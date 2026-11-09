"""
FIX 3 — Stricter evaluations that a 100% score has to survive.

(a) CROSS-CONDITION : train on State=idle  -> test on State=charging, and reverse.
                      Benign/Cryptojacking are single sessions split by time, so the
                      original test set is "the same recording, later". Here the test
                      condition is never seen during training.
(b) LEAVE-ONE-FAMILY-OUT (LOFO): remove one attack family (e.g. Cryptojacking) from
                      training completely; measure how much of it is still flagged.
                      This is the closest thing to a zero-day test the dataset allows.
(c) NON-OVERLAPPING : keep only every W-th window (stride = window) so test windows
                      share no timesteps -> removes the stride-1 inflation.

Models: LDA-LSTM (your Pipeline C, hidden 16 = tuned edge model), LogReg, HGB.
PCA/LDA are re-fitted on each new training set (no leakage from the held-out part).

Run:
    python fixes/04_generalisation_tests.py                 # host + power
    python fixes/04_generalisation_tests.py --modality host --fast
Output:
    fixes_out/generalisation_tests.csv
"""

import argparse

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

from fx_common import (FAMILY_NAMES, SEQ_DIRS, binary_report, fit_dr, load_all,
                       non_overlapping, predict_torch, save_rows, subset, train_torch)


def concat(*parts):
    X = np.concatenate([p[0] for p in parts])
    yb = np.concatenate([p[1] for p in parts])
    ym = np.concatenate([p[2] for p in parts])
    meta = sum((p[3] for p in parts), [])
    return X, yb, ym, meta


def run_models(tag, mod, train, val, test, fast=False):
    out = []
    if len(np.unique(train[1])) < 2 or len(test[1]) == 0:
        print(f"  skip {tag}: train has one class or test empty")
        return out
    tf, dim = fit_dr("lda", train[0], train[2])
    Xtr, Xva, Xte = tf(train[0]), tf(val[0]), tf(test[0])
    # deep: Pipeline C (LDA-LSTM), tuned edge size
    model = train_torch("lstm", Xtr, train[1], Xva, val[1], hidden_size=16,
                        epochs=15 if fast else 40)
    p, pr = predict_torch(model, Xte)
    out.append(("LDA-LSTM(h16)", p, pr))
    for name, clf in (("LogReg", LogisticRegression(max_iter=3000, class_weight="balanced")),
                      ("HGB", HistGradientBoostingClassifier(random_state=0, class_weight="balanced"))):
        A = train[0].reshape(len(train[0]), -1)
        clf.fit(A, train[1])
        B = test[0].reshape(len(test[0]), -1)
        out.append((name + "(raw)", clf.predict(B), clf.predict_proba(B)[:, 1]))
    rows = []
    for name, p, pr in out:
        r = binary_report(test[1], p, pr, test[2])
        r.update(modality=mod, test=tag, model=name, n_train=len(train[1]), n_test=len(test[1]))
        rows.append(r)
        fam = {k: round(v, 3) for k, v in r.items() if k.startswith(("recall_", "tnr_"))}
        print(f"  {tag:32s} {name:14s} F1={r['f1']:.3f} FPR={r['fpr']:.3f} {fam}")
    return rows


def main(modalities, fast):
    rows = []
    for mod in modalities:
        d = load_all(mod)
        tr, va, te = d["train"], d["val"], d["test"]
        allw = concat(tr, va, te)
        state = np.array([m.get("State", "na") for m in allw[3]])
        print(f"\n######## {mod}")

        # (a) cross-condition
        for a, b in (("idle", "charging"), ("charging", "idle")):
            if not ((state == a).any() and (state == b).any()):
                continue
            src = subset(allw, state == a)
            # validation = random 15% of source windows, by session to stay honest
            sess = np.array([m["session_id"] for m in src[3]])
            rng = np.random.default_rng(0)
            u = np.unique(sess)
            val_s = set(rng.choice(u, size=max(1, int(0.15 * len(u))), replace=False))
            is_val = np.array([s in val_s for s in sess])
            # make sure val has both classes; otherwise fall back to the original val split
            v = subset(src, is_val)
            if len(np.unique(v[1])) < 2:
                v = subset(va, np.array([m.get("State") == a for m in va[3]]))
                is_val[:] = False
            rows += run_models(f"cross: train {a} -> test {b}", mod,
                               subset(src, ~is_val), v, subset(allw, state == b), fast)

        # (b) leave-one-family-out
        for fid in np.unique(allw[2]):
            if fid == 0:
                continue
            fam = FAMILY_NAMES[int(fid)]
            train = subset(tr, tr[2] != fid)
            val = subset(va, va[2] != fid)
            unseen = subset(allw, allw[2] == fid)            # every window of that family
            benign_test = subset(te, te[2] == 0)
            rows += run_models(f"LOFO: unseen {fam}", mod, train, val,
                               concat(benign_test, unseen), fast)

        # (c) non-overlapping windows
        W = tr[0].shape[1]
        rows += run_models(f"non-overlap (stride={W})", mod, non_overlapping(tr, W),
                           non_overlapping(va, W), non_overlapping(te, W), fast)
    save_rows(rows, "generalisation_tests.csv")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--modality", choices=list(SEQ_DIRS), action="append")
    ap.add_argument("--fast", action="store_true", help="15 epochs instead of 40")
    a = ap.parse_args()
    main(a.modality or ["host", "power"], a.fast)
