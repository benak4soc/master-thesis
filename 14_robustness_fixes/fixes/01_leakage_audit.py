"""
FIX 1a — Leakage / shortcut audit.

Idea: if ONE feature with ONE threshold (a decision stump) already gives
~100% on the TEST set, the deep models are not learning attack behaviour,
they are exploiting a shortcut (timestamp, port, capture-specific constant).

Run:
    python fixes/01_leakage_audit.py                     # all three modalities
    python fixes/01_leakage_audit.py --seq-dir fixes_out/network_noleak
Output:
    fixes_out/leakage_audit_<name>.csv   (one row per feature, sorted)
Rule of thumb for the thesis:
    any single feature with test accuracy >= 0.99 must be justified or removed.
"""

import argparse
import os

import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.tree import DecisionTreeClassifier

from fx_common import SEQ_DIRS, feature_names, load_all, save_rows


def audit(name, seq_dir):
    data = load_all(name if name in SEQ_DIRS else None, seq_dir)
    feats = feature_names(None, seq_dir)
    Xtr, ytr = data["train"][0][:, -1, :], data["train"][1]   # last timestep = label timestep
    Xte, yte = data["test"][0][:, -1, :], data["test"][1]
    rows = []
    for j, f in enumerate(feats):
        stump = DecisionTreeClassifier(max_depth=1, random_state=0).fit(Xtr[:, [j]], ytr)
        p = stump.predict(Xte[:, [j]])
        rows.append({"feature": f,
                     "test_acc": accuracy_score(yte, p),
                     "test_balanced_acc": balanced_accuracy_score(yte, p),
                     "threshold_scaled": float(stump.tree_.threshold[0])})
    rows.sort(key=lambda r: -r["test_balanced_acc"])
    flagged = [r for r in rows if r["test_balanced_acc"] >= 0.99]
    print(f"\n=== {name}  ({seq_dir})")
    print(f"test: benign={int((yte == 0).sum())} attack={int((yte == 1).sum())}")
    print("Top 8 single-feature stumps (balanced acc):")
    for r in rows[:8]:
        print(f"  {r['test_balanced_acc']:.3f}  {r['feature']}")
    print(f"FLAGGED (>=0.99 alone): {len(flagged)} -> {[r['feature'] for r in flagged]}")
    save_rows(rows, f"leakage_audit_{name}.csv")
    return flagged


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq-dir", help="audit a custom sequence folder")
    a = ap.parse_args()
    if a.seq_dir:
        audit(os.path.basename(os.path.normpath(a.seq_dir)), a.seq_dir)
    else:
        for m, d in SEQ_DIRS.items():
            audit(m, d)
