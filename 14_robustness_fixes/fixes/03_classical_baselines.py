"""
FIX 2 — Classical baselines next to pipelines A–E.

A thesis that claims "LDA + LSTM" is a good detector must show what a
simple model achieves on exactly the same windows. If LogReg already hits
F1 = 1.0, the contribution is EFFICIENCY, not accuracy — say so.

Models (all on flattened windows, same train/val/test):
    LDA-clf    : LDA used directly as the classifier (no deep model at all)
    LogReg     : L2 logistic regression
    RF         : random forest, 200 trees
    HGB        : histogram gradient boosting (XGBoost-like, in sklearn)
Feature spaces: raw | PCA | LDA (PCA/LDA fit on train timesteps only)

Run:
    python fixes/03_classical_baselines.py
Output:
    fixes_out/classical_baselines.csv  (accuracy, F1, FPR + 95% CI, per-family recall,
                                        #params, median/p95 latency per window)
"""

import os
import pickle

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from fx_common import OUT, SEQ_DIRS, binary_report, fit_dr, latency_ms, load_all, save_rows

MODALITIES = dict(SEQ_DIRS)
noleak = os.path.join(OUT, "network_noleak")
if os.path.isdir(noleak):
    MODALITIES["network_noleak"] = noleak
    MODALITIES.pop("network")   # the leaky version should not be reported any more


def models():
    return {
        "LDA-clf": LinearDiscriminantAnalysis(),
        "LogReg": LogisticRegression(max_iter=3000, class_weight="balanced"),
        "RF": RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=0,
                                     class_weight="balanced"),
        "HGB": HistGradientBoostingClassifier(random_state=0, class_weight="balanced"),
    }


def size_proxy(m):
    """#parameters for linear models, #nodes for trees, bytes as fallback."""
    if hasattr(m, "coef_"):
        return int(m.coef_.size + np.size(m.intercept_))
    if hasattr(m, "estimators_"):
        return int(sum(e.tree_.node_count for e in m.estimators_))
    return len(pickle.dumps(m))


rows = []
for mod, seq_dir in MODALITIES.items():
    d = load_all(None, seq_dir)
    (Xtr, ytr, mtr, _), (Xte, yte, mte, _) = d["train"], d["test"]
    for dr in ("none", "pca", "lda"):
        tf, dim = fit_dr(dr, Xtr, mtr)
        A, B = tf(Xtr).reshape(len(Xtr), -1), tf(Xte).reshape(len(Xte), -1)
        for name, m in models().items():
            m.fit(A, ytr)
            pred = m.predict(B)
            prob = m.predict_proba(B)[:, 1]
            r = binary_report(yte, pred, prob, mte)
            med, p95 = latency_ms(lambda x: m.predict(x), B[:1], reps=100)
            r.update(modality=mod, dr=dr, features_per_step=dim, model=name,
                     size=size_proxy(m), latency_ms_median=med, latency_ms_p95=p95)
            rows.append(r)
            print(f"{mod:15s} {dr:4s} dim={dim:3d} {name:8s} F1={r['f1']:.4f} "
                  f"FPR={r['fpr']:.3f} [{r['fpr_ci95_low']:.3f},{r['fpr_ci95_high']:.3f}]")

cols_first = ["modality", "dr", "features_per_step", "model", "accuracy", "f1", "roc_auc",
              "fpr", "fpr_ci95_low", "fpr_ci95_high", "n_benign_test", "size",
              "latency_ms_median", "latency_ms_p95"]
rows = [{**{k: r.get(k) for k in cols_first}, **r} for r in rows]
save_rows(rows, "classical_baselines.csv")
