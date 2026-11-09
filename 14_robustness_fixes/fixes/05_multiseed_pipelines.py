"""
FIX 4 — Pipelines A–E with 5 seeds -> mean ± std (and per-family recall, FPR CI).

One seed (42) cannot tell whether C vs E on power (F1 0.983 vs 0.990) is a real
difference or noise. This re-trains every pipeline with seeds 0..4 using the
same recipe as lstm_common (weighted CE, Adam 1e-3, early stop on val F1, patience 8)
and reports mean ± std plus a paired Wilcoxon test vs Pipeline C.

Run:
    python fixes/05_multiseed_pipelines.py --modality power
    python fixes/05_multiseed_pipelines.py --modality host --non-overlap
    python fixes/05_multiseed_pipelines.py --modality network_noleak --seeds 0 1 2
Output:
    fixes_out/multiseed_<modality>[_nonoverlap]_runs.csv     (one row per run)
    fixes_out/multiseed_<modality>[_nonoverlap]_summary.csv  (mean ± std table for the thesis)
"""

import argparse
import os
import time

import numpy as np
import pandas as pd

from fx_common import (OUT, SEQ_DIRS, binary_report, fit_dr, latency_ms, load_all, n_params,
                       non_overlapping, predict_torch, save_rows, train_torch)

PIPELINES = {          # letter: (DR, model_type)
    "A_LSTM": ("none", "lstm"),
    "B_PCA-LSTM": ("pca", "lstm"),
    "C_LDA-LSTM": ("lda", "lstm"),
    "D_LDA-BiLSTM": ("lda", "bilstm"),
    "E_CNN-BiLSTM-Attn": ("none", "cnn_bilstm_attention"),
}


def main(mod, seeds, non_overlap, epochs):
    seq_dir = os.path.join(OUT, "network_noleak") if mod == "network_noleak" else SEQ_DIRS[mod]
    d = load_all(None, seq_dir)
    tr, va, te = d["train"], d["val"], d["test"]
    if non_overlap:
        W = tr[0].shape[1]
        tr, va, te = (non_overlapping(s, W) for s in (tr, va, te))
    tag = mod + ("_nonoverlap" if non_overlap else "")
    rows = []
    for pname, (dr, mtype) in PIPELINES.items():
        tf, dim = fit_dr(dr, tr[0], tr[2])          # DR is deterministic -> fit once
        Xtr, Xva, Xte = tf(tr[0]), tf(va[0]), tf(te[0])
        for s in seeds:
            t0 = time.time()
            model = train_torch(mtype, Xtr, tr[1], Xva, va[1], seed=s, epochs=epochs)
            secs = time.time() - t0
            p, pr = predict_torch(model, Xte)
            r = binary_report(te[1], p, pr, te[2])
            med, p95 = latency_ms(lambda x: predict_torch(model, x), Xte[:1], reps=100)
            r.update(modality=tag, pipeline=pname, seed=s, features_per_step=dim,
                     n_params=n_params(model), train_seconds=secs,
                     latency_ms_median=med, latency_ms_p95=p95)
            rows.append(r)
            print(f"{tag} {pname:18s} seed={s} F1={r['f1']:.4f} FPR={r['fpr']:.4f} ({secs:.0f}s)")
    save_rows(rows, f"multiseed_{tag}_runs.csv")

    df = pd.DataFrame(rows)
    metrics = [c for c in ("f1", "roc_auc", "fpr", "recall", "train_seconds", "latency_ms_median")
               if c in df] + [c for c in df if c.startswith(("recall_", "tnr_"))]
    summ = []
    for pname, g in df.groupby("pipeline", sort=False):
        row = {"pipeline": pname, "features_per_step": int(g.features_per_step.iloc[0]),
               "n_params": int(g.n_params.iloc[0]), "n_seeds": len(g)}
        for m in metrics:
            row[m] = f"{g[m].mean():.4f} ± {g[m].std(ddof=1):.4f}"
        # paired test against Pipeline C on F1 (same seeds)
        if pname != "C_LDA-LSTM" and len(seeds) >= 5:
            from scipy.stats import wilcoxon
            c = df[df.pipeline == "C_LDA-LSTM"].sort_values("seed").f1.values
            x = g.sort_values("seed").f1.values
            try:
                row["wilcoxon_p_vs_C"] = float(wilcoxon(x, c).pvalue)
            except ValueError:          # all differences zero (saturated metrics)
                row["wilcoxon_p_vs_C"] = 1.0
        summ.append(row)
    path = save_rows(summ, f"multiseed_{tag}_summary.csv")
    print(pd.read_csv(path).to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--modality", default="power",
                    choices=list(SEQ_DIRS) + ["network_noleak"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--non-overlap", action="store_true")
    ap.add_argument("--epochs", type=int, default=40)
    a = ap.parse_args()
    main(a.modality, a.seeds, a.non_overlap, a.epochs)
