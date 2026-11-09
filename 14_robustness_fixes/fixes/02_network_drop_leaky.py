"""
FIX 1b — Remove leaky network features and rebuild the network windows (+PCA/LDA).

Quick path (works with what is in the repo): drop the columns from the
existing windows. StandardScaler is per-column, so dropping columns after
scaling is mathematically identical to re-scaling without them.

Proper path (for the final thesis): also add the same list to METADATA in
03_sequence_preparation/prepare_network_sequences.py so the columns never
enter feature_columns (see FIX_GUIDE.md, section 1).

Run:
    python fixes/02_network_drop_leaky.py
Output:
    fixes_out/network_noleak/{train,val,test}_windows.npz + meta + feature_columns.txt
    fixes_out/network_noleak/{pca,lda_multiclass}/...
"""

import json
import os
import shutil

import numpy as np

from fx_common import NETWORK_LEAKY_COLUMNS, OUT, SEQ_DIRS, feature_names, fit_dr, load_all

SRC = SEQ_DIRS["network"]
DST = os.path.join(OUT, "network_noleak")
os.makedirs(DST, exist_ok=True)

feats = feature_names("network")
keep = [i for i, f in enumerate(feats) if f not in NETWORK_LEAKY_COLUMNS]
dropped = [f for f in feats if f in NETWORK_LEAKY_COLUMNS]
print(f"Features {len(feats)} -> {len(keep)}; dropped: {dropped}")

data = load_all("network")
clean = {}
for split, (X, yb, ym, meta) in data.items():
    Xk = X[:, :, keep]
    clean[split] = (Xk, yb, ym)
    np.savez_compressed(os.path.join(DST, f"{split}_windows.npz"), X=Xk, y_binary=yb, y_multiclass=ym)
    shutil.copy(os.path.join(SRC, f"{split}_meta.json"), os.path.join(DST, f"{split}_meta.json"))
with open(os.path.join(DST, "feature_columns.txt"), "w", encoding="utf-8") as fh:
    fh.write("\n".join(feats[i] for i in keep) + "\n")

# Re-fit PCA and LDA on TRAIN only, on the cleaned features
for kind, sub in (("pca", "pca"), ("lda", "lda_multiclass")):
    tf, dim = fit_dr(kind, clean["train"][0], clean["train"][2])
    d = os.path.join(DST, sub)
    os.makedirs(d, exist_ok=True)
    for split, (X, yb, ym) in clean.items():
        np.savez_compressed(os.path.join(d, f"{split}_windows.npz"), X=tf(X), y_binary=yb, y_multiclass=ym)
    with open(os.path.join(d, "feature_columns.txt"), "w") as fh:
        fh.write("\n".join(f"{kind}_{i}" for i in range(dim)) + "\n")
    print(f"{kind.upper()}: {len(keep)} -> {dim} components")

with open(os.path.join(DST, "README.txt"), "w") as fh:
    json.dump({"source": SRC, "dropped_columns": dropped, "n_features": len(keep)}, fh, indent=2)
print(f"\nDone -> {DST}\nNext: python fixes/01_leakage_audit.py --seq-dir {DST}")
