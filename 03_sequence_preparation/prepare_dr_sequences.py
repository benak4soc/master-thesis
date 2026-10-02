# ============================================================
# CICEVSE2024 - Fit PCA / LDA on train windows only
# Master's Thesis — Pipelines B & C preparation
#
# Input :
#   processed_EVSE_B_Host/sequences/{train,val,test}_windows.npz
#   (from prepare_host_sequences.py; already scaled)
#
# Output:
#   processed_EVSE_B_Host/sequences/pca/
#   processed_EVSE_B_Host/sequences/lda_multiclass/
#
# Rules (thesis §11):
#   PCA and LDA are fitted ONLY on training data, then applied
#   to val/test.
#
# LDA (plan B):
#   Supervised multi-class LDA using Attack_Group labels:
#   Benign / Recon / DoS / Cryptojacking / Backdoor
#   -> at most (C-1) components (Backdoor absent => C=4 => 3 comps)
# ============================================================

import os
import json
import warnings

import numpy as np
import joblib

from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

warnings.filterwarnings("ignore")


SEQ_DIR = os.path.join(
    "processed_EVSE_B_Host",
    "sequences"
)

PCA_DIR = os.path.join(SEQ_DIR, "pca")
LDA_DIR = os.path.join(SEQ_DIR, "lda_multiclass")

os.makedirs(PCA_DIR, exist_ok=True)
os.makedirs(LDA_DIR, exist_ok=True)

# Keep components explaining this fraction of variance
PCA_VARIANCE = 0.95
RANDOM_STATE = 42

# Multi-class label names (must match hosteventspreprocessing.py)
MULTI_CLASS_NAMES = {
    0: "Benign",
    1: "Recon",
    2: "DoS",
    3: "Cryptojacking",
    4: "Backdoor",
}


print("=" * 70)
print("DIMENSIONALITY REDUCTION — PCA & MULTI-CLASS LDA")
print("=" * 70)


def load_split(name):
    path = os.path.join(
        SEQ_DIR,
        f"{name}_windows.npz"
    )

    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Missing {path}. Run prepare_host_sequences.py first."
        )

    data = np.load(path)
    return {
        "X": data["X"].astype(np.float32),
        "y_binary": data["y_binary"].astype(np.int64),
        "y_multiclass": data["y_multiclass"].astype(np.int64),
        "y_attack_family": data["y_attack_family"].astype(np.float64),
    }


train = load_split("train")
val = load_split("val")
test = load_split("test")

n_train, n_steps, n_features = train["X"].shape

print(f"\nTrain windows : {train['X'].shape}")
print(f"Val windows   : {val['X'].shape}")
print(f"Test windows  : {test['X'].shape}")


# ------------------------------------------------------------
# Helpers: reshape windows <-> timesteps
# ------------------------------------------------------------

def flatten_windows(X):
    """(N, T, F) -> (N*T, F)"""
    return X.reshape(-1, X.shape[-1])


def expand_labels(y, n_steps):
    """
    Repeat each window label across its timesteps.
    Valid here because each window lies inside one session
    / Attack_Group block (labels constant within the window).
    """
    return np.repeat(y, n_steps)


def apply_transform_windows(X, transformer):
    """Transform each timestep, keep (N, T, k) shape."""
    n, t, f = X.shape
    flat = flatten_windows(X)
    reduced = transformer.transform(flat).astype(np.float32)
    return reduced.reshape(n, t, -1)


def save_split_windows(out_dir, name, X, source):
    path = os.path.join(out_dir, f"{name}_windows.npz")
    np.savez_compressed(
        path,
        X=X,
        y_binary=source["y_binary"],
        y_multiclass=source["y_multiclass"],
        y_attack_family=source["y_attack_family"],
    )
    return path


# ============================================================
# 1. PCA (unsupervised) — fit on TRAIN timesteps only
# ============================================================

print("\n" + "-" * 70)
print("PCA")
print("-" * 70)

X_train_flat = flatten_windows(train["X"])

pca = PCA(
    n_components=PCA_VARIANCE,
    svd_solver="full",
    random_state=RANDOM_STATE,
)

pca.fit(X_train_flat)

n_pca = int(pca.n_components_)
explained = float(pca.explained_variance_ratio_.sum())

print(f"Components selected : {n_pca}")
print(f"Variance explained  : {explained:.4f}")
print(
    "Per-component var  : "
    + ", ".join(
        f"{v:.4f}"
        for v in pca.explained_variance_ratio_[:10]
    )
    + (" ..." if n_pca > 10 else "")
)

joblib.dump(
    pca,
    os.path.join(PCA_DIR, "pca.joblib")
)

for name, source in [
    ("train", train),
    ("val", val),
    ("test", test),
]:
    X_red = apply_transform_windows(source["X"], pca)
    path = save_split_windows(
        PCA_DIR,
        name,
        X_red,
        source
    )
    print(
        f"  {name:5s} -> {X_red.shape}  saved {os.path.basename(path)}"
    )

pca_cfg = {
    "method": "PCA",
    "variance_threshold": PCA_VARIANCE,
    "n_components": n_pca,
    "explained_variance_ratio_sum": explained,
    "explained_variance_ratio": (
        pca.explained_variance_ratio_.tolist()
    ),
    "input_features": n_features,
    "fit_on": "train_timesteps_only",
}

with open(
    os.path.join(PCA_DIR, "pca_config.json"),
    "w",
    encoding="utf-8"
) as handle:
    json.dump(pca_cfg, handle, indent=2)


# ============================================================
# 2. Multi-class LDA — fit on TRAIN timesteps only
# ============================================================

print("\n" + "-" * 70)
print("MULTI-CLASS LDA (Attack_Group)")
print("-" * 70)

y_train_flat = expand_labels(
    train["y_multiclass"],
    n_steps
)

present_ids = sorted(np.unique(y_train_flat).tolist())
present_names = [
    MULTI_CLASS_NAMES.get(i, f"id_{i}")
    for i in present_ids
]

print(f"Classes present in train: {present_names}")
print(
    "Counts: "
    + str(
        {
            MULTI_CLASS_NAMES.get(i, str(i)): int(
                (y_train_flat == i).sum()
            )
            for i in present_ids
        }
    )
)

n_classes = len(present_ids)
max_lda_components = max(1, n_classes - 1)

lda = LinearDiscriminantAnalysis(
    n_components=max_lda_components,
    solver="svd",
)

lda.fit(X_train_flat, y_train_flat)

n_lda = int(lda.n_components)
print(f"LDA components : {n_lda}  (max C-1 = {max_lda_components})")

# Explained variance ratio between classes (sklearn attribute)
if hasattr(lda, "explained_variance_ratio_"):
    print(
        "Between-class variance ratio: "
        + ", ".join(
            f"{v:.4f}"
            for v in lda.explained_variance_ratio_
        )
    )

joblib.dump(
    lda,
    os.path.join(LDA_DIR, "lda.joblib")
)

for name, source in [
    ("train", train),
    ("val", val),
    ("test", test),
]:
    X_red = apply_transform_windows(source["X"], lda)
    path = save_split_windows(
        LDA_DIR,
        name,
        X_red,
        source
    )
    print(
        f"  {name:5s} -> {X_red.shape}  saved {os.path.basename(path)}"
    )

lda_cfg = {
    "method": "LDA_multiclass",
    "label_source": "target_multiclass / Attack_Group",
    "classes_present": present_names,
    "class_ids_present": present_ids,
    "n_components": n_lda,
    "max_components_c_minus_1": max_lda_components,
    "input_features": n_features,
    "fit_on": "train_timesteps_only",
    "note": (
        "Backdoor not present in EVSE-B host HPC file; "
        "LDA uses Benign/Recon/DoS/Cryptojacking when available."
    ),
}

if hasattr(lda, "explained_variance_ratio_"):
    lda_cfg["explained_variance_ratio"] = (
        lda.explained_variance_ratio_.tolist()
    )

with open(
    os.path.join(LDA_DIR, "lda_config.json"),
    "w",
    encoding="utf-8"
) as handle:
    json.dump(lda_cfg, handle, indent=2)


# ============================================================
# 3. Optional: binary LDA (C-1 = 1) for thesis discussion
# ============================================================

print("\n" + "-" * 70)
print("BINARY LDA (supplementary, C-1 = 1)")
print("-" * 70)

BINARY_LDA_DIR = os.path.join(SEQ_DIR, "lda_binary")
os.makedirs(BINARY_LDA_DIR, exist_ok=True)

y_bin_flat = expand_labels(
    train["y_binary"],
    n_steps
)

lda_bin = LinearDiscriminantAnalysis(
    n_components=1,
    solver="svd",
)
lda_bin.fit(X_train_flat, y_bin_flat)

joblib.dump(
    lda_bin,
    os.path.join(BINARY_LDA_DIR, "lda.joblib")
)

for name, source in [
    ("train", train),
    ("val", val),
    ("test", test),
]:
    X_red = apply_transform_windows(source["X"], lda_bin)
    save_split_windows(
        BINARY_LDA_DIR,
        name,
        X_red,
        source
    )
    print(f"  {name:5s} -> {X_red.shape}")

with open(
    os.path.join(BINARY_LDA_DIR, "lda_config.json"),
    "w",
    encoding="utf-8"
) as handle:
    json.dump(
        {
            "method": "LDA_binary",
            "n_components": 1,
            "label_source": "y_binary",
            "note": (
                "Classical binary LDA yields at most 1 discriminant; "
                "included for thesis limitation discussion."
            ),
        },
        handle,
        indent=2,
    )


print("\n" + "=" * 70)
print("DR PREPARATION COMPLETE")
print("=" * 70)
print(f"PCA features           : {n_features} -> {n_pca}")
print(f"Multi-class LDA feats  : {n_features} -> {n_lda}")
print(f"Binary LDA feats       : {n_features} -> 1")
print(f"\nPCA dir : {PCA_DIR}")
print(f"LDA dir : {LDA_DIR}")
print("\nNext: train_pca_lstm.py / train_lda_lstm.py")
