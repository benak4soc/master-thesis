# ============================================================
# Power EVSE-B — PCA + multi-class LDA (incl. Backdoor)
# ============================================================

import os
import json
import warnings

import numpy as np
import joblib
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

warnings.filterwarnings("ignore")

SEQ_DIR = os.path.join("processed_EVSE_B_Power", "sequences")
PCA_DIR = os.path.join(SEQ_DIR, "pca")
LDA_DIR = os.path.join(SEQ_DIR, "lda_multiclass")
BIN_LDA_DIR = os.path.join(SEQ_DIR, "lda_binary")

for d in [PCA_DIR, LDA_DIR, BIN_LDA_DIR]:
    os.makedirs(d, exist_ok=True)

PCA_VARIANCE = 0.95
RANDOM_STATE = 42
MULTI_CLASS_NAMES = {
    0: "Benign",
    1: "Recon",
    2: "DoS",
    3: "Cryptojacking",
    4: "Backdoor",
}


def load_split(name):
    data = np.load(os.path.join(SEQ_DIR, f"{name}_windows.npz"))
    return {
        "X": data["X"].astype(np.float32),
        "y_binary": data["y_binary"].astype(np.int64),
        "y_multiclass": data["y_multiclass"].astype(np.int64),
        "y_attack_family": data["y_attack_family"].astype(np.float64),
    }


def flatten(X):
    return X.reshape(-1, X.shape[-1])


def expand_labels(y, t):
    return np.repeat(y, t)


def apply_tf(X, tf):
    n, t, f = X.shape
    return tf.transform(flatten(X)).astype(np.float32).reshape(n, t, -1)


def save_split(out_dir, name, X, source):
    np.savez_compressed(
        os.path.join(out_dir, f"{name}_windows.npz"),
        X=X,
        y_binary=source["y_binary"],
        y_multiclass=source["y_multiclass"],
        y_attack_family=source["y_attack_family"],
    )


print("=" * 70)
print("POWER EVSE-B — PCA / LDA")
print("=" * 70)

train, val, test = load_split("train"), load_split("val"), load_split("test")
n_steps = train["X"].shape[1]
n_features = train["X"].shape[2]
X_flat = flatten(train["X"])
print(f"Train: {train['X'].shape}")

# PCA — with only 4 features, may keep most/all
n_pca = min(n_features, max(1, n_features))
pca = PCA(
    n_components=min(PCA_VARIANCE, n_features),
    svd_solver="full",
    random_state=RANDOM_STATE,
)
# If variance float > possible comps, sklearn handles; for n_features=4 use auto
try:
    pca.fit(X_flat)
except Exception:
    pca = PCA(n_components=n_features, svd_solver="full", random_state=RANDOM_STATE)
    pca.fit(X_flat)

print(
    f"PCA: {n_features} -> {pca.n_components_} "
    f"(var={pca.explained_variance_ratio_.sum():.4f})"
)
joblib.dump(pca, os.path.join(PCA_DIR, "pca.joblib"))
for name, src in [("train", train), ("val", val), ("test", test)]:
    Xr = apply_tf(src["X"], pca)
    save_split(PCA_DIR, name, Xr, src)
    print(f"  PCA {name}: {Xr.shape}")

with open(os.path.join(PCA_DIR, "pca_config.json"), "w", encoding="utf-8") as h:
    json.dump(
        {
            "n_components": int(pca.n_components_),
            "explained_variance_ratio_sum": float(
                pca.explained_variance_ratio_.sum()
            ),
            "input_features": n_features,
        },
        h,
        indent=2,
    )

# Multi-class LDA (5 classes incl Backdoor => up to 4 comps)
y_flat = expand_labels(train["y_multiclass"], n_steps)
present = sorted(np.unique(y_flat).tolist())
print("LDA classes:", [MULTI_CLASS_NAMES.get(i, str(i)) for i in present])
n_comp = max(1, len(present) - 1)
# LDA cannot have more comps than features-related limits; also <= n_features
n_comp = min(n_comp, n_features, X_flat.shape[1])
lda = LinearDiscriminantAnalysis(n_components=n_comp, solver="svd")
lda.fit(X_flat, y_flat)
print(f"LDA: {n_features} -> {lda.n_components}")
joblib.dump(lda, os.path.join(LDA_DIR, "lda.joblib"))
for name, src in [("train", train), ("val", val), ("test", test)]:
    Xr = apply_tf(src["X"], lda)
    save_split(LDA_DIR, name, Xr, src)
    print(f"  LDA {name}: {Xr.shape}")

cfg = {
    "n_components": int(lda.n_components),
    "classes": [MULTI_CLASS_NAMES.get(i, str(i)) for i in present],
    "input_features": n_features,
    "note": "Power modality includes Backdoor for full 5-class LDA.",
}
if hasattr(lda, "explained_variance_ratio_"):
    cfg["explained_variance_ratio"] = lda.explained_variance_ratio_.tolist()

with open(os.path.join(LDA_DIR, "lda_config.json"), "w", encoding="utf-8") as h:
    json.dump(cfg, h, indent=2)

# Binary LDA
lda_b = LinearDiscriminantAnalysis(n_components=1, solver="svd")
lda_b.fit(X_flat, expand_labels(train["y_binary"], n_steps))
joblib.dump(lda_b, os.path.join(BIN_LDA_DIR, "lda.joblib"))
for name, src in [("train", train), ("val", val), ("test", test)]:
    save_split(BIN_LDA_DIR, name, apply_tf(src["X"], lda_b), src)

print("\nPower DR preparation complete.")
