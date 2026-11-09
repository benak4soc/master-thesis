"""
Shared helpers for the robustness fixes (fixes/0*.py).

All scripts are meant to be run from the REPO ROOT, e.g.
    python fixes/03_classical_baselines.py

They read the already-prepared windows in 09_processed_datasets/
and write everything new to fixes_out/ (never overwrite originals).
"""

import json
import math
import os
import sys
import time

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(REPO, "09_processed_datasets")
OUT = os.path.join(REPO, "fixes_out")
os.makedirs(OUT, exist_ok=True)

# Make the original model classes importable (BaselineLSTM, BiLSTM, CNN-BiLSTM-Attn)
sys.path.insert(0, os.path.join(REPO, "04_model_training"))

SEQ_DIRS = {
    "host": os.path.join(DATA, "processed_EVSE_B_Host", "sequences"),
    "network": os.path.join(DATA, "processed_EVSE_A", "sequences"),
    "power": os.path.join(DATA, "processed_EVSE_B_Power", "sequences"),
}

FAMILY_NAMES = {0: "Benign", 1: "Recon", 2: "DoS", 3: "Cryptojacking", 4: "Backdoor"}

# Features that identify WHEN/WHERE a flow was captured rather than WHAT it does.
# Absolute timestamps encode the capture file -> perfect but meaningless separation.
NETWORK_LEAKY_COLUMNS = [
    "bidirectional_first_seen_ms", "bidirectional_last_seen_ms",
    "src2dst_first_seen_ms", "src2dst_last_seen_ms",
    "dst2src_first_seen_ms", "dst2src_last_seen_ms",
    "src_port",                 # ephemeral port = per-capture artefact
    "dst_port",                 # optional: keep only if you argue it is behavioural
    "vlan_id", "tunnel_id", "ip_version",          # constant / environment
    "application_is_guessed", "application_confidence",  # nDPI meta, not traffic
]


# ------------------------------------------------------------------ data
def feature_names(modality, seq_dir=None):
    seq_dir = seq_dir or SEQ_DIRS[modality]
    with open(os.path.join(seq_dir, "feature_columns.txt"), encoding="utf-8") as fh:
        return [l.strip() for l in fh if l.strip()]


def load_split(seq_dir, split):
    """Returns X (N,T,F), y_binary, y_multiclass, meta(list of dict) for one split."""
    d = np.load(os.path.join(seq_dir, f"{split}_windows.npz"))
    meta_path = os.path.join(seq_dir, f"{split}_meta.json")
    if not os.path.isfile(meta_path):  # DR sub-folders have no meta -> use parent
        meta_path = os.path.join(os.path.dirname(seq_dir), f"{split}_meta.json")
    with open(meta_path, encoding="utf-8") as fh:
        meta = json.load(fh)
    X = d["X"].astype(np.float32)
    assert len(meta) == len(X), f"meta/window mismatch in {seq_dir}/{split}"
    return X, d["y_binary"].astype(np.int64), d["y_multiclass"].astype(np.int64), meta


def load_all(modality, seq_dir=None):
    seq_dir = seq_dir or SEQ_DIRS[modality]
    return {s: load_split(seq_dir, s) for s in ("train", "val", "test")}


def subset(split_tuple, mask):
    X, yb, ym, meta = split_tuple
    idx = np.where(mask)[0]
    return X[idx], yb[idx], ym[idx], [meta[i] for i in idx]


def non_overlapping(split_tuple, window):
    """Keep every `window`-th window inside each session -> no shared timesteps."""
    _, _, _, meta = split_tuple
    pos, keep = {}, np.zeros(len(meta), dtype=bool)
    for i, m in enumerate(meta):
        k = pos.get(m["session_id"], 0)
        keep[i] = (k % window == 0)
        pos[m["session_id"]] = k + 1
    return subset(split_tuple, keep)


# ------------------------------------------------------------------ DR (train-only fit)
def fit_dr(kind, X_train, y_train_mc, pca_var=0.95):
    """Fit PCA / LDA on TRAIN timesteps only; returns transform(X)->X'."""
    from sklearn.decomposition import PCA
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

    if kind in (None, "none"):
        return lambda X: X, X_train.shape[-1]
    n, t, f = X_train.shape
    flat = X_train.reshape(n * t, f)
    if kind == "pca":
        m = PCA(n_components=pca_var if f > 1 else 1, svd_solver="full", random_state=0).fit(flat)
    elif kind == "lda":
        y_flat = np.repeat(y_train_mc, t)
        n_comp = min(len(np.unique(y_flat)) - 1, f)
        m = LinearDiscriminantAnalysis(n_components=max(1, n_comp), solver="svd").fit(flat, y_flat)
    else:
        raise ValueError(kind)

    def tf(X):
        a, b, c = X.shape
        return m.transform(X.reshape(a * b, c)).reshape(a, b, -1).astype(np.float32)

    return tf, tf(X_train[:1]).shape[-1]


# ------------------------------------------------------------------ metrics
def wilson_ci(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, c - h), min(1.0, c + h))


def binary_report(y_true, y_pred, y_prob=None, y_mc=None):
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    lo, hi = wilson_ci(fp, tn + fp)
    r = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "fpr": fp / max(1, tn + fp),
        "fpr_ci95_low": lo, "fpr_ci95_high": hi,
        "n_benign_test": tn + fp, "n_attack_test": tp + fn,
        "TN": tn, "FP": fp, "FN": fn, "TP": tp,
    }
    if y_prob is not None and len(np.unique(y_true)) == 2:
        r["roc_auc"] = roc_auc_score(y_true, y_prob)
    if y_mc is not None:  # recall per attack family (= detection rate of that family)
        for fid in np.unique(y_mc):
            m = y_mc == fid
            name = FAMILY_NAMES.get(int(fid), str(fid))
            key = "tnr_Benign" if fid == 0 else f"recall_{name}"
            r[key] = float((y_pred[m] == (0 if fid == 0 else 1)).mean())
    return {k: (float(v) if isinstance(v, (np.floating, float)) else v) for k, v in r.items()}


# ------------------------------------------------------------------ deep training
def train_torch(model_type, Xtr, ytr, Xva, yva, seed=42, hidden_size=64, num_layers=1,
                dropout=0.2, epochs=40, lr=1e-3, batch_size=64, patience=8, verbose=False):
    """Same recipe as lstm_common.train_binary_lstm (weighted CE, early stop on val F1)."""
    import torch
    import torch.nn as nn
    from sklearn.metrics import f1_score
    from lstm_common import build_model

    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.use_deterministic_algorithms(False)
    model = build_model(model_type, Xtr.shape[-1], hidden_size=hidden_size,
                        num_layers=num_layers, dropout=dropout)
    counts = np.bincount(ytr, minlength=2).astype(np.float32)
    w = torch.tensor(counts.sum() / (2 * np.maximum(counts, 1)), dtype=torch.float32)
    crit = nn.CrossEntropyLoss(weight=w)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    Xt, yt = torch.from_numpy(Xtr), torch.from_numpy(ytr)
    Xv = torch.from_numpy(Xva)
    g = torch.Generator().manual_seed(seed)
    best, best_f1, bad = None, -1.0, 0
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(Xt), generator=g)
        for i in range(0, len(Xt), batch_size):
            b = perm[i:i + batch_size]
            opt.zero_grad()
            loss = crit(model(Xt[b]), yt[b])
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            pv = model(Xv).argmax(1).numpy()
        f1 = f1_score(yva, pv, zero_division=0)
        if f1 > best_f1:
            best_f1, bad = f1, 0
            best = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
        if verbose:
            print(f"  ep{ep:02d} val_f1={f1:.4f}")
    model.load_state_dict(best)
    model.eval()
    return model


def predict_torch(model, X):
    import torch
    with torch.no_grad():
        logits = model(torch.from_numpy(X))
        prob = torch.softmax(logits, 1)[:, 1].numpy()
    return (prob >= 0.5).astype(int), prob


def n_params(model):
    return int(sum(p.numel() for p in model.parameters()))


def latency_ms(fn, X1, reps=300):
    for _ in range(20):
        fn(X1)
    t = []
    for _ in range(reps):
        s = time.perf_counter()
        fn(X1)
        t.append((time.perf_counter() - s) * 1000)
    return float(np.median(t)), float(np.percentile(t, 95))


def save_rows(rows, name):
    import pandas as pd
    path = os.path.join(OUT, name)
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"\nSaved {path}")
    return path
