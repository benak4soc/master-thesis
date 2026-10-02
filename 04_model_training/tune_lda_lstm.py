# ============================================================
# Hyperparameter tuning — Pipeline C (LDA-LSTM)
# Master's Thesis §16 multi-criteria selection
#
# Primary: validation F1
# Secondary (for edge): n_params, val train time, inference latency
# ============================================================

import os
import json
import time
import itertools
import warnings

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from sklearn.metrics import f1_score, roc_auc_score, accuracy_score

from lstm_common import BaselineLSTM, set_seed, load_binary_split

warnings.filterwarnings("ignore")


SEQ_DIR = os.path.join(
    "processed_EVSE_B_Host",
    "sequences",
    "lda_multiclass",
)

OUT_DIR = os.path.join(
    "models",
    "pipeline_C_lda_lstm_tuned",
)

os.makedirs(OUT_DIR, exist_ok=True)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

SEED = 42
EPOCHS = 25
PATIENCE = 5

# Compact search space suitable for Jetson Nano
SEARCH_SPACE = {
    "hidden_size": [16, 32, 64],
    "num_layers": [1, 2],
    "dropout": [0.1, 0.2, 0.3],
    "learning_rate": [1e-3, 5e-4],
    "batch_size": [32, 64],
}


def make_loader(X, y, batch_size, shuffle):
    return DataLoader(
        TensorDataset(
            torch.from_numpy(X),
            torch.from_numpy(y),
        ),
        batch_size=batch_size,
        shuffle=shuffle,
    )


def eval_f1(model, loader):
    model.eval()
    ys, ps, pr = [], [], []

    with torch.no_grad():
        for Xb, yb in loader:
            Xb = Xb.to(DEVICE)
            logits = model(Xb)
            prob = torch.softmax(logits, dim=1)[:, 1]
            pred = torch.argmax(logits, dim=1)
            ys.append(yb.numpy())
            ps.append(pred.cpu().numpy())
            pr.append(prob.cpu().numpy())

    y_true = np.concatenate(ys)
    y_pred = np.concatenate(ps)
    y_prob = np.concatenate(pr)

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }

    if len(np.unique(y_true)) > 1:
        metrics["roc_auc"] = float(
            roc_auc_score(y_true, y_prob)
        )
    else:
        metrics["roc_auc"] = float("nan")

    return metrics


def measure_latency_ms(model, n_features, n_steps=10, runs=100):
    model.eval()
    x = torch.randn(1, n_steps, n_features, device=DEVICE)

    # warmup
    with torch.no_grad():
        for _ in range(10):
            _ = model(x)

    if DEVICE.type == "cuda":
        torch.cuda.synchronize()

    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(runs):
            _ = model(x)

    if DEVICE.type == "cuda":
        torch.cuda.synchronize()

    elapsed = time.perf_counter() - start
    return (elapsed / runs) * 1000.0


def train_one(cfg, X_train, y_train, X_val, y_val, n_features):
    set_seed(SEED)

    n_benign = max(int((y_train == 0).sum()), 1)
    n_attack = max(int((y_train == 1).sum()), 1)
    w_benign = (n_benign + n_attack) / (2.0 * n_benign)
    w_attack = (n_benign + n_attack) / (2.0 * n_attack)

    class_weights = torch.tensor(
        [w_benign, w_attack],
        dtype=torch.float32,
        device=DEVICE,
    )

    train_loader = make_loader(
        X_train,
        y_train,
        cfg["batch_size"],
        True,
    )
    val_loader = make_loader(
        X_val,
        y_val,
        cfg["batch_size"],
        False,
    )

    model = BaselineLSTM(
        n_features=n_features,
        hidden_size=cfg["hidden_size"],
        num_layers=cfg["num_layers"],
        dropout=cfg["dropout"],
    ).to(DEVICE)

    n_params = sum(p.numel() for p in model.parameters())

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg["learning_rate"],
        weight_decay=1e-5,
    )

    best_state = None
    best_val_f1 = -1.0
    best_epoch = 0
    no_improve = 0

    t0 = time.perf_counter()

    for epoch in range(1, EPOCHS + 1):
        model.train()
        for Xb, yb in train_loader:
            Xb = Xb.to(DEVICE)
            yb = yb.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(Xb), yb)
            loss.backward()
            optimizer.step()

        val_metrics = eval_f1(model, val_loader)

        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            best_epoch = epoch
            no_improve = 0
            best_state = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                break

    train_seconds = time.perf_counter() - t0

    model.load_state_dict(best_state)
    val_metrics = eval_f1(model, val_loader)
    latency_ms = measure_latency_ms(model, n_features)

    return {
        "config": cfg,
        "n_params": n_params,
        "best_epoch": best_epoch,
        "epochs_run": epoch,
        "train_seconds": train_seconds,
        "val_metrics": val_metrics,
        "latency_ms_per_window": latency_ms,
        "state_dict": best_state,
    }


def score_candidate(result):
    """
    Multi-criteria score for edge selection.
    Higher is better.
    """

    f1 = result["val_metrics"]["f1"]
    # Prefer compact / fast models when F1 is equal
    param_penalty = result["n_params"] / 100_000.0
    latency_penalty = result["latency_ms_per_window"] / 10.0
    time_penalty = result["train_seconds"] / 60.0

    return (
        f1 * 100.0
        - 0.5 * param_penalty
        - 0.5 * latency_penalty
        - 0.1 * time_penalty
    )


print("=" * 70)
print("HYPERPARAMETER TUNING — Pipeline C (LDA-LSTM)")
print("=" * 70)
print(f"Device: {DEVICE}")

X_train, y_train = load_binary_split(SEQ_DIR, "train")
X_val, y_val = load_binary_split(SEQ_DIR, "val")
X_test, y_test = load_binary_split(SEQ_DIR, "test")

n_steps = X_train.shape[1]
n_features = X_train.shape[2]

print(f"Features/window: {n_features} x {n_steps}")

keys = list(SEARCH_SPACE.keys())
grid = list(
    itertools.product(
        *[SEARCH_SPACE[k] for k in keys]
    )
)

print(f"Candidates: {len(grid)}")

results = []
best = None
best_score = -1e18

for i, values in enumerate(grid, start=1):
    cfg = dict(zip(keys, values))
    print(
        f"\n[{i}/{len(grid)}] {cfg}"
    )

    result = train_one(
        cfg,
        X_train,
        y_train,
        X_val,
        y_val,
        n_features,
    )

    s = score_candidate(result)
    result["selection_score"] = s

    # Drop state_dict from JSON-serializable copy later
    summary = {
        k: v
        for k, v in result.items()
        if k != "state_dict"
    }
    results.append(summary)

    print(
        f"  val_f1={result['val_metrics']['f1']:.4f}  "
        f"params={result['n_params']:,}  "
        f"lat={result['latency_ms_per_window']:.3f} ms  "
        f"train={result['train_seconds']:.2f}s  "
        f"score={s:.4f}"
    )

    if s > best_score:
        best_score = s
        best = result


# ------------------------------------------------------------
# Evaluate best on test
# ------------------------------------------------------------

best_model = BaselineLSTM(
    n_features=n_features,
    hidden_size=best["config"]["hidden_size"],
    num_layers=best["config"]["num_layers"],
    dropout=best["config"]["dropout"],
).to(DEVICE)

best_model.load_state_dict(best["state_dict"])

test_loader = make_loader(
    X_test,
    y_test,
    best["config"]["batch_size"],
    False,
)
test_metrics = eval_f1(best_model, test_loader)
latency_ms = measure_latency_ms(best_model, n_features)

ckpt_path = os.path.join(OUT_DIR, "best_lstm.pt")
torch.save(
    {
        "model_state": best["state_dict"],
        "model_type": "lstm",
        "n_features": n_features,
        "n_timesteps": n_steps,
        "hidden_size": best["config"]["hidden_size"],
        "num_layers": best["config"]["num_layers"],
        "dropout": best["config"]["dropout"],
        "config": best["config"],
        "val_metrics": best["val_metrics"],
        "test_metrics": test_metrics,
        "n_params": best["n_params"],
        "latency_ms_per_window": latency_ms,
        "pipeline": "C_LDA_LSTM_tuned",
    },
    ckpt_path,
)

# Sort leaderboard
leaderboard = sorted(
    results,
    key=lambda r: r["selection_score"],
    reverse=True,
)

out = {
    "search_space": SEARCH_SPACE,
    "best_config": best["config"],
    "best_n_params": best["n_params"],
    "best_val_metrics": best["val_metrics"],
    "best_test_metrics": test_metrics,
    "best_latency_ms_per_window": latency_ms,
    "best_selection_score": best_score,
    "leaderboard_top10": leaderboard[:10],
    "n_candidates": len(grid),
}

with open(
    os.path.join(OUT_DIR, "tuning_results.json"),
    "w",
    encoding="utf-8",
) as handle:
    json.dump(out, handle, indent=2)

print("\n" + "=" * 70)
print("BEST CONFIG (multi-criteria)")
print("=" * 70)
print(json.dumps(best["config"], indent=2))
print(f"Params     : {best['n_params']:,}")
print(f"Val F1     : {best['val_metrics']['f1']:.4f}")
print(f"Test F1    : {test_metrics['f1']:.4f}")
print(f"Test AUC   : {test_metrics['roc_auc']:.4f}")
print(f"Latency    : {latency_ms:.3f} ms / window")
print(f"Saved      : {ckpt_path}")
print("\nTuning complete.")
