# ============================================================
# CICEVSE2024 - Pipeline A: Baseline LSTM (no PCA / LDA)
# Master's Thesis
#
# Input :
#   processed_EVSE_B_Host/sequences/{train,val,test}_windows.npz
#
# Task  :
#   Binary attack detection (Benign=0, Attack=1)
#
# Metrics (thesis §15):
#   Accuracy, Precision, Recall, F1, ROC-AUC, Training Time
# ============================================================

import os
import json
import time
import warnings

import numpy as np
import joblib

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
)

warnings.filterwarnings("ignore")


# ============================================================
# CONFIGURATION
# ============================================================

SEQ_DIR = os.path.join(
    "processed_EVSE_B_Host",
    "sequences"
)

MODEL_DIR = os.path.join(
    "models",
    "pipeline_A_lstm"
)

os.makedirs(MODEL_DIR, exist_ok=True)

# Model / training hyperparameters
HIDDEN_SIZE = 64
NUM_LAYERS = 1
DROPOUT = 0.2
BATCH_SIZE = 64
EPOCHS = 40
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5
PATIENCE = 8
SEED = 42

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(SEED)


print("=" * 70)
print("PIPELINE A — BASELINE LSTM (no dimensionality reduction)")
print("=" * 70)
print(f"Device: {DEVICE}")


# ============================================================
# 1. LOAD WINDOWS
# ============================================================

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
    return (
        data["X"].astype(np.float32),
        data["y_binary"].astype(np.int64),
    )


X_train, y_train = load_split("train")
X_val, y_val = load_split("val")
X_test, y_test = load_split("test")

n_timesteps = X_train.shape[1]
n_features = X_train.shape[2]

print(f"\nTrain : X={X_train.shape}  y={np.bincount(y_train)}")
print(f"Val   : X={X_val.shape}  y={np.bincount(y_val)}")
print(f"Test  : X={X_test.shape}  y={np.bincount(y_test)}")


# Class weights (handle imbalance)
n_benign = max(int((y_train == 0).sum()), 1)
n_attack = max(int((y_train == 1).sum()), 1)
# Inverse-frequency weights
w_benign = (n_benign + n_attack) / (2.0 * n_benign)
w_attack = (n_benign + n_attack) / (2.0 * n_attack)

class_weights = torch.tensor(
    [w_benign, w_attack],
    dtype=torch.float32,
    device=DEVICE
)

print(
    f"\nClass weights: benign={w_benign:.3f}, "
    f"attack={w_attack:.3f}"
)


# ============================================================
# 2. DATALOADERS
# ============================================================

def make_loader(X, y, shuffle):
    dataset = TensorDataset(
        torch.from_numpy(X),
        torch.from_numpy(y),
    )
    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        drop_last=False,
    )


train_loader = make_loader(X_train, y_train, shuffle=True)
val_loader = make_loader(X_val, y_val, shuffle=False)
test_loader = make_loader(X_test, y_test, shuffle=False)


# ============================================================
# 3. MODEL
# ============================================================

class BaselineLSTM(nn.Module):
    """
    Pipeline A: LSTM encoder + linear classifier.
    Uses the last hidden state for sequence classification.
    """

    def __init__(
        self,
        n_features,
        hidden_size=64,
        num_layers=1,
        dropout=0.2,
        n_classes=2,
    ):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, n_classes)

    def forward(self, x):
        # x: (batch, timesteps, features)
        out, (h_n, _) = self.lstm(x)
        # last layer hidden state
        h_last = h_n[-1]
        h_last = self.dropout(h_last)
        logits = self.fc(h_last)
        return logits


model = BaselineLSTM(
    n_features=n_features,
    hidden_size=HIDDEN_SIZE,
    num_layers=NUM_LAYERS,
    dropout=DROPOUT,
).to(DEVICE)

criterion = nn.CrossEntropyLoss(
    weight=class_weights
)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
)

print(
    f"\nModel parameters: "
    f"{sum(p.numel() for p in model.parameters()):,}"
)


# ============================================================
# 4. TRAIN / EVAL HELPERS
# ============================================================

def run_epoch(loader, train_mode=True):
    if train_mode:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    n_batches = 0

    all_true = []
    all_pred = []
    all_prob = []

    context = torch.enable_grad() if train_mode else torch.no_grad()

    with context:
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)

            if train_mode:
                optimizer.zero_grad()

            logits = model(X_batch)
            loss = criterion(logits, y_batch)

            if train_mode:
                loss.backward()
                optimizer.step()

            probs = torch.softmax(logits, dim=1)[:, 1]
            preds = torch.argmax(logits, dim=1)

            total_loss += loss.item()
            n_batches += 1

            all_true.append(y_batch.detach().cpu().numpy())
            all_pred.append(preds.detach().cpu().numpy())
            all_prob.append(probs.detach().cpu().numpy())

    y_true = np.concatenate(all_true)
    y_pred = np.concatenate(all_pred)
    y_prob = np.concatenate(all_prob)

    metrics = {
        "loss": total_loss / max(n_batches, 1),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(
            precision_score(
                y_true,
                y_pred,
                zero_division=0
            )
        ),
        "recall": float(
            recall_score(
                y_true,
                y_pred,
                zero_division=0
            )
        ),
        "f1": float(
            f1_score(
                y_true,
                y_pred,
                zero_division=0
            )
        ),
    }

    # ROC-AUC needs both classes present
    if len(np.unique(y_true)) > 1:
        metrics["roc_auc"] = float(
            roc_auc_score(y_true, y_prob)
        )
    else:
        metrics["roc_auc"] = float("nan")

    return metrics, y_true, y_pred, y_prob


# ============================================================
# 5. TRAINING LOOP (early stopping on val F1)
# ============================================================

history = []
best_val_f1 = -1.0
best_epoch = -1
epochs_no_improve = 0

best_path = os.path.join(
    MODEL_DIR,
    "best_lstm.pt"
)

print("\nTraining...")
train_start = time.perf_counter()

for epoch in range(1, EPOCHS + 1):
    train_metrics, _, _, _ = run_epoch(
        train_loader,
        train_mode=True
    )
    val_metrics, _, _, _ = run_epoch(
        val_loader,
        train_mode=False
    )

    row = {
        "epoch": epoch,
        "train_loss": train_metrics["loss"],
        "train_f1": train_metrics["f1"],
        "val_loss": val_metrics["loss"],
        "val_accuracy": val_metrics["accuracy"],
        "val_precision": val_metrics["precision"],
        "val_recall": val_metrics["recall"],
        "val_f1": val_metrics["f1"],
        "val_roc_auc": val_metrics["roc_auc"],
    }
    history.append(row)

    print(
        f"Epoch {epoch:02d}/{EPOCHS} | "
        f"train_loss={train_metrics['loss']:.4f} "
        f"train_f1={train_metrics['f1']:.4f} | "
        f"val_loss={val_metrics['loss']:.4f} "
        f"val_f1={val_metrics['f1']:.4f} "
        f"val_auc={val_metrics['roc_auc']:.4f}"
    )

    if val_metrics["f1"] > best_val_f1:
        best_val_f1 = val_metrics["f1"]
        best_epoch = epoch
        epochs_no_improve = 0

        torch.save(
            {
                "model_state": model.state_dict(),
                "n_features": n_features,
                "n_timesteps": n_timesteps,
                "hidden_size": HIDDEN_SIZE,
                "num_layers": NUM_LAYERS,
                "dropout": DROPOUT,
                "best_val_f1": best_val_f1,
                "epoch": epoch,
            },
            best_path,
        )
    else:
        epochs_no_improve += 1

        if epochs_no_improve >= PATIENCE:
            print(
                f"\nEarly stopping at epoch {epoch} "
                f"(best val F1={best_val_f1:.4f} @ epoch {best_epoch})"
            )
            break

train_seconds = time.perf_counter() - train_start

print(
    f"\nTraining time: {train_seconds:.2f} s "
    f"(best epoch={best_epoch})"
)


# ============================================================
# 6. TEST EVALUATION (best checkpoint)
# ============================================================

checkpoint = torch.load(
    best_path,
    map_location=DEVICE,
    weights_only=False
)

model.load_state_dict(
    checkpoint["model_state"]
)

test_metrics, y_true, y_pred, y_prob = run_epoch(
    test_loader,
    train_mode=False
)

print("\n" + "=" * 70)
print("TEST RESULTS — Pipeline A (Baseline LSTM)")
print("=" * 70)

print(f"Accuracy  : {test_metrics['accuracy']:.4f}")
print(f"Precision : {test_metrics['precision']:.4f}")
print(f"Recall    : {test_metrics['recall']:.4f}")
print(f"F1-score  : {test_metrics['f1']:.4f}")
print(f"ROC-AUC   : {test_metrics['roc_auc']:.4f}")
print(f"Train time: {train_seconds:.2f} s")

print("\nConfusion matrix [ [TN FP] [FN TP] ]:")
print(confusion_matrix(y_true, y_pred))

print("\nClassification report:")
print(
    classification_report(
        y_true,
        y_pred,
        target_names=["Benign", "Attack"],
        digits=4,
        zero_division=0,
    )
)


# ============================================================
# 7. SAVE ARTIFACTS
# ============================================================

results = {
    "pipeline": "A_LSTM_baseline",
    "device": str(DEVICE),
    "window_features_shape": {
        "timesteps": n_timesteps,
        "features": n_features,
    },
    "hyperparameters": {
        "hidden_size": HIDDEN_SIZE,
        "num_layers": NUM_LAYERS,
        "dropout": DROPOUT,
        "batch_size": BATCH_SIZE,
        "epochs_max": EPOCHS,
        "epochs_run": len(history),
        "best_epoch": best_epoch,
        "learning_rate": LEARNING_RATE,
        "patience": PATIENCE,
        "class_weights": {
            "benign": w_benign,
            "attack": w_attack,
        },
    },
    "training_time_seconds": train_seconds,
    "best_val_f1": best_val_f1,
    "test_metrics": test_metrics,
    "confusion_matrix": confusion_matrix(
        y_true,
        y_pred
    ).tolist(),
}

results_path = os.path.join(
    MODEL_DIR,
    "test_results.json"
)

with open(
    results_path,
    "w",
    encoding="utf-8"
) as handle:
    json.dump(results, handle, indent=2)

history_path = os.path.join(
    MODEL_DIR,
    "training_history.json"
)

with open(
    history_path,
    "w",
    encoding="utf-8"
) as handle:
    json.dump(history, handle, indent=2)

print(f"\nSaved model : {best_path}")
print(f"Saved results: {results_path}")
print(f"Saved history: {history_path}")
print("\nPipeline A complete.")
