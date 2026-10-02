# ============================================================
# Shared LSTM trainer for Pipelines A/B/C/...
# ============================================================

import os
import json
import time
import warnings

import numpy as np

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


class BaselineLSTM(nn.Module):
    """LSTM encoder + linear classifier (last hidden state)."""

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
        _, (h_n, _) = self.lstm(x)
        h_last = self.dropout(h_n[-1])
        return self.fc(h_last)


class BiLSTMClassifier(nn.Module):
    """
    Pipeline D: Bidirectional LSTM + linear classifier.
    Concatenates last forward and backward hidden states.
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
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size * 2, n_classes)

    def forward(self, x):
        # h_n: (num_layers * 2, batch, hidden)
        _, (h_n, _) = self.lstm(x)
        # last layer forward / backward
        h_fwd = h_n[-2]
        h_bwd = h_n[-1]
        h = torch.cat([h_fwd, h_bwd], dim=1)
        h = self.dropout(h)
        return self.fc(h)


class CNNBiLSTMAttention(nn.Module):
    """
    Pipeline E: 1D CNN -> BiLSTM -> temporal attention -> classifier.

    CNN extracts local patterns along time.
    BiLSTM models bidirectional temporal context.
    Attention weights informative timesteps.
    """

    def __init__(
        self,
        n_features,
        hidden_size=64,
        num_layers=1,
        dropout=0.2,
        n_classes=2,
        cnn_channels=64,
        cnn_kernel=3,
    ):
        super().__init__()

        padding = cnn_kernel // 2

        self.conv = nn.Sequential(
            nn.Conv1d(
                in_channels=n_features,
                out_channels=cnn_channels,
                kernel_size=cnn_kernel,
                padding=padding,
            ),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.lstm = nn.LSTM(
            input_size=cnn_channels,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        attn_dim = hidden_size * 2
        self.attn_fc = nn.Linear(attn_dim, attn_dim)
        self.attn_v = nn.Linear(attn_dim, 1, bias=False)

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(attn_dim, n_classes)

    def forward(self, x):
        # x: (batch, timesteps, features)
        # Conv1d expects (batch, channels, length)
        x = x.transpose(1, 2)
        x = self.conv(x)
        x = x.transpose(1, 2)

        # lstm_out: (batch, timesteps, 2*hidden)
        lstm_out, _ = self.lstm(x)

        # Additive attention over time
        scores = self.attn_v(torch.tanh(self.attn_fc(lstm_out)))
        # scores: (batch, timesteps, 1)
        weights = torch.softmax(scores, dim=1)
        context = torch.sum(weights * lstm_out, dim=1)

        context = self.dropout(context)
        return self.fc(context)


def build_model(model_type, n_features, **kwargs):
    model_type = model_type.lower()

    if model_type in ("lstm", "baseline"):
        return BaselineLSTM(n_features=n_features, **kwargs)

    if model_type in ("bilstm", "bi-lstm"):
        return BiLSTMClassifier(n_features=n_features, **kwargs)

    if model_type in (
        "cnn_bilstm_attention",
        "cnn-bilstm-attention",
        "cnn_bilstm_attn",
    ):
        return CNNBiLSTMAttention(
            n_features=n_features,
            **kwargs
        )

    raise ValueError(f"Unknown model_type: {model_type}")


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_binary_split(seq_dir, name):
    path = os.path.join(seq_dir, f"{name}_windows.npz")
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Missing {path}."
        )
    data = np.load(path)
    return (
        data["X"].astype(np.float32),
        data["y_binary"].astype(np.int64),
    )


def train_binary_lstm(
    *,
    pipeline_name,
    seq_dir,
    model_dir,
    model_type="lstm",
    hidden_size=64,
    num_layers=1,
    dropout=0.2,
    batch_size=64,
    epochs=40,
    learning_rate=1e-3,
    weight_decay=1e-5,
    patience=8,
    seed=42,
    model_kwargs=None,
):
    """
    Train / validate / test a binary sequence model on windows.
    model_type: 'lstm' | 'bilstm' | 'cnn_bilstm_attention'
    """

    os.makedirs(model_dir, exist_ok=True)
    set_seed(seed)
    model_kwargs = model_kwargs or {}

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("=" * 70)
    print(pipeline_name)
    print("=" * 70)
    print(f"Device : {device}")
    print(f"Data   : {seq_dir}")
    print(f"Model  : {model_type}")

    X_train, y_train = load_binary_split(seq_dir, "train")
    X_val, y_val = load_binary_split(seq_dir, "val")
    X_test, y_test = load_binary_split(seq_dir, "test")

    n_timesteps = X_train.shape[1]
    n_features = X_train.shape[2]

    print(f"\nTrain : X={X_train.shape}  y={np.bincount(y_train)}")
    print(f"Val   : X={X_val.shape}  y={np.bincount(y_val)}")
    print(f"Test  : X={X_test.shape}  y={np.bincount(y_test)}")

    n_benign = max(int((y_train == 0).sum()), 1)
    n_attack = max(int((y_train == 1).sum()), 1)
    w_benign = (n_benign + n_attack) / (2.0 * n_benign)
    w_attack = (n_benign + n_attack) / (2.0 * n_attack)

    class_weights = torch.tensor(
        [w_benign, w_attack],
        dtype=torch.float32,
        device=device,
    )

    print(
        f"\nClass weights: benign={w_benign:.3f}, "
        f"attack={w_attack:.3f}"
    )

    def make_loader(X, y, shuffle):
        return DataLoader(
            TensorDataset(
                torch.from_numpy(X),
                torch.from_numpy(y),
            ),
            batch_size=batch_size,
            shuffle=shuffle,
        )

    train_loader = make_loader(X_train, y_train, True)
    val_loader = make_loader(X_val, y_val, False)
    test_loader = make_loader(X_test, y_test, False)

    build_kwargs = {
        "hidden_size": hidden_size,
        "num_layers": num_layers,
        "dropout": dropout,
        **model_kwargs,
    }

    model = build_model(
        model_type,
        n_features=n_features,
        **build_kwargs,
    ).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    n_params = sum(
        p.numel() for p in model.parameters()
    )
    print(f"Model parameters: {n_params:,}")

    def run_epoch(loader, train_mode=True):
        model.train(train_mode)
        total_loss = 0.0
        n_batches = 0
        all_true, all_pred, all_prob = [], [], []

        context = (
            torch.enable_grad()
            if train_mode
            else torch.no_grad()
        )

        with context:
            for X_batch, y_batch in loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)

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
                precision_score(y_true, y_pred, zero_division=0)
            ),
            "recall": float(
                recall_score(y_true, y_pred, zero_division=0)
            ),
            "f1": float(
                f1_score(y_true, y_pred, zero_division=0)
            ),
        }

        if len(np.unique(y_true)) > 1:
            metrics["roc_auc"] = float(
                roc_auc_score(y_true, y_prob)
            )
        else:
            metrics["roc_auc"] = float("nan")

        return metrics, y_true, y_pred, y_prob

    history = []
    best_val_f1 = -1.0
    best_epoch = -1
    epochs_no_improve = 0
    best_path = os.path.join(model_dir, "best_lstm.pt")

    print("\nTraining...")
    train_start = time.perf_counter()

    for epoch in range(1, epochs + 1):
        train_metrics, _, _, _ = run_epoch(
            train_loader, True
        )
        val_metrics, _, _, _ = run_epoch(
            val_loader, False
        )

        history.append(
            {
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
        )

        print(
            f"Epoch {epoch:02d}/{epochs} | "
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
                    "model_type": model_type,
                    "n_features": n_features,
                    "n_timesteps": n_timesteps,
                    "hidden_size": hidden_size,
                    "num_layers": num_layers,
                    "dropout": dropout,
                    "model_kwargs": model_kwargs,
                    "best_val_f1": best_val_f1,
                    "epoch": epoch,
                    "pipeline": pipeline_name,
                    "n_params": n_params,
                },
                best_path,
            )
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(
                    f"\nEarly stopping at epoch {epoch} "
                    f"(best val F1={best_val_f1:.4f} "
                    f"@ epoch {best_epoch})"
                )
                break

    train_seconds = time.perf_counter() - train_start

    checkpoint = torch.load(
        best_path,
        map_location=device,
        weights_only=False,
    )
    model.load_state_dict(checkpoint["model_state"])

    test_metrics, y_true, y_pred, y_prob = run_epoch(
        test_loader, False
    )

    print("\n" + "=" * 70)
    print(f"TEST RESULTS — {pipeline_name}")
    print("=" * 70)
    print(f"Features  : {n_features}")
    print(f"Params    : {n_params:,}")
    print(f"Accuracy  : {test_metrics['accuracy']:.4f}")
    print(f"Precision : {test_metrics['precision']:.4f}")
    print(f"Recall    : {test_metrics['recall']:.4f}")
    print(f"F1-score  : {test_metrics['f1']:.4f}")
    print(f"ROC-AUC   : {test_metrics['roc_auc']:.4f}")
    print(f"Train time: {train_seconds:.2f} s")
    print("\nConfusion matrix [[TN FP] [FN TP]]:")
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

    results = {
        "pipeline": pipeline_name,
        "model_type": model_type,
        "device": str(device),
        "seq_dir": seq_dir,
        "n_params": n_params,
        "window_features_shape": {
            "timesteps": n_timesteps,
            "features": n_features,
        },
        "hyperparameters": {
            "hidden_size": hidden_size,
            "num_layers": num_layers,
            "dropout": dropout,
            "batch_size": batch_size,
            "epochs_max": epochs,
            "epochs_run": len(history),
            "best_epoch": best_epoch,
            "learning_rate": learning_rate,
            "patience": patience,
            "model_kwargs": model_kwargs,
            "class_weights": {
                "benign": w_benign,
                "attack": w_attack,
            },
        },
        "training_time_seconds": train_seconds,
        "best_val_f1": best_val_f1,
        "test_metrics": test_metrics,
        "confusion_matrix": confusion_matrix(
            y_true, y_pred
        ).tolist(),
    }

    results_path = os.path.join(model_dir, "test_results.json")
    with open(results_path, "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)

    history_path = os.path.join(
        model_dir, "training_history.json"
    )
    with open(history_path, "w", encoding="utf-8") as handle:
        json.dump(history, handle, indent=2)

    print(f"\nSaved model : {best_path}")
    print(f"Saved results: {results_path}")
    print(f"Saved history: {history_path}")

    return results
