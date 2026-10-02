# ============================================================
# Generate Confusion Matrix for EACH trained model
#
# Usage:
#   python plot_confusion_matrices.py
#   python plot_confusion_matrices.py --model host_C_lda_lstm
#
# Output:
#   figures/confusion_matrices/<model_name>.png
# ============================================================

import argparse
import os
import warnings

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from lstm_common import build_model
from model_plot_config import MODEL_REGISTRY, iter_available_models

warnings.filterwarnings("ignore")

OUT_DIR = os.path.join("figures", "confusion_matrices")
LABELS = ["Benign", "Attack"]


def load_test_xy(seq_dir):
    path = os.path.join(seq_dir, "test_windows.npz")
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    data = np.load(path)
    return (
        data["X"].astype(np.float32),
        data["y_binary"].astype(np.int64),
    )


def predict(ckpt_path, seq_dir, model_type_hint):
    ckpt = torch.load(
        ckpt_path,
        map_location="cpu",
        weights_only=False,
    )
    X, y = load_test_xy(seq_dir)

    model_type = ckpt.get("model_type", model_type_hint)
    kwargs = {
        "hidden_size": int(ckpt.get("hidden_size", 64)),
        "num_layers": int(ckpt.get("num_layers", 1)),
        "dropout": 0.0,
    }
    mk = ckpt.get("model_kwargs") or {}
    if "cnn" in str(model_type).lower():
        kwargs.update(mk)
        if "cnn_channels" not in kwargs:
            # sensible defaults if missing in older checkpoints
            kwargs["cnn_channels"] = 64 if X.shape[2] > 4 else 32
            kwargs["cnn_kernel"] = 3

    model = build_model(
        model_type,
        n_features=X.shape[2],
        **kwargs,
    )
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    loader = DataLoader(
        TensorDataset(torch.from_numpy(X), torch.from_numpy(y)),
        batch_size=256,
        shuffle=False,
    )

    preds, trues = [], []
    with torch.no_grad():
        for xb, yb in loader:
            logits = model(xb)
            pred = torch.argmax(logits, dim=1).cpu().numpy()
            preds.append(pred)
            trues.append(yb.numpy())

    return np.concatenate(trues), np.concatenate(preds)


def save_confusion_matrix(y_true, y_pred, title, out_path):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(5.5, 4.8))
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=LABELS,
    )
    disp.plot(ax=ax, cmap="Blues", colorbar=False, values_format="d")
    ax.set_title(title)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    return cm


def main():
    parser = argparse.ArgumentParser(
        description="Plot confusion matrices for trained pipelines"
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Optional single model name from registry "
        "(e.g. host_C_lda_lstm)",
    )
    parser.add_argument(
        "--out-dir",
        default=OUT_DIR,
    )
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    models = list(iter_available_models())
    if args.model:
        models = [m for m in models if m[0] == args.model]
        if not models:
            known = [n for n, *_ in MODEL_REGISTRY]
            raise SystemExit(
                f"Model '{args.model}' not found. Known: {known}"
            )

    print("=" * 70)
    print("CONFUSION MATRICES")
    print("=" * 70)

    for name, model_dir, seq_dir, model_type, ckpt in models:
        try:
            y_true, y_pred = predict(ckpt, seq_dir, model_type)
            out_path = os.path.join(args.out_dir, f"{name}.png")
            cm = save_confusion_matrix(
                y_true,
                y_pred,
                f"Confusion Matrix — {name}",
                out_path,
            )
            print(
                f"[OK] {name:35s} CM={cm.tolist()} -> {out_path}"
            )
        except Exception as exc:
            print(f"[SKIP] {name}: {exc}")

    print("\nDone.")
    print(f"Figures saved in: {args.out_dir}")


if __name__ == "__main__":
    main()
