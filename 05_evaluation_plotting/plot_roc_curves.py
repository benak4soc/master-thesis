# ============================================================
# Generate ROC plots for EACH trained model
#
# Usage:
#   python plot_roc_curves.py
#   python plot_roc_curves.py --model power_B_pca_lstm
#   python plot_roc_curves.py --combined
#
# Output:
#   figures/roc_curves/<model_name>.png
#   figures/roc_curves/combined_<modality>.png  (optional)
# ============================================================

import argparse
import os
import warnings
from collections import defaultdict

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_curve, auc

from lstm_common import build_model
from model_plot_config import MODEL_REGISTRY, iter_available_models

warnings.filterwarnings("ignore")

OUT_DIR = os.path.join("figures", "roc_curves")


def load_test_xy(seq_dir):
    path = os.path.join(seq_dir, "test_windows.npz")
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    data = np.load(path)
    return (
        data["X"].astype(np.float32),
        data["y_binary"].astype(np.int64),
    )


def predict_proba(ckpt_path, seq_dir, model_type_hint):
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

    probs, trues = [], []
    with torch.no_grad():
        for xb, yb in loader:
            logits = model(xb)
            p = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            probs.append(p)
            trues.append(yb.numpy())

    return np.concatenate(trues), np.concatenate(probs)


def save_roc(y_true, y_prob, title, out_path):
    if len(np.unique(y_true)) < 2:
        raise ValueError(
            "ROC requires both classes in y_true"
        )

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6.2, 5.2))
    plt.plot(
        fpr,
        tpr,
        lw=2.2,
        label=f"ROC curve (AUC = {roc_auc:.4f})",
    )
    plt.plot([0, 1], [0, 1], "k--", lw=1, label="Chance")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    return roc_auc


def save_combined(mod_curves, modality, out_dir):
    if not mod_curves:
        return

    plt.figure(figsize=(7.2, 5.6))
    for name, fpr, tpr, roc_auc in mod_curves:
        short = name.split("_", 1)[1] if "_" in name else name
        plt.plot(
            fpr,
            tpr,
            lw=1.8,
            label=f"{short} (AUC={roc_auc:.3f})",
        )
    plt.plot([0, 1], [0, 1], "k--", lw=1)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"Combined ROC — {modality}")
    plt.legend(fontsize=8, loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    out_path = os.path.join(out_dir, f"combined_{modality}.png")
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[OK] combined_{modality} -> {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Plot ROC curves for trained pipelines"
    )
    parser.add_argument("--model", default=None)
    parser.add_argument("--out-dir", default=OUT_DIR)
    parser.add_argument(
        "--combined",
        action="store_true",
        help="Also save one combined ROC per modality",
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
    print("ROC CURVES")
    print("=" * 70)

    by_modality = defaultdict(list)

    for name, model_dir, seq_dir, model_type, ckpt in models:
        try:
            y_true, y_prob = predict_proba(
                ckpt, seq_dir, model_type
            )
            out_path = os.path.join(args.out_dir, f"{name}.png")
            roc_auc = save_roc(
                y_true,
                y_prob,
                f"ROC Curve — {name}",
                out_path,
            )
            print(
                f"[OK] {name:35s} AUC={roc_auc:.4f} -> {out_path}"
            )

            if args.combined:
                fpr, tpr, _ = roc_curve(y_true, y_prob)
                modality = name.split("_")[0]
                by_modality[modality].append(
                    (name, fpr, tpr, roc_auc)
                )
        except Exception as exc:
            print(f"[SKIP] {name}: {exc}")

    if args.combined:
        for modality, curves in by_modality.items():
            save_combined(curves, modality, args.out_dir)

    print("\nDone.")
    print(f"Figures saved in: {args.out_dir}")


if __name__ == "__main__":
    main()
