# ============================================================
# Generate Training & Validation Loss plots for EACH model
#
# Usage:
#   python plot_train_val_loss.py
#   python plot_train_val_loss.py --model host_A_lstm
#
# Output:
#   figures/train_val_loss/<model_name>.png
#
# Reads training_history.json saved during training
# (no GPU/model reload required).
# ============================================================

import argparse
import json
import os
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from model_plot_config import MODEL_REGISTRY, iter_available_models

warnings.filterwarnings("ignore")

OUT_DIR = os.path.join("figures", "train_val_loss")


def load_history(model_dir):
    path = os.path.join(model_dir, "training_history.json")
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    with open(path, encoding="utf-8") as handle:
        history = json.load(handle)
    if not history:
        raise ValueError("Empty training history")
    return history


def save_loss_plot(history, title, out_path):
    epochs = [h["epoch"] for h in history]

    # Support both binary trainer keys and multiclass keys
    if "train_loss" in history[0]:
        train_loss = [h["train_loss"] for h in history]
    else:
        raise KeyError("train_loss not found in history")

    if "val_loss" in history[0]:
        val_loss = [h["val_loss"] for h in history]
    else:
        raise KeyError("val_loss not found in history")

    plt.figure(figsize=(7.0, 4.6))
    plt.plot(
        epochs,
        train_loss,
        marker="o",
        markersize=3,
        linewidth=2,
        label="Training loss",
    )
    plt.plot(
        epochs,
        val_loss,
        marker="s",
        markersize=3,
        linewidth=2,
        label="Validation loss",
    )
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()

    # Optional secondary panel info in companion json
    meta = {
        "epochs": len(epochs),
        "final_train_loss": float(train_loss[-1]),
        "final_val_loss": float(val_loss[-1]),
        "best_val_loss": float(min(val_loss)),
        "best_val_epoch": int(
            epochs[int(np_argmin(val_loss))]
        ),
    }
    return meta


def np_argmin(values):
    best_i = 0
    best_v = values[0]
    for i, v in enumerate(values):
        if v < best_v:
            best_v = v
            best_i = i
    return best_i


def main():
    parser = argparse.ArgumentParser(
        description="Plot train/val loss for trained pipelines"
    )
    parser.add_argument("--model", default=None)
    parser.add_argument("--out-dir", default=OUT_DIR)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Prefer registry entries that have history; include even if ckpt missing
    entries = []
    for name, model_dir, seq_dir, model_type in MODEL_REGISTRY:
        hist = os.path.join(model_dir, "training_history.json")
        if os.path.isfile(hist):
            entries.append((name, model_dir))

    # Also include any discovered via checkpoints (covers tuned models etc.)
    seen = {n for n, _ in entries}
    for name, model_dir, seq_dir, model_type, ckpt in iter_available_models():
        hist = os.path.join(model_dir, "training_history.json")
        if name not in seen and os.path.isfile(hist):
            entries.append((name, model_dir))

    if args.model:
        entries = [e for e in entries if e[0] == args.model]
        if not entries:
            known = [n for n, _ in entries] or [
                n for n, *_ in MODEL_REGISTRY
            ]
            raise SystemExit(
                f"Model '{args.model}' not found / no history. "
                f"Known registry names: {[n for n, *_ in MODEL_REGISTRY]}"
            )

    print("=" * 70)
    print("TRAINING / VALIDATION LOSS")
    print("=" * 70)

    for name, model_dir in entries:
        try:
            history = load_history(model_dir)
            out_path = os.path.join(args.out_dir, f"{name}.png")
            meta = save_loss_plot(
                history,
                f"Training & Validation Loss — {name}",
                out_path,
            )
            with open(
                os.path.join(args.out_dir, f"{name}_loss_meta.json"),
                "w",
                encoding="utf-8",
            ) as handle:
                json.dump(meta, handle, indent=2)

            print(
                f"[OK] {name:35s} "
                f"epochs={meta['epochs']} "
                f"best_val={meta['best_val_loss']:.4f} "
                f"@ ep {meta['best_val_epoch']} -> {out_path}"
            )
        except Exception as exc:
            print(f"[SKIP] {name}: {exc}")

    print("\nDone.")
    print(f"Figures saved in: {args.out_dir}")


if __name__ == "__main__":
    main()
