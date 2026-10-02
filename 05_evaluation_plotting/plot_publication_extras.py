# ============================================================
# Extra publication-quality plots for the EVSE IDS thesis
#
# Beyond CM / ROC / Loss, this script generates:
#   1. Precision-Recall curves (per model)
#   2. Metric comparison bar charts (per modality)
#   3. Efficiency trade-off plots (params / time / features vs F1)
#   4. Validation F1 learning curves
#   5. PCA / LDA explained-variance plots
#   6. Class / Attack_Group distribution charts
#   7. Multi-class performance bars
#   8. Metrics heatmap across all pipelines
#
# Usage:
#   python plot_publication_extras.py
# ============================================================

import json
import os
import warnings
from collections import defaultdict

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (
    precision_recall_curve,
    average_precision_score,
)

from lstm_common import build_model
from model_plot_config import iter_available_models

warnings.filterwarnings("ignore")

OUT = os.path.join("figures", "publication_extras")
os.makedirs(OUT, exist_ok=True)

PIPE_SHORT = {
    "A": "A-LSTM",
    "B": "B-PCA-LSTM",
    "C": "C-LDA-LSTM",
    "D": "D-LDA-BiLSTM",
    "E": "E-CNN-BiLSTM-Attn",
}


def ensure(path):
    os.makedirs(path, exist_ok=True)
    return path


def savefig(path):
    plt.tight_layout()
    plt.savefig(path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"[OK] {path}")


def load_test_results(model_dir):
    path = os.path.join(model_dir, "test_results.json")
    if not os.path.isfile(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_history(model_dir):
    path = os.path.join(model_dir, "training_history.json")
    if not os.path.isfile(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def parse_name(name):
    # host_A_lstm / network_B_pca_lstm / power_E_...
    parts = name.split("_")
    modality = parts[0]
    pipe = parts[1]
    return modality, pipe


def predict_proba(ckpt_path, seq_dir, model_type_hint):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    data = np.load(os.path.join(seq_dir, "test_windows.npz"))
    X = data["X"].astype(np.float32)
    y = data["y_binary"].astype(np.int64)

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

    model = build_model(model_type, n_features=X.shape[2], **kwargs)
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
            p = torch.softmax(model(xb), dim=1)[:, 1].cpu().numpy()
            probs.append(p)
            trues.append(yb.numpy())
    return np.concatenate(trues), np.concatenate(probs)


# ------------------------------------------------------------
# 1) Precision-Recall curves
# ------------------------------------------------------------
def plot_precision_recall():
    folder = ensure(os.path.join(OUT, "precision_recall"))
    by_mod = defaultdict(list)

    for name, model_dir, seq_dir, model_type, ckpt in iter_available_models():
        try:
            y_true, y_prob = predict_proba(ckpt, seq_dir, model_type)
            if len(np.unique(y_true)) < 2:
                continue
            prec, rec, _ = precision_recall_curve(y_true, y_prob)
            ap = average_precision_score(y_true, y_prob)

            plt.figure(figsize=(6.2, 5.0))
            plt.plot(rec, prec, lw=2.2, label=f"AP = {ap:.4f}")
            plt.xlabel("Recall")
            plt.ylabel("Precision")
            plt.title(f"Precision-Recall — {name}")
            plt.legend(loc="lower left")
            plt.grid(True, alpha=0.3)
            savefig(os.path.join(folder, f"{name}.png"))

            modality, pipe = parse_name(name)
            by_mod[modality].append(
                (PIPE_SHORT.get(pipe, pipe), rec, prec, ap)
            )
        except Exception as exc:
            print(f"[SKIP PR] {name}: {exc}")

    for modality, curves in by_mod.items():
        plt.figure(figsize=(7.2, 5.4))
        for label, rec, prec, ap in curves:
            plt.plot(rec, prec, lw=1.8, label=f"{label} (AP={ap:.3f})")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title(f"Combined Precision-Recall — {modality}")
        plt.legend(fontsize=8, loc="lower left")
        plt.grid(True, alpha=0.3)
        savefig(os.path.join(folder, f"combined_{modality}.png"))


# ------------------------------------------------------------
# 2) Metric comparison bars
# ------------------------------------------------------------
def collect_metrics():
    rows = []
    for name, model_dir, seq_dir, model_type, ckpt in iter_available_models():
        res = load_test_results(model_dir)
        if not res:
            continue
        m = res["test_metrics"]
        modality, pipe = parse_name(name)
        rows.append(
            {
                "name": name,
                "modality": modality,
                "pipe": PIPE_SHORT.get(pipe, pipe),
                "features": res["window_features_shape"]["features"],
                "params": res.get("n_params")
                or {
                    "host_A_lstm": 73346,
                    "host_B_pca_lstm": 23682,
                    "host_C_lda_lstm": 17794,
                    "host_D_lda_bilstm": 35586,
                    "host_E_cnn_bilstm_attention": 125762,
                }.get(name),
                "accuracy": m["accuracy"],
                "precision": m["precision"],
                "recall": m["recall"],
                "f1": m["f1"],
                "roc_auc": m["roc_auc"],
                "train_seconds": res["training_time_seconds"],
            }
        )
    return rows


def plot_metric_bars(rows):
    folder = ensure(os.path.join(OUT, "metric_bars"))
    metrics = ["accuracy", "precision", "recall", "f1", "roc_auc"]

    for modality in ["host", "network", "power"]:
        subset = [r for r in rows if r["modality"] == modality]
        if not subset:
            continue
        labels = [r["pipe"] for r in subset]
        x = np.arange(len(labels))
        width = 0.15

        plt.figure(figsize=(10, 5.2))
        for i, metric in enumerate(metrics):
            vals = [r[metric] for r in subset]
            plt.bar(x + (i - 2) * width, vals, width, label=metric.upper())
        plt.xticks(x, labels, rotation=15, ha="right")
        plt.ylim(0.95 if modality != "power" else 0.96, 1.005)
        if modality == "power":
            plt.ylim(0.96, 1.005)
        plt.ylabel("Score")
        plt.title(f"Test Metrics Comparison — {modality}")
        plt.legend(ncol=5, fontsize=8)
        plt.grid(True, axis="y", alpha=0.3)
        savefig(os.path.join(folder, f"metrics_{modality}.png"))

        # dedicated F1-only bar (clean for papers)
        plt.figure(figsize=(8, 4.6))
        f1s = [r["f1"] for r in subset]
        bars = plt.bar(labels, f1s, color="#4C78A8")
        plt.ylabel("F1-score")
        plt.title(f"F1-score by Pipeline — {modality}")
        plt.xticks(rotation=15, ha="right")
        for b, v in zip(bars, f1s):
            plt.text(
                b.get_x() + b.get_width() / 2,
                v + 0.001,
                f"{v:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
        plt.ylim(min(f1s) - 0.02, 1.01)
        plt.grid(True, axis="y", alpha=0.3)
        savefig(os.path.join(folder, f"f1_{modality}.png"))


# ------------------------------------------------------------
# 3) Efficiency trade-offs
# ------------------------------------------------------------
def plot_efficiency(rows):
    folder = ensure(os.path.join(OUT, "efficiency"))

    # params vs F1
    plt.figure(figsize=(7.5, 5.2))
    for modality, marker in [
        ("host", "o"),
        ("network", "s"),
        ("power", "^"),
    ]:
        subset = [r for r in rows if r["modality"] == modality and r["params"]]
        if not subset:
            continue
        plt.scatter(
            [r["params"] for r in subset],
            [r["f1"] for r in subset],
            s=70,
            marker=marker,
            label=modality,
        )
        for r in subset:
            plt.annotate(
                r["pipe"].split("-")[0],
                (r["params"], r["f1"]),
                textcoords="offset points",
                xytext=(5, 4),
                fontsize=7,
            )
    plt.xscale("log")
    plt.xlabel("Parameters (log scale)")
    plt.ylabel("Test F1-score")
    plt.title("Efficiency Trade-off: Parameters vs F1")
    plt.legend()
    plt.grid(True, alpha=0.3)
    savefig(os.path.join(folder, "params_vs_f1.png"))

    # train time vs F1
    plt.figure(figsize=(7.5, 5.2))
    for modality, marker in [
        ("host", "o"),
        ("network", "s"),
        ("power", "^"),
    ]:
        subset = [r for r in rows if r["modality"] == modality]
        plt.scatter(
            [r["train_seconds"] for r in subset],
            [r["f1"] for r in subset],
            s=70,
            marker=marker,
            label=modality,
        )
        for r in subset:
            plt.annotate(
                r["pipe"].split("-")[0],
                (r["train_seconds"], r["f1"]),
                textcoords="offset points",
                xytext=(5, 4),
                fontsize=7,
            )
    plt.xlabel("Training time (s)")
    plt.ylabel("Test F1-score")
    plt.title("Efficiency Trade-off: Training Time vs F1")
    plt.legend()
    plt.grid(True, alpha=0.3)
    savefig(os.path.join(folder, "train_time_vs_f1.png"))

    # features vs F1
    plt.figure(figsize=(7.5, 5.2))
    for modality, marker in [
        ("host", "o"),
        ("network", "s"),
        ("power", "^"),
    ]:
        subset = [r for r in rows if r["modality"] == modality]
        plt.scatter(
            [r["features"] for r in subset],
            [r["f1"] for r in subset],
            s=70,
            marker=marker,
            label=modality,
        )
        for r in subset:
            plt.annotate(
                r["pipe"].split("-")[0],
                (r["features"], r["f1"]),
                textcoords="offset points",
                xytext=(5, 4),
                fontsize=7,
            )
    plt.xlabel("Input feature dimension")
    plt.ylabel("Test F1-score")
    plt.title("Feature Compactness vs Detection Performance")
    plt.legend()
    plt.grid(True, alpha=0.3)
    savefig(os.path.join(folder, "features_vs_f1.png"))

    # grouped bars: params and features by pipeline/modality
    for modality in ["host", "network", "power"]:
        subset = [r for r in rows if r["modality"] == modality and r["params"]]
        if not subset:
            continue
        labels = [r["pipe"] for r in subset]
        fig, ax1 = plt.subplots(figsize=(9, 4.8))
        x = np.arange(len(labels))
        ax1.bar(x - 0.2, [r["features"] for r in subset], 0.4, color="#4C78A8", label="Features")
        ax1.set_ylabel("Features")
        ax2 = ax1.twinx()
        ax2.bar(x + 0.2, [r["params"] for r in subset], 0.4, color="#F58518", label="Params")
        ax2.set_ylabel("Parameters")
        ax1.set_xticks(x)
        ax1.set_xticklabels(labels, rotation=15, ha="right")
        ax1.set_title(f"Model Complexity — {modality}")
        h1, l1 = ax1.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax1.legend(h1 + h2, l1 + l2, loc="upper left")
        savefig(os.path.join(folder, f"complexity_{modality}.png"))


# ------------------------------------------------------------
# 4) Validation F1 learning curves
# ------------------------------------------------------------
def plot_val_f1_curves():
    folder = ensure(os.path.join(OUT, "learning_curves"))
    by_mod = defaultdict(list)

    for name, model_dir, seq_dir, model_type, ckpt in iter_available_models():
        hist = load_history(model_dir)
        if not hist or "val_f1" not in hist[0]:
            continue
        epochs = [h["epoch"] for h in hist]
        val_f1 = [h["val_f1"] for h in hist]
        train_f1 = [h.get("train_f1", np.nan) for h in hist]

        plt.figure(figsize=(7.0, 4.6))
        plt.plot(epochs, train_f1, lw=2, label="Train F1")
        plt.plot(epochs, val_f1, lw=2, label="Validation F1")
        plt.xlabel("Epoch")
        plt.ylabel("F1-score")
        plt.title(f"Learning Curve (F1) — {name}")
        plt.legend()
        plt.grid(True, alpha=0.3)
        savefig(os.path.join(folder, f"f1_{name}.png"))

        modality, pipe = parse_name(name)
        by_mod[modality].append((PIPE_SHORT.get(pipe, pipe), epochs, val_f1))

    for modality, curves in by_mod.items():
        plt.figure(figsize=(8, 5))
        for label, epochs, val_f1 in curves:
            plt.plot(epochs, val_f1, lw=1.8, marker="o", markersize=3, label=label)
        plt.xlabel("Epoch")
        plt.ylabel("Validation F1")
        plt.title(f"Validation F1 Learning Curves — {modality}")
        plt.legend(fontsize=8)
        plt.grid(True, alpha=0.3)
        savefig(os.path.join(folder, f"val_f1_combined_{modality}.png"))


# ------------------------------------------------------------
# 5) PCA / LDA variance plots
# ------------------------------------------------------------
def plot_dr_variance():
    folder = ensure(os.path.join(OUT, "dimensionality_reduction"))
    pca_cfgs = [
        ("host", os.path.join("processed_EVSE_B_Host", "sequences", "pca", "pca_config.json")),
        ("network", os.path.join("processed_EVSE_A", "sequences", "pca", "pca_config.json")),
        ("power", os.path.join("processed_EVSE_B_Power", "sequences", "pca", "pca_config.json")),
    ]
    lda_cfgs = [
        ("host", os.path.join("processed_EVSE_B_Host", "sequences", "lda_multiclass", "lda_config.json")),
        ("network", os.path.join("processed_EVSE_A", "sequences", "lda_multiclass", "lda_config.json")),
        ("power", os.path.join("processed_EVSE_B_Power", "sequences", "lda_multiclass", "lda_config.json")),
    ]

    for modality, path in pca_cfgs:
        if not os.path.isfile(path):
            continue
        with open(path, encoding="utf-8") as f:
            cfg = json.load(f)
        # reload ratios from joblib if needed
        ratios = cfg.get("explained_variance_ratio")
        if ratios is None:
            try:
                import joblib
                pca = joblib.load(
                    os.path.join(os.path.dirname(path), "pca.joblib")
                )
                ratios = pca.explained_variance_ratio_.tolist()
            except Exception:
                continue
        ratios = np.asarray(ratios, dtype=float)
        cum = np.cumsum(ratios)

        fig, ax1 = plt.subplots(figsize=(7.2, 4.6))
        ax1.bar(
            np.arange(1, len(ratios) + 1),
            ratios,
            color="#4C78A8",
            label="Individual variance",
        )
        ax1.set_xlabel("Principal component")
        ax1.set_ylabel("Explained variance ratio")
        ax2 = ax1.twinx()
        ax2.plot(
            np.arange(1, len(ratios) + 1),
            cum,
            color="#F58518",
            marker="o",
            label="Cumulative",
        )
        ax2.axhline(0.95, color="gray", ls="--", lw=1, label="95% threshold")
        ax2.set_ylabel("Cumulative variance")
        ax1.set_title(f"PCA Scree / Cumulative Variance — {modality}")
        h1, l1 = ax1.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax1.legend(h1 + h2, l1 + l2, loc="center right", fontsize=8)
        savefig(os.path.join(folder, f"pca_variance_{modality}.png"))

    for modality, path in lda_cfgs:
        if not os.path.isfile(path):
            continue
        with open(path, encoding="utf-8") as f:
            cfg = json.load(f)
        ratios = cfg.get("explained_variance_ratio")
        if not ratios:
            try:
                import joblib
                lda = joblib.load(
                    os.path.join(os.path.dirname(path), "lda.joblib")
                )
                if hasattr(lda, "explained_variance_ratio_"):
                    ratios = lda.explained_variance_ratio_.tolist()
            except Exception:
                ratios = None
        if not ratios:
            continue
        ratios = np.asarray(ratios, dtype=float)
        plt.figure(figsize=(6.4, 4.4))
        plt.bar(
            [f"LD{i+1}" for i in range(len(ratios))],
            ratios,
            color="#54A24B",
        )
        plt.ylabel("Between-class variance ratio")
        plt.title(f"LDA Discriminant Variance — {modality}")
        for i, v in enumerate(ratios):
            plt.text(i, v + 0.01, f"{v:.3f}", ha="center", fontsize=8)
        plt.ylim(0, min(1.05, max(ratios) * 1.25 + 0.05))
        plt.grid(True, axis="y", alpha=0.3)
        savefig(os.path.join(folder, f"lda_variance_{modality}.png"))


# ------------------------------------------------------------
# 6) Class / Attack_Group distributions
# ------------------------------------------------------------
def plot_distributions():
    folder = ensure(os.path.join(OUT, "distributions"))
    import pandas as pd

    specs = [
        (
            "host",
            os.path.join(
                "processed_EVSE_B_Host",
                "EVSE_B_Host_Events_Clean.csv",
            ),
            "Attack_Group",
            "Label",
        ),
        (
            "network",
            os.path.join(
                "processed_EVSE_A",
                "EVSE_A_Network_Clean.csv",
            ),
            "Attack",
            "Label",
            ),
        (
            "power",
            os.path.join(
                "processed_EVSE_B_Power",
                "EVSE_B_Power_Clean.csv",
            ),
            "Attack_Group",
            "Label",
        ),
    ]

    for item in specs:
        modality, path, group_col, label_col = item[0], item[1], item[2], item[3]
        if not os.path.isfile(path):
            continue
        # network full file is huge; use usecols
        df = pd.read_csv(path, usecols=[group_col, label_col])

        # binary distribution
        vc = df[label_col].astype(str).str.lower().value_counts()
        plt.figure(figsize=(5.5, 4.4))
        plt.bar(vc.index.astype(str), vc.values, color=["#4C78A8", "#E45756"][: len(vc)])
        plt.title(f"Binary Label Distribution — {modality}")
        plt.ylabel("Count")
        for i, v in enumerate(vc.values):
            plt.text(i, v, str(v), ha="center", va="bottom", fontsize=8)
        plt.grid(True, axis="y", alpha=0.3)
        savefig(os.path.join(folder, f"binary_dist_{modality}.png"))

        # family / attack distribution
        gvc = df[group_col].astype(str).value_counts()
        plt.figure(figsize=(8.5, 4.8))
        plt.bar(gvc.index.astype(str), gvc.values, color="#72B7B2")
        plt.title(f"{group_col} Distribution — {modality}")
        plt.ylabel("Count")
        plt.xticks(rotation=30, ha="right")
        plt.grid(True, axis="y", alpha=0.3)
        savefig(os.path.join(folder, f"group_dist_{modality}.png"))


# ------------------------------------------------------------
# 7) Multi-class performance bars
# ------------------------------------------------------------
def plot_multiclass_bars():
    folder = ensure(os.path.join(OUT, "multiclass"))
    files = [
        ("host", os.path.join("thesis_figures", "multiclass_host.json")),
        ("network", os.path.join("thesis_figures", "multiclass_network.json")),
        ("power", os.path.join("thesis_figures", "multiclass_power.json")),
    ]
    for modality, path in files:
        if not os.path.isfile(path):
            continue
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        report = data["classification_report"]
        classes = data["metrics"]["classes"]
        precs, recs, f1s = [], [], []
        for c in classes:
            precs.append(report[c]["precision"])
            recs.append(report[c]["recall"])
            f1s.append(report[c]["f1-score"])

        x = np.arange(len(classes))
        width = 0.25
        plt.figure(figsize=(9, 4.8))
        plt.bar(x - width, precs, width, label="Precision")
        plt.bar(x, recs, width, label="Recall")
        plt.bar(x + width, f1s, width, label="F1")
        plt.xticks(x, classes, rotation=15, ha="right")
        plt.ylim(0, 1.05)
        plt.ylabel("Score")
        plt.title(f"Multi-class Per-Category Metrics — {modality}")
        plt.legend()
        plt.grid(True, axis="y", alpha=0.3)
        savefig(os.path.join(folder, f"per_class_metrics_{modality}.png"))

        # ROC AUC bars if available
        aucs = data.get("roc_aucs", {})
        class_aucs = [(c, aucs[c]) for c in classes if c in aucs]
        if class_aucs:
            plt.figure(figsize=(8, 4.4))
            plt.bar(
                [c for c, _ in class_aucs],
                [v for _, v in class_aucs],
                color="#B279A2",
            )
            if "macro_ovr" in aucs:
                plt.axhline(
                    aucs["macro_ovr"],
                    color="gray",
                    ls="--",
                    label=f"macro-OVR={aucs['macro_ovr']:.3f}",
                )
                plt.legend()
            plt.ylim(0, 1.05)
            plt.ylabel("OvR ROC-AUC")
            plt.title(f"Multi-class Per-Class ROC-AUC — {modality}")
            plt.grid(True, axis="y", alpha=0.3)
            savefig(os.path.join(folder, f"per_class_auc_{modality}.png"))


# ------------------------------------------------------------
# 8) Metrics heatmap
# ------------------------------------------------------------
def plot_heatmap(rows):
    folder = ensure(os.path.join(OUT, "heatmap"))
    # rows as modality|pipeline
    labels = [f"{r['modality'][:3]}-{r['pipe'].split('-')[0]}" for r in rows]
    metrics = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    mat = np.array([[r[m] for m in metrics] for r in rows], dtype=float)

    plt.figure(figsize=(8.5, max(5.5, 0.35 * len(labels) + 2)))
    im = plt.imshow(mat, aspect="auto", cmap="YlGnBu", vmin=0.95, vmax=1.0)
    plt.colorbar(im, fraction=0.03, pad=0.04, label="Score")
    plt.xticks(np.arange(len(metrics)), [m.upper() for m in metrics])
    plt.yticks(np.arange(len(labels)), labels)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            plt.text(
                j,
                i,
                f"{mat[i, j]:.3f}",
                ha="center",
                va="center",
                fontsize=7,
                color="black",
            )
    plt.title("Test Metrics Heatmap — All Pipelines")
    savefig(os.path.join(folder, "metrics_heatmap_all.png"))


def main():
    print("=" * 70)
    print("PUBLICATION EXTRA PLOTS")
    print("=" * 70)

    print("\n[1/8] Precision-Recall curves...")
    plot_precision_recall()

    print("\n[2/8] Collecting metrics...")
    rows = collect_metrics()

    print("\n[3/8] Metric comparison bars...")
    plot_metric_bars(rows)

    print("\n[4/8] Efficiency trade-off plots...")
    plot_efficiency(rows)

    print("\n[5/8] Validation F1 learning curves...")
    plot_val_f1_curves()

    print("\n[6/8] PCA/LDA variance plots...")
    plot_dr_variance()

    print("\n[7/8] Class distribution plots...")
    plot_distributions()

    print("\n[8/8] Multi-class bars + heatmap...")
    plot_multiclass_bars()
    plot_heatmap(rows)

    # index
    index = {
        "output_root": OUT,
        "categories": [
            "precision_recall",
            "metric_bars",
            "efficiency",
            "learning_curves",
            "dimensionality_reduction",
            "distributions",
            "multiclass",
            "heatmap",
        ],
        "n_models": len(rows),
    }
    with open(os.path.join(OUT, "index.json"), "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2)

    print("\nAll extra publication figures written to:", OUT)


if __name__ == "__main__":
    main()
