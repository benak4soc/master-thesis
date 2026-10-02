# ============================================================
# Generate thesis figures + multi-class supplementary evaluation
# ============================================================

import os
import json
import warnings

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    auc,
    roc_auc_score,
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.preprocessing import label_binarize

from lstm_common import BaselineLSTM, set_seed

warnings.filterwarnings("ignore")

FIG_DIR = "thesis_figures"
os.makedirs(FIG_DIR, exist_ok=True)

CLASS_NAMES_5 = [
    "Benign",
    "Recon",
    "DoS",
    "Cryptojacking",
    "Backdoor",
]


def load_history(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def plot_loss(history, title, out_path):
    epochs = [h["epoch"] for h in history]
    tr = [h["train_loss"] for h in history]
    va = [h["val_loss"] for h in history]
    plt.figure(figsize=(7, 4))
    plt.plot(epochs, tr, label="Train loss", linewidth=2)
    plt.plot(epochs, va, label="Validation loss", linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_cm(cm, labels, title, out_path):
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=labels,
    )
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_binary_roc(y_true, y_prob, title, out_path):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, lw=2, label=f"AUC = {roc_auc:.4f}")
    plt.plot([0, 1], [0, 1], "k--", lw=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    return roc_auc


def plot_multiclass_roc(y_true, y_prob, class_names, title, out_path):
    # y_prob: (N, C)
    present = sorted(np.unique(y_true).tolist())
    names = [class_names[i] for i in present if i < len(class_names)]
    y_bin = label_binarize(y_true, classes=present)
    if y_bin.ndim == 1:
        y_bin = y_bin.reshape(-1, 1)

    plt.figure(figsize=(7, 5.5))
    aucs = {}
    for i, cls in enumerate(present):
        if y_bin.shape[1] == 1 and i == 0:
            # binary edge case
            fpr, tpr, _ = roc_curve(y_bin[:, 0], y_prob[:, cls])
        else:
            fpr, tpr, _ = roc_curve(y_bin[:, i], y_prob[:, cls])
        a = auc(fpr, tpr)
        label = class_names[cls] if cls < len(class_names) else str(cls)
        aucs[label] = float(a)
        plt.plot(fpr, tpr, lw=2, label=f"{label} (AUC={a:.3f})")

    # macro average approx
    try:
        macro = roc_auc_score(
            y_true,
            y_prob[:, present],
            multi_class="ovr",
            average="macro",
            labels=present,
        )
        aucs["macro_ovr"] = float(macro)
    except Exception:
        macro = float("nan")

    plt.plot([0, 1], [0, 1], "k--", lw=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title + (f"\nmacro-OVR AUC={macro:.4f}" if macro == macro else ""))
    plt.legend(fontsize=8, loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    return aucs


def load_npz_split(seq_dir, name):
    data = np.load(os.path.join(seq_dir, f"{name}_windows.npz"))
    return (
        data["X"].astype(np.float32),
        data["y_binary"].astype(np.int64),
        data["y_multiclass"].astype(np.int64),
    )


def predict_binary_model(ckpt_path, seq_dir, model_type="lstm"):
    from lstm_common import build_model

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    X_test, y_test, _ = load_npz_split(seq_dir, "test")
    n_features = X_test.shape[2]
    kwargs = {
        "hidden_size": ckpt.get("hidden_size", 64),
        "num_layers": ckpt.get("num_layers", 1),
        "dropout": 0.0,
    }
    # CNN may need cnn kwargs
    mk = ckpt.get("model_kwargs") or {}
    if model_type.startswith("cnn"):
        kwargs.update(mk)

    model = build_model(
        ckpt.get("model_type", model_type),
        n_features=n_features,
        **kwargs,
    )
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    loader = DataLoader(
        TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test)),
        batch_size=256,
        shuffle=False,
    )
    probs, preds, trues = [], [], []
    with torch.no_grad():
        for xb, yb in loader:
            logits = model(xb)
            p = torch.softmax(logits, dim=1)[:, 1].numpy()
            pred = torch.argmax(logits, dim=1).numpy()
            probs.append(p)
            preds.append(pred)
            trues.append(yb.numpy())
    return (
        np.concatenate(trues),
        np.concatenate(preds),
        np.concatenate(probs),
    )


def train_multiclass_lstm(seq_dir, out_dir, modality_name, epochs=25):
    """Supplementary multi-class LSTM on provided windows (uses y_multiclass)."""
    os.makedirs(out_dir, exist_ok=True)
    set_seed(42)
    device = torch.device("cpu")

    Xtr, _, ytr = load_npz_split(seq_dir, "train")
    Xva, _, yva = load_npz_split(seq_dir, "val")
    Xte, _, yte = load_npz_split(seq_dir, "test")

    # Keep only classes present in train
    present = sorted(np.unique(ytr).tolist())
    remap = {c: i for i, c in enumerate(present)}
    inv = {i: c for c, i in remap.items()}

    def remap_y(y):
        return np.asarray([remap[int(v)] for v in y], dtype=np.int64)

    ytr_m, yva_m, yte_m = remap_y(ytr), remap_y(yva), remap_y(yte)
    n_classes = len(present)
    names = [CLASS_NAMES_5[c] for c in present]

    # class weights
    counts = np.bincount(ytr_m, minlength=n_classes).astype(np.float64)
    weights = (counts.sum() / (n_classes * np.maximum(counts, 1))).astype(np.float32)
    class_weights = torch.tensor(weights, device=device)

    model = BaselineLSTM(
        n_features=Xtr.shape[2],
        hidden_size=64,
        num_layers=1,
        dropout=0.2,
        n_classes=n_classes,
    ).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    crit = nn.CrossEntropyLoss(weight=class_weights)

    def loader(X, y, shuffle):
        return DataLoader(
            TensorDataset(torch.from_numpy(X), torch.from_numpy(y)),
            batch_size=64,
            shuffle=shuffle,
        )

    tr_loader = loader(Xtr, ytr_m, True)
    va_loader = loader(Xva, yva_m, False)
    te_loader = loader(Xte, yte_m, False)

    history = []
    best_state = None
    best_f1 = -1
    patience, bad = 6, 0

    for epoch in range(1, epochs + 1):
        model.train()
        total = 0.0
        n = 0
        for xb, yb in tr_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward()
            opt.step()
            total += loss.item()
            n += 1
        train_loss = total / max(n, 1)

        model.eval()
        ys, ps = [], []
        vloss, vn = 0.0, 0
        with torch.no_grad():
            for xb, yb in va_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                loss = crit(logits, yb)
                vloss += loss.item()
                vn += 1
                ys.append(yb.cpu().numpy())
                ps.append(torch.argmax(logits, dim=1).cpu().numpy())
        y_true = np.concatenate(ys)
        y_pred = np.concatenate(ps)
        val_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": vloss / max(vn, 1),
                "val_macro_f1": float(val_f1),
            }
        )
        if val_f1 > best_f1:
            best_f1 = val_f1
            bad = 0
            best_state = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }
        else:
            bad += 1
            if bad >= patience:
                break

    model.load_state_dict(best_state)
    model.eval()
    ys, ps, pr = [], [], []
    with torch.no_grad():
        for xb, yb in te_loader:
            logits = model(xb)
            prob = torch.softmax(logits, dim=1).numpy()
            pred = torch.argmax(logits, dim=1).numpy()
            ys.append(yb.numpy())
            ps.append(pred)
            pr.append(prob)

    y_true = np.concatenate(ys)
    y_pred = np.concatenate(ps)
    y_prob = np.concatenate(pr)

    # map back to original class ids for naming in CM display order
    cm = confusion_matrix(y_true, y_pred, labels=list(range(n_classes)))
    report = classification_report(
        y_true,
        y_pred,
        target_names=names,
        digits=4,
        zero_division=0,
        output_dict=True,
    )
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(
            precision_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "recall_macro": float(
            recall_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "f1_macro": float(
            f1_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "classes": names,
        "present_original_ids": present,
    }

    plot_cm(
        cm,
        names,
        f"Multi-class CM — {modality_name}",
        os.path.join(out_dir, f"cm_multiclass_{modality_name}.png"),
    )
    aucs = plot_multiclass_roc(
        y_true,
        y_prob,
        names,  # remapped 0..C-1 names
        f"Multi-class ROC (OvR) — {modality_name}",
        os.path.join(out_dir, f"roc_multiclass_{modality_name}.png"),
    )
    plot_loss(
        [
            {
                "epoch": h["epoch"],
                "train_loss": h["train_loss"],
                "val_loss": h["val_loss"],
            }
            for h in history
        ],
        f"Multi-class train/val loss — {modality_name}",
        os.path.join(out_dir, f"loss_multiclass_{modality_name}.png"),
    )

    # also save CM as markdown-friendly csv/json
    result = {
        "modality": modality_name,
        "metrics": metrics,
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
        "roc_aucs": aucs,
        "history": history,
    }
    with open(
        os.path.join(out_dir, f"multiclass_{modality_name}.json"),
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(result, f, indent=2)

    return result


# ------------------------------------------------------------
# 1) Binary loss curves for selected pipelines
# ------------------------------------------------------------
print("Generating binary loss / CM / ROC figures...")

BINARY_SPECS = [
    # host
    (
        "host",
        "A",
        "models/pipeline_A_lstm",
        "processed_EVSE_B_Host/sequences",
        "lstm",
    ),
    (
        "host",
        "B",
        "models/pipeline_B_pca_lstm",
        "processed_EVSE_B_Host/sequences/pca",
        "lstm",
    ),
    (
        "host",
        "C",
        "models/pipeline_C_lda_lstm",
        "processed_EVSE_B_Host/sequences/lda_multiclass",
        "lstm",
    ),
    (
        "host",
        "D",
        "models/pipeline_D_lda_bilstm",
        "processed_EVSE_B_Host/sequences/lda_multiclass",
        "bilstm",
    ),
    (
        "host",
        "E",
        "models/pipeline_E_cnn_bilstm_attention",
        "processed_EVSE_B_Host/sequences",
        "cnn_bilstm_attention",
    ),
    # network
    (
        "network",
        "A",
        "models/network_EVSE_A/pipeline_A_lstm",
        "processed_EVSE_A/sequences",
        "lstm",
    ),
    (
        "network",
        "C",
        "models/network_EVSE_A/pipeline_C_lda_lstm",
        "processed_EVSE_A/sequences/lda_multiclass",
        "lstm",
    ),
    # power
    (
        "power",
        "A",
        "models/power_EVSE_B/pipeline_A_lstm",
        "processed_EVSE_B_Power/sequences",
        "lstm",
    ),
    (
        "power",
        "B",
        "models/power_EVSE_B/pipeline_B_pca_lstm",
        "processed_EVSE_B_Power/sequences/pca",
        "lstm",
    ),
    (
        "power",
        "C",
        "models/power_EVSE_B/pipeline_C_lda_lstm",
        "processed_EVSE_B_Power/sequences/lda_multiclass",
        "lstm",
    ),
    (
        "power",
        "E",
        "models/power_EVSE_B/pipeline_E_cnn_bilstm_attention",
        "processed_EVSE_B_Power/sequences",
        "cnn_bilstm_attention",
    ),
]

binary_summary = {}

for modality, pipe, model_dir, seq_dir, model_type in BINARY_SPECS:
    hist_path = os.path.join(model_dir, "training_history.json")
    ckpt_path = os.path.join(model_dir, "best_lstm.pt")
    res_path = os.path.join(model_dir, "test_results.json")
    if not os.path.isfile(hist_path):
        continue

    tag = f"{modality}_{pipe}"
    hist = load_history(hist_path)
    plot_loss(
        hist,
        f"Train/Val Loss — {modality.upper()} Pipeline {pipe}",
        os.path.join(FIG_DIR, f"loss_{tag}.png"),
    )

    # confusion from saved test_results if present
    if os.path.isfile(res_path):
        with open(res_path, encoding="utf-8") as f:
            res = json.load(f)
        cm = np.asarray(res["confusion_matrix"])
        plot_cm(
            cm,
            ["Benign", "Attack"],
            f"Binary CM — {modality.upper()} Pipeline {pipe}",
            os.path.join(FIG_DIR, f"cm_binary_{tag}.png"),
        )
        binary_summary[tag] = res

    # ROC needs probabilities — reload model
    if os.path.isfile(ckpt_path) and os.path.isdir(seq_dir):
        try:
            y_true, y_pred, y_prob = predict_binary_model(
                ckpt_path, seq_dir, model_type
            )
            plot_binary_roc(
                y_true,
                y_prob,
                f"Binary ROC — {modality.upper()} Pipeline {pipe}",
                os.path.join(FIG_DIR, f"roc_binary_{tag}.png"),
            )
            # per-attack-category detection if meta exists
            meta_path = os.path.join(seq_dir, "test_meta.json")
            # for DR folders, meta is in parent
            if not os.path.isfile(meta_path):
                parent = os.path.dirname(seq_dir.rstrip("\\/"))
                # if seq_dir ends with pca/lda_multiclass, parent is sequences
                cand = os.path.join(parent, "test_meta.json")
                if os.path.isfile(cand):
                    meta_path = cand
                else:
                    # host DR: processed_EVSE_B_Host/sequences/pca -> sequences
                    cand2 = os.path.join(
                        os.path.dirname(seq_dir),
                        "..",
                    )
            if os.path.isfile(meta_path):
                with open(meta_path, encoding="utf-8") as f:
                    meta = json.load(f)
                if len(meta) == len(y_true):
                    groups = [m.get("Attack_Group", "NA") for m in meta]
                    per = {}
                    for g in sorted(set(groups)):
                        idx = [i for i, gg in enumerate(groups) if gg == g]
                        yt = y_true[idx]
                        yp = y_pred[idx]
                        # for attack groups, recall = detection rate
                        if g == "Benign":
                            # specificity-ish: correctly predicted benign
                            per[g] = {
                                "n": len(idx),
                                "correct": int((yt == yp).sum()),
                                "accuracy": float((yt == yp).mean()),
                            }
                        else:
                            # all should be attack=1
                            per[g] = {
                                "n": len(idx),
                                "detected": int((yp == 1).sum()),
                                "missed": int((yp == 0).sum()),
                                "detection_rate": float((yp == 1).mean())
                                if len(idx)
                                else 0.0,
                            }
                    with open(
                        os.path.join(
                            FIG_DIR,
                            f"per_attack_detection_{tag}.json",
                        ),
                        "w",
                        encoding="utf-8",
                    ) as f:
                        json.dump(per, f, indent=2)
        except Exception as exc:
            print(f"  skip ROC for {tag}: {exc}")


# ------------------------------------------------------------
# 2) Multi-class supplementary (host + power; network has no Backdoor)
# ------------------------------------------------------------
print("\nTraining supplementary multi-class models...")

mc_host = train_multiclass_lstm(
    os.path.join(
        "processed_EVSE_B_Host",
        "sequences",
        "lda_multiclass",
    ),
    FIG_DIR,
    "host",
    epochs=30,
)
print("Host multi-class:", mc_host["metrics"])

mc_power = train_multiclass_lstm(
    os.path.join(
        "processed_EVSE_B_Power",
        "sequences",
        "lda_multiclass",
    ),
    FIG_DIR,
    "power",
    epochs=30,
)
print("Power multi-class:", mc_power["metrics"])

# Network multi-class (Benign/Recon/DoS only)
mc_net = train_multiclass_lstm(
    os.path.join(
        "processed_EVSE_A",
        "sequences",
        "lda_multiclass",
    ),
    FIG_DIR,
    "network",
    epochs=25,
)
print("Network multi-class:", mc_net["metrics"])

with open(
    os.path.join(FIG_DIR, "figure_index.json"),
    "w",
    encoding="utf-8",
) as f:
    json.dump(
        {
            "binary_pipelines_plotted": list(binary_summary.keys()),
            "multiclass": {
                "host": mc_host["metrics"],
                "power": mc_power["metrics"],
                "network": mc_net["metrics"],
            },
        },
        f,
        indent=2,
    )

print("\nFigures written to", FIG_DIR)
print("Done.")
