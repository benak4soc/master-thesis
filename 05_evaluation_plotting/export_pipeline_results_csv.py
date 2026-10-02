# Export all pipeline test results to a single CSV
import os
import json
import csv

import numpy as np
import torch


def infer_modality_and_folder(model_dir):
    rel = os.path.relpath(model_dir, "models")
    segs = rel.split(os.sep)
    if segs[0] == "network_EVSE_A":
        return "network", segs[1] if len(segs) > 1 else segs[0]
    if segs[0] == "power_EVSE_B":
        return "power", segs[1] if len(segs) > 1 else segs[0]
    return "host", segs[0]


def pipeline_letter(pipe_folder):
    name = pipe_folder.lower()
    for letter in ["a", "b", "c", "d", "e"]:
        if f"pipeline_{letter}_" in name or name.startswith(f"pipeline_{letter}"):
            return letter.upper()
    return ""


def dim_reduction(pipe_folder, description):
    text = f"{pipe_folder} {description}".lower()
    if "pca" in text:
        return "PCA"
    if "lda" in text:
        return "LDA"
    return "None"


def model_type_from_folder(pipe_folder, explicit=None):
    if explicit:
        return explicit
    name = pipe_folder.lower()
    if "cnn" in name:
        return "cnn_bilstm_attention"
    if "bilstm" in name:
        return "bilstm"
    return "lstm"


def count_params_from_ckpt(model_dir):
    ckpt_path = os.path.join(model_dir, "best_lstm.pt")
    if not os.path.isfile(ckpt_path):
        return None
    try:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if ckpt.get("n_params") is not None:
            return int(ckpt["n_params"])
        if "model_state" in ckpt:
            return int(
                sum(
                    int(np.prod(v.shape))
                    for v in ckpt["model_state"].values()
                )
            )
    except Exception:
        return None
    return None


def main():
    result_files = []
    for root, _, files in os.walk("models"):
        if "test_results.json" in files:
            result_files.append(os.path.join(root, "test_results.json"))
    result_files = sorted(result_files)

    rows = []
    for path in result_files:
        with open(path, encoding="utf-8") as handle:
            data = json.load(handle)

        model_dir = os.path.dirname(path)
        modality, pipe_folder = infer_modality_and_folder(model_dir)
        description = data.get("pipeline", pipe_folder)
        metrics = data.get("test_metrics", {})
        hp = data.get("hyperparameters", {})
        shape = data.get("window_features_shape", {})
        cm = data.get("confusion_matrix", [[None, None], [None, None]])

        n_params = data.get("n_params")
        if n_params is None:
            n_params = count_params_from_ckpt(model_dir)

        rows.append(
            {
                "modality": modality,
                "pipeline_folder": pipe_folder,
                "pipeline_letter": pipeline_letter(pipe_folder),
                "pipeline_description": description,
                "model_type": model_type_from_folder(
                    pipe_folder, data.get("model_type")
                ),
                "dim_reduction": dim_reduction(pipe_folder, description),
                "n_timesteps": shape.get("timesteps"),
                "n_features": shape.get("features"),
                "n_params": n_params,
                "accuracy": metrics.get("accuracy"),
                "precision": metrics.get("precision"),
                "recall": metrics.get("recall"),
                "f1": metrics.get("f1"),
                "roc_auc": metrics.get("roc_auc"),
                "test_loss": metrics.get("loss"),
                "train_seconds": data.get("training_time_seconds"),
                "best_val_f1": data.get("best_val_f1"),
                "best_epoch": hp.get("best_epoch"),
                "epochs_run": hp.get("epochs_run"),
                "epochs_max": hp.get("epochs_max"),
                "hidden_size": hp.get("hidden_size"),
                "num_layers": hp.get("num_layers"),
                "dropout": hp.get("dropout"),
                "batch_size": hp.get("batch_size"),
                "learning_rate": hp.get("learning_rate"),
                "patience": hp.get("patience"),
                "cm_TN": cm[0][0],
                "cm_FP": cm[0][1],
                "cm_FN": cm[1][0],
                "cm_TP": cm[1][1],
                "device": data.get("device"),
                "seq_dir": data.get("seq_dir"),
                "results_path": path,
            }
        )

    out_csv = os.path.join("models", "all_pipeline_results.csv")
    fieldnames = list(rows[0].keys())
    with open(out_csv, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    # also save a compact copy at project root
    root_csv = "all_pipeline_results.csv"
    with open(root_csv, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved {len(rows)} rows")
    print(f"  {out_csv}")
    print(f"  {root_csv}")
    for row in rows:
        print(
            f"{row['modality']:8s} {row['pipeline_folder']:40s} "
            f"F1={row['f1']:.4f} Acc={row['accuracy']:.4f} "
            f"feats={row['n_features']} params={row['n_params']}"
        )


if __name__ == "__main__":
    main()
