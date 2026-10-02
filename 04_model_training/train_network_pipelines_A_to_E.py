# ============================================================
# Train Pipelines A–E on EVSE-A network traffic sequences
# ============================================================

import os
import json

from lstm_common import train_binary_lstm


BASE = os.path.join("processed_EVSE_A", "sequences")
MODEL_ROOT = os.path.join("models", "network_EVSE_A")


PIPELINES = [
    {
        "name": "PIPELINE A — LSTM (network, no DR)",
        "model_type": "lstm",
        "seq_dir": BASE,
        "model_dir": os.path.join(MODEL_ROOT, "pipeline_A_lstm"),
    },
    {
        "name": "PIPELINE B — PCA-LSTM (network)",
        "model_type": "lstm",
        "seq_dir": os.path.join(BASE, "pca"),
        "model_dir": os.path.join(MODEL_ROOT, "pipeline_B_pca_lstm"),
    },
    {
        "name": "PIPELINE C — LDA-LSTM (network)",
        "model_type": "lstm",
        "seq_dir": os.path.join(BASE, "lda_multiclass"),
        "model_dir": os.path.join(MODEL_ROOT, "pipeline_C_lda_lstm"),
    },
    {
        "name": "PIPELINE D — LDA-BiLSTM (network)",
        "model_type": "bilstm",
        "seq_dir": os.path.join(BASE, "lda_multiclass"),
        "model_dir": os.path.join(MODEL_ROOT, "pipeline_D_lda_bilstm"),
    },
    {
        "name": "PIPELINE E — CNN-BiLSTM+Attention (network)",
        "model_type": "cnn_bilstm_attention",
        "seq_dir": BASE,
        "model_dir": os.path.join(
            MODEL_ROOT,
            "pipeline_E_cnn_bilstm_attention",
        ),
        "model_kwargs": {"cnn_channels": 64, "cnn_kernel": 3},
    },
]


def main():
    os.makedirs(MODEL_ROOT, exist_ok=True)
    comparison = []

    for pipe in PIPELINES:
        result = train_binary_lstm(
            pipeline_name=pipe["name"],
            seq_dir=pipe["seq_dir"],
            model_dir=pipe["model_dir"],
            model_type=pipe["model_type"],
            model_kwargs=pipe.get("model_kwargs"),
        )
        comparison.append(
            {
                "pipeline": pipe["name"],
                "model_type": pipe["model_type"],
                "features": result["window_features_shape"]["features"],
                "params": result.get("n_params"),
                "accuracy": result["test_metrics"]["accuracy"],
                "precision": result["test_metrics"]["precision"],
                "recall": result["test_metrics"]["recall"],
                "f1": result["test_metrics"]["f1"],
                "roc_auc": result["test_metrics"]["roc_auc"],
                "train_seconds": round(
                    result["training_time_seconds"], 2
                ),
            }
        )
        print("\n")

    out_path = os.path.join(
        MODEL_ROOT,
        "comparison_A_to_E_network.json",
    )
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "modality": "network_EVSE_A",
                "comparison": comparison,
                "note": (
                    "Trained on undersampled network windows "
                    "(all benign kept). Compare with host results separately."
                ),
            },
            handle,
            indent=2,
        )

    print("=" * 70)
    print("NETWORK A–E COMPARISON")
    print("=" * 70)
    for row in comparison:
        print(
            f"{row['pipeline'][:40]:40s} "
            f"F={row['features']:3d}  "
            f"F1={row['f1']:.4f}  "
            f"AUC={row['roc_auc']:.4f}  "
            f"t={row['train_seconds']:.1f}s  "
            f"P={row['params']}"
        )
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
