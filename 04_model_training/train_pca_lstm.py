# ============================================================
# CICEVSE2024 - Pipeline B: PCA-LSTM
# Master's Thesis
#
# Dimensionality reduction: PCA (train-only, 95% variance)
# Classifier: LSTM binary (Benign vs Attack)
# ============================================================

import os

from lstm_common import train_binary_lstm


SEQ_DIR = os.path.join(
    "processed_EVSE_B_Host",
    "sequences",
    "pca",
)

MODEL_DIR = os.path.join(
    "models",
    "pipeline_B_pca_lstm",
)


if __name__ == "__main__":
    train_binary_lstm(
        pipeline_name="PIPELINE B — PCA-LSTM",
        seq_dir=SEQ_DIR,
        model_dir=MODEL_DIR,
    )
    print("\nPipeline B complete.")
