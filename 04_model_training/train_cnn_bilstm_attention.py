# ============================================================
# CICEVSE2024 - Pipeline E: CNN-BiLSTM + Attention
# Master's Thesis
#
# Feature representation:
#   Full scaled host features (no PCA/LDA) — thesis table:
#   "None / selected feature representation"
#
# Model:
#   1D CNN -> BiLSTM -> temporal attention -> classifier
# ============================================================

import os

from lstm_common import train_binary_lstm


SEQ_DIR = os.path.join(
    "processed_EVSE_B_Host",
    "sequences",
)

MODEL_DIR = os.path.join(
    "models",
    "pipeline_E_cnn_bilstm_attention",
)


if __name__ == "__main__":
    train_binary_lstm(
        pipeline_name=(
            "PIPELINE E — CNN-BiLSTM + Attention "
            "(full features)"
        ),
        seq_dir=SEQ_DIR,
        model_dir=MODEL_DIR,
        model_type="cnn_bilstm_attention",
        model_kwargs={
            "cnn_channels": 64,
            "cnn_kernel": 3,
        },
    )
    print("\nPipeline E complete.")
