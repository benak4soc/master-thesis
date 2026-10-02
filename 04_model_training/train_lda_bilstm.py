# ============================================================
# CICEVSE2024 - Pipeline D: LDA-BiLSTM
# Master's Thesis
#
# Dimensionality reduction:
#   Multi-class LDA on Attack_Group (same features as Pipeline C)
#
# Model:
#   Bidirectional LSTM for binary Benign vs Attack detection
# ============================================================

import os

from lstm_common import train_binary_lstm


SEQ_DIR = os.path.join(
    "processed_EVSE_B_Host",
    "sequences",
    "lda_multiclass",
)

MODEL_DIR = os.path.join(
    "models",
    "pipeline_D_lda_bilstm",
)


if __name__ == "__main__":
    train_binary_lstm(
        pipeline_name=(
            "PIPELINE D — LDA-BiLSTM "
            "(multi-class LDA features)"
        ),
        seq_dir=SEQ_DIR,
        model_dir=MODEL_DIR,
        model_type="bilstm",
    )
    print("\nPipeline D complete.")
