# ============================================================
# CICEVSE2024 - Pipeline C: LDA-LSTM
# Master's Thesis
#
# Dimensionality reduction:
#   Multi-class LDA on Attack_Group
#   (Benign / Recon / DoS / Cryptojacking / Backdoor)
#   fitted on training data only.
#
# Classifier:
#   LSTM binary detection (Benign vs Attack)
#   — LDA projection is supervised multi-class; LSTM still
#     solves the primary binary IDS task.
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
    "pipeline_C_lda_lstm",
)


if __name__ == "__main__":
    train_binary_lstm(
        pipeline_name=(
            "PIPELINE C — LDA-LSTM "
            "(multi-class LDA features)"
        ),
        seq_dir=SEQ_DIR,
        model_dir=MODEL_DIR,
    )
    print("\nPipeline C complete.")
