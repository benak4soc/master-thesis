# ============================================================
# Shared registry: trained models -> sequence folders
# Used by the three plotting scripts.
# ============================================================

import os

# (display_name, model_dir, seq_dir, model_type_hint)
# model_type is also read from checkpoint when available.
MODEL_REGISTRY = [
    # Host HPC / Kernel
    (
        "host_A_lstm",
        os.path.join("models", "pipeline_A_lstm"),
        os.path.join("processed_EVSE_B_Host", "sequences"),
        "lstm",
    ),
    (
        "host_B_pca_lstm",
        os.path.join("models", "pipeline_B_pca_lstm"),
        os.path.join("processed_EVSE_B_Host", "sequences", "pca"),
        "lstm",
    ),
    (
        "host_C_lda_lstm",
        os.path.join("models", "pipeline_C_lda_lstm"),
        os.path.join(
            "processed_EVSE_B_Host", "sequences", "lda_multiclass"
        ),
        "lstm",
    ),
    (
        "host_D_lda_bilstm",
        os.path.join("models", "pipeline_D_lda_bilstm"),
        os.path.join(
            "processed_EVSE_B_Host", "sequences", "lda_multiclass"
        ),
        "bilstm",
    ),
    (
        "host_E_cnn_bilstm_attention",
        os.path.join("models", "pipeline_E_cnn_bilstm_attention"),
        os.path.join("processed_EVSE_B_Host", "sequences"),
        "cnn_bilstm_attention",
    ),
    # Network EVSE-A
    (
        "network_A_lstm",
        os.path.join("models", "network_EVSE_A", "pipeline_A_lstm"),
        os.path.join("processed_EVSE_A", "sequences"),
        "lstm",
    ),
    (
        "network_B_pca_lstm",
        os.path.join(
            "models", "network_EVSE_A", "pipeline_B_pca_lstm"
        ),
        os.path.join("processed_EVSE_A", "sequences", "pca"),
        "lstm",
    ),
    (
        "network_C_lda_lstm",
        os.path.join(
            "models", "network_EVSE_A", "pipeline_C_lda_lstm"
        ),
        os.path.join(
            "processed_EVSE_A", "sequences", "lda_multiclass"
        ),
        "lstm",
    ),
    (
        "network_D_lda_bilstm",
        os.path.join(
            "models", "network_EVSE_A", "pipeline_D_lda_bilstm"
        ),
        os.path.join(
            "processed_EVSE_A", "sequences", "lda_multiclass"
        ),
        "bilstm",
    ),
    (
        "network_E_cnn_bilstm_attention",
        os.path.join(
            "models",
            "network_EVSE_A",
            "pipeline_E_cnn_bilstm_attention",
        ),
        os.path.join("processed_EVSE_A", "sequences"),
        "cnn_bilstm_attention",
    ),
    # Power EVSE-B
    (
        "power_A_lstm",
        os.path.join("models", "power_EVSE_B", "pipeline_A_lstm"),
        os.path.join("processed_EVSE_B_Power", "sequences"),
        "lstm",
    ),
    (
        "power_B_pca_lstm",
        os.path.join(
            "models", "power_EVSE_B", "pipeline_B_pca_lstm"
        ),
        os.path.join("processed_EVSE_B_Power", "sequences", "pca"),
        "lstm",
    ),
    (
        "power_C_lda_lstm",
        os.path.join(
            "models", "power_EVSE_B", "pipeline_C_lda_lstm"
        ),
        os.path.join(
            "processed_EVSE_B_Power", "sequences", "lda_multiclass"
        ),
        "lstm",
    ),
    (
        "power_D_lda_bilstm",
        os.path.join(
            "models", "power_EVSE_B", "pipeline_D_lda_bilstm"
        ),
        os.path.join(
            "processed_EVSE_B_Power", "sequences", "lda_multiclass"
        ),
        "bilstm",
    ),
    (
        "power_E_cnn_bilstm_attention",
        os.path.join(
            "models",
            "power_EVSE_B",
            "pipeline_E_cnn_bilstm_attention",
        ),
        os.path.join("processed_EVSE_B_Power", "sequences"),
        "cnn_bilstm_attention",
    ),
]


def iter_available_models():
    """Yield registry entries that have a checkpoint on disk."""
    for name, model_dir, seq_dir, model_type in MODEL_REGISTRY:
        ckpt = os.path.join(model_dir, "best_lstm.pt")
        if os.path.isfile(ckpt):
            yield name, model_dir, seq_dir, model_type, ckpt
