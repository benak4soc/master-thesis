# ============================================================
# Export tuned Pipeline C (LDA-LSTM) to ONNX
# + package preprocessing artifacts for Jetson Nano
# ============================================================

import os
import json
import shutil
import warnings

import numpy as np
import torch
import torch.nn as nn
import joblib

from lstm_common import BaselineLSTM

warnings.filterwarnings("ignore")


CKPT = os.path.join(
    "models",
    "pipeline_C_lda_lstm_tuned",
    "best_lstm.pt",
)

# Fallback to untuned C if tuning not run yet
FALLBACK_CKPT = os.path.join(
    "models",
    "pipeline_C_lda_lstm",
    "best_lstm.pt",
)

LDA_PATH = os.path.join(
    "processed_EVSE_B_Host",
    "sequences",
    "lda_multiclass",
    "lda.joblib",
)

SCALER_PATH = os.path.join(
    "processed_EVSE_B_Host",
    "sequences",
    "scaler.joblib",
)

FEATURE_PATH = os.path.join(
    "processed_EVSE_B_Host",
    "sequences",
    "feature_columns.txt",
)

DEPLOY_DIR = os.path.join(
    "jetson_deploy",
    "pipeline_C_lda_lstm",
)

os.makedirs(DEPLOY_DIR, exist_ok=True)


print("=" * 70)
print("ONNX EXPORT — Pipeline C (LDA-LSTM)")
print("=" * 70)

ckpt_path = CKPT if os.path.isfile(CKPT) else FALLBACK_CKPT

if not os.path.isfile(ckpt_path):
    raise FileNotFoundError(
        "No Pipeline C checkpoint found. "
        "Run train_lda_lstm.py or tune_lda_lstm.py first."
    )

print(f"Checkpoint: {ckpt_path}")

ckpt = torch.load(
    ckpt_path,
    map_location="cpu",
    weights_only=False,
)

n_features = int(ckpt["n_features"])
n_timesteps = int(ckpt["n_timesteps"])
hidden_size = int(ckpt["hidden_size"])
num_layers = int(ckpt["num_layers"])
dropout = float(ckpt.get("dropout", 0.0))

model = BaselineLSTM(
    n_features=n_features,
    hidden_size=hidden_size,
    num_layers=num_layers,
    dropout=0.0,  # disable dropout for export/inference
)

# Rebuild with matching dropout modules then load
model = BaselineLSTM(
    n_features=n_features,
    hidden_size=hidden_size,
    num_layers=num_layers,
    dropout=dropout,
)
model.load_state_dict(ckpt["model_state"])
model.eval()

# For export, force eval mode (dropout off)
dummy = torch.randn(1, n_timesteps, n_features)

onnx_path = os.path.join(
    DEPLOY_DIR,
    "lda_lstm.onnx",
)

# Prefer legacy exporter (more stable on Python 3.14 / Windows)
try:
    torch.onnx.export(
        model,
        dummy,
        onnx_path,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={
            "input": {0: "batch"},
            "logits": {0: "batch"},
        },
        opset_version=17,
        dynamo=False,
    )
except TypeError:
    # Older torch without dynamo kwarg
    torch.onnx.export(
        model,
        dummy,
        onnx_path,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={
            "input": {0: "batch"},
            "logits": {0: "batch"},
        },
        opset_version=17,
    )

print(f"ONNX saved: {onnx_path}")

# Quick ORT validation
try:
    import onnxruntime as ort

    sess = ort.InferenceSession(
        onnx_path,
        providers=["CPUExecutionProvider"],
    )
    out = sess.run(
        None,
        {"input": dummy.numpy()},
    )[0]
    print(
        f"ORT check OK: logits shape={out.shape}"
    )
except Exception as exc:
    print(f"ORT check warning: {exc}")

# Copy preprocessing artifacts
for src, name in [
    (SCALER_PATH, "scaler.joblib"),
    (LDA_PATH, "lda.joblib"),
    (FEATURE_PATH, "feature_columns.txt"),
]:
    if not os.path.isfile(src):
        raise FileNotFoundError(f"Missing artifact: {src}")
    dst = os.path.join(DEPLOY_DIR, name)
    shutil.copy2(src, dst)
    print(f"Copied: {name}")

# Save checkpoint copy too
shutil.copy2(
    ckpt_path,
    os.path.join(DEPLOY_DIR, "best_lstm.pt"),
)

meta = {
    "pipeline": "C_LDA_LSTM",
    "n_features_raw": None,
    "n_features_lda": n_features,
    "n_timesteps": n_timesteps,
    "hidden_size": hidden_size,
    "num_layers": num_layers,
    "dropout": dropout,
    "onnx_input": "[batch, timesteps, lda_features]",
    "onnx_output": "logits [batch, 2]  (0=Benign, 1=Attack)",
    "label_map": {"0": "Benign", "1": "Attack"},
    "checkpoint_metrics": {
        "val": ckpt.get("val_metrics"),
        "test": ckpt.get("test_metrics"),
        "n_params": ckpt.get("n_params"),
        "latency_ms_per_window": ckpt.get(
            "latency_ms_per_window"
        ),
        "config": ckpt.get("config"),
    },
}

# Fill raw feature count
with open(FEATURE_PATH, encoding="utf-8") as handle:
    feats = [ln.strip() for ln in handle if ln.strip()]
meta["n_features_raw"] = len(feats)
meta["feature_columns_file"] = "feature_columns.txt"

with open(
    os.path.join(DEPLOY_DIR, "model_meta.json"),
    "w",
    encoding="utf-8",
) as handle:
    json.dump(meta, handle, indent=2)

print(f"\nDeploy package: {DEPLOY_DIR}")
print("Export complete.")
