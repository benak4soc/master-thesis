"""
Rebuild Jetson/RPi Pipeline C package from the *fixed* host sequences:
  train h16 LDA-LSTM -> ONNX + best_lstm.pt + scaler/lda + preprocess_numpy.npz
  copy into 06_edge_deployment/jetson_deploy and 11_jetson_nano_demo.
Run from master-thesis repo root:
  python 06_edge_deployment/rebuild_edge_package.py
"""
from __future__ import print_function

import json
import os
import shutil
import sys

import joblib
import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "04_model_training"))

from lstm_common import BaselineLSTM, load_binary_split, set_seed, train_binary_lstm  # noqa: E402

HOST = os.path.join(REPO, "09_processed_datasets", "processed_EVSE_B_Host")
if not os.path.isdir(HOST):
    HOST = os.path.join(REPO, "processed_EVSE_B_Host")

SEQ_RAW = os.path.join(HOST, "sequences")
SEQ_LDA = os.path.join(SEQ_RAW, "lda_multiclass")
OUT_CKPT_DIR = os.path.join(REPO, "models", "pipeline_C_lda_lstm_tuned")
DEPLOY_DIRS = [
    os.path.join(REPO, "06_edge_deployment", "jetson_deploy", "pipeline_C_lda_lstm"),
    os.path.join(REPO, "11_jetson_nano_demo", "pipeline_C_lda_lstm"),
    os.path.join(REPO, "13_rpi4_algorithm_dashboard", "host_lda_lstm"),
]

HIDDEN = 16
NUM_LAYERS = 1
DROPOUT = 0.2
LR = 1e-3
BATCH = 64
EPOCHS = 40
PATIENCE = 8
SEED = 42


def main():
    set_seed(SEED)
    os.makedirs(OUT_CKPT_DIR, exist_ok=True)

    Xtr, _ = load_binary_split(SEQ_LDA, "train")
    print("LDA train windows:", Xtr.shape)

    results = train_binary_lstm(
        pipeline_name="C_LDA_LSTM_edge_h16",
        seq_dir=SEQ_LDA,
        model_dir=OUT_CKPT_DIR,
        model_type="lstm",
        hidden_size=HIDDEN,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
        batch_size=BATCH,
        epochs=EPOCHS,
        learning_rate=LR,
        patience=PATIENCE,
        seed=SEED,
    )
    test_metrics = results["test_metrics"]
    print("Test metrics:", test_metrics)

    ckpt_path = os.path.join(OUT_CKPT_DIR, "best_lstm.pt")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    n_features = int(ckpt["n_features"])
    n_timesteps = int(ckpt["n_timesteps"])

    # Fresh model for ONNX (eval)
    export_model = BaselineLSTM(
        n_features=n_features,
        hidden_size=int(ckpt["hidden_size"]),
        num_layers=int(ckpt["num_layers"]),
        dropout=float(ckpt.get("dropout", DROPOUT)),
    )
    export_model.load_state_dict(ckpt["model_state"])
    export_model.eval()
    dummy = torch.randn(1, n_timesteps, n_features)

    scaler = joblib.load(os.path.join(SEQ_RAW, "scaler.joblib"))
    lda = joblib.load(os.path.join(SEQ_LDA, "lda.joblib"))
    feat_src = os.path.join(SEQ_RAW, "feature_columns.txt")

    for deploy in DEPLOY_DIRS:
        os.makedirs(deploy, exist_ok=True)
        onnx_path = os.path.join(deploy, "lda_lstm.onnx")
        try:
            torch.onnx.export(
                export_model,
                dummy,
                onnx_path,
                input_names=["input"],
                output_names=["logits"],
                dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
                opset_version=17,
                dynamo=False,
            )
        except TypeError:
            torch.onnx.export(
                export_model,
                dummy,
                onnx_path,
                input_names=["input"],
                output_names=["logits"],
                dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
                opset_version=17,
            )

        shutil.copy2(ckpt_path, os.path.join(deploy, "best_lstm.pt"))
        shutil.copy2(os.path.join(SEQ_RAW, "scaler.joblib"), os.path.join(deploy, "scaler.joblib"))
        shutil.copy2(os.path.join(SEQ_LDA, "lda.joblib"), os.path.join(deploy, "lda.joblib"))
        shutil.copy2(feat_src, os.path.join(deploy, "feature_columns.txt"))

        np.savez_compressed(
            os.path.join(deploy, "preprocess_numpy.npz"),
            scaler_mean=np.asarray(scaler.mean_, dtype=np.float64),
            scaler_scale=np.asarray(scaler.scale_, dtype=np.float64),
            lda_xbar=np.asarray(lda.xbar_, dtype=np.float64),
            lda_scalings=np.asarray(lda.scalings_, dtype=np.float64),
        )

        # sequences/*/test_windows.npz are ALREADY StandardScaler-transformed.
        # Recover approximate raw counters so on-device e2e (raw->scaler->LDA)
        # matches training. Double-scaling previously collapsed accuracy to ~0.4.
        scaled_te = np.load(os.path.join(SEQ_RAW, "test_windows.npz"))
        X_scaled = scaled_te["X"].astype(np.float64)
        X_raw = X_scaled * np.asarray(scaler.scale_, dtype=np.float64) + np.asarray(
            scaler.mean_, dtype=np.float64
        )
        np.savez_compressed(
            os.path.join(deploy, "test_windows_raw.npz"),
            X=X_raw.astype(np.float32),
            y_binary=scaled_te["y_binary"],
            y_multiclass=scaled_te["y_multiclass"],
        )

        with open(feat_src, encoding="utf-8") as fh:
            n_raw = len([ln for ln in fh if ln.strip()])

        meta = {
            "pipeline": "C_LDA_LSTM",
            "n_features_raw": n_raw,
            "n_features_lda": n_features,
            "n_timesteps": n_timesteps,
            "hidden_size": int(ckpt["hidden_size"]),
            "num_layers": int(ckpt["num_layers"]),
            "dropout": float(ckpt.get("dropout", DROPOUT)),
            "onnx_input": "[batch, timesteps, lda_features]",
            "onnx_output": "logits [batch, 2]  (0=Benign, 1=Attack)",
            "label_map": {"0": "Benign", "1": "Attack"},
            "checkpoint_metrics": {
                "test": test_metrics,
                "n_params": ckpt.get("n_params"),
                "config": {
                    "hidden_size": HIDDEN,
                    "num_layers": NUM_LAYERS,
                    "dropout": DROPOUT,
                    "learning_rate": LR,
                    "batch_size": BATCH,
                    "seed": SEED,
                    "built_from": "fixed_host_Attack_name_first",
                },
            },
            "feature_columns_file": "feature_columns.txt",
        }
        with open(os.path.join(deploy, "model_meta.json"), "w", encoding="utf-8") as fh:
            json.dump(meta, fh, indent=2)
        print("Updated", deploy)

    print("DONE. Edge packages rebuilt from fixed host LDA sequences.")


if __name__ == "__main__":
    main()
