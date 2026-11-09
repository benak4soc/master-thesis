"""
Rebuild RPi4 NumPy package from the *fixed* host Pipeline C (h16).
Run from master-thesis root:
  python 13_rpi4_algorithm_dashboard/export_fixed_rpi4_package.py
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
sys.path.insert(0, os.path.join(REPO, "13_rpi4_algorithm_dashboard"))

# Prefer fixed tuned checkpoint from rebuild_edge_package
CKPT = os.path.join(REPO, "models", "pipeline_C_lda_lstm_tuned", "best_lstm.pt")
HOST = os.path.join(REPO, "09_processed_datasets", "processed_EVSE_B_Host")
SEQ = os.path.join(HOST, "sequences")
SCALER_PATH = os.path.join(SEQ, "scaler.joblib")
LDA_PATH = os.path.join(SEQ, "lda_multiclass", "lda.joblib")
FEATURE_PATH = os.path.join(SEQ, "feature_columns.txt")
RAW_TEST = os.path.join(SEQ, "test_windows.npz")
LDA_TEST = os.path.join(SEQ, "lda_multiclass", "test_windows.npz")

OUT_DIRS = [
    os.path.join(REPO, "13_rpi4_algorithm_dashboard", "host_lda_lstm"),
    r"E:\rpi_thesisproject\dataset\rpi4_deploy\host_lda_lstm",
]

# Copy rpi4_infer from E: project if missing locally
INFER_SRC_CANDIDATES = [
    os.path.join(REPO, "13_rpi4_algorithm_dashboard", "rpi4_infer.py"),
    r"E:\rpi_thesisproject\dataset\rpi4_infer.py",
]


def fused_affine(scaler, lda):
    mean = np.asarray(scaler.mean_, dtype=np.float64)
    scale = np.asarray(scaler.scale_, dtype=np.float64)
    scale = np.where(scale == 0.0, 1.0, scale)
    xbar = np.asarray(lda.xbar_, dtype=np.float64)
    scalings = np.asarray(lda.scalings_, dtype=np.float64)
    n_components = int(getattr(lda, "n_components_", lda.n_components))
    scalings = scalings[:, :n_components]
    affine_a = scalings / scale[:, None]
    affine_b = -((mean / scale) + xbar).dot(scalings)
    return affine_a, affine_b


def lstm_weights(state):
    return {
        "lstm_weight_ih": state["lstm.weight_ih_l0"].detach().cpu().numpy(),
        "lstm_weight_hh": state["lstm.weight_hh_l0"].detach().cpu().numpy(),
        "lstm_bias_ih": state["lstm.bias_ih_l0"].detach().cpu().numpy(),
        "lstm_bias_hh": state["lstm.bias_hh_l0"].detach().cpu().numpy(),
        "fc_weight": state["fc.weight"].detach().cpu().numpy(),
        "fc_bias": state["fc.bias"].detach().cpu().numpy(),
    }


def main():
    infer_src = next(p for p in INFER_SRC_CANDIDATES if os.path.isfile(p))
    # Import helpers from that file's directory
    sys.path.insert(0, os.path.dirname(infer_src))
    from rpi4_infer import (  # noqa: E402
        binary_metrics,
        family_detection,
        load_package,
        logits_from_raw,
        predict,
        run_self_test,
    )
    from lstm_common import BaselineLSTM  # noqa: E402

    ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
    scaler = joblib.load(SCALER_PATH)
    lda = joblib.load(LDA_PATH)
    affine_a, affine_b = fused_affine(scaler, lda)
    weights = lstm_weights(ckpt["model_state"])

    scaled_test = np.load(RAW_TEST)
    scaled = scaled_test["X"].astype(np.float64)
    y_bin = scaled_test["y_binary"].astype(np.int64)
    y_multi = scaled_test["y_multiclass"].astype(np.int64)
    mean = np.asarray(scaler.mean_, dtype=np.float64)
    scale = np.asarray(scaler.scale_, dtype=np.float64)
    scale = np.where(scale == 0.0, 1.0, scale)
    raw = scaled * scale + mean
    raw32 = raw.astype(np.float32)

    lda_hat = (raw.reshape(-1, raw.shape[-1]).dot(affine_a) + affine_b).reshape(
        raw.shape[0], raw.shape[1], -1
    )
    lda_ref = np.load(LDA_TEST)["X"].astype(np.float64)
    lda_err = float(np.max(np.abs(lda_hat - lda_ref)))
    print("Max |fused LDA - sklearn LDA| : %.3e" % lda_err)

    model = BaselineLSTM(
        n_features=int(ckpt["n_features"]),
        hidden_size=int(ckpt["hidden_size"]),
        num_layers=int(ckpt["num_layers"]),
        dropout=float(ckpt.get("dropout", 0.0)),
    )
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    with torch.no_grad():
        ref_logits = model(torch.from_numpy(lda_ref.astype(np.float32))).cpu().numpy()

    n_params = int(sum(v.size for v in weights.values()))
    meta = {
        "role": "Raspberry Pi 4 acts as the EVSE host",
        "pipeline": "C_LDA_LSTM_TUNED",
        "modality": "EVSE-B host HPC/kernel counters",
        "task": "binary Benign vs Attack",
        "runtime": "numpy fused StandardScaler+LDA + one-layer LSTM",
        "n_features_raw": int(raw.shape[-1]),
        "n_features_lda": int(ckpt["n_features"]),
        "n_timesteps": int(ckpt["n_timesteps"]),
        "hidden_size": int(ckpt["hidden_size"]),
        "num_layers": int(ckpt["num_layers"]),
        "n_params": n_params,
        "decision_threshold": 0.5,
        "built_from": "fixed_host_Attack_name_first",
        "label_map": {"0": "Benign", "1": "Attack"},
        "attack_families_in_host_test": ["Recon", "DoS", "Cryptojacking"],
        "note": "Backdoor not in host file. Family scores = binary detection recall.",
    }

    for deploy in OUT_DIRS:
        os.makedirs(deploy, exist_ok=True)
        np.savez_compressed(
            os.path.join(deploy, "weights.npz"),
            affine_A=affine_a.astype(np.float32),
            affine_b=affine_b.astype(np.float32),
            **{k: v.astype(np.float32) for k, v in weights.items()},
        )
        shutil.copy2(FEATURE_PATH, os.path.join(deploy, "feature_columns.txt"))
        shutil.copy2(infer_src, os.path.join(deploy, "rpi4_infer.py"))
        with open(os.path.join(deploy, "model_meta.json"), "w", encoding="utf-8") as fh:
            json.dump(meta, fh, indent=2)
        np.savez_compressed(
            os.path.join(deploy, "test_windows_raw.npz"),
            X=raw32,
            y_binary=y_bin,
            y_multiclass=y_multi,
        )
        sample_dir = os.path.join(deploy, "samples")
        os.makedirs(sample_dir, exist_ok=True)
        saved = {}
        for name, class_id in (("benign", 0), ("recon", 1), ("dos", 2), ("cryptojacking", 3)):
            hits = np.flatnonzero(y_multi == class_id)
            if len(hits) == 0:
                continue
            idx = int(hits[0])
            np.save(os.path.join(sample_dir, name + ".npy"), raw32[idx])
            saved[name] = {"index": idx, "y_binary": int(y_bin[idx])}
        with open(os.path.join(sample_dir, "labels.json"), "w", encoding="utf-8") as fh:
            json.dump(saved, fh, indent=2)

        pack = load_package(deploy)
        pi_logits = logits_from_raw(raw32, pack)
        logit_err = float(np.max(np.abs(pi_logits - ref_logits)))
        ref_pred = ref_logits.argmax(axis=1)
        pi_pred, pi_prob, _ = predict(raw32, pack, threshold=0.5)
        agreement = float(np.mean(pi_pred == ref_pred))
        metrics = binary_metrics(y_bin, pi_pred, pi_prob[:, 1])
        metrics["per_family"] = family_detection(y_multi, pi_pred)
        timing = run_self_test(deploy, 0.5, 50)
        report = {
            "target": "Raspberry Pi 4 (CPU, NumPy)",
            "built_from": "fixed_host_Attack_name_first",
            "n_params": n_params,
            "max_abs_lda_error": lda_err,
            "max_abs_logit_error_vs_torch": logit_err,
            "prediction_agreement_vs_torch": agreement,
            "test_metrics": {
                "accuracy": metrics["accuracy"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
                "roc_auc": metrics["roc_auc"],
                "confusion_matrix": metrics["confusion_matrix"],
            },
            "per_family": metrics["per_family"],
            "latency_ms_per_window_this_pc": timing["latency_ms_per_window"],
        }
        with open(os.path.join(deploy, "detection_report.json"), "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2)
        print("Updated", deploy)
        print(
            "  agreement=%.4f f1=%.4f lda_err=%.2e logit_err=%.2e"
            % (agreement, metrics["f1"], lda_err, logit_err)
        )
        if agreement < 0.999 or metrics["f1"] < 0.999:
            raise SystemExit("Export quality check failed for %s" % deploy)

    print("DONE")


if __name__ == "__main__":
    main()
