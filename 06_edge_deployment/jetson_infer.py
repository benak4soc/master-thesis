# ============================================================
# Jetson Nano / edge inference — Pipeline C (LDA-LSTM)
#
# Usage (on PC or Jetson):
#   python jetson_infer.py --demo
#   python jetson_infer.py --input window.npy
#
# window.npy shape: (timesteps, raw_features)  OR
#                   (batch, timesteps, raw_features)
#
# Pipeline:
#   raw features -> StandardScaler -> LDA -> ONNX LSTM -> probs
# ============================================================

import argparse
import json
import os
import time
import warnings

import numpy as np
import joblib

warnings.filterwarnings("ignore")


def default_deploy_dir():
    return os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "jetson_deploy",
        "pipeline_C_lda_lstm",
    )


def load_artifacts(deploy_dir):
    meta_path = os.path.join(deploy_dir, "model_meta.json")
    with open(meta_path, encoding="utf-8") as handle:
        meta = json.load(handle)

    scaler = joblib.load(
        os.path.join(deploy_dir, "scaler.joblib")
    )
    lda = joblib.load(
        os.path.join(deploy_dir, "lda.joblib")
    )

    feature_path = os.path.join(
        deploy_dir,
        "feature_columns.txt",
    )
    with open(feature_path, encoding="utf-8") as handle:
        feature_columns = [
            ln.strip() for ln in handle if ln.strip()
        ]

    onnx_path = os.path.join(deploy_dir, "lda_lstm.onnx")

    return meta, scaler, lda, feature_columns, onnx_path


def preprocess_windows(raw_x, scaler, lda):
    """
    raw_x: (T, F) or (B, T, F) raw host features (unscaled)
    returns: (B, T, lda_dim) float32
    """

    x = np.asarray(raw_x, dtype=np.float32)

    if x.ndim == 2:
        x = x[None, ...]

    if x.ndim != 3:
        raise ValueError(
            f"Expected (T,F) or (B,T,F), got {x.shape}"
        )

    b, t, f = x.shape
    flat = x.reshape(-1, f)
    flat = np.nan_to_num(flat, nan=0.0, posinf=0.0, neginf=0.0)
    scaled = scaler.transform(flat)
    reduced = lda.transform(scaled).astype(np.float32)
    return reduced.reshape(b, t, -1)


def run_onnx(onnx_path, x_lda):
    import onnxruntime as ort

    session = ort.InferenceSession(
        onnx_path,
        providers=["CPUExecutionProvider"],
    )
    input_name = session.get_inputs()[0].name
    logits = session.run(
        None,
        {input_name: x_lda.astype(np.float32)},
    )[0]
    return logits


def softmax(logits):
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def predict(raw_x, deploy_dir, benchmark_runs=0):
    meta, scaler, lda, _, onnx_path = load_artifacts(
        deploy_dir
    )

    x_lda = preprocess_windows(raw_x, scaler, lda)
    logits = run_onnx(onnx_path, x_lda)
    probs = softmax(logits)
    preds = probs.argmax(axis=1)

    latency_ms = None
    if benchmark_runs > 0:
        # warmup
        for _ in range(10):
            _ = run_onnx(onnx_path, x_lda)

        start = time.perf_counter()
        for _ in range(benchmark_runs):
            _ = run_onnx(onnx_path, x_lda)
        latency_ms = (
            (time.perf_counter() - start) / benchmark_runs
        ) * 1000.0

    return {
        "preds": preds,
        "probs": probs,
        "logits": logits,
        "latency_ms": latency_ms,
        "meta": meta,
        "x_lda_shape": x_lda.shape,
    }


def demo(deploy_dir, benchmark_runs=100):
    meta, _, _, feature_columns, _ = load_artifacts(
        deploy_dir
    )

    t = int(meta["n_timesteps"])
    f = int(meta["n_features_raw"])

    # Synthetic demo window (replace with real host features)
    rng = np.random.default_rng(0)
    raw = rng.normal(size=(t, f)).astype(np.float32)

    result = predict(
        raw,
        deploy_dir,
        benchmark_runs=benchmark_runs,
    )

    print("=" * 70)
    print("JETSON / EDGE INFERENCE DEMO — Pipeline C")
    print("=" * 70)
    print(f"Deploy dir   : {deploy_dir}")
    print(f"Raw shape    : {(t, f)}")
    print(f"LDA shape    : {result['x_lda_shape']}")
    print(
        f"Prediction   : {int(result['preds'][0])} "
        f"({meta['label_map'][str(int(result['preds'][0]))]})"
    )
    print(
        f"Probabilities: Benign={result['probs'][0, 0]:.4f}  "
        f"Attack={result['probs'][0, 1]:.4f}"
    )

    if result["latency_ms"] is not None:
        print(
            f"ONNX latency : {result['latency_ms']:.3f} ms "
            f"/ window (CPU ORT, {benchmark_runs} runs)"
        )

    print(
        "\nOn Jetson Nano, optionally convert ONNX -> TensorRT "
        "with trtexec for lower latency."
    )


def main():
    parser = argparse.ArgumentParser(
        description="EVSE LDA-LSTM edge inference"
    )
    parser.add_argument(
        "--deploy-dir",
        default=default_deploy_dir(),
        help="Path to jetson_deploy/pipeline_C_lda_lstm",
    )
    parser.add_argument(
        "--input",
        default=None,
        help="Path to .npy window array (T,F) or (B,T,F)",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run synthetic demo + latency benchmark",
    )
    parser.add_argument(
        "--benchmark-runs",
        type=int,
        default=100,
    )
    args = parser.parse_args()

    if args.demo or args.input is None:
        demo(args.deploy_dir, args.benchmark_runs)
        return

    raw = np.load(args.input)
    result = predict(
        raw,
        args.deploy_dir,
        benchmark_runs=args.benchmark_runs,
    )

    for i, (pred, prob) in enumerate(
        zip(result["preds"], result["probs"])
    ):
        label = result["meta"]["label_map"][str(int(pred))]
        print(
            f"[{i}] pred={int(pred)} ({label})  "
            f"p_benign={prob[0]:.4f} p_attack={prob[1]:.4f}"
        )

    if result["latency_ms"] is not None:
        print(
            f"latency_ms={result['latency_ms']:.3f}"
        )


if __name__ == "__main__":
    main()
