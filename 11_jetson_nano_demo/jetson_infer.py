# ============================================================
# Jetson Nano / edge inference — Pipeline C (LDA-LSTM)
#
# Usage:
#   python3 jetson_infer.py --demo
#   python3 jetson_infer.py --input window.npy
#
# Prefers:
#   1) preprocess_numpy.npz  (avoids sklearn version issues)
#   2) PyTorch best_lstm.pt  (Jetson-friendly; onnxruntime pip
#      wheels often crash with Illegal instruction on Nano)
# ============================================================

from __future__ import print_function

import argparse
import json
import os
import time
import warnings

import numpy as np

warnings.filterwarnings("ignore")


def default_deploy_dir():
    here = os.path.dirname(os.path.abspath(__file__))
    local = os.path.join(here, "pipeline_C_lda_lstm")
    if os.path.isdir(local):
        return local
    return os.path.join(
        here,
        "..",
        "06_edge_deployment",
        "jetson_deploy",
        "pipeline_C_lda_lstm",
    )


def load_meta(deploy_dir):
    with open(os.path.join(deploy_dir, "model_meta.json"), "r") as handle:
        return json.load(handle)


def load_preprocess(deploy_dir):
    npz_path = os.path.join(deploy_dir, "preprocess_numpy.npz")
    if os.path.isfile(npz_path):
        data = np.load(npz_path)
        return {
            "mode": "numpy",
            "mean": data["scaler_mean"].astype(np.float32),
            "scale": data["scaler_scale"].astype(np.float32),
            "xbar": data["lda_xbar"].astype(np.float32),
            "scalings": data["lda_scalings"].astype(np.float32),
        }

    import joblib

    scaler = joblib.load(os.path.join(deploy_dir, "scaler.joblib"))
    lda = joblib.load(os.path.join(deploy_dir, "lda.joblib"))
    return {"mode": "sklearn", "scaler": scaler, "lda": lda}


def load_artifacts(deploy_dir):
    """Kept for the section 11 desktop GUI: meta, scaler, lda, feature names, onnx path."""
    meta = load_meta(deploy_dir)
    prep = load_preprocess(deploy_dir)
    feature_path = os.path.join(deploy_dir, "feature_columns.txt")
    features = []
    if os.path.isfile(feature_path):
        with open(feature_path, "r") as handle:
            features = [ln.strip() for ln in handle if ln.strip()]
    onnx_path = os.path.join(deploy_dir, "lda_lstm.onnx")
    return meta, prep.get("scaler"), prep.get("lda"), features, onnx_path


def preprocess_windows(raw_x, prep):
    x = np.asarray(raw_x, dtype=np.float32)
    if x.ndim == 2:
        x = x[None, ...]
    if x.ndim != 3:
        raise ValueError("Expected (T,F) or (B,T,F), got %s" % (x.shape,))

    b, t, f = x.shape
    flat = x.reshape(-1, f).astype(np.float32)
    # numpy 1.13 (Jetson apt) has limited nan_to_num kwargs
    flat = np.nan_to_num(flat)
    flat[np.isinf(flat)] = 0.0

    if prep["mode"] == "numpy":
        scaled = (flat - prep["mean"]) / prep["scale"]
        reduced = (scaled - prep["xbar"]).dot(prep["scalings"])
    else:
        scaled = prep["scaler"].transform(flat)
        reduced = prep["lda"].transform(scaled)

    return reduced.astype(np.float32).reshape(b, t, -1)


class BaselineLSTMTorch(object):
    def __init__(self, n_features, hidden_size, num_layers, dropout, n_classes=2):
        import torch
        import torch.nn as nn

        class _M(nn.Module):
            def __init__(self):
                super(_M, self).__init__()
                self.lstm = nn.LSTM(
                    input_size=n_features,
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    batch_first=True,
                    dropout=dropout if num_layers > 1 else 0.0,
                )
                self.dropout = nn.Dropout(dropout)
                self.fc = nn.Linear(hidden_size, n_classes)

            def forward(self, x):
                _, (h_n, _) = self.lstm(x)
                return self.fc(self.dropout(h_n[-1]))

        self.torch = torch
        self.model = _M()

    def load(self, ckpt_path, map_location="cpu"):
        ckpt = self.torch.load(ckpt_path, map_location=map_location)
        state = ckpt["model_state"] if isinstance(ckpt, dict) and "model_state" in ckpt else ckpt
        self.model.load_state_dict(state)
        self.model.eval()

    def __call__(self, x_lda):
        with self.torch.no_grad():
            t = self.torch.from_numpy(x_lda.astype(np.float32))
            logits = self.model(t).cpu().numpy()
        return logits


def run_onnx(onnx_path, x_lda):
    import onnxruntime as ort

    session = ort.InferenceSession(
        onnx_path,
        providers=["CPUExecutionProvider"],
    )
    input_name = session.get_inputs()[0].name
    return session.run(None, {input_name: x_lda.astype(np.float32)})[0]


def softmax(logits):
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def build_backend(deploy_dir, meta):
    # Try ONNX first, then PyTorch
    onnx_path = os.path.join(deploy_dir, "lda_lstm.onnx")
    ckpt_path = os.path.join(deploy_dir, "best_lstm.pt")

    try:
        import onnxruntime  # noqa: F401

        def _infer(x):
            return run_onnx(onnx_path, x)

        # smoke
        smoke = np.zeros((1, int(meta["n_timesteps"]), int(meta["n_features_lda"])), np.float32)
        _ = _infer(smoke)
        return "onnx", _infer
    except Exception as exc:
        print("ONNX backend unavailable (%s); trying PyTorch..." % exc)

    model = BaselineLSTMTorch(
        n_features=int(meta["n_features_lda"]),
        hidden_size=int(meta.get("hidden_size", 16)),
        num_layers=int(meta.get("num_layers", 1)),
        dropout=float(meta.get("dropout", 0.2)),
        n_classes=2,
    )
    model.load(ckpt_path, map_location="cpu")

    def _infer(x):
        return model(x)

    return "torch", _infer


def predict(raw_x, deploy_dir, benchmark_runs=0):
    meta = load_meta(deploy_dir)
    prep = load_preprocess(deploy_dir)
    backend_name, infer_fn = build_backend(deploy_dir, meta)

    x_lda = preprocess_windows(raw_x, prep)
    logits = infer_fn(x_lda)
    probs = softmax(logits)
    preds = probs.argmax(axis=1)

    latency_ms = None
    if benchmark_runs > 0:
        for _ in range(5):
            _ = infer_fn(x_lda)
        start = time.time()
        for _ in range(benchmark_runs):
            _ = infer_fn(x_lda)
        latency_ms = ((time.time() - start) / float(benchmark_runs)) * 1000.0

    return {
        "preds": preds,
        "probs": probs,
        "logits": logits,
        "latency_ms": latency_ms,
        "meta": meta,
        "x_lda_shape": x_lda.shape,
        "backend": backend_name,
        "preprocess": prep["mode"],
    }


def demo(deploy_dir, benchmark_runs=50):
    meta = load_meta(deploy_dir)
    t = int(meta["n_timesteps"])
    f = int(meta["n_features_raw"])
    rng = np.random.RandomState(0)
    raw = rng.randn(t, f).astype(np.float32)

    result = predict(raw, deploy_dir, benchmark_runs=benchmark_runs)
    pred = int(result["preds"][0])
    label = meta["label_map"][str(pred)]

    print("=" * 70)
    print("JETSON / EDGE INFERENCE DEMO — Pipeline C")
    print("=" * 70)
    print("Deploy dir   :", deploy_dir)
    print("Preprocess   :", result["preprocess"])
    print("Backend      :", result["backend"])
    print("Raw shape    :", (t, f))
    print("LDA shape    :", result["x_lda_shape"])
    print("Prediction   : %d (%s)" % (pred, label))
    print(
        "Probabilities: Benign=%.4f  Attack=%.4f"
        % (result["probs"][0, 0], result["probs"][0, 1])
    )
    if result["latency_ms"] is not None:
        print(
            "Latency      : %.3f ms / window (%d runs)"
            % (result["latency_ms"], benchmark_runs)
        )


def main():
    parser = argparse.ArgumentParser(description="EVSE LDA-LSTM edge inference")
    parser.add_argument("--deploy-dir", default=default_deploy_dir())
    parser.add_argument("--input", default=None)
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--benchmark-runs", type=int, default=50)
    args = parser.parse_args()

    if args.demo or args.input is None:
        demo(args.deploy_dir, args.benchmark_runs)
        return

    raw = np.load(args.input)
    result = predict(raw, args.deploy_dir, benchmark_runs=args.benchmark_runs)
    for i, (pred, prob) in enumerate(zip(result["preds"], result["probs"])):
        label = result["meta"]["label_map"][str(int(pred))]
        print(
            "[%d] pred=%d (%s)  p_benign=%.4f p_attack=%.4f"
            % (i, int(pred), label, prob[0], prob[1])
        )
    if result["latency_ms"] is not None:
        print("latency_ms=%.3f backend=%s" % (result["latency_ms"], result["backend"]))


if __name__ == "__main__":
    main()
