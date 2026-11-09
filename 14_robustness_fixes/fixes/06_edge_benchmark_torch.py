"""
On-device edge benchmark for Jetson Nano (PyTorch backend).

onnxruntime pip wheels often crash with Illegal instruction on Jetson Nano, so
this script mirrors fixes/06_edge_benchmark.py using best_lstm.pt +
preprocess_numpy.npz (same path as 11_jetson_nano_demo/jetson_infer.py).

Compatible with Python 3.6 / NumPy 1.13 on JetPack 4.6.

Usage on the board:
    python3 fixes/06_edge_benchmark_torch.py \
        --model-dir pipeline_C_lda_lstm \
        --raw-windows test_windows_raw.npz \
        --device-name "Jetson Nano 4GB MAXN"
"""
from __future__ import print_function

import argparse
import json
import os
import platform
import time

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--model-dir", default="pipeline_C_lda_lstm")
ap.add_argument("--raw-windows", default="test_windows_raw.npz")
ap.add_argument("--device-name", default=platform.node())
ap.add_argument("--reps", type=int, default=500)
ap.add_argument("--warmup", type=int, default=30)
a = ap.parse_args()

try:
    import psutil
    proc = psutil.Process()
except Exception:
    psutil, proc = None, None

import torch
import torch.nn as nn

pre = np.load(os.path.join(a.model_dir, "preprocess_numpy.npz"))
mean = pre["scaler_mean"].astype(np.float32)
scale = pre["scaler_scale"].astype(np.float32)
xbar = pre["lda_xbar"].astype(np.float32)
scal = pre["lda_scalings"].astype(np.float32)

with open(os.path.join(a.model_dir, "model_meta.json"), "r") as fh:
    meta = json.load(fh)

hidden = int(meta.get("hidden_size", 16))
n_layers = int(meta.get("num_layers", 1))
dropout = float(meta.get("dropout", 0.2))
n_feat = int(meta.get("n_features_lda", 3))


class BaselineLSTM(nn.Module):
    def __init__(self):
        super(BaselineLSTM, self).__init__()
        self.lstm = nn.LSTM(
            input_size=n_feat,
            hidden_size=hidden,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden, 2)

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        return self.fc(self.drop(h_n[-1]))


def preprocess(X):
    # X: (B, T, 220) raw
    X = np.asarray(X, dtype=np.float32)
    b, t, f = X.shape
    flat = X.reshape(-1, f)
    flat = np.nan_to_num(flat)
    flat[np.isinf(flat)] = 0.0
    Z = (flat - mean) / scale
    R = (Z - xbar).dot(scal)
    return R.astype(np.float32).reshape(b, t, -1)


rss0 = proc.memory_info().rss if proc else 0
t0 = time.time()
model = BaselineLSTM()
ckpt = torch.load(os.path.join(a.model_dir, "best_lstm.pt"), map_location="cpu")
state = ckpt["model_state"] if isinstance(ckpt, dict) and "model_state" in ckpt else ckpt
model.load_state_dict(state)
model.eval()
cold_ms = (time.time() - t0) * 1000.0

raw = np.load(a.raw_windows)
X = raw["X"].astype(np.float32)
if "y_binary" in raw.files:
    y = raw["y_binary"]
else:
    y = raw["y"]


def infer(xb):
    with torch.no_grad():
        lda = preprocess(xb)
        logits = model(torch.from_numpy(lda))
        return logits.cpu().numpy()


# correctness
preds = []
for i in range(0, len(X), 64):
    preds.append(infer(X[i:i + 64]).argmax(1))
pred = np.concatenate(preds)
acc = float((pred == y).mean())

# warmup + latency batch-1 end-to-end
for _ in range(a.warmup):
    infer(X[:1])
if psutil:
    proc.cpu_percent(None)

lat = []
for i in range(a.reps):
    xb = X[i % len(X)][None]
    s = time.time()
    infer(xb)
    lat.append((time.time() - s) * 1000.0)
cpu = proc.cpu_percent(None) if psutil else None
lat = np.asarray(lat, dtype=np.float64)

thr = {}
for b in (1, 32, 64):
    # np.resize exists on 1.13
    xb = np.resize(X, (b,) + X.shape[1:]).astype(np.float32)
    n = 0
    s = time.time()
    while time.time() - s < 3.0:
        infer(xb)
        n += b
    thr["batch_%d" % b] = n / (time.time() - s)

pt_size = os.path.getsize(os.path.join(a.model_dir, "best_lstm.pt")) / 1024.0
onnx_path = os.path.join(a.model_dir, "lda_lstm.onnx")
onnx_kb = os.path.getsize(onnx_path) / 1024.0 if os.path.isfile(onnx_path) else None

res = {
    "device": a.device_name,
    "machine": platform.machine(),
    "python": platform.python_version(),
    "backend": "torch",
    "torch": torch.__version__,
    "numpy": np.__version__,
    "pt_size_kb": pt_size,
    "onnx_size_kb": onnx_kb,
    "cold_start_ms": cold_ms,
    "latency_ms_e2e_batch1": {
        "median": float(np.median(lat)),
        "p95": float(np.percentile(lat, 95)),
        "p99": float(np.percentile(lat, 99)),
        "max": float(lat.max()),
        "mean": float(lat.mean()),
    },
    "throughput_windows_per_s": thr,
    "peak_rss_mb": (proc.memory_info().rss / float(2 ** 20)) if proc else None,
    "rss_increase_mb": ((proc.memory_info().rss - rss0) / float(2 ** 20)) if proc else None,
    "cpu_percent_during_latency_loop": cpu,
    "test_accuracy_on_device": acc,
    "n_test_windows": int(len(y)),
    "reps": a.reps,
}

safe = a.device_name.replace(" ", "_").replace("/", "-")
out = "edge_benchmark_%s.json" % safe
with open(out, "w") as fh:
    json.dump(res, fh, indent=2)
print(json.dumps(res, indent=2))
print("Wrote", os.path.abspath(out))
