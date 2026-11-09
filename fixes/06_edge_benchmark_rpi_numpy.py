"""
On-device edge benchmark for Raspberry Pi 4 (NumPy LDA-LSTM).
No torch / onnxruntime required.

  python3 06_edge_benchmark_rpi_numpy.py \
    --model-dir . \
    --raw-windows test_windows_raw.npz \
    --device-name "RPi4 4GB"
"""
from __future__ import print_function

import argparse
import json
import os
import platform
import sys
import time

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--model-dir", default=".")
ap.add_argument("--raw-windows", default="test_windows_raw.npz")
ap.add_argument("--device-name", default="RPi4")
ap.add_argument("--reps", type=int, default=500)
ap.add_argument("--warmup", type=int, default=30)
a = ap.parse_args()

sys.path.insert(0, a.model_dir)
from rpi4_infer import load_package, predict  # noqa: E402

try:
    import psutil
    proc = psutil.Process()
except Exception:
    psutil, proc = None, None

rss0 = proc.memory_info().rss if proc else 0
t0 = time.time()
pack = load_package(a.model_dir)
cold_ms = (time.time() - t0) * 1000.0

raw = np.load(a.raw_windows)
X = raw["X"].astype(np.float32)
y = raw["y_binary"] if "y_binary" in raw.files else raw["y"]

# correctness
pred, prob, _ = predict(X, pack, threshold=0.5)
acc = float(np.mean(pred == y))

# warmup + latency batch-1 end-to-end
for _ in range(a.warmup):
    predict(X[:1], pack, threshold=0.5)
if psutil:
    proc.cpu_percent(None)

lat = []
for i in range(a.reps):
    xb = X[i % len(X)][None]
    s = time.time()
    predict(xb, pack, threshold=0.5)
    lat.append((time.time() - s) * 1000.0)
cpu = proc.cpu_percent(None) if psutil else None
lat = np.asarray(lat, dtype=np.float64)

thr = {}
for b in (1, 32, 64):
    xb = np.resize(X, (b,) + X.shape[1:]).astype(np.float32)
    n = 0
    s = time.time()
    while time.time() - s < 3.0:
        predict(xb, pack, threshold=0.5)
        n += b
    thr["batch_%d" % b] = n / (time.time() - s)

wpath = os.path.join(a.model_dir, "weights.npz")
res = {
    "device": a.device_name,
    "machine": platform.machine(),
    "python": platform.python_version(),
    "backend": "numpy",
    "numpy": np.__version__,
    "weights_size_kb": os.path.getsize(wpath) / 1024.0,
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
