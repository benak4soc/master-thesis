"""
FIX 5 — On-device benchmark (run this ON the Jetson Nano and ON the Raspberry Pi 4).

Measures the FULL inference path the thesis claims is edge-ready:
    raw 220 host counters -> StandardScaler -> LDA (220->3) -> LSTM(h16) ONNX -> logits
and reports what a reviewer will ask for:
    model size, cold-start time, latency median/p95/p99 (batch 1),
    throughput (windows/s) at batch 1/32/256, peak RSS memory, CPU %,
    and a correctness check against test_windows_raw.npz.

Power draw: run in parallel and attach the log
    Jetson : sudo tegrastats --interval 500 --logfile tegrastats.log
    RPi4   : USB power meter (e.g. UM25C) or `vcgencmd measure_volts/measure_temp`

Usage (copy 11_jetson_nano_demo/pipeline_C_lda_lstm + this file + the raw test windows):
    pip install numpy onnxruntime psutil
    python 06_edge_benchmark.py --model-dir pipeline_C_lda_lstm \
        --raw-windows test_windows_raw.npz --device-name "Jetson Nano 4GB (MAXN)"
    # GPU on Jetson (if onnxruntime-gpu / TensorRT EP installed):
    python 06_edge_benchmark.py ... --provider CUDAExecutionProvider
Output:
    edge_benchmark_<device>.json   -> put the numbers in the thesis table
"""

import argparse
import json
import os
import platform
import time

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--model-dir", default="11_jetson_nano_demo/pipeline_C_lda_lstm")
ap.add_argument("--raw-windows", default="13_rpi4_algorithm_dashboard/host_lda_lstm/test_windows_raw.npz",
                help="npz with X (N,10,220) UNSCALED counters and y / y_binary")
ap.add_argument("--device-name", default=platform.node())
ap.add_argument("--provider", default="CPUExecutionProvider")
ap.add_argument("--reps", type=int, default=2000)
a = ap.parse_args()

import onnxruntime as ort  # noqa: E402

try:
    import psutil
    proc = psutil.Process()
except ImportError:
    psutil, proc = None, None

onnx_path = os.path.join(a.model_dir, "lda_lstm.onnx")
pre = np.load(os.path.join(a.model_dir, "preprocess_numpy.npz"))
mean, scale = pre["scaler_mean"], pre["scaler_scale"]
xbar, scal = pre["lda_xbar"], pre["lda_scalings"]


def preprocess(X):                       # X: (B,10,220) raw counters
    Z = (X - mean) / scale               # StandardScaler
    return ((Z - xbar) @ scal).astype(np.float32)   # sklearn LDA(svd).transform


rss0 = proc.memory_info().rss if proc else 0
t0 = time.perf_counter()
so = ort.SessionOptions()
so.intra_op_num_threads = os.cpu_count()
sess = ort.InferenceSession(onnx_path, so, providers=[a.provider])
cold_ms = (time.perf_counter() - t0) * 1000
inp = sess.get_inputs()[0].name

raw = np.load(a.raw_windows)
X = raw["X"].astype(np.float64)
y = raw["y_binary"] if "y_binary" in raw else raw["y"]


def infer(xb):
    return sess.run(None, {inp: preprocess(xb)})[0]


# correctness on the full test set
pred = np.concatenate([infer(X[i:i + 256]).argmax(1) for i in range(0, len(X), 256)])
acc = float((pred == y).mean())

# latency, batch 1, end-to-end (preprocess + model)
for _ in range(50):
    infer(X[:1])
if psutil:
    proc.cpu_percent(None)
lat = []
for i in range(a.reps):
    xb = X[i % len(X)][None]
    s = time.perf_counter()
    infer(xb)
    lat.append((time.perf_counter() - s) * 1000)
cpu = proc.cpu_percent(None) if psutil else None
lat = np.array(lat)

thr = {}
for b in (1, 32, 256):
    xb = np.resize(X, (b,) + X.shape[1:])
    n, s = 0, time.perf_counter()
    while time.perf_counter() - s < 3.0:
        infer(xb)
        n += b
    thr[f"batch_{b}"] = n / (time.perf_counter() - s)

res = {
    "device": a.device_name, "machine": platform.machine(), "python": platform.python_version(),
    "onnxruntime": ort.__version__, "provider": a.provider,
    "onnx_size_kb": os.path.getsize(onnx_path) / 1024,
    "cold_start_ms": cold_ms,
    "latency_ms_e2e_batch1": {"median": float(np.median(lat)), "p95": float(np.percentile(lat, 95)),
                              "p99": float(np.percentile(lat, 99)), "max": float(lat.max())},
    "throughput_windows_per_s": thr,
    "peak_rss_mb": (proc.memory_info().rss / 2**20) if proc else None,
    "rss_increase_mb": ((proc.memory_info().rss - rss0) / 2**20) if proc else None,
    "cpu_percent_during_latency_loop": cpu,
    "test_accuracy_on_device": acc, "n_test_windows": int(len(y)),
}
out = f"edge_benchmark_{a.device_name.replace(' ', '_').replace('/', '-')}.json"
with open(out, "w") as fh:
    json.dump(res, fh, indent=2)
print(json.dumps(res, indent=2))
print(f"\nSaved {out}")
