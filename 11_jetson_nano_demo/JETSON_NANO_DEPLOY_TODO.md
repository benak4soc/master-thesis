# Jetson Nano — Deploy TODO & What the Model Detects

Edge model: **Pipeline C (LDA-LSTM)** on **EVSE-B Host HPC / kernel events**  
Package: `jetson_deploy/pipeline_C_lda_lstm/`  
Demo GUI: `cd 11_jetson_nano_demo && python jetson_demo_gui.py`
(GitHub folder: `11_jetson_nano_demo/`)

---

## What the model has to detect

| Item | Detail |
|------|--------|
| **Modality** | Host HPC / kernel performance counters (220 raw features → LDA → 3 dims) |
| **Input window** | Shape `(10, 220)` = 10 timesteps × 220 host features |
| **Task** | **Binary IDS**: `0 = Benign`, `1 = Attack` |
| **Not multi-class on device** | Family labels (Recon / DoS / Cryptojacking / Backdoor) are thesis analysis; the ONNX edge model outputs Benign vs Attack only |
| **Attack meaning** | Any malicious host activity in CICEVSE2024 host events (scans, floods, crypto-mining, backdoor-style activity reflected in HPC/kernel counters) |
| **Pipeline on device** | raw window → `StandardScaler` → `LDA` → ONNX LSTM → softmax → Benign / Attack + confidence |

**Demo message for examiners:**  
> “This Jetson Nano runs a lightweight LDA-LSTM that watches short windows of EVSE host telemetry and raises an **Attack** alert when the sequence looks malicious.”

---

## TODO checklist — prepare the Jetson Nano

### A. Hardware & OS
- [ ] Jetson Nano (2GB or 4GB) powered and networked (Ethernet or Wi‑Fi)
- [ ] MicroSD with **NVIDIA JetPack** (recommended: JetPack 4.6.x for Nano)
- [ ] First boot: set user, password, locale; enable SSH if you will push files remotely
- [ ] Expand filesystem if needed; free disk space (≥2 GB free for deps + models)
- [ ] Optional: attach display + keyboard for live GUI demo (or use SSH + X11 / VNC)

### B. Copy deploy package to Jetson
- [ ] Copy this folder tree to the Nano, e.g. `~/evse-ids/`:
  - `jetson_deploy/pipeline_C_lda_lstm/` (onnx, joblibs, meta, features)
  - `jetson_infer.py`
  - `jetson_demo_gui.py`
  - `JETSON_NANO_DEPLOY_TODO.md` (this file)
- [ ] Confirm files exist:
  - `lda_lstm.onnx`
  - `scaler.joblib`
  - `lda.joblib`
  - `feature_columns.txt`
  - `model_meta.json`

### C. Python environment on Jetson
- [ ] Prefer system Python 3 that ships with JetPack (`python3`)
- [ ] Install packages (CPU ONNX Runtime is enough for the demo):
  ```bash
  sudo apt-get update
  sudo apt-get install -y python3-pip python3-tk
  pip3 install --user numpy joblib scikit-learn onnxruntime
  ```
- [ ] If `joblib` / `sklearn` versions mismatch with PC-exported `.joblib`, recreate scaler/LDA on the Nano with matching sklearn, **or** keep the same major sklearn version as the training PC
- [ ] Smoke test CLI:
  ```bash
  cd ~/evse-ids
  python3 jetson_infer.py --demo
  ```
- [ ] Expected: prints Benign/Attack probabilities + latency (ms/window)

### D. Optional — TensorRT (faster, thesis edge metrics)
- [ ] Confirm TensorRT / `trtexec` present (JetPack)
- [ ] Convert ONNX → engine:
  ```bash
  /usr/src/tensorrt/bin/trtexec \
    --onnx=jetson_deploy/pipeline_C_lda_lstm/lda_lstm.onnx \
    --saveEngine=jetson_deploy/pipeline_C_lda_lstm/lda_lstm.trt \
    --fp16
  ```
- [ ] Record engine size and compare latency vs ONNX Runtime (thesis § edge metrics)

### E. GUI demonstration
- [ ] On Nano with display:
  ```bash
  python3 jetson_demo_gui.py
  ```
- [ ] Or on PC first (same GUI) to rehearse the talk, then on Jetson for the live show
- [ ] Rehearse buttons: **Load artifacts** → **Synthetic demo** → **Load .npy window** → **Benchmark**
- [ ] Have 1–2 saved `.npy` windows ready (benign-like / attack-like) if you collect real host windows later

### F. Thesis metrics to capture on Jetson
- [ ] Inference latency (ms/window) — use GUI Benchmark or `jetson_infer.py --demo`
- [ ] Throughput (windows/s) ≈ `1000 / latency_ms`
- [ ] Model size: ONNX (and TRT if built)
- [ ] RAM / CPU (and GPU if TRT) during inference (`tegrastats`)
- [ ] Optional: power draw (USB power meter or `tegrastats`)

### G. Demo script (suggested live flow)
1. Open GUI → show **What this detects** panel  
2. Load artifacts → green status  
3. Run **Synthetic window** → explain pipeline (Scaler → LDA → LSTM)  
4. Run **Benchmark** → quote latency on Jetson  
5. Load a real `.npy` window if available → show Benign vs Attack + confidence bar  
6. Mention: full multi-class / network / power evaluation stayed offline; **edge deploy = host binary Pipeline C**

---

## Quick troubleshooting

| Symptom | Fix |
|---------|-----|
| `No module named 'onnxruntime'` | `pip3 install --user onnxruntime` |
| `No module named 'tkinter'` | `sudo apt-get install python3-tk` |
| joblib/sklearn load error | Align `scikit-learn` version with training PC |
| Very slow first run | Normal (ORT / TRT warmup); use Benchmark after warmup |
| Out of memory (2GB Nano) | Close browsers; use ORT CPU only; avoid heavy desktop apps |
