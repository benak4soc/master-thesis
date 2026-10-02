# 11 — Jetson Nano Demo (GUI)

Live demonstration package for **Pipeline C (LDA-LSTM)** on EVSE-B host HPC/kernel events.

## What it detects
- **Binary IDS:** Benign vs Attack
- **Input:** window `(10, 220)` host features
- **On-device pipeline:** StandardScaler → LDA (3-D) → ONNX LSTM

## Contents
| File / folder | Role |
|---------------|------|
| `JETSON_NANO_DEPLOY_TODO.md` | Full Jetson setup checklist |
| `jetson_demo_gui.py` | Presentation GUI |
| `jetson_infer.py` | CLI + inference backend |
| `pipeline_C_lda_lstm/` | ONNX model + scaler + LDA + meta |

## Run on PC (rehearsal)
```bash
cd 11_jetson_nano_demo
pip install numpy joblib scikit-learn onnxruntime
python jetson_demo_gui.py
```

## Run on Jetson Nano
1. Follow `JETSON_NANO_DEPLOY_TODO.md`
2. Copy this whole `11_jetson_nano_demo/` folder to the Nano
3. `sudo apt-get install -y python3-tk`
4. `pip3 install --user numpy joblib scikit-learn onnxruntime`
5. `python3 jetson_demo_gui.py`

CLI alternative:
```bash
python3 jetson_infer.py --demo
```

Export / packaging scripts remain in `06_edge_deployment/`.
