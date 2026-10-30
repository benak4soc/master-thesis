# 13 — Raspberry Pi 4: algorithm runtime and dashboard

This section records the edge algorithm that was run on a Raspberry Pi 4 acting as the EVSE, and the dashboard used to demonstrate it. The detector is the tuned host **Pipeline C** model (LDA-LSTM). It is the same binary model as the Jetson package, exported as a NumPy graph so the Pi does not need PyTorch, ONNX Runtime, or scikit-learn.

## Algorithm that runs on the Pi

The Pi is the EVSE host. Each decision uses one window of its own host counters.

| Step | What happens | Size |
|------|----------------|------|
| Window | 10 timesteps of raw HPC / kernel counters | `(10, 220)` |
| Fused scaler + LDA | Train-only StandardScaler and multi-class LDA, stored as one affine map | 220 → 3 |
| LSTM | One layer, hidden size 16, last hidden state, then a linear head | 1,378 parameters |
| Decision | Softmax, threshold 0.5 | `0` Benign, `1` Attack |

The weight file is about **9.6 KB**. Family names are not a fourth model output. Recon, DoS, and Cryptojacking are labels of the replayed host window. The network only scores Benign versus Attack. Backdoor is not in the EVSE-B host file. RomCom is not in that file either.

The NumPy graph was checked against the PyTorch checkpoint on all 843 held-out windows: prediction agreement **1.0**, maximum absolute logit error about **5.2e-6**.

## Held-out detection

Binary result on the host test split (`detection_report.json`):

| Metric | Value |
|--------|------:|
| Windows | 843 |
| Accuracy | 1.000 |
| Precision | 1.000 |
| Recall | 1.000 |
| F1 | 1.000 |
| ROC-AUC | 1.000 |
| Confusion | TN 330, FP 0, FN 0, TP 513 |

Attack-family recall of that same binary alert:

| Family | Detected |
|--------|----------|
| Benign (specificity) | 330 / 330 |
| Recon | 157 / 157 |
| DoS | 103 / 103 |
| Cryptojacking | 253 / 253 |

These scores are the same saturated host split discussed in the thesis reference. They show that the compressed runtime did not drop a test window. They are not a claim about unseen sites or zero-day attacks.

## What was measured on the Raspberry Pi 4

Board: Raspberry Pi 4 Model B Rev 1.4, 64-bit OS, Python 3.9.2, NumPy 1.19.5 already installed.

| Check | Result |
|-------|--------|
| `rpi4_infer.py --self-test` | 843 / 843 windows, about **3.7 ms/window** (about 270 windows/s) |
| Dashboard, DoS replay | Attack, attack probability **0.914**, **4.6 ms**, EVSE state Faulted |
| Dashboard, Cryptojacking replay | Attack. One core at **99%**, package CPU **25%**, **39.4°C**, 1800 MHz, throttling **no** |

The cryptojacking button starts a one-core load on Linux so the health chart moves, and it scores a recorded cryptojacking window. Stop clears that load. The load is not a miner binary.

## Dashboard

`rpi4_dashboard.py` serves one page on port 8080. No extra packages beyond NumPy.

| Control | Relation to the algorithm |
|---------|---------------------------|
| Run Benign | Replays `samples/benign.npy`. Badge follows the model. |
| Simulate DoS | Replays `samples/dos.npy`. Packet rate, half-open count, and the dropped OCPP session are scenario playback so the spike is visible. |
| Simulate Cryptojacking | Replays `samples/cryptojacking.npy`. CPU, temperature, RAM, clock, and throttling are read from the Pi. |
| Simulate RomCom | Scripted RAT storyboard (unexpected shell, beacon, new file). The page states that RomCom is not an LDA-LSTM class. The model is still scored on a benign window and that score is shown separately. |
| Block IP / Throttle / Isolate | Written into the demo log only. Firewall rules are not changed. |
| Alert sound | Optional beep when the attack class changes. |

The summary card keeps two different counts: the live session, and the held-out test metrics above.

## How to run

From this folder, on the Pi or on a PC that has the files:

```bash
python3 rpi4_dashboard.py --host 0.0.0.0 --port 8080
```

Open `http://<pi-address>:8080`.

Command-line check without the page:

```bash
python3 rpi4_infer.py --self-test
```

`rpi4_infer.py` finds `host_lda_lstm/weights.npz` next to itself.

## Limits to keep in the discussion

- The on-device head is binary. DoS and Cryptojacking names on the badge are the replayed window, not a 4-class softmax.
- RomCom and Backdoor are outside the host test set. RomCom on the dashboard is illustrative.
- OCPP values and the DoS packet series are generated for the demo. They are not a live charger capture or a packet tap.
- Host binary scores on this split are saturated. Power remains the harder modality in the rest of the thesis.
