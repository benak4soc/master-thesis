Raspberry Pi 4 EVSE detector
================================
The Pi is the EVSE host. This package scores 10-step windows of the
same 220 host counters used in training.

Pipeline: raw counters -> fused scaler+LDA (220 -> 3) -> LSTM (hidden 16) -> Benign/Attack

On the Pi (64-bit Raspberry Pi OS, Python 3), from this folder:
    sudo apt-get install -y python3-numpy
    python3 rpi4_infer.py --self-test
    python3 rpi4_infer.py --demo

Copy this whole folder to the Pi. rpi4_infer.py sits next to weights.npz.

Live charging session: push one raw counter vector per sample.

    from rpi4_infer import EVSEWindowDetector
    ids = EVSEWindowDetector()
    decision = ids.push(counter_vector)   # shape (220,)
    # decision["label"] is "Benign" or "Attack" once 10 samples are buffered

--self-test replays the held-out host windows and prints recall for
Recon, DoS, and Cryptojacking. Backdoor is not in the host dataset.
