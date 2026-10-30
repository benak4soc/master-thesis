# ============================================================
# Raspberry Pi 4 EVSE inference — tuned host LDA-LSTM
#
# The Pi is the EVSE host. Each new host-counter sample is
# pushed into a 10-step window. The detector applies the
# fused StandardScaler+LDA map, then a NumPy LSTM.
# No PyTorch, scikit-learn, or ONNX Runtime required.
#
#   python3 rpi4_infer.py --self-test
#   python3 rpi4_infer.py --input window.npy
#   python3 rpi4_infer.py --csv counters.csv
# ============================================================

from __future__ import print_function

import argparse
import csv
import json
import os
import time

import numpy as np


LABELS = {0: "Benign", 1: "Attack"}
FAMILY = {
    0: "Benign",
    1: "Recon",
    2: "DoS",
    3: "Cryptojacking",
    4: "Backdoor",
}


def default_deploy_dir():
    here = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        here,
        os.path.join(here, "rpi4_deploy", "host_lda_lstm"),
        os.path.join(here, "host_lda_lstm"),
    ]
    for path in candidates:
        if os.path.isfile(os.path.join(path, "weights.npz")):
            return path
    return candidates[0]


def load_package(deploy_dir):
    weight_path = os.path.join(deploy_dir, "weights.npz")
    if not os.path.isfile(weight_path):
        raise FileNotFoundError(
            "Missing %s. Run export_rpi4_ids.py on the training PC first."
            % weight_path
        )
    data = np.load(weight_path)
    meta_path = os.path.join(deploy_dir, "model_meta.json")
    with open(meta_path, "r", encoding="utf-8") as handle:
        meta = json.load(handle)
    feature_path = os.path.join(deploy_dir, "feature_columns.txt")
    with open(feature_path, "r", encoding="utf-8") as handle:
        features = [line.strip() for line in handle if line.strip()]
    pack = {
        "A": data["affine_A"].astype(np.float32),
        "b": data["affine_b"].astype(np.float32),
        "w_ih": data["lstm_weight_ih"].astype(np.float32),
        "w_hh": data["lstm_weight_hh"].astype(np.float32),
        "b_ih": data["lstm_bias_ih"].astype(np.float32),
        "b_hh": data["lstm_bias_hh"].astype(np.float32),
        "fc_w": data["fc_weight"].astype(np.float32),
        "fc_b": data["fc_bias"].astype(np.float32),
        "meta": meta,
        "features": features,
    }
    return pack


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -40.0, 40.0)))


def project(raw_x, pack):
    """Raw host counters (T, F) or (B, T, F) -> LDA windows (B, T, 3)."""
    x = np.asarray(raw_x, dtype=np.float32)
    if x.ndim == 2:
        x = x[None, ...]
    if x.ndim != 3:
        raise ValueError("Expected (T, F) or (B, T, F), got %s" % (x.shape,))
    batch, steps, feats = x.shape
    expected = int(pack["meta"]["n_features_raw"])
    if feats != expected:
        raise ValueError(
            "Expected %d host features, got %d" % (expected, feats)
        )
    flat = np.nan_to_num(x.reshape(-1, feats)).astype(np.float32)
    flat[np.isinf(flat)] = 0.0
    reduced = flat.dot(pack["A"]) + pack["b"]
    lda_dim = int(pack["meta"]["n_features_lda"])
    return reduced.reshape(batch, steps, lda_dim)


def lstm_last_hidden(x_lda, pack):
    """One-layer LSTM. x_lda is (B, T, 3). Returns h_T (B, H)."""
    w_ih = pack["w_ih"]
    w_hh = pack["w_hh"]
    b_ih = pack["b_ih"]
    b_hh = pack["b_hh"]
    batch, steps, _ = x_lda.shape
    hidden = w_hh.shape[1]
    h = np.zeros((batch, hidden), dtype=np.float32)
    c = np.zeros((batch, hidden), dtype=np.float32)
    for t in range(steps):
        gates = x_lda[:, t, :].dot(w_ih.T) + b_ih
        gates += h.dot(w_hh.T) + b_hh
        i_gate, f_gate, g_gate, o_gate = np.split(gates, 4, axis=1)
        i_gate = _sigmoid(i_gate)
        f_gate = _sigmoid(f_gate)
        g_gate = np.tanh(g_gate)
        o_gate = _sigmoid(o_gate)
        c = f_gate * c + i_gate * g_gate
        h = o_gate * np.tanh(c)
    return h


def logits_from_raw(raw_x, pack):
    hidden = lstm_last_hidden(project(raw_x, pack), pack)
    return hidden.dot(pack["fc_w"].T) + pack["fc_b"]


def softmax(logits):
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def predict(raw_x, pack, threshold=0.5):
    logits = logits_from_raw(raw_x, pack)
    probs = softmax(logits)
    attack_p = probs[:, 1]
    preds = (attack_p >= threshold).astype(np.int64)
    return preds, probs, logits


def binary_metrics(y_true, y_pred, attack_prob):
    y_true = np.asarray(y_true).astype(np.int64)
    y_pred = np.asarray(y_pred).astype(np.int64)
    tp = int(np.logical_and(y_true == 1, y_pred == 1).sum())
    tn = int(np.logical_and(y_true == 0, y_pred == 0).sum())
    fp = int(np.logical_and(y_true == 0, y_pred == 1).sum())
    fn = int(np.logical_and(y_true == 1, y_pred == 0).sum())
    precision = tp / float(tp + fp) if (tp + fp) else 0.0
    recall = tp / float(tp + fn) if (tp + fn) else 0.0
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if (precision + recall)
        else 0.0
    )
    accuracy = (tp + tn) / float(len(y_true)) if len(y_true) else 0.0
    return {
        "n": int(len(y_true)),
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": _roc_auc(y_true, np.asarray(attack_prob, dtype=np.float64)),
        "confusion_matrix": [[tn, fp], [fn, tp]],
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def _roc_auc(y_true, scores):
    y_true = np.asarray(y_true).astype(np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    n_pos = int((y_true == 1).sum())
    n_neg = int((y_true == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return None
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks = np.empty(len(scores), dtype=np.float64)
    start = 0
    while start < len(scores):
        stop = start
        while (
            stop + 1 < len(scores)
            and sorted_scores[stop + 1] == sorted_scores[start]
        ):
            stop += 1
        avg = 0.5 * (start + stop) + 1.0
        ranks[order[start : stop + 1]] = avg
        start = stop + 1
    sum_pos = float(ranks[y_true == 1].sum())
    return (sum_pos - n_pos * (n_pos + 1) / 2.0) / float(n_pos * n_neg)


def family_detection(y_multi, y_pred):
    """Binary detector recall inside each attack family."""
    y_multi = np.asarray(y_multi).astype(np.int64)
    y_pred = np.asarray(y_pred).astype(np.int64)
    rows = {}
    for class_id, name in FAMILY.items():
        mask = y_multi == class_id
        n = int(mask.sum())
        if n == 0:
            continue
        if class_id == 0:
            correct = int((y_pred[mask] == 0).sum())
            rows[name] = {
                "n": n,
                "correct": correct,
                "rate": correct / float(n),
                "meaning": "specificity",
            }
        else:
            detected = int((y_pred[mask] == 1).sum())
            rows[name] = {
                "n": n,
                "detected": detected,
                "missed": n - detected,
                "recall": detected / float(n),
            }
    return rows


class EVSEWindowDetector(object):
    """Sliding-window IDS for the Raspberry Pi running as the EVSE."""

    def __init__(self, deploy_dir=None, threshold=0.5):
        self.deploy_dir = deploy_dir or default_deploy_dir()
        self.pack = load_package(self.deploy_dir)
        self.threshold = float(threshold)
        self.window = int(self.pack["meta"]["n_timesteps"])
        self.n_features = int(self.pack["meta"]["n_features_raw"])
        self.buffer = []

    def reset(self):
        self.buffer = []

    def push(self, sample):
        """Add one raw counter vector of shape (n_features,).

        Returns a decision dict once the window is full, else None.
        """
        vec = np.asarray(sample, dtype=np.float32).reshape(-1)
        if vec.shape[0] != self.n_features:
            raise ValueError(
                "Expected %d features, got %d"
                % (self.n_features, vec.shape[0])
            )
        self.buffer.append(vec)
        if len(self.buffer) < self.window:
            return None
        if len(self.buffer) > self.window:
            self.buffer = self.buffer[-self.window :]
        window = np.stack(self.buffer, axis=0)
        preds, probs, _ = predict(window, self.pack, self.threshold)
        pred = int(preds[0])
        return {
            "pred": pred,
            "label": LABELS[pred],
            "p_benign": float(probs[0, 0]),
            "p_attack": float(probs[0, 1]),
        }


def _load_feature_table(path, features):
    rows = []
    with open(path, "r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        header = [cell.strip() for cell in header]
        index = []
        if all(name in header for name in features):
            index = [header.index(name) for name in features]
        else:
            numeric_header = True
            try:
                [float(cell) for cell in header]
            except ValueError:
                numeric_header = False
            if numeric_header and len(header) == len(features):
                rows.append([float(cell) for cell in header])
                index = list(range(len(features)))
            elif len(header) >= len(features) and not any(
                name in header for name in features[:5]
            ):
                raise ValueError(
                    "CSV header does not contain the %d host feature names."
                    % len(features)
                )
            else:
                raise ValueError(
                    "CSV must use the training feature column names."
                )
        for record in reader:
            if not record or all(not cell.strip() for cell in record):
                continue
            values = []
            for col in index:
                text = record[col].strip() if col < len(record) else ""
                if text == "" or text.lower() == "nan":
                    values.append(0.0)
                else:
                    values.append(float(text))
            rows.append(values)
    if not rows:
        raise ValueError("CSV has no counter rows: %s" % path)
    return np.asarray(rows, dtype=np.float32)


def run_self_test(deploy_dir, threshold, benchmark_runs):
    pack = load_package(deploy_dir)
    test_path = os.path.join(deploy_dir, "test_windows_raw.npz")
    if not os.path.isfile(test_path):
        raise FileNotFoundError("Missing held-out windows: %s" % test_path)
    data = np.load(test_path)
    raw = data["X"]
    y_bin = data["y_binary"].astype(np.int64)
    y_multi = data["y_multiclass"].astype(np.int64)

    start = time.perf_counter()
    preds, probs, _ = predict(raw, pack, threshold)
    batch_ms = (time.perf_counter() - start) * 1000.0

    one = raw[:1]
    for _ in range(5):
        predict(one, pack, threshold)
    start = time.perf_counter()
    for _ in range(benchmark_runs):
        predict(one, pack, threshold)
    latency_ms = (time.perf_counter() - start) * 1000.0 / float(benchmark_runs)

    metrics = binary_metrics(y_bin, preds, probs[:, 1])
    families = family_detection(y_multi, preds)
    metrics["per_family"] = families
    metrics["batch_ms"] = batch_ms
    metrics["latency_ms_per_window"] = latency_ms
    metrics["windows_per_second"] = 1000.0 / latency_ms if latency_ms else None
    return metrics


def print_metrics(metrics):
    cm = metrics["confusion_matrix"]
    print("Held-out host windows : %d" % metrics["n"])
    print(
        "Accuracy %.4f | Precision %.4f | Recall %.4f | F1 %.4f | ROC-AUC %s"
        % (
            metrics["accuracy"],
            metrics["precision"],
            metrics["recall"],
            metrics["f1"],
            "n/a" if metrics["roc_auc"] is None else "%.4f" % metrics["roc_auc"],
        )
    )
    print("Confusion [TN FP; FN TP]: %s" % cm)
    print("Attack-family detection (binary alert):")
    for name, row in metrics["per_family"].items():
        if "recall" in row:
            print(
                "  %-14s n=%-4d detected=%-4d missed=%-3d recall=%.4f"
                % (name, row["n"], row["detected"], row["missed"], row["recall"])
            )
        else:
            print(
                "  %-14s n=%-4d correct=%-4d specificity=%.4f"
                % (name, row["n"], row["correct"], row["rate"])
            )
    print(
        "Latency          : %.3f ms/window  (%.1f windows/s)"
        % (metrics["latency_ms_per_window"], metrics["windows_per_second"])
    )


def demo(deploy_dir, threshold, benchmark_runs):
    pack = load_package(deploy_dir)
    sample_dir = os.path.join(deploy_dir, "samples")
    print("=" * 70)
    print("RPI4 EVSE IDS — tuned host LDA-LSTM")
    print("=" * 70)
    print("Deploy dir :", deploy_dir)
    print(
        "Input      : (%d, %d) raw host counters -> LDA %d -> LSTM hidden %d"
        % (
            pack["meta"]["n_timesteps"],
            pack["meta"]["n_features_raw"],
            pack["meta"]["n_features_lda"],
            pack["meta"]["hidden_size"],
        )
    )
    names = ["benign", "recon", "dos", "cryptojacking"]
    found = False
    for name in names:
        path = os.path.join(sample_dir, name + ".npy")
        if not os.path.isfile(path):
            continue
        found = True
        raw = np.load(path)
        preds, probs, _ = predict(raw, pack, threshold)
        print(
            "%-16s -> %s   p_attack=%.4f"
            % (name, LABELS[int(preds[0])], float(probs[0, 1]))
        )
    if not found:
        print("No sample windows in", sample_dir)

    metrics = run_self_test(deploy_dir, threshold, benchmark_runs)
    print("-" * 70)
    print_metrics(metrics)


def watch_csv(path, deploy_dir, threshold):
    pack = load_package(deploy_dir)
    table = _load_feature_table(path, pack["features"])
    detector = EVSEWindowDetector(deploy_dir, threshold)
    attacks = 0
    decisions = 0
    last = None
    for row in table:
        result = detector.push(row)
        if result is None:
            continue
        decisions += 1
        if result["pred"] == 1:
            attacks += 1
        if result["label"] != last:
            print(
                "window %d  %s  p_attack=%.4f"
                % (decisions, result["label"], result["p_attack"])
            )
            last = result["label"]
    print(
        "Decisions=%d  Attack windows=%d  Benign windows=%d"
        % (decisions, attacks, decisions - attacks)
    )


def main():
    parser = argparse.ArgumentParser(
        description="Raspberry Pi 4 EVSE host-attack detector"
    )
    parser.add_argument("--deploy-dir", default=default_deploy_dir())
    parser.add_argument("--input", default=None, help="Raw window .npy, shape (T,F) or (B,T,F)")
    parser.add_argument("--csv", default=None, help="CSV of raw host-counter rows")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--benchmark-runs", type=int, default=100)
    args = parser.parse_args()

    if args.demo or (
        args.input is None and args.csv is None and not args.self_test
    ):
        demo(args.deploy_dir, args.threshold, args.benchmark_runs)
        return

    if args.self_test:
        metrics = run_self_test(
            args.deploy_dir, args.threshold, args.benchmark_runs
        )
        print_metrics(metrics)
        return

    if args.csv:
        watch_csv(args.csv, args.deploy_dir, args.threshold)
        return

    raw = np.load(args.input)
    pack = load_package(args.deploy_dir)
    preds, probs, _ = predict(raw, pack, args.threshold)
    for i, (pred, prob) in enumerate(zip(preds, probs)):
        print(
            "[%d] %s  p_benign=%.4f  p_attack=%.4f"
            % (i, LABELS[int(pred)], float(prob[0]), float(prob[1]))
        )


if __name__ == "__main__":
    main()
