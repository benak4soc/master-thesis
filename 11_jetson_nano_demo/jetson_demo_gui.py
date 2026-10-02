# ============================================================
# EVSE Attack Detection — Jetson / Edge Demo GUI
# Pipeline C: LDA-LSTM (Host HPC/kernel → Benign vs Attack)
#
# Run (PC or Jetson Nano):
#   python jetson_demo_gui.py
#   python jetson_demo_gui.py --deploy-dir path/to/pipeline_C_lda_lstm
# ============================================================

from __future__ import annotations

import argparse
import os
import threading
import time
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import numpy as np

from jetson_infer import default_deploy_dir, load_artifacts, predict


ATTACK_FAMILIES_TEXT = (
    "Binary edge decision: Benign  |  Attack\n\n"
    "Input: EVSE-B host HPC / kernel counter windows\n"
    "  • Shape: (10 timesteps × 220 features)\n"
    "  • Pipeline: StandardScaler → LDA (3-D) → LSTM (ONNX)\n\n"
    "Attack covers malicious host activity seen in CICEVSE2024,\n"
    "including Recon, DoS, Cryptojacking, and Backdoor-style\n"
    "behaviour reflected in HPC/kernel telemetry.\n\n"
    "This GUI is for live demonstration on PC or Jetson Nano."
)


class EvseIdsGui(tk.Tk):
    def __init__(self, deploy_dir: str):
        super().__init__()
        self.title("EVSE IDS Demo — Pipeline C (LDA-LSTM)")
        self.geometry("920x640")
        self.minsize(820, 580)

        self.deploy_dir = tk.StringVar(value=deploy_dir)
        self.status = tk.StringVar(value="Ready — load artifacts to start")
        self.pred_label = tk.StringVar(value="—")
        self.prob_benign = tk.StringVar(value="—")
        self.prob_attack = tk.StringVar(value="—")
        self.latency = tk.StringVar(value="—")
        self.shape_info = tk.StringVar(value="—")
        self.meta_info = tk.StringVar(value="—")

        self._meta = None
        self._busy = False
        self._build()

    def _build(self):
        pad = {"padx": 10, "pady": 6}
        root = ttk.Frame(self, padding=12)
        root.pack(fill=tk.BOTH, expand=True)

        # Header
        header = ttk.Frame(root)
        header.pack(fill=tk.X, **pad)
        ttk.Label(
            header,
            text="EVSE Edge Attack Detection",
            font=("Segoe UI", 16, "bold"),
        ).pack(anchor=tk.W)
        ttk.Label(
            header,
            text="Jetson Nano demo · Host modality · Pipeline C (LDA-LSTM)",
            font=("Segoe UI", 10),
        ).pack(anchor=tk.W)

        body = ttk.Panedwindow(root, orient=tk.HORIZONTAL)
        body.pack(fill=tk.BOTH, expand=True, **pad)

        # Left: what it detects + controls
        left = ttk.Frame(body, padding=4)
        right = ttk.Frame(body, padding=4)
        body.add(left, weight=1)
        body.add(right, weight=1)

        info = ttk.LabelFrame(left, text="What this model detects", padding=8)
        info.pack(fill=tk.BOTH, expand=True)
        txt = tk.Text(info, height=14, wrap=tk.WORD, font=("Consolas", 9))
        txt.insert("1.0", ATTACK_FAMILIES_TEXT)
        txt.configure(state=tk.DISABLED)
        txt.pack(fill=tk.BOTH, expand=True)

        path_fr = ttk.LabelFrame(left, text="Deploy package", padding=8)
        path_fr.pack(fill=tk.X, pady=(8, 0))
        ttk.Entry(path_fr, textvariable=self.deploy_dir).pack(
            side=tk.LEFT, fill=tk.X, expand=True
        )
        ttk.Button(path_fr, text="Browse…", command=self._browse_dir).pack(
            side=tk.LEFT, padx=(6, 0)
        )

        btns = ttk.Frame(left)
        btns.pack(fill=tk.X, pady=8)
        ttk.Button(btns, text="1. Load artifacts", command=self._load_artifacts).pack(
            side=tk.LEFT, padx=(0, 6)
        )
        ttk.Button(
            btns, text="2. Synthetic demo", command=self._run_synthetic
        ).pack(side=tk.LEFT, padx=(0, 6))
        ttk.Button(
            btns, text="3. Load .npy window", command=self._run_npy
        ).pack(side=tk.LEFT, padx=(0, 6))
        ttk.Button(btns, text="4. Benchmark", command=self._run_benchmark).pack(
            side=tk.LEFT
        )

        # Right: results
        res = ttk.LabelFrame(right, text="Live result", padding=10)
        res.pack(fill=tk.BOTH, expand=True)

        ttk.Label(res, text="Decision", font=("Segoe UI", 10)).pack(anchor=tk.W)
        self.decision_lbl = ttk.Label(
            res,
            textvariable=self.pred_label,
            font=("Segoe UI", 28, "bold"),
        )
        self.decision_lbl.pack(anchor=tk.W, pady=(0, 10))

        ttk.Label(res, text="P(Benign)").pack(anchor=tk.W)
        self.bar_benign = ttk.Progressbar(res, maximum=100, length=360)
        self.bar_benign.pack(fill=tk.X, pady=2)
        ttk.Label(res, textvariable=self.prob_benign).pack(anchor=tk.W)

        ttk.Label(res, text="P(Attack)").pack(anchor=tk.W, pady=(8, 0))
        self.bar_attack = ttk.Progressbar(res, maximum=100, length=360)
        self.bar_attack.pack(fill=tk.X, pady=2)
        ttk.Label(res, textvariable=self.prob_attack).pack(anchor=tk.W)

        meta_fr = ttk.Frame(res)
        meta_fr.pack(fill=tk.X, pady=16)
        ttk.Label(meta_fr, text="Latency:").grid(row=0, column=0, sticky=tk.W)
        ttk.Label(meta_fr, textvariable=self.latency).grid(
            row=0, column=1, sticky=tk.W, padx=8
        )
        ttk.Label(meta_fr, text="Window / LDA:").grid(
            row=1, column=0, sticky=tk.W
        )
        ttk.Label(meta_fr, textvariable=self.shape_info).grid(
            row=1, column=1, sticky=tk.W, padx=8
        )
        ttk.Label(meta_fr, text="Model:").grid(row=2, column=0, sticky=tk.W)
        ttk.Label(meta_fr, textvariable=self.meta_info).grid(
            row=2, column=1, sticky=tk.W, padx=8
        )

        # Status bar
        status = ttk.Frame(root)
        status.pack(fill=tk.X, **pad)
        ttk.Label(status, textvariable=self.status).pack(anchor=tk.W)

        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass

    def _browse_dir(self):
        path = filedialog.askdirectory(
            title="Select pipeline_C_lda_lstm folder",
            initialdir=self.deploy_dir.get(),
        )
        if path:
            self.deploy_dir.set(path)

    def _set_busy(self, busy: bool, msg: str | None = None):
        self._busy = busy
        if msg:
            self.status.set(msg)

    def _load_artifacts(self):
        if self._busy:
            return

        def work():
            try:
                self._set_busy(True, "Loading scaler / LDA / ONNX meta…")
                meta, _, _, feats, onnx_path = load_artifacts(
                    self.deploy_dir.get()
                )
                self._meta = meta
                size_mb = (
                    os.path.getsize(onnx_path) / (1024 * 1024)
                    if os.path.isfile(onnx_path)
                    else 0.0
                )
                self.meta_info.set(
                    f"{meta.get('pipeline', 'C')}  |  "
                    f"raw {meta['n_features_raw']} → LDA {meta['n_features_lda']}  |  "
                    f"T={meta['n_timesteps']}  |  ONNX {size_mb:.2f} MB  |  "
                    f"{len(feats)} feature names"
                )
                self.status.set("Artifacts loaded — ready to infer")
            except Exception as exc:
                self.status.set("Load failed")
                messagebox.showerror("Load artifacts", str(exc))
            finally:
                self._set_busy(False)

        threading.Thread(target=work, daemon=True).start()

    def _apply_result(self, result, source: str):
        pred = int(result["preds"][0])
        probs = result["probs"][0]
        label = result["meta"]["label_map"][str(pred)]
        self.pred_label.set(label.upper())
        self.prob_benign.set(f"{probs[0] * 100:.2f}%")
        self.prob_attack.set(f"{probs[1] * 100:.2f}%")
        self.bar_benign["value"] = float(probs[0] * 100)
        self.bar_attack["value"] = float(probs[1] * 100)

        if label.lower() == "attack":
            self.decision_lbl.configure(foreground="#b00020")
        else:
            self.decision_lbl.configure(foreground="#0b6e4f")

        if result["latency_ms"] is not None:
            self.latency.set(
                f"{result['latency_ms']:.3f} ms / window"
            )
        else:
            self.latency.set("n/a (enable Benchmark for timing)")

        t = result["x_lda_shape"][1]
        d = result["x_lda_shape"][2]
        self.shape_info.set(f"LDA window ({t}, {d})")
        self.status.set(f"Done — {source}: {label}")

    def _run_synthetic(self):
        if self._busy:
            return

        def work():
            try:
                self._set_busy(True, "Running synthetic window…")
                meta, _, _, _, _ = load_artifacts(self.deploy_dir.get())
                self._meta = meta
                t = int(meta["n_timesteps"])
                f = int(meta["n_features_raw"])
                rng = np.random.default_rng(int(time.time()) % 10_000)
                raw = rng.normal(size=(t, f)).astype(np.float32)
                result = predict(raw, self.deploy_dir.get(), benchmark_runs=0)
                self.after(0, lambda: self._apply_result(result, "synthetic"))
            except Exception as exc:
                self.after(
                    0,
                    lambda: messagebox.showerror("Synthetic demo", str(exc)),
                )
                self.status.set("Inference failed")
            finally:
                self._set_busy(False)

        threading.Thread(target=work, daemon=True).start()

    def _run_npy(self):
        if self._busy:
            return
        path = filedialog.askopenfilename(
            title="Select window .npy (T,F) or (B,T,F)",
            filetypes=[("NumPy", "*.npy"), ("All", "*.*")],
        )
        if not path:
            return

        def work():
            try:
                self._set_busy(True, f"Inferring {os.path.basename(path)}…")
                raw = np.load(path)
                result = predict(raw, self.deploy_dir.get(), benchmark_runs=0)
                self.after(
                    0,
                    lambda: self._apply_result(
                        result, os.path.basename(path)
                    ),
                )
            except Exception as exc:
                self.after(
                    0,
                    lambda: messagebox.showerror("Load .npy", str(exc)),
                )
                self.status.set("Inference failed")
            finally:
                self._set_busy(False)

        threading.Thread(target=work, daemon=True).start()

    def _run_benchmark(self):
        if self._busy:
            return

        def work():
            try:
                self._set_busy(True, "Benchmarking ONNX (warmup + 100 runs)…")
                meta, _, _, _, _ = load_artifacts(self.deploy_dir.get())
                t = int(meta["n_timesteps"])
                f = int(meta["n_features_raw"])
                raw = np.random.default_rng(0).normal(
                    size=(t, f)
                ).astype(np.float32)
                result = predict(
                    raw,
                    self.deploy_dir.get(),
                    benchmark_runs=100,
                )
                self.after(
                    0,
                    lambda: self._apply_result(result, "benchmark"),
                )
                thr = 1000.0 / max(result["latency_ms"], 1e-6)
                self.after(
                    0,
                    lambda: self.status.set(
                        f"Benchmark done — ~{thr:.1f} windows/s "
                        f"({result['latency_ms']:.3f} ms)"
                    ),
                )
            except Exception as exc:
                self.after(
                    0,
                    lambda: messagebox.showerror("Benchmark", str(exc)),
                )
                self.status.set("Benchmark failed")
            finally:
                self._set_busy(False)

        threading.Thread(target=work, daemon=True).start()


def main():
    parser = argparse.ArgumentParser(
        description="EVSE Pipeline C demo GUI"
    )
    parser.add_argument(
        "--deploy-dir",
        default=default_deploy_dir(),
        help="Path to jetson_deploy/pipeline_C_lda_lstm",
    )
    args = parser.parse_args()

    if not os.path.isdir(args.deploy_dir):
        raise SystemExit(
            f"Deploy dir not found: {args.deploy_dir}"
        )

    app = EvseIdsGui(args.deploy_dir)
    app.mainloop()


if __name__ == "__main__":
    main()
