# Robustness Fix Guide — EVSE Attack Detection Thesis

This guide closes the evaluation gaps in the current repo. Every fix comes with
a script in `fixes/`, the exact command to run it, the result it gave on the data
already in the repo, and what to write in the thesis.

> All scripts run from the **repo root**, read `09_processed_datasets/`, and
> write only to `fixes_out/`. Nothing original is overwritten.
>
> ```bash
> pip install -r requirements.txt
> python fixes/01_leakage_audit.py
> ```

| # | Gap | Severity | Script / change | Time |
|---|-----|----------|-----------------|------|
| 1 | Network timestamps leak the label | 🔴 critical | `fixes/02_network_drop_leaky.py` + patch in `prepare_network_sequences.py` | 1 h |
| 2 | Network test has only 7 benign windows | 🔴 critical | §2 (re-split, report FPR CI) | 2–4 h |
| 3 | No classical baselines | 🔴 critical | `fixes/03_classical_baselines.py` | 30 min |
| 4 | Single-session benign → test = same recording | 🔴 critical | `fixes/04_generalisation_tests.py` | 30 min |
| 5 | One seed only | 🟠 major | `fixes/05_multiseed_pipelines.py` | 1–3 h (CPU) |
| 6 | No per-family results | 🟠 major | built into scripts 03/04/05 | – |
| 7 | Stride-1 window overlap | 🟠 major | `--non-overlap` in 05, part (c) of 04 | – |
| 8 | `tcp-flood` labelled Recon | 🟡 minor | patch in `hosteventspreprocessing.py` | 10 min |
| 9 | LDA on power does not reduce (4→4) | 🟡 wording | thesis text only | – |
| 10 | No on-device numbers | 🟠 major | `fixes/06_edge_benchmark.py` on Jetson + RPi4 | 2 h |
| 11 | Reproducibility (paths, deps, raw data) | 🟡 minor | `requirements.txt`, §11 | 1 h |

---

## 1. Network: remove leaky columns

**Problem.** A decision stump on `dst2src_first_seen_ms` alone gets 100% test
accuracy (14 single features reach 100%). Absolute timestamps tell the model
which capture file a flow came from, and each file is one label.

**Fix (already applied on branch `robustness-fixes`).** Add the columns to
`METADATA` in `03_sequence_preparation/prepare_network_sequences.py`. They stay
in the dataframe for sorting but no longer go into the model:

```python
METADATA = [ ...,
    "bidirectional_first_seen_ms", "bidirectional_last_seen_ms",
    "src2dst_first_seen_ms", "src2dst_last_seen_ms",
    "dst2src_first_seen_ms", "dst2src_last_seen_ms",
    "src_port", "dst_port", "vlan_id", "tunnel_id", "ip_version",
    "application_is_guessed", "application_confidence",
]
```

Then rerun `prepare_network_sequences.py` → `prepare_network_dr_sequences.py`
→ `train_network_pipelines_A_to_E.py`.

**Quick path without raw data.** `python fixes/02_network_drop_leaky.py` drops
the columns from the existing windows (identical result, since StandardScaler
works per column) and re-fits PCA/LDA → `fixes_out/network_noleak/`.

**Result.** 71 → 58 features; PCA 14 comps, LDA 2 comps. Scores are **still
~100%**, but the remaining 1.0 features are now behavioural (`dst2src_packets`,
`*_ack_packets`, packet-size stats: floods and scans get no or abnormal replies).
That is a legitimate signal. Only the timestamp version was invalid.

**Thesis text.**
> "Absolute flow timestamps and port/tunnel identifiers were excluded from the
> model input because a leakage audit (single-feature decision stumps) showed
> that they identify the capture file and thus the label. After exclusion, the
> strongest single features are reply-direction packet counts, which reflect
> attack behaviour."

Always include the audit (`fixes_out/leakage_audit_*.csv`) as an appendix table.
It is the evidence a reviewer looks for.

---

## 2. Network: too few benign windows

**Problem.** Test = 7 benign vs 924 attack, all 7 from `EVSE-A-idle-benign.csv`.
Benign *charging* (5 windows) appears only in train. A FPR of 0/7 has a 95% CI
of **[0%, 35%]** (Wilson). No claim about false alarms is possible.

**Fixes, best first:**

1. **Flow-level evaluation (window = 1).** Network flows are already aggregated
   records, so windows of 5 consecutive flows add little. Use `WINDOW_SIZE = 1`
   in `prepare_network_sequences.py`. That gives ~5× more benign samples.
2. **Chronological split of EACH benign file separately**, so both idle and
   charging benign show up in val and test. In the "rare group" branch of the
   split, the code already splits per session; with window 1 every session has
   enough rows.
3. **Leave-one-file-out (GroupKFold over `source_file`)** for attack files, with
   benign flows split by time. Report mean ± std over folds.
4. **Lower `MAX_ATTACK_PER_GROUP`** (e.g. 4000 → 500) so the class ratio is not
   130:1. Report balanced accuracy.
5. **Always report FPR with a 95% CI and the benign count.**
   `fx_common.binary_report()` does this (`fpr_ci95_low/high`, `n_benign_test`).

If benign stays this small, present network as a **supplementary** modality
and say so: "the EVSE-A benign capture contains only N flows, so network FPR
cannot be estimated reliably."

---

## 3. Classical baselines

`python fixes/03_classical_baselines.py` → `fixes_out/classical_baselines.csv`

Models: LDA used as the classifier, Logistic Regression, Random Forest,
HistGradientBoosting. Feature spaces: raw / PCA / LDA, all fitted on train only.

**Results (test set, same windows as A–E):**

| Modality | Best classical | F1 | FPR [95% CI] | Best deep (current repo) | F1 |
|---|---|---|---|---|---|
| Host | LogReg (raw 220) | **1.000** | 0.000 [0, 0.012] | all A–E | 1.000 |
| Host | LDA-clf on LDA(3) | **1.000** | 0.000 | C LDA-LSTM | 1.000 |
| Power | **HGB (raw 4)** | **0.996** | **0.021** [0.014, 0.033] | A LSTM | 0.990 (FPR 0.045) |
| Network (no leak) | LogReg / RF | 1.000 | 0.000 [0, **0.354**] | – | – |

**What this means.**
- On host, *any* model is perfect, so the deep models don't win on accuracy.
- On power, gradient boosting **beats every deep pipeline** and halves the FPR.

**How to frame it (this makes the thesis stronger):**
> "Detection quality saturates for linear and non-linear models alike on the
> host modality; the contribution of LDA is therefore efficiency: 220 → 3 input
> features and a 1.4k-parameter LSTM with no loss in F1. On the low-dimensional
> power modality, tree ensembles outperform the temporal models, indicating that
> temporal modelling is not the limiting factor there."

Add the classical rows to **every** results table, next to A–E.

---

## 4. Generalisation tests (the most important new chapter)

`python fixes/04_generalisation_tests.py` → `fixes_out/generalisation_tests.csv`

Three tests, each re-fitting LDA on its own training set:

### (a) Cross-condition: train on idle → test on charging (and reverse)

| Host | F1 | FPR | Recon | DoS | Crypto |
|---|---|---|---|---|---|
| LDA-LSTM idle→charging | **0.64** | 0.00 | 0.92 | 0.98 | **0.00** |
| LDA-LSTM charging→idle | **0.65** | 0.00 | **0.02** | **0.01** | 0.99 |
| HGB idle→charging | 0.80 | **1.00** | 1.0 | 1.0 | 1.0 |

| Power | F1 | FPR |
|---|---|---|
| all models idle→charging | 0.85 | **~1.00** (charging current looks like an attack) |
| all models charging→idle | 0.41 | 0.00 (Recon 0%, Backdoor ~2%) |

→ **The 100% does not carry over to an unseen operating state.** The models learn
"what this state looks like", not just the attack. This is the main honest
finding, and a real contribution: *train on both states, or normalise per state*.

**Follow-up worth doing:** per-state normalisation (fit one scaler per `State`,
or add `State` as an input feature) and rerun test (a).

### (b) Leave-one-family-out (closest to zero-day)

| Unseen family | Host LDA-LSTM | Host HGB | Power LDA-LSTM | Power HGB |
|---|---|---|---|---|
| Recon | 0.65 | 0.82 | 0.96 | 1.00 |
| DoS | 0.36 | 0.96 | 0.99 | 1.00 |
| Cryptojacking | **1.00** | 0.90 | 1.00 | 1.00 |
| Backdoor | – | – | **0.12** | 0.34 |

(values = recall on the never-seen family; benign FPR stays ≤ 7%)

→ Cryptojacking on host and most power attacks are detected without being trained
on. DoS (host) and Backdoor (power) are not. Report this table as is.

### (c) Non-overlapping windows (stride = window)

| | Host F1 / FPR | Power F1 / FPR |
|---|---|---|
| LDA-LSTM | 1.000 / 0.000 | **0.939 / 0.244** |
| HGB | 1.000 / 0.000 | 0.989 / 0.067 |

→ Host is unaffected (overlap is not why host is 100%). On power, LSTM FPR rises
from ~9% to 24% with 10× fewer and independent windows. Stride-1 overlap was
inflating the power results.

---

## 5. Multiple seeds

```bash
python fixes/05_multiseed_pipelines.py --modality host
python fixes/05_multiseed_pipelines.py --modality power          # ~40 min CPU, GPU faster
python fixes/05_multiseed_pipelines.py --modality power --non-overlap
python fixes/05_multiseed_pipelines.py --modality network_noleak
```

Outputs `*_runs.csv` and `*_summary.csv` (mean ± std, per-family recall, FPR,
train time, latency, Wilcoxon p-value vs Pipeline C).

**Host result (5 seeds):** every pipeline F1 = 1.0000 ± 0.0000, so the
saturation is stable, not luck. Train time per run: C 1.43 ± 0.09 s vs E 4.92 ± 0.43 s.

**Power (to run):** the current single-seed gap C 0.983 vs E 0.990 must be
shown with ± std. If the Wilcoxon p > 0.05, write "no significant difference".

Table format for the thesis:

| Pipeline | Feat. | Params | F1 (mean ± std) | FPR (mean ± std) | Train s |
|---|---|---|---|---|---|

---

## 6. Per-family results

All three scripts already output `recall_Recon`, `recall_DoS`,
`recall_Cryptojacking`, `recall_Backdoor`, and `tnr_Benign`. Add one table per
modality. Binary F1 alone hides that, e.g., Backdoor is the weak family.

---

## 7. Window overlap

Covered by 4(c) and `--non-overlap` in script 05. In the methodology, state:
"Overlapping windows (stride 1) were used for training; robustness was verified
with non-overlapping windows (stride = w)."

---

## 8. Label bug: idle `tcp-flood` labelled as Recon

The raw host file has `Scenario = Recon` for the idle tcp-flood run.
`map_attack_group()` used Scenario first, so this DoS run became Recon (it sits
in the **val** split). **Fixed** in `02_preprocessing/hosteventspreprocessing.py`:
the Attack name is now checked first, Scenario only as a fallback. Rerun host
preprocessing → sequences → DR → training.

---

## 9. LDA on power is not a reduction

4 raw features, 5 classes → LDA gives min(C−1, d) = 4 components: **4 → 4**.
PCA keeps 1 component. Rewrite as:
> "For the 4-dimensional power modality, LDA acts as a supervised linear
> re-projection rather than a reduction (4 → 4); the dimensionality-reduction
> argument applies to the host (220 → 3) and network (58 → 2) modalities."

---

## 10. On-device measurement (Jetson Nano + Raspberry Pi 4)

Copy `11_jetson_nano_demo/pipeline_C_lda_lstm/`,
`13_rpi4_algorithm_dashboard/host_lda_lstm/test_windows_raw.npz` and
`fixes/06_edge_benchmark.py` to each device:

```bash
pip install numpy onnxruntime psutil
# terminal 1 (Jetson power/thermal log)
sudo nvpmodel -m 0 && sudo jetson_clocks
sudo tegrastats --interval 500 --logfile tegrastats.log
# terminal 2
python 06_edge_benchmark.py --model-dir pipeline_C_lda_lstm \
   --raw-windows test_windows_raw.npz --device-name "Jetson Nano 4GB MAXN"
```

RPi4: same command, with `--device-name "RPi4 4GB"`. Use a USB power meter for watts.

The script measures the **end-to-end** path (raw 220 counters → scaler → LDA →
ONNX LSTM), plus cold start, p50/p95/p99 latency, throughput at batch 1/32/256,
RAM, CPU %, and checks accuracy on-device. Reference run on a cloud x86 CPU:
median **0.028 ms**, p99 0.069 ms, ~32k windows/s, 7 KB model, accuracy 1.0.

Edge results table:

| Device | Latency p50 / p99 (ms) | Throughput (win/s) | RAM (MB) | Power (W) | Acc |
|---|---|---|---|---|---|

Also benchmark one classical model (LogReg on LDA features, ~16 parameters) on
the same devices. If it is faster with the same accuracy, say so honestly. The
LSTM can still be argued for because of temporal context (see 4(b), where
LDA-LSTM detects unseen Cryptojacking 100% and LogReg 0%).

---

## 11. Reproducibility

- `requirements.txt` added at repo root.
- Training scripts use paths like `processed_EVSE_B_Host/...` and `models/...`.
  Either run them from inside `09_processed_datasets/`, or (better) add a
  `config.py` with `REPO = Path(__file__).resolve().parents[1]` and build all
  paths from it.
- `all_pipeline_results.csv` stores Windows paths (`models\...`). Store them
  with `Path(...).as_posix()`.
- `10_raw_data/` is referenced in the README but missing: add at least
  `README_DATA.md` with the CICEVSE2024 download link, file names and checksums.
- Write down the α, β, γ of the selection score (Phase G) and the cost
  normalisation. Otherwise "multi-criteria selection" can't be reproduced.

---

## Suggested new thesis structure for results

1. Main comparison A–E **+ classical baselines**, mean ± std over 5 seeds
2. Per-family recall table
3. Leakage audit (appendix)
4. **Robustness:** cross-condition, leave-one-family-out, non-overlapping windows
5. Edge benchmark on Jetson Nano and RPi4
6. Discussion: accuracy saturates in-distribution; generalisation across
   operating states is the open problem; LDA's value is efficiency

## Run everything

```bash
pip install -r requirements.txt
python fixes/01_leakage_audit.py
python fixes/02_network_drop_leaky.py
python fixes/01_leakage_audit.py --seq-dir fixes_out/network_noleak
python fixes/03_classical_baselines.py
python fixes/04_generalisation_tests.py
for m in host power network_noleak; do python fixes/05_multiseed_pipelines.py --modality $m; done
python fixes/05_multiseed_pipelines.py --modality power --non-overlap
# on each edge device:
python fixes/06_edge_benchmark.py --device-name "<device>"
```
