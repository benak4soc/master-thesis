# Master’s Thesis Experimental Reference Guide

**Title:** A Supervised Dimensionality-Reduction and Temporal Deep-Learning Framework for Attack Detection in EV Charging Infrastructure  

**Dataset:** CICEVSE2024 (Canadian Institute for Cybersecurity, University of New Brunswick)  
**Student focus:** EVSE attack detection using host HPC/kernel events, network traffic, and power consumption, with edge-oriented model selection  

This document explains **every major step** of the implemented work, maps results to **research gaps, objectives and questions**, and collects **tables + figures** for thesis writing.

Figures are stored in: `thesis_figures/`  
Key result JSON files are stored under: `models/`

---

## Table of contents

1. [Thesis aim in one paragraph](#1-thesis-aim-in-one-paragraph)
2. [End-to-end workflow (what was done)](#2-end-to-end-workflow-what-was-done)
3. [Datasets and preprocessing (step-by-step)](#3-datasets-and-preprocessing-step-by-step)
4. [Temporal windows, split strategy and scaling](#4-temporal-windows-split-strategy-and-scaling)
5. [Dimensionality reduction (PCA vs LDA)](#5-dimensionality-reduction-pca-vs-lda)
6. [Proposed pipelines A–E](#6-proposed-pipelines-ae)
7. [Hyperparameter tables](#7-hyperparameter-tables)
8. [Binary detection results (all modalities)](#8-binary-detection-results-all-modalities)
9. [Confusion matrices (binary)](#9-confusion-matrices-binary)
10. [Multi-class / attack-category results](#10-multi-class--attack-category-results)
11. [Training & validation loss and ROC plots](#11-training--validation-loss-and-roc-plots)
12. [Edge selection and Jetson packaging](#12-edge-selection-and-jetson-packaging)
13. [How research gaps are fulfilled](#13-how-research-gaps-are-fulfilled)
14. [How research objectives are met](#14-how-research-objectives-are-met)
15. [How research questions are answered](#15-how-research-questions-are-answered)
16. [Limitations and interpretation](#16-limitations-and-interpretation)
17. [Final conclusion](#17-final-conclusion)
18. [File map (where everything lives)](#18-file-map-where-everything-lives)

---

## 1. Thesis aim in one paragraph

EV charging stations (EVSE) produce high-dimensional cyber and cyber-physical telemetry. Attacks can unfold over time, so a single snapshot is often insufficient. This thesis therefore compares **five temporal deep-learning pipelines**, with and without **PCA/LDA**, across **three CICEVSE2024 modalities** (host HPC/kernel, network traffic, power). The goal is not only high detection scores, but a model that is **accurate enough and small/fast enough** for **edge deployment** (NVIDIA Jetson Nano).

---

## 2. End-to-end workflow (what was done)

```text
CICEVSE2024 raw CSVs
        │
        ▼
 Preprocessing (clean labels, features, Attack_Group)
        │
        ▼
 Session-aware / chronological split
        │
        ▼
 Train-only StandardScaler
        │
        ▼
 Sliding temporal windows
        │
        ├── Pipeline A: LSTM (no DR)
        ├── Pipeline B: PCA → LSTM
        ├── Pipeline C: multi-class LDA → LSTM
        ├── Pipeline D: multi-class LDA → BiLSTM
        └── Pipeline E: CNN-BiLSTM + Attention (no DR)
        │
        ▼
 Compare Acc / Prec / Rec / F1 / ROC-AUC / train time / params
        │
        ▼
 Select best efficiency trade-off (mainly host Pipeline C)
        │
        ▼
 Hyperparameter tuning → ONNX export → Jetson package
```

### Why this workflow matches the proposal

| Proposal stage | What it means | What was implemented |
|---|---|---|
| Preprocessing | Remove leakage/noise; define labels | Host / network / power cleaners |
| Dimensionality reduction | PCA & LDA on train only | Variance PCA + Attack_Group LDA |
| Temporal modelling | Sequences for LSTM-family models | Sliding windows |
| Controlled comparison | Same split/metrics for A–E | Shared trainer `lstm_common.py` |
| Edge focus | Deploy practical model | Tuned LDA-LSTM + ONNX package |

---

## 3. Datasets and preprocessing (step-by-step)

### 3.1 Three modalities used

| Modality | Source folder | Clean output | Main role in thesis |
|---|---|---|---|
| **Host HPC/kernel** | `host events/` | `processed_EVSE_B_Host/` | **Primary** high-dimensional DR study (~220 features) |
| **Network traffic** | `network traffic/EVSE A/` | `processed_EVSE_A/` | Network-level Recon/DoS view |
| **Power consumption** | `power consumption/` | `processed_EVSE_B_Power/` | Cyber-physical view; **includes Backdoor** |

### 3.2 Host preprocessing (`hosteventspreprocessing.py`)

**Problem in raw file:** mixed/invalid labels (`0`, `NaN`, repeated header rows), ~900 columns with many zero-variance counters.

**Steps:**
1. Drop `Unnamed:*` artifact columns and garbage rows  
2. Keep only `Label ∈ {benign, attack}`  
3. Derive `Attack_Group` from `Scenario` / `Attack`  
4. Create:
   - `target` (binary: 0/1)
   - `target_multiclass` (Benign/Recon/DoS/Cryptojacking/Backdoor)
5. Convert HPC features to numeric; remove zero-variance features  
6. Save clean CSV + attack-family subset  

**Result:**
- Rows: **6,166**
- Features: **220**
- Labels: Benign 2,302 / Attack 3,864  
- Groups present: Benign, Recon, DoS, Cryptojacking (**Backdoor = 0** in host file)

### 3.3 Network preprocessing (`nwtrafficpreprocessing.py`)

**Problem:** NFStream CSVs had **no Label column**; labels taken from filenames.

**Steps:**
1. Read all EVSE-A charging/idle CSVs  
2. Derive Label/Attack from filename (`benign` vs attack names)  
3. Remove IDs/IPs/MACs (leakage / non-generalizable identifiers)  
4. Keep numeric flow features; binary target  

**Result:**
- Rows: **547,854**
- Features: **71**
- Extreme imbalance: Benign **82** / Attack **547,772**

For training A–E, attacks were **undersampled** (all benign kept; 4,000 Recon + 4,000 DoS) because full attack volume is unnecessary for a controlled comparison and would dominate training.

### 3.4 Power preprocessing (`powerpreprocessing.py`)

**Steps:**
1. Normalize Label / Attack / Attack-Group / State  
2. Map power `Attack-Group` + `Attack` into thesis families  
   - especially `host-attack` → **Backdoor** or **Cryptojacking** via Attack name  
3. Create binary + multi-class targets  
4. Keep 4 numeric sensors  

**Result:**
- Clean rows: **114,198**
- Features: `shunt_voltage`, `bus_voltage_V`, `current_mA`, `power_mW`
- Full 5-class family set including **Backdoor (21,090)**

### 3.5 Attack_Group definition (shared thesis taxonomy)

| Attack_Group | Typical examples | Present in |
|---|---|---|
| Benign | normal idle/charging | Host, Network, Power |
| Recon | portscan, vuln-scan, syn-stealth, fingerprinting | Host, Network, Power |
| DoS | syn/tcp/udp/icmp floods | Host, Network, Power |
| Cryptojacking | cryptojacking | Host, Power |
| Backdoor | backdoor | **Power only** |

This directly supports Gap 3 (attack scope beyond DDoS-only).

---

## 4. Temporal windows, split strategy and scaling

### 4.1 Why windows?

LSTM/BiLSTM/CNN-BiLSTM need sequences:

\[
(x_{t-w+1},\ldots,x_t) \rightarrow y_t
\]

where \(y_t\) is the label of the **last** timestep in the window (common IDS practice).

| Modality | Window size | Stride | Reason |
|---|---:|---:|---|
| Host | 10 | 1 | Enough temporal context; sessions are long enough |
| Network | 5 | 1 | Benign files are tiny (14 and 68 rows) |
| Power | 10 | 1 | Dense sampling; long sessions |

### 4.2 Leakage-aware split (important)

A random row split would leak near-duplicate samples from the same attack session into train and test.

**Used strategy:**
- Prefer **session-aware** split when enough sessions exist  
- For rare groups (≤3 sessions): **chronological** within-session 70/15/15 split  

### 4.3 Train-only scaling / DR fitting

To avoid information leakage:
1. Fit `StandardScaler` on **train rows only**
2. Fit PCA / LDA on **train timesteps only**
3. Transform val/test with frozen train parameters  

This follows thesis methodology §11.

---

## 5. Dimensionality reduction (PCA vs LDA)

### 5.1 PCA (unsupervised)

- Goal: keep components explaining ~**95%** variance  
- Does **not** use class labels  

| Modality | Input features | PCA components |
|---|---:|---:|
| Host | 220 | **26** |
| Network | 71 | **17** |
| Power | 4 | **1** |

### 5.2 Multi-class LDA (supervised; plan B)

- Uses `Attack_Group` labels  
- Maximum components = \(C-1\)  
- Then LSTM still performs **binary** Benign vs Attack (primary IDS task)  
- Separate **multi-class LSTM** was also trained as a supplementary experiment for attack-category analysis  

| Modality | Classes in LDA fit | LDA components |
|---|---|---:|
| Host | Benign, Recon, DoS, Cryptojacking | **3** |
| Network | Benign, Recon, DoS | **2** |
| Power | Benign, Recon, DoS, Cryptojacking, Backdoor | **4** |

### 5.3 Binary LDA (limitation discussion)

Classical binary LDA yields at most **1** component. It was exported for discussion, but multi-class LDA is the main supervised DR contribution.

---

## 6. Proposed pipelines A–E

| Pipeline | Dimensionality reduction | Model | Purpose |
|---|---|---|---|
| **A** | None | LSTM | Baseline temporal model |
| **B** | PCA | LSTM | Unsupervised compression + LSTM |
| **C** | Multi-class LDA | LSTM | Supervised compact features + LSTM |
| **D** | Multi-class LDA | BiLSTM | Does bidirectional context help? |
| **E** | None | CNN-BiLSTM + Attention | Most complex temporal architecture |

### Architecture intuition (short)

- **LSTM:** remembers temporal dependencies in one direction  
- **BiLSTM:** forward + backward context inside each window  
- **CNN-BiLSTM+Attention:** CNN finds local patterns; BiLSTM models sequence; attention weights informative timesteps  

All pipelines share the same split, metrics and early-stopping protocol where applicable (`lstm_common.py`).

---

## 7. Hyperparameter tables

### 7.1 Common training hyperparameters (Pipelines A–E default)

| Hyperparameter | Value | Meaning |
|---|---|---|
| Optimizer | Adam | Adaptive gradient method |
| Learning rate | \(1\times10^{-3}\) | Step size |
| Weight decay | \(1\times10^{-5}\) | Mild L2 regularization |
| Batch size | 64 | Sequences per update |
| Max epochs | 40 | Upper training limit |
| Early stopping patience | 8 | Stop if val F1 does not improve |
| Hidden size | 64 | LSTM units (per direction for BiLSTM) |
| LSTM layers | 1 | Depth |
| Dropout | 0.2 | Regularization |
| Loss | Weighted Cross-Entropy | Handles class imbalance |
| Selection metric | Validation F1 | Security-relevant balance of P/R |

### 7.2 Pipeline-specific settings

| Pipeline | Extra settings |
|---|---|
| A | Full features → LSTM |
| B | PCA 95% variance → LSTM |
| C | Multi-class LDA → LSTM |
| D | Same LDA features as C → BiLSTM |
| E | Conv channels 64 (host/network) / 32 (power), kernel 3 → BiLSTM → attention |

### 7.3 Tuned edge model (Host Pipeline C selected for Jetson)

After 72-config search on host LDA-LSTM:

| Hyperparameter | Default C | Tuned C (edge) |
|---|---:|---:|
| Hidden size | 64 | **16** |
| Layers | 1 | 1 |
| Dropout | 0.2 | 0.2 |
| Learning rate | 0.001 | 0.001 |
| Batch size | 64 | 64 |
| Parameters | 17,794 | **1,378** |
| ONNX size | — | **~7.0 KB** |
| PyTorch latency (PC CPU) | — | **~0.40 ms/window** |

---

## 8. Binary detection results (all modalities)

Primary task: **Benign (0) vs Attack (1)**.

### 8.1 Host HPC/kernel (main high-dimensional modality)

| Pipeline | Features | Params | Accuracy | Precision | Recall | F1 | ROC-AUC | Train (s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A LSTM | 220 | 73,346 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 9.59 |
| B PCA-LSTM | 26 | 23,682 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 6.64 |
| **C LDA-LSTM** | **3** | **17,794** | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | **3.52** |
| D LDA-BiLSTM | 3 | 35,586 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 6.50 |
| E CNN-BiLSTM+Attn | 220 | 125,762 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 11.43 |

**Interpretation:** scores are saturated; ranking is by **efficiency**. Host **C** is the best compact candidate.

### 8.2 Network EVSE-A

| Pipeline | Features | Params | Accuracy | Precision | Recall | F1 | ROC-AUC | Train (s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A LSTM | 71 | 35,202 | 0.9979 | 1.0000 | 0.9978 | 0.9989 | 1.0000 | 5.26 |
| B PCA-LSTM | 17 | 21,378 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 4.59 |
| **C LDA-LSTM** | **2** | **17,538** | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | **4.48** |
| D LDA-BiLSTM | 2 | 35,074 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 7.08 |
| E CNN-BiLSTM+Attn | 71 | 97,154 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 10.77 |

**Caution:** only **7 benign test windows** after undersampling/windowing; metrics are optimistic and must be discussed carefully.

### 8.3 Power EVSE-B (most realistic numeric separation)

| Pipeline | Features | Params | Accuracy | Precision | Recall | F1 | ROC-AUC | Train (s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A LSTM | 4 | 18,050 | 0.9835 | 0.9887 | 0.9907 | 0.9897 | 0.9981 | 95.71 |
| **B PCA-LSTM** | **1** | **17,282** | **0.9853** | **0.9915** | 0.9901 | **0.9908** | 0.9984 | 99.13 |
| C LDA-LSTM | 4 | 18,050 | 0.9724 | 0.9784 | 0.9873 | 0.9828 | 0.9918 | **73.58** |
| D LDA-BiLSTM | 4 | 36,098 | 0.9778 | 0.9837 | 0.9887 | 0.9862 | 0.9894 | 92.64 |
| E CNN-BiLSTM+Attn | 4 | 67,490 | 0.9837 | 0.9906 | 0.9890 | 0.9898 | **0.9987** | 109.14 |

**Interpretation:** with only 4 sensors, PCA (1-D) works very well; complex E gives tiny AUC gain at much higher cost. This is a strong thesis discussion point: **DR + simple temporal model can beat complexity**.

---

## 9. Confusion matrices (binary)

Format: \(\begin{bmatrix} TN & FP \\ FN & TP \end{bmatrix}\)

### 9.1 Host (all pipelines identical on this split)

| | Predicted Benign | Predicted Attack |
|---|---:|---:|
| **Actual Benign** | 330 | 0 |
| **Actual Attack** | 0 | 513 |

Figure example: `thesis_figures/cm_binary_host_C.png`

### 9.2 Network Pipeline A (only non-perfect binary case among defaults)

| | Predicted Benign | Predicted Attack |
|---|---:|---:|
| **Actual Benign** | 7 | 0 |
| **Actual Attack** | 2 | 922 |

Figure: `thesis_figures/cm_binary_network_A.png`

### 9.3 Power Pipeline B (best F1 on power)

| | Predicted Benign | Predicted Attack |
|---|---:|---:|
| **Actual Benign** | 854 | 30 |
| **Actual Attack** | 35 | 3499 |

Figure: `thesis_figures/cm_binary_power_B.png`

Additional binary CMs were generated for key pipelines under `thesis_figures/cm_binary_*.png`.

---

## 10. Multi-class / attack-category results

Primary pipelines answer **binary IDS**.  
For thesis Gap 3 / plan B, a **supplementary multi-class LSTM** was trained on **LDA features** to predict Attack_Group.

### 10.1 Host multi-class (Benign / Recon / DoS / Cryptojacking)

| Metric | Value |
|---|---:|
| Accuracy | 1.000 |
| Precision (macro) | 1.000 |
| Recall (macro) | 1.000 |
| F1 (macro) | 1.000 |
| Macro-OVR ROC-AUC | 1.000 |

**Confusion matrix (rows = true, cols = predicted):**

|  | Benign | Recon | DoS | Cryptojacking |
|---|---:|---:|---:|---:|
| Benign | 330 | 0 | 0 | 0 |
| Recon | 0 | 157 | 0 | 0 |
| DoS | 0 | 0 | 103 | 0 |
| Cryptojacking | 0 | 0 | 0 | 253 |

Figures:
- `thesis_figures/cm_multiclass_host.png`
- `thesis_figures/roc_multiclass_host.png`
- `thesis_figures/loss_multiclass_host.png`

> Note: host has **no Backdoor** samples.

### 10.2 Network multi-class (Benign / Recon / DoS)

| Metric | Value |
|---|---:|
| Accuracy | 1.000 |
| F1 macro | 1.000 |
| Macro-OVR AUC | 1.000 |

|  | Benign | Recon | DoS |
|---|---:|---:|---:|
| Benign | 7 | 0 | 0 |
| Recon | 0 | 311 | 0 |
| DoS | 0 | 0 | 613 |

Figures: `cm_multiclass_network.png`, `roc_multiclass_network.png`

### 10.3 Power multi-class (full 5 classes including Backdoor)

This is the **most informative multi-class result**.

| Metric | Value |
|---|---:|
| Accuracy | 0.802 |
| Precision (macro) | 0.821 |
| Recall (macro) | 0.802 |
| F1 (macro) | 0.791 |
| Macro-OVR ROC-AUC | **0.954** |

**Confusion matrix:**

|  | Benign | Recon | DoS | Cryptojacking | Backdoor |
|---|---:|---:|---:|---:|---:|
| Benign | 742 | 79 | 2 | 0 | 61 |
| Recon | 7 | 743 | 86 | 0 | 48 |
| DoS | 1 | 447 | 325 | 30 | 79 |
| Cryptojacking | 0 | 0 | 7 | 877 | 0 |
| Backdoor | 14 | 11 | 3 | 0 | 856 |

**Per-class OvR ROC-AUC (power):**

| Class | AUC |
|---|---:|
| Benign | 0.989 |
| Recon | 0.912 |
| DoS | 0.884 |
| Cryptojacking | 0.999 |
| Backdoor | 0.986 |

Figures:
- `thesis_figures/cm_multiclass_power.png`
- `thesis_figures/roc_multiclass_power.png`
- `thesis_figures/loss_multiclass_power.png`

**Reading the matrix:** DoS is most confused with Recon (447 DoS→Recon). Cryptojacking and Backdoor are comparatively well separated. This is a useful thesis discussion: binary detection can be near-perfect while fine-grained family classification remains harder on power sensors alone.

---

## 11. Training & validation loss and ROC plots

### 11.1 Where to find plots

All generated under `thesis_figures/`:

| Type | Pattern | Examples |
|---|---|---|
| Train/Val loss (binary) | `loss_{modality}_{pipeline}.png` | `loss_host_C.png`, `loss_power_B.png` |
| Binary ROC | `roc_binary_{modality}_{pipeline}.png` | `roc_binary_host_A.png` |
| Binary CM | `cm_binary_{modality}_{pipeline}.png` | `cm_binary_power_E.png` |
| Multi-class ROC | `roc_multiclass_{modality}.png` | `roc_multiclass_power.png` |
| Multi-class CM | `cm_multiclass_{modality}.png` | `cm_multiclass_host.png` |
| Multi-class loss | `loss_multiclass_{modality}.png` | `loss_multiclass_power.png` |

### 11.2 How to discuss loss curves in the thesis

- **Host/network binary losses** fall very quickly and validation F1 saturates early → strong class separation and/or optimistic split effects.  
- **Power binary losses** decrease more gradually over many epochs → more realistic learning dynamics.  
- Early stopping prevents overfitting when validation F1 plateaus.

### 11.3 How to discuss ROC plots

- Binary ROC-AUC near 1.0 means ranking of attack vs benign scores is excellent on the tested windows.  
- Multi-class OvR ROC (especially power) shows class-wise separability; DoS has the lowest AUC (0.884), matching the confusion with Recon.

---

## 12. Edge selection and Jetson packaging

### 12.1 Selection rule (thesis §16)

Do **not** choose by accuracy alone. Prefer:

Detection (F1/Recall/AUC) **+** low feature count **+** low parameters **+** low train/inference cost **+** Jetson compatibility

### 12.2 Selected model

**Host tuned Pipeline C (LDA-LSTM)**  
Reason: on the main high-dimensional modality, it preserved saturated detection while using only **3 LDA features** and, after tuning, only **1,378 parameters**.

### 12.3 Deployment package

Folder: `jetson_deploy/pipeline_C_lda_lstm/`

Contains:
- `lda_lstm.onnx`
- `scaler.joblib`, `lda.joblib`
- `feature_columns.txt`
- `model_meta.json`
- `best_lstm.pt`

Run locally/on Jetson:
```bash
python jetson_infer.py --demo
```

Optional TensorRT:
```bash
trtexec --onnx=lda_lstm.onnx --saveEngine=lda_lstm.trt --fp16
```

### 12.4 Jetson metrics still to measure on device

| Metric | Status |
|---|---|
| Inference latency | PC baseline done; Jetson TBD |
| Throughput | TBD on Jetson |
| CPU/GPU/RAM | TBD on Jetson |
| Power draw | TBD on Jetson |
| TensorRT engine size | TBD |

This still completes the **method + prototype packaging** side of Objective 5 / RQ4; on-device numbers should be inserted when hardware is available.

---

## 13. How research gaps are fulfilled

### Gap 1 — No systematic PCA/LDA comparison before temporal DL

**Fulfilled by:** Pipelines A (none), B (PCA), C (LDA) under identical splits/metrics on three modalities.  

**Evidence:** Host 220→26 (PCA) and 220→3 (LDA) with no binary F1 loss on the evaluated split; power shows PCA-1 can even be best F1. This answers whether supervised LDA or unsupervised PCA is more useful: **LDA is best for aggressive compression on host; PCA is highly effective on low-dimensional power.**

### Gap 2 — No controlled comparison of LSTM / PCA-LSTM / LDA-LSTM / LDA-BiLSTM / CNN-BiLSTM+Attention

**Fulfilled by:** Full A–E comparison on host, network and power with shared trainer and metrics.  

**Evidence:** Tables in §8. BiLSTM/Attention rarely improve binary F1 enough to justify complexity when LDA/PCA already separate classes well.

### Gap 3 — Limited attack scope (DDoS-only)

**Fulfilled by:** Attack_Group covering Recon, DoS, Cryptojacking and Backdoor; multi-class evaluation on power includes all four attack families. Network/host cover Recon+DoS(+Cryptojacking on host).  

**Evidence:** §3.5 and §10.3.

### Gap 4 — No edge deployment prototype

**Fulfilled by:** Multi-criteria selection, hyperparameter compression, ONNX export, preprocessing artifact packaging, inference script and Jetson README.  

**Evidence:** §12. Remaining work is on-device measurement, not missing methodology.

---

## 14. How research objectives are met

| Objective | Status | How |
|---|---|---|
| **O1** Preprocess CICEVSE2024 for temporal DL | **Met** | Host/network/power cleaners + windows + leakage-aware splits |
| **O2** Investigate PCA and LDA | **Met** | Train-only PCA/LDA; component counts and pipeline impact reported |
| **O3** Develop/compare A–E | **Met** | All five pipelines on all three modalities |
| **O4** Evaluate with Acc/P/R/F1/AUC/time and select by performance+cost | **Met** | Full tables; selection prefers compact LDA-LSTM |
| **O5** Optimize/deploy on Jetson Nano | **Mostly met (prototype ready)** | Tuned model + ONNX package; physical Jetson numbers pending |

---

## 15. How research questions are answered

### RQ1 — How do PCA and LDA affect performance and cost?

**Answer:**  
On host/network, both preserve binary detection while cutting features sharply; LDA compresses more aggressively (3 or 2 features) and usually trains fastest. On power (already 4-D), PCA to 1-D gave the best binary F1, while LDA (4-D supervised projection) was slightly weaker on binary F1 but supports multi-class family separation. **DR reduces cost; it does not inherently hurt detection when class structure is strong.**

### RQ2 — How do LSTM, BiLSTM and CNN-BiLSTM+Attention compare?

**Answer:**  
LSTM with good features is already strong. BiLSTM and CNN-BiLSTM+Attention increase parameters/time; binary gains are small or none on host/network, and only marginal on power (E best AUC, B best F1). **Complexity is not automatically better for this dataset.**

### RQ3 — Which pipeline has the best performance–efficiency trade-off?

**Answer:**  
- **Host (main DR story):** Pipeline **C LDA-LSTM**, then tuned to 1,378 params.  
- **Network:** also **C**.  
- **Power:** **B PCA-LSTM** slightly best F1; **C** fastest train.  

Overall thesis recommendation for edge IDS prototype: **tuned Host LDA-LSTM (C)**.

### RQ4 — Can the selected model be optimized/deployed on Jetson Nano?

**Answer:**  
Yes at the software/prototype level: model was reduced, exported to ONNX (~7 KB), and packaged with exact train-time scaler/LDA. Feasibility is demonstrated for edge packaging; **final latency/power figures require running the package on the Jetson device**.

---

## 16. Limitations and interpretation

1. **Host/network binary scores are likely optimistic** due to chronological partial splits, overlapping windows, and (for network) very few benign windows.  
2. **Multi-class host/network saturation** should be treated similarly with caution.  
3. **Power multi-class (~0.80 accuracy, 0.95 macro AUC)** is the more realistic fine-grained result and should be emphasized.  
4. LDA is supervised and needs labels at training time.  
5. Jetson hardware metrics are not yet filled.  
6. Results are for CICEVSE2024 testbed conditions; not automatic proof of zero-day production detection.

---

## 17. Final conclusion

This work built a complete experimental framework for EVSE attack detection on CICEVSE2024 using **host, network and power** modalities. Five temporal pipelines were compared under a consistent protocol with train-only PCA/LDA. The study shows that:

1. **High detection performance is achievable** with temporal deep learning on CICEVSE2024 windows.  
2. **Dimensionality reduction is valuable**: PCA and especially multi-class LDA drastically reduce feature dimensionality while preserving binary detection quality on the evaluated splits.  
3. **Heavier architectures are not always justified**: BiLSTM and CNN-BiLSTM+Attention seldom outperform compact LDA/PCA-LSTM enough to warrant their cost.  
4. **Attack scope was broadened** beyond DDoS-only through Recon/DoS/Cryptojacking/Backdoor labelling and multi-class evaluation (with Backdoor available in power).  
5. **An edge-ready prototype** was produced by selecting and tuning host LDA-LSTM, exporting ONNX, and packaging preprocessing for Jetson Nano deployment.

Therefore, the thesis aim is experimentally supported: a **supervised dimensionality-reduction + temporal deep-learning framework** can detect EVSE attacks effectively, and a compact LDA-LSTM is a practical candidate for **edge-based EV charging security monitoring**. The immediate next step for the final manuscript is to insert on-device Jetson measurements and optionally add stricter session-holdout robustness tests.

---

## 18. File map (where everything lives)

### Preprocessing
- `hosteventspreprocessing.py`
- `nwtrafficpreprocessing.py`
- `powerpreprocessing.py`

### Sequence / DR preparation
- Host: `prepare_host_sequences.py`, `prepare_dr_sequences.py`
- Network: `prepare_network_sequences.py`, `prepare_network_dr_sequences.py`
- Power: `prepare_power_sequences.py`, `prepare_power_dr_sequences.py`

### Training
- Shared: `lstm_common.py`
- Host A–E scripts / models under `models/pipeline_*`
- Network: `train_network_pipelines_A_to_E.py` → `models/network_EVSE_A/`
- Power: `train_power_pipelines_A_to_E.py` → `models/power_EVSE_B/`

### Edge
- `tune_lda_lstm.py`
- `export_onnx.py`
- `jetson_infer.py`
- `jetson_deploy/pipeline_C_lda_lstm/`

### Figures / this guide
- `generate_thesis_figures.py`
- `thesis_figures/` (PNG + multiclass JSON)
- `MASTER_THESIS_EXPERIMENTAL_REFERENCE.md` *(this file)*

### Quick result JSON pointers
- `models/comparison_A_to_E.json` (host)
- `models/network_EVSE_A/comparison_A_to_E_network.json`
- `models/power_EVSE_B/comparison_A_to_E_power.json`
- `models/edge_selection_summary.json`
- `thesis_figures/multiclass_host.json`
- `thesis_figures/multiclass_power.json`
- `thesis_figures/multiclass_network.json`

---

*Generated from the experimental artefacts of this project for thesis drafting and viva preparation. Update Jetson TBD cells after on-device runs.*
