# Results Chapter Draft (Short)
# Master's Thesis:
# A Supervised Dimensionality-Reduction and Temporal Deep-Learning
# Framework for Attack Detection in EV Charging Infrastructure
#
# Status: draft text for Chapter "Results and Discussion"
# Based on CICEVSE2024 EVSE-B host HPC/kernel experiments (Pipelines A–E)
# and edge-oriented selection / ONNX packaging of Pipeline C.

---

## X Results and Discussion

### X.1 Experimental Setup

The primary experimental modality was the CICEVSE2024 EVSE-B host HPC/kernel event stream. After cleaning invalid and corrupted rows, the processed host dataset contained 6,166 labelled observations with 220 informative numerical features. Binary labels distinguished Benign and Attack traffic. For supervised dimensionality reduction, Attack_Group labels were derived as Benign, Recon, DoS and Cryptojacking (Backdoor was not present in the host HPC file; it appears in the power-consumption modality and was retained for multi-class labelling consistency).

Temporal windows of length 10 (stride 1) were constructed. A hybrid split was used to reduce leakage: session-aware assignment for Recon and DoS sessions, and chronological within-session splitting for Benign and Cryptojacking (only two sessions each). The StandardScaler, PCA and LDA transformers were fitted on training data only and then applied to validation and test partitions.

Five pipelines were evaluated under a common protocol:

| Pipeline | Dimensionality reduction | Temporal model |
|---|---|---|
| A | None | LSTM |
| B | PCA (95% variance → 26 components) | LSTM |
| C | Multi-class LDA (→ 3 components) | LSTM |
| D | Multi-class LDA (→ 3 components) | BiLSTM |
| E | None | CNN–BiLSTM + Attention |

Evaluation metrics followed the thesis plan: Accuracy, Precision, Recall, F1-score, ROC-AUC and training time. Model size (parameter count) was recorded as an edge-relevant secondary criterion.

### X.2 Comparative Detection Performance (Pipelines A–E)

Table X.1 summarises test-set results on the host temporal windows.

**Table X.1.** Test performance and efficiency of Pipelines A–E.

| Pipeline | Features | Parameters | Acc. | Prec. | Rec. | F1 | ROC-AUC | Train time (s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A LSTM | 220 | 73,346 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 9.59 |
| B PCA-LSTM | 26 | 23,682 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 6.64 |
| C LDA-LSTM | 3 | 17,794 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 3.52 |
| D LDA-BiLSTM | 3 | 35,586 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 6.50 |
| E CNN-BiLSTM+Attn | 220 | 125,762 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 11.43 |

All five pipelines achieved saturated classification metrics on the held-out test windows (F1 = 1.000, ROC-AUC = 1.000). Consequently, discrimination performance alone does not differentiate the architectures on this split. The comparative value of the study therefore shifts to computational efficiency and edge suitability, which is consistent with the multi-criteria selection strategy defined in the methodology.

Two observations are particularly relevant:

1. **Dimensionality reduction preserves detection quality.** PCA reduced the feature space from 220 to 26 components (≈95% explained variance), and multi-class LDA reduced it further to three discriminative components, without degrading Accuracy/F1/ROC-AUC on this test set.
2. **Architectural complexity did not yield additional detection benefit.** Replacing LSTM with BiLSTM (Pipeline D) or introducing CNN + attention (Pipeline E) increased parameter count and training time but did not improve test metrics relative to the simpler LDA-LSTM baseline.

These findings support the thesis hypothesis that supervised LDA can provide a compact, attack-discriminative representation for temporal EVSE intrusion detection, at least under the present dataset organisation and binary detection task.

### X.3 Effect of PCA versus LDA

PCA retained a larger number of components (26) because it optimises variance rather than class separability. Multi-class LDA, trained on Attack_Group labels (Benign / Recon / DoS / Cryptojacking), produced at most \(C-1 = 3\) components. The first LDA axis accounted for approximately 98.4% of the between-class scatter, indicating strong linear class structure in the host feature space.

For the primary binary IDS task, LDA features were still consumed by an LSTM classifier predicting Benign versus Attack. This design separates (i) supervised projection using fine-grained family labels from (ii) the operational binary decision required by an endpoint monitor. A classical binary LDA baseline (\(C-1 = 1\)) was also prepared for limitation discussion, reflecting the known rank constraint of LDA.

### X.4 Model Selection for Edge Deployment

Because detection metrics were saturated, selection followed the multi-criteria rule stated in the proposal: detection quality, complemented by parameter count, training cost and inference cost. Under this rule, **Pipeline C (LDA-LSTM)** was selected as the deployment candidate: it matched the best F1/ROC-AUC while using the fewest features and the lowest training time among A–E before further tuning.

Hyperparameter tuning of Pipeline C (72 configurations) further reduced capacity for edge use:

**Table X.2.** Tuned Pipeline C selected for Jetson packaging.

| Item | Value |
|---|---|
| Hidden size | 16 |
| LSTM layers | 1 |
| Dropout | 0.2 |
| Learning rate | \(10^{-3}\) |
| Batch size | 64 |
| Parameters | 1,378 |
| Test F1 / ROC-AUC | 1.000 / 1.000 |
| PyTorch latency (PC CPU) | ≈ 0.40 ms / window |
| ONNX model size | ≈ 7.0 KB |

The tuned model was exported to ONNX and packaged with the training-time StandardScaler and LDA transforms, ensuring that edge inference applies the same preprocessing chain as training. A CPU ONNX Runtime demo on the development workstation measured roughly 3.2 ms per window for the packaged preprocessing + inference path (synthetic input). Physical Jetson Nano measurements (latency, throughput, CPU/GPU utilisation, RAM, power) remain to be recorded on-device and should be inserted into Table X.3 once available.

**Table X.3.** Edge metrics on NVIDIA Jetson Nano (to be completed on device).

| Metric | Value |
|---|---|
| Inference latency (ms/window) | *TBD* |
| Throughput (windows/s) | *TBD* |
| CPU utilisation (%) | *TBD* |
| GPU utilisation (%) | *TBD* |
| RAM usage (MB) | *TBD* |
| Model size ONNX / TensorRT | 7.0 KB / *TBD* |
| Power consumption (W) | *TBD* |

### X.5 Supporting Modalities (Network and Power)

Network-traffic preprocessing for EVSE-A and power-consumption preprocessing for EVSE-B were completed as supporting cyber and cyber-physical views. The power modality is particularly important for multi-class completeness: unlike the host HPC file, it contains Backdoor samples (Attack_Group = Backdoor) alongside Recon, DoS and Cryptojacking. The present A–E comparison, however, focuses on host HPC/kernel features because they constitute the high-dimensional setting that motivates PCA/LDA. Extension to power-only or multi-modal fusion is left as future / supplementary work.

### X.6 Discussion of Optimistic Scores

The perfect test scores should be interpreted cautiously. Contributing factors include: (i) chronological splitting within Benign/Cryptojacking sessions, which may retain local temporal correlation; (ii) overlapping windows (stride = 1); and (iii) strong linear separability of host counters under the labelled scenarios. The results therefore demonstrate feasibility and efficiency ranking under a controlled protocol, not a claim of guaranteed zero-day detection in production EVSE networks. Stricter session hold-outs, larger stride values, and cross-scenario evaluation are recommended robustness checks for the final thesis revision.

### X.7 Summary of Findings

1. Pipelines A–E all achieved high binary detection performance on the processed CICEVSE2024 host windows.
2. PCA and especially multi-class LDA substantially reduced dimensionality without loss of test F1/ROC-AUC on this split.
3. BiLSTM and CNN–BiLSTM–Attention increased complexity without measurable detection gain here.
4. Tuned LDA-LSTM (Pipeline C) offers the best efficiency trade-off for edge deployment and was exported to ONNX for Jetson Nano validation.
5. Remaining work is on-device Jetson measurement and, optionally, stricter leakage-resistant evaluation and multi-modal experiments using power/network streams.

---

*Draft generated from experimental artefacts in `models/comparison_A_to_E.json` and `models/edge_selection_summary.json`. Replace TBD Jetson cells after device runs; adjust chapter numbering to the final thesis outline.*
