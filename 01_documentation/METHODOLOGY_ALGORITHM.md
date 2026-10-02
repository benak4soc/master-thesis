# Methodology Algorithm (Publication-Ready)

Use the following as **Algorithm 1** (or split into Algorithms 1–3) in the thesis/journal paper.  
Notation is aligned with the implemented CICEVSE2024 pipelines A–E.

---

## Algorithm 1: Supervised Dimensionality-Reduction and Temporal Deep-Learning Framework for EVSE Attack Detection

**Require:**  
Raw multimodal CICEVSE2024 datasets \(\mathcal{D} = \{\mathcal{D}_{\text{host}}, \mathcal{D}_{\text{net}}, \mathcal{D}_{\text{power}}\}\)  
Window length \(w\), stride \(s\), train/val/test ratios \((\rho_{\text{tr}},\rho_{\text{va}},\rho_{\text{te}})\)  
PCA variance threshold \(\tau\) (e.g., \(0.95\))  
Deep models \(\mathcal{M} = \{\text{LSTM}, \text{BiLSTM}, \text{CNN-BiLSTM+Attn}\}\)  
Evaluation metrics \(\mathcal{E} = \{\text{Acc}, \text{Prec}, \text{Rec}, \text{F1}, \text{ROC-AUC}, \text{TrainTime}\}\)

**Ensure:**  
Best edge-ready detector \(f^\star\) with preprocessing transforms \((\phi_{\text{scale}}, \phi_{\text{DR}})\)

---

### Phase A — Data Cleaning and Label Construction

1. **for** each modality \(m \in \{\text{host}, \text{net}, \text{power}\}\) **do**
2.     Load raw table \(X^{(m)}\)
3.     Remove artifact columns / empty columns / duplicate rows / invalid records
4.     Normalize metadata fields (`Label`, `Attack`, `Scenario`/`Attack-Group`, `State`)
5.     Construct binary label:
       \[
       y_i =
       \begin{cases}
       0, & \text{if sample } i \text{ is Benign}\\
       1, & \text{if sample } i \text{ is Attack}
       \end{cases}
       \]
6.     Map each sample to attack family  
       \(g_i \in \{\text{Benign}, \text{Recon}, \text{DoS}, \text{Cryptojacking}, \text{Backdoor}\}\)
7.     Select numeric feature matrix \(X^{(m)} \in \mathbb{R}^{N_m \times d_m}\)  
       (exclude identifiers and label-leaking fields)
8. **end for**

---

### Phase B — Leakage-Aware Partitioning

9. **for** each modality \(m\) **do**
10.    Define sessions \(S^{(m)}\) (e.g., file/scenario/state/interface blocks)
11.    **if** group \(g\) has enough sessions **then**
12.        Assign whole sessions to train/val/test with ratios \((\rho_{\text{tr}},\rho_{\text{va}},\rho_{\text{te}})\)
13.    **else**
14.        Apply chronological split within each session (prefix/mid/suffix)
15.    **end if**
16.    Obtain disjoint sets \(\mathcal{T}_{\text{tr}}^{(m)}, \mathcal{T}_{\text{va}}^{(m)}, \mathcal{T}_{\text{te}}^{(m)}\)
17. **end for**

---

### Phase C — Train-Only Scaling

18. **for** each modality \(m\) **do**
19.    Fit scaler \(\phi_{\text{scale}}^{(m)}\) on training features only
20.    Transform train/val/test:
       \[
       \tilde{X} = \phi_{\text{scale}}^{(m)}(X)
       \]
21. **end for**

---

### Phase D — Temporal Sequence Construction

22. **for** each modality \(m\) and split \(p \in \{\text{tr},\text{va},\text{te}\}\) **do**
23.    **for** each session in split \(p\) **do**
24.        For \(t = w, w+s, w+2s, \ldots\) form window
           \[
           \mathbf{z}_t = (\tilde{x}_{t-w+1},\ldots,\tilde{x}_t)
           \]
25.        Assign window label \(y_t\) (and family \(g_t\)) from the last timestep
26.    **end for**
27.    Store sequence dataset \(\mathcal{W}_p^{(m)} = \{(\mathbf{z}_t, y_t, g_t)\}\)
28. **end for**

---

### Phase E — Dimensionality Reduction (fit on training windows only)

29. **for** each modality \(m\) **do**
30.    Flatten training windows to timestep matrix \(U_{\text{tr}}^{(m)}\)
31.    **PCA:** fit \(\phi_{\text{PCA}}^{(m)}\) to retain variance \(\ge \tau\)
32.    **LDA:** fit \(\phi_{\text{LDA}}^{(m)}\) using multi-class labels \(g_t\)  
        with at most \(C-1\) components
33.    Apply \(\phi_{\text{PCA}}^{(m)}\) and \(\phi_{\text{LDA}}^{(m)}\) to all splits (train parameters frozen)
34. **end for**

---

### Phase F — Pipeline Training and Evaluation

35. Define pipelines:
    - **A:** no DR + LSTM  
    - **B:** PCA + LSTM  
    - **C:** LDA + LSTM  
    - **D:** LDA + BiLSTM  
    - **E:** no DR + CNN–BiLSTM–Attention
36. **for** each modality \(m\) **do**
37.    **for** each pipeline \(\pi \in \{A,B,C,D,E\}\) **do**
38.        Initialize model \(f_\pi^{(m)}\) with weighted cross-entropy loss
39.        Train on \(\mathcal{W}_{\text{tr}}^{(m)}\) with early stopping on validation F1
40.        Evaluate \(f_\pi^{(m)}\) on \(\mathcal{W}_{\text{te}}^{(m)}\) using metrics \(\mathcal{E}\)
41.        Record parameter count, feature dimension, and training time
42.    **end for**
43. **end for**

---

### Phase G — Multi-Criteria Model Selection

44. Rank pipelines by detection quality and efficiency:
    \[
    \text{Score}(\pi)
    =
    \alpha\cdot \text{F1}(\pi)
    +
    \beta\cdot \text{AUC}(\pi)
    -
    \gamma\cdot \widehat{\text{Cost}}(\pi)
    \]
    where \(\widehat{\text{Cost}}\) includes feature dimension, parameter count, and training/inference time
45. Select best candidate \(f^\star\) (in this work: compact **LDA–LSTM** on host modality)
46. Optionally tune hyperparameters of \(f^\star\) on validation data

---

### Phase H — Edge Packaging and Deployment

47. Export \(f^\star\) to ONNX (optionally convert to TensorRT on Jetson)
48. Package frozen transforms \((\phi_{\text{scale}}^\star, \phi_{\text{DR}}^\star)\) with the model
49. On edge device, for each incoming window \(\mathbf{z}\):
50.    \(\mathbf{u} \leftarrow \phi_{\text{DR}}^\star(\phi_{\text{scale}}^\star(\mathbf{z}))\)
51.    \(\hat{y} \leftarrow f^\star(\mathbf{u})\)
52.    Measure latency, throughput, CPU/GPU/RAM (and power if available)
53. **return** \(f^\star\), transforms, and evaluation reports

---

## Algorithm 2 (Optional Compact Form for Journal Page Limits)

If the venue needs a shorter algorithm, use this condensed version:

**Algorithm 2:** EVSE Attack Detection via DR–Temporal DL

**Require:** Dataset \(\mathcal{D}\), window \(w\), pipelines \(\Pi=\{A..E\}\)  
**Ensure:** Selected detector \(f^\star\)

1. Clean \(\mathcal{D}\); build binary labels \(y\) and family labels \(g\)
2. Split by session/time into train/val/test (no leakage)
3. Fit scaler on train; transform all splits
4. Build sliding windows of length \(w\)
5. Fit PCA and multi-class LDA on train windows only; transform all splits
6. **for** \(\pi \in \Pi\) **do**
7.     Train temporal model of pipeline \(\pi\); evaluate on test set
8. **end for**
9. Select \(f^\star\) by F1/AUC versus computational cost
10. Export \(f^\star\) (+ scaler/DR) for edge inference
11. **return** \(f^\star\)

---

## Suggested in-text description (1 paragraph for paper)

> The proposed methodology first cleans each CICEVSE2024 modality and constructs binary Benign/Attack labels together with multi-class attack-family labels. Data are partitioned using a session-aware or chronological protocol to reduce leakage, after which a StandardScaler is fitted on training data only. Overlapping temporal windows are then formed for sequence learning. Unsupervised PCA and supervised multi-class LDA are fitted exclusively on the training partition and applied to validation/test data. Five pipelines—LSTM, PCA–LSTM, LDA–LSTM, LDA–BiLSTM, and CNN–BiLSTM with Attention—are trained under a common protocol and compared using Accuracy, Precision, Recall, F1-score, ROC-AUC, and training time, complemented by model-size metrics. The best efficiency–performance trade-off is selected, optimized, and exported for edge deployment with frozen preprocessing transforms.

---

## Symbol table (useful for the paper)

| Symbol | Meaning |
|---|---|
| \(X^{(m)}\) | Feature matrix of modality \(m\) |
| \(d_m\) | Original feature dimension |
| \(y_i\) | Binary label (0 Benign, 1 Attack) |
| \(g_i\) | Attack-family label |
| \(w\) | Window length |
| \(\phi_{\text{scale}}\) | StandardScaler |
| \(\phi_{\text{PCA}}, \phi_{\text{LDA}}\) | Dimensionality-reduction maps |
| \(f_\pi\) | Model of pipeline \(\pi\) |
| \(f^\star\) | Selected final model |

---

## LaTeX tip

In LaTeX, format with `algorithm` + `algpseudocode` (or `algorithm2e`).  
Keep Phase A–H as comments/`\Statex` headers so reviewers can follow the pipeline visually.
