# System Architecture Diagram

**Figure suggestion title:**  
*System architecture of the proposed supervised dimensionality-reduction and temporal deep-learning framework for EVSE attack detection.*

Static figure file: `thesis_figures/system_architecture.png`

---

## 1) Overall system architecture (Mermaid)

```mermaid
flowchart TB
  subgraph SRC["CICEVSE2024 Data Sources"]
    H["Host HPC / Kernel Events<br/>EVSE-B"]
    N["Network Traffic<br/>EVSE-A"]
    P["Power Consumption<br/>EVSE-B"]
  end

  subgraph PRE["Preprocessing Layer"]
    C1["Data Cleaning"]
    C2["Binary Label y = Benign/Attack"]
    C3["Attack_Group Mapping<br/>Recon / DoS / Cryptojacking / Backdoor"]
    C4["Feature Selection<br/>no ID / label leakage"]
    C5["Session / Chronological Split"]
    C6["Train-only StandardScaler"]
  end

  subgraph SEQ["Temporal Modelling Input"]
    W["Sliding Window Construction<br/>window length w"]
  end

  subgraph DR["Dimensionality Reduction<br/>fit on Train only"]
    PCA["PCA<br/>retain ~95% variance"]
    LDA["Multi-class LDA<br/>max C-1 components"]
  end

  subgraph PIP["Experimental Pipelines"]
    A["A: LSTM"]
    B["B: PCA-LSTM"]
    C["C: LDA-LSTM"]
    D["D: LDA-BiLSTM"]
    E["E: CNN-BiLSTM + Attention"]
  end

  subgraph EVAL["Evaluation & Selection"]
    M["Acc / Prec / Rec / F1 / ROC-AUC / Train Time"]
    S["Multi-criteria selection<br/>performance + efficiency"]
  end

  subgraph EDGE["Edge Deployment"]
    T["Hyperparameter Tuning"]
    O["ONNX Export<br/>+ frozen Scaler/LDA"]
    J["NVIDIA Jetson Nano<br/>near-EVSE inference"]
  end

  H --> PRE
  N --> PRE
  P --> PRE
  PRE --> SEQ
  SEQ --> DR
  SEQ --> A
  PCA --> B
  LDA --> C
  LDA --> D
  SEQ --> E
  DR --> PCA
  DR --> LDA
  A --> EVAL
  B --> EVAL
  C --> EVAL
  D --> EVAL
  E --> EVAL
  EVAL --> S
  S --> T --> O --> J
```

---

## 2) Deployment / runtime architecture (Mermaid)

```mermaid
flowchart LR
  subgraph EVSE["EV Charging Station Side"]
    EVSEHW["EVSE Device<br/>Host / Network / Power sensors"]
  end

  subgraph EDGEBOX["Edge Security Node"]
    BUF["Window Buffer"]
    SC["StandardScaler"]
    DR2["LDA / PCA Transform"]
    INF["Selected Temporal Model<br/>e.g., LDA-LSTM ONNX"]
    OUT["Alert: Benign / Attack"]
  end

  subgraph CLOUD["Optional Backend"]
    LOG["Logging / Monitoring Dashboard"]
  end

  EVSEHW --> BUF --> SC --> DR2 --> INF --> OUT
  OUT -.-> LOG
```

---

## 3) Caption text for thesis/journal

> **Figure X.** System architecture of the proposed framework. Multimodal CICEVSE2024 telemetry (host HPC/kernel events, network traffic, and power consumption) is cleaned and labelled, split with a leakage-aware protocol, and converted into temporal windows. PCA and multi-class LDA are fitted on training data only. Five temporal deep-learning pipelines (A–E) are trained and compared using detection and efficiency metrics. The selected model is optimized and deployed on an NVIDIA Jetson Nano for near-EVSE edge inference.

---

## 4) Layer summary (for text around the figure)

| Layer | Function |
|---|---|
| Data sources | Host, network, and power modalities from CICEVSE2024 |
| Preprocessing | Cleaning, binary + family labels, leakage-safe features, split, scaling |
| Sequence construction | Sliding windows for temporal models |
| Dimensionality reduction | Train-only PCA and multi-class LDA |
| Pipelines A–E | Controlled temporal DL comparison |
| Evaluation & selection | Accuracy/F1/AUC vs computational cost |
| Edge deployment | Tuned model + ONNX/TensorRT on Jetson Nano |
