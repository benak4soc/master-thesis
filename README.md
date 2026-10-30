# Master Thesis — EVSE Attack Detection (CICEVSE2024)

**Title:** A Supervised Dimensionality-Reduction and Temporal Deep-Learning Framework for Attack Detection in EV Charging Infrastructure

**Repository flow (run / read in numbered order):**

| Step | Folder | Contents |
|------|--------|----------|
| 01 | `01_documentation/` | Thesis reference, methodology algorithm, architecture, results draft |
| 02 | `02_preprocessing/` | Host / network / power cleaning scripts |
| 03 | `03_sequence_preparation/` | Windowing + PCA/LDA preparation |
| 04 | `04_model_training/` | Pipelines A–E trainers + shared LSTM module + tuning |
| 05 | `05_evaluation_plotting/` | Confusion matrix, ROC, loss, publication plots, CSV export |
| 06 | `06_edge_deployment/` | ONNX export, Jetson inference package |
| 07 | `07_results_and_models/` | Trained checkpoints, JSON metrics, `all_pipeline_results.csv` |
| 08 | `08_figures/` | All generated figures |
| 09 | `09_processed_datasets/` | Cleaned/processed datasets & sequences |
| 10 | `10_raw_data/` | Host + power raw data; network samples + download notes |
| 11 | `11_jetson_nano_demo/` | Jetson Nano setup TODO + Pipeline C inference and desktop GUI |
| 12 | `12_evse_live_dashboard/` | Browser dashboard: Benign / DoS / RomCom / Cryptojacking replay on Pipeline C |
| 13 | `13_rpi4_algorithm_dashboard/` | Raspberry Pi 4 discussion: NumPy LDA-LSTM runtime, on-device results, and the demo dashboard |

## Pipelines
- **A** LSTM (no DR)
- **B** PCA-LSTM
- **C** LDA-LSTM
- **D** LDA-BiLSTM
- **E** CNN-BiLSTM + Attention

Modalities: Host HPC/Kernel, Network EVSE-A, Power EVSE-B.

## Quick start
1. Place full CICEVSE2024 raw files as described in `10_raw_data/README_DATA.md`.
2. Run preprocessing scripts in `02_preprocessing/` (paths may need adjusting to your local data location).
3. Run sequence scripts in `03_sequence_preparation/`.
4. Train with scripts in `04_model_training/`.
5. Plot/export with `05_evaluation_plotting/`.
6. Deploy with `06_edge_deployment/`.
7. Demonstrate on Jetson with `11_jetson_nano_demo/` (`python jetson_demo_gui.py`).
8. Open the scenario dashboard with `12_evse_live_dashboard/` (`python evse_dashboard.py`).
9. Read the Raspberry Pi 4 runtime discussion in `13_rpi4_algorithm_dashboard/` and run `python rpi4_dashboard.py`.

## Note on large files
Files larger than ~95MB (especially full cleaned network CSV and large network attack CSVs) are omitted due to GitHub limits. Regenerate them with the preprocessing scripts after downloading the official dataset.

## Key result file
`07_results_and_models/all_pipeline_results.csv`
