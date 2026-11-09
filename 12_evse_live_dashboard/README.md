# EVSE Live Attack-Detection Dashboard

Browser dashboard for Jetson Nano / PC demos of **Pipeline C (LDA-LSTM)**.

The ONNX/Torch model outputs **Benign vs Attack**. Scenario buttons (DoS / RomCom / Cryptojacking) drive the supporting telemetry story (network, CPU/temp, OCPP, processes) while the live score stays grounded in the real model.

## Panels

1. Top status badge (Benign / Attack + family), EVSE state, uptime, model/version, latency  
2. Live detection — confidence gauge, 4-class bars, probability history chart  
3. Attack timeline — filterable log (time, class, confidence, source IP, action)  
4. Jetson health — CPU / per-core / temp / RAM / throttle (+ live `/proc` on Nano)  
5. Network — pps, B/s, connections, SYN half-open, top talkers  
6. EVSE/OCPP — message mix, V/A/kW/kWh  
7. Process/host — suspicious procs, outbound, beaconing, file alerts  
8. Attack summary — donut, counters, confusion matrix, published Acc/F1/AUC  
9. Demo controls — scenarios, start/stop, mitigation (block / throttle / isolate)  
10. Alerts — banner + optional sound + recommended mitigation text  

## Run (PC)

```bash
cd 12_evse_live_dashboard
python evse_dashboard.py --self-check
python evse_dashboard.py --deploy-dir ../11_jetson_nano_demo/pipeline_C_lda_lstm
# opens http://127.0.0.1:8765/
```

## Run (Jetson Nano)

```bash
cd ~/evse-ids   # or copy this folder next to pipeline_C_lda_lstm
python3 evse_dashboard.py --host 0.0.0.0 --port 8765 --no-browser \
  --deploy-dir pipeline_C_lda_lstm
```

Then open in TigerVNC browser: `http://127.0.0.1:8765/`  
or from the PC: `http://192.168.1.4:8765/`

On the Nano, host CPU/temp/RAM are blended from live `/proc` sensors when available (pill shows **Jetson live sensors**).

## Demo script

1. Start detection → **Run Benign** (green badge, charging, calm charts)  
2. **Simulate DoS** → red badge, SYN/pps spike, Faulted EVSE → **Block IP** recovers  
3. **Simulate Cryptojacking** → high CPU/temp/throttle → **Throttle** recovers  
4. **Simulate RomCom** → suspicious process, beacon, file alerts → **Isolate EVSE**  
