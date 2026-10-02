# Jetson Nano deployment — Pipeline C (LDA-LSTM)
#
# Contents of jetson_deploy/pipeline_C_lda_lstm/:
#   lda_lstm.onnx          ONNX model
#   scaler.joblib          StandardScaler (fit on train)
#   lda.joblib             Multi-class LDA (fit on train)
#   feature_columns.txt    Raw HPC/kernel feature order
#   best_lstm.pt           PyTorch checkpoint
#   model_meta.json        Shapes / metrics / labels
#
# PC / Jetson Python deps:
#   pip install numpy joblib scikit-learn onnxruntime
#
# Run demo:
#   python jetson_infer.py --demo
#
# Real window (.npy with shape [T, F] or [B, T, F]):
#   python jetson_infer.py --input window.npy
#
# Optional TensorRT on Jetson (after JetPack install):
#   /usr/src/tensorrt/bin/trtexec \
#     --onnx=lda_lstm.onnx \
#     --saveEngine=lda_lstm.trt \
#     --fp16
#
# Edge metrics to record on Jetson (thesis §15.7):
#   - inference latency (ms/window)
#   - throughput (windows/s)
#   - CPU / GPU utilization
#   - RAM usage
#   - model size (onnx / trt)
#   - power draw if measurable
