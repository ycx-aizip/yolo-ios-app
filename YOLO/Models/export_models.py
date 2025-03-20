from ultralytics import YOLO

# Loop through all YOLO11 model sizes
for size in ("n", "s", "m", "l", "x"):      # nano, small, medium, large, xlarge
    # Load a base YOLO11 PyTorch model
    model = YOLO(f"yolo11{size}.pt")
    # Export to CoreML with INT8 quantization and NMS layers
    model.export(format="coreml", int8=True, nms=True)