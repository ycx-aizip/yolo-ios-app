from ultralytics import YOLO


# Load a trained YOLO11 PyTorch model
model = YOLO("yolo11l.pt")

# Export the PyTorch model to CoreML INT8 format with NMS layers
# The imgsz property may be adjusted when you export a trained model
model.export(format="coreml", int8=True, nms=True, imgsz=640)

# for size in ("n", "s", "m", "l", "x"):      # nano, small, medium, large, xlarge
#     # Load a base YOLO11 PyTorch model
#     model = YOLO(f"yolo11{size}.pt")
#     # Export to CoreML with INT8 quantization and NMS layers
#     model.export(format="coreml", int8=True, nms=True)