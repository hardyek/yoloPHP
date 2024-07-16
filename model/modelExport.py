# Should be run from within the model directory.
from ultralytics import YOLO

model = YOLO("yolov8n.pt") # Update with the path to desired model to convert into TorchScript.

model.export(format="torchscript", imgsz=640)