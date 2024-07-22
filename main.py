from ultralytics import YOLO
model = YOLO("yolov8n-cls.pt")
results = model.train(data="data", epochs=12, imgsz=64)