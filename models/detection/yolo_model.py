from ultralytics import YOLO
from models.detection_regisrty import DetectionModelRegistry


@DetectionModelRegistry.register("yolo_v8")
class YOLOv8Wrapper:

    def __init__(self, model_name="yolov8n.pt"):
        self.model = YOLO(model_name)

    def train(self, data_yaml, epochs, imgsz):
        self.model.train(
            data=data_yaml,
            epochs=epochs,
            imgsz=imgsz
        )