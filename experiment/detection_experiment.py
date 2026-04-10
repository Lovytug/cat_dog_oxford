from ultralytics import YOLO
from tools.covert_to_yolo import OxfordPetsToYOLO


class DetectionExperiment:

    def __init__(self, config, images_dir, annotations_dir):
        self.config = config
        self.images_dir = images_dir
        self.annotations_dir = annotations_dir

    def run(self):

        # 🔥 1. Конвертация (автоматически)
        converter = OxfordPetsToYOLO(
            images_dir=self.images_dir,
            annotations_dir=self.annotations_dir,
            output_dir="yolo_data"
        )
        converter.convert()

        # 🔥 2. Загружаем модель
        model = YOLO(self.config["model"]["params"]["model_name"])

        # 🔥 3. Запуск обучения
        model.train(
            data="yolo_data/dataset.yaml",
            epochs=self.config["epochs"],
            imgsz=self.config.get("img_size", 640)
        )