import shutil
import random
from pathlib import Path
import xml.etree.ElementTree as ET
from PIL import Image


class OxfordPetsToYOLO:

    def __init__(self, images_dir, annotations_dir, output_dir="yolo_data"):
        self.images_dir = Path(images_dir)
        self.annotations_dir = Path(annotations_dir)
        self.output_dir = Path(output_dir)

    def convert(self):

        print("🔄 Конвертация датасета в YOLO формат...")

        all_images = list(self.images_dir.glob("*.jpg"))

        if len(all_images) == 0:
            raise ValueError("❌ В папке нет изображений")

        random.shuffle(all_images)

        split_idx = int(0.8 * len(all_images))

        train_images = all_images[:split_idx]
        val_images = all_images[split_idx:]

        print(f"📊 Train: {len(train_images)}, Val: {len(val_images)}")

        self._process_list("train", train_images)
        self._process_list("val", val_images)

        self._create_yaml()

        print("✅ Готово")


    def _process_list(self, split, image_paths):

        img_out = self.output_dir / "images" / split
        lbl_out = self.output_dir / "labels" / split

        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)

        count = 0

        # 👉 читаем mapping из txt
        split_file = self.annotations_dir / "trainval.txt"
        mapping = {}

        with open(split_file) as f:
            for line in f:
                image_id, _, species, _ = line.strip().split()
                mapping[image_id] = int(species)

        for img_path in image_paths:

            image_id = img_path.stem
            xml_path = self.annotations_dir / "xmls" / f"{image_id}.xml"

            if not xml_path.exists():
                continue

            if image_id not in mapping:
                continue

            species = mapping[image_id]

            shutil.copy(img_path, img_out / img_path.name)

            self._convert_xml(
                xml_path,
                lbl_out / f"{image_id}.txt",
                img_path,
                species
            )

            count += 1

        print(f"✅ {split}: {count} изображений")


    def _convert_xml(self, xml_path, out_path, img_path, species):

        tree = ET.parse(xml_path)
        root = tree.getroot()

        img = Image.open(img_path)
        w, h = img.size

        lines = []

        # 👉 переводим species → class_id
        class_id = species - 1  # 1→0 (cat), 2→1 (dog)

        for obj in root.findall("object"):
            bbox = obj.find("bndbox")

            xmin = int(bbox.find("xmin").text)
            ymin = int(bbox.find("ymin").text)
            xmax = int(bbox.find("xmax").text)
            ymax = int(bbox.find("ymax").text)

            x_center = ((xmin + xmax) / 2) / w
            y_center = ((ymin + ymax) / 2) / h
            bw = (xmax - xmin) / w
            bh = (ymax - ymin) / h

            lines.append(f"{class_id} {x_center} {y_center} {bw} {bh}")

        with open(out_path, "w") as f:
            f.write("\n".join(lines))

    def _create_yaml(self):

        yaml_path = self.output_dir / "dataset.yaml"

        content = f"""path: {self.output_dir.resolve()}

train: images/train
val: images/val

nc: 1
names: ["pet"]
"""

        with open(yaml_path, "w") as f:
            f.write(content)

        print(f"📄 dataset.yaml создан: {yaml_path}")