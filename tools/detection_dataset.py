import torch
from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image
import xml.etree.ElementTree as ET


class OxfordPetsDetectionDataset(Dataset):

    def __init__(self, images_dir, annotations_dir, split_file, transform=None):
        self.images_dir = Path(images_dir)
        self.annotations_dir = Path(annotations_dir)
        self.transform = transform

        self.samples = []

        with open(split_file, "r") as f:
            for line in f:
                image_id = line.strip().split()[0]

                img_path = self.images_dir / f"{image_id}.jpg"
                xml_path = self.annotations_dir / "xmls" / f"{image_id}.xml"

                self.samples.append((img_path, xml_path))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, xml_path = self.samples[idx]

        image = Image.open(img_path).convert("RGB")

        boxes = []
        labels = []

        tree = ET.parse(xml_path)
        root = tree.getroot()

        for obj in root.findall("object"):
            bbox = obj.find("bndbox")

            xmin = int(bbox.find("xmin").text)
            ymin = int(bbox.find("ymin").text)
            xmax = int(bbox.find("xmax").text)
            ymax = int(bbox.find("ymax").text)

            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(1)  # один класс (pet)

        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.long)
        }

        if self.transform:
            image = self.transform(image)

        return image, target