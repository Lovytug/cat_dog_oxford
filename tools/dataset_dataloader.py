
import torch
from torch.utils.data import Dataset, DataLoader, Subset

from PIL import Image

from pathlib import Path


class CatDogOxfortDataset(Dataset):

    def __init__(self, root_dir, split_file, transform=None):

        self.root_dir = Path(root_dir) # подразумиваем что это папка подпапки images
        self.transform = transform

        self.images = []
        self.labels = []

        with open(split_file, 'r') as f:
            for line in f:
                image_id, class_id, species, breed_id = line.strip().split()


                img_path = self.root_dir / f"{image_id}.jpg"

                label = int(class_id) - 1 

                self.images.append(img_path)
                self.labels.append(label)
    

    def __len__(self):
        return len(self.labels)

    
    def __getitem__(self, index):
        img_path = self.images[index]

        label = self.labels[index]
        img = Image.open(img_path).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        return img, torch.tensor(label, dtype=torch.long)



class CreaterTrainValDataset:

    def __init__(
            self, 
            images_dir,
            annotaions_dir, 
            transformer_train=None, 
            transformer_val=None
        ):

        self.images_dir = images_dir
        self.annotaions_dir = annotaions_dir
        self.transformer_train = transformer_train
        self.transformer_val = transformer_val
    

    def train_dataset(self):
        return CatDogOxfortDataset(
            root_dir=self.images_dir,
            split_file=Path(self.annotaions_dir) / "trainval.txt",
            transform=self.transformer_train
        )
    

    def val_dataset(self):
        return CatDogOxfortDataset(
            root_dir=self.images_dir,
            split_file=Path(self.annotaions_dir) / "test.txt",
            transform=self.transformer_val
        )


class CreaterDataloader():

    def __init__(self):
        pass

    def create(self, dataset, batch_size, shuffle=True):
        return DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
        )