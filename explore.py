# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%

import numpy as np

import torch
import torch.nn.functional as F
from torch import nn

from torchvision import transforms

# %%

from tools.dataset_dataloader import CreaterTrainValDataset, CreaterDataloader
from models.baseline import BaselineModel
from tools.trainer import ModelTrainer
from tools.tblogger import TBLogger
from util.util import PROJECT_ROOT

# %%

import argparse
from pathlib import Path

# %%

def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--images_dir", type=str, required=True,
                        help="Path to images directory")
    parser.add_argument("--annotations_dir", type=str, required=True,
                        help="Path to annotations directory")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--log_dir", type=str, default="runs/exp")

    return parser.parse_args()

# %%
args = parse_args()

images_path = Path(args.images_dir)
annotations_path = Path(args.annotations_dir)

# %%

size_img = (224, 224)

train_transform = transforms.Compose([
    transforms.Resize(size_img),
    transforms.ToTensor()
])

val_transform = transforms.Compose([
    transforms.Resize(size_img),
    transforms.ToTensor()
])
    
# %%

creater_dataset = CreaterTrainValDataset(
    images_dir=images_path,
    annotaions_dir=annotations_path,
    transformer_val=val_transform,
    transformer_train=train_transform
)

train_ds = creater_dataset.train_dataset()
val_ds = creater_dataset.val_dataset()

# %%

creater_dataloader = CreaterDataloader()

train_loader = creater_dataloader.create(train_ds, batch_size=32)
val_loader = creater_dataloader.create(val_ds, batch_size=32)

# %%

model = BaselineModel()
optimizer = torch.optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss()

# %%
logger = TBLogger(
    log_dir="runs/exp_test"
)

trainer = ModelTrainer(
    model=model,
    optimizer=optimizer,
    criterion=criterion,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    logger=logger
)

trainer.train(
    num_epochs=5,
    train_loader=train_loader,
    val_loader=val_loader
)

logger.close()
