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
import argparse
from pathlib import Path

import numpy as np
import torch
from torch import nn

from models.baseline.baseline import ShortBaselineModel, DeepBaselineModel

from tools.experiment import Experiment

# %%[markdown]
# Команда чтобы настраивать пути (изображений и анотации) во время запуска консоли

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

# %%[markdown]
### Эксперименты
# Далее производятся эксперименты по моделям

# %%[markdown]
# Для начала рассмотрим какие результаты даст сырая неглубокая сеть
# и глубокая сеть, где они будут без сложной аугоментации и доп слоев

# %%
model = ShortBaselineModel()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
criterion = nn.CrossEntropyLoss()

expirement_1 = Experiment(
    images_dir=images_path,
    annotations_dir=annotations_path
)

expirement_1.setup_transforms()
expirement_1.setup_data()
expirement_1.setup_logger(log_dir="runs/exp_001_vanila_short_baseline")
expirement_1.setup_model(model, optimizer, criterion)
expirement_1.run()

# %%
model = DeepBaselineModel()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
criterion = nn.CrossEntropyLoss()

expirement_1 = Experiment(
    images_dir=images_path,
    annotations_dir=annotations_path
)

expirement_1.setup_transforms()
expirement_1.setup_data()
expirement_1.setup_logger(log_dir="runs/exp_002_vanila_deep_baseline")
expirement_1.setup_model(model, optimizer, criterion)
expirement_1.run()
