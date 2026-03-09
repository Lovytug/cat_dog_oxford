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
import yaml

import numpy as np
import torch
from torch import nn
import torchvision.transforms.v2 as T

from models.baseline.vanila.baseline import ShortBaselineModel, DeepBaselineModel
from models.baseline.upgrade.baseline import BatchDeepBaselineModel, ResidualDeepBaselineModel

from tools.loggers.filter_activ_logger import FilterActivityLogger
from tools.loggers.gradients_logger import GradientsLogger
from tools.loggers.metric_logger import MetricLogger
from tools.loggers.weight_update_logger import WeightUpdateLogger
from tools.loggers.lr_logger import LRLogger
from tools.schedulers import build_scheduler

from experiment.experiment import Experiment
from experiment.experiment_tracker import ExperimentTracker

# %%[markdown]
# Команда чтобы настраивать пути (изображений и анотации) во время запуска консоли

# %%

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images_dir", type=str, required=True)
    parser.add_argument("--annotations_dir", type=str, required=True)
    parser.add_argument("--config", type=str, default="experiments_runs.yaml")
    parser.add_argument("--experiment", type=str, default="all",
                        help="Run specific experiment or 'all'")
    return parser.parse_args()

# %%

def get_model(model_name: str):

    models = {
        "ShortBaselineModel": ShortBaselineModel,
        "DeepBaselineModel": DeepBaselineModel,
        "BatchDeepBaselineModel": BatchDeepBaselineModel,
        "ResidualDeepBaselineModel": ResidualDeepBaselineModel
    }
    if model_name not in models:
        raise ValueError(f"Unknown model: {model_name}")
    return models[model_name]()

# %%
args = parse_args()

with open(args.config, "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

tracker = ExperimentTracker(file_path="result/result.csv")

# %%[markdown]
### Эксперименты
# Далее производятся эксперименты по моделям

# %%[markdown]
# Трансформер для аугментации трейна. Применяется ко всем экпериментам один и тот же

# %%

color_random = T.RandomChoice([
    T.ColorJitter(0.2, 0.2, 0.2, 0.1),
    T.RandomChannelPermutation(),
    T.RandomPhotometricDistort(),
    T.RandomGrayscale(),
])

train_transormer = T.Compose([
    T.RandomRotation(degrees=15),
    color_random,
    T.RandomErasing(
        p=0.15,
        scale=(0.3, 0.3),
        ratio=(0.33, 3.33),
        value=(128, 128, 128))
])

# %%
callbackers = [
    MetricLogger,
    GradientsLogger,
    WeightUpdateLogger,
    FilterActivityLogger,
    LRLogger
]

# %%
for exp_name, exp_config in config["experiments"].items():

    # Пропускаем если не этот эксперимент
    if args.experiment != "all" and args.experiment != exp_name:
        continue
    
    # Пропускаем если run: false
    if not exp_config.get("run", True):
        print(f"Пропускаем {exp_name}")
        continue
    
    print(f"\nОсуществляется запуск: {exp_name}")
    

    model = get_model(exp_config["model"])
    
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=exp_config["lr"],
        weight_decay=exp_config.get("weight_decay", 0.0)
    )

    scheduler = build_scheduler(
        optimizer,
        exp_config.get("scheduler")
    )

    criterion = nn.CrossEntropyLoss()

    experiment = Experiment(
        images_dir=Path(args.images_dir),
        annotations_dir=Path(args.annotations_dir)
    )
    
    experiment.setup_transforms(train_transormer)
    experiment.setup_data()
    experiment.setup_logger(callbackers=callbackers, log_dir=exp_config["log_dir"])
    experiment.setup_model(model, optimizer, criterion, scheduler)
    
    result = experiment.run(epochs=exp_config["epochs"])
    tracker.save(result)
    
    print(f"{exp_name} завершена!")
