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

from models.baseline.baseline import ShortBaselineModel, DeepBaselineModel

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

# %%
for exp_name, exp_config in config["experiments"].items():

    # Пропускаем если не этот эксперимент
    if args.experiment != "all" and args.experiment != exp_name:
        continue
    
    # Пропускаем если run: false
    if not exp_config.get("run", True):
        print(f"⏭️  Пропускаем {exp_name}")
        continue
    
    print(f"\n🔬 Осуществляется запуск: {exp_name}")
    

    model = get_model(exp_config["model"])
    
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=exp_config["lr"],
        weight_decay=exp_config.get("weight_decay", 0.0)
    )
    
    criterion = nn.CrossEntropyLoss()

    experiment = Experiment(
        images_dir=Path(args.images_dir),
        annotations_dir=Path(args.annotations_dir)
    )
    
    experiment.setup_transforms()
    experiment.setup_data()
    experiment.setup_logger(log_dir=exp_config["log_dir"])
    experiment.setup_model(model, optimizer, criterion)
    
    result = experiment.run(epochs=exp_config["epochs"])
    tracker.save(result)
    
    print(f"✅ {exp_name} завершена!")
