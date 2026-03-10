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


from experiment.runner import ExperimentRunner

# %%[markdown]
# Команда чтобы настраивать пути (изображений и анотации) во время запуска консоли

# %%

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images_dir", type=str, required=False, default="./images")
    parser.add_argument("--annotations_dir", type=str, required=False, default="./annotations")
    parser.add_argument("--config", type=str, default="experiments_runs.yaml")
    parser.add_argument("--experiment", type=str, default="all",
                        help="Run specific experiment or 'all'")
    return parser.parse_args()

# %%

# %%
def main():
    args = parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    runner = ExperimentRunner(
        config=config,
        images_dir=args.images_dir,
        annotations_dir=args.annotations_dir
    )

    runner.run(args.experiment)

# %%

if __name__ == "__main__":
    main()
