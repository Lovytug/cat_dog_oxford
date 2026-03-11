import torch

from models.baseline.upgrade.baseline import BatchDeepBaselineModel, ResidualDeepBaselineModel
from models.baseline.vanila.baseline import DeepBaselineModel, ShortBaselineModel
from models.baseline.upgrade.bacth_deep import BatchDeepNewStartFiltersBaselineModel, BatchDeepNewEndFilterBaselineModel
from models.res_net.res_net50 import ResNetTransfer
from builder.experiment_builder import ExperimentBuilder
from experiment.experiment import Experiment


class ExperimentRunner:

    def __init__(self, config, images_dir, annotations_dir):

        self.config = config
        self.images_dir = images_dir
        self.annotations_dir = annotations_dir

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )


    def run(self, experiment_name=None):

        experiments = self.config["experiments"]

        for name, exp_config in experiments.items():

            if experiment_name != "all" and name != experiment_name:
                continue

            if not exp_config.get("run", True):
                print(f"Пропускаем {name}")
                continue

            self._run_single(name, exp_config)


    def _run_single(self, name, config):

        print(f"\nЗапущена: {name}")

        builder = ExperimentBuilder(config)

        objects = builder.build()

        experiment = Experiment(
            images_dir=self.images_dir,
            annotations_dir=self.annotations_dir,
            batch_size=objects["batch_size"],
            device=self.device
        )

        experiment.setup_transforms(
            train_tf=objects["train_transform"],
            val_tf=objects["val_transform"]
        )

        experiment.setup_data()

        experiment.setup_logger(
            callbackers=objects["callbacks"],
            log_dir=config["log_dir"]
        )

        experiment.setup_model(
            model=objects["model"],
            optimizer=objects["optimizer"],
            scheduler=objects["scheduler"]
        )

        experiment.run(
            epochs=config["epochs"]
        )

        print(f"{name} закончена\n")