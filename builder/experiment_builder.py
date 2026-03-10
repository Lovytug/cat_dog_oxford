from builder.model_factory import ModelFactory
from builder.optimizer_factory import OptimizerFactory
from builder.scheduler_factory import SchedulerFactory
from builder.transform_factory import TransformFactory
from builder.callback_factory import CallbackFactory


class ExperimentBuilder:

    def __init__(self, config: dict):
        self.config = config


    def build(self):

        model = ModelFactory.build(
            self.config["model"]
        )

        optimizer = OptimizerFactory.build(
            model,
            self.config["optimizer"]
        )

        scheduler = SchedulerFactory.build(
            optimizer,
            self.config.get("scheduler")
        )

        train_tf, val_tf = self._build_transforms()

        callbacks = self._build_callbacks()

        batch_size = self._get_batch_size()

        return {
            "model": model,
            "optimizer": optimizer,
            "scheduler": scheduler,
            "train_transform": train_tf,
            "val_transform": val_tf,
            "callbacks": callbacks,
            "batch_size": batch_size
        }


    def _build_transforms(self):

        tf_config = self.config.get("transforms", {})

        train_tf = TransformFactory.build_pipeline(
            tf_config.get("train", [])
        )

        val_tf = TransformFactory.build_pipeline(
            tf_config.get("val", [])
        )

        return train_tf, val_tf


    def _build_callbacks(self):

        cb_config = self.config.get("callbacks", [])

        return CallbackFactory.build(cb_config)


    def _get_batch_size(self):

        batch_size = self.config.get("batch_size", 32)

        if batch_size <= 0:
            raise ValueError("batch_size должен быть > 0")

        if batch_size > 1024:
            raise ValueError("batch_size выглядит слишком большим")

        return batch_size