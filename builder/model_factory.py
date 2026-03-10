from models.registry import ModelRegistry


class ModelFactory:

    @staticmethod
    def build(config):

        name = config["name"]

        model_cls = ModelRegistry.get(name)

        params = config.get("params", {})

        return model_cls(**params)