class DetectionModelRegistry:

    _registry = {}

    @classmethod
    def register(cls, name):
        def decorator(model_cls):
            cls._registry[name] = model_cls
            return model_cls
        return decorator

    @classmethod
    def get(cls, name):
        if name not in cls._registry:
            raise ValueError(f"Unknown detection model: {name}")
        return cls._registry[name]