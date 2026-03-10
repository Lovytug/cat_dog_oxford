import torchvision.transforms.v2 as T


class TransformFactory:

    registry = {

        "random_rotation": T.RandomRotation,
        "color_jitter": T.ColorJitter,
        "random_grayscale": T.RandomGrayscale,
        "random_erasing": T.RandomErasing,
        "random_photometric_distort": T.RandomPhotometricDistort,
        "channel_permutation": T.RandomChannelPermutation
    }

    @classmethod
    def build_pipeline(cls, config_list):
        
        augmentations = []
        for item in config_list:

            name = list(item.keys())[0]
            params = item[name] or {}

            if name not in cls.registry:
                raise ValueError(f"Неизвестный Трансформер {name}")
            
            augmentations.append(cls.registry[name](**params))

        return augmentations