from torchvision import transforms


class TransformBuilder:
    
    def __init__(self, size_img, mean, std):
        self.size_img = size_img
        self.mean = mean
        self.std = std


    def build_train(self, augmentations=None):
        augmentations = augmentations or []
        return transforms.Compose([
            transforms.Resize(self.size_img),
            *augmentations,
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std),
        ])
    

    def build_val(self):
        return transforms.Compose([
            transforms.Resize(self.size_img),
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std),
        ])
    

    