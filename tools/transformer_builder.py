import torchvision.transforms.v2 as T

class TransformBuilder:
    
    def __init__(self, size_img, mean, std):
        self.size_img = size_img
        self.mean = mean
        self.std = std


    def build_train(self, augmentations=None):
        augmentations = augmentations or []
        return T.Compose([
            T.RandomResizedCrop(self.size_img),
            T.RandomHorizontalFlip(),
            *augmentations,
            T.ToTensor(),
            T.Normalize(self.mean, self.std),
        ])
    

    def build_val(self):
        return T.Compose([
            T.Resize(256),
            T.CenterCrop(self.size_img),
            T.ToTensor(),
            T.Normalize(self.mean, self.std),
        ])
    

    