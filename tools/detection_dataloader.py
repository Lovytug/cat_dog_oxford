from torch.utils.data import DataLoader


def detection_collate_fn(batch):
    images, targets = zip(*batch)

    return list(images), list(targets)


class DetectionDataLoaderCreator:

    def create(self, dataset, batch_size, shuffle=True):
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=detection_collate_fn
        )