import torch
from torch.utils.data import Dataset
from torchvision.datasets.folder import default_loader


import os
from glob import glob


class ExposureErrors(Dataset):
    def __init__(self, root, transform=None, loader=default_loader):
        self.root = root
        self.transform = transform
        self.loader = loader
        self.exposure_images = glob(os.path.join(self.root, 'INPUT_IMAGES', '*.JPG'))
        self.gt_images = glob(os.path.join(self.root, 'GT_IMAGES', '*.jpg'))
        self.num_classes = len(self.gt_images)
        self.images = self.gt_images + self.exposure_images

    def __getitem__(self, index):
        path = self.images[index]
        image = self.loader(path)
        if self.transform is not None:
            image = self.transform(image)
        return image

    def __len__(self):
        return len(self.images)

    def get_views_of_class(self, index):
        path = self.gt_images[index]
        image_id = path.split('/')[-1].split('-')[0]
        image_idxs = [index]
        for i, path in enumerate(self.exposure_images):
            if image_id in path:
                image_idxs.append(i + self.num_classes)
        return torch.stack([self.__getitem__(i) for i in image_idxs])

    
if __name__ == '__main__':
    from torchvision import transforms
    import matplotlib.pyplot as plt
    import numpy as np

    image_size = 224

    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor()
    ])

    dataset = ExposureErrors('../data/ExposureErrors', transform=transform)
    for i in range(dataset.num_classes):
        batch = dataset.get_views_of_class(i)
        print(i, batch.shape)
        fig, axs = plt.subplots(1, len(batch), figsize=(15, 3))
        for i, ax in enumerate(axs):
            ax.imshow(np.transpose(batch[i], (1, 2, 0)))
        plt.show()
