import torch
from torch.utils.data import Dataset
from torchvision.datasets.folder import default_loader

import numpy as np
from PIL import Image
from scipy.io import loadmat

import os
from glob import glob


class DaLI(Dataset):
    def __init__(self, root, transform=None, loader=default_loader):
        self.root = root
        self.transform = transform
        self.loader = loader
        self.images = glob(os.path.join(self.root, 'input', 'png', '**', '*.png'))
        self.images = sorted(self.images)
        self.num_classes = 12

    def __getitem__(self, index):
        path = self.images[index]
        image = self.loader(path)
        if self.transform is not None:
            image = self.transform(image)
        return image

    def __len__(self):
        return len(self.images)

    def get_views_of_class(self, class_idx):
        image_idxs = [i for i, filename in enumerate(self.images) if int(filename.split('/')[-2]) == class_idx + 1]
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

    dataset = DaLI('../data/DaLI', transform=transform)
    for i in range(dataset.num_classes):
        batch = dataset.get_views_of_class(i)
        print(i, batch.shape)
        fig, axs = plt.subplots(1, len(batch), figsize=(32, 2))
        for i, ax in enumerate(axs):
            ax.imshow(np.transpose(batch[i], (1, 2, 0)))
        plt.show()
