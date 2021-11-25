import torch
from torch.utils.data import Dataset
from torchvision.datasets.folder import default_loader

import os
from glob import glob

class Flickr1024(Dataset):
    def __init__(self, root, transform=None, loader=default_loader):
        self.root = root
        self.transform = transform
        self.loader = loader
        self.images = glob(os.path.join(self.root, '**', '*_L.png'))
        self.num_classes = len(self.images)

    def __getitem__(self, index):
        path = self.images[index]
        left = self.loader(path)
        right = self.loader(path[:-5] + 'R.png')
        if self.transform is not None:
            left, right = self.transform(left), self.transform(right)
        return left, right

    def __len__(self):
        return len(self.images)

    def get_views_of_class(self, class_idx):
        return torch.stack(list(self.__getitem__(class_idx)))

if __name__ == '__main__':
    from torchvision import transforms

    image_size = 224

    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor()
    ])

    dataset = Flickr1024('../data/Flickr1024', transform=transform)

    len(dataset)

    import matplotlib.pyplot as plt
    import numpy as np

    fig, axs = plt.subplots(1, 2, figsize=(8, 6))
    axs[0].imshow(np.transpose(dataset[0][0], (1, 2, 0)))
    axs[1].imshow(np.transpose(dataset[0][1], (1, 2, 0)))
    axs[0].axis('off')
    axs[1].axis('off');
    plt.show()
