import torch
from torch.utils.data import Dataset
from torchvision.datasets.folder import default_loader

import os
from glob import glob

class RealBlur(Dataset):
    def __init__(self, root, transform=None, loader=default_loader):
        self.root = root
        self.transform = transform
        self.loader = loader
        self.sharp = sorted(glob(os.path.join(self.root, 'RealBlur-J_ECC_IMCORR_centroid_itensity_ref', '**', 'gt', '*.png')))
        self.blur = sorted(glob(os.path.join(self.root, 'RealBlur-J_ECC_IMCORR_centroid_itensity_ref', '**', 'blur', '*.png')))
        self.num_classes = len(self.sharp)
        self.images = self.sharp + self.blur

    def __getitem__(self, index):
        sharp = self.loader(self.images[index])
        blur = self.loader(self.images[index + self.num_classes])
        if self.transform is not None:
            sharp, blur = self.transform(sharp), self.transform(blur)
        return sharp, blur

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

    dataset = RealBlur('../data/RealBlur', transform=transform)

    import matplotlib.pyplot as plt
    import numpy as np
    for i in range(dataset.num_classes):
        images = dataset.get_views_of_class(i)
        print(images.shape)
        fig, axs = plt.subplots(1, 2, figsize=(16, 8))
        axs[0].imshow(np.transpose(images[0], (1, 2, 0)))
        axs[1].imshow(np.transpose(images[1], (1, 2, 0)))
        axs[0].axis('off');
        axs[1].axis('off');
        plt.show()
