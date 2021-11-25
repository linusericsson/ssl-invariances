import torch
from torch.utils.data import Dataset
from torchvision.datasets.folder import default_loader


import os
from glob import glob

# viewpoint, illumination, temperature

class ALOT(Dataset):
    def __init__(self, root, transform=None, loader=default_loader, mode='viewpoint', camera=None):
        self.root = root
        self.transform = transform
        self.loader = loader
        self.images = glob(os.path.join(self.root, '**', '*.png'))
        self.images = sorted(self.images)
        if mode == 'viewpoint':
            self.num_classes = 1500
        if mode == 'illumination':
            self.num_classes = 4000
        if mode == 'temperature':
            self.num_classes = 1000
        self.mode = mode
        self.camera = camera

    def __getitem__(self, index):
        path = self.images[index]
        image = self.loader(path)
        if self.transform is not None:
            image = self.transform(image)
        return image

    def __len__(self):
        return len(self.images)

    def get_views_of_class(self, class_idx):
        if self.mode == 'viewpoint':
            class_idx = class_idx % 250
            light = class_idx % 6
            l = [1, 2, 3, 4, 5, 8][light]
            image_idxs = []
            for i, filename in enumerate(self.images):
                if int(filename.split('/')[-2]) == class_idx + 1:
                    if 'i' not in filename.split('/')[-1]:
                        if int(filename.split('/')[-1].split('l')[1][0]) == l:
                            image_idxs.append(i)
        if self.mode == 'illumination':
            camera = ((class_idx % 1000) // 250)
            angle = (class_idx // 1000)
            class_idx = class_idx % 250
            image_idxs = []
            for i, filename in enumerate(self.images):
                if int(filename.split('/')[-2]) == class_idx + 1:
                    if int(filename.split('/')[-1].split('_')[1][1:2]) == camera + 1:
                        if 'r' in filename.split('/')[-1]:
                            a = int(filename.split('/')[-1].split('r')[1].split('.')[0])
                        else:
                            a = 0
                        if a == angle * 60:
                            if 'i' not in filename.split('/')[-1]:
                                image_idxs.append(i)
        if self.mode == 'temperature':
            camera = (class_idx // 250)
            class_idx = class_idx % 250
            image_idxs = []
            for i, filename in enumerate(self.images):
                if int(filename.split('/')[-2]) == class_idx + 1:
                    if int(filename.split('/')[-1].split('_')[1][1:2]) == camera + 1:
                        if 'i.' in filename.split('/')[-1].split('_')[1] or 'l8.' in filename.split('/')[-1].split('_')[1]:
                            image_idxs.append(i)
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

    camera = None
    mode = 'temperature'
    dataset = ALOT('../data/ALOT', transform=transform, mode=mode, camera=camera)
    for i in range(dataset.num_classes):
        batch = dataset.get_views_of_class(i)
        print(i, batch.shape)
        fig, axs = plt.subplots(1, len(batch), figsize=(32, 2))
        for i, ax in enumerate(axs):
            ax.imshow(np.transpose(batch[i], (1, 2, 0)))
        plt.show()
