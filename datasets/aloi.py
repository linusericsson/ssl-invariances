import torch
from torch.utils.data import Dataset
from torchvision.datasets.folder import default_loader


import os
from glob import glob

# viewpoint, illumination, temperature

class ALOI(Dataset):
    def __init__(self, root, transform=None, loader=default_loader, mode='viewpoint', camera=None):
        self.root = root
        self.transform = transform
        self.loader = loader
        self.images = glob(os.path.join(self.root, mode, '**', '*.png'))
        self.images = sorted(self.images)
        self.num_classes = 3000 if mode == 'illumination' else 1000
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
        if self.mode == 'illumination':
            camera = (class_idx // 1000)
            class_idx = class_idx % 1000
            image_idxs = [i for i, filename in enumerate(self.images)
                          if int(filename.split('/')[-2]) == class_idx + 1 and int(filename.split('/')[-1][-5:-4]) == camera + 1]
        else:
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

    mode = 'illumination' # viewpoint: {camera: None}, illumination: {camera: 1}, temperature: {camera: None}
    camera = None
    dataset = ALOI('../data/ALOI', transform=transform, mode=mode, camera=camera)
    for i in range(dataset.num_classes):
        batch = dataset.get_views_of_class(i)
        print(i, batch.shape)
        #for x in batch:
        #    plt.imshow(np.transpose(x, (1, 2, 0)))
        #    plt.show()
        fig, axs = plt.subplots(1, len(batch), figsize=(32, 2))
        for i, ax in enumerate(axs):
            ax.imshow(np.transpose(batch[i], (1, 2, 0)))
        plt.show()
