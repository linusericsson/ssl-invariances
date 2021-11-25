import torch
from torch.utils.data import Dataset
from torchvision.datasets.folder import default_loader


import os
from glob import glob


class COIL100(Dataset):
    def __init__(self, root, transform=None, loader=default_loader):
        self.root = root
        self.transform = transform
        self.loader = loader
        self.images = glob(os.path.join(self.root, '*.png'))
        self.images = sorted(self.images)
        self.num_classes = 100

    def __getitem__(self, index):
        path = self.images[index]
        image = self.loader(path)
        if self.transform is not None:
            image = self.transform(image)
        return image

    def __len__(self):
        return len(self.images)

    def get_views_of_class(self, class_idx):
        image_idxs = [i for i, filename in enumerate(self.images) if int(filename.split('/')[-1].split('__')[0][3:]) == class_idx + 1]
        return torch.stack([self.__getitem__(i) for i in image_idxs])

    
if __name__ == '__main__':
    from torchvision import transforms

    image_size = 224

    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor()
    ])

    dataset = COIL100('../data/COIL100', transform=transform)
    for i in range(dataset.num_classes):
        batch = dataset.get_views_of_class(i)
        print(batch.shape)
