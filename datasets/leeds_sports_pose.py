import torch
from torch.utils.data import Dataset
from torchvision.datasets.folder import default_loader
import torchvision.transforms.functional as FT

import os
import numpy as np
from math import ceil
from glob import glob
from random import shuffle
from scipy.io import loadmat

class LeedsSportsPose(Dataset):
    def __init__(self, root, split, transform=None, loader=default_loader, download=False):
        self.root = root
        self.split = split
        self.transform = transform
        self.loader = loader
        images = glob(os.path.join(self.root, 'images', '*.jpg'))
        images = sorted(images)
        joints = loadmat(os.path.join(self.root, 'joints.mat'))['joints']
        joints = np.array([[(joints[0, j, i], joints[1, j, i], joints[2, j, i]) for j in range(joints.shape[1])] for i in range(joints.shape[2])])

        split_path = os.path.join(self.root, f'{split}.npy')
        while not os.path.exists(split_path):
            self.generate_dataset_splits(len(images))
        split_idxs = np.load(split_path)
        print(max(split_idxs), len(images), len(joints))
        self.images = [images[i] for i in split_idxs]
        self.joints = [joints[i] for i in split_idxs]

    def generate_dataset_splits(self, l, split_sizes=[0.6, 0.4]):
        np.random.seed(0)
        assert sum(split_sizes) == 1
        idxs = np.arange(l)
        np.random.shuffle(idxs)
        split1 = int(l * split_sizes[0])
        train_idx = idxs[:split1]
        test_idx = idxs[split1:]
        print(max(train_idx), max(test_idx))
        np.save(os.path.join(self.root, 'train'), train_idx)
        np.save(os.path.join(self.root, 'test'), test_idx)

    def __getitem__(self, index):
        # get image in original resolution
        path = self.images[index]
        image = self.loader(path)
        h, w = image.height, image.width
        min_side = min(h, w)

        # get keypoints in original resolution
        joints = self.joints[index]

        bbox_x1 = int((w - min_side) / 2) if w >= min_side else 0
        bbox_x2 = bbox_x1 + min_side
        bbox_y1 = int((h - min_side) / 2) if h >= min_side else 0
        bbox_y2 = bbox_y1 + min_side

        image = FT.crop(image, top=bbox_y1, left=bbox_x1, height=min_side, width=min_side)
        joints = torch.tensor([(x - bbox_x1, y - bbox_y1) for x, y, _ in joints])
        
        h, w = image.height, image.width
        min_side = min(h, w)

        if self.transform is not None:
            image = self.transform(image)
        new_h, new_w = image.shape[1:]

        joints = torch.tensor([[
            ((x - ((w - min_side) / 2)) / min_side) * new_w,
            ((y - ((h - min_side) / 2)) / min_side) * new_h,
        ] for x, y in joints])
        joints = joints.flatten()

        return image, joints

    def __len__(self):
        return len(self.images)


if __name__ == '__main__':
    import PIL
    from torchvision import transforms

    image_size = 224

    transform = transforms.Compose([
        transforms.Resize(image_size, PIL.Image.BILINEAR),
        transforms.CenterCrop(image_size),
        transforms.ToTensor()
    ])

    dataset = LeedsSportsPose('../data/LeedsSportsPose', 'train', transform=transform)

    print(len(dataset))

    import matplotlib.pyplot as plt
    import numpy as np

    for i in range(len(dataset)):
        image, joints = dataset[i]

        plt.figure(figsize=(8, 8))
        plt.imshow(np.transpose(image, (1, 2, 0)))
        plt.scatter(x=joints[::2], y=joints[1::2])
        plt.axis('off')
        plt.show()
