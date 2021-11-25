import torch
from torch.utils.data import Dataset
from torchvision.datasets.folder import default_loader
import torchvision.transforms.functional as FT

import os
import numpy as np
from math import ceil
from glob import glob
from random import shuffle

class FacesInTheWild300W(Dataset):
    def __init__(self, root, split, mode='indoor+outdoor', transform=None, loader=default_loader, download=False):
        self.root = root
        self.split = split
        self.mode = mode
        self.transform = transform
        self.loader = loader
        images = []
        keypoints = []
        if 'indoor' in mode:
            print('Loading indoor images')
            images += glob(os.path.join(self.root, '01_Indoor', '*.png'))
            keypoints += glob(os.path.join(self.root, '01_Indoor', '*.pts'))
        if 'outdoor' in mode:
            print('Loading outdoor images')
            images += glob(os.path.join(self.root, '02_Outdoor', '*.png'))
            keypoints += glob(os.path.join(self.root, '02_Outdoor', '*.pts'))
        images = list(sorted(images))
        keypoints = list(sorted(keypoints))

        split_path = os.path.join(self.root, f'{mode}_{split}.npy')
        while not os.path.exists(split_path):
            self.generate_dataset_splits(len(images))
        split_idxs = np.load(split_path)
        print(max(split_idxs), len(images), len(keypoints))
        self.images = [images[i] for i in split_idxs]
        self.keypoints = [keypoints[i] for i in split_idxs]

    def generate_dataset_splits(self, l, split_sizes=[0.3, 0.3, 0.4]):
        np.random.seed(0)
        assert sum(split_sizes) == 1
        idxs = np.arange(l)
        np.random.shuffle(idxs)
        split1, split2 = int(l * split_sizes[0]), int(l * sum(split_sizes[:2]))
        train_idx = idxs[:split1]
        valid_idx = idxs[split1:split2]
        test_idx = idxs[split2:]
        print(max(train_idx), max(valid_idx), max(test_idx))
        np.save(os.path.join(self.root, f'{self.mode}_train'), train_idx)
        np.save(os.path.join(self.root, f'{self.mode}_valid'), valid_idx)
        np.save(os.path.join(self.root, f'{self.mode}_test'), test_idx)

    def __getitem__(self, index):
        # get image in original resolution
        path = self.images[index]
        image = self.loader(path)
        h, w = image.height, image.width
        min_side = min(h, w)

        # get keypoints in original resolution
        keypoint = open(self.keypoints[index], 'r').readlines()
        keypoint = keypoint[3:-1]
        keypoint = [s.strip().split(' ') for s in keypoint]
        keypoint = torch.tensor([(float(x), float(y)) for x, y in keypoint])
        bbox_x1, bbox_x2 = keypoint[:, 0].min().item(), keypoint[:, 0].max().item()
        bbox_y1, bbox_y2 = keypoint[:, 1].min().item(), keypoint[:, 1].max().item()
        bbox_width = ceil(bbox_x2 - bbox_x1)
        bbox_height = ceil(bbox_y2 - bbox_y1)
        bbox_length = max(bbox_width, bbox_height)

        image = FT.crop(image, top=bbox_y1, left=bbox_x1, height=bbox_length, width=bbox_length)
        keypoint = torch.tensor([(x - bbox_x1, y - bbox_y1) for x, y in keypoint])
        
        h, w = image.height, image.width
        min_side = min(h, w)

        if self.transform is not None:
            image = self.transform(image)
        new_h, new_w = image.shape[1:]

        keypoint = torch.tensor([[
            ((x - ((w - min_side) / 2)) / min_side) * new_w,
            ((y - ((h - min_side) / 2)) / min_side) * new_h,
        ] for x, y in keypoint])
        keypoint = keypoint.flatten()
        return image, keypoint

    def __len__(self):
        return len(self.images)


if __name__ == '__main__':
    import PIL
    from torchvision import transforms

    image_size = 64

    transform = transforms.Compose([
        transforms.Resize(image_size, PIL.Image.BILINEAR),
        transforms.CenterCrop(image_size),
        transforms.ToTensor()
    ])

    dataset = FacesInTheWild300W('../data/300W', 'train', 'indoor+outdoor', transform=transform)

    print(len(dataset))

    i = 0
    print(dataset[i][1])

    import matplotlib.pyplot as plt
    import numpy as np

    plt.figure(figsize=(8, 8))
    plt.imshow(np.transpose(dataset[i][0], (1, 2, 0)))
    h, w = dataset[0][0].shape[1:]
    plt.scatter(x=dataset[i][1][::2], y=dataset[i][1][1::2])
    plt.axis('off')
    plt.show()
