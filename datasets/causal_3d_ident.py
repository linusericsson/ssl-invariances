import os
import numpy as np
from torchvision import datasets, transforms

class Causal3DIdent(datasets.ImageFolder):
    # 3600 images over 7 classes
    # teapot, hare, dragon, cow, armadillo, horse, head
    # latents
    # (obj pos X, Y, Z), (obj rot X, Y, Z), (light angle), (obj hue), (light hue), (bg hue)
    def __init__(self, root, split, transform, target_type='class'):
        split_path = os.path.join(root, split + 'set')
        super().__init__(split_path, transform=transform)
        self.target_type = target_type
        self.num_classes = 7
        self.latents = np.concatenate([np.load(os.path.join(split_path, f'latents_{i}.npy')) for i in range(self.num_classes)])

    def __getitem__(self, i):
        img, class_label = super().__getitem__(i)
        latent = self.latents[i]
        target = class_label if self.target_type == 'class' else latent[int(self.target_type)]
        return img, target

if __name__ == '__main__':
    train_set = Causal3DIdent('../../data/Causal3DIdent', 'train', transform=transforms.ToTensor(), target_type='latents')
    test_set = Causal3DIdent('../../data/Causal3DIdent', 'test', transform=transforms.ToTensor(), target_type='latents')
    print(len(train_set), len(test_set))

    import matplotlib.pyplot as plt
    plt.imshow(np.transpose(test_set[0][0], (1, 2, 0)))
    plt.show()
