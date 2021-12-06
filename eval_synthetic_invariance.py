#!/usr/bin/env python
# coding: utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
import torchvision.transforms.functional as FT

import os
import PIL
import pickle
import argparse
from glob import glob
from tqdm import tqdm
from pathlib import Path
from itertools import product
from collections import OrderedDict

import numpy as np
import albumentations
from scipy.spatial.distance import mahalanobis

from datasets import Flowers, Caltech101

import model_utils


def deform(img, points, sigma):
    # convert to numpy
    img = np.array(img)
    # apply deformation
    img = albumentations.geometric.transforms.ElasticTransform(sigma=sigma, always_apply=True, approximate=True)(image=img)['image']
    # return PIL image
    return PIL.Image.fromarray(img)

FT.deform = deform


class ManualTransform(object):
    def __init__(self, name, k, norm, resize=256, crop_size=224):
        self.norm = norm
        name_to_fn = {
            'rotation': 'rotate',
            'translation': 'affine',
            'scale': 'affine',
            'shear': 'affine',
            'resized_crop': 'resized_crop',
            'h_flip': 'hflip',
            'v_flip': 'vflip',
            'deform': 'deform',
            'grayscale': 'rgb_to_grayscale',
            'brightness': 'adjust_brightness',
            'contrast': 'adjust_contrast',
            'saturation': 'adjust_saturation',
            'hue': 'adjust_hue',
            'blur': 'gaussian_blur',
            'sharpness': 'adjust_sharpness',
            'invert': 'invert',
            'equalize': 'equalize',
            'posterize': 'posterize',
        }
        self.fn = name_to_fn[name]
        self.k = k
        self.resize = resize
        self.crop_size = crop_size
        if name == 'rotation':
            self.param_keys = ['angle']
            self.param_vals = [torch.linspace(0, 360, self.k + 1)[:self.k].to(torch.float32).tolist()]
            self.original_idx = 0
        elif name == 'translation':
            self.param_keys = ['translate', 'angle', 'scale', 'shear']
            space = (1 - (crop_size / resize)) / 2
            a = torch.linspace(-space, space, int(np.sqrt(self.k)) + 1)[:int(np.sqrt(self.k))].to(torch.float32)
            translate_params = [(float(a * resize), float(b * resize)) for a, b in product(a, a)]
            self.param_vals = [
                translate_params,
                torch.zeros(self.k).tolist(),
                torch.ones(self.k).tolist(),
                torch.ones(self.k).tolist()
            ]
            self.original_idx = translate_params.index((0.0, 0.0))
        elif name == 'scale':
            self.param_keys = ['scale', 'translate', 'angle', 'shear']
            self.param_vals = [
                torch.linspace(1 / 4, 2, self.k).to(torch.float32).tolist(),
                torch.zeros((self.k, 2)).tolist(),
                torch.zeros(self.k).tolist(),
                torch.ones(self.k).tolist()
            ]
            self.original_idx = 0
        elif name == 'shear':
            self.param_keys = ['shear', 'translate', 'angle', 'scale']
            a = torch.linspace(-160, 160, int(np.sqrt(self.k)) + 1)[:int(np.sqrt(self.k))].to(torch.float32).tolist()
            shear_params = [(a, b) for a, b in product(a, a)]
            self.param_vals = [
                shear_params,
                torch.zeros((self.k, 2)).tolist(),
                torch.zeros(self.k).tolist(),
                torch.ones(self.k).tolist()
            ]
            self.original_idx = shear_params.index((0.0, 0.0))
        elif name == 'resized_crop':
            self.param_keys = ['top', 'left', 'height', 'width', 'size']
            n = int(np.sqrt(np.sqrt(self.k)))
            a = (torch.linspace(0, 0.25, n) * resize).to(torch.float32).tolist()
            b = (torch.linspace(0.75, 0.25, n) * resize).to(torch.float32).tolist()
            p = product(a, a, b, b)
            a, b, c, d = tuple(zip(*p))
            self.param_vals = [
                a, b, c, d,
                [(s.item(), s.item()) for s in torch.ones(self.k, dtype=int) * crop_size]
            ]
            self.original_idx = 0
        elif name in ['h_flip', 'v_flip']:
            self.param_keys = ['aug']
            self.param_vals = [[False, True]]
            self.original_idx = 0
        elif name == 'deform':
            torch.manual_seed(0)
            np.random.seed(0)
            self.param_keys = ['points', 'sigma'] # 10, 50, 3, 9
            points = torch.repeat_interleave(torch.arange(2, 10), 32).tolist()
            sigma = torch.linspace(10, 50, 8).to(int).repeat(32).tolist()
            self.param_vals = [
                points,
                sigma
            ]
            self.original_idx = 0
        elif name == 'grayscale':
            self.param_keys = ['aug', 'num_output_channels']
            self.param_vals = [[False, True], [3, 3]]
            self.original_idx = 0
        elif name == 'brightness':
            self.param_keys = ['brightness_factor']
            self.param_vals = [torch.linspace(0.25, 5, self.k).to(torch.float32).tolist()]
            self.original_idx = self.k // 2
        elif name == 'contrast':
            self.param_keys = ['contrast_factor']
            self.param_vals = [torch.linspace(0.25, 5, self.k).to(torch.float32).tolist()]
            self.original_idx = self.k // 2
        elif name == 'saturation':
            self.param_keys = ['saturation_factor']
            self.param_vals = [torch.linspace(0.25, 5, self.k).to(torch.float32).tolist()]
            self.original_idx = self.k // 2
        elif name == 'hue':
            self.param_keys = ['hue_factor']
            self.param_vals = [torch.linspace(-0.5, 0.5, self.k).to(torch.float32).tolist()]
            self.original_idx = self.k // 2
        elif name == 'blur':
            self.param_keys = ['sigma', 'kernel_size']
            self.param_vals = [
                torch.linspace(1e-5, 20.0, self.k).to(torch.float32).tolist(),
                (torch.ones(self.k).to(int) + (crop_size // 20 * 2)).tolist(),
            ]
            self.original_idx = 0
        elif name == 'sharpness':
            self.param_keys = ['sharpness_factor']
            self.param_vals = [torch.linspace(1, 30.0, self.k).to(torch.float32).tolist()]
            self.original_idx = 0
        elif name in ['invert', 'equalize']:
            self.param_keys = ['aug']
            self.param_vals = [[False, True]]
            self.original_idx = 0
        elif name == 'posterize':
            self.param_keys = ['bits']
            self.param_vals = [torch.arange(1, 8).tolist()]
            self.original_idx = 0

    def T(self, image, **params):
        if 'aug' in params:
            if params['aug']:
                del params['aug']
                image = eval(f'FT.{self.fn}(image, **params)')
        elif self.fn == 'translation':
            pass
        else:
            image = eval(f'FT.{self.fn}(image, **params)')
        if self.fn != 'resized_crop':
            image = FT.resize(image, self.resize)
            if self.fn == 'translation':
                image = eval(f'FT.{self.fn}(image, **params)')
            image = FT.center_crop(image, self.crop_size)
        image = FT.pil_to_tensor(image).to(torch.float32)
        image = FT.normalize(image / 255., *self.norm)
        return image

    def __call__(self, x):
        xs = []
        for i in range(self.k):
            params = dict([(k, v[i]) for k, v in zip(self.param_keys, self.param_vals)])
            xs.append(self.T(x, **params))
        return tuple(xs)


def distance(a, b, normalize=False):
    if normalize:
        a = F.normalize(a, dim=0)
        b = F.normalize(b, dim=0)
        return torch.dist(a, b, p=2)
    else:
        return torch.dist(a, b, p=2)


def D(a, b): # cosine similarity
    return F.cosine_similarity(a, b, dim=-1).mean()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('data', type=str, metavar='DIR',
                        help='path to dataset')
    parser.add_argument('--dataset', default='imagenet', type=str, metavar='D',
                        help='dataset to train on (imagenet/coco)')
    parser.add_argument('--split', default='val', type=str, metavar='S',
                        help='dataset split to train on (train/val)')
    parser.add_argument('--model', default='default', type=str, metavar='M',
                        help='model to evaluate invariance of (random/supervised/default/ventral/dorsal)')
    parser.add_argument('--mean-embedding', default='', type=str,
                        help='whether to use mean-embedding model (ventral/dorsal)')
    parser.add_argument('--mean-embedding-k', default=32, type=int,
                        help='number of samples in mean-embedding model (default: 256)')
    parser.add_argument('--transform', default='rotation', type=str, metavar='T',
                        help='transform to evaluate invariance of (rotation/translation/colour jitter/blur etc.)')
    parser.add_argument('--device', default='cuda:0', type=str, metavar='D',
                        help='GPU device')
    parser.add_argument('--feature-layer', default='backbone', type=str, metavar='F',
                        help='layer to extract features from (default: backbone)')
    parser.add_argument('--batch-size', default=256, type=int, metavar='N',
                        help='mini-batch size')
    parser.add_argument('--num-images', default=1000, type=int, metavar='N',
                        help='number of images to evaluate invariance on.')
    parser.add_argument('--resize', default=256, type=int, metavar='R',
                        help='resize')
    parser.add_argument('--crop-size', default=224, type=int, metavar='C',
                        help='crop size')
    parser.add_argument('--k', default=None, type=int,
                        help='number of transformations')
    parser.add_argument('--ckpt-dir', default='models/', type=Path,
                        metavar='DIR', help='path to checkpoint directory')
    parser.add_argument('--results-dir', default='./results/', type=Path,
                        metavar='DIR', help='path to results directory')
    args = parser.parse_args()

    model = model_utils.load_model(args.model, args)
    model = model.to(args.device)

    clean_transform = transforms.Compose([
        transforms.Resize(args.resize),
        transforms.CenterCrop(args.crop_size),
        transforms.ToTensor(),
        transforms.Normalize(*model_utils.imagenet_mean_std)
    ])

    data_path = args.data[:-1] if '\r' in args.data else args.data
    split_path = os.path.join(data_path, args.split)

    if os.path.exists(f"results/{args.model}_{args.feature_layer}_feature_cov_matrix.pth"):
        print(f"Found precomputed covariance matrix for {args.model}, skipping it.")
    else:
        print(f"Computing covariance matrix for {args.model}.")
        dataset = datasets.ImageFolder(split_path, transform=clean_transform)
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

        all_features = []
        with torch.no_grad():
            progress = tqdm(data_loader)
            for i, (data, _) in enumerate(progress):
                data = data.to(args.device)
                features = model(data).detach().cpu()
                all_features.append(features)
        all_features = torch.cat(all_features)

        mean_feature = all_features.mean(dim=0)
        cov_matrix = np.cov(all_features, rowvar=False)

        torch.save(mean_feature, open(f"{args.results_dir}/{args.model}_{args.feature_layer}_mean_feature.pth", 'wb'))
        torch.save(cov_matrix, open(f"{args.results_dir}/{args.model}_{args.feature_layer}_feature_cov_matrix.pth", 'wb'))


    if args.k is not None:
        k = args.k
    elif args.transform in ['h_flip', 'v_flip', 'grayscale', 'invert', 'equalize']:
        k = 2
    elif args.transform == 'posterize':
        k = 7
    else:
        k = 256

    transform = ManualTransform(args.transform, k, norm=model_utils.imagenet_mean_std, resize=args.resize, crop_size=args.crop_size)

    dataset = datasets.ImageFolder(split_path, transform=transform)
    clean_dataset = datasets.ImageFolder(split_path, transform=clean_transform)

    np.random.seed(0)
    torch.manual_seed(0)
    sampler = np.random.choice(np.arange(len(dataset)), args.num_images)

    def get_same_batch(sampler, d1, d2):
        for i in sampler:
            img1, _ = d1[i]
            img2, _ = d2[i]
            yield (torch.stack(img1), img2.unsqueeze(0))

    batch_generator = get_same_batch(sampler, dataset, clean_dataset)

    mean_feature = torch.load(open(f"{args.results_dir}/{args.model}_{args.feature_layer}_mean_feature.pth", 'rb'))
    cov_matrix = torch.load(open(f"{args.results_dir}/{args.model}_{args.feature_layer}_feature_cov_matrix.pth", 'rb'))
    if args.model == 'random':
        cov_matrix = cov_matrix + 1e-10 * np.eye(cov_matrix.shape[0])
    inv_cov_matrix = np.linalg.inv(cov_matrix)
    cholesky_matrix = torch.linalg.cholesky(torch.from_numpy(inv_cov_matrix).to(torch.float32))

    L = torch.zeros((args.num_images, k))
    S = torch.zeros((args.num_images, k))

    with torch.no_grad():
        for i in tqdm(range(args.num_images)):
            data, clean_data = next(batch_generator)
            clean_feature = model(clean_data.to(args.device)).detach()
            features = model(data.to(args.device)).detach()

            a = (mean_feature - clean_feature.cpu()) @ cholesky_matrix

            for j in range(features.shape[0]):
                b = (mean_feature - features[j].cpu()) @ cholesky_matrix
                S[i, j] = D(a, b) # cosine similarity
                L[i, j] = mahalanobis(clean_feature.cpu(), features[j].cpu(), inv_cov_matrix) # mahalanobis distance

    L = torch.from_numpy(np.nanmean(L, axis=0))
    S = torch.from_numpy(np.nanmean(S, axis=0))
    print(f"{args.model} on {args.transform}:")
    print(f"\t distance {L.mean():.6f} and similarity {S.mean():.6f}")

    if args.mean_embedding:
        model_name = args.model + '_m_e_' + args.mean_embedding
    else:
        model_name = args.model

    if args.dataset == 'imagenet':
        dataset_name = ''
    else:
        dataset_name = '_' + args.dataset

    torch.save(L, open(f"{args.results_dir}/{model_name}_{args.feature_layer}{dataset_name}_{args.transform}_invariance_distance.pth", 'wb'))
    torch.save(S, open(f"{args.results_dir}/{model_name}_{args.feature_layer}{dataset_name}_{args.transform}_invariance_similarity.pth", 'wb'))
