#!/usr/bin/env python
# coding: utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models

import argparse
import numpy as np
from tqdm import tqdm
from glob import glob
from pathlib import Path

from datasets import COIL100, ALOI, ALOT, ExposureErrors, DaLI, Flickr1024, RealBlur

import model_utils

from scipy.spatial.distance import mahalanobis


def D(a, b): # cosine similarity
    return F.cosine_similarity(a, b, dim=-1).mean()


def pw_cos_sim(x):
    m = torch.zeros((x.shape[0], x.shape[0]))
    for i in range(x.shape[0]):
        for j in range(x.shape[0]):
            m[i, j] = D(x[i], x[j])
    return m.mean()


DATA_INFO = {
    'COIL100': {
        'class': COIL100,
        'root': 'COIL100',
        'kwargs': {},
        'k': 72
    },
    'ALOI-viewpoint': {
        'class': ALOI,
        'root': 'ALOI',
        'kwargs': {'mode': 'viewpoint', 'camera': None},
        'k': 72
    },
    'ALOI-illumination': {
        'class': ALOI,
        'root': 'ALOI',
        'kwargs': {'mode': 'illumination', 'camera': None},
        'k':  8
    },
    'ALOI-temperature': {
        'class': ALOI,
        'root': 'ALOI',
        'kwargs': {'mode': 'temperature', 'camera': None},
        'k': 12
    },
    'ALOT-viewpoint': {
        'class': ALOT,
        'root': 'ALOT',
        'kwargs': {'mode': 'viewpoint', 'camera': None},
        'k': 16
    },
    'ALOT-illumination': {
        'class': ALOT,
        'root': 'ALOT',
        'kwargs': {'mode': 'illumination', 'camera': None},
        'k':  6
    },
    'ALOT-temperature': {
        'class': ALOT,
        'root': 'ALOT',
        'kwargs': {'mode': 'temperature', 'camera': None},
        'k': 2
    },
    'ExposureErrors': {
        'class': ExposureErrors,
        'root': 'ExposureErrors',
        'kwargs': {},
        'k': 6
    },
    'DaLI': {
        'class': DaLI,
        'root': 'DaLI',
        'kwargs': {},
        'k': 32
    },
    'Flickr1024': {
        'class': Flickr1024,
        'root': 'Flickr1024',
        'kwargs': {},
        'k': 2
    },
    'RealBlur': {
        'class': RealBlur,
        'root': 'RealBlur',
        'kwargs': {},
        'k': 2
    },
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='default', type=str, metavar='M',
                        help='model to evaluate invariance of (random/supervised/default/ventral/dorsal)')
    parser.add_argument('--dataset', default='Flickr1024', type=str, metavar='DS',
                        help='dataset to evaluate invariance (Flickr1024/COIL100/ALOI-viewpoint/ALOI-illumination/ALOI-temperature/ALOT-viewpoint/ALOT-illumination/ALOT-temperature/ExposureErrors/DaLI/RealBlur)')
    parser.add_argument('--device', default='cuda:0', type=str, metavar='D',
                        help='GPU device')
    parser.add_argument('--feature-layer', default='backbone', type=str, metavar='F',
                        help='layer to extract features from (default: backbone)')
    parser.add_argument('--resize', default=224, type=int, metavar='R',
                        help='resize')
    parser.add_argument('--crop-size', default=224, type=int, metavar='C',
                        help='crop size')
    parser.add_argument('--ckpt-dir', default='./models/', type=Path,
                        metavar='DIR', help='path to checkpoint directory')
    parser.add_argument('--results-dir', default='./results/', type=Path,
                        metavar='DIR', help='path to results directory')
    parser.add_argument('--data-root', default='../data/', type=Path,
                        metavar='DIR', help='data root directory')
    args = parser.parse_args()

    resize = 224
    crop_size = 224

    if 'camera' == DATA_INFO[args.dataset]:
        camera = DATA_INFO[args.dataset]['camera']
    k = DATA_INFO[args.dataset]['k']

    transform = transforms.Compose([
        transforms.Resize(args.resize),
        transforms.CenterCrop(args.crop_size),
        transforms.ToTensor(),
        transforms.Normalize(*model_utils.imagenet_mean_std)
    ])

    dataset = DATA_INFO[args.dataset]['class'](
        os.path.join(args.data_root, DATA_INFO[args.dataset]['root']),
        transform=transform,
        **DATA_INFO[args.dataset]['kwargs']
    )

    # load model
    model = model_utils.load_model(args.model, args)
    model = model.to(args.device)

    mean_feature = torch.load(open(f"{args.results_dir}/{args.model}_{args.feature_layer}_mean_feature.pth", 'rb'))
    cov_matrix = torch.load(open(f"{args.results_dir}/{args.model}_{args.feature_layer}_feature_cov_matrix.pth", 'rb'))
    if args.model == 'random':
        cov_matrix = cov_matrix + 1e-10 * np.eye(cov_matrix.shape[0])
    inv_cov_matrix = np.linalg.inv(cov_matrix)
    cholesky_matrix = torch.linalg.cholesky(torch.from_numpy(inv_cov_matrix).to(torch.float32))

    L = torch.zeros(dataset.num_classes)
    S = torch.zeros(dataset.num_classes)

    with torch.no_grad():
        for i in tqdm(range(dataset.num_classes), total=dataset.num_classes):
            images = dataset.get_views_of_class(i)
            features = model(images.to(args.device)).cpu().detach()
            features = (mean_feature - features) @ cholesky_matrix
            S[i] = pw_cos_sim(features)
            L[i] = torch.pdist(features, p=2).mean()


    print(f'{args.model}: {L.mean().item()}, {S.mean().item()}')

    torch.save(L, open(f"results/{args.model}_{args.feature_layer}_{args.dataset}_invariance_distance.pth", 'wb'))
    torch.save(S, open(f"results/{args.model}_{args.feature_layer}_{args.dataset}_invariance_similarity.pth", 'wb'))
