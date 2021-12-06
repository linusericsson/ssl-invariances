#!/usr/bin/env python
# coding: utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, ConcatDataset, Subset

import os
import inspect
import argparse
from glob import glob
from tqdm import tqdm
from pathlib import Path
from itertools import product
from collections import OrderedDict

import numpy as np

from datasets import Causal3DIdent

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import KFold

import model_utils


class CVTester():
    def __init__(self, model, trainval, test, device, k=5, batch_size=64,
                 feature_dim=2048, wd_range=None, debug=False):
        self.model = model
        self.trainval = trainval
        self.test = test
        self.kf = KFold(n_splits=k, shuffle=True)
        self.batch_size = batch_size
        self.device = device
        self.feature_dim = feature_dim
        self.debug = debug
        self.best_params = {}

        self.X_trainval_feature, self.y_trainval = self._inference(self.trainval, self.model, 'trainval')
        self.X_test_feature, self.y_test = self._inference(self.test, self.model, 'test')

        if wd_range is None:
            self.wd_range = torch.logspace(-6, 5, 45)
        else:
            self.wd_range = wd_range

        self.clf = KernelRidge(kernel='rbf')
        self.pipe = make_pipeline(StandardScaler(), KernelRidge(kernel='rbf'))

        self.alphas = np.logspace(0, -4, 5)
        self.gammas = np.logspace(-5, 2, 8)

    def _inference(self, data_set, model, split):
        model.eval()
        feature_vector = []
        labels_vector = []
        c = 10 if split == 'test' else 5
        max_iter = int((4096 * c) / self.batch_size)
        loader = DataLoader(data_set, batch_size=self.batch_size, shuffle=True)
        for i, data in enumerate(tqdm(loader, desc=f'Computing features for {split} set', total=max_iter)):
            if self.debug and i >= 100:
                print('DEBUG: stopping early.')
                break
            elif i >= max_iter:
                break
            batch_x, batch_y = data
            batch_x = batch_x.to(self.device)
            labels_vector.extend(np.array(batch_y))

            features = model(batch_x)
            feature_vector.extend(features.cpu().detach().numpy())

        feature_vector = np.array(feature_vector)
        labels_vector = np.array(labels_vector).reshape(-1, 1)

        return feature_vector, labels_vector

    def validate(self):
        best_score = -np.inf
        for alpha, gamma in tqdm(product(self.alphas, self.gammas), total=len(self.alphas) * len(self.gammas), desc='Cross-validating'):
            self.pipe.set_params(**{'kernelridge__alpha': alpha, 'kernelridge__gamma': gamma})

            cv_scores = []
            for i, (train, val) in enumerate(self.kf.split(self.X_trainval_feature)):
                self.pipe.fit(self.X_trainval_feature[train], self.y_trainval[train])
                test_score = self.pipe.score(self.X_trainval_feature[val], self.y_trainval[val])
                cv_scores.append(test_score)
            score = np.mean(cv_scores)

            if score > best_score:
                best_score = score
                self.best_params['alpha'] = alpha
                self.best_params['gamma'] = gamma

    def evaluate(self):
        print(f"Best hyperparameters {self.best_params}")
        self.pipe.set_params(**{'kernelridge__alpha': self.best_params['alpha'], 'kernelridge__gamma': self.best_params['gamma']})
        self.pipe.fit(self.X_trainval_feature, self.y_trainval)  # apply scaling on training data
        test_score = self.pipe.score(self.X_test_feature, self.y_test)  # apply scaling on testing data, without leaking training data.
        return test_score


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='default', type=str, metavar='M',
                        help='model to evaluate invariance of (random/supervised/default/ventral/dorsal)')
    parser.add_argument('--fuse-mode', default='cat', type=str, metavar='F',
                        help='method of fusing multiple representations (cat/add/mean)')
    parser.add_argument('--mean-embedding', default='', type=str,
                        help='whether to use mean-embedding model (ventral/dorsal)')
    parser.add_argument('--mean-embedding-k', default=32, type=int,
                        help='number of samples in mean-embedding model (default: 32)')
    parser.add_argument('--target', default='0', type=str, metavar='DS',
                        help='latent variable to regress (0-9)')
    parser.add_argument('--cv-folds', default=5, type=int,
                        help='number of cross-validation folds (default: 5)')
    parser.add_argument('--device', default='cuda:0', type=str, metavar='D',
                        help='GPU device')
    parser.add_argument('--feature-layer', default='backbone', type=str, metavar='F',
                        help='layer to extract features from (default: backbone)')
    parser.add_argument('--batch-size', default=64, type=int, metavar='N',
                        help='mini-batch size')
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
    parser.add_argument('--quick', dest='quick', action='store_true')
    parser.set_defaults(quick=False)
    args = parser.parse_args()


    # load model
    model = model_utils.load_model(args.model, args)
    model = model.to(args.device)

    transform = transforms.Compose([
        transforms.Resize(args.resize),
        transforms.CenterCrop(args.crop_size),
        transforms.ToTensor(),
        transforms.Normalize(*model_utils.imagenet_mean_std)
    ])
    trainset = Causal3DIdent(os.path.join(args.data_root, 'Causal3DIdent'), split='train', target_type=args.target, transform=transform)
    testset = Causal3DIdent(os.path.join(args.data_root, 'Causal3DIdent'), split='test', target_type=args.target, transform=transform)

    tester = CVTester(model, trainset, testset, args.device, k=args.cv_folds, batch_size=args.batch_size, debug=args.quick)
    tester.validate()
    score = tester.evaluate()

    print(score)

    if args.mean_embedding:
        model_name = args.model + '_m_e_' + args.mean_embedding
    else:
        model_name = args.model

    torch.save(score, open(f"results/{model_name}_{args.feature_layer}_causal3dident_{args.target}.pth", 'wb'))
