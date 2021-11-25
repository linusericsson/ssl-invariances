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
from tqdm import tqdm
from pathlib import Path
from collections import OrderedDict

from sklearn.linear_model import Ridge
from sklearn.linear_model import LogisticRegression as LogReg
from sklearn.metrics import confusion_matrix, precision_recall_curve
from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import r2_score
from sklearn.model_selection import KFold

import numpy as np

import model_utils

from datasets import Flowers, Caltech101, FacesInTheWild300W, CelebA, LeedsSportsPose


dataset_info = {
    'cifar10': {
        'class': datasets.CIFAR10, 'dir': 'CIFAR10', 'num_classes': 10,
        'splits': ['train', 'train', 'test'], 'split_size': 0.8,
        'mode': 'classification'
    },
    'flowers': {
        'class': Flowers, 'dir': 'Flowers', 'num_classes': 102,
        'splits': ['train', 'val', 'test'], 'split_size': 0.5,
        'mode': 'classification'
    },
    'caltech101': {
        'class': Caltech101, 'dir': 'Caltech101', 'num_classes': 102,
        'splits': ['train', 'train', 'test'], 'split_size': 0.5,
        'mode': 'classification'
    },
    '300w': {
        'class': FacesInTheWild300W, 'dir': '300W', 'num_classes': None,
        'splits': ['train', 'val', 'test'], 'split_size': 0.5,
        'mode': 'regression'
    },
    'celeba': {
        'class': CelebA, 'dir': 'CelebA', 'num_classes': 40,
        'splits': ['train', 'val', 'test'], 'split_size': 0.5,
        'target_type': 'landmarks',
        'mode': 'regression'
    },
    'leeds_sports_pose': {
        'class': LeedsSportsPose, 'dir': 'LeedsSportsPose', 'num_classes': None,
        'splits': ['train', 'train', 'test'], 'split_size': 0.8,
        'mode': 'regression'
    }
}


class LogisticRegression(nn.Module):
    def __init__(self, input_dim, num_features, num_classes, multilabel, metric):
        super().__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.multilabel = multilabel
        self.metric = metric
        self.clf = LogReg(solver='lbfgs', multi_class='multinomial', warm_start=True)

        print('Logistic regression:')
        print(f'\t solver = L-BFGS')
        print(f"\t classes = {self.num_classes}")
        print(f"\t multilabel = {self.multilabel}")
        print(f"\t metric = {self.metric}")

    def set_params(self, d):
        self.clf.set_params(**d)

    @ignore_warnings(category=ConvergenceWarning)
    def fit(self, X_train, y_train, X_test, y_test):
        if not self.multilabel:
            self.clf = self.clf.fit(X_train, y_train)
            test_acc = self.clf.score(X_test, y_test)

            pred_test = self.clf.predict(X_test)
            #Get the confusion matrix
            cm = confusion_matrix(y_test, pred_test)

            if self.metric == 'mean per-class accuracy':
                _cm = cm.diagonal() / cm.sum(axis=1) 
                test_acc = _cm.mean()

            return test_acc, cm
        else:
            per_class_acc = []
            for cls in range(self.num_classes):
                self.clf.fit(X_train, y_train[:, cls])
                acc = self.clf.score(X_test, y_test[:, cls])
                per_class_acc.append(acc)

            test_acc = np.mean(per_class_acc)

            return test_acc, per_class_acc


class LinearRegression(nn.Module):
    def __init__(self, input_dim, num_features):
        super().__init__()
        self.input_dim = input_dim
        self.num_features = num_features
        self.clf = Ridge()

    def set_params(self, d):
        d['alpha'] = d['C']
        del d['C']
        self.clf.set_params(**d)

    @ignore_warnings(category=ConvergenceWarning)
    def fit(self, X_train, y_train, X_test, y_test, metric='r2'):
        self.clf = self.clf.fit(X_train, y_train)
        r2 = self.clf.score(X_test, y_test)
        mse_loss = F.mse_loss(torch.from_numpy(self.clf.predict(X_test)), torch.from_numpy(y_test)).item()
        if metric =='mse':
            return mse_loss, None
        elif metric == 'r2':
            return r2, None


class CVTester():
    def __init__(self, mode, model, trainval, test, device, num_classes, num_features, k=5, batch_size=256,
                 feature_dim=2048, wd_range=None, debug=False):
        self.mode = mode
        self.model = model
        self.trainval = trainval
        self.test = test
        self.kf = KFold(n_splits=k, shuffle=True)
        self.batch_size = batch_size
        self.device = device
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.debug = debug
        self.best_params = {}

        self.X_trainval_feature, self.y_trainval = self._inference(self.trainval, self.model, 'trainval')
        self.X_test_feature, self.y_test = self._inference(self.test, self.model, 'test')

        multilabel = (mode == 'multi-label classification')
        metric = 'mean per-class accuracy' if bool(
            sum([isinstance(trainval, d) for d in [Caltech101, Flowers]])
        ) else 'accuracy'

        if wd_range is None:
            self.wd_range = torch.logspace(-6, 5, 45)
        else:
            self.wd_range = wd_range

        if 'classification' in self.mode:
            self.classifier = LogisticRegression(self.feature_dim, num_features, self.num_classes,
                                                 multilabel, metric).to(self.device)
        elif self.mode == 'regression':
            self.classifier = LinearRegression(self.feature_dim, num_features).to(self.device)

    def _inference(self, data_set, model, split):
        model.eval()
        feature_vector = []
        labels_vector = []
        loader = DataLoader(data_set, batch_size=self.batch_size, shuffle=True)
        for i, data in enumerate(tqdm(loader, desc=f'Computing features for {split} set')):
            if self.debug and i >= 100:
                print('DEBUG: stopping early.')
                break
            batch_x, batch_y = data
            batch_x = batch_x.to(self.device)
            labels_vector.extend(np.array(batch_y))

            features = model(batch_x)
            feature_vector.extend(features.cpu().detach().numpy())

        feature_vector = np.array(feature_vector)
        if 'classification' in self.mode:
            labels_vector = np.array(labels_vector, dtype=int)
        else:
            labels_vector = np.array(labels_vector)

        return feature_vector, labels_vector

    def validate(self):
        best_score = -np.inf
        for wd in tqdm(self.wd_range, desc='Cross-validating'):
            C = 1. / wd.item()
            self.classifier.set_params({'C': C})

            cv_scores = []
            for i, (train, val) in enumerate(self.kf.split(self.X_trainval_feature)):
                test_score, _ = self.classifier.fit(self.X_trainval_feature[train], self.y_trainval[train],
                                                    self.X_trainval_feature[val], self.y_trainval[val])
                cv_scores.append(test_score)
            score = np.mean(cv_scores)
            #print(f'{C}: {score}, {cv_scores}')

            if score > best_score:
                best_score = score
                self.best_params['C'] = C

    def evaluate(self):
        print(f"Best hyperparameters {self.best_params}")
        self.classifier.set_params({'C': self.best_params['C']})
        test_score, per_class_acc = self.classifier.fit(self.X_trainval_feature, self.y_trainval, self.X_test_feature, self.y_test)
        return test_score, per_class_acc


def get_dataset(args, c, d, s, t):
    if d == 'CelebA':
        return c(os.path.join(args.data_root, d), split=s, target_type=dataset_info[args.dataset]['target_type'], transform=t, download=True)
    elif d == 'CIFAR10':
        return c(os.path.join(args.data_root, d), train=s == 'train', transform=t, download=True)
    else:
        if 'split' in inspect.getfullargspec(c.__init__)[0]:
            if s == 'valid':
                try:
                    return c(os.path.join(args.data_root, d), split=s, transform=t)
                except:
                    return c(os.path.join(args.data_root, d), split='val', transform=t)
            else:
                return c(os.path.join(args.data_root, d), split=s, transform=t)
        else:
            return c(os.path.join(args.data_root, d), train=s == 'train', transform=t)


def prepare_data(args, norm):
    transform = transforms.Compose([
        transforms.Resize(args.resize),
        transforms.CenterCrop(args.crop_size),
        transforms.ToTensor(),
        transforms.Normalize(*norm)
    ])

    if dataset_info[args.dataset]['splits'][1] == 'val':
        train_dataset = get_dataset(args, dataset_info[args.dataset]['class'],
                                    dataset_info[args.dataset]['dir'], 'train', transform)
        val_dataset = get_dataset(args, dataset_info[args.dataset]['class'],
                                    dataset_info[args.dataset]['dir'], 'valid', transform)
        trainval = ConcatDataset([train_dataset, val_dataset])

    elif dataset_info[args.dataset]['splits'][1] == 'train':
        trainval = get_dataset(args, dataset_info[args.dataset]['class'],
                              dataset_info[args.dataset]['dir'], 'train', transform)

    test = get_dataset(args, dataset_info[args.dataset]['class'],
                       dataset_info[args.dataset]['dir'], 'test', transform)

    return trainval, test


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
    parser.add_argument('--dataset', default='cifar10', type=str, metavar='DS',
                        help='dataset to evaluate on')
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
    parser.add_argument('--results-dir', default='./results', type=Path,
                        metavar='DIR', help='path to results directory')
    parser.add_argument('--data-root', default='../data/', type=Path,
                        metavar='DIR', help='data root directory')
    parser.add_argument('--quick', dest='quick', action='store_true')
    parser.set_defaults(quick=False)
    args = parser.parse_args()
    args.dataset = args.dataset.lower()

    imagenet_mean_std = [[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]]
    trainval, test = prepare_data(args, norm=imagenet_mean_std)

    models_list = [model_utils.load_model(model_name, args) for model_name in args.model.split('+')]
    model = model_utils.ModelCombiner(args.fuse_mode, *models_list)
    model.to(args.device)

    c = 2 if 'w3' in args.model else args.model.count('+')
    wd_range = torch.logspace(-6 + 2 * c, 5 + 2 * c, 45)
    print(f'Searching regularisation parameter in {wd_range}')

    clf = CVTester(dataset_info[args.dataset]['mode'], model, trainval, test, device=args.device, batch_size=args.batch_size, k=args.cv_folds,
                   num_classes=dataset_info[args.dataset]['num_classes'], num_features=len(models_list), wd_range=wd_range, debug=args.quick)
    clf.validate()
    test_acc, per_class_acc = clf.evaluate()

    print(f'{args.model} on {args.dataset}: {test_acc:.2f}')

    if args.mean_embedding:
        model_name = args.model + '_m_e_' + args.mean_embedding
    else:
        model_name = args.model

    torch.save(test_acc, open(f'{args.results_dir}/{model_name}_{args.dataset}.pth', 'wb'))
