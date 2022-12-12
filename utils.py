#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import os
import copy
import torch
import numpy as np
from scipy.stats import rice
from torchvision import datasets, transforms
from torchvision.datasets import ImageFolder
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import random_split
from sampling import mnist_iid, mnist_noniid, mnist_noniid_unequal
from sampling import cifar_iid, cifar_noniid
from sampling import LEGO_iid


def get_dataset(args):
    """ Returns train and test datasets and a user group which is a dict where
    the keys are the user index and the values are the corresponding data for
    each of those users.
    """

    if args.dataset == 'cifar':
        data_dir = '../data/cifar/'
        apply_transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        train_dataset = datasets.CIFAR10(data_dir, train=True, download=True,
                                         transform=apply_transform)

        test_dataset = datasets.CIFAR10(data_dir, train=False, download=True,
                                        transform=apply_transform)

        # sample training data amongst users
        if args.iid:
            # Sample IID user data from Mnist
            user_groups = cifar_iid(train_dataset, args.num_users)
        else:
            # Sample Non-IID user data from Mnist
            if args.unequal:
                # Chose uneuqal splits for every user
                raise NotImplementedError()
            else:
                # Chose euqal splits for every user
                user_groups = cifar_noniid(train_dataset, args.num_users)

        """
        This part is the LEGO dataset code!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        """

    elif args.dataset == 'LEGO':
        data_dir = './data/LEGO brick images v1/'
        classes = os.listdir(data_dir)
        print('The number of classes in the dataset is: ' + str(len(classes)))

        train_transform = transforms.Compose([
            transforms.RandomRotation(10),  # rotate +/- 10 degrees
            transforms.RandomHorizontalFlip(),  # reverse 50% of images
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])
        dataset = ImageFolder(data_dir, transform=train_transform)
        img, label = dataset[100]
        torch.manual_seed(20)
        test_size = len(dataset) // 5
        train_size = len(dataset) - test_size
        train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
        if args.iid:
            user_groups = LEGO_iid(train_dataset, args.num_users)

    elif args.dataset == 'mnist' or 'fmnist':
        if args.dataset == 'mnist':
            data_dir = './data/mnist/'
        else:
            data_dir = './data/fmnist/'

        apply_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))])

        train_dataset = datasets.MNIST(data_dir, train=True, download=True,
                                       transform=apply_transform)

        test_dataset = datasets.MNIST(data_dir, train=False, download=True,
                                      transform=apply_transform)

        # sample training data amongst users
        if args.iid:
            # Sample IID user data from Mnist
            user_groups = mnist_iid(train_dataset, args.num_users)
        else:
            # Sample Non-IID user data from Mnist
            if args.unequal:
                # Chose uneuqal splits for every user
                user_groups = mnist_noniid_unequal(train_dataset, args.num_users)
            else:
                # Chose euqal splits for every user
                user_groups = mnist_noniid(train_dataset, args.num_users)

    return train_dataset, test_dataset, user_groups


def average_weights(w, hk, args, device):
    w_avg = copy.deepcopy(w[0])
    # calculate standard deviation
    stds = torch.empty(0).to(device)
    for i in range(0, len(w)):
        cur = torch.empty(0).to(device)
        for key in w[i].keys():
            cur = torch.cat((cur, w[i][key].view(-1)))
        stdValue = torch.std(cur, unbiased=False)
        stds = torch.cat((stds, stdValue.unsqueeze(0)))

    # calculate m
    tmp = [stds[i] ** 2 / hk[i] for i in range(len(w))]
    tmp_tensor = torch.Tensor(tmp).to(device)
    tmp_sorted, indices = torch.sort(tmp_tensor, descending=True)
    m = torch.sqrt(tmp_sorted[-args.selected_users])
    max_idx = indices[:(args.num_users - args.selected_users)]

    # add noise & average
    for key in w_avg.keys():
        for i in range(0, len(w)):
            if i == 0:
                w_avg[key] -= w_avg[key]
            if i in max_idx:
                continue
            snr = get_snr(args.snr_dB)
            wgn = torch.normal(0, 1 / snr, w[i][key].shape).to(device)
            w_avg[key] += w[i][key] + m * wgn
        w_avg[key] = torch.div(w_avg[key], args.selected_users)
    return w_avg


def get_beta(w):
    w_squaresum = torch.sum(torch.pow(w, 2))
    d = torch.numel(w)
    return w_squaresum, d


def get_hk(args, alpha):
    cn = np.random.multivariate_normal([0, 0], [[0.5, 0], [0, 0.5]], args.num_users)
    hk = np.sqrt(alpha / (alpha + 1)) + np.sqrt(1 / (alpha + 1)) * cn
    amplitude = [np.sum(hk[i] ** 2) for i in range(args.num_users)]
    return amplitude


def get_snr(snr_dB):
    return np.power(10, snr_dB / 10)


def exp_details(args):
    print('\nExperimental details:')
    print(f'    Model     : {args.model}')
    print(f'    Optimizer : {args.optimizer}')
    print(f'    Learning  : {args.lr}')
    print(f'    Global Rounds   : {args.epochs}\n')

    print('    Federated parameters:')
    if args.iid:
        print('    IID')
    else:
        print('    Non-IID')
    print(f'    Fraction of users  : {args.frac}')
    print(f'    Local Batch size   : {args.local_bs}')
    print(f'    Local Epochs       : {args.local_ep}\n')
    return
