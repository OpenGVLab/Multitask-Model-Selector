#!/usr/bin/env python
# coding: utf-8

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, ConcatDataset
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import datasets, transforms, models

import PIL

from datasets.dtd import DTD
from datasets.pets import Pets
from datasets.cars import Cars
from datasets.food import Food
from datasets.sun397 import SUN397
from datasets.voc2007 import VOC2007
from datasets.flowers import Flowers
from datasets.aircraft import Aircraft
from datasets.caltech101 import Caltech101


# Data classes and functions
def get_dataset(dset, root, split, transform):
    print(dset,root,split)
    # return dset(root, train=(split == 'train'), transform=transform, download=True)
    # for cifar10,cifar100
    return dset(root, split, transform=transform, download=False)
    

def get_train_valid_loader(dset,
                           data_dir,
                           normalise_dict,
                           batch_size,
                           image_size,
                           random_seed,
                           valid_size=0.2,
                           shuffle=True,
                           num_workers=11,
                           pin_memory=True):
    """
    Utility function for loading and returning train and valid
    multi-process iterators over the CIFAR-10 dataset.
    If using CUDA, num_workers should be set to 1 and pin_memory to True.
    Params
    ------
    - dset: dataset class to load.
    - data_dir: path directory to the dataset.
    - normalise_dict: dictionary containing the normalisation parameters.
    - batch_size: how many samples per batch to load.
    - image_size: size of images after transforms.
    - random_seed: fix seed for reproducibility.
    - valid_size: percentage split of the training set used for
      the validation set. Should be a float in the range [0, 1].
    - shuffle: whether to shuffle the train/validation indices.
    - num_workers: number of subprocesses to use when loading the dataset.
    - pin_memory: whether to copy tensors into CUDA pinned memory. Set it to
      True if using GPU.
    Returns
    -------
    - train_loader: training set iterator.
    - valid_loader: validation set iterator.
    - trainval_loader: iterator for the training and validation sets combined.
    """
    error_msg = "[!] valid_size should be in the range [0, 1]."
    assert ((valid_size >= 0) and (valid_size <= 1)), error_msg

    normalize = transforms.Normalize(**normalise_dict)
    print("Train normaliser:", normalize)

    # define transforms
    transform = transforms.Compose([
        transforms.Resize(image_size, interpolation=PIL.Image.BICUBIC),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        normalize,
    ])

    if dset in [Aircraft, DTD, Flowers, VOC2007]:
        # if we have a predefined validation set
        print(data_dir,'data_dir')
        train_dataset = get_dataset(dset, data_dir, 'train', transform)
        valid_dataset = get_dataset(dset, data_dir, 'val', transform)
        trainval_dataset = ConcatDataset([train_dataset, valid_dataset])

        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=shuffle,
            num_workers=num_workers, pin_memory=pin_memory,
        )
        valid_loader = DataLoader(
            valid_dataset, batch_size=batch_size, shuffle=shuffle,
            num_workers=num_workers, pin_memory=pin_memory,
        )
        trainval_loader = DataLoader(
            trainval_dataset, batch_size=batch_size, shuffle=shuffle,
            num_workers=num_workers, pin_memory=pin_memory,
        )
    else:
        # otherwise we select a random subset of the train set to form the validation set
        train_dataset = get_dataset(dset, data_dir, 'train', transform)
        valid_dataset = get_dataset(dset, data_dir, 'train', transform)
        trainval_dataset = get_dataset(dset, data_dir, 'train', transform)

        num_train = len(train_dataset)
        indices = list(range(num_train))
        split = int(np.floor(valid_size * num_train))

        if shuffle:
            np.random.seed(random_seed)
            np.random.shuffle(indices)

        train_idx, valid_idx = indices[split:], indices[:split]
        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)

        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, sampler=train_sampler,
            num_workers=num_workers, pin_memory=pin_memory,
        )
        valid_loader = DataLoader(
            valid_dataset, batch_size=batch_size, sampler=valid_sampler,
            num_workers=num_workers, pin_memory=pin_memory,
        )
        trainval_loader = DataLoader(
            trainval_dataset, batch_size=batch_size, shuffle=shuffle,
            num_workers=num_workers, pin_memory=pin_memory,
        )

    return train_loader, valid_loader, trainval_loader


def get_train_test_loader(dset,
                    data_dir,
                    normalise_dict,
                    batch_size,
                    image_size,
                    shuffle=False,
                    num_workers=11,
                    pin_memory=True):
    normalize = transforms.Normalize(**normalise_dict)
    print("Train normaliser:", normalize)

    # define transforms
    transform = transforms.Compose([
        transforms.Resize(image_size, interpolation=PIL.Image.BICUBIC),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        normalize,
    ])
    train_dataset = get_dataset(dset, data_dir, 'train', transform)
    test_dataset = get_dataset(dset, data_dir, 'test', transform)
    if dset in [Aircraft, DTD, Flowers, VOC2007]:
        valid_dataset = get_dataset(dset, data_dir, 'val', transform)
        all_dataset = ConcatDataset([train_dataset, test_dataset, valid_dataset])
    else:
        all_dataset = ConcatDataset([train_dataset, test_dataset])
    
    data_loader = DataLoader(
        all_dataset, batch_size=batch_size, shuffle=shuffle,
        num_workers=num_workers, pin_memory=pin_memory,
    )

    return data_loader
        

def get_test_loader(dset,
                    data_dir,
                    normalise_dict,
                    batch_size,
                    image_size,
                    shuffle=False,
                    num_workers=11,
                    pin_memory=True):
    """
    Utility function for loading and returning a multi-process
    test iterator over the CIFAR-10 dataset.
    If using CUDA, num_workers should be set to 1 and pin_memory to True.
    Params
    ------
    - dset: dataset class to load.
    - data_dir: path directory to the dataset.
    - normalise_dict: dictionary containing the normalisation parameters.
    - batch_size: how many samples per batch to load.
    - image_size: size of images after transforms.
    - shuffle: whether to shuffle the dataset after every epoch.
    - num_workers: number of subprocesses to use when loading the dataset.
    - pin_memory: whether to copy tensors into CUDA pinned memory. Set it to
      True if using GPU.
    Returns
    -------
    - data_loader: test set iterator.
    """

    normalize = transforms.Normalize(**normalise_dict)
    print("Test normaliser:", normalize)

    # define transform
    transform = transforms.Compose([
        transforms.Resize(image_size, interpolation=PIL.Image.BICUBIC),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        normalize,
    ])

    dataset = get_dataset(dset, data_dir, 'test', transform)

    data_loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle,
        num_workers=num_workers, pin_memory=pin_memory,
    )

    return data_loader


def prepare_data(dset, data_dir, batch_size, image_size, normalisation):
    if normalisation:
        normalise_dict = {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]}
    else:
        normalise_dict = {'mean': [0.0, 0.0, 0.0], 'std': [1.0, 1.0, 1.0]}
    train_loader, val_loader, trainval_loader = get_train_valid_loader(dset, data_dir, normalise_dict,
                                                batch_size, image_size, random_seed=0)
    test_loader = get_test_loader(dset, data_dir, normalise_dict, batch_size, image_size)

    all_loader = get_train_test_loader(dset, data_dir, normalise_dict, batch_size, image_size)

    return train_loader, val_loader, trainval_loader, test_loader, all_loader


def get_data(dataset):
    # name: {class, root, num_classes, metric}
    # download target data to ./data/*
    LINEAR_DATASETS = {
    'aircraft': [Aircraft, './data/Aircraft', 100, 'mean per-class accuracy'],
    'caltech101': [Caltech101, './data/Caltech101', 102, 'mean per-class accuracy'],
    'cars': [Cars, './data/Cars', 196, 'accuracy'],
    'cifar10': [datasets.CIFAR10, './data/CIFAR10', 10, 'accuracy'],
    'cifar100': [datasets.CIFAR100, './data/CIFAR100', 100, 'accuracy'],
    'dtd': [DTD, './data/DTD', 47, 'accuracy'],
    'flowers': [Flowers, './data/Flowers', 102, 'mean per-class accuracy'],
    'food': [Food, './data/Food', 101, 'accuracy'],
    'pets': [Pets, './data/Pets', 37, 'mean per-class accuracy'],
    'sun397': [SUN397, './data/SUN397', 397, 'accuracy'],
    'voc2007': [VOC2007, './data/VOC2007', 20, 'mAP'],
    'dSprites': [VOC2007, './data/VOC2007', 20, 'mAP']
    }   
    dset, data_dir, num_classes, metric = LINEAR_DATASETS[dataset]
    return dset, data_dir, num_classes, metric
