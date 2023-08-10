#!/usr/bin/env python
# coding: utf-8

import os
import argparse
from pprint import pprint

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import datasets, transforms
import models.group1 as models

import PIL
import numpy as np
from tqdm import tqdm

from sklearn.linear_model import LogisticRegression as LogReg
from sklearn.metrics import confusion_matrix, precision_recall_curve
from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning

from temperature_scaling import cross_validate_temp_scaling, DummyDataset
from utils import forward_pass,  load_model_group4_norm
from get_dataloader import prepare_data, get_data
from utils import forward_pass

# pca 80
import json
import time


models_hub = ['deit_tiny', 'deit_small', 'deit_base',
    'dino_small', 'dino_base', 'mocov3_small', 
    'pvtv2_b2', 'pvtv2_b3',
    'pvt_tiny', 'pvt_small', 'pvt_medium', 
     'swin_t', 'swin_s'
]
# models_hub = ['deit_tiny', 'deit_small', 'deit_base',
#     'dino_small', 'dino_base', 
#     'pvtv2_b2', 'pvtv2_b3',
#     'pvt_tiny', 'pvt_small', 'pvt_medium', 
#      'swin_t', 'swin_s'
# ]



# Testing classes and functions
# Main code
if __name__ == "__main__":
    dataset_all = ['food','pets','sun397','flowers','cars','cifar10','cifar100','aircraft']
    parser = argparse.ArgumentParser(description='Evaluate pretrained self-supervised model via logistic regression.')
    parser.add_argument('-m', '--model', type=str, default='deepcluster-v2',
                        help='name of the pretrained model to load and evaluate (deepcluster-v2 | supervised)')
    parser.add_argument('-d', '--dataset', type=str, default='cifar10', help='name of the dataset to evaluate on')
    parser.add_argument('--metric', type=str, default='logme', help='name of the method for measuring transferability')
    parser.add_argument('-b', '--batch-size', type=int, default=256, help='the size of the mini-batches when inferring features')
    parser.add_argument('-i', '--image-size', type=int, default=224, help='the size of the input images')
    parser.add_argument('-w', '--wd-values', type=int, default=45, help='the number of weight decay values to validate')
    parser.add_argument('-c', '--C', type=float, default=None, help='sklearn C value (1 / weight_decay), if not tuning on validation set')
    
    parser.add_argument('-r', '--reg', type=float, default=1.0, help='regularization weight in WDA')
    parser.add_argument('-n', '--no-norm', action='store_true', default=False,
                        help='whether to turn off data normalisation (based on ImageNet values)')
    parser.add_argument('--device', type=str, default='cuda', help='CUDA or CPU training (cuda | cpu)')
    parser.add_argument('-f', '--ft', type=float, default=-1, help='whether using tinetuned model')
    parser.add_argument('--fulldata', type=float, default=-1, help='whether using tinetuned model')
    parser.add_argument('--threshold', type=float, default=0.5, help='whether using tinetuned model')
    args = parser.parse_args()
    args.norm = not args.no_norm
    pprint(args)

    # start_time 
    for dataset in dataset_all:
        args.dataset = dataset
        print('dataset!!!!!',args.dataset)

        # load dataset
        dset, data_dir, num_classes, metric = get_data(args.dataset)
        args.num_classes = num_classes
        
        train_loader, val_loader, trainval_loader, test_loader, all_loader = prepare_data(
            dset, data_dir, args.batch_size, args.image_size, normalisation=args.norm)
        
        print(f'Train:{len(train_loader.dataset)}, Val:{len(val_loader.dataset)},' 
                f'TrainVal:{len(trainval_loader.dataset)}, Test:{len(test_loader.dataset)} '
                f'AllData:{len(all_loader.dataset)}')
        
        #if args.dataset in ['sun397','caltech101','pets', 'voc2007', 'dtd', 'cifar100', 'cars']:
        #    trainval_loader = all_loader
        # if args.fulldata > 0:
        #     trainval_loader = all_loader
        # trainval_loader = all_loader
        trainval_loader = trainval_loader



        fpath = os.path.join('./results_f_trainval', 'group4_bnorm')
        if not os.path.exists(fpath):
            os.makedirs(fpath)
        
        
        for model in models_hub:
            args.model = model
            args.dataset = dataset
            model_npy_feature = os.path.join(fpath, f'{args.model}_{args.dataset}_feature.npy')
            model_npy_label = os.path.join(fpath, f'{args.model}_{args.dataset}_label.npy')

            if os.path.exists(model_npy_feature) and os.path.exists(model_npy_label):
                print(f"Features and Labels of {args.model} on {args.dataset} has been saved.")
                continue

            
            model, fc_layer = load_model_group4_norm(args)
            X_trainval_feature, _ , y_trainval = forward_pass(trainval_loader, model, fc_layer, args.model)   
            
            #X_trainval_feature, y_trainval = forward_pass_feature(trainval_loader, model)   
            if args.dataset == 'voc2007':
                y_trainval = torch.argmax(y_trainval,dim=1)
            print(f'x_trainval shape:{X_trainval_feature.shape} and y_trainval shape:{y_trainval.shape}')
            
            np.save(model_npy_feature, X_trainval_feature.numpy())
            np.save(model_npy_label, y_trainval.numpy())
            print(f"Features and Labels of {args.model} on {args.dataset} has been saved.")
        
        