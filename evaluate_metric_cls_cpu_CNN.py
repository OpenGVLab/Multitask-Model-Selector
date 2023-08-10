#!/usr/bin/env python
# coding: utf-8

import os
import argparse
from pprint import pprint

import torch
import models.group1 as models
import numpy as np
from scipy.stats import weightedtau
import json
import time

from metrics import LEEP, NLEEP, LogME_Score, SFDA_Score, PARC_Score, LogME_optimal, EMMS
from scipy.stats import kendalltau
from scipy.stats import weightedtau
import pprint
import json
from scipy.stats import pearsonr
from w_pearson import wpearson

def save_score(score_dict, fpath):
    with open(fpath, "w") as f:
        # write dict 
        json.dump(score_dict, f)


def exist_score(model_name, fpath):
    with open(fpath, "r") as f:
        result = json.load(f)
        if model_name in result.keys():
            return True
        else:
            return False

def recall_k(score, dset, k):
    #succed = 0
    sorted_score = sorted(score.items(), key=lambda i: i[1], reverse=True)
    sorted_score = {a[0]:a[1] for a in sorted_score}
    
    gt = finetune_acc[dset]
    sorted_gt = sorted(gt.items(), key=lambda i: i[1], reverse=True)
    sorted_gt = {a[0]:a[1] for a in sorted_gt}

    top_k_gt = sorted_gt.keys()[:k]
    succed = 1 if sorted_score.keys()[0] in top_k_gt else 0
    return succed

def rel_k(score, dset, k):
    sorted_score = sorted(score.items(), key=lambda i: i[1], reverse=True)
    
    gt = finetune_acc[dset]
    sorted_gt = sorted(gt.items(), key=lambda i: i[1], reverse=True)
    best_model = sorted_gt[0][0]
    sorted_gt = {a[0]:a[1] for a in sorted_gt}

    max_gt = sorted_gt[best_model]
    topk_score_model = [a[0] for i, a in enumerate(sorted_score) if i < k]
    topk_score_ft = [sorted_gt[a] for a in topk_score_model]
    return max(topk_score_ft) / max_gt

def pearson_coef(score,dset):
    global finetune_acc
    score = score.items()
    metric_score = [a[1] for a in score]
    gt = finetune_acc[dset]
    gt_ = []
    for a in score:
        gt_.append(gt[a[0]])
    tw_metric,_ = pearsonr(metric_score,gt_)
    return tw_metric

def wpearson_coef(score,dset):
    global finetune_acc
    score = score.items()
    metric_score = [a[1] for a in score]
    gt = finetune_acc[dset]
    gt_ = []
    for a in score:
        gt_.append(gt[a[0]])
    tw_metric = wpearson(metric_score,gt_)
    return tw_metric

def w_kendall_metric(score, dset):
    global finetune_acc
    score = score.items()
    metric_score = [a[1] for a in score]
    gt = finetune_acc[dset]
    gt_ = []
    for a in score:
        gt_.append(gt[a[0]])
    tw_metric,_ = weightedtau(metric_score,gt_)
    return tw_metric

def kendall_metric(score, dset):
    global finetune_acc_ssl
    score = score.items()
    metric_score = [a[1] for a in score]
    gt = finetune_acc[dset]
    gt_ = []
    for a in score:
        gt_.append(gt[a[0]])
    t_metric,_ = kendalltau(metric_score,gt_)
    return t_metric


finetune_acc = {
        'aircraft': {'resnet34': 84.06, 'resnet50': 84.64, 'resnet101': 85.53, 'resnet152': 86.29, 'densenet121': 84.66, 
                    'densenet169': 84.19, 'densenet201': 85.38, 'mnasnet1_0': 66.48, 'mobilenet_v2': 79.68, 
                    'googlenet': 80.32, 'inception_v3': 80.15}, 
        'caltech101': {'resnet34': 91.15, 'resnet50': 91.98, 'resnet101': 92.38, 'resnet152': 93.1, 'densenet121': 91.5, 
                    'densenet169': 92.51, 'densenet201': 93.14, 'mnasnet1_0': 89.34, 'mobilenet_v2': 88.64, 
                    'googlenet': 90.85, 'inception_v3': 92.75}, 
        'cars': {'resnet34': 88.63, 'resnet50': 89.09, 'resnet101': 89.47, 'resnet152': 89.88, 'densenet121': 89.34, 
                    'densenet169': 89.02, 'densenet201': 89.44, 'mnasnet1_0': 72.58, 'mobilenet_v2': 86.44, 
                    'googlenet': 87.76, 'inception_v3': 87.74}, 
        'cifar10': {'resnet34': 96.12, 'resnet50': 96.28, 'resnet101': 97.39, 'resnet152': 97.53, 'densenet121': 96.45, 
                    'densenet169': 96.77, 'densenet201': 97.02, 'mnasnet1_0': 92.59, 'mobilenet_v2': 94.74, 
                    'googlenet': 95.54, 
                    'inception_v3': 96.18}, 
        'cifar100': {'resnet34': 81.94, 'resnet50': 82.8, 'resnet101': 84.88, 'resnet152': 85.66, 'densenet121': 82.75, 
                    'densenet169': 84.26, 'densenet201': 84.88, 'mnasnet1_0': 72.04, 'mobilenet_v2': 78.11, 
                    'googlenet': 79.84, 
                    'inception_v3': 81.49}, 
        'dtd': {'resnet34': 72.96, 'resnet50': 74.72, 'resnet101': 74.8, 'resnet152': 76.44, 'densenet121': 74.18, 
                    'densenet169': 74.72, 'densenet201': 76.04, 'mnasnet1_0': 70.12, 'mobilenet_v2': 71.72, 
                    'googlenet': 72.53, 
                    'inception_v3': 72.85}, 
        'flowers': {'resnet34': 95.2, 'resnet50': 96.26, 'resnet101': 96.53, 'resnet152': 96.86, 'densenet121': 97.02, 
                    'densenet169': 97.32, 'densenet201': 97.1, 'mnasnet1_0': 95.39, 'mobilenet_v2': 96.2, 
                    'googlenet': 95.76, 
                    'inception_v3': 95.73},
        'food': {'resnet34': 81.99, 'resnet50': 84.45, 'resnet101': 85.58, 'resnet152': 86.28, 'densenet121': 84.99, 
                    'densenet169': 85.84, 'densenet201': 86.71, 'mnasnet1_0': 71.35, 'mobilenet_v2': 81.12, 
                    'googlenet': 79.3, 
                    'inception_v3': 81.76}, 
        'pets': {'resnet34': 93.5, 'resnet50': 93.88, 'resnet101': 93.92, 'resnet152': 94.42, 'densenet121': 93.07, 
                    'densenet169': 93.62, 'densenet201': 94.03, 'mnasnet1_0': 91.08, 'mobilenet_v2': 91.28, 
                    'googlenet': 91.38, 
                    'inception_v3': 92.14},
        'sun397': {'resnet34': 61.02, 'resnet50': 63.54, 'resnet101': 63.76, 'resnet152': 64.82, 'densenet121': 63.26, 
                    'densenet169': 64.1, 'densenet201': 64.57, 'mnasnet1_0': 56.56, 'mobilenet_v2': 60.29, 
                    'googlenet': 59.89, 
                    'inception_v3': 59.98}, 
        'voc2007': {'resnet34': 84.6, 'resnet50': 85.8, 'resnet101': 85.68, 'resnet152': 86.32, 'densenet121': 85.28, 
                    'densenet169': 85.77, 'densenet201': 85.67, 'mnasnet1_0': 81.06, 'mobilenet_v2': 82.8, 
                    'googlenet': 82.58, 
                    'inception_v3': 83.84}
        }


# Main code
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Calculate transferability score.')
    parser.add_argument('-m', '--model', type=str, default='deepcluster-v2',
                        help='name of the pretrained model to load and evaluate (deepcluster-v2 | supervised)')
    parser.add_argument('-d', '--dataset', type=str, default='dtd', 
                        help='name of the dataset to evaluate on')
    parser.add_argument('-me', '--metric', type=str, default='logme', 
                        help='name of the method for measuring transferability')   
    parser.add_argument('--nleep-ratio', type=float, default=5, 
                        help='the ratio of the Gaussian components and target data classess')
    parser.add_argument('--parc-ratio', type=float, default=2,
                        help='PCA reduction dimension')
    parser.add_argument('--output_dir', type=str, default='./results_metrics/group1', 
                        help='dir of output score')
    args = parser.parse_args()   
    print(args)

    score_dict = {}   
    fpath = os.path.join(args.output_dir, args.metric)
    if not os.path.exists(fpath):
        os.makedirs(fpath)
    fpath = os.path.join(fpath, f'{args.dataset}_metrics_clip.json')
    finetune = []
    score = []
    models_hub = ['inception_v3', 'mobilenet_v2', 'mnasnet1_0', 'densenet121', 'densenet169', 'densenet201', 
                    'resnet34', 'resnet50', 'resnet101', 'resnet152', 'googlenet']

    datasets_hub = ['aircraft','caltech101','cars','cifar10','cifar100','dtd','flowers','food','pets','sun397','voc2007']
    for dataset in datasets_hub:
        start_time = time.time()
        args.dataset = dataset
        finetune = []
        score = []
        score_dict = {}   

        for model in models_hub:
            args.model = model
            model_npy_feature = os.path.join('/data/results_f/group1', f'{args.model}_{args.dataset}_feature.npy')
            model_npy_label = os.path.join('/data/results_f/group1', f'{args.model}_{args.dataset}_label.npy')
            X_features, y_labels = np.load(model_npy_feature), np.load(model_npy_label)

            print(y_labels.max())
            
            embedding_npy_label = f'{args.dataset}_bert_1024_nonorm.npy'
            embedding_npy_label2 = f'{args.dataset}_clip_1024_nonorm.npy'
            embedding_npy_label3 = f'{args.dataset}_gpt2_1024_nonorm.npy'
            embedding_label = np.load(embedding_npy_label)  #47,512
            embedding_label2 = np.load(embedding_npy_label2)  #47,512
            embedding_label3 = np.load(embedding_npy_label3)  #47,512
            y_labels_0 = np.zeros([y_labels.shape[0],embedding_label.shape[1]])
            y_labels_1 = np.zeros([y_labels.shape[0],embedding_label2.shape[1]])
            y_labels_2 = np.zeros([y_labels.shape[0],embedding_label3.shape[1]])
            for i in range(y_labels.shape[0]):
                y_labels_0[i] = embedding_label[y_labels[i]]
                y_labels_1[i] = embedding_label2[y_labels[i]]
                y_labels_2[i] = embedding_label3[y_labels[i]]
            # y_labels = y_labels2
            y_labels1 = np.stack((y_labels_0, y_labels_1, y_labels_2), axis=2)
            print(X_features.shape,y_labels1.shape,dataset)
            args.metric = 'EMMS'
            if args.metric == 'EMMS':
                score_dict[args.model] = EMMS(X_features, y_labels1)
            elif args.metric == 'logme':
                score_dict[args.model] = LogME_Score(X_features, y_labels)
            elif args.metric == 'transrate':
                score_dict[args.model] = Transrate(X_features, y_labels)
            elif args.metric == 'leep':     
                score_dict[args.model] = LEEP(X_features, y_labels, model_name=args.model)
            elif args.metric == 'nleep':           
                ratio = 1 if args.dataset in ('food', 'pets') else args.nleep_ratio
                score_dict[args.model] = NLEEP(X_features, y_labels, component_ratio=ratio)  
            else:
                raise NotImplementedError
            finetune.append(finetune_acc[args.dataset][args.model])
            score.append(score_dict[args.model])
            print(f'{args.metric} of {args.model}: {score_dict[args.model]}\n')
        end_time = time.time()
        elapsed_time = end_time - start_time
        print("Elapsed time: {:.2f} s".format(elapsed_time))
        tw_metric, _ = weightedtau(score, finetune)
        print(tw_metric,args.dataset)
        results = sorted(score_dict.items(), key=lambda i: i[1], reverse=True)
        print(f'Models ranking on {args.dataset} based on {args.metric}: ')
        print(results)

        tw = w_kendall_metric(score_dict, args.dataset)
        t = kendall_metric(score_dict, args.dataset)
        pear = pearson_coef(score_dict, args.dataset)
        wpear = wpearson_coef(score_dict, args.dataset)

        rel_3 = rel_k(score_dict, args.dataset, k=3)
        rel_1 = rel_k(score_dict, args.dataset, k=1)
        # results = {a[0]: a[1] for a in results}
        # save_score(results, fpath)

        print("Rel@1    dataset:{:12s} our:{:2.3f}".format(args.dataset,rel_1))
        print("Rel@3    dataset:{:12s} our:{:2.3f}".format(args.dataset,rel_3))
        print("Pearson  dataset:{:12s} our:{:2.3f}".format(args.dataset,pear))
        print("WPearson dataset:{:12s} our:{:2.3f}".format(args.dataset,wpear))
        print("Kendall  dataset:{:12s} our:{:2.3f}".format(args.dataset,t))
        print("WKendall dataset:{:12s} our:{:2.3f}".format(args.dataset,tw))
    
        
        print('*'*80)
        # results = {a[0]: a[1] for a in results}
        # save_score(results, fpath)
