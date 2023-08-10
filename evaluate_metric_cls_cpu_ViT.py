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


finetune_acc = {'aircraft': {'deit_tiny': 71.26, 'deit_small': 73.12, 'deit_base': 78.39, 'dino_small': 72.18, 'dino_base': 67.13, 'mocov3_small': 76.04, 'pvtv2_b2': 84.14, 'pvtv2_b3': 84.7, 'pvt_tiny': 69.76, 'pvt_small': 75.2, 'pvt_medium': 76.7, 'swin_t': 81.9, 'swin_s': 83.24}, 
                'caltech101': {'deit_tiny': 89.39, 'deit_small': 92.7, 'deit_base': 93.47, 'dino_small': 86.76, 'dino_base': 92.34, 'mocov3_small': 89.84, 'pvtv2_b2': 93.13, 'pvtv2_b3': 94.4, 'pvt_tiny': 90.04, 'pvt_small': 93.02, 'pvt_medium': 93.75, 'swin_t': 91.9, 'swin_s': 94.0}, 
                'cars': {'deit_tiny': 82.09, 'deit_small': 86.72, 'deit_base': 89.26, 'dino_small': 79.81, 'dino_base': 80.74, 'mocov3_small': 82.18, 'pvtv2_b2': 90.6, 'pvtv2_b3': 91.22, 'pvt_tiny': 84.1, 'pvt_small': 87.61, 'pvt_medium': 87.66, 'swin_t': 88.93, 'swin_s': 89.81}, 
                'cifar10': {'deit_tiny': 96.52, 'deit_small': 97.69, 'deit_base': 98.56, 'dino_small': 97.96, 'dino_base': 98.31, 'mocov3_small': 97.92, 'pvtv2_b2': 97.96, 'pvtv2_b3': 98.44, 'pvt_tiny': 94.87, 'pvt_small': 97.34, 'pvt_medium': 97.93, 'swin_t': 97.34, 'swin_s': 98.06}, 
                'cifar100': {'deit_tiny': 81.58, 'deit_small': 86.62, 'deit_base': 89.96, 'dino_small': 85.66, 'dino_base': 89.38, 'mocov3_small': 85.84, 'pvtv2_b2': 88.24, 'pvtv2_b3': 89.3, 'pvt_tiny': 75.26, 'pvt_small': 86.2, 'pvt_medium': 87.36, 'swin_t': 85.97, 'swin_s': 88.42}, 
                'dtd': {'deit_tiny': 71.86, 'deit_small': 75.08, 'deit_base': 77.66, 'dino_small': 75.96, 'dino_base': 76.01, 'mocov3_small': 71.88, 'pvtv2_b2': 77.16, 'pvtv2_b3': 77.37, 'pvt_tiny': 72.92, 'pvt_small': 75.77, 'pvt_medium': 77.1, 'swin_t': 77.04, 'swin_s': 77.34}, 
                'flowers': {'deit_tiny': 95.5, 'deit_small': 96.79, 'deit_base': 97.98, 'dino_small': 95.96, 'dino_base': 96.28, 'mocov3_small': 93.89, 'pvtv2_b2': 97.89, 'pvtv2_b3': 98.06, 'pvt_tiny': 95.8, 'pvt_small': 97.32, 'pvt_medium': 97.36, 'swin_t': 97.4, 'swin_s': 96.87}, 
                'food': {'deit_tiny': 81.96, 'deit_small': 86.26, 'deit_base': 88.96, 'dino_small': 85.69, 'dino_base': 87.1, 'mocov3_small': 82.84, 'pvtv2_b2': 88.67, 'pvtv2_b3': 89.08, 'pvt_tiny': 83.78, 'pvt_small': 86.98, 'pvt_medium': 85.56, 'swin_t': 86.67, 'swin_s': 87.7}, 
                'pets': {'deit_tiny': 91.44, 'deit_small': 94.02, 'deit_base': 94.61, 'dino_small': 92.59, 'dino_base': 93.41, 'mocov3_small': 90.44, 'pvtv2_b2': 93.86, 'pvtv2_b3': 95.14, 'pvt_tiny': 91.48, 'pvt_small': 94.13, 'pvt_medium': 94.48, 'swin_t': 94.5, 'swin_s': 94.8}, 
                'sun397': {'deit_tiny': 58.4, 'deit_small': 64.76, 'deit_base': 68.62, 'dino_small': 64.14, 'dino_base': 64.78, 'mocov3_small': 60.6, 'pvtv2_b2': 66.44, 'pvtv2_b3': 67.54, 'pvt_tiny': 61.86, 'pvt_small': 65.78, 'pvt_medium': 67.22, 'swin_t': 65.51, 'swin_s': 67.03}, 
                'voc2007': {'deit_tiny': 83.1, 'deit_small': 86.62, 'deit_base': 87.88, 'dino_small': 84.8, 'dino_base': 86.72, 'mocov3_small': 81.84, 'pvtv2_b2': 86.44, 'pvtv2_b3': 88.08, 'pvt_tiny': 84.6, 'pvt_small': 86.62, 'pvt_medium': 87.36, 'swin_t': 87.54, 'swin_s': 88.26}
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
    models_hub = ['deit_base', 'deit_tiny', 'deit_small',
    'dino_small', 'mocov3_small', 
    'pvtv2_b2', 
    'pvt_tiny', 'pvt_small', 'pvt_medium', 
     'swin_t'
    ]

    datasets_hub = ['aircraft','caltech101','cars','cifar10','cifar100','dtd','flowers','food','pets','sun397','voc2007']
    for dataset in datasets_hub:
        start_time = time.time()
        args.dataset = dataset
        finetune = []
        score = []
        score_dict = {}   

        for model in models_hub:
            args.model = model
            model_npy_feature = os.path.join('/data/results_f/group4_bnorm', f'{args.model}_{args.dataset}_feature.npy')
            model_npy_label = os.path.join('/data/results_f/group4_bnorm', f'{args.model}_{args.dataset}_label.npy')
            X_features, y_labels = np.load(model_npy_feature), np.load(model_npy_label)

            print(y_labels.max())
            
            embedding_npy_label = f'/{args.dataset}_nonorm_bert.npy'
            embedding_npy_label2 = f'/{args.dataset}_clip_1024_nonorm.npy'
            embedding_npy_label3 = f'/{args.dataset}_gpt2_1024_nonorm.npy'
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
