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

from metrics import LEEP, NLEEP, LogME_Score, SFDA_Score, PARC_Score, LogME_optimal, EMMS, TransRate


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

finetune_acc =  {
                'flickr8k': {'vit_bert': 0.18513, 'vit_roberta': 0.20527, 'vit_bart': 0.21897, 'swinvit_bert': 0.22913, 'swinvit_roberta': 0.23986, 'swinvit_bart': 0.2468, 'swin2vit_bert': 0.25686, 'swin2vit_roberta': 0.23402, 'swin2vit_bart': 0.26235},
                'flickr30k': {'vit_bert':0.26648, 'vit_roberta': 0.23701, 'vit_bart': 0.25134, 'swinvit_bert': 0.26614, 'swinvit_roberta': 0.28838, 'swinvit_bart': 0.28032, 'swin2vit_bert': 0.32331, 'swin2vit_roberta': 0.28814, 'swin2vit_bart': 0.30352}, 
                'RSICD': {'vit_bert': 0.30389, 'vit_roberta': 0.28921, 'vit_bart': 0.30347, 'swinvit_bert': 0.32539,'swinvit_roberta': 0.3207, 'swinvit_bart': 0.31989, 'swin2vit_bert': 0.34449, 'swin2vit_roberta': 0.35218, 'swin2vit_bart': 0.33715},  
                'flickr10kH': {'vit_bert': 0.04312, 'vit_roberta': 0.04882, 'vit_bart': 0.04753, 'swinvit_bert': 0.05245, 'swinvit_roberta': 0.06115, 'swinvit_bart': 0.05099, 'swin2vit_bert': 0.04863, 'swin2vit_roberta': 0.05799, 'swin2vit_bart': 0.069},  
                'flickr10kR': {'vit_bert': 0.04184, 'vit_roberta': 0.0448, 'vit_bart': 0.04526, 'swinvit_bert': 0.04665, 'swinvit_roberta': 0.04492, 'swinvit_bart': 0.04946, 'swin2vit_bert': 0.04489, 'swin2vit_roberta': 0.06134, 'swin2vit_bart': 0.04956},  
                }

# Main code
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Calculate transferability score.')
    parser.add_argument('-m', '--model', type=str, default='deepcluster-v2',
                        help='name of the pretrained model to load and evaluate (deepcluster-v2 | supervised)')
    parser.add_argument('-d', '--dataset', type=str, default='flickr10kR', 
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
    pprint(args)

    score_dict = {}   
    fpath = os.path.join(args.output_dir, args.metric)
    if not os.path.exists(fpath):
        os.makedirs(fpath)
    fpath = os.path.join(fpath, f'{args.dataset}_metrics_clip.json')

    if not os.path.exists(fpath):
        save_score(score_dict, fpath)
    else:
        with open(fpath, "r") as f:
            score_dict = json.load(f)
    finetune = []
    score = []
    encoders_hub = ['vit','swinvit','swin2vit']
    decoders_hub = ['bert','roberta','bart']
    for encoder in encoders_hub:
        for decoder in decoders_hub:
            model = encoder + '_' + decoder
            args.model = model
            model_npy_feature = os.path.join(f'/data/feature_bnorm_caption_1024/{args.dataset}_feature_bnorm', f'{args.model}_10.npy')
            model_npy_label = os.path.join(f'{args.dataset}_captions_nonorm_clip_1024.npy')
            model_npy_label2 = os.path.join(f'{args.dataset}_captions_nonorm_gpt2_1024.npy')
            model_npy_label3 = os.path.join(f'args.dataset}_captions_nonorm_bert_1024.npy')
            X_features, y_labels0 = np.load(model_npy_feature), np.load(model_npy_label)
            y_labels2 = np.load(model_npy_label2)
            y_labels3 = np.load(model_npy_label3)
            y_labels = np.stack((y_labels0, y_labels2, y_labels3), axis=2)
            print(y_labels.shape)
                
            print(f'x_trainval shape:{X_features.shape} and y_trainval shape:{y_labels.shape}')        
            print(f'Calc Transferabilities of {args.model} on {args.dataset}')
            args.metric = 'emms'
            if args.metric == 'logme':
                score_dict[args.model] = EMMS(X_features, y_labels)
            elif args.metric == 'logme_clip':
                score_dict[args.model] = LogME_Score(X_features, y_labels0)
            elif args.metric == 'logme_gpt2':
                score_dict[args.model] = LogME_Score(X_features, y_labels1)
            elif args.metric == 'logme_bert':
                score_dict[args.model] = LogME_Score(X_features, y_labels2)  
            else:
                raise NotImplementedError
            finetune.append(finetune_acc[args.dataset][args.model])
            score.append(score_dict[args.model])
            print(f'{args.metric} of {args.model}: {score_dict[args.model]}\n')
            # save_score(score_dict, fpath)
    tw_metric, _ = weightedtau(score, finetune)
    print(tw_metric,args.dataset)
    results = sorted(score_dict.items(), key=lambda i: i[1], reverse=True)
    print(f'Models ranking on {args.dataset} based on {args.metric}: ')
    print(results)
        # results = {a[0]: a[1] for a in results}
        # save_score(results, fpath)
