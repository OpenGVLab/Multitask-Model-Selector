#!/usr/bin/env python
# coding: utf-8

import os
import sys
from math import sqrt

from collections import OrderedDict

import torch
import torch.nn as nn
import models.group4 as models

import numpy as np
from sklearn.metrics import precision_recall_curve
import logging


def get_logger0(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    fh = logging.FileHandler(filename, "a+")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger


def voc_ap(rec, prec):
    """
    average precision calculations for PASCAL VOC 2007 metric, 11-recall-point based AP
    [precision integrated to recall]
    :param rec: recall
    :param prec: precision
    :return: average precision
    """
    ap = 0.
    for t in np.linspace(0, 1, 11):
        if np.sum(rec >= t) == 0:
            p = 0
        else:
            p = np.max(prec[rec >= t])
        ap += p / 11.
    return ap


def voc_eval_cls(y_true, y_pred):
    # get precision and recall
    prec, rec, _ = precision_recall_curve(y_true, y_pred)
    # compute average precision
    ap = voc_ap(rec, prec)
    return ap


class ResNetBackbone(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.model_name = model_name

        self.model = models.resnet50(pretrained=False)
        del self.model.fc

        state_dict = torch.load(os.path.join('models', self.model_name + '.pth'))
        self.model.load_state_dict(state_dict)

        self.model.eval()
        print("Number of model parameters:", sum(p.numel() for p in self.model.parameters()))

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)

        x = self.model.avgpool(x)
        x = torch.flatten(x, 1)

        return x


def load_model(configs):   
    print("Using torchvision Pretrained Models")
    if configs.model in ('inception_v3', 'googlenet'):
        model = models.__dict__[configs.model](pretrained=True, aux_logits=False).cuda()
    else:
        model = models.__dict__[configs.model](pretrained=True).cuda()

    if configs.model in ['mobilenet_v2', 'mnasnet1_0']:
        fc_layer = model.classifier[-1]
    elif configs.model in ['densenet121', 'densenet169', 'densenet201']:
        fc_layer = model.classifier
    elif configs.model in ['resnet34', 'resnet50', 'resnet101', 'resnet152', 'googlenet', 'inception_v3']:
        fc_layer = model.fc
    else:
        # try your customized model
        raise NotImplementedError
    feature_dim = fc_layer.in_features
    return model, fc_layer, feature_dim


def load_model_group4_norm(configs):   
    print("Using torchvision Pretrained Models")
    if configs.model in ('inception_v3', 'googlenet'):
        model = models.__dict__[configs.model](pretrained=True, aux_logits=False).cuda()
    else:
        model = models.__dict__[configs.model](pretrained=True).cuda()

    if configs.model in ['swin_t', 'swin_s','swin_b','pvt_tiny','pvt_small','pvt_medium','deit_tiny', 'deit_small', 'deit_base','dino_small', 'dino_base','mocov3_small']:
        fc_layer = model.norm
    elif configs.model in ['pvtv2_b3', 'pvtv2_b2']:
        fc_layer = model.norm4
    # elif configs.model in ['resnet34', 'resnet50', 'resnet101', 'resnet152', 'googlenet', 'inception_v3']:
    #     fc_layer = model.fc
    else:
        # try your customized model
        raise NotImplementedError
    # feature_dim = fc_layer.in_features
    # return model, fc_layer, feature_dim
    return model, fc_layer

def forward_pass(score_loader, model, fc_layer, model_name='resnet50'):
    """
    a forward pass on target dataset
    :params score_loader: the dataloader for scoring transferability
    :params model: the model for scoring transferability
    :params fc_layer: the fc layer of the model, for registering hooks
    returns
        features: extracted features of model
        outputs: outputs of model
        targets: ground-truth labels of dataset
    """
    features = []
    outputs = []
    targets = []
    model = model.cuda()

    def hook_fn_forward(module, input, output):
        #features.append(input[0].detach().cpu())
        # print(input.shape,'input.shape')
        # print(input[0].detach().shape,'input[0].detach().shape')  torch.Size([256, 1024])
        features.append(input[0].detach().cpu())
        #outputs.append(output.detach().cpu())

    forward_hook = fc_layer.register_forward_hook(hook_fn_forward)

    model.eval()
    with torch.no_grad():
        for _, (data, target) in enumerate(score_loader):
            targets.append(target)
            data = data.cuda()
            _ = model(data)

    forward_hook.remove()
    if model_name in ['pvt_tiny', 'pvt_small', 'pvt_medium', 'deit_small', 
                    'deit_tiny', 'deit_base', 'dino_base', 'dino_small', 
                    'mocov3_small']:
        features = torch.cat([x[:, 0] for x in features])

    elif model_name in ['pvtv2_b2', 'pvtv2_b3']:
        features = torch.cat([x.mean(dim=1) for x in features])
    
    elif model_name in ['swin_t', 'swin_s']:
        avgpool = nn.AdaptiveAvgPool1d(1).cuda()
        features = torch.cat([torch.flatten(avgpool(x.transpose(1, 2)), 1) for x in features])

    else:
        features = torch.cat([x for x in features])
    # outputs = torch.cat([x for x in outputs])
    
    targets = torch.cat([x for x in targets])
    outputs = targets   ## no use  ,上面一句报错，这玩意也不用，随便附一个只
    
    return features.cpu(), outputs, targets

def forward_pass_anorm(score_loader, model, fc_layer, model_name='resnet50'):
    """
    a forward pass on target dataset
    :params score_loader: the dataloader for scoring transferability
    :params model: the model for scoring transferability
    :params fc_layer: the fc layer of the model, for registering hooks
    returns
        features: extracted features of model
        outputs: outputs of model
        targets: ground-truth labels of dataset
    """
    features = []
    outputs = []
    targets = []
    model = model.cuda()

    def hook_fn_forward(module, input, output):
        #features.append(input[0].detach().cpu())
        # print(input.shape,'input.shape')
        # print(input[0].detach().shape,'input[0].detach().shape')  torch.Size([256, 1024])
        features.append(input[0].detach().cpu())
        #outputs.append(output.detach().cpu())

    forward_hook = fc_layer.register_forward_hook(hook_fn_forward)

    model.eval()
    with torch.no_grad():
        for _, (data, target) in enumerate(score_loader):
            targets.append(target)
            data = data.cuda()
            _ = model(data)

    forward_hook.remove()
    features = torch.cat([x for x in features])
    # outputs = torch.cat([x for x in outputs])
    
    targets = torch.cat([x for x in targets])
    outputs = targets   ## no use  ,上面一句报错，这玩意也不用，随便附一个只
    
    return features.cpu(), outputs, targets


def forward_pass_feature(score_loader, model):
    """
    a forward pass on target dataset
    :params score_loader: the dataloader for scoring transferability
    :params model: the model for scoring transferability
    returns
        features: extracted features of model
        targets: ground-truth labels of dataset
    """
    features = []
    targets = []

    model.eval()
    with torch.no_grad():
        for _, (data, target) in enumerate(score_loader):
            targets.append(target)
            data = data.cuda()
            _features = model(data)
            features.append(_features)

    features = torch.cat([x for x in features])
    targets = torch.cat([x for x in targets])

    return features.detach().cpu(), targets.detach().cpu()


def initLabeled(y, p=0.2):
    # random selected the labeled instances' index
    n = len(y)
    labeledIndex = []
    labelDict = OrderedDict()
    for label in np.unique(y):
        labelDict[label] = []
    for i, label in enumerate(y):
        labelDict[label].append(i)
    for value in labelDict.values():
        #print(len(value))
        for idx in np.random.choice(value, size=int(p*len(value)), replace=False, p=None):
            labeledIndex.append(idx)
    return labeledIndex


def KA(feat1, feat2, remove_mean=True):
    """
        feat1, feat2: n x d
    """
    from numpy.linalg import norm
    if remove_mean:
        feat1 -= np.mean(feat1, axis=0, keepdims=1)
        feat2 -= np.mean(feat2, axis=0, keepdims=1)
    norm12 = norm(feat1.T.dot(feat2))**2
    norm11 = norm(feat1.T.dot(feat1))
    norm22 = norm(feat2.T.dot(feat2))
    return norm12 / (norm11 * norm22)


def compute_sim(feat_files):
    N = len(feat_files)
    sim = np.eye(N)
    for i in range(N):
        feat_i = np.load(feat_files[i])
        for j in range(i+1, N):
            feat_j = np.load(feat_files[j])
            sim[i, j] = KA(feat_i, feat_j, remove_mean=True)
            sim[j, i] = sim[i, j]
            print(i, j, sim[i, j], flush=True)
    return sim


def iterative_A(A, max_iterations=3):
    '''
    calculate the largest eigenvalue of A
    '''
    x = A.sum(axis=1)
    #k = 3
    for _ in range(max_iterations):
        temp = np.dot(A, x)
        y = temp / np.linalg.norm(temp, 2)
        temp = np.dot(A, y)
        x = temp / np.linalg.norm(temp, 2)
    return np.dot(np.dot(x.T, A), y)


def wpearson(vec_1, vec_2, weights=None, r=4):
    if weights is None:
        weights = [len(vec_1)-i for i in range(len(vec_1))]
    list_length = len(vec_1)
    weights = list(map(float, weights))
    vec_1 = list(map(float, vec_1))
    vec_2 = list(map(float, vec_2))
    if any(len(x) != list_length for x in [vec_2, weights]):
        print('Vector/Weight sizes not equal.')
        sys.exit(1)
    w_sum = sum(weights)

    # Calculate the weighted average relative value of vector 1 and vector 2.
    vec1_sum = 0.0
    vec2_sum = 0.0
    for x in range(len(vec_1)):
        vec1_sum += (weights[x] * vec_1[x])
        vec2_sum += (weights[x] * vec_2[x])	
    vec1_avg = (vec1_sum / w_sum)
    vec2_avg = (vec2_sum / w_sum)

    # Calculate wPCC
    sum_top = 0.0
    sum_bottom1 = 0.0
    sum_bottom2 = 0.0
    for x in range(len(vec_1)):
        dif_1 = (vec_1[x] - vec1_avg)
        dif_2 = (vec_2[x] - vec2_avg)
        sum_top += (weights[x] * dif_1 * dif_2)
        sum_bottom1 += (dif_1 ** 2 ) * (weights[x])
        sum_bottom2 += (dif_2 ** 2) * (weights[x])

    cor = sum_top / (sqrt(sum_bottom1 * sum_bottom2))
    return round(cor, r)
 