#!/usr/bin/env python
# coding: utf-8

import os
import argparse
from pprint import pprint

import torch
import torch.nn as nn
import numpy as np
from utils import load_model, forward_pass
from get_dataloader import prepare_data, get_data


def setup_distributed(backend="nccl", port=29595):
    """Initialize slurm distributed training environment. (from mmcv)"""
    proc_id = int(os.environ["SLURM_PROCID"])
    ntasks = int(os.environ["SLURM_NTASKS"])
    node_list = os.environ["SLURM_NODELIST"]
    num_gpus = torch.cuda.device_count()
    torch.cuda.set_device(proc_id % num_gpus)
    addr = subprocess.getoutput(f"scontrol show hostname {node_list} | head -n1")
    # specify master port
    if port is not None:
        os.environ["MASTER_PORT"] = str(port)
    elif "MASTER_PORT" in os.environ:
        pass  # use MASTER_PORT in the environment variable
    else:
        os.environ["MASTER_PORT"] = "29500"
    if "MASTER_ADDR" not in os.environ:
        os.environ["MASTER_ADDR"] = addr
    os.environ["WORLD_SIZE"] = str(ntasks)
    os.environ["LOCAL_RANK"] = str(proc_id % num_gpus)
    os.environ["RANK"] = str(proc_id)

    dist.init_process_group(backend=backend)


# Main code
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Extract feature for supervised models.')
    parser.add_argument('-m', '--model', type=str, default='resnet50',
                        help='name of the pretrained model to load and evaluate (resnet50 | supervised)')
    parser.add_argument('-d', '--dataset', type=str, default='cars', 
                        help='name of the dataset to evaluate on')
    parser.add_argument('-b', '--batch-size', type=int, default=256, 
                        help='the size of the mini-batches when inferring features')
    parser.add_argument('-i', '--image-size', type=int, default=224, 
                        help='the size of the input images')
    parser.add_argument('--device', type=str, default='cuda', 
                        help='CUDA or CPU training (cuda | cpu)')
    parser.add_argument('-n', '--no-norm', action='store_true', default=False,
                        help='whether to turn off data normalisation (based on ImageNet values)')
    
    args = parser.parse_args()
    args.norm = not args.no_norm
    pprint(args)

    # load dataset
    dset, data_dir, num_classes, metric = get_data(args.dataset)
    args.num_classes = num_classes
    
    train_loader, val_loader, trainval_loader, test_loader, all_loader = prepare_data(
        dset, data_dir, args.batch_size, args.image_size, normalisation=args.norm)
    
    print(f'Train:{len(train_loader.dataset)}, Val:{len(val_loader.dataset)},' 
            f'TrainVal:{len(trainval_loader.dataset)}, Test:{len(test_loader.dataset)} '
            f'AllData:{len(all_loader.dataset)}')
    
    trainval_loader = all_loader

    fpath = os.path.join('./results_f_another', 'group1')
    if not os.path.exists(fpath):
        os.makedirs(fpath)  
    
    models_hub = ['inception_v3', 'mobilenet_v2', 'mnasnet1_0', 'densenet121', 'densenet169', 'densenet201', 
             'resnet34', 'resnet50', 'resnet101', 'resnet152', 'googlenet']
    for model in models_hub:
        args.model = model
        model_npy_feature = os.path.join(fpath, f'{args.model}_{args.dataset}_feature.npy')
        model_npy_label = os.path.join(fpath, f'{args.model}_{args.dataset}_label.npy')

        if os.path.exists(model_npy_feature) and os.path.exists(model_npy_label):
            print(f"Features and Labels of {args.model} on {args.dataset} has been saved.")
            continue
        model, fc_layer, _ = load_model(args)
        model = nn.DataParallel(model)
        X_trainval_feature, X_output, y_trainval = forward_pass(trainval_loader, model, fc_layer)   
        # print(X_trainval_feature.shape,X_output.shape,y_trainval.shape)
        
        #X_trainval_feature, y_trainval = forward_pass_feature(trainval_loader, model)   
        if args.dataset == 'voc2007':
            y_trainval = torch.argmax(y_trainval, dim=1)
        print(f'x_trainval shape:{X_trainval_feature.shape} and y_trainval shape:{y_trainval.shape}')
        
        np.save(model_npy_feature, X_trainval_feature.numpy())
        np.save(model_npy_label, y_trainval.numpy())
        print(f"Features and Labels of {args.model} on {args.dataset} has been saved.")
      