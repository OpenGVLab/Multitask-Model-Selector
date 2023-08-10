import torch
from torch.utils.data import DataLoader
# from torchvision.datasets import CocoCaptions
import torchvision.transforms as transforms
from transformers import VisionEncoderDecoderModel, EncoderDecoderConfig
from transformers import GPT2TokenizerFast, ViTImageProcessor, VisionEncoderDecoderModel
from datasets.image_caption_dataset_feature import CaptionDataset
import transformers
import argparse
from PIL import Image
import os
from nltk.translate.bleu_score import corpus_bleu
import numpy as np
from transformers import ViTModel, T5ForConditionalGeneration, ViTFeatureExtractor
from transformers import T5Tokenizer, T5ForConditionalGeneration

def forward_pass(all_loader, model, encoder_name, decoder_name = "bert"):
    features = []
    targets = []   #img name (测试顺序是否一样的)
    outputs = []
    print(decoder_name)
    def hook_fn_forward(module, input, output):
        features.append(input[0].detach().cpu())
    if decoder_name == 'bert':
        forward_hook = model.decoder.bert.encoder.layer[11].output.LayerNorm.register_forward_hook(hook_fn_forward)
    elif decoder_name == "bart":
        # decoder.bert.encoder.layer.23.output.LayerNorm
        forward_hook = model.decoder.model.decoder.layers[5].final_layer_norm.register_forward_hook(hook_fn_forward)
    elif decoder_name == "roberta":
        forward_hook = model.decoder.roberta.encoder.layer[11].output.LayerNorm.register_forward_hook(hook_fn_forward)
    else:
        print('no hook')
    
    model = model.eval()
    with torch.no_grad():
        cnt = 0
        for _, (imgs, encoded_captions, length) in enumerate(all_loader):
            imgs, encoded_captions = imgs.to('cuda'), encoded_captions.to('cuda')
            # model.eval()
            _ = model(pixel_values=imgs, labels=encoded_captions).loss
            cnt += 1
    forward_hook.remove()
    features = torch.cat([x.mean(dim=1) for x in features])
    return features.cpu()



encoder_name_list = ['vit','swinvit','swin2vit']
decoder_name_list = ['bert','bart','roberta']


for encoder_name_tmp in encoder_name_list:
    for decoder_name_tmp in decoder_name_list:
        if encoder_name_tmp == 'vit':
            encoder_name = "google/vit-base-patch16-224-in21k"
        elif encoder_name_tmp == 'swinvit':
            encoder_name = "microsoft/swin-base-patch4-window7-224-in22k"

        elif encoder_name_tmp == 'swin2vit':
            encoder_name = "microsoft/swinv2-base-patch4-window12-192-22k"
        else:
            print('no encoder')
        if decoder_name_tmp == 'bert':
            decoder_name = "bert-base-uncased"
        elif decoder_name_tmp == 'roberta':
            decoder_name = "roberta-base"
        elif decoder_name_tmp == 'bart':
            decoder_name = "facebook/bart-base"
        else:
            print('no decoder')
        print(encoder_name,decoder_name)
        feature_extractor = transformers.AutoFeatureExtractor.from_pretrained(encoder_name)
        tokenizer = transformers.AutoTokenizer.from_pretrained(decoder_name)
        if encoder_name_tmp == 'vit':
            if decoder_name_tmp == 'bert':
                model = VisionEncoderDecoderModel.from_pretrained('/data/ckp/vit_bert_oncoco_tune_1e-05_1e-06_10') 
            elif decoder_name_tmp == 'bart':
                model = VisionEncoderDecoderModel.from_pretrained('/data/ckp/vit_bart_oncoco_tune_0.0001_1e-06_10') 

            elif decoder_name_tmp == 'roberta':
                model = VisionEncoderDecoderModel.from_pretrained('/data/ckp/vit_roberta_oncoco_tune_1e-05_1e-06_10') 
            else:
                print('no decoder')
        elif encoder_name_tmp == 'swinvit':
            if decoder_name_tmp == 'bert':
                print('/data/ckp/swinvit_bert_tune_onflickr8k1e-05_0.0001_10')
                model = VisionEncoderDecoderModel.from_pretrained('/data/ckp/swinvit_bert_tune_onflickr8k1e-05_0.0001_10') 
            elif decoder_name_tmp == 'bart':
                model = VisionEncoderDecoderModel.from_pretrained('/data/ckp/swinvit_bart_oncoco_tune_0.0001_1e-06_10') 

            elif decoder_name_tmp == 'roberta':
                model = VisionEncoderDecoderModel.from_pretrained('/data/ckp/swinvit_roberta_oncoco_tune_1e-05_1e-05_10') 
            else:
                print('no decoder')

        elif encoder_name_tmp == 'swin2vit':
            if decoder_name_tmp == 'bert':
                model = VisionEncoderDecoderModel.from_pretrained('/data/ckp/swin2vit_bert_oncoco_tune_1e-05_1e-06_10') 
            elif decoder_name_tmp == 'bart':
                model = VisionEncoderDecoderModel.from_pretrained('/data/ckp/swin2vit_bart_oncoco_tune_1e-05_1e-06_10') 

            elif decoder_name_tmp == 'roberta':
                model = VisionEncoderDecoderModel.from_pretrained('/data/ckp/swin2vit_roberta_oncoco_tune_1e-05_0.0001_10') 
            else:
                print('no decoder')
        else:
            print('no decoder')
        model.config.decoder_start_token_id = tokenizer.cls_token_id
        model.config.pad_token_id = tokenizer.pad_token_id
        model = model.cuda()
        model = model.eval()
        train_dataset = CaptionDataset(root_dir="/data/Flicker8k_Dataset/", annotations_file="/data/flickr8k/all.txt", feature_extractor = feature_extractor, tokenizer =tokenizer)
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=False,num_workers=4)
        X_trainval_feature = forward_pass(train_loader, model, encoder_name_tmp, decoder_name_tmp)  
        model_npy_feature = 'feature.npy'
        np.save(model_npy_feature, X_trainval_feature.numpy())
        print('saved!!',encoder_name_tmp,decoder_name_tmp)
        print('end')
