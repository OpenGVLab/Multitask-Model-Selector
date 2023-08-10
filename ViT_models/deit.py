import torch
import torch.nn as nn
from functools import partial

from timm.models.vision_transformer import VisionTransformer, _cfg
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_

import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), './'))

__all__ = [
    'deit_tiny', 'deit_small', 'deit_base',
    'dino_small', 'dino_base',
]

model_urls = {
    'deit_tiny': '/ViT_models/checkpoints/deit_tiny_patch16_224-a1311bcf.pth',
    'deit_small': '/ViT_models/checkpoints/deit_small_patch16_224-cd65a155.pth',
    'deit_base': '/ViT_models/checkpoints/deit_base_patch16_224-b5f2ef4d.pth',
    'dino_small': '/ViT_models/checkpoints/dino_deitsmall16_pretrain.pth',
    'dino_base': '/ViT_models/checkpoints/dino_vitbase16_pretrain.pth',
}

def _init_weights(m):
    if isinstance(m, nn.Linear):
        trunc_normal_(m.weight, std=.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.constant_(m.bias, 0)

@register_model
def deit_tiny(pretrained=False, task_cls=10, **kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    embed_dim = model.head.in_features
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.load(
            model_urls['deit_tiny'],
            map_location="cpu"
            )
        model.load_state_dict(checkpoint["model"])
        print("Pretrained Models Loaded")
    num_classes = task_cls
    model.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
    _init_weights(model.head)
    return model


@register_model
def deit_small(pretrained=False, task_cls=10, **kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    embed_dim = model.head.in_features
    model.default_cfg = _cfg()

    #del model.head
    if pretrained:
        checkpoint = torch.load(
            model_urls['deit_small'],
            map_location="cpu"
            )
        model.load_state_dict(checkpoint["model"])
        print("Pretrained Models Loaded")
    num_classes = task_cls
    model.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
    _init_weights(model.head)

    return model

@register_model
def deit_base(pretrained=False, task_cls=10, **kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    embed_dim = model.head.in_features
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.load(
            model_urls['deit_base'],
            map_location="cpu"
            )
        model.load_state_dict(checkpoint["model"])
        print("Pretrained Models Loaded")
    num_classes = task_cls
    model.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
    _init_weights(model.head)

    return model

@register_model
def dino_small(pretrained=False, task_cls=10, **kwargs):
    
    model = VisionTransformer(
        patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    embed_dim = model.head.in_features
    model.default_cfg = _cfg()
    del model.head
    #print(model)
    if pretrained:
        checkpoint = torch.load(
            model_urls['dino_small'],
            map_location="cpu"
            )
        model.load_state_dict(checkpoint)
        print("Pretrained Models Loaded")

    num_classes = task_cls
    model.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
    _init_weights(model.head)
    
    return model

@register_model
def dino_base(pretrained=False, task_cls=10, **kwargs):
    
    model = VisionTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    embed_dim = model.head.in_features
    model.default_cfg = _cfg()
    del model.head
    #print(model)
    if pretrained:
        checkpoint = torch.load(
            model_urls['dino_base'],
            map_location="cpu"
            )
        model.load_state_dict(checkpoint)
        print("Pretrained Models Loaded")
    num_classes = task_cls
    model.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
    _init_weights(model.head)
    
    return model



if __name__ == '__main__':
    model = dino_small(pretrained=False)
    fc_layer = model.fc_norm
    features = []
    def hook_fn_forward(module, input, output):
        #features.append(input[0].detach().cpu())
        # print(input.shape,'input.shape')
        # print(input[0].detach().shape,'input[0].detach().shape')  torch.Size([256, 1024])
        features.append(input[0].detach().cpu())
    forward_hook = fc_layer.register_forward_hook(hook_fn_forward)
    
    
    print(model)
    for n, m in model.named_modules():
        print(n,m)
    x = torch.rand(2,3,224,224)
    y=model(x)
    forward_hook.remove()
    print(y.size())
    
    print(len(features))
    print(features[0].shape)
    features = torch.cat([x[:, 0] for x in features])
    print(features.shape)
