import os

import torch
import torch.nn as nn

from typing import Dict

from .layers import LoRALayer, PlainMultiheadAttentionLoRA, MultiheadAttentionLoLoRA_v3, MultiheadAttentionDoRA_v4,\
                    MultiheadAttentionAdaLoRA, MultiheadAttentionDyLoRA, MultiheadAttentionLoLoRA, MultiheadAttentionDoRA_v3, \
                    MultiheadAttentionLoLoRA_v2, PlainMultiheadAttention, MultiheadAttentionDoRA, MultiheadAttentionDoRA_v2, \
                    MultiheadAttentionDoRA_v5, MultiheadAttentionDoRA_v7, MultiheadAttentionDoRA_v7_2, MultiheadAttentionDoRA_v7_3, \
                    MultiheadAttentionDoRA_v8, MultiheadAttentionDoRA_v9, MultiheadAttentionNoRA_A, MultiheadAttentionNoRA_B, \
                    MultiheadAttentionNoRA_C
from .lora_lib import DoraLinear_v7, DoraLinear_v7_2, DoraLinear_v7_3, DoraLinear_v8, DoraLinear_v9, NoRALinear_A, NoRALinear_B, \
                    NoRALinear_C

INDEX_POSITIONS_TEXT = {
    'top1': [11],
    'top2': [10, 11],
    'top3': [9, 10, 11],
    'bottom': [0, 1, 2, 3],
    'mid': [4, 5, 6, 7],
    'up': [8, 9, 10, 11],
    'half-up': [6, 7, 8, 9, 10, 11],
    'half-bottom': [0, 1, 2, 3, 4, 5],
    'all': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]}


INDEX_POSITIONS_VISION = {
    'ViT-B/16': {
        'top': [11],
        'top3': [9, 10, 11],
        'bottom': [0, 1, 2, 3],
        'mid': [4, 5, 6, 7],
        'up': [8, 9, 10, 11],
        'half-up': [6, 7, 8, 9, 10, 11],
        'half-bottom': [0, 1, 2, 3, 4, 5],
        'all': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]},
    'ViT-B/32': {
        'bottom': [0, 1, 2, 3],
        'mid': [4, 5, 6, 7],
        'up': [8, 9, 10, 11],
        'half-up': [6, 7, 8, 9, 10, 11],
        'half-bottom': [0, 1, 2, 3, 4, 5],
        'all': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]},

    'ViT-L/14': {
        'bottom': [0, 1, 2, 3],
        'mid': [4, 5, 6, 7],
        'up': [8, 9, 10, 11],
        'half-up': [6, 7, 8, 9, 10, 11],
        'half-bottom': [0, 1, 2, 3, 4, 5],
        'all': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]}
}


def mark_only_lora_as_trainable(model: nn.Module, bias: str = 'none') -> None:
    for n, p in model.named_parameters():
        trainable_prefixes = ['lora_', 'vera_', 'dora_', 'side_a', 'side_b', 'W_A', 'W_B', "W_a", "W_b"]
        if not any(prefix in n for prefix in trainable_prefixes):
            p.requires_grad = False
    if bias == 'none':
        return
    elif bias == 'all':
        for n, p in model.named_parameters():
            if 'bias' in n:
                p.requires_grad = True
    elif bias == 'lora_only':
        for m in model.modules():
            if isinstance(m, LoRALayer) and \
                    hasattr(m, 'bias') and \
                    m.bias is not None:
                m.bias.requires_grad = True
    else:
        raise NotImplementedError


def lora_state_dict(model: nn.Module, bias: str = 'none') -> Dict[str, torch.Tensor]:
    my_state_dict = model.state_dict()
    if bias == 'none':
        return {k: my_state_dict[k] for k in my_state_dict if 'lora_' in k}
    elif bias == 'all':
        return {k: my_state_dict[k] for k in my_state_dict if 'lora_' in k or 'bias' in k}
    elif bias == 'lora_only':
        to_return = {}
        for k in my_state_dict:
            if 'lora_' in k:
                to_return[k] = my_state_dict[k]
                bias_name = k.split('lora_')[0]+'bias'
                if bias_name in my_state_dict:
                    to_return[bias_name] = my_state_dict[bias_name]
        return to_return
    else:
        raise NotImplementedError


def get_lora_parameters(model, bias='none'):
    params = []
    lora_prefixes = ['lora_', 'vera_', 'dora_', 'side_a', 'side_b', 'W_A', 'W_B', "W_a", "W_b"]
    
    def is_lora_param(name):
        return any(prefix in name for prefix in lora_prefixes)
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        
        if bias == 'none' and is_lora_param(name):
            params.append(param)
        elif bias == 'all' and (is_lora_param(name) or 'bias' in name):
            params.append(param)
        elif bias == 'lora_only':
            if is_lora_param(name):
                params.append(param)
                bias_name = name.split('lora_')[0] + 'bias'
                if bias_name in model.state_dict():
                    bias_param = dict(model.named_parameters())[bias_name]
                    if bias_param.requires_grad:
                        params.append(bias_param)
        else:
            raise NotImplementedError(f"不支持的偏置模式: {bias}")
    return params


def apply_lora(args, clip_model):
    list_lora_layers = []
    if args.encoder == 'text' or args.encoder == 'both':
        indices = INDEX_POSITIONS_TEXT[args.position]
        text_encoder = clip_model.transformer
        for i, block in enumerate(text_encoder.resblocks):
            
            if i in indices:
                for name, submodule in block.named_children():
                    if isinstance(submodule, nn.MultiheadAttention):
                        new_multi_head_lora = PlainMultiheadAttentionLoRA(
                            submodule, enable_lora=args.params, r=args.r, lora_alpha=args.alpha, dropout_rate=args.dropout_rate)
                        setattr(block, name, new_multi_head_lora)
                        list_lora_layers.append(new_multi_head_lora)

    if args.encoder == 'vision' or args.encoder == 'both':
        indices = INDEX_POSITIONS_VISION[args.backbone][args.position]
        vision_encoder = clip_model.visual.transformer
        for i, block in enumerate(vision_encoder.resblocks):
            
            if i in indices:
                for name, submodule in block.named_children():
                    if isinstance(submodule, nn.MultiheadAttention):
                        new_multi_head_lora = PlainMultiheadAttentionLoRA(
                            submodule, enable_lora=args.params, r=args.r, lora_alpha=args.alpha, dropout_rate=args.dropout_rate)
                        setattr(block, name, new_multi_head_lora)
                        list_lora_layers.append(new_multi_head_lora)
    return list_lora_layers


def apply_attn(args, clip_model):
    list_linear_layers = []
    if args.encoder == 'text' or args.encoder == 'both':
        indices = INDEX_POSITIONS_TEXT[args.position]
        text_encoder = clip_model.transformer
        for i, block in enumerate(text_encoder.resblocks):
            if i in indices:
                for name, submodule in block.named_children():
                    if isinstance(submodule, nn.MultiheadAttention):
                        new_multi_head = PlainMultiheadAttention(existing_mha=submodule)
                        setattr(block, name, new_multi_head)
                        list_linear_layers.append(new_multi_head)

    if args.encoder == 'vision' or args.encoder == 'both':
        indices = INDEX_POSITIONS_VISION[args.backbone][args.position]
        vision_encoder = clip_model.visual.transformer
        for i, block in enumerate(vision_encoder.resblocks):
            if i in indices:
                for name, submodule in block.named_children():
                    if isinstance(submodule, nn.MultiheadAttention):
                        new_multi_head = PlainMultiheadAttention(existing_mha=submodule)
                        setattr(block, name, new_multi_head)
                        list_linear_layers.append(new_multi_head)
    return list_linear_layers


def apply_Adalora(args, clip_model):
    list_lora_layers = []
    if args.encoder == 'text' or args.encoder == 'both':
        indices = INDEX_POSITIONS_TEXT[args.position]
        text_encoder = clip_model.transformer
        for i, block in enumerate(text_encoder.resblocks):
            
            if i in indices:
                for name, submodule in block.named_children():
                    if isinstance(submodule, nn.MultiheadAttention):
                        new_multi_head_lora = MultiheadAttentionAdaLoRA(
                            submodule, enable_lora=args.params, r=args.r, lora_alpha=args.alpha, dropout_rate=args.dropout_rate)
                        setattr(block, name, new_multi_head_lora)
                        list_lora_layers.append(new_multi_head_lora)

    if args.encoder == 'vision' or args.encoder == 'both':
        indices = INDEX_POSITIONS_VISION[args.backbone][args.position]
        vision_encoder = clip_model.visual.transformer
        for i, block in enumerate(vision_encoder.resblocks):
            
            if i in indices:
                for name, submodule in block.named_children():
                    if isinstance(submodule, nn.MultiheadAttention):
                        new_multi_head_lora = MultiheadAttentionAdaLoRA(
                            submodule, enable_lora=args.params, r=args.r, lora_alpha=args.alpha, dropout_rate=args.dropout_rate)
                        setattr(block, name, new_multi_head_lora)
                        list_lora_layers.append(new_multi_head_lora)
    return list_lora_layers

def apply_Dylora(args, clip_model):
    list_lora_layers = []
    if args.encoder == 'text' or args.encoder == 'both':
        indices = INDEX_POSITIONS_TEXT[args.position]
        text_encoder = clip_model.transformer
        for i, block in enumerate(text_encoder.resblocks):
            
            if i in indices:
                for name, submodule in block.named_children():
                    if isinstance(submodule, nn.MultiheadAttention):
                        new_multi_head_lora = MultiheadAttentionDyLoRA(
                            submodule, enable_lora=args.params, r=args.r, lora_alpha=args.alpha, dropout_rate=args.dropout_rate)
                        setattr(block, name, new_multi_head_lora)
                        list_lora_layers.append(new_multi_head_lora)

    if args.encoder == 'vision' or args.encoder == 'both':
        indices = INDEX_POSITIONS_VISION[args.backbone][args.position]
        vision_encoder = clip_model.visual.transformer
        for i, block in enumerate(vision_encoder.resblocks):
            
            if i in indices:
                for name, submodule in block.named_children():
                    if isinstance(submodule, nn.MultiheadAttention):
                        new_multi_head_lora = MultiheadAttentionDyLoRA(
                            submodule, enable_lora=args.params, r=args.r, lora_alpha=args.alpha, dropout_rate=args.dropout_rate)
                        setattr(block, name, new_multi_head_lora)
                        list_lora_layers.append(new_multi_head_lora)
    return list_lora_layers


def apply_Bilora(args, clip_model):
    list_lora_layers = []
    if args.encoder == 'text' or args.encoder == 'both':
        indices = INDEX_POSITIONS_TEXT[args.position]
        text_encoder = clip_model.transformer
        for i, block in enumerate(text_encoder.resblocks):
            
            if i in indices:
                for name, submodule in block.named_children():
                    if isinstance(submodule, nn.MultiheadAttention):
                        new_multi_head_lora = PlainMultiheadAttentionLoRA(
                            submodule, enable_lora=args.params, r=args.r, lora_alpha=args.alpha, dropout_rate=args.dropout_rate)
                        setattr(block, name, new_multi_head_lora)
                        list_lora_layers.append(new_multi_head_lora)

    if args.encoder == 'vision' or args.encoder == 'both':
        indices = INDEX_POSITIONS_VISION[args.backbone][args.position]
        vision_encoder = clip_model.visual.transformer
        for i, block in enumerate(vision_encoder.resblocks):
            
            if i in indices:
                for name, submodule in block.named_children():
                    if isinstance(submodule, nn.MultiheadAttention):
                        new_multi_head_lora = PlainMultiheadAttentionLoRA(
                            submodule, enable_lora=args.params, r=args.r, lora_alpha=args.alpha, dropout_rate=args.dropout_rate)
                        setattr(block, name, new_multi_head_lora)
                        list_lora_layers.append(new_multi_head_lora)
    return list_lora_layers

def apply_Lolora(args, clip_model):
    list_lora_layers = []
    if args.encoder == 'text' or args.encoder == 'both':
        indices = INDEX_POSITIONS_TEXT[args.position]
        text_encoder = clip_model.transformer
        for i, block in enumerate(text_encoder.resblocks):
            
            if i in indices:
                for name, submodule in block.named_children():
                    if isinstance(submodule, nn.MultiheadAttention):
                        new_multi_head_lora = MultiheadAttentionLoLoRA(
                            submodule, enable_lora=args.params, r=args.r, rank_ratio=args.rank_ratio,
                            max_rank=args.max_rank, min_rank=args.min_rank,
                            lora_alpha=args.alpha, dropout_rate=args.dropout_rate)
                        setattr(block, name, new_multi_head_lora)
                        list_lora_layers.append(new_multi_head_lora)

    if args.encoder == 'vision' or args.encoder == 'both':
        indices = INDEX_POSITIONS_VISION[args.backbone][args.position]
        vision_encoder = clip_model.visual.transformer
        for i, block in enumerate(vision_encoder.resblocks):
            
            if i in indices:
                for name, submodule in block.named_children():
                    if isinstance(submodule, nn.MultiheadAttention):
                        new_multi_head_lora = MultiheadAttentionLoLoRA(
                            submodule, enable_lora=args.params, r=args.r, rank_ratio=args.rank_ratio,
                            max_rank=args.max_rank, min_rank=args.min_rank,
                            lora_alpha=args.alpha, dropout_rate=args.dropout_rate)
                        setattr(block, name, new_multi_head_lora)
                        list_lora_layers.append(new_multi_head_lora)
    return list_lora_layers

def apply_Lolora_v2(args, clip_model):
    list_lora_layers = []
    if args.encoder == 'text' or args.encoder == 'both':
        indices = INDEX_POSITIONS_TEXT[args.position]
        text_encoder = clip_model.transformer
        for i, block in enumerate(text_encoder.resblocks):
            
            if i in indices:
                for name, submodule in block.named_children():
                    if isinstance(submodule, nn.MultiheadAttention):
                        new_multi_head_lora = MultiheadAttentionLoLoRA_v2(
                            submodule, enable_lora=args.params, rank_ratio=args.rank_ratio,
                            inner_max_rank=args.inner_max_rank, inner_min_rank=args.inner_min_rank,
                            lora_alpha=args.alpha, dropout_rate=args.dropout_rate)
                        setattr(block, name, new_multi_head_lora)
                        list_lora_layers.append(new_multi_head_lora)

    if args.encoder == 'vision' or args.encoder == 'both':
        indices = INDEX_POSITIONS_VISION[args.backbone][args.position]
        vision_encoder = clip_model.visual.transformer
        for i, block in enumerate(vision_encoder.resblocks):
            
            if i in indices:
                for name, submodule in block.named_children():
                    if isinstance(submodule, nn.MultiheadAttention):
                        new_multi_head_lora = MultiheadAttentionLoLoRA_v2(
                            submodule, enable_lora=args.params, rank_ratio=args.rank_ratio,
                            inner_max_rank=args.inner_max_rank, inner_min_rank=args.inner_min_rank,
                            lora_alpha=args.alpha, dropout_rate=args.dropout_rate)
                        setattr(block, name, new_multi_head_lora)
                        list_lora_layers.append(new_multi_head_lora)
    return list_lora_layers

def plot_individual_bars(data):
    import matplotlib.pyplot as plt
    import numpy as np

    # 准备数据
    x = np.arange(12)  # 0到11的数字
    text_values = list(data['text'].values())
    vision_values = list(data['vision'].values())

    # 绘制text数据的柱状图
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x, text_values, color='skyblue')
    ax.set_xticks(x)
    ax.set_xlabel('Layer Index')
    ax.set_ylabel('Text Values')
    ax.set_title('Text Data across Layers')
    ax.grid(True)
    plt.tight_layout()
    plt.show()

    # 绘制vision数据的柱状图
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x, vision_values, color='lightgreen')
    ax.set_xticks(x)
    ax.set_xlabel('Layer Index')
    ax.set_ylabel('Vision Values')
    ax.set_title('Vision Data across Layers')
    ax.grid(True)
    plt.tight_layout()
    plt.show()

def apply_Lolora_v3(args, clip_model, text_ratios, vision_ratios):
    list_lora_layers = []
    
    if args.encoder == 'text' or args.encoder == 'both':
        indices = INDEX_POSITIONS_TEXT[args.position]
        text_encoder = clip_model.transformer
        for i, block in enumerate(text_encoder.resblocks):
            
            if i in indices:
                for name, submodule in block.named_children():
                    if isinstance(submodule, nn.MultiheadAttention):
                        new_multi_head_lora = MultiheadAttentionLoLoRA_v3(
                            submodule, enable_lora=args.params, rank_ratio=text_ratios[i],
                            inner_max_rank=args.inner_max_rank, inner_min_rank=args.inner_min_rank,
                            lora_alpha=args.alpha, dropout_rate=args.dropout_rate)
                        setattr(block, name, new_multi_head_lora)
                        list_lora_layers.append(new_multi_head_lora)

    if args.encoder == 'vision' or args.encoder == 'both':
        indices = INDEX_POSITIONS_VISION[args.backbone][args.position]
        vision_encoder = clip_model.visual.transformer
        for i, block in enumerate(vision_encoder.resblocks):
            
            if i in indices:
                for name, submodule in block.named_children():
                    if isinstance(submodule, nn.MultiheadAttention):
                        new_multi_head_lora = MultiheadAttentionLoLoRA_v3(
                            submodule, enable_lora=args.params, rank_ratio=vision_ratios[i],
                            inner_max_rank=args.inner_max_rank, inner_min_rank=args.inner_min_rank,
                            lora_alpha=args.alpha, dropout_rate=args.dropout_rate)
                        setattr(block, name, new_multi_head_lora)
                        list_lora_layers.append(new_multi_head_lora)
    return list_lora_layers

def apply_Dora(args, clip_model):
    list_lora_layers = []
    if args.encoder == 'text' or args.encoder == 'both':
        indices = INDEX_POSITIONS_TEXT[args.position]
        text_encoder = clip_model.transformer
        for i, block in enumerate(text_encoder.resblocks):
            
            if i in indices:
                for name, submodule in block.named_children():
                    if isinstance(submodule, nn.MultiheadAttention):
                        new_multi_head_lora = MultiheadAttentionDoRA(
                            submodule, enable_lora=args.params, r=args.r, lora_alpha=args.alpha, 
                            dropout_rate=args.dropout_rate, Wdecompose=args.W_decompose, dora_simple=args.dora_simple)
                        setattr(block, name, new_multi_head_lora)
                        list_lora_layers.append(new_multi_head_lora)

    if args.encoder == 'vision' or args.encoder == 'both':
        indices = INDEX_POSITIONS_VISION[args.backbone][args.position]
        vision_encoder = clip_model.visual.transformer
        for i, block in enumerate(vision_encoder.resblocks):
            
            if i in indices:
                for name, submodule in block.named_children():
                    if isinstance(submodule, nn.MultiheadAttention):
                        new_multi_head_lora = MultiheadAttentionDoRA(
                            submodule, enable_lora=args.params, r=args.r, lora_alpha=args.alpha, 
                            dropout_rate=args.dropout_rate, Wdecompose=args.W_decompose, dora_simple=args.dora_simple)
                        setattr(block, name, new_multi_head_lora)
                        list_lora_layers.append(new_multi_head_lora)
    return list_lora_layers

def apply_Dora_v2(args, clip_model):
    list_lora_layers = []
    if args.encoder == 'text' or args.encoder == 'both':
        indices = INDEX_POSITIONS_TEXT[args.position]
        text_encoder = clip_model.transformer
        for i, block in enumerate(text_encoder.resblocks):
            
            if i in indices:
                for name, submodule in block.named_children():
                    if isinstance(submodule, nn.MultiheadAttention):
                        new_multi_head_lora = MultiheadAttentionDoRA_v2(
                            submodule, enable_lora=args.params, r=args.r, lora_alpha=args.alpha, 
                            dropout_rate=args.dropout_rate, Wdecompose=args.W_decompose, dora_simple=args.dora_simple)
                        setattr(block, name, new_multi_head_lora)
                        list_lora_layers.append(new_multi_head_lora)

    if args.encoder == 'vision' or args.encoder == 'both':
        indices = INDEX_POSITIONS_VISION[args.backbone][args.position]
        vision_encoder = clip_model.visual.transformer
        for i, block in enumerate(vision_encoder.resblocks):
            
            if i in indices:
                for name, submodule in block.named_children():
                    if isinstance(submodule, nn.MultiheadAttention):
                        new_multi_head_lora = MultiheadAttentionDoRA_v2(
                            submodule, enable_lora=args.params, r=args.r, lora_alpha=args.alpha, 
                            dropout_rate=args.dropout_rate, Wdecompose=args.W_decompose, dora_simple=args.dora_simple)
                        setattr(block, name, new_multi_head_lora)
                        list_lora_layers.append(new_multi_head_lora)
    return list_lora_layers

def apply_Dora_v3(args, clip_model):
    list_lora_layers = []
    if args.encoder == 'text' or args.encoder == 'both':
        indices = INDEX_POSITIONS_TEXT[args.position]
        text_encoder = clip_model.transformer
        for i, block in enumerate(text_encoder.resblocks):
            
            if i in indices:
                for name, submodule in block.named_children():
                    if isinstance(submodule, nn.MultiheadAttention):
                        new_multi_head_lora = MultiheadAttentionDoRA_v3(
                            submodule, enable_lora=args.params, r=args.r, lora_alpha=args.alpha, 
                            dropout_rate=args.dropout_rate, Wdecompose=args.W_decompose, dora_simple=args.dora_simple)
                        setattr(block, name, new_multi_head_lora)
                        list_lora_layers.append(new_multi_head_lora)

    if args.encoder == 'vision' or args.encoder == 'both':
        indices = INDEX_POSITIONS_VISION[args.backbone][args.position]
        vision_encoder = clip_model.visual.transformer
        for i, block in enumerate(vision_encoder.resblocks):
            
            if i in indices:
                for name, submodule in block.named_children():
                    if isinstance(submodule, nn.MultiheadAttention):
                        new_multi_head_lora = MultiheadAttentionDoRA_v3(
                            submodule, enable_lora=args.params, r=args.r, lora_alpha=args.alpha, 
                            dropout_rate=args.dropout_rate, Wdecompose=args.W_decompose, dora_simple=args.dora_simple)
                        setattr(block, name, new_multi_head_lora)
                        list_lora_layers.append(new_multi_head_lora)
    return list_lora_layers

def apply_Dora_v4(args, clip_model):
    list_lora_layers = []
    if args.encoder == 'text' or args.encoder == 'both':
        indices = INDEX_POSITIONS_TEXT[args.position]
        text_encoder = clip_model.transformer
        for i, block in enumerate(text_encoder.resblocks):
            
            if i in indices:
                for name, submodule in block.named_children():
                    if isinstance(submodule, nn.MultiheadAttention):
                        new_multi_head_lora = MultiheadAttentionDoRA_v4(
                            submodule, enable_lora=args.params, r=args.r, lora_alpha=args.alpha, 
                            dropout_rate=args.dropout_rate, Wdecompose=args.W_decompose, dora_simple=args.dora_simple, r_init_method=args.init_method)
                        setattr(block, name, new_multi_head_lora)
                        list_lora_layers.append(new_multi_head_lora)

    if args.encoder == 'vision' or args.encoder == 'both':
        indices = INDEX_POSITIONS_VISION[args.backbone][args.position]
        vision_encoder = clip_model.visual.transformer
        for i, block in enumerate(vision_encoder.resblocks):
            
            if i in indices:
                for name, submodule in block.named_children():
                    if isinstance(submodule, nn.MultiheadAttention):
                        new_multi_head_lora = MultiheadAttentionDoRA_v4(
                            submodule, enable_lora=args.params, r=args.r, lora_alpha=args.alpha, 
                            dropout_rate=args.dropout_rate, Wdecompose=args.W_decompose, dora_simple=args.dora_simple, r_init_method=args.init_method)
                        setattr(block, name, new_multi_head_lora)
                        list_lora_layers.append(new_multi_head_lora)
    return list_lora_layers

def apply_Dora_v5(args, clip_model):
    list_lora_layers = []
    if args.encoder == 'text' or args.encoder == 'both':
        indices = INDEX_POSITIONS_TEXT[args.position]
        text_encoder = clip_model.transformer
        for i, block in enumerate(text_encoder.resblocks):
            
            if i in indices:
                for name, submodule in block.named_children():
                    if isinstance(submodule, nn.MultiheadAttention):
                        new_multi_head_lora = MultiheadAttentionDoRA_v5(
                            submodule, enable_lora=args.params, r=args.r, lora_alpha=args.alpha, 
                            dropout_rate=args.dropout_rate, Wdecompose=args.W_decompose, dora_simple=args.dora_simple, r_init_method=args.init_method)
                        setattr(block, name, new_multi_head_lora)
                        list_lora_layers.append(new_multi_head_lora)

    if args.encoder == 'vision' or args.encoder == 'both':
        indices = INDEX_POSITIONS_VISION[args.backbone][args.position]
        vision_encoder = clip_model.visual.transformer
        for i, block in enumerate(vision_encoder.resblocks):
            
            if i in indices:
                for name, submodule in block.named_children():
                    if isinstance(submodule, nn.MultiheadAttention):
                        new_multi_head_lora = MultiheadAttentionDoRA_v5(
                            submodule, enable_lora=args.params, r=args.r, lora_alpha=args.alpha, 
                            dropout_rate=args.dropout_rate, Wdecompose=args.W_decompose, dora_simple=args.dora_simple, r_init_method=args.init_method)
                        setattr(block, name, new_multi_head_lora)
                        list_lora_layers.append(new_multi_head_lora)
    return list_lora_layers

def apply_Dora_v7(args, clip_model, text_W_prime_dict, text_S_dict, vision_W_prime_dict, vision_S_dict):
    list_lora_layers = []
    if args.encoder == 'text' or args.encoder == 'both':
        indices = INDEX_POSITIONS_TEXT[args.position]
        text_encoder = clip_model.transformer
        for i, block in enumerate(text_encoder.resblocks):
            
            if i in indices:
                for name, submodule in block.named_children():
                    if isinstance(submodule, nn.MultiheadAttention):
                        new_multi_head_lora = MultiheadAttentionDoRA_v7(
                            submodule, enable_lora=args.params, r=args.r, lora_alpha=args.alpha, 
                            dropout_rate=args.dropout_rate, Wdecompose=args.W_decompose, dora_simple=args.dora_simple)
                        setattr(block, name, new_multi_head_lora)
                        list_lora_layers.append(new_multi_head_lora)
        
        for i, block in enumerate(text_encoder.resblocks):
            
            if i in indices:
                for name, submodule in block.named_modules():
                    if isinstance(submodule, DoraLinear_v7):
                        submodule.init_lora_AB_parameters(text_W_prime_dict[f'text.resblocks.{i}.{name}'], text_S_dict[f'text.resblocks.{i}.{name}'])


    if args.encoder == 'vision' or args.encoder == 'both':
        indices = INDEX_POSITIONS_VISION[args.backbone][args.position]
        vision_encoder = clip_model.visual.transformer
        for i, block in enumerate(vision_encoder.resblocks):
            
            if i in indices:
                for name, submodule in block.named_children():
                    if isinstance(submodule, nn.MultiheadAttention):
                        new_multi_head_lora = MultiheadAttentionDoRA_v7(
                            submodule, enable_lora=args.params, r=args.r, lora_alpha=args.alpha, 
                            dropout_rate=args.dropout_rate, Wdecompose=args.W_decompose, dora_simple=args.dora_simple)
                        setattr(block, name, new_multi_head_lora)
                        list_lora_layers.append(new_multi_head_lora)
        
        for i, block in enumerate(vision_encoder.resblocks):
            
            if i in indices:
                for name, submodule in block.named_modules():
                    if isinstance(submodule, DoraLinear_v7):
                        submodule.init_lora_AB_parameters(vision_W_prime_dict[f'vision.resblocks.{i}.{name}'], vision_S_dict[f'vision.resblocks.{i}.{name}'])

    return list_lora_layers

def apply_Dora_v7_2(args, clip_model, text_W_prime_dict, text_S_dict, vision_W_prime_dict, vision_S_dict):
    list_lora_layers = []
    if args.encoder == 'text' or args.encoder == 'both':
        indices = INDEX_POSITIONS_TEXT[args.position]
        text_encoder = clip_model.transformer        
        for i, block in enumerate(text_encoder.resblocks):
            
            if i in indices:
                for name, submodule in block.named_children():
                    if isinstance(submodule, nn.MultiheadAttention):
                        new_multi_head_lora = MultiheadAttentionDoRA_v7_2(
                            submodule, enable_lora=args.params, r=args.r, lora_alpha=args.alpha, 
                            dropout_rate=args.dropout_rate, Wdecompose=args.W_decompose, dora_simple=args.dora_simple)
                        setattr(block, name, new_multi_head_lora)
                        list_lora_layers.append(new_multi_head_lora)
        for i, block in enumerate(text_encoder.resblocks):
            
            if i in indices:
                for name, submodule in block.named_modules():
                    if isinstance(submodule, DoraLinear_v7_2):
                        submodule.init_lora_AB_parameters(text_W_prime_dict[f'text.resblocks.{i}.{name}'], text_S_dict[f'text.resblocks.{i}.{name}'])


    if args.encoder == 'vision' or args.encoder == 'both':
        indices = INDEX_POSITIONS_VISION[args.backbone][args.position]
        vision_encoder = clip_model.visual.transformer
        for i, block in enumerate(vision_encoder.resblocks):
            
            if i in indices:
                for name, submodule in block.named_children():
                    if isinstance(submodule, nn.MultiheadAttention):
                        new_multi_head_lora = MultiheadAttentionDoRA_v7_2(
                            submodule, enable_lora=args.params, r=args.r, lora_alpha=args.alpha, 
                            dropout_rate=args.dropout_rate, Wdecompose=args.W_decompose, dora_simple=args.dora_simple)
                        setattr(block, name, new_multi_head_lora)
                        list_lora_layers.append(new_multi_head_lora)
        for i, block in enumerate(vision_encoder.resblocks):
            
            if i in indices:
                for name, submodule in block.named_modules():
                    if isinstance(submodule, DoraLinear_v7_2):
                        submodule.init_lora_AB_parameters(vision_W_prime_dict[f'vision.resblocks.{i}.{name}'], vision_S_dict[f'vision.resblocks.{i}.{name}'])

    return list_lora_layers

def apply_Dora_v7_3(args, clip_model, text_W_prime_dict, text_S_dict, vision_W_prime_dict, vision_S_dict):
    list_lora_layers = []
    if args.encoder == 'text' or args.encoder == 'both':
        indices = INDEX_POSITIONS_TEXT[args.position]
        text_encoder = clip_model.transformer        
        for i, block in enumerate(text_encoder.resblocks):
            
            if i in indices:
                for name, submodule in block.named_children():
                    if isinstance(submodule, nn.MultiheadAttention):
                        new_multi_head_lora = MultiheadAttentionDoRA_v7_3(
                            submodule, enable_lora=args.params, r=args.r, lora_alpha=args.alpha, 
                            dropout_rate=args.dropout_rate, Wdecompose=args.W_decompose, dora_simple=args.dora_simple)
                        setattr(block, name, new_multi_head_lora)
                        list_lora_layers.append(new_multi_head_lora)
        for i, block in enumerate(text_encoder.resblocks):
            
            if i in indices:
                for name, submodule in block.named_modules():
                    if isinstance(submodule, DoraLinear_v7_3):
                        submodule.init_lora_AB_parameters(text_W_prime_dict[f'text.resblocks.{i}.{name}'], text_S_dict[f'text.resblocks.{i}.{name}'])


    if args.encoder == 'vision' or args.encoder == 'both':
        indices = INDEX_POSITIONS_VISION[args.backbone][args.position]
        vision_encoder = clip_model.visual.transformer
        for i, block in enumerate(vision_encoder.resblocks):
            
            if i in indices:
                for name, submodule in block.named_children():
                    if isinstance(submodule, nn.MultiheadAttention):
                        new_multi_head_lora = MultiheadAttentionDoRA_v7_3(
                            submodule, enable_lora=args.params, r=args.r, lora_alpha=args.alpha, 
                            dropout_rate=args.dropout_rate, Wdecompose=args.W_decompose, dora_simple=args.dora_simple)
                        setattr(block, name, new_multi_head_lora)
                        list_lora_layers.append(new_multi_head_lora)
        for i, block in enumerate(vision_encoder.resblocks):
            
            if i in indices:
                for name, submodule in block.named_modules():
                    if isinstance(submodule, DoraLinear_v7_3):
                        submodule.init_lora_AB_parameters(vision_W_prime_dict[f'vision.resblocks.{i}.{name}'], vision_S_dict[f'vision.resblocks.{i}.{name}'])

    return list_lora_layers

def apply_Dora_v8(args, clip_model, text_W_prime_dict, text_S_dict, vision_W_prime_dict, vision_S_dict):
    list_lora_layers = []
    if args.encoder == 'text' or args.encoder == 'both':
        indices = INDEX_POSITIONS_TEXT[args.position]
        text_encoder = clip_model.transformer
        for i, block in enumerate(text_encoder.resblocks):
            if i in indices:
                for name, submodule in block.named_children():
                    if isinstance(submodule, nn.MultiheadAttention):
                        new_multi_head_lora = MultiheadAttentionDoRA_v8(
                            submodule, enable_lora=args.params, r=args.r, lora_alpha=args.alpha, 
                            dropout_rate=args.dropout_rate, Wdecompose=args.W_decompose, dora_simple=args.dora_simple, side_r=args.side_r)
                        setattr(block, name, new_multi_head_lora)
                        list_lora_layers.append(new_multi_head_lora)
        
        for i, block in enumerate(text_encoder.resblocks):
            
            if i in indices:
                for name, submodule in block.named_modules():
                    if isinstance(submodule, DoraLinear_v8):
                        submodule.init_lora_AB_parameters(text_W_prime_dict[f'text.resblocks.{i}.{name}'], text_S_dict[f'text.resblocks.{i}.{name}'])


    if args.encoder == 'vision' or args.encoder == 'both':
        indices = INDEX_POSITIONS_VISION[args.backbone][args.position]
        vision_encoder = clip_model.visual.transformer
        for i, block in enumerate(vision_encoder.resblocks):
            if i in indices:
                for name, submodule in block.named_children():
                    if isinstance(submodule, nn.MultiheadAttention):
                        new_multi_head_lora = MultiheadAttentionDoRA_v8(
                            submodule, enable_lora=args.params, r=args.r, lora_alpha=args.alpha, 
                            dropout_rate=args.dropout_rate, Wdecompose=args.W_decompose, dora_simple=args.dora_simple, side_r=args.side_r)
                        setattr(block, name, new_multi_head_lora)
                        list_lora_layers.append(new_multi_head_lora)
        
        for i, block in enumerate(vision_encoder.resblocks):
            
            if i in indices:
                for name, submodule in block.named_modules():
                    if isinstance(submodule, DoraLinear_v8):
                        submodule.init_lora_AB_parameters(vision_W_prime_dict[f'vision.resblocks.{i}.{name}'], vision_S_dict[f'vision.resblocks.{i}.{name}'])

    return list_lora_layers

def apply_Dora_v9(args, clip_model, text_W_prime_dict, text_S_dict, vision_W_prime_dict, vision_S_dict):
    list_lora_layers = []
    if args.encoder == 'text' or args.encoder == 'both':
        indices = INDEX_POSITIONS_TEXT[args.position]
        text_encoder = clip_model.transformer
        for i, block in enumerate(text_encoder.resblocks):
            
            if i in indices:
                for name, submodule in block.named_children():
                    if isinstance(submodule, nn.MultiheadAttention):
                        new_multi_head_lora = MultiheadAttentionDoRA_v9(
                            submodule, enable_lora=args.params, r=args.r, lora_alpha=args.alpha, 
                            dropout_rate=args.dropout_rate, Wdecompose=args.W_decompose, dora_simple=args.dora_simple)
                        setattr(block, name, new_multi_head_lora)
                        list_lora_layers.append(new_multi_head_lora)
        
        for i, block in enumerate(text_encoder.resblocks):
            
            if i in indices:
                for name, submodule in block.named_modules():
                    if isinstance(submodule, DoraLinear_v9):
                        submodule.init_lora_AB_parameters(text_W_prime_dict[f'text.resblocks.{i}.{name}'], text_S_dict[f'text.resblocks.{i}.{name}'])


    if args.encoder == 'vision' or args.encoder == 'both':
        indices = INDEX_POSITIONS_VISION[args.backbone][args.position]
        vision_encoder = clip_model.visual.transformer
        for i, block in enumerate(vision_encoder.resblocks):
            
            if i in indices:
                for name, submodule in block.named_children():
                    if isinstance(submodule, nn.MultiheadAttention):
                        new_multi_head_lora = MultiheadAttentionDoRA_v9(
                            submodule, enable_lora=args.params, r=args.r, lora_alpha=args.alpha, 
                            dropout_rate=args.dropout_rate, Wdecompose=args.W_decompose, dora_simple=args.dora_simple)
                        setattr(block, name, new_multi_head_lora)
                        list_lora_layers.append(new_multi_head_lora)
        
        for i, block in enumerate(vision_encoder.resblocks):
            
            if i in indices:
                for name, submodule in block.named_modules():
                    if isinstance(submodule, DoraLinear_v9):
                        submodule.init_lora_AB_parameters(vision_W_prime_dict[f'vision.resblocks.{i}.{name}'], vision_S_dict[f'vision.resblocks.{i}.{name}'])

    return list_lora_layers

def apply_NoRA_A(args, clip_model, text_W_prime_dict, text_S_dict, vision_W_prime_dict, vision_S_dict):
    list_lora_layers = []
    if args.encoder == 'text' or args.encoder == 'both':
        indices = INDEX_POSITIONS_TEXT[args.position]
        text_encoder = clip_model.transformer
        for i, block in enumerate(text_encoder.resblocks):
            if i in indices:
                for name, submodule in block.named_children():
                    if isinstance(submodule, nn.MultiheadAttention):
                        new_multi_head_lora = MultiheadAttentionNoRA_A(
                            submodule, enable_lora=args.params, r=args.r, lora_alpha=args.alpha, 
                            dropout_rate=args.dropout_rate)
                        setattr(block, name, new_multi_head_lora)
                        list_lora_layers.append(new_multi_head_lora)
        
        for i, block in enumerate(text_encoder.resblocks):
            
            if i in indices:
                for name, submodule in block.named_modules():
                    if isinstance(submodule, NoRALinear_A):
                        submodule.init_lora_param(text_W_prime_dict[f'text.resblocks.{i}.{name}'], text_S_dict[f'text.resblocks.{i}.{name}'], init_method=args.init_method)


    if args.encoder == 'vision' or args.encoder == 'both':
        indices = INDEX_POSITIONS_VISION[args.backbone][args.position]
        vision_encoder = clip_model.visual.transformer
        for i, block in enumerate(vision_encoder.resblocks):
            if i in indices:
                for name, submodule in block.named_children():
                    if isinstance(submodule, nn.MultiheadAttention):
                        new_multi_head_lora = MultiheadAttentionNoRA_A(
                            submodule, enable_lora=args.params, r=args.r, lora_alpha=args.alpha, 
                            dropout_rate=args.dropout_rate)
                        setattr(block, name, new_multi_head_lora)
                        list_lora_layers.append(new_multi_head_lora)
        
        for i, block in enumerate(vision_encoder.resblocks):
            
            if i in indices:
                for name, submodule in block.named_modules():
                    if isinstance(submodule, NoRALinear_A):
                        submodule.init_lora_param(vision_W_prime_dict[f'vision.resblocks.{i}.{name}'], vision_S_dict[f'vision.resblocks.{i}.{name}'], init_method=args.init_method)
    return list_lora_layers

def apply_NoRA_B(args, clip_model, text_W_prime_dict, text_S_dict, vision_W_prime_dict, vision_S_dict):
    list_lora_layers = []
    if args.encoder == 'text' or args.encoder == 'both':
        indices = INDEX_POSITIONS_TEXT[args.position]
        text_encoder = clip_model.transformer
        for i, block in enumerate(text_encoder.resblocks):
            if i in indices:
                for name, submodule in block.named_children():
                    if isinstance(submodule, nn.MultiheadAttention):
                        new_multi_head_lora = MultiheadAttentionNoRA_B(
                            submodule, enable_lora=args.params, r=args.r, lora_alpha=args.alpha, 
                            dropout_rate=args.dropout_rate, structure=args.structure)
                        setattr(block, name, new_multi_head_lora)
                        list_lora_layers.append(new_multi_head_lora)
        
        for i, block in enumerate(text_encoder.resblocks):
            print(f"Residual Attention Block {i}: {block}")
            if i in indices:
                for name, submodule in block.named_modules():
                    if isinstance(submodule, NoRALinear_B):
                        submodule.init_W_AB_param(text_W_prime_dict[f'text.resblocks.{i}.{name}'], text_S_dict[f'text.resblocks.{i}.{name}'], init_method=args.init_method)


    if args.encoder == 'vision' or args.encoder == 'both':
        indices = INDEX_POSITIONS_VISION[args.backbone][args.position]
        vision_encoder = clip_model.visual.transformer
        for i, block in enumerate(vision_encoder.resblocks):
            
            if i in indices:
                for name, submodule in block.named_children():
                    if isinstance(submodule, nn.MultiheadAttention):
                        new_multi_head_lora = MultiheadAttentionNoRA_B(
                            submodule, enable_lora=args.params, r=args.r, lora_alpha=args.alpha, 
                            dropout_rate=args.dropout_rate, structure=args.structure)
                        setattr(block, name, new_multi_head_lora)
                        list_lora_layers.append(new_multi_head_lora)
        
        for i, block in enumerate(vision_encoder.resblocks):
            print(f"Residual Attention Block {i}: {block}")
            if i in indices:
                for name, submodule in block.named_modules():
                    if isinstance(submodule, NoRALinear_B):
                        submodule.init_W_AB_param(vision_W_prime_dict[f'vision.resblocks.{i}.{name}'], vision_S_dict[f'vision.resblocks.{i}.{name}'], init_method=args.init_method)
    return list_lora_layers

def apply_NoRA_C(args, clip_model, text_W_prime_dict, text_S_dict, vision_W_prime_dict, vision_S_dict):
    list_lora_layers = []
    if args.encoder == 'text' or args.encoder == 'both':
        indices = INDEX_POSITIONS_TEXT[args.position]
        text_encoder = clip_model.transformer
        for i, block in enumerate(text_encoder.resblocks):
            if i in indices:
                for name, submodule in block.named_children():
                    if isinstance(submodule, nn.MultiheadAttention):
                        new_multi_head_lora = MultiheadAttentionNoRA_C(
                            submodule, enable_lora=args.params, r=args.r, lora_alpha=args.alpha, 
                            dropout_rate=args.dropout_rate, Wab_structure=args.Wab_structure)
                        setattr(block, name, new_multi_head_lora)
                        list_lora_layers.append(new_multi_head_lora)
        
        for i, block in enumerate(text_encoder.resblocks):
            
            if i in indices:
                for name, submodule in block.named_modules():
                    if isinstance(submodule, NoRALinear_C):
                        submodule.init_W_AB_param(text_W_prime_dict[f'text.resblocks.{i}.{name}'], text_S_dict[f'text.resblocks.{i}.{name}'], init_method=args.init_method)


    if args.encoder == 'vision' or args.encoder == 'both':
        indices = INDEX_POSITIONS_VISION[args.backbone][args.position]
        vision_encoder = clip_model.visual.transformer
        for i, block in enumerate(vision_encoder.resblocks):
            if i in indices:
                for name, submodule in block.named_children():
                    if isinstance(submodule, nn.MultiheadAttention):
                        new_multi_head_lora = MultiheadAttentionNoRA_C(
                            submodule, enable_lora=args.params, r=args.r, lora_alpha=args.alpha, 
                            dropout_rate=args.dropout_rate, Wab_structure=args.Wab_structure)
                        setattr(block, name, new_multi_head_lora)
                        list_lora_layers.append(new_multi_head_lora)
        
        for i, block in enumerate(vision_encoder.resblocks):
            print(f"Residual Attention Block {i}: {block}")
            if i in indices:
                for name, submodule in block.named_modules():
                    if isinstance(submodule, NoRALinear_C):
                        submodule.init_W_AB_param(vision_W_prime_dict[f'vision.resblocks.{i}.{name}'], vision_S_dict[f'vision.resblocks.{i}.{name}'], init_method=args.init_method)
    return list_lora_layers


def save_lora(args, list_lora_layers):
    weights = {}
    for i, layer in enumerate(list_lora_layers):
        layer_weights = {}
        if 'q' in args.params:
            layer_weights['q_proj'] = {
                'w_lora_A': layer.q_proj.w_lora_A.data,
                'w_lora_B': layer.q_proj.w_lora_B.data
            }
        if 'k' in args.params:
            layer_weights['k_proj'] = {
                'w_lora_A': layer.k_proj.w_lora_A.data,
                'w_lora_B': layer.k_proj.w_lora_B.data
            }
        if 'v' in args.params:
            layer_weights['v_proj'] = {
                'w_lora_A': layer.v_proj.w_lora_A.data,
                'w_lora_B': layer.v_proj.w_lora_B.data
            }
        if 'o' in args.params:
            layer_weights['proj'] = {
                'w_lora_A': layer.proj.w_lora_A.data,
                'w_lora_B': layer.proj.w_lora_B.data
            }

        weights[f'layer_{i}'] = layer_weights

    metadata = {
        'r': args.r,
        'alpha': args.alpha,
        'encoder': args.encoder,
        'params': args.params,
        'position': args.position
    }

    save_data = {
        'weights': weights,
        'metadata': metadata
    }

    # to manage names like ViT-B/16
    backbone = args.backbone.replace('/', '').replace('-', '').lower()
    save_dir = f'{args.save_path}/{backbone}/{args.dataset}/{args.shots}shots/seed{args.seed}'
    os.makedirs(save_dir, exist_ok=True)

    save_path = f'{save_dir}/{args.filename}.pt'
    torch.save(save_data, save_path)
    print(f'LoRA weights saved to {save_path}')

def load_lora(args, list_lora_layers):
    # to manage names like ViT-B/16
    backbone = args.backbone.replace('/', '').replace('-', '').lower()
    load_path = f'{args.save_path}/{backbone}/{args.dataset}/{args.shots}shots/seed{args.seed}/{args.filename}.pt'

    if not os.path.exists(load_path):
        raise FileNotFoundError(f'File {load_path} does not exist.')

    loaded_data = torch.load(load_path)

    metadata = loaded_data['metadata']
    if metadata['r'] != args.r:
        raise ValueError(
            f"r mismatch: expected {args.r}, found {metadata['r']}")
    if metadata['alpha'] != args.alpha:
        raise ValueError(
            f"alpha mismatch: expected {args.alpha}, found {metadata['alpha']}")
    if metadata['encoder'] != args.encoder:
        raise ValueError(
            f"Encoder mismatch: expected {args.encoder}, found {metadata['encoder']}")
    if metadata['params'] != args.params:
        raise ValueError(
            f"Params mismatch: expected {args.params}, found {metadata['params']}")
    if metadata['position'] != args.position:
        raise ValueError(
            f"Position mismatch: expected {args.position}, found {metadata['position']}")

    weights = loaded_data['weights']
    for i, layer in enumerate(list_lora_layers):
        layer_weights = weights[f'layer_{i}']
        if 'q' in args.params and 'q_proj' in layer_weights:
            layer.q_proj.w_lora_A.data.copy_(
                layer_weights['q_proj']['w_lora_A'])
            layer.q_proj.w_lora_B.data.copy_(
                layer_weights['q_proj']['w_lora_B'])
        if 'k' in args.params and 'k_proj' in layer_weights:
            layer.k_proj.w_lora_A.data.copy_(
                layer_weights['k_proj']['w_lora_A'])
            layer.k_proj.w_lora_B.data.copy_(
                layer_weights['k_proj']['w_lora_B'])
        if 'v' in args.params and 'v_proj' in layer_weights:
            layer.v_proj.w_lora_A.data.copy_(
                layer_weights['v_proj']['w_lora_A'])
            layer.v_proj.w_lora_B.data.copy_(
                layer_weights['v_proj']['w_lora_B'])
        if 'o' in args.params and 'proj' in layer_weights:
            layer.proj.w_lora_A.data.copy_(layer_weights['proj']['w_lora_A'])
            layer.proj.w_lora_B.data.copy_(layer_weights['proj']['w_lora_B'])

    print(f'LoRA weights loaded from {load_path}')
