import os
import json
import torch
import torch.nn as nn
import pickle
from typing import Dict

from .layers import LoRALayer, PlainMultiheadAttentionLoRA, PlainMultiheadAttention, MultiheadAttentionAwLoRA
from .lora_lib import AwLinear

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

def apply_awlora(args, clip_model):
    list_lora_layers = []
    
    # 加载rank分配文件
    rank_allocation = {}
    json_path = f'output/awlora/json/rank_allocation_{args.theta_type}_{args.dataset}_{args.shots}.json'
    if os.path.exists(json_path):
        with open(json_path, 'r') as f:
            rank_allocation = json.load(f)
    
    if args.encoder == 'text' or args.encoder == 'both':
        indices = INDEX_POSITIONS_TEXT[args.position]
        text_encoder = clip_model.transformer
        for i, block in enumerate(text_encoder.resblocks):
            if i in indices:
                for name, submodule in block.named_children():
                    if isinstance(submodule, nn.MultiheadAttention):
                        # 为每个投影层获取对应的rank
                        ranks = {
                            'q': int(rank_allocation.get(f'text_resblock_{i}_attn.q_proj', args.r)),
                            'k': int(rank_allocation.get(f'text_resblock_{i}_attn.k_proj', args.r)),
                            'v': int(rank_allocation.get(f'text_resblock_{i}_attn.v_proj', args.r)),
                            'o': int(rank_allocation.get(f'text_resblock_{i}_attn.o_proj', args.r))
                        }
                        new_multi_head_lora = MultiheadAttentionAwLoRA(
                            submodule, 
                            enable_lora=args.params, 
                            ranks=ranks, 
                            lora_alpha=args.alpha, 
                            dropout_rate=args.dropout_rate, 
                            svd_lora=args.svd_lora, 
                            svd_step=args.svd_step,
                            rand_lora=args.rand_lora)
                        setattr(block, name, new_multi_head_lora)
                        list_lora_layers.append(new_multi_head_lora)

    if args.encoder == 'vision' or args.encoder == 'both':
        indices = INDEX_POSITIONS_VISION[args.backbone][args.position]
        vision_encoder = clip_model.visual.transformer
        for i, block in enumerate(vision_encoder.resblocks):
            if i in indices:
                for name, submodule in block.named_children():
                    if isinstance(submodule, nn.MultiheadAttention):
                        # 为每个投影层获取对应的rank
                        ranks = {
                            'q': int(rank_allocation.get(f'vision_resblock_{i}_attn.q_proj', args.r)),
                            'k': int(rank_allocation.get(f'vision_resblock_{i}_attn.k_proj', args.r)),
                            'v': int(rank_allocation.get(f'vision_resblock_{i}_attn.v_proj', args.r)),
                            'o': int(rank_allocation.get(f'vision_resblock_{i}_attn.o_proj', args.r))
                        }
                        new_multi_head_lora = MultiheadAttentionAwLoRA(
                            submodule, 
                            enable_lora=args.params, 
                            ranks=ranks, 
                            lora_alpha=args.alpha, 
                            dropout_rate=args.dropout_rate, 
                            svd_lora=args.svd_lora, 
                            svd_step=args.svd_step,
                            rand_lora=args.rand_lora)
                        setattr(block, name, new_multi_head_lora)
                        list_lora_layers.append(new_multi_head_lora)
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
