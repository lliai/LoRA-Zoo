import os
os.environ["CUDA_VISIBLE_DEVICES"] = "6"
import torch
import torch.nn as nn
import random
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import pickle
import numpy as np
from scipy.optimize import minimize
from utils import clip_classifier

from loralib.utils import apply_attn


# 假设已有辅助函数：find_layers，用于遍历模块中符合条件的层
def find_layers(module, layers=[nn.Linear]):
    """
    遍历 module 内的所有子模块，将符合 layers 类型的模块以字典形式返回。
    """
    results = {}
    for name, sub_module in module.named_modules():
        if type(sub_module) in layers:
            results[name] = sub_module
    return results

def cal_layers_lora_rank(args, clip_model, train_loader, dataset):
    """
    使用优化方法求解 CLIP 模型中各层 LoRA 的 rank 分配。

    思路：
    1. 通过 hook 收集各层的激活信息（这里用 L2 范数均值作为激活重要性指标）。
    2. 对每层，根据其权重尺寸计算参数因子（例如 in_features + out_features）。
    3. 构造目标函数 f(x) = -sum_i (importance_i * log(x_i))，
       使得激活越大的层（importance_i 较大）获得较大的 rank。
    4. 添加约束条件：总参数量 sum_i (factor_i * x_i) <= args.rank_budget。
    5. 设定每层 rank 的边界（下界为 1，上界为 args.max_rank），并使用 SLSQP 求解。
    """
    # 用于存储每层激活的重要性和权重信息
    layer_activations = {}
    layer_factors = {}  # 记录每层增加 LoRA 参数时的因子

    def create_activation_hook(layer_name):
        def hook_fn(module, input, output):
            # 使用输出 tensor 的 L2 范数均值作为该层激活的重要性指标
            act = output[0].detach()
            importance = act.norm(p=2, dim=-1).mean().item()
            layer_activations[layer_name] = importance

        return hook_fn

    hooks = []
    # 同时遍历 text 与 vision 分支的 resblocks（假设二者数量一致）
    for i, (text_resblock, vision_resblock) in enumerate(zip(clip_model.transformer.resblocks,
                                                              clip_model.visual.transformer.resblocks)):
        # 遍历当前 resblock 中的 nn.Linear 层
        text_layers = find_layers(text_resblock, layers=[nn.Linear])
        vision_layers = find_layers(vision_resblock, layers=[nn.Linear])
        for name, layer in text_layers.items():
            hook_name = f"text_resblock_{i}_{name}"
            hooks.append(layer.register_forward_hook(create_activation_hook(hook_name)))
            # 参数因子采用权重尺寸之和
            layer_factors[hook_name] = layer.weight.shape[0] + layer.weight.shape[1]
        for name, layer in vision_layers.items():
            hook_name = f"vision_resblock_{i}_{name}"
            hooks.append(layer.register_forward_hook(create_activation_hook(hook_name)))
            layer_factors[hook_name] = layer.weight.shape[0] + layer.weight.shape[1]

    # 利用一个 batch 进行前向传播以触发 hook 收集激活数据
    clip_model.eval()
    for images, target in tqdm(train_loader, desc="Collecting activations"):
        images, target = images.cuda(), target.cuda()
        with torch.no_grad():
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                if args.encoder in ['text', 'both']:
                    # 对 text 分支，随机选择一个 class name
                    class_name = random.choice(dataset.classnames)
                    clip_classifier([class_name], dataset.template, clip_model)
                if args.encoder in ['vision', 'both']:
                    clip_model.encode_image(images)
        break  # 仅使用第一个 batch

    # 移除所有 hook
    for h in hooks:
        h.remove()

    # 整理各层数据，并保持顺序一致
    keys = list(layer_activations.keys())
    importance = np.array([layer_activations[k] for k in keys])
    factors = np.array([layer_factors[k] for k in keys])
    n_layers = len(keys)

    # 定义目标函数：最小化 -sum_i (importance_i * log(x_i))
    def objective(x):
        return -np.sum(importance * np.log(x))

    # 目标函数梯度
    def objective_grad(x):
        return -importance / x

    # 约束函数：总参数量 sum_i (factors[i] * x[i]) <= args.rank_budget
    def constraint_func(x):
        return args.rank_budget - np.sum(factors * x)

    cons = {'type': 'ineq', 'fun': constraint_func}

    # 设置每层 rank 的搜索边界：下界 1，上界 args.max_rank
    bounds = [(1, args.max_rank) for _ in range(n_layers)]

    # 初始猜测：各层初始都设置为 1
    x0 = np.ones(n_layers, dtype=np.float32) * 8

    result = minimize(fun=objective, x0=x0, jac=objective_grad,
                      bounds=bounds, constraints=cons, method='SLSQP')
    
    if not result.success:
        raise ValueError("优化求解失败: " + result.message)

    optimal_ranks = result.x

    # 整理结果：将每层名称与对应的最优 rank 关联起来
    rank_allocation = {k: r for k, r in zip(keys, optimal_ranks)}
    
    # 可选：将分配结果保存到文件
    with open('rank_allocation.pkl', 'wb') as f:
        pickle.dump(rank_allocation, f)

    return rank_allocation, layer_factors

# --------------------- 使用示例 ---------------------
if __name__ == "__main__":
    import clip
    from types import SimpleNamespace
    
    # 构造参数
    args = SimpleNamespace(
        rank_budget=2*1024*1024,  # LoRA参数总预算
        max_rank=32,      # 每层最大rank值
        encoder='both',    # 同时优化text和vision编码器
        position='all',     # 所有层都添加LoRA
        backbone='ViT-B/16'
    )
    


    # 加载CLIP模型
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model, preprocess = clip.load(args.backbone, device=device)


    # 替换规范原有的attn层
    list_linear_layers = apply_attn(args, clip_model)
    clip_model = clip_model.cuda()

    # 构造一个简单的数据集
    class DummyDataset:
        def __init__(self):
            self.classnames = ["dog", "cat", "bird", "fish"]
            self.template = ["a photo of a {}."]
    
    # 生成随机数据用于演示
    batch_size = 4
    dummy_images = torch.randn(16, 3, 224, 224)  # 16张图片
    dummy_labels = torch.randint(0, 4, (16,))    # 随机标签
    dataset = DummyDataset()
    
    # 创建DataLoader
    train_loader = DataLoader(
        TensorDataset(dummy_images, dummy_labels),
        batch_size=batch_size,
        shuffle=True
    )
    
    # 计算最优rank分配
    try:
        rank_allocation, layer_factors = cal_layers_lora_rank(args, clip_model, train_loader, dataset)
        

        # 打印结果
        print("\n最优rank分配结果:")
        # rank_allocation : {layer_name: rank}
        # For example
        # text_resblock_0_q_proj: 16
        # vision_resblock_0_q_proj: 16
        # ...
        for layer_name, rank in rank_allocation.items():
            print(f"{layer_name}: {int(rank)}")
            

        # 计算总参数量
        total_params = sum(layer_factors[k] * rank_allocation[k] for k in rank_allocation)
        print(f"\n总参数量: {int(total_params)}")
        
    except Exception as e:
        print(f"计算过程出错: {str(e)}")
