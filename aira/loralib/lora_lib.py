#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import math
from re import S
from regex import W
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import wandb
from sklearn.decomposition import TruncatedSVD
from typing import Optional, List, Tuple
import random

def get_lora_plus_parameters(model, bias='none'):
    """
    Returns the parameters of the model that should be optimized.
    """
    lora_a_params = []
    lora_b_params = []
    params = []
    for name, param in model.named_parameters():
        if bias == 'none':
            if 'lora_A' in name:
                lora_a_params.append(param)
            if 'lora_B' in name:
                lora_b_params.append(param)
        elif bias == 'all':
            if 'lora_A' in name:
                lora_a_params.append(param)
            if 'lora_B' in name:
                lora_b_params.append(param)
            if 'bias' in name:
                params.append(param)
        elif bias == 'lora_only':
            if 'lora_' in name:
                params.append(param)
                bias_name = name.split('lora_')[0] + 'bias'
                if bias_name in model.state_dict():
                    bias_param = dict(model.named_parameters())[bias_name]
                    params.append(bias_param)
        else:
            raise NotImplementedError
    if params:
        return params, lora_a_params, lora_b_params
    else:
        return lora_a_params, lora_b_params

class LoRALayer():
    def __init__(
        self, 
        r: int, 
        lora_alpha: float, 
        lora_dropout: float,
        merge_weights: bool,
    ):
        self.r = r
        self.lora_alpha = lora_alpha
        # Optional dropout
        if lora_dropout > 0.:
            self.lora_dropout = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout = lambda x: x
        # Mark the weight as unmerged
        self.merged = False
        self.merge_weights = merge_weights

#正交初始化小loar矩阵
def init_module_weights(target: torch.Tensor, sigma: float):
    # Initialize weights with orthogonal initialization
    nn.init.orthogonal_(target)

class AwLinear(nn.Linear, LoRALayer):
    # LoRA implemented in a dense layer
    # 激活敏感的秩分量动态加权（Activation-aware Rank Weighting）
    # 核心思路：通过输入激活的统计特征，动态调整不同秩分量在更新中的贡献权重。
    def __init__(
        self, 
        existing_linear: nn.Linear,
        r: int = 0, 
        lora_alpha: int = 1, 
        lora_dropout: float = 0.,
        fan_in_fan_out: bool = False, # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        merge_weights: bool = True,
        svd_lora: bool = False,
        svd_step: int = 1,
        rand_lora: bool = False,
        **kwargs
    ):
        super().__init__(
            in_features=existing_linear.in_features, 
            out_features=existing_linear.out_features)
        self.load_state_dict(existing_linear.state_dict())
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                           merge_weights=merge_weights)
        self.fan_in_fan_out = fan_in_fan_out
        
        self.svd_lora = svd_lora     
        self.svd_step = svd_step
        self.rand_lora = rand_lora

        # Actual trainable parameters
        in_features = existing_linear.in_features
        out_features = existing_linear.out_features
        if r > 0:
            self.lora_A = nn.Parameter(self.weight.new_zeros((r, in_features)))
            self.lora_B = nn.Parameter(self.weight.new_zeros((out_features, r)))

            self.scaling = self.lora_alpha
            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False
        self.reset_parameters()
        if fan_in_fan_out:
            self.weight.data = self.weight.data.T

    def reset_parameters(self):
        if hasattr(self, 'lora_A'):
            if self.svd_lora and self.svd_step == 1:
                # 使用SVD初始化
                U, S, V = torch.svd(self.weight)
                self.lora_A.data = V[:self.r, :]
                self.lora_B.data = U[:, :self.r] @ torch.diag(S[:self.r])
            elif self.svd_lora and self.svd_step == 2:
                U, S, V = torch.svd(self.weight)
                self.lora_A.data = torch.diag(S[:self.r]) @ V[:self.r, :]
                self.lora_B.data = U[:, :self.r]
            elif self.svd_lora and self.svd_step == 3:
                U, S, V = torch.svd(self.weight)
                self.lora_A.data = torch.diag(torch.sqrt(S[:self.r])) @ V[:self.r, :]
                self.lora_B.data = U[:, :self.r] @ torch.diag(torch.sqrt(S[:self.r]))
            elif self.rand_lora:
                self.lora_A.data = torch.randn(self.r, self.in_features, requires_grad=True)
                self.lora_B.data = torch.randn(self.out_features, self.r, requires_grad=True)
            else:
                # initialize A the same way as the default for nn.Linear and B to zero
                nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
                nn.init.zeros_(self.lora_B)

    def train(self, mode: bool = True):
        def T(w):
            return w.T if self.fan_in_fan_out else w
        nn.Linear.train(self, mode)
        if self.merge_weights and self.merged:
            # Make sure that the weights are not merged
            if self.r > 0:
                lora_A = self.lora_A
                lora_B = self.lora_B
                self.weight.data -= T(lora_B @ lora_A) * self.scaling
            self.merged = False
    
    def eval(self):
        def T(w):
            return w.T if self.fan_in_fan_out else w
        nn.Linear.eval(self)
        if self.merge_weights and not self.merged:
            # Merge the weights and mark it
            if self.r > 0:
                lora_A = self.lora_A
                lora_B = self.lora_B
                self.weight.data += T(lora_B @ lora_A) * self.scaling
            self.merged = True

    def forward(self, x: torch.Tensor):
        def T(w):
            return w.T if self.fan_in_fan_out else w
        if self.r > 0 and not self.merged:
            result = F.linear(x, T(self.weight), bias=self.bias)
            
            # 新增激活感知加权逻辑
            with torch.no_grad():
                # 计算激活感知的通道重要性分数 
                s_i = torch.mean(torch.abs(result), dim=0)  
                # 归一化s_i
                s_i = (s_i - s_i.min()) / (s_i.max() - s_i.min())
            
            # 应用激活权重到输入特征（需要保持梯度）
            # 激活感知输出
            x = self.lora_dropout(x)
            lora_A = self.lora_A
            lora_B = self.lora_B
            lora_output = F.linear(x, T(lora_B @ lora_A)) * s_i.detach()

            result += lora_output * self.scaling
            
            return result
        else:
            return F.linear(x, T(self.weight), bias=self.bias)

'''           
class AwRankLinear(nn.Linear, LoRALayer):
    def __init__(self, existing_linear: nn.Linear, r: int = 0, lora_alpha: int = 1, 
                 lora_dropout: float = 0., fan_in_fan_out: bool = False, 
                 merge_weights: bool = True, beta: float = 0.1, budget: float = 0.5,
                 R_max: int = 8, update_freq: int = 100, **kwargs):
        super().__init__(existing_linear.in_features, existing_linear.out_features)
        self.load_state_dict(existing_linear.state_dict())
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout, 
                          merge_weights=merge_weights)
        
        self.fan_in_fan_out = fan_in_fan_out
        # 动态秩配置
        self.beta = beta  # 重要性平滑因子
        self.budget = budget  # 参数预算比例
        self.R_max = R_max  # 最大允许秩
        self.update_freq = update_freq  # 更新频率（迭代次数）
        
        # 动态秩相关参数
        self.current_rank = r
        self.importance = torch.zeros(self.in_features)  # 通道重要性累计
        self.step_counter = 0  # 更新计数器
        
        # 预分配最大秩参数空间
        self.lora_A = nn.Parameter(torch.zeros(R_max, self.in_features))
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, R_max))
        self.scaling = self.lora_alpha / self.current_rank if self.current_rank > 0 else 0
        
        # 参数合并相关属性
        self.merge_weights = merge_weights
        self.merged = False  # 初始状态为未合并
        
        # 初始化参数
        self.reset_parameters()
        if fan_in_fan_out:
            self.weight.data = self.weight.data.T

    def reset_parameters(self):
        if hasattr(self, 'lora_A'):
            # 正交初始化 + 缩放
            nn.init.orthogonal_(self.lora_A, gain=math.sqrt(5))
            nn.init.zeros_(self.lora_B)
            
    def _update_importance(self, x):
        """更新通道重要性统计"""
        with torch.no_grad():
            # 计算当前batch的通道重要性（L1范数）
            batch_imp = torch.mean(torch.abs(x), dim=0)  # [in_features]
            # 指数移动平均
            self.importance = self.beta * self.importance + (1 - self.beta) * batch_imp
            
    def _allocate_rank(self):
        """动态分配当前秩"""
        # 计算可用参数预算
        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        target_params = total_params * self.budget
        
        # 获取重要性排序
        sorted_indices = torch.argsort(self.importance, descending=True)
        
        # 贪心选择满足预算的最大秩
        selected_rank = 0
        accumulated_params = 0
        for r in range(1, self.R_max+1):
            req_params = r * (self.in_features + self.out_features)
            if accumulated_params + req_params <= target_params:
                selected_rank = r
                accumulated_params += req_params
            else:
                break
                
        # 更新当前秩
        self.current_rank = selected_rank
        self.scaling = self.lora_alpha / self.current_rank if self.current_rank > 0 else 0
        
        # 参数裁剪（保留重要通道）
        active_indices = sorted_indices[:self.current_rank]
        self.lora_A.data = self.lora_A.data[:self.current_rank, active_indices]
        self.lora_B.data = self.lora_B.data[:, :self.current_rank]
        
    def forward(self, x: torch.Tensor):
        def T(w):
            return w.T if self.fan_in_fan_out else w
        
        # 更新统计量
        self._update_importance(x)
        self.step_counter += 1
        
        # 周期性更新秩分配
        if self.training and (self.step_counter % self.update_freq == 0):
            self._allocate_rank()
            self.step_counter = 0  # 重置计数器
            
        if self.current_rank > 0 and not self.merged:
            # 应用激活加权的低秩更新
            x_weighted = self.lora_dropout(x) * self.importance.detach()
            lora_output = (x_weighted @ self.lora_A[:self.current_rank, :].T 
                           @ self.lora_B[:, :self.current_rank].T) * self.scaling
            return F.linear(x, T(self.weight), self.bias) + lora_output
        else:
            return F.linear(x, T(self.weight), self.bias)
            
    def train(self, mode: bool = True):
        """ 切换到训练模式 """
        super().train(mode)
        if mode:
            # 如果启用了合并功能且当前已合并，则取消合并
            if self.merge_weights and self.merged:
                if self.current_rank > 0:
                    # 根据当前秩取消合并
                    lora_A = self.lora_A[:self.current_rank, :]
                    lora_B = self.lora_B[:, :self.current_rank]
                    self.weight.data -= (lora_B @ lora_A) * self.scaling
                self.merged = False
            # 重置重要性统计
            self.importance = torch.zeros_like(self.importance)
        return self

    def eval(self):
        """ 切换到评估模式 """
        super().eval()
        # 如果启用了合并功能且当前未合并，则执行合并
        if self.merge_weights and not self.merged:
            if self.current_rank > 0:
                # 根据当前秩合并参数
                lora_A = self.lora_A[:self.current_rank, :]
                lora_B = self.lora_B[:, :self.current_rank]
                self.weight.data += (lora_B @ lora_A) * self.scaling
            self.merged = True
        return self

    def _merge_weights(self):
        """ 手动合并权重 """
        if not self.merged:
            self.eval()

    def _unmerge_weights(self):
        """ 手动取消合并权重 """
        if self.merged:
            self.train()
'''           
