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


class RankAllocator(object):
    """
    The RankAllocator for AdaLoRA Model that will be called every training step. 
    Paper: https://openreview.net/pdf?id=lq62uWRJjiY

    Args:
        model: the model that we apply AdaLoRA to.
        lora_r (`int`): The initial rank for each incremental matrix.
        target_rank (`int`): The target average rank of incremental matrix.
        init_warmup (`int`): The steps of initial fine-tuning warmup.
        final_warmup (`int`): The step of final fine-tuning.
        mask_interval (`int`): The time internval between two budget allocations.
        beta1 (`float`): The hyperparameter of EMA for sensitivity smoothing.
        beta2 (`float`): The hyperparameter of EMA for undertainty quantification.
        total_step (`int`): The total training steps, correctly configured before training.
        target_total_rank (`Optinal[int]`): The speficified final total rank. 
        wandb_writter (`Bool`): if use Wandb LoggerWritter. 
        wandb_writter_loginterval (`int`): The logging interval of LoggerWritter. 
    """
    def __init__(
        self, model, 
        lora_r:int,
        target_rank:int, 
        init_warmup:int, 
        final_warmup:int,
        mask_interval:int,
        beta1:float, 
        beta2:float, 
        total_step:Optional[int]=None, 
        target_total_rank:Optional[int]=None,
        wandb_writter:bool=True,
        wandb_writter_loginterval:int=500, 
    ):
        self.ave_target_rank = target_rank 
        self.target_rank = target_total_rank if target_total_rank is not None else 0
        self.lora_init_rank = lora_r 
        self.initial_warmup = init_warmup
        self.final_warmup = final_warmup 
        self.mask_interval = mask_interval
        self.beta1 = beta1
        self.beta2 = beta2
        self.total_step = total_step if total_step is not None else 0

        self.model = model
        self.ipt = {} 
        self.exp_avg_ipt = {}
        self.exp_avg_unc = {}
        self.cat_ipt = {}
        self.rank_pattern = {} 
        self.get_lora_param_name()

        self.wandb_writter = wandb_writter
        self.log_interval = wandb_writter_loginterval 

        assert (self.beta1<1 and self.beta1>0)
        assert (self.beta2<1 and self.beta2>0)

    def set_total_step(self, total_step:int): 
        # Set total step number 
        self.total_step = total_step
        assert self.total_step>self.initial_warmup+self.final_warmup

    def get_rank_pattern(self):
        # Return rank pattern 
        return self.rank_pattern

    def get_lora_param_name(self):
        # Prepare the budget scheduler 
        self.name_set = set() 
        self.total_rank = 0 
        self.shape_dict = {}
        for n,p in self.model.named_parameters():
            if "lora_A" in n: 
                name_mat = n.replace("lora_A", "%s")
                self.name_set.add(name_mat)
                self.total_rank += p.size(0) 
                self.shape_dict[n] = p.shape
            if "lora_B" in n:
                self.shape_dict[n] = p.shape
        self.name_set = list(sorted(self.name_set)) 
        if self.target_rank is None:
            self.target_rank = self.ave_target_rank * len(self.name_set) 

    def schedule_threshold(self, step:int):
        # Global budget schedule
        mask_ind = False 
        target_rank = self.target_rank 
        initial_warmup = self.initial_warmup 
        final_warmup = self.final_warmup 
        total_step = self.total_step 
        self.global_step = step
        if step <= initial_warmup: 
            # Initial warmup 
            curr_rank = self.total_rank 
            mask_ind = False 
        elif step > total_step - final_warmup: 
            # Final fine-tuning 
            curr_rank = self.target_rank 
            # Fix the rank pattern by 
            # always masking the same unimportant singluar values 
            mask_ind = True 
        else: 
            # Budget decreasing 
            mul_coeff = 1-(step-initial_warmup)/(total_step-final_warmup-initial_warmup)
            curr_rank = target_rank + (self.total_rank-target_rank)*(mul_coeff**3)
            curr_rank = int(curr_rank)
            mask_ind = True if step % self.mask_interval == 0 else False 
        return curr_rank, mask_ind 


    def update_ipt(self, model): 
        for n,p in model.named_parameters():
            if "lora_" in n: 
                if n not in self.ipt:
                    self.ipt[n] = torch.zeros_like(p)
                    self.exp_avg_ipt[n] = torch.zeros_like(p) 
                    self.exp_avg_unc[n] = torch.zeros_like(p) 
                with torch.no_grad():
                    # Calculate sensitivity 
                    self.ipt[n] = (p * p.grad).abs().detach()
                    # Update sensitivity 
                    self.exp_avg_ipt[n] = self.beta1 * self.exp_avg_ipt[n] + \
                                        (1-self.beta1)*self.ipt[n]
                    # Update uncertainty 
                    self.exp_avg_unc[n] = self.beta2 * self.exp_avg_unc[n] + \
                                        (1-self.beta2)*(self.ipt[n]-self.exp_avg_ipt[n]).abs()

    def calculate_score(self, n, p=None, metric="ipt"):
        if metric == "ipt":
            # Combine the senstivity and uncertainty 
            ipt_score = self.exp_avg_ipt[n] * self.exp_avg_unc[n]
        elif metric == "mag":
            ipt_score = p.abs().detach().clone() 
        else:
            raise ValueError("Unexcptected Metric: %s"%metric)
        return ipt_score 

    def _combine_ipt(self, ipt_E, ipt_AB):
        ipt_AB = ipt_AB.sum(dim=1, keepdim=False)
        sum_ipt = ipt_E.view(-1) + ipt_AB.view(-1)
        return sum_ipt

    def mask_to_target_rank(self, model, curr_rank): 
        is_dict = {}
        combine_dict = {} 
        singular_dict = {}
        # Calculate the importance score for each sub matrix 
        for n,p in model.named_parameters(): 
            if "lora_A" in n: 
                rdim, hdim_a = p.shape
                ipt_score = self.calculate_score(n, metric="ipt")
                comb_ipt = torch.mean(ipt_score, dim=1, keepdim=True)
                name_mat = n.replace("lora_A", "%s")
                if name_mat not in combine_dict: 
                    combine_dict[name_mat] = [comb_ipt]
                else:
                    combine_dict[name_mat].append(comb_ipt)
            if "lora_B" in n: 
                hdim_b, rdim = p.shape 
                ipt_score = self.calculate_score(n, metric="ipt")
                comb_ipt = torch.mean(ipt_score, dim=0, keepdim=False).view(-1, 1)
                name_mat = n.replace("lora_B", "%s")
                if name_mat not in combine_dict: 
                    combine_dict[name_mat] = [comb_ipt]
                else:
                    combine_dict[name_mat].append(comb_ipt)
            if "lora_E" in n:
                ipt_score = self.calculate_score(n, p=p, metric="ipt")                
                name_mat = n.replace("lora_E", "%s")
                singular_dict[name_mat] = ipt_score

        # Combine the importance scores 
        all_is = []
        for name_mat in combine_dict: 
            ipt_E = singular_dict[name_mat] 
            ipt_AB = torch.cat(combine_dict[name_mat], dim=1)
            sum_ipt = self._combine_ipt(ipt_E, ipt_AB)
            name_E = name_mat%"lora_E"
            is_dict[name_E] = sum_ipt.view(-1, 1)
            all_is.append(sum_ipt.view(-1))

        # Calculate the masking threshold 
        mask_threshold = torch.kthvalue(torch.cat(all_is), (self.total_rank-curr_rank))[0].item()

        # Mask out unimportant singular values 
        with torch.no_grad():
            curr_sum_rank = 0
            sum_param = 0
            for n,p in model.named_parameters():
                if "lora_E" in n: 
                    p.data.masked_fill_(is_dict[n]<=mask_threshold, 0.0)
                    ranknum = (is_dict[n]>mask_threshold).sum().item() 

                    if self.global_step%self.log_interval==0:
                        wandb.log({"Ranknum/%s"%(n,): ranknum}, step=self.global_step) 
                        self.rank_pattern[n] = ranknum 
                        curr_sum_rank += ranknum 
                        sum_param += ranknum*self.shape_dict[n.replace("lora_E", "lora_A")][1]  
                        sum_param += ranknum*self.shape_dict[n.replace("lora_E", "lora_B")][0]  

            if self.global_step%self.log_interval==0:
                wandb.log({"Budget/total_rank": curr_sum_rank}, step=self.global_step)
                wandb.log({"Budget/mask_threshold": mask_threshold}, step=self.global_step)
                wandb.log({"Budget/sum_param": sum_param}, step=self.global_step)

        return mask_threshold


    def update_and_mask(self, model, global_step):
        if global_step<self.total_step-self.final_warmup:
            # Update importance scores element-wise 
            self.update_ipt(model)
            # do not update ipt during final fine-tuning 
        # Budget schedule
        curr_rank, mask_ind = self.schedule_threshold(global_step)
        if mask_ind:
            # Mask to target budget 
            mask_threshold = self.mask_to_target_rank(model, curr_rank) 
        else:
            mask_threshold = None 
        self._maybe_wandb_writter_log(model)
        return curr_rank, mask_threshold

    def _maybe_wandb_writter_log(self, model):
        if self.global_step%self.log_interval==0:
            with torch.no_grad():
                regu_loss = []
                for n,p in model.named_parameters():
                    if "lora_A" in n or "lora_B" in n:
                        mat = p.data.detach().clone()
                        mat_cov = mat @ mat.T if "lora_A" in n else mat.T @ mat 
                        I = torch.eye(*mat_cov.size(), out=torch.empty_like(mat_cov))
                        I.requires_grad = False
                        orth_regu = torch.norm(mat_cov-I, p="fro")
                        regu_loss.append(orth_regu.item())
                        wandb.log({"Orth_regu_loss/%s"%n: orth_regu.item()}, step=self.global_step)
                wandb.log({"train/orth_regu_loss": sum(regu_loss)/len(regu_loss)}, step=self.global_step)

def compute_orth_regu(model, regu_weight=0.1):
    # The function to compute orthongonal regularization for SVDLinear in `model`. 
    regu_loss, num_param = 0., 0
    for n,p in model.named_parameters():
        if "lora_A" in n or "lora_B" in n:
            para_cov = p @ p.T if "lora_A" in n else p.T @ p 
            I = torch.eye(*para_cov.size(), out=torch.empty_like(para_cov))
            I.requires_grad = False
            regu_loss += torch.norm(para_cov-I, p="fro")
            num_param += 1
    return regu_weight*regu_loss/num_param

class SVDLinear(nn.Linear, LoRALayer):
    def __init__(
            self, 
            existing_linear: nn.Linear,
            r: int = 0, 
            lora_alpha: int = 1, 
            lora_dropout: float = 0.,
            fan_in_fan_out: bool = False, 
            merge_weights: bool = True,
            **kwargs
    ):
        super().__init__(
            in_features=existing_linear.in_features, 
            out_features=existing_linear.out_features)
        
        self.load_state_dict(existing_linear.state_dict())
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout, merge_weights=merge_weights)
        self.fan_in_fan_out = fan_in_fan_out

        # Actual trainable parameters
        in_features = existing_linear.in_features
        out_features = existing_linear.out_features
        if r > 0:
            self.lora_A = nn.Parameter(
                self.weight.new_zeros((r, in_features))
            )
            self.lora_E = nn.Parameter(
                self.weight.new_zeros(r, 1)
            ) 
            self.lora_B = nn.Parameter(
                self.weight.new_zeros((out_features, r))
            )
            self.ranknum = nn.Parameter(
                self.weight.new_zeros(1), requires_grad=False
            )
            self.ranknum.data.fill_(float(self.r))
            self.scaling = self.lora_alpha if self.lora_alpha>0 else float(self.r)   
            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False
            self.ranknum.requires_grad = False
        self.init_lora_param()
        if fan_in_fan_out:
            self.weight.data = self.weight.data.T

    def init_lora_param(self):
        if hasattr(self, 'lora_A'):
            # initialize A,B the same way as the default for nn.Linear 
            # and E (singular values) for zero 
            nn.init.zeros_(self.lora_E)
            nn.init.normal_(self.lora_A, mean=0.0, std=0.02)
            nn.init.normal_(self.lora_B, mean=0.0, std=0.02)

    def train(self, mode: bool = True):
        def T(w):
            return w.T if self.fan_in_fan_out else w
        nn.Linear.train(self, mode)
        if self.merge_weights and self.merged:
            # Make sure that the weights are not merged
            if self.r > 0:
                self.weight.data -= T(
                    self.lora_B @ (self.lora_A*self.lora_E)
                ) * self.scaling / (self.ranknum+1e-5)
            self.merged = False
    
    def eval(self):
        def T(w):
            return w.T if self.fan_in_fan_out else w
        nn.Linear.eval(self)
        if self.merge_weights and not self.merged:
            # Merge the weights and mark it
            if self.r > 0:
                self.weight.data += T(
                    self.lora_B @ (self.lora_A * self.lora_E)
                ) * self.scaling / (self.ranknum+1e-5)
            self.merged = True

    def forward(self, x: torch.Tensor):
        def T(w):
            return w.T if self.fan_in_fan_out else w
        if self.r > 0 and not self.merged:
            result = F.linear(x, T(self.weight), bias=self.bias)
            if self.r > 0:
                result += (
                    self.lora_dropout(x) @ (self.lora_A * self.lora_E).T @ self.lora_B.T
                ) * self.scaling / (self.ranknum+1e-5)
            return result
        else:
            return F.linear(x, T(self.weight), bias=self.bias)
        
class dynamic(nn.Module):
    def __init__(
        self,
        maximum_rank: int = 1,
    ):
        '''
        maximum_rank: maximum rank of the input matrix
        '''
        super(dynamic, self).__init__()
        self.maximum_rank = maximum_rank

        self.frozen = False
        self.current_rank = 0
        self.prune_list = []

    def get_dimension(self):
        return self.maximum_rank
    
    def get_rank(self):
        return self.current_rank
    
    def set_rank(self, 
                 rank, 
                 frozen=False,):
        self.current_rank = max(0, min(rank, self.get_dimension()))
        self.frozen = frozen

    def forward(self, inputs: torch.Tensor, mode: bool = False):
        if self.training or mode:
            if self.frozen:
                
                pr = inputs[:,:self.get_rank()].detach()
                r = inputs[:,self.get_rank()]
                
                if len(r.shape) == 1:
                    r = r.unsqueeze(-1)
                result = torch.cat([pr,r],dim=-1)

                return result * math.sqrt(self.get_dimension()/(self.get_rank()+1)) 
            else:
                return inputs[:,:self.get_rank()+1] * math.sqrt(self.get_dimension()/(self.get_rank()+1))

        else:
            # at test time, just return the reduced rank inputs
            return inputs[:,:self.get_rank()+1] * math.sqrt(self.get_dimension()/(self.get_rank()+1)) 

class DyLinear(nn.Linear, LoRALayer):
    # LoRA implemented in a dense layer
    def __init__(
        self, 
        existing_linear: nn.Linear,
        r: int = 0, 
        lora_alpha: int = 1, 
        lora_dropout: float = 0.,
        fan_in_fan_out: bool = False, # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        merge_weights: bool = True,
        **kwargs
    ):
        super().__init__(
            in_features=existing_linear.in_features, 
            out_features=existing_linear.out_features)
        self.load_state_dict(existing_linear.state_dict())
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                           merge_weights=merge_weights)
        self.fan_in_fan_out = fan_in_fan_out
        
        # Actual trainable parameters
        in_features = existing_linear.in_features
        out_features = existing_linear.out_features
        if r > 0:
            self.lora_A = nn.Parameter(self.weight.new_zeros((r, in_features)))
            self.lora_B = nn.Parameter(self.weight.new_zeros((out_features, r)))

            self.nd_lora_A = dynamic(maximum_rank=self.r)
            self.nd_lora_B = dynamic(maximum_rank=self.r)

            self.scaling = self.lora_alpha / self.r
            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False
        self.reset_parameters()
        if fan_in_fan_out:
            self.weight.data = self.weight.data.T

    def get_dimension(self):
        assert self.nd_lora_A.get_dimension() == self.nd_lora_B.get_dimension()
        return self.nd_lora_A.get_dimension()

    def get_rank(self):
        assert self.nd_lora_A.get_rank() == self.nd_lora_B.get_rank()
        return self.nd_lora_A.get_rank()

    def set_rank(self, rank, frozen=False):
        self.nd_lora_A.set_rank(rank, frozen=frozen)
        self.nd_lora_B.set_rank(rank, frozen=frozen)

    def reset_parameters(self):
        if hasattr(self, 'lora_A'):
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
                lora_A = self.nd_lora_A(self.lora_A.T, mode=mode).T
                lora_B = self.nd_lora_B(self.lora_B, mode=mode)
                self.weight.data -= T(lora_B @ lora_A) * self.scaling
            self.merged = False
    
    def eval(self):
        def T(w):
            return w.T if self.fan_in_fan_out else w
        nn.Linear.eval(self)
        if self.merge_weights and not self.merged:
            # Merge the weights and mark it
            if self.r > 0:
                lora_A = self.nd_lora_A(self.lora_A.T).T
                lora_B = self.nd_lora_B(self.lora_B)
                self.weight.data += T(lora_B @ lora_A) * self.scaling
            self.merged = True

    def forward(self, x: torch.Tensor):
        def T(w):
            return w.T if self.fan_in_fan_out else w
        if self.r > 0 and not self.merged:
            result = F.linear(x, T(self.weight), bias=self.bias)
            if self.r > 0:
                lora_A = self.nd_lora_A(self.lora_A.T).T
                lora_B = self.nd_lora_B(self.lora_B)
                result += (self.lora_dropout(x) @ lora_A.T @ lora_B.T) * self.scaling
            return result
        else:
            return F.linear(x, T(self.weight), bias=self.bias)
        
class AutoloraLinear(nn.Linear, LoRALayer):
    def __init__(
        self, 
        existing_linear: nn.Linear,
        r: int = 8, 
        lora_alpha: int = 1, 
        lora_dropout: float = 0.,
        fan_in_fan_out: bool = False, 
        merge_weights: bool = True,
        **kwargs
    ):
        in_features = existing_linear.in_features
        out_features = existing_linear.out_features

        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                           merge_weights=merge_weights)
        self.fan_in_fan_out = fan_in_fan_out
        # Actual trainable parameters
        if r > 0:
            self.lora_A = nn.Parameter(
                self.weight.new_zeros((r, in_features))
            )
            self.lora_B = nn.Parameter(
                self.weight.new_zeros((out_features, r))
            )
            self.ranknum = r
            self.scaling = self.lora_alpha if self.lora_alpha>0 else float(self.r)   
            # Freezing the pre-trained weight matrix
            #self.weight.requires_grad = False
            #self.merged = False
        self.reset_parameters()
        if fan_in_fan_out:
            self.weight.data = self.weight.data.T
            
    def reset_parameters(self):
        if hasattr(self, 'lora_A'):
            # initialize A,B the same way as the default for nn.Linear 
            nn.init.normal_(self.lora_A, mean=0.0, std=0.02)
            nn.init.normal_(self.lora_B, mean=0.0, std=0.02)
 

    def train(self, alphas):
        def T(w):
            return w.T if self.fan_in_fan_out else w
        nn.Linear.train(self)
        if self.merge_weights and self.merged:
            # Make sure that the weights are not merged
            if self.r > 0:
                for i in range(0, self.r):
                    # self.weight.data -= T(
                    #     torch.clamp(alphas[i], 1e-6, 1-1e-6) * \
                    #         torch.matmul(torch.unsqueeze(self.lora_B[:,i], 1), torch.unsqueeze(self.lora_A[i,:], 0))
                    # ) * self.scaling / (self.ranknum+1e-5)
                    self.weight.data -= T(
                        alphas[i] * \
                            torch.matmul(torch.unsqueeze(self.lora_B[:,i], 1), torch.unsqueeze(self.lora_A[i,:], 0))
                    ) * self.scaling / (self.ranknum+1e-5)
            self.merged = False
    
    def eval(self, alphas):
        def T(w):
            return w.T if self.fan_in_fan_out else w
        nn.Linear.eval(self)
        if self.merge_weights and not self.merged:
            # Merge the weights and mark it
            if self.r > 0:
                for i in range(0, self.r):
                    # self.weight.data += T(
                    #     torch.clamp(alphas[i], 1e-6, 1-1e-6) * \
                    #        torch.matmul(torch.unsqueeze(self.lora_B[:,i], 1), torch.unsqueeze(self.lora_A[i,:], 0))
                    # ) * self.scaling / (self.ranknum+1e-5)
                    self.weight.data += T(
                        alphas[i] * \
                            torch.matmul(torch.unsqueeze(self.lora_B[:,i], 1), torch.unsqueeze(self.lora_A[i,:], 0))
                    ) * self.scaling / (self.ranknum+1e-5)
            self.merged = True

    def forward(self, x: torch.Tensor, alphas):
        def T(w):
            return w.T if self.fan_in_fan_out else w
        if self.r > 0 and not self.merged:
            #print(self.weight[0])
            
            result = F.linear(x, T(self.weight), bias=self.bias)
            #print(self.weight.shape)
            #print(F.softmax(alphas, dim=0))
            x = self.lora_dropout(x)
            if self.r > 0:
                for i in range(0, self.r):
                    # result += (
                    #     self.lora_dropout(x) @ (self.lora_A * self.lora_E).T @ self.lora_B.T
                    # ) * self.scaling / (self.ranknum+1e-5)
                    #print(F.softmax(alphas, dim=0)[i].data * torch.matmul(torch.unsqueeze(self.lora_B[:,i], 1), torch.unsqueeze(self.lora_A[i,:], 0)).shape        
                    # result += x @ T((
                    #     torch.clamp(alphas[i], 1e-6, 1-1e-6) * \
                    #         torch.matmul(torch.unsqueeze(self.lora_B[:,i], 1), torch.unsqueeze(self.lora_A[i,:], 0))
                    # ).T) * self.scaling / (self.ranknum+1e-5)
                    result += x @ T((
                        alphas[i] * \
                            torch.matmul(torch.unsqueeze(self.lora_B[:,i], 1), torch.unsqueeze(self.lora_A[i,:], 0))
                    ).T) * self.scaling / (self.ranknum+1e-5)
            return result
        else:
            return F.linear(x, T(self.weight), bias=self.bias)
        
    def maskforward(self, x, alphas, mark):
        def T(w):
            return w.T if self.fan_in_fan_out else w
        if self.r > 0 and not self.merged:
            result = F.linear(x, T(self.weight), bias=self.bias)
            x = self.lora_dropout(x)
            if self.r > 0:
                for i in range(0, self.r):
                    # result += (
                    #     self.lora_dropout(x) @ (self.lora_A * self.lora_E).T @ self.lora_B.T
                    # ) * self.scaling / (self.ranknum+1e-5)
                    # result += x @ T((
                    #     mark[i].item() * torch.clamp(alphas[i], 1e-6, 1-1e-6) * \
                    #         torch.matmul(torch.unsqueeze(self.lora_B[:,i], 1), torch.unsqueeze(self.lora_A[i,:], 0))
                    # ).T) * self.scaling / (self.ranknum+1e-5)
                    result += x @ T((
                        mark[i].item() * alphas[i] * \
                            torch.matmul(torch.unsqueeze(self.lora_B[:,i], 1), torch.unsqueeze(self.lora_A[i,:], 0))
                    ).T) * self.scaling / (self.ranknum+1e-5)
            return result
        else:
            return F.linear(x, T(self.weight), bias=self.bias)    


#正交初始化小loar矩阵
def init_module_weights(target: torch.Tensor, sigma: float):
    # Initialize weights with orthogonal initialization
    nn.init.orthogonal_(target)



class LoLinear(nn.Linear, LoRALayer):
    '''
    Lora in Lora Linear
    r: larger matrix rank
    lora_A_rank_ratio: ratio of lora A/B rank by weight rank
    max_rank: max rank of lora RA/RB
    '''
    def __init__(self, 
                 existing_linear: nn.Linear,
                 r: int = 0,
                 rank_ratio: float = 0.1,
                 max_rank: int = 0,
                 min_rank: int = 1,
                 lora_alpha: float = 1.0, 
                 lora_dropout: float = 0.0, 
                 merge_weights: bool = True, 
                 fan_in_fan_out: bool = False, **kwargs):
        super().__init__(
            in_features=existing_linear.in_features, 
            out_features=existing_linear.out_features)
        self.load_state_dict(existing_linear.state_dict())
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                           merge_weights=merge_weights)
        self.fan_in_fan_out = fan_in_fan_out
        
        # Actual trainable parameters
        in_features = existing_linear.in_features
        out_features = existing_linear.out_features
        self.r = math.floor(in_features * rank_ratio)
        if self.r > 0:
            self.lora_A = nn.Parameter(self.weight.new_zeros((self.r, in_features)))
            self.lora_B = nn.Parameter(self.weight.new_zeros((out_features, self.r)))

            # 需要设置初始较大的r
            max_rank = max_rank if max_rank < self.r else math.floor(self.r / 2)
            assert max_rank > min_rank, "max_rank must be larger than min_rank"
            self.lora_RA = nn.Parameter(torch.empty(max_rank, self.r))
            self.lora_RB = nn.Parameter(torch.empty(self.r, max_rank))

            # 动态调整rank
            self.nd_lora_RA = dynamic(maximum_rank=max_rank)
            self.nd_lora_RB = dynamic(maximum_rank=max_rank)

            self.scaling = self.lora_alpha / self.r

            init_module_weights(self.lora_RA, sigma=1e-5)
            init_module_weights(self.lora_RB, sigma=1e-5)
            # Freezing the pre-trained weight matrix
            self.lora_A.requires_grad = False
            self.lora_B.requires_grad = False
            self.weight.requires_grad = False
        self.init_lora_param(existing_linear)
        if fan_in_fan_out:
            self.weight.data = self.weight.data.T

    def init_lora_param(self, existing_linear: nn.Linear):
        
        # weight = existing_linear.weight.data
        # # Tensor转换np
        # U, S, Vh = torch.linalg.svd(weight)
        # a = Vh[: self.r, :].to(weight.device)
        # b = U[:, : self.r].to(weight.device)
        # S = torch.diag(S)
        # S_selected = S[: self.r, : self.r]

        # if hasattr(self, 'lora_A'):
        #     self.lora_A.data.copy_(a)
        #     self.lora_B.data.copy_(b)
        nn.init.kaiming_uniform_(self.lora_A)
        nn.init.kaiming_uniform_(self.lora_B)

    def get_dimension(self):
        assert self.nd_lora_RA.get_dimension() == self.nd_lora_RB.get_dimension()
        return self.nd_lora_RA.get_dimension()

    def get_rank(self):
        assert self.nd_lora_RA.get_rank() == self.nd_lora_RB.get_rank()
        return self.nd_lora_RA.get_rank()

    def compute_lora_saliency_score(self):
        saliency_scores = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                saliency_score = (param.grad * param).abs().sum()
                saliency_scores[name] = saliency_score
        return saliency_scores
    

    def set_rank(self, rank, frozen=False):
        self.nd_lora_RA.set_rank(rank, frozen=frozen)
        self.nd_lora_RB.set_rank(rank, frozen=frozen)

    def merge_BA(self, mode=True):
        def T(w):
            return w.T if self.fan_in_fan_out else w
        if self.r > 0:
            lora_RA = self.nd_lora_RA(self.lora_RA.T, mode=mode).T
            lora_RB = self.nd_lora_RB(self.lora_RB, mode=mode)           
            return T(self.lora_B @ lora_RB @ lora_RA @ self.lora_A).view(self.weight.shape)
        else:
            return 0

    def train(self, mode: bool = True):
        nn.Linear.train(self, mode)
        if self.merge_weights and self.merged:
            # Make sure that the weights are not merged
            if self.r > 0:
                merged_weight_lora = self.merge_BA()
                self.weight.data -= merged_weight_lora * self.scaling
            self.merged = False
    
    def eval(self):
        nn.Linear.eval(self)
        if self.merge_weights and not self.merged:
            # Merge the weights and mark it
            if self.r > 0:
                merged_weight_lora = self.merge_BA()
                self.weight.data += merged_weight_lora * self.scaling
            self.merged = True

    def forward(self, x: torch.Tensor):
        def T(w):
            return w.T if self.fan_in_fan_out else w
        if self.r > 0 and not self.merged:
            result = F.linear(x, T(self.weight), bias=self.bias)
            if self.r > 0:
                weight_BA = self.merge_BA()
                result += (self.lora_dropout(x) @ weight_BA) * self.scaling
            return result
        else:
            return F.linear(x, T(self.weight), bias=self.bias)    
        
# 检查LoLinear
# x = torch.randn(32, 512)
# existing_linear = nn.Linear(512, 512)
# lolinear = LoLinear(existing_linear, r=x.shape[0], max_rank=12, min_rank=1)
# lolinear.set_rank(2)
# output = lolinear(x)
# print(output.shape)

class dynamic_v2(nn.Module):
    def __init__(
        self,
        maximum_rank: int = 1,
    ):
        '''
        maximum_rank: maximum rank of the input matrix
        '''
        super(dynamic_v2, self).__init__()
        self.maximum_rank = maximum_rank

        self.frozen = False
        self.current_rank = 0
        self.prune_list = []

    def get_dimension(self):
        return self.maximum_rank
    
    def set_dimension(self, rank):
        self.maximum_rank = rank
    
    def get_rank(self):
        return self.current_rank
    
    def set_rank(self, 
                 rank, 
                 frozen=False,
                 prune: list = [bool]):
        self.prune_list = torch.tensor(prune, dtype=torch.bool)  # 转化为tensor bool
        self.current_rank = max(0, min(rank, self.get_dimension()))
        self.frozen = frozen

    def forward(self, inputs: torch.Tensor, mode: bool = False):
    
        # 根据prune的True值的位置来决定inputs需要训练的位置
        inputs = inputs[:, ~self.prune_list] # type: ignore
        # pr_inputs = inputs[:, self.prune_list] 
        if self.training or mode:

            if self.frozen:
                
                pr = inputs[:,:self.get_rank()].detach()
                r = inputs[:,self.get_rank()]
                
                if len(r.shape) == 1:
                    r = r.unsqueeze(-1)
                result = torch.cat([pr,r],dim=-1)
                return result * math.sqrt(self.get_dimension()/(self.get_rank()+1)) 
            else:
                return inputs[:,:self.get_rank()+1] * math.sqrt(self.get_dimension()/(self.get_rank()+1))

        else:
            # at test time, just return the reduced rank inputs
            return inputs[:,:self.get_rank()+1] * math.sqrt(self.get_dimension()/(self.get_rank()+1)) 


class LoLinear_v2(nn.Linear, LoRALayer):
    '''
    Lora in Lora Linear
    r: larger matrix rank
    lora_A_rank_ratio: ratio of lora A/B rank by weight rank
    max_rank: max rank of lora RA/RB
    '''
    def __init__(self, 
                 existing_linear: nn.Linear,
                 lora_alpha: float = 1.0, 
                 lora_dropout: float = 0.0, 
                 merge_weights: bool = True, 
                 fan_in_fan_out: bool = False,  
                 outer_rank: int = 0,           # 外层rank是多少，一般由rank_ratio决定
                 rank_ratio: float = 0.1,       
                 inner_max_rank: int = 0,           # 内层rank是多少
                 inner_min_rank: int = 1,             # 剪枝后最小的rank是多少 
                 if_prune: bool = True,         # 是否剪枝
                 **kwargs):
        super().__init__(
            in_features=existing_linear.in_features, 
            out_features=existing_linear.out_features)
        self.load_state_dict(existing_linear.state_dict())
        LoRALayer.__init__(self, r=outer_rank, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                           merge_weights=merge_weights)
        self.fan_in_fan_out = fan_in_fan_out
        
        # Actual trainable parameters
        in_features = existing_linear.in_features
        out_features = existing_linear.out_features

        self.outer_rank = math.floor(in_features * rank_ratio)
        self.inner_max_rank = inner_max_rank
        self.inner_min_rank = inner_min_rank
        self.if_prune = if_prune
        self.pruned = [False] * inner_max_rank  # 用于跟踪被剪枝的rank
        
        if self.outer_rank > 0:
            self.lora_A = nn.Parameter(self.weight.new_zeros((self.outer_rank, in_features)))
            self.lora_B = nn.Parameter(self.weight.new_zeros((out_features, self.outer_rank)))

            # 需要设置初始较大的r
            inner_max_rank = inner_max_rank if inner_max_rank < self.outer_rank else math.floor(self.outer_rank / 2)
            assert inner_max_rank > inner_min_rank, "inner_rank must be larger than min_rank"
            self.lora_RA = nn.Parameter(torch.empty(inner_max_rank, self.outer_rank))
            self.lora_RB = nn.Parameter(torch.empty(self.outer_rank, inner_max_rank))

            # 动态调整rank
            self.nd_lora_RA = dynamic_v2(maximum_rank=inner_max_rank)
            self.nd_lora_RB = dynamic_v2(maximum_rank=inner_max_rank)

            self.scaling = self.lora_alpha / self.outer_rank

            init_module_weights(self.lora_RA, sigma=1e-5)
            init_module_weights(self.lora_RB, sigma=1e-5)
            # Freezing the pre-trained weight matrix
            self.lora_A.requires_grad = False
            self.lora_B.requires_grad = False
            self.weight.requires_grad = False
        self.init_lora_param(existing_linear)
        if fan_in_fan_out:
            self.weight.data = self.weight.data.T

    def init_lora_param(self, existing_linear: nn.Linear):
        
        weight = existing_linear.weight.data
        # Tensor转换np
        U, S, Vh = torch.linalg.svd(weight)
        a = Vh[: self.outer_rank, :].to(weight.device)
        b = U[:, : self.outer_rank].to(weight.device)
        S = torch.diag(S)
        S_selected = S[: self.outer_rank, : self.outer_rank]

        if hasattr(self, 'lora_A'):
            self.lora_A.data.copy_(a)
            self.lora_B.data.copy_(b)

    def get_dimension(self):
        assert self.nd_lora_RA.get_dimension() == self.nd_lora_RB.get_dimension()
        return self.nd_lora_RA.get_dimension()

    def get_rank(self):
        assert self.nd_lora_RA.get_rank() == self.nd_lora_RB.get_rank()
        return self.nd_lora_RA.get_rank()

    def compute_lora_saliency_score(self):
        '''
        暂时不可用
        计算内层lora的基于梯度的重要性评分
        '''
        saliency_scores = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                saliency_score = (param.grad * param).abs().sum()
                saliency_scores[name] = saliency_score
        return saliency_scores

    def compute_frobenius_scores(self):
        '''
        计算内层lora的基于Frobenius范数重要性评分
        '''
        # self.lora_RA: (inner_rank, self.outer_rank)
        # self.lora_RB: (self.outer_rank, inner_rank)
        scores = [torch.norm(self.lora_RB[:, i] @ self.lora_RA[i, :]).item() if not self.pruned[i] else 0 for i in range(self.inner_max_rank)]
        total_norm = sum(scores)
        return [score / total_norm for score in scores] if total_norm != 0 else [0 for _ in scores]
    
    def prune(self):

        importance_scores = self.compute_frobenius_scores()
        self.importance_scores = importance_scores
        
        # 排除已被剪枝的rank，选择未剪枝rank中最不重要的进行剪枝
        unpruned_indices = [i for i in range(len(importance_scores)) if not self.pruned[i]]
        if len(unpruned_indices) > self.inner_min_rank:
            self.if_prune = True
            sorted_indices = sorted(unpruned_indices, key=lambda i: importance_scores[i])
            prune_idx = sorted_indices[0]
            with torch.no_grad():
                self.lora_RA[prune_idx, :] = torch.zeros_like(self.lora_RA[prune_idx, :], requires_grad=False)
                self.lora_RB[:, prune_idx] = torch.zeros_like(self.lora_RB[:, prune_idx], requires_grad=False)
                self.pruned[prune_idx] = True  # 标记为已剪枝
        else:
            self.if_prune = False

    def set_rank(self, frozen=False):
        
        self.inner_unpruned_rank = sum(not _ for _ in self.pruned)  # 内层lora未剪枝的rank长度
        min_rank = self.inner_min_rank     # 内层lora目标最小rank
        self.nd_lora_RA.set_dimension(self.inner_unpruned_rank)
        self.nd_lora_RB.set_dimension(self.inner_unpruned_rank)
        rank = random.randint(min_rank, self.inner_unpruned_rank+1)
        self.nd_lora_RA.set_rank(rank, frozen=frozen, prune=self.pruned)
        self.nd_lora_RB.set_rank(rank, frozen=frozen, prune=self.pruned)

    def merge_BA(self, mode=True):
        def T(w):
            return w.T if self.fan_in_fan_out else w
        if self.outer_rank > 0:
            lora_RA = self.nd_lora_RA(self.lora_RA.T, mode=mode).T
            lora_RB = self.nd_lora_RB(self.lora_RB, mode=mode)           
            return T(self.lora_B @ lora_RB @ lora_RA @ self.lora_A).view(self.weight.shape)
        else:
            return 0

    def train(self, mode: bool = True):
        nn.Linear.train(self, mode)
        if self.merge_weights and self.merged:
            # Make sure that the weights are not merged
            if self.outer_rank > 0:
                merged_weight_lora = self.merge_BA()
                self.weight.data -= merged_weight_lora * self.scaling
            self.merged = False
    
    def eval(self):
        nn.Linear.eval(self)
        if self.merge_weights and not self.merged:
            # Merge the weights and mark it
            if self.outer_rank > 0:
                merged_weight_lora = self.merge_BA()
                self.weight.data += merged_weight_lora * self.scaling
            self.merged = True

    def forward(self, x: torch.Tensor):
        def T(w):
            return w.T if self.fan_in_fan_out else w
        if self.outer_rank > 0 and not self.merged:
            result = F.linear(x, T(self.weight), bias=self.bias)
            if self.outer_rank > 0:
                weight_BA = self.merge_BA()
                result += (self.lora_dropout(x) @ weight_BA) * self.scaling
            return result
        else:
            return F.linear(x, T(self.weight), bias=self.bias)    


def calculate_lod_score(model: nn.Module, input: torch.Tensor, M: int=5):
    '''
    计算lora的基于lod的重要性评分
    '''
    lod_values = []
    for layer in model.layers:
        W = layer.weight  # 获取层的权重矩阵
        X = inputs  # 假设输入已通过该层的输入矩阵
        A = torch.norm(X, dim=1, p=2).unsqueeze(1) * torch.abs(W)  # 计算异常值分数
        A_mean = A.mean()
        lod = (A > M * A_mean).float().mean().item()  # 计算LOD
        lod_values.append(lod)
        inputs = layer(inputs)  # 计算下一层的输入
    return lod_values


class DoraLinear(nn.Linear, LoRALayer):
    # Lora implemented in a dense layer
    def __init__(
        self,
        existing_linear: nn.Linear,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        fan_in_fan_out: bool = False,  # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        merge_weights: bool = True,
        Wdecompose: bool = False,
        dora_simple: bool = True,
        **kwargs,
    ):
        in_features = existing_linear.in_features
        out_features = existing_linear.out_features
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        self.load_state_dict(existing_linear.state_dict())
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout, merge_weights=merge_weights)

        self.dora_weight_m_wdecomp = nn.Linear(1, out_features, bias=False) # self.weight_m_wdecomp.weight # shape: out_features, 1

        self.fan_in_fan_out = fan_in_fan_out
        self.Wdecompose = Wdecompose # whether to tune only the magnitude component of Wdecompose or not
        self.dora_simple = dora_simple # whether to use dora simple to save up GPU memory
        if self.Wdecompose == False:
            if r > 0:
                self.lora_A = nn.Linear(in_features, r, bias=False)
                self.lora_B = nn.Linear(r, out_features, bias=False)
                self.scaling = self.lora_alpha / self.r
                # Freezing the pre-trained weight matrix

        self.weight.requires_grad = False
        self.init_lora_parameters()
        if fan_in_fan_out:
            self.weight.data = self.weight.data.T

    def init_lora_parameters(self):
        with torch.no_grad():
            self.dora_weight_m_wdecomp.weight.copy_((torch.linalg.norm(self.weight.detach(),dim=1)).unsqueeze(1).detach())
        if hasattr(self, "lora_A"):
            # initialize A the same way as the default for nn.Linear and B to zero
            nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B.weight)

    def train(self, mode: bool = True):
        def transpose(w, fan_in_fan_out):
            return w.T if fan_in_fan_out else w
        
        nn.Linear.train(self, mode)
        if self.Wdecompose == False:
            self.lora_A.train(mode)
            self.lora_B.train(mode)
        self.dora_weight_m_wdecomp.train(mode)

        if mode and self.merge_weights and self.merged:
            # Unmerge the weights and mark it
            if self.Wdecompose:
                norm_scale = (self.dora_weight_m_wdecomp.weight / (torch.linalg.norm(self.weight,dim=1)).unsqueeze(1))
                weight = norm_scale * self.weight
                self.weight.data.copy_(weight.detach())
            else:
                if self.r > 0:
                    new_weight_v = self.weight - transpose(self.lora_B.weight @ self.lora_A.weight, fan_in_fan_out=self.fan_in_fan_out) * self.scaling
                    weight = (self.dora_weight_m_wdecomp.weight / (torch.linalg.norm(new_weight_v,dim=1)).unsqueeze(1)) * new_weight_v
                    self.weight.data.copy_(weight.detach())
            self.merged = False
        elif not mode and self.merge_weights and not self.merged:
            # Merge the weights and mark it
            if self.Wdecompose:
                norm_scale = (self.dora_weight_m_wdecomp.weight / (torch.linalg.norm(self.weight,dim=1)).unsqueeze(1))
                weight = norm_scale * self.weight
                self.weight.data.copy_(weight.detach())
            else:
                if self.r > 0:
                    new_weight_v = self.weight + transpose(self.lora_B.weight @ self.lora_A.weight, fan_in_fan_out=self.fan_in_fan_out) * self.scaling
                    weight = (self.dora_weight_m_wdecomp.weight / (torch.linalg.norm(new_weight_v,dim=1)).unsqueeze(1)) * new_weight_v
                    self.weight.data.copy_(weight.detach())
            self.merged = True


    def forward(self, x: torch.Tensor):
        def transpose(w, fan_in_fan_out):
            return w.T if fan_in_fan_out else w

        previous_dtype = self.weight.dtype
        
        if self.Wdecompose and not self.merged:


            norm_scale = self.dora_weight_m_wdecomp.weight.view(-1) / (torch.linalg.norm(self.weight,dim=1))

            org_result = (F.linear(x, transpose(self.weight, self.fan_in_fan_out)))

            result = org_result + (norm_scale-1) * (F.linear(self.lora_dropout(x), transpose(self.weight, self.fan_in_fan_out)))

            if not self.bias is None:
                    result += self.bias.view(1, -1).expand_as(result)

        elif self.r > 0 and not self.merged:
            
            new_weight_v = self.weight + (self.lora_B.weight @ self.lora_A.weight) * self.scaling

            if self.dora_simple:
                norm_scale = self.dora_weight_m_wdecomp.weight.view(-1) / (torch.linalg.norm(new_weight_v,dim=1)).detach()
            else:
                norm_scale = self.dora_weight_m_wdecomp.weight.view(-1) / (torch.linalg.norm(new_weight_v,dim=1))

            org_result = (F.linear(x, transpose(self.weight, self.fan_in_fan_out)))

            dropout_x = self.lora_dropout(x)

            result = org_result + (norm_scale-1) * (F.linear(dropout_x, transpose(self.weight, self.fan_in_fan_out)))

            if not self.bias is None:
                    result += self.bias.view(1, -1).expand_as(result)

            result += ( norm_scale * (self.lora_B(self.lora_A(dropout_x.to(self.lora_A.weight.dtype))))) * self.scaling
            
        else:
             result = F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)

        if result.dtype != previous_dtype:
            result = result.to(previous_dtype)

        return result

class DoraLinear_v2(nn.Linear, LoRALayer):
    '''
    DoraLinear_v2类结合了以下几种技术：

    1. DoRA (Decomposed Rank Adaptation):
       - 将权重矩阵分解为方向和幅度两个分量
       - 只更新幅度分量，保持方向不变
       - 通过self.weight_m_wdecomp实现幅度的调整

    2. VERA (Vector-based Random Matrix Adaptation):
       - 引入vera_lambda_b和vera_lambda_d两个可学习的向量参数
       - 用于动态调整LoRA矩阵A和B的影响
       - 通过element-wise乘法实现更细粒度的适应

    '''
    # Lora implemented in a dense layer
    def __init__(
        self,
        existing_linear: nn.Linear,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        fan_in_fan_out: bool = False,  # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        merge_weights: bool = True,
        Wdecompose: bool = False,
        dora_simple: bool = True,
        **kwargs,
    ):
        in_features = existing_linear.in_features
        out_features = existing_linear.out_features
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        self.load_state_dict(existing_linear.state_dict())
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout, merge_weights=merge_weights)

        self.dora_weight_m_wdecomp = nn.Linear(1, out_features, bias=False) # self.weight_m_wdecomp.weight # shape: out_features, 1

        self.fan_in_fan_out = fan_in_fan_out
        self.Wdecompose = Wdecompose # whether to tune only the magnitude component of Wdecompose or not
        self.dora_simple = dora_simple # whether to use dora simple to save up GPU memory
        if self.Wdecompose == False:
            if r > 0:
                self.lora_A = nn.Linear(in_features, r, bias=False)
                self.lora_B = nn.Linear(r, out_features, bias=False)
                self.scaling = self.lora_alpha / self.r
                # Freezing the pre-trained weight matrix
                self.lora_A.weight.requires_grad = False
                self.lora_B.weight.requires_grad = False

        self.weight.requires_grad = False
        self.init_lora_parameters()
        if fan_in_fan_out:
            self.weight.data = self.weight.data.T

        # Initialize vera_lambda_b and vera_lambda_d
        self.vera_lambda_b = nn.Parameter(torch.ones(out_features), requires_grad=True)
        self.vera_lambda_d = nn.Parameter(torch.randn(r), requires_grad=True)

    def init_lora_parameters(self):
        with torch.no_grad():
            self.dora_weight_m_wdecomp.weight.copy_((torch.linalg.norm(self.weight.detach(),dim=1)).unsqueeze(1).detach())
        if hasattr(self, "lora_A"):
            # 使用Kaiming均匀初始化方法初始化lora_A和lora_B的权重
            nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.lora_B.weight, a=math.sqrt(5))

    def train(self, mode: bool = True):        
        nn.Linear.train(self, mode)
        if self.Wdecompose == False:
            self.lora_A.train(mode)
            self.lora_B.train(mode)
        self.dora_weight_m_wdecomp.train(mode)

        if mode and self.merge_weights and self.merged:
            # Unmerge the weights and mark it
            if self.Wdecompose:
                norm_scale = (self.dora_weight_m_wdecomp.weight / (torch.linalg.norm(self.weight,dim=1)).unsqueeze(1))
                weight = norm_scale * self.weight
                self.weight.data.copy_(weight.detach())
            else:
                if self.r > 0:
                    # 使用Vera的更新公式
                    vera_lambda_b = self.vera_lambda_b.unsqueeze(-1) # shape: out_features, 1
                    vera_lambda_d = self.vera_lambda_d.unsqueeze(-1) # shape: r, 1

                    new_weight_v = self.weight - ((vera_lambda_b * self.lora_B.weight) @ (vera_lambda_d * self.lora_A.weight)) * self.scaling
                    weight = (self.dora_weight_m_wdecomp.weight / (torch.linalg.norm(new_weight_v,dim=1)).unsqueeze(1)) * new_weight_v
                    self.weight.data.copy_(weight.detach())
            self.merged = False
        elif not mode and self.merge_weights and not self.merged:
            # Merge the weights and mark it
            if self.Wdecompose:
                norm_scale = (self.dora_weight_m_wdecomp.weight / (torch.linalg.norm(self.weight,dim=1)).unsqueeze(1))
                weight = norm_scale * self.weight
                self.weight.data.copy_(weight.detach())
            else:
                if self.r > 0:
                    # 使用Vera的更新公式
                    vera_lambda_b = self.vera_lambda_b.unsqueeze(-1) # shape: out_features, 1
                    vera_lambda_d = self.vera_lambda_d.unsqueeze(-1) # shape: r, 1

                    new_weight_v = self.weight + ((vera_lambda_b * self.lora_B.weight) @ (vera_lambda_d * self.lora_A.weight)) * self.scaling
                    weight = (self.dora_weight_m_wdecomp.weight / (torch.linalg.norm(new_weight_v,dim=1)).unsqueeze(1)) * new_weight_v
                    self.weight.data.copy_(weight.detach())
            self.merged = True

    def forward(self, x: torch.Tensor):
        def transpose(w, fan_in_fan_out):
            return w.T if fan_in_fan_out else w

        previous_dtype = self.weight.dtype
        
        if self.Wdecompose and not self.merged:
            norm_scale = self.dora_weight_m_wdecomp.weight.view(-1) / (torch.linalg.norm(self.weight,dim=1))
            org_result = (F.linear(x, transpose(self.weight, self.fan_in_fan_out)))
            result = org_result + (norm_scale-1) * (F.linear(self.lora_dropout(x), transpose(self.weight, self.fan_in_fan_out)))
            if not self.bias is None:
                result += self.bias.view(1, -1).expand_as(result)

        elif self.r > 0 and not self.merged:

            new_weight_v = self.weight + ((self.vera_lambda_b.unsqueeze(1) * self.lora_B.weight) @ (self.vera_lambda_d.unsqueeze(1) * self.lora_A.weight)) * self.scaling
            if self.dora_simple:
                norm_scale = self.dora_weight_m_wdecomp.weight.view(-1) / (torch.linalg.norm(new_weight_v,dim=1)).detach()
            else:
                norm_scale = self.dora_weight_m_wdecomp.weight.view(-1) / (torch.linalg.norm(new_weight_v,dim=1))
            org_result = (F.linear(x, transpose(self.weight, self.fan_in_fan_out)))
            dropout_x = self.lora_dropout(x)
            result = org_result + (norm_scale-1) * (F.linear(dropout_x, transpose(self.weight, self.fan_in_fan_out)))
            if not self.bias is None:
                result += self.bias.view(1, -1).expand_as(result)
            
            dropout_x = dropout_x.to(self.lora_A.weight.dtype)
            dropout_x = self.lora_A(dropout_x)
            dropout_x = self.vera_lambda_d.unsqueeze(0).unsqueeze(0) * dropout_x
            dropout_x = self.lora_B(dropout_x)
            dropout_x = self.vera_lambda_b.unsqueeze(0).unsqueeze(0) * dropout_x
            result += (norm_scale * dropout_x) * self.scaling
        else:
            result = F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)

        if result.dtype != previous_dtype:
            result = result.to(previous_dtype)

        return result

class DoraLinear_v3(nn.Linear, LoRALayer):
    '''
    DoraLinear_v3类结合了以下三种技术:
    
    1. DoRA (Decomposed Rank Adaptation):
       - 将权重矩阵分解为方向和幅度两个分量
       - 只更新幅度分量,保持方向不变
       - 通过self.weight_m_wdecomp实现幅度的调整
    
    2. VERA (Vector-based Random Matrix Adaptation):
       - 引入vera_lambda_b和vera_lambda_d两个可学习的向量参数
       - 用于动态调整LoRA矩阵A和B的影响
       - 通过element-wise乘法实现更细粒度的适应
    
    3. rsLoRA (Rank Stabilized LoRA):
       - 调整LoRA适配器的缩放因子，使其与秩的平方根成反比
       - 保证不同秩下的学习稳定性
       - 避免高秩情况下的梯度崩塌问题
    
    4. SVD初始化LoRA_A和LoRA_B
    
    这种组合设计旨在实现更灵活、稳定和高效的参数高效微调。
    '''
    def __init__(
        self,
        existing_linear: nn.Linear,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        fan_in_fan_out: bool = False,  # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        merge_weights: bool = True,
        Wdecompose: bool = False,
        dora_simple: bool = True,
        **kwargs,
    ):
        in_features = existing_linear.in_features
        out_features = existing_linear.out_features
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        self.load_state_dict(existing_linear.state_dict())
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout, merge_weights=merge_weights)

        self.dora_weight_m_wdecomp = nn.Linear(1, out_features, bias=False) # self.weight_m_wdecomp.weight # shape: out_features, 1

        self.fan_in_fan_out = fan_in_fan_out
        self.Wdecompose = Wdecompose # whether to tune only the magnitude component of Wdecompose or not
        self.dora_simple = dora_simple # whether to use dora simple to save up GPU memory
        if self.Wdecompose == False:
            if r > 0:
                self.lora_A = nn.Linear(in_features, r, bias=False)
                self.lora_B = nn.Linear(r, out_features, bias=False)
                self.scaling = self.lora_alpha / math.sqrt(self.r)  # rsLoRA: 将缩放因子改为与秩的平方根成反比
                # Freezing the pre-trained weight matrix
                self.lora_A.weight.requires_grad = False
                self.lora_B.weight.requires_grad = False

        self.weight.requires_grad = False
        self.init_lora_parameters()
        if fan_in_fan_out:
            self.weight.data = self.weight.data.T

        # Initialize vera_lambda_b and vera_lambda_d
        self.vera_lambda_b = nn.Parameter(torch.ones(out_features), requires_grad=True)
        self.vera_lambda_d = nn.Parameter(torch.randn(r), requires_grad=True)

    def init_lora_parameters(self):
        with torch.no_grad():
            self.dora_weight_m_wdecomp.weight.copy_((torch.linalg.norm(self.weight.detach(),dim=1)).unsqueeze(1).detach())
        if hasattr(self, "lora_A"):
            # 使用SVD初始化初始化lora_A/B
            U, S, Vt = torch.linalg.svd(self.weight.data, full_matrices=False)
            U_r = U[:, :self.r]
            S_r = S[:self.r]
            Vt_r = Vt[:self.r, :]

            self.lora_A.weight.data.copy_(torch.diag(S_r) @ U_r.T)
            self.lora_B.weight.data.copy_(Vt_r.T)

    def train(self, mode: bool = True):
        def transpose(w, fan_in_fan_out):
            return w.T if fan_in_fan_out else w
        
        nn.Linear.train(self, mode)
        if self.Wdecompose == False:
            self.lora_A.train(mode)
            self.lora_B.train(mode)
        self.dora_weight_m_wdecomp.train(mode)

        if mode and self.merge_weights and self.merged:
            # Unmerge the weights and mark it
            if self.Wdecompose:
                norm_scale = (self.dora_weight_m_wdecomp.weight / (torch.linalg.norm(self.weight,dim=1)).unsqueeze(1))
                weight = norm_scale * self.weight
                self.weight.data.copy_(weight.detach())
            else:
                if self.r > 0:
                    # 使用Vera的更新公式
                    vera_lambda_b = self.vera_lambda_b.unsqueeze(-1) # shape: out_features, 1
                    vera_lambda_d = self.vera_lambda_d.unsqueeze(-1) # shape: r, 1

                    new_weight_v = self.weight - ((vera_lambda_b * self.lora_B.weight) @ (vera_lambda_d * self.lora_A.weight)) * self.scaling
                    weight = (self.dora_weight_m_wdecomp.weight / (torch.linalg.norm(new_weight_v,dim=1)).unsqueeze(1)) * new_weight_v
                    self.weight.data.copy_(weight.detach())
            self.merged = False
        elif not mode and self.merge_weights and not self.merged:
            # Merge the weights and mark it
            if self.Wdecompose:
                norm_scale = (self.dora_weight_m_wdecomp.weight / (torch.linalg.norm(self.weight,dim=1)).unsqueeze(1))
                weight = norm_scale * self.weight
                self.weight.data.copy_(weight.detach())
            else:
                if self.r > 0:
                    # 使用Vera的更新公式
                    vera_lambda_b = self.vera_lambda_b.unsqueeze(-1) # shape: out_features, 1
                    vera_lambda_d = self.vera_lambda_d.unsqueeze(-1) # shape: r, 1

                    new_weight_v = self.weight + ((vera_lambda_b * self.lora_B.weight) @ (vera_lambda_d * self.lora_A.weight)) * self.scaling
                    weight = (self.dora_weight_m_wdecomp.weight / (torch.linalg.norm(new_weight_v,dim=1)).unsqueeze(1)) * new_weight_v
                    self.weight.data.copy_(weight.detach())
            self.merged = True

    def forward(self, x: torch.Tensor):
        def transpose(w, fan_in_fan_out):
            return w.T if fan_in_fan_out else w

        previous_dtype = self.weight.dtype
        
        if self.Wdecompose and not self.merged:
            norm_scale = self.dora_weight_m_wdecomp.weight.view(-1) / (torch.linalg.norm(self.weight,dim=1))
            org_result = (F.linear(x, transpose(self.weight, self.fan_in_fan_out)))
            result = org_result + (norm_scale-1) * (F.linear(self.lora_dropout(x), transpose(self.weight, self.fan_in_fan_out)))
            if not self.bias is None:
                result += self.bias.view(1, -1).expand_as(result)

        elif self.r > 0 and not self.merged:

            new_weight_v = self.weight + ((self.vera_lambda_b.unsqueeze(1) * self.lora_B.weight) @ (self.vera_lambda_d.unsqueeze(1) * self.lora_A.weight)) * self.scaling
            if self.dora_simple:
                norm_scale = self.dora_weight_m_wdecomp.weight.view(-1) / (torch.linalg.norm(new_weight_v,dim=1)).detach()
            else:
                norm_scale = self.dora_weight_m_wdecomp.weight.view(-1) / (torch.linalg.norm(new_weight_v,dim=1))
            org_result = (F.linear(x, transpose(self.weight, self.fan_in_fan_out)))
            dropout_x = self.lora_dropout(x)
            result = org_result + (norm_scale-1) * (F.linear(dropout_x, transpose(self.weight, self.fan_in_fan_out)))
            if not self.bias is None:
                result += self.bias.view(1, -1).expand_as(result)
            
            dropout_x = dropout_x.to(self.lora_A.weight.dtype)
            dropout_x = self.lora_A(dropout_x)
            dropout_x = self.vera_lambda_d.unsqueeze(0).unsqueeze(0) * dropout_x
            dropout_x = self.lora_B(dropout_x)
            dropout_x = self.vera_lambda_b.unsqueeze(0).unsqueeze(0) * dropout_x
            result += (norm_scale * dropout_x) * self.scaling
        else:
            result = F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)

        if result.dtype != previous_dtype:
            result = result.to(previous_dtype)

        return result

class DoraLinear_v4(nn.Linear, LoRALayer):
    """
    DoraLinear_v4 是一个结合了DoRA、LoRA-XS和VERA三种方法的线性层实现。
    
    - DoRA (Decomposed Rank Adaptation): 通过分解权重矩阵的范数和方向来实现更有效的参数适应。
    - LoRA-XS: 使用SVD初始化低秩矩阵A和B，并引入可训练的R矩阵来增强表达能力。
    - VERA (VERsatile Adaptation): 引入可训练的缩放因子vera_lambda_b、vera_lambda_d和vera_lambda_xs，分别用于调整LoRA-XS中B、A和R矩阵的输出。
    
    """
    def __init__(
        self,
        existing_linear: nn.Linear,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        fan_in_fan_out: bool = False,
        merge_weights: bool = True,
        Wdecompose: bool = False,
        dora_simple: bool = True,
        r_init_method: str = 'eye',  # 新增参数，用于选择lora_R的初始化方法
        **kwargs,
    ):
        in_features = existing_linear.in_features
        out_features = existing_linear.out_features
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        self.load_state_dict(existing_linear.state_dict())
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout, merge_weights=merge_weights)

        self.dora_weight_m_wdecomp = nn.Linear(1, out_features, bias=False)

        self.fan_in_fan_out = fan_in_fan_out
        self.Wdecompose = Wdecompose
        self.dora_simple = dora_simple
        self.r_init_method = r_init_method  # 保存初始化方法
        
        if self.Wdecompose == False and r > 0:
            # LoRA-XS: 使用SVD初始化A和B,并引入可训练的R矩阵
            self.lora_A = nn.Linear(in_features, r, bias=False)
            self.lora_B = nn.Linear(r, out_features, bias=False)
            self.lora_R = nn.Linear(r, r, bias=False)
            self.scaling = self.lora_alpha / r

        self.weight.requires_grad = False
        self.lora_A.weight.requires_grad = False
        self.lora_B.weight.requires_grad = False
        self.lora_R.weight.requires_grad = False

        # VERA: 初始化vera_lambda_b和vera_lambda_d
        # vera_lambda_b在lora_B之后应用,用于调整lora_B的输出
        # vera_lambda_d在lora_A之后应用,用于调整lora_A的输出
        self.vera_lambda_b = nn.Parameter(torch.ones(out_features), requires_grad=True)
        self.vera_lambda_d = nn.Parameter(torch.randn(r), requires_grad=True)
        self.vera_lambda_xs = nn.Parameter(torch.randn(r), requires_grad=True)

        self.init_lora_parameters()
        if fan_in_fan_out:
            self.weight.data = self.weight.data.T

    def init_lora_parameters(self):
        with torch.no_grad():
            self.dora_weight_m_wdecomp.weight.copy_((torch.linalg.norm(self.weight.detach(),dim=1)).unsqueeze(1).detach())

            if hasattr(self, "lora_A"):
                # LoRA-XS: 使用SVD初始化A和B
                U, S, Vt = torch.linalg.svd(self.weight.data, full_matrices=False)
                self.lora_A.weight.data.copy_((U[:, :self.r] @ torch.diag(S[:self.r])).T)
                self.lora_B.weight.data.copy_(Vt[:self.r, :].T)
                
                # 初始化lora_R
                if self.r_init_method == 'kaiming':
                    nn.init.kaiming_uniform_(self.lora_R.weight, a=math.sqrt(5))
                elif self.r_init_method == 'eye':
                    self.lora_R.weight.data.copy_(torch.eye(self.r))
                else:
                    raise ValueError(f"不支持的初始化方法: {self.r_init_method}")

    def train(self, mode: bool = True):        
        nn.Linear.train(self, mode)
        if self.Wdecompose == False:
            self.lora_A.train(mode)
            self.lora_B.train(mode)
        self.dora_weight_m_wdecomp.train(mode)

        if mode and self.merge_weights and self.merged:
            # Unmerge the weights and mark it
            if self.Wdecompose:
                norm_scale = (self.dora_weight_m_wdecomp.weight / (torch.linalg.norm(self.weight,dim=1)).unsqueeze(1))
                weight = norm_scale * self.weight
                self.weight.data.copy_(weight.detach())
            else:
                if self.r > 0:
                    # 使用Vera的更新公式
                    vera_lambda_b = self.vera_lambda_b.unsqueeze(-1) # shape: out_features, 1
                    vera_lambda_d = self.vera_lambda_d.unsqueeze(-1) # shape: r, 1
                    vera_lambda_xs = self.vera_lambda_xs.unsqueeze(-1) # shape: r, 1

                    new_weight_v = self.weight - ((vera_lambda_b * self.lora_B.weight) @ (vera_lambda_xs * self.lora_R.weight) @ (vera_lambda_d * self.lora_A.weight)) * self.scaling
                    weight = (self.dora_weight_m_wdecomp.weight / (torch.linalg.norm(new_weight_v,dim=1)).unsqueeze(1)) * new_weight_v
                    self.weight.data.copy_(weight.detach())
            self.merged = False
        elif not mode and self.merge_weights and not self.merged:
            # Merge the weights and mark it
            if self.Wdecompose:
                norm_scale = (self.dora_weight_m_wdecomp.weight / (torch.linalg.norm(self.weight,dim=1)).unsqueeze(1))
                weight = norm_scale * self.weight
                self.weight.data.copy_(weight.detach())
            else:
                if self.r > 0:
                    # 使用Vera的更新公式
                    vera_lambda_b = self.vera_lambda_b.unsqueeze(-1) # shape: out_features, 1
                    vera_lambda_d = self.vera_lambda_d.unsqueeze(-1) # shape: r, 1
                    vera_lambda_xs = self.vera_lambda_xs.unsqueeze(-1) # shape: r, 1

                    new_weight_v = self.weight + ((vera_lambda_b * self.lora_B.weight) @ (vera_lambda_xs * self.lora_R.weight) @ (vera_lambda_d * self.lora_A.weight)) * self.scaling
                    weight = (self.dora_weight_m_wdecomp.weight / (torch.linalg.norm(new_weight_v,dim=1)).unsqueeze(1)) * new_weight_v
                    self.weight.data.copy_(weight.detach())
            self.merged = True

    def forward(self, x: torch.Tensor):
        def transpose(w, fan_in_fan_out):
            return w.T if fan_in_fan_out else w

        previous_dtype = self.weight.dtype
        
        if self.Wdecompose and not self.merged:
            norm_scale = self.dora_weight_m_wdecomp.weight.view(-1) / (torch.linalg.norm(self.weight,dim=1))
            org_result = (F.linear(x, transpose(self.weight, self.fan_in_fan_out)))
            result = org_result + (norm_scale-1) * (F.linear(self.lora_dropout(x), transpose(self.weight, self.fan_in_fan_out)))
            if not self.bias is None:
                result += self.bias.view(1, -1).expand_as(result)

        elif self.r > 0 and not self.merged:

            new_weight_v = self.weight + ((self.vera_lambda_b.unsqueeze(1) * self.lora_B.weight) @ (self.vera_lambda_xs.unsqueeze(1) * self.lora_R.weight) @ (self.vera_lambda_d.unsqueeze(1) * self.lora_A.weight)) * self.scaling
            if self.dora_simple:
                norm_scale = self.dora_weight_m_wdecomp.weight.view(-1) / (torch.linalg.norm(new_weight_v,dim=1)).detach()
            else:
                norm_scale = self.dora_weight_m_wdecomp.weight.view(-1) / (torch.linalg.norm(new_weight_v,dim=1))
            org_result = (F.linear(x, transpose(self.weight, self.fan_in_fan_out)))
            dropout_x = self.lora_dropout(x)
            result = org_result + (norm_scale-1) * (F.linear(dropout_x, transpose(self.weight, self.fan_in_fan_out)))
            if not self.bias is None:
                result += self.bias.view(1, -1).expand_as(result)
            
            dropout_x = dropout_x.to(self.lora_A.weight.dtype)
            dropout_x = self.lora_A(dropout_x)
            dropout_x = self.vera_lambda_d.unsqueeze(0).unsqueeze(0) * dropout_x
            dropout_x = self.lora_R(dropout_x)
            dropout_x = self.vera_lambda_xs.unsqueeze(0).unsqueeze(0) * dropout_x
            dropout_x = self.lora_B(dropout_x)
            dropout_x = self.vera_lambda_b.unsqueeze(0).unsqueeze(0) * dropout_x
            result += (norm_scale * dropout_x) * self.scaling
        else:
            result = F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)

        if result.dtype != previous_dtype:
            result = result.to(previous_dtype)

        return result
    

class DoraLinear_v5(nn.Linear, LoRALayer):
    """
    DoraLinear_v5 是一个结合了DoRA、LoRA-XS和VERA三种方法的线性层实现。
    
    - DoRA (Decomposed Rank Adaptation): 通过分解权重矩阵的范数和方向来实现更有效的参数适应。
    - LoRA-XS: 使用SVD初始化低秩矩阵A和B，并引入可训练的R矩阵来增强表达能力。
    - VERA (VERsatile Adaptation): 引入可训练的缩放因子vera_lambda_b、vera_lambda_d和vera_lambda_xs，分别用于调整LoRA-XS中B、A和R矩阵的输出。
    
    """
    def __init__(
        self,
        existing_linear: nn.Linear,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        fan_in_fan_out: bool = False,
        merge_weights: bool = True,
        Wdecompose: bool = False,
        dora_simple: bool = True,
        r_init_method: str = 'eye',  # 新增参数，用于选择lora_R的初始化方法
        **kwargs,
    ):
        in_features = existing_linear.in_features
        out_features = existing_linear.out_features
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        self.load_state_dict(existing_linear.state_dict())
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout, merge_weights=merge_weights)

        self.dora_weight_m_wdecomp = nn.Linear(1, out_features, bias=False)

        self.fan_in_fan_out = fan_in_fan_out
        self.Wdecompose = Wdecompose
        self.dora_simple = dora_simple
        self.r_init_method = r_init_method  # 保存初始化方法
        
        if self.Wdecompose == False and r > 0:
            # LoRA-XS: 使用SVD初始化A和B,并引入可训练的R矩阵
            self.lora_A = nn.Linear(in_features, r, bias=False)
            self.lora_B = nn.Linear(r, out_features, bias=False)
            self.lora_R = nn.Linear(r, r, bias=False)
            self.scaling = self.lora_alpha / math.sqrt(r) # rslora 方法

        self.weight.requires_grad = False
        self.lora_A.weight.requires_grad = False
        self.lora_B.weight.requires_grad = False
        self.lora_R.weight.requires_grad = False

        # VERA: 初始化vera_lambda_b和vera_lambda_d
        # vera_lambda_b在lora_B之后应用,用于调整lora_B的输出
        # vera_lambda_d在lora_A之后应用,用于调整lora_A的输出
        self.vera_lambda_b = nn.Parameter(torch.ones(out_features), requires_grad=True)
        self.vera_lambda_d = nn.Parameter(torch.randn(r), requires_grad=True)
        self.vera_lambda_xs = nn.Parameter(torch.randn(r), requires_grad=True)

        self.init_lora_parameters()
        if fan_in_fan_out:
            self.weight.data = self.weight.data.T

    def init_lora_parameters(self):
        with torch.no_grad():
            self.dora_weight_m_wdecomp.weight.copy_((torch.linalg.norm(self.weight.detach(),dim=1)).unsqueeze(1).detach())

            if hasattr(self, "lora_A"):
                # LoRA-XS: 使用SVD初始化A和B
                U, S, Vt = torch.linalg.svd(self.weight.data, full_matrices=False)
                self.lora_A.weight.data.copy_((U[:, :self.r] @ torch.diag(S[:self.r])).T)
                self.lora_B.weight.data.copy_(Vt[:self.r, :].T)
                
                # 初始化lora_R
                if self.r_init_method == 'kaiming':
                    nn.init.kaiming_uniform_(self.lora_R.weight, a=math.sqrt(5))
                elif self.r_init_method == 'eye':
                    self.lora_R.weight.data.copy_(torch.eye(self.r))
                else:
                    raise ValueError(f"不支持的初始化方法: {self.r_init_method}")

    def train(self, mode: bool = True):        
        nn.Linear.train(self, mode)
        if self.Wdecompose == False:
            self.lora_A.train(mode)
            self.lora_B.train(mode)
        self.dora_weight_m_wdecomp.train(mode)

        if mode and self.merge_weights and self.merged:
            # Unmerge the weights and mark it
            if self.Wdecompose:
                norm_scale = (self.dora_weight_m_wdecomp.weight / (torch.linalg.norm(self.weight,dim=1)).unsqueeze(1))
                weight = norm_scale * self.weight
                self.weight.data.copy_(weight.detach())
            else:
                if self.r > 0:
                    # 使用Vera的更新公式
                    vera_lambda_b = self.vera_lambda_b.unsqueeze(-1) # shape: out_features, 1
                    vera_lambda_d = self.vera_lambda_d.unsqueeze(-1) # shape: r, 1
                    vera_lambda_xs = self.vera_lambda_xs.unsqueeze(-1) # shape: r, 1

                    new_weight_v = self.weight - ((vera_lambda_b * self.lora_B.weight) @ (vera_lambda_xs * self.lora_R.weight) @ (vera_lambda_d * self.lora_A.weight)) * self.scaling
                    weight = (self.dora_weight_m_wdecomp.weight / (torch.linalg.norm(new_weight_v,dim=1)).unsqueeze(1)) * new_weight_v
                    self.weight.data.copy_(weight.detach())
            self.merged = False
        elif not mode and self.merge_weights and not self.merged:
            # Merge the weights and mark it
            if self.Wdecompose:
                norm_scale = (self.dora_weight_m_wdecomp.weight / (torch.linalg.norm(self.weight,dim=1)).unsqueeze(1))
                weight = norm_scale * self.weight
                self.weight.data.copy_(weight.detach())
            else:
                if self.r > 0:
                    # 使用Vera的更新公式
                    vera_lambda_b = self.vera_lambda_b.unsqueeze(-1) # shape: out_features, 1
                    vera_lambda_d = self.vera_lambda_d.unsqueeze(-1) # shape: r, 1
                    vera_lambda_xs = self.vera_lambda_xs.unsqueeze(-1) # shape: r, 1

                    new_weight_v = self.weight + ((vera_lambda_b * self.lora_B.weight) @ (vera_lambda_xs * self.lora_R.weight) @ (vera_lambda_d * self.lora_A.weight)) * self.scaling
                    weight = (self.dora_weight_m_wdecomp.weight / (torch.linalg.norm(new_weight_v,dim=1)).unsqueeze(1)) * new_weight_v
                    self.weight.data.copy_(weight.detach())
            self.merged = True

    def forward(self, x: torch.Tensor):
        def transpose(w, fan_in_fan_out):
            return w.T if fan_in_fan_out else w

        previous_dtype = self.weight.dtype
        
        if self.Wdecompose and not self.merged:
            norm_scale = self.dora_weight_m_wdecomp.weight.view(-1) / (torch.linalg.norm(self.weight,dim=1))
            org_result = (F.linear(x, transpose(self.weight, self.fan_in_fan_out)))
            result = org_result + (norm_scale-1) * (F.linear(self.lora_dropout(x), transpose(self.weight, self.fan_in_fan_out)))
            if not self.bias is None:
                result += self.bias.view(1, -1).expand_as(result)

        elif self.r > 0 and not self.merged:

            new_weight_v = self.weight + ((self.vera_lambda_b.unsqueeze(1) * self.lora_B.weight) @ (self.vera_lambda_xs.unsqueeze(1) * self.lora_R.weight) @ (self.vera_lambda_d.unsqueeze(1) * self.lora_A.weight)) * self.scaling
            if self.dora_simple:
                norm_scale = self.dora_weight_m_wdecomp.weight.view(-1) / (torch.linalg.norm(new_weight_v,dim=1)).detach()
            else:
                norm_scale = self.dora_weight_m_wdecomp.weight.view(-1) / (torch.linalg.norm(new_weight_v,dim=1))
            org_result = (F.linear(x, transpose(self.weight, self.fan_in_fan_out)))
            dropout_x = self.lora_dropout(x)
            result = org_result + (norm_scale-1) * (F.linear(dropout_x, transpose(self.weight, self.fan_in_fan_out)))
            if not self.bias is None:
                result += self.bias.view(1, -1).expand_as(result)
            
            dropout_x = dropout_x.to(self.lora_A.weight.dtype)
            dropout_x = self.lora_A(dropout_x)
            dropout_x = self.vera_lambda_d.unsqueeze(0).unsqueeze(0) * dropout_x
            dropout_x = self.lora_R(dropout_x)
            dropout_x = self.vera_lambda_xs.unsqueeze(0).unsqueeze(0) * dropout_x
            dropout_x = self.lora_B(dropout_x)
            dropout_x = self.vera_lambda_b.unsqueeze(0).unsqueeze(0) * dropout_x
            result += (norm_scale * dropout_x) * self.scaling
        else:
            result = F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)

        if result.dtype != previous_dtype:
            result = result.to(previous_dtype)

        return result
 
class DoraLinear_v7(nn.Linear, LoRALayer):
    '''

    DoraLinear_v7类结合了以下三种技术:
    
    1. DoRA (Decomposed Rank Adaptation):
       - 将权重矩阵分解为方向和幅度两个分量
       - 只更新幅度分量,保持方向不变
       - 通过self.weight_m_wdecomp实现幅度的调整
    
    2. VERA (Vector-based Random Matrix Adaptation):
       - 引入vera_lambda_b和vera_lambda_d两个可学习的向量参数
       - 用于动态调整LoRA矩阵A和B的影响
       - 通过element-wise乘法实现更细粒度的适应
    
    3. 通过ASVD的加权方法SVD初始化LoRA_A和LoRA_B
    
    这种组合设计旨在实现更灵活、稳定和高效的参数高效微调。
    '''
    def __init__(
        self,
        existing_linear: nn.Linear,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        fan_in_fan_out: bool = False,  # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        merge_weights: bool = True,
        Wdecompose: bool = False,
        dora_simple: bool = True,
        NoRA_loss: bool = True,
        **kwargs,
    ):
        in_features = existing_linear.in_features
        out_features = existing_linear.out_features
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        self.load_state_dict(existing_linear.state_dict())
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout, merge_weights=merge_weights)

        self.dora_weight_m_wdecomp = nn.Linear(1, out_features, bias=False) # self.weight_m_wdecomp.weight # shape: out_features, 1

        self.fan_in_fan_out = fan_in_fan_out
        self.Wdecompose = Wdecompose # whether to tune only the magnitude component of Wdecompose or not
        self.dora_simple = dora_simple # whether to use dora simple to save up GPU memory  
        if self.Wdecompose == False:
            if r > 0:
                self.lora_A = nn.Linear(in_features, r, bias=False) # lora_A weight shape: 
                self.lora_B = nn.Linear(r, out_features, bias=False)
                self.scaling = self.lora_alpha / self.r
                # Freezing the pre-trained weight matrix
                self.lora_A.weight.requires_grad = False
                self.lora_B.weight.requires_grad = False

        self.weight.requires_grad = False
        self.init_lora_parameters()
        if fan_in_fan_out:
            self.weight.data = self.weight.data.T

        # Initialize vera_lambda_b and vera_lambda_d
        self.vera_lambda_b = nn.Parameter(torch.ones(out_features), requires_grad=True)
        self.vera_lambda_d = nn.Parameter(torch.randn(r), requires_grad=True)

    def init_lora_parameters(self):
        with torch.no_grad():
            self.dora_weight_m_wdecomp.weight.copy_((torch.linalg.norm(self.weight.detach(),dim=1)).unsqueeze(1).detach())

    def init_lora_AB_parameters(self, W_prime, S_diag):
        
        U, S, Vh = torch.linalg.svd(W_prime, full_matrices=False)
        U = U[:, :self.r]
        S = S[:self.r]
        Vh = Vh[:self.r , :]
        weight_A = U @ torch.diag(S)
        weight_B = Vh @ torch.inverse(S_diag)

        weight_A = weight_A.T
        weight_B = weight_B.T

        with torch.no_grad():
            self.lora_A.weight.copy_(weight_A)
            self.lora_B.weight.copy_(weight_B)

    def train(self, mode: bool = True):
        def transpose(w, fan_in_fan_out):
            return w.T if fan_in_fan_out else w
        
        nn.Linear.train(self, mode)
        if self.Wdecompose == False:
            self.lora_A.train(mode)
            self.lora_B.train(mode)
        self.dora_weight_m_wdecomp.train(mode)

        if mode and self.merge_weights and self.merged:
            # Unmerge the weights and mark it
            if self.Wdecompose:
                norm_scale = (self.dora_weight_m_wdecomp.weight / (torch.linalg.norm(self.weight,dim=1)).unsqueeze(1))
                weight = norm_scale * self.weight
                self.weight.data.copy_(weight.detach())
            else:
                if self.r > 0:
                    # 使用Vera的更新公式
                    vera_lambda_b = self.vera_lambda_b.unsqueeze(-1) # shape: out_features, 1
                    vera_lambda_d = self.vera_lambda_d.unsqueeze(-1) # shape: r, 1

                    new_weight_v = self.weight - ((vera_lambda_b * self.lora_B.weight) @ (vera_lambda_d * self.lora_A.weight)) * self.scaling
                    weight = (self.dora_weight_m_wdecomp.weight / (torch.linalg.norm(new_weight_v,dim=1)).unsqueeze(1)) * new_weight_v
                    self.weight.data.copy_(weight.detach())
            self.merged = False
        elif not mode and self.merge_weights and not self.merged:
            # Merge the weights and mark it
            if self.Wdecompose:
                norm_scale = (self.dora_weight_m_wdecomp.weight / (torch.linalg.norm(self.weight,dim=1)).unsqueeze(1))
                weight = norm_scale * self.weight
                self.weight.data.copy_(weight.detach())
            else:
                if self.r > 0:
                    # 使用Vera的更新公式
                    vera_lambda_b = self.vera_lambda_b.unsqueeze(-1) # shape: out_features, 1
                    vera_lambda_d = self.vera_lambda_d.unsqueeze(-1) # shape: r, 1

                    new_weight_v = self.weight + ((vera_lambda_b * self.lora_B.weight) @ (vera_lambda_d * self.lora_A.weight)) * self.scaling
                    weight = (self.dora_weight_m_wdecomp.weight / (torch.linalg.norm(new_weight_v,dim=1)).unsqueeze(1)) * new_weight_v
                    self.weight.data.copy_(weight.detach())
            self.merged = True

    def forward(self, x: torch.Tensor):
        def transpose(w, fan_in_fan_out):
            return w.T if fan_in_fan_out else w

        previous_dtype = self.weight.dtype
        
        if self.Wdecompose and not self.merged:
            norm_scale = self.dora_weight_m_wdecomp.weight.view(-1) / (torch.linalg.norm(self.weight,dim=1))
            org_result = (F.linear(x, transpose(self.weight, self.fan_in_fan_out)))
            result = org_result + (norm_scale-1) * (F.linear(self.lora_dropout(x), transpose(self.weight, self.fan_in_fan_out)))
            if not self.bias is None:
                result += self.bias.view(1, -1).expand_as(result)

        elif self.r > 0 and not self.merged:

            new_weight_v = self.weight + ((self.vera_lambda_b.unsqueeze(1) * self.lora_B.weight) @ (self.vera_lambda_d.unsqueeze(1) * self.lora_A.weight)) * self.scaling
            if self.dora_simple:
                norm_scale = self.dora_weight_m_wdecomp.weight.view(-1) / (torch.linalg.norm(new_weight_v,dim=1)).detach()
            else:
                norm_scale = self.dora_weight_m_wdecomp.weight.view(-1) / (torch.linalg.norm(new_weight_v,dim=1))
            org_result = (F.linear(x, transpose(self.weight, self.fan_in_fan_out)))
            dropout_x = self.lora_dropout(x)
            result = org_result + (norm_scale-1) * (F.linear(dropout_x, transpose(self.weight, self.fan_in_fan_out)))
            if not self.bias is None:
                result += self.bias.view(1, -1).expand_as(result)
            
            dropout_x = dropout_x.to(self.lora_A.weight.dtype)
            dropout_x = self.lora_A(dropout_x)
            dropout_x = self.vera_lambda_d.unsqueeze(0).unsqueeze(0) * dropout_x
            dropout_x = self.lora_B(dropout_x)
            dropout_x = self.vera_lambda_b.unsqueeze(0).unsqueeze(0) * dropout_x
            result += (norm_scale * dropout_x) * self.scaling
        else:
            result = F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)

        if result.dtype != previous_dtype:
            result = result.to(previous_dtype)

        return result

class DoraLinear_v7_2(nn.Linear, LoRALayer):
    '''

    DoraLinear_v7类结合了以下三种技术
    ------
    
    1. DoRA (Decomposed Rank Adaptation):
       - 将权重矩阵分解为方向和幅度两个分量
       - 只更新幅度分量,保持方向不变
       - 通过`self.weight_m_wdecomp`实现幅度的调整
    
    2. VERA (Vector-based Random Matrix Adaptation):
       - 引入`vera_lambda_b`和`vera_lambda_d`两个可学习的向量参数
       - 用于动态调整LoRA矩阵A和B的影响
       - 通过element-wise乘法实现更细粒度的适应
    
    3. 通过ASVD的加权方法SVD初始化LoRA_A和LoRA_B
    
    这种组合设计旨在实现更灵活、稳定和高效的参数高效微调。
    '''
    def __init__(
        self,
        existing_linear: nn.Linear,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        fan_in_fan_out: bool = False,  # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        merge_weights: bool = True,
        Wdecompose: bool = False,
        dora_simple: bool = True,
        NoRA_loss: bool = True,
        **kwargs,
    ):
        in_features = existing_linear.in_features
        out_features = existing_linear.out_features
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        self.load_state_dict(existing_linear.state_dict())
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout, merge_weights=merge_weights)

        self.dora_weight_m_wdecomp = nn.Linear(1, out_features, bias=False) # self.weight_m_wdecomp.weight # shape: out_features, 1

        self.fan_in_fan_out = fan_in_fan_out
        self.Wdecompose = Wdecompose # whether to tune only the magnitude component of Wdecompose or not
        self.dora_simple = dora_simple # whether to use dora simple to save up GPU memory
        self.NoRA_loss = NoRA_loss #是否使用正则化loss
        if self.Wdecompose == False:
            if r > 0:
                self.lora_A = nn.Linear(in_features, r, bias=False) # lora_A weight shape: 
                self.lora_B = nn.Linear(r, out_features, bias=False)
                self.scaling = self.lora_alpha / self.r
                # Freezing the pre-trained weight matrix
                self.lora_A.weight.requires_grad = False
                self.lora_B.weight.requires_grad = False

        self.weight.requires_grad = False
        self.init_lora_parameters()
        if fan_in_fan_out:
            self.weight.data = self.weight.data.T

        # Initialize vera_lambda_b and vera_lambda_d
        self.vera_lambda_b = nn.Parameter(torch.ones(out_features), requires_grad=True)
        self.vera_lambda_d = nn.Parameter(torch.randn(r), requires_grad=True)

    def init_lora_parameters(self):
        with torch.no_grad():
            self.dora_weight_m_wdecomp.weight.copy_((torch.linalg.norm(self.weight.detach(),dim=1)).unsqueeze(1).detach())

    def init_lora_AB_parameters(self, W_prime, S_diag):
        
        U, S, Vh = torch.linalg.svd(W_prime, full_matrices=False)
        U = U[:, :self.r]
        S = S[:self.r]
        Vh = Vh[:self.r , :]
        weight_A = U @ torch.diag(S)
        weight_B = Vh @ torch.inverse(S_diag)

        weight_A = weight_A.T
        weight_B = weight_B.T

        with torch.no_grad():
            self.lora_A.weight.copy_(weight_A)
            self.lora_B.weight.copy_(weight_B)

    def train(self, mode: bool = True):
        def transpose(w, fan_in_fan_out):
            return w.T if fan_in_fan_out else w
        
        nn.Linear.train(self, mode)
        if self.Wdecompose == False:
            self.lora_A.train(mode)
            self.lora_B.train(mode)
        self.dora_weight_m_wdecomp.train(mode)
        self.NoRA_loss = mode

        if mode and self.merge_weights and self.merged:
            # Unmerge the weights and mark it
            if self.Wdecompose:
                norm_scale = (self.dora_weight_m_wdecomp.weight / (torch.linalg.norm(self.weight,dim=1)).unsqueeze(1))
                weight = norm_scale * self.weight
                self.weight.data.copy_(weight.detach())
            else:
                if self.r > 0:
                    # 使用Vera的更新公式
                    vera_lambda_b = self.vera_lambda_b.unsqueeze(-1) # shape: out_features, 1
                    vera_lambda_d = self.vera_lambda_d.unsqueeze(-1) # shape: r, 1

                    new_weight_v = self.weight - ((vera_lambda_b * self.lora_B.weight) @ (vera_lambda_d * self.lora_A.weight)) * self.scaling
                    weight = (self.dora_weight_m_wdecomp.weight / (torch.linalg.norm(new_weight_v,dim=1)).unsqueeze(1)) * new_weight_v
                    self.weight.data.copy_(weight.detach())
            self.merged = False
        elif not mode and self.merge_weights and not self.merged:
            # Merge the weights and mark it
            if self.Wdecompose:
                norm_scale = (self.dora_weight_m_wdecomp.weight / (torch.linalg.norm(self.weight,dim=1)).unsqueeze(1))
                weight = norm_scale * self.weight
                self.weight.data.copy_(weight.detach())
            else:
                if self.r > 0:
                    # 使用Vera的更新公式
                    vera_lambda_b = self.vera_lambda_b.unsqueeze(-1) # shape: out_features, 1
                    vera_lambda_d = self.vera_lambda_d.unsqueeze(-1) # shape: r, 1

                    new_weight_v = self.weight + ((vera_lambda_b * self.lora_B.weight) @ (vera_lambda_d * self.lora_A.weight)) * self.scaling
                    weight = (self.dora_weight_m_wdecomp.weight / (torch.linalg.norm(new_weight_v,dim=1)).unsqueeze(1)) * new_weight_v
                    self.weight.data.copy_(weight.detach())
            self.merged = True

    def regularize_loss(self):
        # 计算lora_a.weight、vera_d、lora_b.weight相乘后与weight之间的mse损失
        vera_lambda_b = self.vera_lambda_b.unsqueeze(-1)
        vera_lambda_d = self.vera_lambda_d.unsqueeze(-1)

        lora_product = ((vera_lambda_b * self.lora_B.weight) @ (vera_lambda_d * self.lora_A.weight)) * self.scaling
        if self.merged:
            weight = self.weight - (((vera_lambda_b * self.lora_B.weight) @ (vera_lambda_d * self.lora_A.weight)) * self.scaling)
        else:
            weight = self.weight
        
        mse_loss = F.mse_loss(lora_product, weight)
        return mse_loss

    def forward(self, x: torch.Tensor):
        def transpose(w, fan_in_fan_out):
            return w.T if fan_in_fan_out else w

        previous_dtype = self.weight.dtype
        
        if self.Wdecompose and not self.merged:
            norm_scale = self.dora_weight_m_wdecomp.weight.view(-1) / (torch.linalg.norm(self.weight,dim=1))
            org_result = (F.linear(x, transpose(self.weight, self.fan_in_fan_out)))
            result = org_result + (norm_scale-1) * (F.linear(self.lora_dropout(x), transpose(self.weight, self.fan_in_fan_out)))
            if not self.bias is None:
                result += self.bias.view(1, -1).expand_as(result)

        elif self.r > 0 and not self.merged:

            new_weight_v = self.weight + ((self.vera_lambda_b.unsqueeze(1) * self.lora_B.weight) @ (self.vera_lambda_d.unsqueeze(1) * self.lora_A.weight)) * self.scaling
            if self.dora_simple:
                norm_scale = self.dora_weight_m_wdecomp.weight.view(-1) / (torch.linalg.norm(new_weight_v,dim=1)).detach()
            else:
                norm_scale = self.dora_weight_m_wdecomp.weight.view(-1) / (torch.linalg.norm(new_weight_v,dim=1))
            org_result = (F.linear(x, transpose(self.weight, self.fan_in_fan_out)))
            dropout_x = self.lora_dropout(x)
            result = org_result + (norm_scale-1) * (F.linear(dropout_x, transpose(self.weight, self.fan_in_fan_out)))
            if not self.bias is None:
                result += self.bias.view(1, -1).expand_as(result)
            
            dropout_x = dropout_x.to(self.lora_A.weight.dtype)
            dropout_x = self.lora_A(dropout_x)
            dropout_x = self.vera_lambda_d.unsqueeze(0).unsqueeze(0) * dropout_x
            dropout_x = self.lora_B(dropout_x)
            dropout_x = self.vera_lambda_b.unsqueeze(0).unsqueeze(0) * dropout_x
            result += (norm_scale * dropout_x) * self.scaling
        else:
            result = F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)

        if result.dtype != previous_dtype:
            result = result.to(previous_dtype)
        
        if self.NoRA_loss:
            mse_loss = self.regularize_loss()
            return result, mse_loss

        else:
            return result, None

class DoraLinear_v7_3(nn.Linear, LoRALayer):
    '''

    DoraLinear_v7类结合了以下三种技术
    ------
    
    1. DoRA (Decomposed Rank Adaptation):
       - 将权重矩阵分解为方向和幅度两个分量
       - 只更新幅度分量,保持方向不变
       - 通过`self.weight_m_wdecomp`实现幅度的调整
    
    2. VERA (Vector-based Random Matrix Adaptation):
       - 引入`vera_lambda_b`和`vera_lambda_d`两个可学习的向量参数
       - 用于动态调整LoRA矩阵A和B的影响
       - 通过element-wise乘法实现更细粒度的适应
    
    3. 通过ASVD的加权方法SVD初始化LoRA_A和LoRA_B

    4. 使用SVD分解后的S矩阵的平方根作为权重

    5. 计算lora分支和weight分支的mse损失
    
    这种组合设计旨在实现更灵活、稳定和高效的参数高效微调。
    '''
    def __init__(
        self,
        existing_linear: nn.Linear,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        fan_in_fan_out: bool = False,  # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        merge_weights: bool = True,
        Wdecompose: bool = False,
        dora_simple: bool = True,
        NoRA_loss: bool = True,
        **kwargs,
    ):
        in_features = existing_linear.in_features
        out_features = existing_linear.out_features
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        self.load_state_dict(existing_linear.state_dict())
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout, merge_weights=merge_weights)

        self.dora_weight_m_wdecomp = nn.Linear(1, out_features, bias=False) # self.weight_m_wdecomp.weight # shape: out_features, 1

        self.fan_in_fan_out = fan_in_fan_out
        self.Wdecompose = Wdecompose # whether to tune only the magnitude component of Wdecompose or not
        self.dora_simple = dora_simple # whether to use dora simple to save up GPU memory
        self.NoRA_loss = NoRA_loss #是否使用正则化loss
        if self.Wdecompose == False:
            if r > 0:
                self.lora_A = nn.Linear(in_features, r, bias=False) # lora_A weight shape: 
                self.lora_B = nn.Linear(r, out_features, bias=False)
                self.scaling = self.lora_alpha / self.r
                # Freezing the pre-trained weight matrix
                self.lora_A.weight.requires_grad = False
                self.lora_B.weight.requires_grad = False

        self.weight.requires_grad = False
        self.init_lora_parameters()
        if fan_in_fan_out:
            self.weight.data = self.weight.data.T

        # Initialize vera_lambda_b and vera_lambda_d
        self.vera_lambda_b = nn.Parameter(torch.ones(out_features), requires_grad=True)
        self.vera_lambda_d = nn.Parameter(torch.randn(r), requires_grad=True)

    def init_lora_parameters(self):
        with torch.no_grad():
            self.dora_weight_m_wdecomp.weight.copy_((torch.linalg.norm(self.weight.detach(),dim=1)).unsqueeze(1).detach())

    def init_lora_AB_parameters(self, W_prime, S_diag):
        
        U, S, Vh = torch.linalg.svd(W_prime, full_matrices=False)
        U = U[:, :self.r]
        S = torch.sqrt(S[:self.r])
        Vh = Vh[:self.r , :]
        weight_A = U @ torch.diag(S)
        weight_B = torch.diag(S) @ Vh @ torch.inverse(S_diag)

        weight_A = weight_A.T
        weight_B = weight_B.T

        with torch.no_grad():
            self.lora_A.weight.copy_(weight_A)
            self.lora_B.weight.copy_(weight_B)

    def train(self, mode: bool = True):
        def transpose(w, fan_in_fan_out):
            return w.T if fan_in_fan_out else w
        
        nn.Linear.train(self, mode)
        if self.Wdecompose == False:
            self.lora_A.train(mode)
            self.lora_B.train(mode)
        self.dora_weight_m_wdecomp.train(mode)
        self.NoRA_loss = mode

        if mode and self.merge_weights and self.merged:
            # Unmerge the weights and mark it
            if self.Wdecompose:
                norm_scale = (self.dora_weight_m_wdecomp.weight / (torch.linalg.norm(self.weight,dim=1)).unsqueeze(1))
                weight = norm_scale * self.weight
                self.weight.data.copy_(weight.detach())
            else:
                if self.r > 0:
                    # 使用Vera的更新公式
                    vera_lambda_b = self.vera_lambda_b.unsqueeze(-1) # shape: out_features, 1
                    vera_lambda_d = self.vera_lambda_d.unsqueeze(-1) # shape: r, 1

                    new_weight_v = self.weight - ((vera_lambda_b * self.lora_B.weight) @ (vera_lambda_d * self.lora_A.weight)) * self.scaling
                    weight = (self.dora_weight_m_wdecomp.weight / (torch.linalg.norm(new_weight_v,dim=1)).unsqueeze(1)) * new_weight_v
                    self.weight.data.copy_(weight.detach())
            self.merged = False
        elif not mode and self.merge_weights and not self.merged:
            # Merge the weights and mark it
            if self.Wdecompose:
                norm_scale = (self.dora_weight_m_wdecomp.weight / (torch.linalg.norm(self.weight,dim=1)).unsqueeze(1))
                weight = norm_scale * self.weight
                self.weight.data.copy_(weight.detach())
            else:
                if self.r > 0:
                    # 使用Vera的更新公式
                    vera_lambda_b = self.vera_lambda_b.unsqueeze(-1) # shape: out_features, 1
                    vera_lambda_d = self.vera_lambda_d.unsqueeze(-1) # shape: r, 1

                    new_weight_v = self.weight + ((vera_lambda_b * self.lora_B.weight) @ (vera_lambda_d * self.lora_A.weight)) * self.scaling
                    weight = (self.dora_weight_m_wdecomp.weight / (torch.linalg.norm(new_weight_v,dim=1)).unsqueeze(1)) * new_weight_v
                    self.weight.data.copy_(weight.detach())
            self.merged = True

    def regularize_loss(self):
        # 计算lora_a.weight、vera_d、lora_b.weight相乘后与weight之间的mse损失
        vera_lambda_b = self.vera_lambda_b.unsqueeze(-1)
        vera_lambda_d = self.vera_lambda_d.unsqueeze(-1)

        lora_product = ((vera_lambda_b * self.lora_B.weight) @ (vera_lambda_d * self.lora_A.weight)) * self.scaling
        if self.merged:
            weight = self.weight - (((vera_lambda_b * self.lora_B.weight) @ (vera_lambda_d * self.lora_A.weight)) * self.scaling)
        else:
            weight = self.weight
        
        mse_loss = F.mse_loss(lora_product, weight)
        return mse_loss

    def forward(self, x: torch.Tensor):
        def transpose(w, fan_in_fan_out):
            return w.T if fan_in_fan_out else w

        previous_dtype = self.weight.dtype
        
        if self.Wdecompose and not self.merged:
            norm_scale = self.dora_weight_m_wdecomp.weight.view(-1) / (torch.linalg.norm(self.weight,dim=1))
            org_result = (F.linear(x, transpose(self.weight, self.fan_in_fan_out)))
            result = org_result + (norm_scale-1) * (F.linear(self.lora_dropout(x), transpose(self.weight, self.fan_in_fan_out)))
            if not self.bias is None:
                result += self.bias.view(1, -1).expand_as(result)

        elif self.r > 0 and not self.merged:

            new_weight_v = self.weight + ((self.vera_lambda_b.unsqueeze(1) * self.lora_B.weight) @ (self.vera_lambda_d.unsqueeze(1) * self.lora_A.weight)) * self.scaling
            if self.dora_simple:
                norm_scale = self.dora_weight_m_wdecomp.weight.view(-1) / (torch.linalg.norm(new_weight_v,dim=1)).detach()
            else:
                norm_scale = self.dora_weight_m_wdecomp.weight.view(-1) / (torch.linalg.norm(new_weight_v,dim=1))
            org_result = (F.linear(x, transpose(self.weight, self.fan_in_fan_out)))
            dropout_x = self.lora_dropout(x)
            result = org_result + (norm_scale-1) * (F.linear(dropout_x, transpose(self.weight, self.fan_in_fan_out)))
            if not self.bias is None:
                result += self.bias.view(1, -1).expand_as(result)
            
            dropout_x = dropout_x.to(self.lora_A.weight.dtype)
            dropout_x = self.lora_A(dropout_x)
            dropout_x = self.vera_lambda_d.unsqueeze(0).unsqueeze(0) * dropout_x
            dropout_x = self.lora_B(dropout_x)
            dropout_x = self.vera_lambda_b.unsqueeze(0).unsqueeze(0) * dropout_x
            result += (norm_scale * dropout_x) * self.scaling
        else:
            result = F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)

        if result.dtype != previous_dtype:
            result = result.to(previous_dtype)
        
        if self.NoRA_loss:
            mse_loss = self.regularize_loss()
            return result, mse_loss

        else:
            return result, None

class DoraLinear_v8(nn.Linear, LoRALayer):
    '''

    DoraLinear_v8类结合了以下三种技术:
    
    1. DoRA (Decomposed Rank Adaptation):
       - 将权重矩阵分解为方向和幅度两个分量
       - 只更新幅度分量,保持方向不变
       - 通过self.weight_m_wdecomp实现幅度的调整
    
    2. VERA (Vector-based Random Matrix Adaptation):
       - 引入vera_lambda_b和vera_lambda_d两个可学习的向量参数
       - 用于动态调整LoRA矩阵A和B的影响
       - 通过element-wise乘法实现更细粒度的适应
    
    3. 通过ASVD的加权方法SVD初始化LoRA_A和LoRA_B

    4. 加入side_r, 加入并行的lora_side ，作为残差项加入lora_a进行计算
    
    这种组合设计旨在实现更灵活、稳定和高效的参数高效微调。
    '''
    def __init__(
        self,
        existing_linear: nn.Linear,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        fan_in_fan_out: bool = False,  # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        merge_weights: bool = True,
        Wdecompose: bool = False,
        dora_simple: bool = True,
        side_r: int = 1000,
        **kwargs,
    ):
        assert side_r < r, "side_r must be less than r"
        in_features = existing_linear.in_features
        out_features = existing_linear.out_features
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        self.load_state_dict(existing_linear.state_dict())
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout, merge_weights=merge_weights)

        self.dora_weight_m_wdecomp = nn.Linear(1, out_features, bias=False) # self.weight_m_wdecomp.weight # shape: out_features, 1

        self.fan_in_fan_out = fan_in_fan_out
        self.Wdecompose = Wdecompose # whether to tune only the magnitude component of Wdecompose or not
        self.dora_simple = dora_simple # whether to use dora simple to save up GPU memory  
        if self.Wdecompose == False:
            if r > 0:
                self.lora_A = nn.Linear(in_features, r, bias=False) # lora_A weight shape: 
                self.lora_B = nn.Linear(r, out_features, bias=False)
                self.scaling = self.lora_alpha / self.r
                # Freezing the pre-trained weight matrix
                self.lora_A.weight.requires_grad = False
                self.lora_B.weight.requires_grad = False

        self.weight.requires_grad = False
        if fan_in_fan_out:
            self.weight.data = self.weight.data.T

        # Initialize vera_lambda_b
        self.vera_lambda_b = nn.Parameter(torch.ones(out_features), requires_grad=True)
        # Initialize lora-side
        self.side_r = side_r
        self.side_a = nn.Parameter(torch.randn(in_features, side_r), requires_grad=True)
        self.side_b = nn.Parameter(torch.randn(side_r, r), requires_grad=True)

        self.init_lora_parameters()
        
    def init_lora_parameters(self):
        with torch.no_grad():
            self.dora_weight_m_wdecomp.weight.copy_((torch.linalg.norm(self.weight.detach(),dim=1)).unsqueeze(1).detach())
            torch.nn.init.kaiming_uniform_(self.side_a, a=math.sqrt(5))
            torch.nn.init.kaiming_uniform_(self.side_b, a=math.sqrt(5))

    def init_lora_AB_parameters(self, W_prime, S_diag):
        
        U, S, Vh = torch.linalg.svd(W_prime, full_matrices=False)
        U = U[:, :self.r]
        S = S[:self.r]
        Vh = Vh[:self.r , :]
        weight_A = U @ torch.diag(S)
        weight_B = Vh @ torch.inverse(S_diag)

        weight_A = weight_A.T
        weight_B = weight_B.T

        with torch.no_grad():
            self.lora_A.weight.copy_(weight_A)
            self.lora_B.weight.copy_(weight_B)

    def get_side_ab_weight(self):
        return self.side_a @ self.side_b

    def train(self, mode: bool = True):
        def transpose(w, fan_in_fan_out):
            return w.T if fan_in_fan_out else w
        
        nn.Linear.train(self, mode)
        if self.Wdecompose == False:
            self.lora_A.train(mode)
            self.lora_B.train(mode)
        self.dora_weight_m_wdecomp.train(mode)

        if mode and self.merge_weights and self.merged:
            # Unmerge the weights and mark it
            if self.Wdecompose:
                norm_scale = (self.dora_weight_m_wdecomp.weight / (torch.linalg.norm(self.weight,dim=1)).unsqueeze(1))
                weight = norm_scale * self.weight
                self.weight.data.copy_(weight.detach())
            else:
                if self.r > 0:
                    # 使用Vera的更新公式
                    vera_lambda_b = self.vera_lambda_b.unsqueeze(-1) # shape: out_features, 1

                    new_weight_v = self.weight - ((vera_lambda_b * self.lora_B.weight) @ (self.lora_A.weight + self.get_side_ab_weight().T)) * self.scaling
                    weight = (self.dora_weight_m_wdecomp.weight / (torch.linalg.norm(new_weight_v,dim=1)).unsqueeze(1)) * new_weight_v
                    self.weight.data.copy_(weight.detach())
            self.merged = False
        elif not mode and self.merge_weights and not self.merged:
            # Merge the weights and mark it
            if self.Wdecompose:
                norm_scale = (self.dora_weight_m_wdecomp.weight / (torch.linalg.norm(self.weight,dim=1)).unsqueeze(1))
                weight = norm_scale * self.weight
                self.weight.data.copy_(weight.detach())
            else:
                if self.r > 0:
                    # 使用Vera的更新公式
                    vera_lambda_b = self.vera_lambda_b.unsqueeze(-1) # shape: out_features, 1

                    new_weight_v = self.weight + ((vera_lambda_b * self.lora_B.weight) @ (self.lora_A.weight + self.get_side_ab_weight().T)) * self.scaling
                    weight = (self.dora_weight_m_wdecomp.weight / (torch.linalg.norm(new_weight_v,dim=1)).unsqueeze(1)) * new_weight_v
                    self.weight.data.copy_(weight.detach())
            self.merged = True

    def forward(self, x: torch.Tensor):
        def transpose(w, fan_in_fan_out):
            return w.T if fan_in_fan_out else w

        previous_dtype = self.weight.dtype
        
        if self.Wdecompose and not self.merged:
            norm_scale = self.dora_weight_m_wdecomp.weight.view(-1) / (torch.linalg.norm(self.weight,dim=1))
            org_result = (F.linear(x, transpose(self.weight, self.fan_in_fan_out)))
            result = org_result + (norm_scale-1) * (F.linear(self.lora_dropout(x), transpose(self.weight, self.fan_in_fan_out)))
            if not self.bias is None:
                result += self.bias.view(1, -1).expand_as(result)

        elif self.r > 0 and not self.merged:

            new_weight_v = self.weight + ((self.vera_lambda_b.unsqueeze(1) * self.lora_B.weight) @ (self.lora_A.weight + self.get_side_ab_weight().T)) * self.scaling
            if self.dora_simple:
                norm_scale = self.dora_weight_m_wdecomp.weight.view(-1) / (torch.linalg.norm(new_weight_v,dim=1)).detach()
            else:
                norm_scale = self.dora_weight_m_wdecomp.weight.view(-1) / (torch.linalg.norm(new_weight_v,dim=1))
            org_result = (F.linear(x, transpose(self.weight, self.fan_in_fan_out)))
            dropout_x = self.lora_dropout(x)
            result = org_result + (norm_scale-1) * (F.linear(dropout_x, transpose(self.weight, self.fan_in_fan_out)))
            if not self.bias is None:
                result += self.bias.view(1, -1).expand_as(result)
            
            dropout_x = dropout_x.to(self.lora_A.weight.dtype)
            dropout_x = self.lora_A(dropout_x) + F.linear(dropout_x, self.get_side_ab_weight().T)
            dropout_x = self.lora_B(dropout_x)
            dropout_x = self.vera_lambda_b.unsqueeze(0).unsqueeze(0) * dropout_x
            result += (norm_scale * dropout_x) * self.scaling
        else:
            result = F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)

        if result.dtype != previous_dtype:
            result = result.to(previous_dtype)

        return result

class DoraLinear_v9(nn.Linear, LoRALayer):
    '''

    DoraLinear_v7类结合了以下三种技术:
    
    1. DoRA (Decomposed Rank Adaptation):
       - 将权重矩阵分解为方向和幅度两个分量
       - 只更新幅度分量,保持方向不变
       - 通过self.weight_m_wdecomp实现幅度的调整
    
    2. VERA (Vector-based Random Matrix Adaptation):
       - 引入vera_lambda_b和vera_lambda_d两个可学习的向量参数
       - 用于动态调整LoRA矩阵A和B的影响
       - 通过element-wise乘法实现更细粒度的适应
    
    3. 通过ASVD的加权方法SVD初始化LoRA_A和LoRA_B
    
    这种组合设计旨在实现更灵活、稳定和高效的参数高效微调。
    '''
    def __init__(
        self,
        existing_linear: nn.Linear,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        fan_in_fan_out: bool = False,  # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        merge_weights: bool = True,
        Wdecompose: bool = False,
        dora_simple: bool = True,
        NoRA_loss: bool = True,
        **kwargs,
    ):
        in_features = existing_linear.in_features
        out_features = existing_linear.out_features
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        self.load_state_dict(existing_linear.state_dict())
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout, merge_weights=merge_weights)

        self.dora_weight_m_wdecomp = nn.Linear(1, out_features, bias=False) # self.weight_m_wdecomp.weight # shape: out_features, 1

        self.fan_in_fan_out = fan_in_fan_out
        self.Wdecompose = Wdecompose # whether to tune only the magnitude component of Wdecompose or not
        self.dora_simple = dora_simple # whether to use dora simple to save up GPU memory  
        if self.Wdecompose == False:
            if r > 0:
                self.lora_A = nn.Linear(in_features, r, bias=False) # lora_A weight shape: 
                self.lora_B = nn.Linear(r, out_features, bias=False)
                self.scaling = self.lora_alpha / self.r
                # Freezing the pre-trained weight matrix
                self.lora_A.weight.requires_grad = False
                self.lora_B.weight.requires_grad = False

        self.weight.requires_grad = False
        self.init_lora_parameters()
        if fan_in_fan_out:
            self.weight.data = self.weight.data.T

        # Initialize vera_lambda_b and vera_lambda_d
        self.vera_lambda_b = nn.Parameter(torch.ones(out_features), requires_grad=True)
        self.vera_lambda_d = nn.Parameter(torch.ones(r), requires_grad=True)

    def init_lora_parameters(self):
        with torch.no_grad():
            self.dora_weight_m_wdecomp.weight.copy_((torch.linalg.norm(self.weight.detach(),dim=1)).unsqueeze(1).detach())

    def init_lora_AB_parameters(self, W_prime, S_diag):
        
        U, S, Vh = torch.linalg.svd(W_prime, full_matrices=False)
        U = U[:, :self.r]
        S = S[:self.r]
        Vh = Vh[:self.r , :]
        weight_A = U @ torch.diag(S)
        weight_B = Vh @ torch.inverse(S_diag)

        weight_A = weight_A.T
        weight_B = weight_B.T

        with torch.no_grad():
            self.lora_A.weight.copy_(weight_A)
            self.lora_B.weight.copy_(weight_B)

    def orthonormal_loss(self):
        vera_lambda_b = self.vera_lambda_b.unsqueeze(-1)
        vera_lambda_d = self.vera_lambda_d.unsqueeze(-1)

        W_a = vera_lambda_d * self.lora_A.weight # shape: r, in_features
        W_b = vera_lambda_b * self.lora_B.weight # shape: out_features, r

        W_a_orthonormal = W_a @ W_a.T # shape: r, r
        W_b_orthonormal = W_b.T @ W_b # shape: r, r

        I_a = torch.eye(self.r).to(W_a_orthonormal.device)
        I_a.requires_grad = False
        I_b = torch.eye(self.r).to(W_b_orthonormal.device)
        I_b.requires_grad = False

        loss = torch.norm(W_a_orthonormal - I_a, p="fro") + torch.norm(W_b_orthonormal - I_b, p="fro")
        return loss

    def train(self, mode: bool = True):
        def transpose(w, fan_in_fan_out):
            return w.T if fan_in_fan_out else w
        
        nn.Linear.train(self, mode)
        if self.Wdecompose == False:
            self.lora_A.train(mode)
            self.lora_B.train(mode)
        self.dora_weight_m_wdecomp.train(mode)

        if mode and self.merge_weights and self.merged:
            # Unmerge the weights and mark it
            if self.Wdecompose:
                norm_scale = (self.dora_weight_m_wdecomp.weight / (torch.linalg.norm(self.weight,dim=1)).unsqueeze(1))
                weight = norm_scale * self.weight
                self.weight.data.copy_(weight.detach())
            else:
                if self.r > 0:
                    # 使用Vera的更新公式
                    vera_lambda_b = self.vera_lambda_b.unsqueeze(-1) # shape: out_features, 1
                    vera_lambda_d = self.vera_lambda_d.unsqueeze(-1) # shape: r, 1

                    new_weight_v = self.weight - ((vera_lambda_b * self.lora_B.weight) @ (vera_lambda_d * self.lora_A.weight)) * self.scaling
                    weight = (self.dora_weight_m_wdecomp.weight / (torch.linalg.norm(new_weight_v,dim=1)).unsqueeze(1)) * new_weight_v
                    self.weight.data.copy_(weight.detach())
            self.merged = False
        elif not mode and self.merge_weights and not self.merged:
            # Merge the weights and mark it
            if self.Wdecompose:
                norm_scale = (self.dora_weight_m_wdecomp.weight / (torch.linalg.norm(self.weight,dim=1)).unsqueeze(1))
                weight = norm_scale * self.weight
                self.weight.data.copy_(weight.detach())
            else:
                if self.r > 0:
                    # 使用Vera的更新公式
                    vera_lambda_b = self.vera_lambda_b.unsqueeze(-1) # shape: out_features, 1
                    vera_lambda_d = self.vera_lambda_d.unsqueeze(-1) # shape: r, 1

                    new_weight_v = self.weight + ((vera_lambda_b * self.lora_B.weight) @ (vera_lambda_d * self.lora_A.weight)) * self.scaling
                    weight = (self.dora_weight_m_wdecomp.weight / (torch.linalg.norm(new_weight_v,dim=1)).unsqueeze(1)) * new_weight_v
                    self.weight.data.copy_(weight.detach())
            self.merged = True

    def forward(self, x: torch.Tensor):
        def transpose(w, fan_in_fan_out):
            return w.T if fan_in_fan_out else w

        previous_dtype = self.weight.dtype
        
        if self.Wdecompose and not self.merged:
            norm_scale = self.dora_weight_m_wdecomp.weight.view(-1) / (torch.linalg.norm(self.weight,dim=1))
            org_result = (F.linear(x, transpose(self.weight, self.fan_in_fan_out)))
            result = org_result + (norm_scale-1) * (F.linear(self.lora_dropout(x), transpose(self.weight, self.fan_in_fan_out)))
            if not self.bias is None:
                result += self.bias.view(1, -1).expand_as(result)

        elif self.r > 0 and not self.merged:

            new_weight_v = self.weight + ((self.vera_lambda_b.unsqueeze(1) * self.lora_B.weight) @ (self.vera_lambda_d.unsqueeze(1) * self.lora_A.weight)) * self.scaling
            if self.dora_simple:
                norm_scale = self.dora_weight_m_wdecomp.weight.view(-1) / (torch.linalg.norm(new_weight_v,dim=1)).detach()
            else:
                norm_scale = self.dora_weight_m_wdecomp.weight.view(-1) / (torch.linalg.norm(new_weight_v,dim=1))
            org_result = (F.linear(x, transpose(self.weight, self.fan_in_fan_out)))
            dropout_x = self.lora_dropout(x)
            result = org_result + (norm_scale-1) * (F.linear(dropout_x, transpose(self.weight, self.fan_in_fan_out)))
            if not self.bias is None:
                result += self.bias.view(1, -1).expand_as(result)
            
            dropout_x = dropout_x.to(self.lora_A.weight.dtype)
            dropout_x = self.lora_A(dropout_x)
            dropout_x = self.vera_lambda_d.unsqueeze(0).unsqueeze(0) * dropout_x
            dropout_x = self.lora_B(dropout_x)
            dropout_x = self.vera_lambda_b.unsqueeze(0).unsqueeze(0) * dropout_x
            result += (norm_scale * dropout_x) * self.scaling
        else:
            result = F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)

        if result.dtype != previous_dtype:
            result = result.to(previous_dtype)

        return result

class NoRALinear_v1(nn.Linear, LoRALayer):
    '''

    NoRALinear_v1类结合了以下三种技术:
    
    1. DoRA (Decomposed Rank Adaptation):
       - 将权重矩阵分解为方向和幅度两个分量
       - 只更新幅度分量,保持方向不变
       - 通过self.weight_m_wdecomp实现幅度的调整
    
    2. VERA (Vector-based Random Matrix Adaptation):
       - 引入vera_lambda_b和vera_lambda_d两个可学习的向量参数
       - 用于动态调整LoRA矩阵A和B的影响
       - 通过element-wise乘法实现更细粒度的适应
    
    3. 通过ASVD的加权方法SVD初始化W_A和W_B
    
    这种组合设计旨在实现更灵活、稳定和高效的参数高效微调。
    '''
    def __init__(
        self,
        existing_linear: nn.Linear,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        fan_in_fan_out: bool = False,  # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        merge_weights: bool = True,
        Wdecompose: bool = False,
        dora_simple: bool = True,
        NoRA_loss: bool = True,
        **kwargs,
    ):
        in_features = existing_linear.in_features
        out_features = existing_linear.out_features
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        self.load_state_dict(existing_linear.state_dict())
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout, merge_weights=merge_weights)

        self.dora_weight_m_wdecomp = nn.Linear(1, out_features, bias=False) # self.weight_m_wdecomp.weight # shape: out_features, 1

        self.fan_in_fan_out = fan_in_fan_out
        self.Wdecompose = Wdecompose # whether to tune only the magnitude component of Wdecompose or not
        self.dora_simple = dora_simple # whether to use dora simple to save up GPU memory  
        if self.Wdecompose == False:
            if r > 0:
                self.lora_A = nn.Linear(in_features, r, bias=False) # lora_A weight shape: 
                self.lora_B = nn.Linear(r, out_features, bias=False)
                self.scaling = self.lora_alpha / self.r
                # Freezing the pre-trained weight matrix
                self.lora_A.weight.requires_grad = False
                self.lora_B.weight.requires_grad = False

        self.weight.requires_grad = False
        self.init_lora_parameters()
        if fan_in_fan_out:
            self.weight.data = self.weight.data.T

        # Initialize vera_lambda_b and vera_lambda_d
        self.vera_lambda_b = nn.Parameter(torch.ones(out_features), requires_grad=True)
        self.vera_lambda_d = nn.Parameter(torch.randn(r), requires_grad=True)

    def init_lora_parameters(self):
        with torch.no_grad():
            self.dora_weight_m_wdecomp.weight.copy_((torch.linalg.norm(self.weight.detach(),dim=1)).unsqueeze(1).detach())

    def init_lora_AB_parameters(self, W_prime, S_diag):
        
        U, S, Vh = torch.linalg.svd(W_prime, full_matrices=False)
        U = U[:, :self.r]
        S = S[:self.r]
        Vh = Vh[:self.r , :]
        weight_A = U @ torch.diag(S)
        weight_B = Vh @ torch.inverse(S_diag)

        weight_A = weight_A.T
        weight_B = weight_B.T

        with torch.no_grad():
            self.lora_A.weight.copy_(weight_A)
            self.lora_B.weight.copy_(weight_B)

    def train(self, mode: bool = True):
        def transpose(w, fan_in_fan_out):
            return w.T if fan_in_fan_out else w
        
        nn.Linear.train(self, mode)
        if self.Wdecompose == False:
            self.lora_A.train(mode)
            self.lora_B.train(mode)
        self.dora_weight_m_wdecomp.train(mode)

        if mode and self.merge_weights and self.merged:
            # Unmerge the weights and mark it
            if self.Wdecompose:
                norm_scale = (self.dora_weight_m_wdecomp.weight / (torch.linalg.norm(self.weight,dim=1)).unsqueeze(1))
                weight = norm_scale * self.weight
                self.weight.data.copy_(weight.detach())
            else:
                if self.r > 0:
                    # 使用Vera的更新公式
                    vera_lambda_b = self.vera_lambda_b.unsqueeze(-1) # shape: out_features, 1
                    vera_lambda_d = self.vera_lambda_d.unsqueeze(-1) # shape: r, 1

                    new_weight_v = self.weight - ((vera_lambda_b * self.lora_B.weight) @ (vera_lambda_d * self.lora_A.weight)) * self.scaling
                    weight = (self.dora_weight_m_wdecomp.weight / (torch.linalg.norm(new_weight_v,dim=1)).unsqueeze(1)) * new_weight_v
                    self.weight.data.copy_(weight.detach())
            self.merged = False
        elif not mode and self.merge_weights and not self.merged:
            # Merge the weights and mark it
            if self.Wdecompose:
                norm_scale = (self.dora_weight_m_wdecomp.weight / (torch.linalg.norm(self.weight,dim=1)).unsqueeze(1))
                weight = norm_scale * self.weight
                self.weight.data.copy_(weight.detach())
            else:
                if self.r > 0:
                    # 使用Vera的更新公式
                    vera_lambda_b = self.vera_lambda_b.unsqueeze(-1) # shape: out_features, 1
                    vera_lambda_d = self.vera_lambda_d.unsqueeze(-1) # shape: r, 1

                    new_weight_v = self.weight + ((vera_lambda_b * self.lora_B.weight) @ (vera_lambda_d * self.lora_A.weight)) * self.scaling
                    weight = (self.dora_weight_m_wdecomp.weight / (torch.linalg.norm(new_weight_v,dim=1)).unsqueeze(1)) * new_weight_v
                    self.weight.data.copy_(weight.detach())
            self.merged = True

    def forward(self, x: torch.Tensor):
        def transpose(w, fan_in_fan_out):
            return w.T if fan_in_fan_out else w

        previous_dtype = self.weight.dtype
        
        if self.Wdecompose and not self.merged:
            norm_scale = self.dora_weight_m_wdecomp.weight.view(-1) / (torch.linalg.norm(self.weight,dim=1))
            org_result = (F.linear(x, transpose(self.weight, self.fan_in_fan_out)))
            result = org_result + (norm_scale-1) * (F.linear(self.lora_dropout(x), transpose(self.weight, self.fan_in_fan_out)))
            if not self.bias is None:
                result += self.bias.view(1, -1).expand_as(result)

        elif self.r > 0 and not self.merged:

            new_weight_v = self.weight + ((self.vera_lambda_b.unsqueeze(1) * self.lora_B.weight) @ (self.vera_lambda_d.unsqueeze(1) * self.lora_A.weight)) * self.scaling
            if self.dora_simple:
                norm_scale = self.dora_weight_m_wdecomp.weight.view(-1) / (torch.linalg.norm(new_weight_v,dim=1)).detach()
            else:
                norm_scale = self.dora_weight_m_wdecomp.weight.view(-1) / (torch.linalg.norm(new_weight_v,dim=1))
            org_result = (F.linear(x, transpose(self.weight, self.fan_in_fan_out)))
            dropout_x = self.lora_dropout(x)
            result = org_result + (norm_scale-1) * (F.linear(dropout_x, transpose(self.weight, self.fan_in_fan_out)))
            if not self.bias is None:
                result += self.bias.view(1, -1).expand_as(result)
            
            dropout_x = dropout_x.to(self.lora_A.weight.dtype)
            dropout_x = self.lora_A(dropout_x)
            dropout_x = self.vera_lambda_d.unsqueeze(0).unsqueeze(0) * dropout_x
            dropout_x = self.lora_B(dropout_x)
            dropout_x = self.vera_lambda_b.unsqueeze(0).unsqueeze(0) * dropout_x
            result += (norm_scale * dropout_x) * self.scaling
        else:
            result = F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)

        if result.dtype != previous_dtype:
            result = result.to(previous_dtype)

        return result

class NoRALinear_A(nn.Linear, LoRALayer):
    def __init__(self, 
                 existing_linear: nn.Linear,
                 r: int = 0,
                 lora_alpha: float = 1.0, 
                 lora_dropout: float = 0.0, 
                 merge_weights: bool = True, 
                 fan_in_fan_out: bool = False,
                 **kwargs):
        super().__init__(
            in_features=existing_linear.in_features, 
            out_features=existing_linear.out_features)
        self.load_state_dict(existing_linear.state_dict())
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                           merge_weights=merge_weights)
        self.fan_in_fan_out = fan_in_fan_out
        
        # Actual trainable parameters
        in_features = existing_linear.in_features
        out_features = existing_linear.out_features
        if self.r > 0:
            self.lora_A = nn.Parameter(torch.randn(in_features, r), requires_grad=True)
            self.lora_B = nn.Parameter(torch.randn(r, out_features), requires_grad=True)
            self.scaling = self.lora_alpha / self.r

            # Freezing the pre-trained weight matrix
            self.lora_A.requires_grad = True
            self.lora_B.requires_grad = True
            self.weight.requires_grad = False
        if fan_in_fan_out:
            self.weight.data = self.weight.data.T

    def init_lora_param(self, W_prime, S_diag, init_method: str):
        def T(w):
            return w.T if self.fan_in_fan_out else w
        with torch.no_grad():
            weight = T(W_prime) # out, in -> in, out
            if init_method == '1':
                # kaiming初始化
                nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
                nn.init.zeros_(self.lora_B)

            elif init_method == '2':
                # svd初始化: U @ Sigma,	V
                U, S, Vh = torch.linalg.svd(weight, full_matrices=False)
                U = U[:, :self.r]
                S = S[:self.r]
                Vh = Vh[:self.r, :]
                
                self.lora_A.data.copy_(torch.inverse(S_diag.T) @ U @ torch.diag(S))
                self.lora_B.data.copy_(Vh)
            elif init_method == '3':
                # svd初始化： U, Sigma @ Vh
                U, S, Vh = torch.linalg.svd(weight, full_matrices=False)
                U = U[:, :self.r]
                S = S[:self.r]
                Vh = Vh[:self.r, :]
                if hasattr(self, 'lora_A'):
                    self.lora_A.data.copy_(torch.inverse(S_diag.T) @ U)
                    self.lora_B.data.copy_(torch.diag(S) @ Vh)
            elif init_method == '4':
                # svd初始化: U @ sqrt(Sigma), sqrt(Sigma) @ Vh
                U, S, Vh = torch.linalg.svd(weight, full_matrices=False)
                U = U[:, :self.r]
                S = torch.sqrt(S[:self.r])
                Vh = Vh[:self.r, :]
                if hasattr(self, 'lora_A'):
                    self.lora_A.data.copy_(torch.inverse(S_diag.T) @ U @ torch.diag(S))
                    self.lora_B.data.copy_(torch.diag(S) @ Vh)
            

    def merge_AB(self):
        return self.lora_A @ self.lora_B

    def train(self, mode: bool = True):
        nn.Linear.train(self, mode)
        if self.merge_weights and self.merged:
            # Make sure that the weights are not merged
            if self.r > 0:
                merged_weight_lora = self.merge_AB()
                self.weight.data -= merged_weight_lora * self.scaling
            self.merged = False
    
    def eval(self):
        nn.Linear.eval(self)
        if self.merge_weights and not self.merged:
            # Merge the weights and mark it
            if self.r > 0:
                merged_weight_lora = self.merge_AB()
                self.weight.data += merged_weight_lora * self.scaling
            self.merged = True

    def forward(self, x: torch.Tensor):
        def T(w):
            return w.T if self.fan_in_fan_out else w
        if self.r > 0 and not self.merged:
            result = F.linear(x, T(self.weight), bias=self.bias)
            if self.r > 0:
                weight_AB = self.merge_AB()
                result += F.linear(x, T(weight_AB), bias=self.bias) * self.scaling
            return result
        else:
            return F.linear(x, T(self.weight), bias=self.bias)    
  
class NoRALinear_B(nn.Linear, LoRALayer):
    def __init__(self, 
                 existing_linear: nn.Linear,
                 r: int = 0,
                 lora_alpha: float = 1.0, 
                 lora_dropout: float = 0.0, 
                 merge_weights: bool = True, 
                 fan_in_fan_out: bool = False,
                 structure: str = '1',
                 **kwargs):
        super().__init__(
            in_features=existing_linear.in_features, 
            out_features=existing_linear.out_features)
        self.load_state_dict(existing_linear.state_dict())
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                           merge_weights=merge_weights)
        self.fan_in_fan_out = fan_in_fan_out
        self.structure = structure

        # Actual trainable parameters
        in_features = existing_linear.in_features
        out_features = existing_linear.out_features
        if self.r > 0:
            self.lora_A = nn.Parameter(torch.randn(in_features, r), requires_grad=True)
            self.lora_B = nn.Parameter(torch.randn(r, out_features), requires_grad=True)
            self.scaling = self.lora_alpha / self.r

            # Freezing the pre-trained weight matrix
            self.lora_A.requires_grad = False
            self.lora_B.requires_grad = False
            self.weight.requires_grad = False

            self.init_W_ab_structure(structure)

           
        if fan_in_fan_out:
            self.weight.data = self.weight.data.T

    def init_W_AB_param(self, W_prime, S_diag, init_method: str):
        def T(w):
            return w.T if self.fan_in_fan_out else w
        with torch.no_grad():
            weight = T(W_prime) # out, in -> in, out
            if init_method == '1':
                # kaiming初始化
                nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
                nn.init.zeros_(self.lora_B)

            elif init_method == '2':
                # svd初始化: U @ Sigma,	V
                U, S, Vh = torch.linalg.svd(weight, full_matrices=False)
                U = U[:, :self.r]
                S = S[:self.r]
                Vh = Vh[:self.r, :]
                
                self.lora_A.data.copy_(torch.inverse(S_diag.T) @ U)
                self.lora_B.data.copy_(Vh)
            elif init_method == '3':
                # svd初始化： U, Sigma @ Vh
                U, S, Vh = torch.linalg.svd(weight, full_matrices=False)
                U = U[:, :self.r]
                S = S[:self.r]
                Vh = Vh[:self.r, :]
                if hasattr(self, 'lora_A'):
                    self.lora_A.data.copy_(torch.inverse(S_diag.T) @ U)
                    self.lora_B.data.copy_(torch.diag(S) @ Vh)
            elif init_method == '4':
                # svd初始化: U @ sqrt(Sigma), sqrt(Sigma) @ Vh
                U, S, Vh = torch.linalg.svd(weight, full_matrices=False)
                U = U[:, :self.r]
                S = torch.sqrt(S[:self.r])
                Vh = Vh[:self.r, :]
                if hasattr(self, 'lora_A'):
                    self.lora_A.data.copy_(torch.inverse(S_diag.T) @ U @ torch.diag(S))
                    self.lora_B.data.copy_(torch.diag(S) @ Vh)
            
    def init_W_ab_structure(self, structure: str):
        r = self.r
        in_features = self.in_features
        out_features = self.out_features
        if structure == '1':
            # LoRA_a并行结构
            self.W_a = nn.ParameterDict({
                'lora_a1': nn.Parameter(torch.randn(in_features, r // 2), requires_grad=True),
                'lora_a2': nn.Parameter(torch.randn(r // 2, r), requires_grad=True)
            })
            # 初始化
            nn.init.kaiming_uniform_(self.W_a['lora_a1'], a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.W_a['lora_a2'])
        elif structure == '2':
            # LoRA_a串联结构
            self.W_a = nn.ParameterDict({
                'lora_a1': nn.Parameter(torch.randn(r, 16), requires_grad=True),
                'lora_a2': nn.Parameter(torch.randn(16, r), requires_grad=True)
            })
            # 初始化
            nn.init.kaiming_uniform_(self.W_a['lora_a1'], a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.W_a['lora_a2'])
        elif structure == '3':
            # W_a r*r串行结构
            self.W_a = nn.ParameterDict({
                'lora_a1': nn.Parameter(torch.randn(r, r), requires_grad=True)
            })
            # 初始化
            nn.init.kaiming_uniform_(self.W_a['lora_a1'], a=math.sqrt(5))
        elif structure == '4':
            # LoRA_b并行结构
            self.W_b = nn.ParameterDict({
                'lora_b1': nn.Parameter(torch.randn(r, r // 2), requires_grad=True),
                'lora_b2': nn.Parameter(torch.randn(r // 2, out_features), requires_grad=True)
            })
            # 初始化
            nn.init.kaiming_uniform_(self.W_b['lora_b1'], a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.W_b['lora_b2'])
        elif structure == '5':
            # LoRA_b串联结构
            self.W_b = nn.ParameterDict({
                'lora_b1': nn.Parameter(torch.randn(out_features, r // 2), requires_grad=True),
                'lora_b2': nn.Parameter(torch.randn(r // 2, out_features), requires_grad=True)
            })
            # 初始化
            nn.init.kaiming_uniform_(self.W_b['lora_b1'], a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.W_b['lora_b2'])
        elif structure == '6':
            # W_b r*r串行结构
            self.W_b = nn.ParameterDict({
                'lora_b1': nn.Parameter(torch.randn(out_features, out_features), requires_grad=True)
            })
            # 初始化
            nn.init.kaiming_uniform_(self.W_b['lora_b1'], a=math.sqrt(5))
        else:
            raise ValueError(f"Unknown Structure: {structure}")

    def merge_lora(self):
        if self.structure == '1':
            W_AB = (self.lora_A + self.W_a['lora_a1'] @ self.W_a['lora_a2']) @ self.lora_B
        elif self.structure == '2':
            W_AB = (self.lora_A @ self.W_a['lora_a1'] @ self.W_a['lora_a2']) @ self.lora_B
        elif self.structure == '3':
            W_AB = self.lora_A @ self.W_a['lora_a1'] @ self.lora_B
        elif self.structure == '4':
            W_AB = self.lora_A @ (self.lora_B + self.W_b['lora_b1'] @ self.W_b['lora_b2'])
        elif self.structure == '5':
            W_AB = self.lora_A @ (self.lora_B @ self.W_b['lora_b1'] @ self.W_b['lora_b2'])
        elif self.structure == '6':
            W_AB = self.lora_A @ self.lora_B @ self.W_b['lora_b1']
        else:
            raise ValueError(f"未知的结构: {self.structure}")

        return W_AB

    def train(self, mode: bool = True):
        nn.Linear.train(self, mode)
        if self.merge_weights and self.merged:
            # Make sure that the weights are not merged
            if self.r > 0:
                merged_weight_lora = self.merge_lora()
                self.weight.data -= merged_weight_lora * self.scaling
            self.merged = False
    
    def eval(self):
        nn.Linear.eval(self)
        if self.merge_weights and not self.merged:
            # Merge the weights and mark it
            if self.r > 0:
                merged_weight_lora = self.merge_lora()
                self.weight.data += merged_weight_lora * self.scaling
            self.merged = True

    def forward(self, x: torch.Tensor):
        def T(w):
            return w.T if self.fan_in_fan_out else w
        if self.r > 0 and not self.merged:
            result = F.linear(x, T(self.weight), bias=self.bias)
            if self.r > 0:
                weight_AB = self.merge_lora()
                result += F.linear(x, T(weight_AB), bias=self.bias) * self.scaling
            return result
        else:
            return F.linear(x, T(self.weight), bias=self.bias)    

class NoRALinear_C(nn.Linear, LoRALayer):
    def __init__(self, 
                 existing_linear: nn.Linear,
                 r: int = 0,
                 lora_alpha: float = 1.0, 
                 lora_dropout: float = 0.0, 
                 merge_weights: bool = True, 
                 fan_in_fan_out: bool = False,
                 Wab_structure: str = '1',
                 **kwargs):
        feature_size = existing_linear.in_features if existing_linear.in_features < existing_linear.out_features else existing_linear.out_features
        r = math.floor(feature_size * 0.8)
        super().__init__(
            in_features=existing_linear.in_features, 
            out_features=existing_linear.out_features)
        self.load_state_dict(existing_linear.state_dict())
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                           merge_weights=merge_weights)
        self.fan_in_fan_out = fan_in_fan_out
        self.Wab_structure = Wab_structure

        # Actual trainable parameters
        in_features = existing_linear.in_features
        out_features = existing_linear.out_features
        if self.r > 0:
            self.lora_A = nn.Parameter(torch.randn(in_features, r), requires_grad=True)
            self.lora_B = nn.Parameter(torch.randn(r, out_features), requires_grad=True)
            self.scaling = self.lora_alpha / self.r

            # Freezing the pre-trained weight matrix
            self.lora_A.requires_grad = False
            self.lora_B.requires_grad = False
            self.weight.requires_grad = False

            self.init_Wab_structure(Wab_structure)

        if fan_in_fan_out:
            self.weight.data = self.weight.data.T

    def init_W_AB_param(self, W_prime, S_diag, init_method: str):
        def T(w):
            return w.T if self.fan_in_fan_out else w
        with torch.no_grad():
            weight = T(W_prime) # out, in -> in, out

            if init_method == '1':
                # svd初始化: U @ Sigma,	V
                U, S, Vh = torch.linalg.svd(weight, full_matrices=False)
                U = U[:, :self.r]
                S = S[:self.r]
                Vh = Vh[:self.r, :]
                
                self.lora_A.data.copy_(torch.inverse(S_diag.T) @ U @ torch.diag(S))
                self.lora_B.data.copy_(Vh)
            elif init_method == '2':
                # svd初始化： U, Sigma @ Vh
                U, S, Vh = torch.linalg.svd(weight, full_matrices=False)
                U = U[:, :self.r]
                S = S[:self.r]
                Vh = Vh[:self.r, :]
                if hasattr(self, 'lora_A'):
                    self.lora_A.data.copy_(torch.inverse(S_diag.T) @ U)
                    self.lora_B.data.copy_(torch.diag(S) @ Vh)
            elif init_method == '3':
                # svd初始化: U @ sqrt(Sigma), sqrt(Sigma) @ Vh
                U, S, Vh = torch.linalg.svd(weight, full_matrices=False)
                U = U[:, :self.r]
                S = torch.sqrt(S[:self.r])
                Vh = Vh[:self.r, :]
                if hasattr(self, 'lora_A'):
                    self.lora_A.data.copy_(torch.inverse(S_diag.T) @ U @ torch.diag(S))
                    self.lora_B.data.copy_(torch.diag(S) @ Vh)
            
    def init_Wab_structure(self, Wab_structure: str):
        r = self.r
        in_features = self.in_features
        out_features = self.out_features
        if Wab_structure == '1':
            self.W_a = nn.ParameterDict({
                'lora_a1': nn.Parameter(torch.randn(r, 16), requires_grad=True),
                'lora_a2': nn.Parameter(torch.randn(16, r), requires_grad=True)
            })
            self.W_b = None
        elif Wab_structure == '2':
            self.W_a = None
            self.W_b = nn.ParameterDict({
                'lora_b1': nn.Parameter(torch.randn(out_features, 16), requires_grad=True),
                'lora_b2': nn.Parameter(torch.randn(16, out_features), requires_grad=True)
            })
        else:
            raise ValueError(f"Unknown Wab Structure: {Wab_structure}")
    
    
    def merge_lora(self):
        if self.Wab_structure == '1':
            W_AB = self.lora_A @ self.W_a['lora_a1'] @ self.W_a['lora_a2'] @ self.lora_B
        elif self.Wab_structure == '2':
            W_AB = self.lora_A @ self.lora_B @ self.W_b['lora_b1'] @ self.W_b['lora_b2']
        else:
            raise ValueError(f"Unknown Forward Structure: {self.forward_structure}")

        return W_AB

    def train(self, mode: bool = True):
        nn.Linear.train(self, mode)
        if self.merge_weights and self.merged:
            # Make sure that the weights are not merged
            if self.r > 0:
                merged_weight_lora = self.merge_lora()
                self.weight.data -= merged_weight_lora * self.scaling
            self.merged = False
    
    def eval(self):
        nn.Linear.eval(self)
        if self.merge_weights and not self.merged:
            # Merge the weights and mark it
            if self.r > 0:
                merged_weight_lora = self.merge_lora()
                self.weight.data += merged_weight_lora * self.scaling
            self.merged = True

    def forward(self, x: torch.Tensor):
        def T(w):
            return w.T if self.fan_in_fan_out else w
        if self.r > 0 and not self.merged:
            result = F.linear(x, T(self.weight), bias=self.bias)
            if self.r > 0:
                weight_AB = self.merge_lora()
                result += F.linear(x, T(weight_AB), bias=self.bias) * self.scaling
            return result
        else:
            return F.linear(x, T(self.weight), bias=self.bias)    
  