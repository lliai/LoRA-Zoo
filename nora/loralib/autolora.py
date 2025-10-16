import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from .layers import LoRALayer 
from typing import Optional, List 

class AutoloraLinear(nn.Linear, LoRALayer):
    def __init__(
        self, 
        in_features: int, 
        out_features: int, 
        r: int = 8, 
        lora_alpha: int = 1, 
        lora_dropout: float = 0.,
        fan_in_fan_out: bool = False, 
        merge_weights: bool = True,
        **kwargs
    ):
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
        nn.Linear.reset_parameters(self)
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
    
    
class Architecture(nn.Module):
    def __init__(self, r):
        super(Architecture, self).__init__()
        
        alpha = torch.randn(r)
        nn.init.normal_(alpha, mean=0.5, std=0.02)
        self.log_alpha = nn.Parameter(torch.clamp(alpha, 1e-6, 1-1e-6).log())
        #print(self.alpha)
        print(self.log_alpha.exp())
        # nn.init.normal_(alpha, mean=1.0, std=1.5)
        # self.alpha = nn.Parameter(alpha)
        # print(F.softmax(self.alpha))
        # print(self.alpha)
        
    def forward(self):
        return self.log_alpha.exp()
        #return F.softmax(self.alpha)
    
    def regularizer(self, r):
        ones = self.log_alpha.new_ones(r)
        alpha = torch.clamp(self.log_alpha.exp(), 1e-6, 1-1e-6)
        entropy = -1e-2*torch.sum((ones-alpha) * torch.log(ones-alpha) + alpha * torch.log(alpha))
        #print(entropy)
        return entropy
        
        
class AutoEmbedding(nn.Embedding, LoRALayer):
    # LoRA implemented in a dense layer
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        r: int = 0,
        lora_alpha: int = 1,
        merge_weights: bool = True,
        **kwargs
    ):
        nn.Embedding.__init__(self, num_embeddings, embedding_dim, **kwargs)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=0,
                           merge_weights=merge_weights)
        # Actual trainable parameters
        if r > 0:
            self.lora_A = nn.Parameter(self.weight.new_zeros((r, num_embeddings)))
            self.lora_B = nn.Parameter(self.weight.new_zeros((embedding_dim, r)))
            self.scaling = self.lora_alpha / self.r
            self.ranknum = r
            # Freezing the pre-trained weight matrix
            # self.weight.requires_grad = False
        self.reset_parameters()

    def reset_parameters(self):
        nn.Embedding.reset_parameters(self)
        if hasattr(self, 'lora_A'):
            # initialize A,B the same way as the default for nn.Linear 
            nn.init.normal_(self.lora_A, mean=0.0, std=0.02)
            nn.init.normal_(self.lora_B, mean=0.0, std=0.02)

    def train(self, alphas, mode: bool = True):
        nn.Embedding.train(self, mode)
        if self.merge_weights and self.merged:
            # Make sure that the weights are not merged
            if self.r > 0:
                for i in range(0, self.r):
                    # self.weight.data -= T(
                    #     torch.clamp(alphas[i], 1e-6, 1-1e-6) * \
                    #         torch.matmul(torch.unsqueeze(self.lora_B[:,i], 1), torch.unsqueeze(self.lora_A[i,:], 0))
                    # ) * self.scaling / (self.ranknum+1e-5)
                    self.weight.data -= (
                        alphas[i] * \
                            torch.matmul(torch.unsqueeze(self.lora_B[:,i], 1), torch.unsqueeze(self.lora_A[i,:], 0))
                    ).T * self.scaling
            self.merged = False
    
    def eval(self, alphas):
        nn.Linear.eval(self)
        if self.merge_weights and not self.merged:
            # Merge the weights and mark it
            if self.r > 0:
                for i in range(0, self.r):
                    # self.weight.data += T(
                    #     torch.clamp(alphas[i], 1e-6, 1-1e-6) * \
                    #        torch.matmul(torch.unsqueeze(self.lora_B[:,i], 1), torch.unsqueeze(self.lora_A[i,:], 0))
                    # ) * self.scaling / (self.ranknum+1e-5)
                    self.weight.data += (
                        alphas[i] * \
                            torch.matmul(torch.unsqueeze(self.lora_B[:,i], 1), torch.unsqueeze(self.lora_A[i,:], 0))
                    ).T * self.scaling
            self.merged = True

    def forward(self, x: torch.Tensor, alphas):
        if self.r > 0 and not self.merged:
            result = nn.Embedding.forward(self, x)
            if self.r > 0:
                # result += (
                #     self.lora_dropout(x) @ (self.lora_A * self.lora_E).T @ self.lora_B.T
                # ) * self.scaling / (self.ranknum+1e-5)
                #print(F.softmax(alphas, dim=0)[i].data * torch.matmul(torch.unsqueeze(self.lora_B[:,i], 1), torch.unsqueeze(self.lora_A[i,:], 0)).shape        
                # result += x @ T((
                #     torch.clamp(alphas[i], 1e-6, 1-1e-6) * \
                #         torch.matmul(torch.unsqueeze(self.lora_B[:,i], 1), torch.unsqueeze(self.lora_A[i,:], 0))
                # ).T) * self.scaling / (self.ranknum+1e-5)
                after_A = F.embedding(
                    x, self.lora_A.T, self.padding_idx, self.max_norm,
                    self.norm_type, self.scale_grad_by_freq, self.sparse
                )
                #print(after_A.shape)
                for i in range(0, self.r):
                    result += (
                        alphas[i] * \
                            torch.matmul(torch.unsqueeze(after_A[:, :, i], 2), torch.unsqueeze((self.lora_B.T)[i,:], 0))
                    ) * self.scaling
            return result
        else:
            return nn.Embedding.forward(self, x)
        
        
    def maskforward(self, x, alphas, mark):
        if self.r > 0 and not self.merged:
            result = nn.Embedding.forward(self, x)
            if self.r > 0:
                
                # result += (
                #     self.lora_dropout(x) @ (self.lora_A * self.lora_E).T @ self.lora_B.T
                # ) * self.scaling / (self.ranknum+1e-5)
                # result += x @ T((
                #     mark[i].item() * torch.clamp(alphas[i], 1e-6, 1-1e-6) * \
                #         torch.matmul(torch.unsqueeze(self.lora_B[:,i], 1), torch.unsqueeze(self.lora_A[i,:], 0))
                # ).T) * self.scaling / (self.ranknum+1e-5)
                after_A = F.embedding(
                x, self.lora_A.T, self.padding_idx, self.max_norm,
                self.norm_type, self.scale_grad_by_freq, self.sparse
                )
                for i in range(0, self.r):
                    result += (
                        mark[i].item() * alphas[i] * \
                            torch.matmul(torch.unsqueeze(after_A[:,i], 1), torch.unsqueeze((self.lora_B.T)[i,:], 0))
                    ) * self.scaling 
            return result
        else:
            return nn.Embedding.forward(self, x)
        
        
class AutoMergedLinear(nn.Linear, LoRALayer):
    # LoRA implemented in a dense layer
    def __init__(
        self, 
        in_features: int, 
        out_features: int, 
        r: int = 0, 
        lora_alpha: int = 1, 
        lora_dropout: float = 0.,
        enable_lora: List[bool] = [False],
        fan_in_fan_out: bool = False,
        merge_weights: bool = True,
        **kwargs
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                           merge_weights=merge_weights)
        assert out_features % len(enable_lora) == 0, \
            'The length of enable_lora must divide out_features'
        self.enable_lora = enable_lora
        self.fan_in_fan_out = fan_in_fan_out
        # Actual trainable parameters
        if r > 0 and any(enable_lora):
            self.lora_A = nn.Parameter(
                self.weight.new_zeros((r * sum(enable_lora), in_features)))
            self.lora_B = nn.Parameter(
                self.weight.new_zeros((out_features // len(enable_lora) * sum(enable_lora), r))
            ) # weights for Conv1D with groups=sum(enable_lora)
            self.scaling = self.lora_alpha / self.r
            # Freezing the pre-trained weight matrix
            #self.weight.requires_grad = False
            # Compute the indices
            self.lora_ind = self.weight.new_zeros(
                (out_features, ), dtype=torch.bool
            ).view(len(enable_lora), -1)
            self.lora_ind[enable_lora, :] = True
            self.lora_ind = self.lora_ind.view(-1)
        self.reset_parameters()
        if fan_in_fan_out:
            self.weight.data = self.weight.data.transpose(0, 1).contiguous()

    def reset_parameters(self):
        nn.Linear.reset_parameters(self)
        if hasattr(self, 'lora_A'):
            # initialize A the same way as the default for nn.Linear and B to zero
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

    def zero_pad(self, x):
        result = x.new_zeros((len(self.lora_ind), *x.shape[1:]), device=x.device)
        result[self.lora_ind] = x
        return result

    def merge_AB(self, alphas):
        def T(w):
            return w.transpose(0, 1).to(w.device) if self.fan_in_fan_out else w.to(w.device)
        
        delta_w = torch.zeros((self.lora_B.shape[0], self.lora_A.shape[1]), device=alphas.device)
        
        for i in range(0, self.r):
            delta_w += alphas[i] * F.conv1d(
                self.lora_A.unsqueeze(0)[:, 2*i:2*i+2, :].to(alphas.device), 
                self.lora_B.unsqueeze(-1)[:, i:i+1, :].to(alphas.device), 
                groups=sum(self.enable_lora)
            ).squeeze(0)
            
        return T(self.zero_pad(delta_w))

    
    def merge_maskAB(self, alphas, mask):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w
        delta_w = torch.zeros((self.lora_B.shape[0], self.lora_A.shape[1]))
        for i in range(0, self.r):
            delta_w += mask[i] * alphas[i] * F.conv1d(
                self.lora_A.unsqueeze(0)[:, 2*i:2*i+2, :], 
                self.lora_B.unsqueeze(-1)[:, i:i+1, :], 
                groups=sum(self.enable_lora)
            ).squeeze(0)
            
        return T(self.zero_pad(delta_w))

    def train(self, alphas, mode: bool = True):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w
        nn.Linear.train(self, mode)
        if mode:
            if self.merge_weights and self.merged:
                # Make sure that the weights are not merged
                if self.r > 0 and any(self.enable_lora):
                    self.weight.data -= self.merge_AB(alphas) * self.scaling
                self.merged = False
        else:
            if self.merge_weights and not self.merged:
                # Merge the weights and mark it
                if self.r > 0 and any(self.enable_lora):
                    self.weight.data += self.merge_AB(alphas) * self.scaling
                self.merged = True        

    def forward(self, x: torch.Tensor, alphas):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w
        if self.merged:
            return F.linear(x, T(self.weight), bias=self.bias)
        else:
            result = F.linear(x, T(self.weight), bias=self.bias)
            if self.r > 0:
                result += self.lora_dropout(x) @ T(self.merge_AB(alphas).T) * self.scaling
            return result
    
    def maskforward(self, x, alphas, mask):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w
        if self.merged:
            return F.linear(x, T(self.weight), bias=self.bias)
        else:
            result = F.linear(x, T(self.weight), bias=self.bias)
            if self.r > 0:
                result += self.lora_dropout(x) @ T(self.merge_maskAB(alphas, mask).T) * self.scaling
            return result
        
        
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