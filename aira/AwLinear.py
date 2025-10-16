import torch.nn as nn
from loralib.lora_lib import AwLinear
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

def test_awlinear_params():
    # 设置随机种子以确保可重复性
    torch.manual_seed(42)
    
    # 创建一个基础的线性层
    original_linear = nn.Linear(256, 512)
    
    # 测试用的输入数据
    x = torch.randn(32, 256)
    
    def print_params_stats(model_name, lora_A, lora_B):
        print(f"\n=== {model_name} 参数统计 ===")
        print(f"lora_A 均值: {lora_A.mean():.4f}, 标准差: {lora_A.std():.4f}")
        print(f"lora_B 均值: {lora_B.mean():.4f}, 标准差: {lora_B.std():.4f}")
    
    # 1. 测试 svd_lora
    print("\n测试 SVD 初始化:")
    model_svd = AwLinear(
        existing_linear=original_linear,
        r=4,
        svd_lora=True,
        rand_lora=False
    )
    
    print_params_stats("SVD LoRA", model_svd.lora_A.data, model_svd.lora_B.data)
    
    # 2. 测试 rand_lora
    print("\n测试随机初始化:")
    model_rand = AwLinear(
        existing_linear=original_linear,
        r=4,
        svd_lora=False,
        rand_lora=True
    )
    
    print_params_stats("Random LoRA", model_rand.lora_A.data, model_rand.lora_B.data)
    
    # 3. 测试 left_act
    print("\n测试激活感知位置:")
        
    # 右侧激活
    model_right = AwLinear(
        existing_linear=original_linear,
        r=4,
    )
    
    # 运行前向传播
    with torch.no_grad():
        out_right = model_right(x)
        
        print(f"右侧激活输出形状: {out_right.shape}")

if __name__ == "__main__":
    test_awlinear_params()  