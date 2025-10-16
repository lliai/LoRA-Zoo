import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
from loralib.lora_lib import AwRankLinear

# 示例模型定义
class DynamicLoRAModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        # 全部使用AwRankLinear层
        original_linear1 = nn.Linear(input_dim, 256)
        self.lora_layer1 = AwRankLinear(
            existing_linear=original_linear1,
            R_max=16,        
            budget=0.4,      
            beta=0.9,        
            update_freq=100, 
            lora_alpha=32    
        )
        self.relu = nn.ReLU()
        
        original_linear2 = nn.Linear(256, output_dim)
        self.lora_layer2 = AwRankLinear(
            existing_linear=original_linear2,
            R_max=16,        
            budget=0.4,      
            beta=0.9,        
            update_freq=100, 
            lora_alpha=32   
        )

    def forward(self, x):
        x = self.lora_layer1(x)
        x = self.relu(x)
        return self.lora_layer2(x)

# 示例训练流程
def train_model():
    # 1. 准备数据
    input_dim = 128
    output_dim = 10
    train_data = torch.randn(1000, input_dim)
    train_labels = torch.randint(0, output_dim, (1000,))
    train_loader = DataLoader(TensorDataset(train_data, train_labels), batch_size=32)

    # 2. 初始化模型和优化器
    model = DynamicLoRAModel(input_dim, output_dim)
    optimizer = optim.AdamW([
        {'params': model.parameters(), 'lr': 1e-3}
    ])
    criterion = nn.CrossEntropyLoss()

    # 3. 训练循环
    for epoch in range(10):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            
           
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            
            # 梯度裁剪（防止动态结构不稳定）
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()

            # 打印当前秩信息
            if batch_idx % 50 == 0:
                current_rank = model.lora_layer1.current_rank
                print(f'Epoch: {epoch} | Step: {batch_idx} | Loss: {loss.item():.4f} | Current Rank: {current_rank}')
                print(f'Active Parameters: {sum(p.numel() for p in model.lora_layer1.parameters() if p.requires_grad)}')


def test_model(model, test_loader):
    model.eval() 
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            pred = output.argmax(dim=1)
            correct += (pred == target).sum().item()
    
    print(f'Test Accuracy: {100 * correct / len(test_loader.dataset):.2f}%')


if __name__ == "__main__":

    train_model()
    

    test_data = torch.randn(200, 128)
    test_labels = torch.randint(0, 10, (200,))
    test_loader = DataLoader(TensorDataset(test_data, test_labels), batch_size=32)
    

    best_model = DynamicLoRAModel(128, 10)
    test_model(best_model, test_loader)