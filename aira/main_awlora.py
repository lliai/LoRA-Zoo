import os
os.environ["CUDA_VISIBLE_DEVICES"] = "5"
# os.environ["WANDB_MODE"] = "offline"
os.environ['WANDB_API_KEY'] = '83b5aa31dee1c29cbcad6a05b53018482c83cfa1'
import wandb
import torch
import torch.utils.data
import torchvision.transforms as transforms
import clip
from datasets import build_dataset
from datasets.utils import build_data_loader

from utils import *
from run_utils_awlora import *
from lora import run_awlora, run_warmup_awlora

def main(config=None):

    # Load config file and set AwLora parameters
    args = get_arguments()

    if args.step == 1:
        # 利用rank分配训练CLIP
        # Load wandb.agent hy-config
        wandb.init(config=config, name='awlora', notes="构建rank分配")
        config = wandb.config
        args.theta_type = config.theta_type
        args.dataset = config.dataset

        wandb.config.update(vars(args))


        # Test config

        # Set random seed
        set_random_seed(args.seed)

        # CLIP
        clip_model, preprocess = clip.load(args.backbone)
        clip_model.eval()
        logit_scale = 100

        # Prepare dataset
        print("Preparing dataset.")

        dataset = build_dataset(args.dataset, args.root_path, args.shots, preprocess)
        
        if args.dataset == 'imagenet':
            val_loader = torch.utils.data.DataLoader(dataset.val, batch_size=256, num_workers=8, shuffle=False, pin_memory=True)
            test_loader = torch.utils.data.DataLoader(dataset.test, batch_size=256, num_workers=8, shuffle=False, pin_memory=True)
        else:
            val_loader = build_data_loader(data_source=dataset.val, batch_size=256, is_train=False, tfm=preprocess, shuffle=False,  num_workers=8)
            test_loader = build_data_loader(data_source=dataset.test, batch_size=256, is_train=False, tfm=preprocess, shuffle=False,  num_workers=8)
            
        train_loader = None
        if not args.eval_only:
            train_tranform = transforms.Compose([
                transforms.RandomResizedCrop(size=224, scale=(0.08, 1), interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
            ])
            
            if args.dataset == 'imagenet':
                train_loader = torch.utils.data.DataLoader(dataset.train_x, batch_size=args.batch_size, num_workers=8, shuffle=True, pin_memory=True)
            else:
                train_loader = build_data_loader(data_source=dataset.train_x, batch_size=args.batch_size, tfm=train_tranform, is_train=True, shuffle=True, num_workers=8)

        # 获取rank分配
        rank_allocation, layer_factors = run_warmup_awlora(args, clip_model, dataset, train_loader)
        
        # 打印结果
        print("\n最优rank分配结果:")
        # rank_allocation : {layer_name: rank}
        # For example
        # text_resblock_0_q_proj: 16
        # vision_resblock_0_q_proj: 16
        # ...
        # 将rank_allocation的value转换为int
        rank_allocation = {k: int(v) for k, v in rank_allocation.items()}
        for layer_name, rank in rank_allocation.items():
            print(f"{layer_name}: {rank}")
            

        # 计算总参数量
        total_params = sum(layer_factors[k] * rank_allocation[k] for k in rank_allocation)
        print(f"\n总参数量: {int(total_params)}")

        # 将分配结果保存到文件
        import json
        with open(f'./output/awlora/json/rank_allocation_{args.theta_type}_{args.dataset}_{args.shots}.json', 'w') as f:
            json.dump(rank_allocation, f, indent=4)
        with open(f'./output/awlora/json/layer_factors_{args.theta_type}_{args.dataset}_{args.shots}.json', 'w') as f:
            json.dump(layer_factors, f, indent=4)
        
        wandb.finish()

    elif args.step == 2:
        # 利用rank分配训练CLIP
        # Load wandb.agent hy-config
        wandb.init(config=config, name='awlora', 
                notes="使用经过SLSQP优化的rank分配（影响因子为act/lod），训练过程主干网络激活感知帮助lora分支训练，参数量为1.4M（相当于r=16）")
        config = wandb.config
        args.theta_type = config.theta_type
        args.dataset = config.dataset
        args.lr = config.lr
        wandb.config.update(vars(args))

        # Test config

        # Set random seed
        set_random_seed(args.seed)
        
        # CLIP
        clip_model, preprocess = clip.load(args.backbone)
        clip_model.eval()
        logit_scale = 100

        # Prepare dataset
        print("Preparing dataset.")

        dataset = build_dataset(args.dataset, args.root_path, args.shots, preprocess)
        
        if args.dataset == 'imagenet':
            val_loader = torch.utils.data.DataLoader(dataset.val, batch_size=256, num_workers=8, shuffle=False, pin_memory=True)
            test_loader = torch.utils.data.DataLoader(dataset.test, batch_size=256, num_workers=8, shuffle=False, pin_memory=True)
        else:
            val_loader = build_data_loader(data_source=dataset.val, batch_size=256, is_train=False, tfm=preprocess, shuffle=False,  num_workers=8)
            test_loader = build_data_loader(data_source=dataset.test, batch_size=256, is_train=False, tfm=preprocess, shuffle=False,  num_workers=8)
            
        train_loader = None
        if not args.eval_only:
            train_tranform = transforms.Compose([
                transforms.RandomResizedCrop(size=224, scale=(0.08, 1), interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
            ])
            
            if args.dataset == 'imagenet':
                train_loader = torch.utils.data.DataLoader(dataset.train_x, batch_size=args.batch_size, num_workers=8, shuffle=True, pin_memory=True)
            else:
                train_loader = build_data_loader(data_source=dataset.train_x, batch_size=args.batch_size, tfm=train_tranform, is_train=True, shuffle=True, num_workers=8)

        # Run AwLora
        run_awlora(args, clip_model, logit_scale, dataset, train_loader, val_loader, test_loader)
        
        wandb.finish()

    elif args.step == 3:
        # None
        return None

if __name__ == '__main__':
    
    wandb.login()
    sweep_config = {
        "method": "grid",
        "metric": {"goal": "maximize", "name": "Test Accuracy"},
        "parameters": {
            'dataset':{
                'values': ['dtd', 'food101', 'ucf101', 'stanford_cars', 'oxford_pets']
            },
            'theta_type':{
                'values': ['act', 'lod']
            },
            'lr':{
                'values': [1e-4, 5e-4, 1e-3]
            }
            
        },
    }
    sweep_id = wandb.sweep(sweep_config, project="AwLora")
    wandb.agent(sweep_id, main, count=30)