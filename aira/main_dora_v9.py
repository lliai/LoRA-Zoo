import os
os.environ["CUDA_VISIBLE_DEVICES"] = "6"
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
from run_utils_dora_v9 import *
from lora import run_dora_v9, run_warmup_asvd

def main(config=None):

    # Load config file and set DoRA_v9 parameters
    args = get_arguments()

    # Load wandb.agent hy-config
    wandb.init(config=config, name='DoRA_v9', 
               notes="dora+vera+经过asvd加权的lora_a/b的svd初始化(修复Wcompose错误的版本)" + 
                     "+" + "ortho loss")
    config = wandb.config
    args.shots = config.shots
    args.r = config.r
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

    # 获取加权Weight和缩放矩阵S
    text_W_prime_dict, text_S_dict, vision_W_prime_dict, vision_S_dict = run_warmup_asvd(args, clip_model, dataset, train_loader)
    
    # Reload CLIP
    clip_model, preprocess = clip.load(args.backbone)
    clip_model.eval()

    run_dora_v9(args, clip_model, logit_scale, dataset, train_loader, val_loader, test_loader, text_W_prime_dict, text_S_dict, vision_W_prime_dict, vision_S_dict)
    
    wandb.finish()


if __name__ == '__main__':
    
    wandb.login()
    sweep_config = {
        "method": "grid",
        "metric": {"goal": "maximize", "name": "val_acc"},
        "parameters": {
            'shots': {
                'values': [4]
            },
            'r':{
                'values': [32, 128, 256]
            },
            'lr':{
                'values': [1e-2, 5e-2, 1e-3, 5e-3, 1e-4]
            },
            'ortho_loss_weight': {
                'values': [0.1, 0.05]
            }
        },
    }
    sweep_id = wandb.sweep(sweep_config, project="CLIP-Awesome-LoRA")
    wandb.agent(sweep_id, main, count=30)