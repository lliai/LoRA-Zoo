import os
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
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
from run_utils_nora_c import *
from lora import run_nora_c, run_warmup_asvd

def main(config=None):

    # Load config file and set noRA_C parameters
    args = get_arguments()

    # Load wandb.agent hy-config
    wandb.init(config=config, name='NoRA_C', notes="表C: 验证经过ASVD的初始化W_A和W_B后，对W_a、W_b是否设置成LoRA对性能的影响，介于nora的独特特性，需要增大nora的内部r")
    config = wandb.config
    args.dataset = config.dataset
    args.init_method = config.init_method
    args.Wab_structure = config.Wab_structure
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

    run_nora_c(args, clip_model, logit_scale, dataset, train_loader, val_loader, test_loader, text_W_prime_dict, text_S_dict, vision_W_prime_dict, vision_S_dict)
    
    wandb.finish()


if __name__ == '__main__':
    
    wandb.login()
    sweep_config = {
        "method": "grid",
        "metric": {"goal": "maximize", "name": "Val Accuracy"},
        "parameters": {
            'dataset': {
                'values': ['stanford_cars']
            },
            'init_method': {
                'values': ['1', '3']
            },
            'Wab_structure': {
                'values': ['1', '2']
            },
            'shots': {
                'values': [16]
            },
            'r':{
                'values': [16]
            },
            'lr':{
                'values': [1e-2, 1e-3]
            },
        },
    }
    sweep_id = wandb.sweep(sweep_config, project="NoRA")
    wandb.agent(sweep_id, main, count=20)