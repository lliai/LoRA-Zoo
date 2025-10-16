import random
import argparse  
import numpy as np 
import torch


    

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_arguments():

    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default=1997, type=int)
    # Dataset arguments
    parser.add_argument('--root_path', type=str, default='data')
    parser.add_argument('--dataset', type=str, default='dtd')
    parser.add_argument('--shots', default=4, type=int)
    # Model arguments
    parser.add_argument('--backbone', default='ViT-B/16', type=str)
    # Training arguments
    parser.add_argument('--lr', default=5e-4, type=float)
    parser.add_argument('--n_iters', default=300, type=int)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--weight_decay', default=0.005, type=float, help='weight decay')
    # LoRA arguments
    parser.add_argument('--position', type=str, default='all', choices=['bottom', 'mid', 'up', 'half-up', 'half-bottom', 'all', 'top3'], help='where to put the LoRA modules')
    parser.add_argument('--encoder', type=str, choices=['text', 'vision', 'both'], default='both')
    parser.add_argument('--params', metavar='N', type=str, nargs='+',default=['q', 'k', 'v'], help='list of attention matrices where putting a LoRA') 
    parser.add_argument('--r', default=16, type=int, help='the rank of the low-rank matrices')
    parser.add_argument('--alpha', default=2, type=int, help='scaling (see LoRA paper)')
    parser.add_argument('--dropout_rate', default=0.25, type=float, help='dropout rate applie+d before the LoRA module')
    # AwLora arguments
    # Step 1
    parser.add_argument('--step', default=2, type=int, choices=[1, 2, 3], help='the step of AwLora')
    parser.add_argument('--linear_types', nargs='+', type=str, default=['q', 'k', 'v'], help='linear types to apply AwLora')
    parser.add_argument('--max_rank', default=32, type=int, help='Max rank of optimization range')
    parser.add_argument('--rank_budget', default=0.7, type=float, help='Rank budget of AwLora (M)')
    parser.add_argument('--lod_M', default=5, type=float, help='LOD threshold')
    # Step 2
    # Activation arguments
    parser.add_argument('--svd_lora', default=True, action='store_false', help='use SVD to get the initialization of low-rank matrices')
    parser.add_argument('--svd_step', default=1, type=int, choices=[1, 2, 3], help='the step of SVD to get the initialization of low-rank matrices')
    parser.add_argument('--rand_lora', default=False, action='store_true', help='use random initialization of low-rank matrices')   
    parser.add_argument('--theta_type', default='act', choices=['act', 'lod']) 

    parser.add_argument('--run_name', default=None, help='The N-th experiment is recorded, and the default is None')
    parser.add_argument('--save_path', default=None, help='path to save the lora modules after training, not saved if None')
    parser.add_argument('--filename', default='lora_weights', help='file name to save the lora weights (.pt extension will be added)')
    
    parser.add_argument('--eval_only', default=False, action='store_true', help='only evaluate the LoRA modules (save_path should not be None)')
    args = parser.parse_args()

    return args
    

        
