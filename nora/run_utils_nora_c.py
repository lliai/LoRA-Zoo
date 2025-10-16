
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
    parser.add_argument('--seed', default=1, type=int)
    # Dataset arguments
    parser.add_argument('--root_path', type=str, default='data')
    parser.add_argument('--dataset', type=str, default='dtd')
    parser.add_argument('--shots', default=16, type=int)
    # Model arguments
    parser.add_argument('--backbone', default='ViT-B/16', type=str)
    # Training arguments
    parser.add_argument('--lr', default=1e-2, type=float)
    parser.add_argument('--n_iters', default=400, type=int)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--weight_decay', default=0.01, type=float, help='weight decay')
    # LoRA arguments
    parser.add_argument('--position', type=str, default='all', choices=['bottom', 'mid', 'up', 'half-up', 'half-bottom', 'all', 'top3'], help='where to put the LoRA modules')
    parser.add_argument('--encoder', type=str, choices=['text', 'vision', 'both'], default='both')
    parser.add_argument('--params', metavar='N', type=str, nargs='+',default=['q', 'k', 'v'], help='list of attention matrices where putting a LoRA') 
    parser.add_argument('--r', default=2, type=int, help='the rank of the low-rank matrices')
    parser.add_argument('--alpha', default=1, type=int, help='scaling (see LoRA paper)')
    parser.add_argument('--dropout_rate', default=0.25, type=float, help='dropout rate applied before the LoRA module')
    # UoRA_B arguments
    parser.add_argument('--init_method', type=str, default='1', choices=['1', '2', '3'], help='choose the initialization method for W_A and W_B')
    parser.add_argument('--Wab_structure', type=str, default='1', choices=['1', '2'], help='choose the structure for W_a and W_b')

    parser.add_argument('--run_name', default=None, help='The N-th experiment is recorded, and the default is None')
    parser.add_argument('--save_path', default=None, help='path to save the lora modules after training, not saved if None')
    parser.add_argument('--filename', default='lora_weights', help='file name to save the lora weights (.pt extension will be added)')
    
    parser.add_argument('--eval_only', default=False, action='store_true', help='only evaluate the LoRA modules (save_path should not be None)')
    args = parser.parse_args()

    return args
    

        
