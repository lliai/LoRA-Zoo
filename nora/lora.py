from ast import arg
import wandb
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import *
from loralib.lora_lib import get_lora_plus_parameters, compute_orth_regu, RankAllocator
from loralib.utils import plot_individual_bars, apply_Lolora_v2, mark_only_lora_as_trainable, apply_lora, apply_Adalora, \
                            apply_Dylora, apply_attn, apply_Dora, apply_Dora_v2, apply_Dora_v3, apply_Dora_v4, apply_Dora_v5, \
                            apply_Lolora, apply_Lolora_v3, apply_Dora_v7, apply_Dora_v7_2, get_lora_parameters, lora_state_dict, \
                            apply_Dora_v7_3, apply_Dora_v8, apply_Dora_v9, apply_NoRA_A, apply_NoRA_B, apply_NoRA_C, \
                            save_lora, load_lora
from loralib import layers as lora_layers
import random
from loralib.lora_lib import DyLinear, LoLinear, LoLinear_v2, DoraLinear_v9

INDEX_POSITIONS_TEXT = {
    'top1': [11],
    'top2': [10, 11],
    'top3': [9, 10, 11],
    'bottom': [0, 1, 2, 3],
    'mid': [4, 5, 6, 7],
    'up': [8, 9, 10, 11],
    'half-up': [6, 7, 8, 9, 10, 11],
    'half-bottom': [0, 1, 2, 3, 4, 5],
    'all': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]}


INDEX_POSITIONS_VISION = {
    'ViT-B/16': {
        'top': [11],
        'top3': [9, 10, 11],
        'bottom': [0, 1, 2, 3],
        'mid': [4, 5, 6, 7],
        'up': [8, 9, 10, 11],
        'half-up': [6, 7, 8, 9, 10, 11],
        'half-bottom': [0, 1, 2, 3, 4, 5],
        'all': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]},
    'ViT-B/32': {
        'bottom': [0, 1, 2, 3],
        'mid': [4, 5, 6, 7],
        'up': [8, 9, 10, 11],
        'half-up': [6, 7, 8, 9, 10, 11],
        'half-bottom': [0, 1, 2, 3, 4, 5],
        'all': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]},

    'ViT-L/14': {
        'bottom': [0, 1, 2, 3],
        'mid': [4, 5, 6, 7],
        'up': [8, 9, 10, 11],
        'half-up': [6, 7, 8, 9, 10, 11],
        'half-bottom': [0, 1, 2, 3, 4, 5],
        'all': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]}
}

INDEX_POSITIONS_LAYERS = {
    'attn_weights': ['q_proj_weight', 'k_proj_weight', 'v_proj_weight'],
    'attn_linears': ['o_proj']
}

def evaluate_lora(args, clip_model, loader, dataset):
    clip_model.eval()
    with torch.no_grad():
        template = dataset.template[0]
        texts = [template.format(classname.replace('_', ' ')) for classname in dataset.classnames]
        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            texts = clip.tokenize(texts).cuda()
            class_embeddings = clip_model.encode_text(texts)
        text_features = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)

    acc = 0.
    tot_samples = 0
    with torch.no_grad():
        for i, (images, target) in enumerate(loader):
            images, target = images.cuda(), target.cuda()
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                image_features = clip_model.encode_image(images)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            cosine_similarity = image_features @ text_features.t()
            acc += cls_acc(cosine_similarity, target) * len(cosine_similarity)
            tot_samples += len(cosine_similarity)
    acc /= tot_samples

    return acc

def run_lora(args, clip_model, logit_scale, dataset, train_loader, val_loader, test_loader):
    VALIDATION = True

    # Textual features
    print("\nGetting textual features as CLIP's classifier.")
    textual_features = clip_classifier(dataset.classnames, dataset.template, clip_model)

    # Pre-load val features
    print("\nLoading visual features and labels from val set.")
    val_features, val_labels = pre_load_features(clip_model, val_loader)

    # Pre-load test features
    print("\nLoading visual features and labels from test set.")
    test_features, test_labels = pre_load_features(clip_model, test_loader)

    test_features = test_features.cuda()
    test_labels = test_labels.cuda()

    # Zero-shot CLIP
    clip_logits = logit_scale * test_features @ textual_features
    zs_acc = cls_acc(clip_logits, test_labels)
    print("\n**** Zero-shot CLIP's test accuracy: {:.2f}. ****\n".format(zs_acc))

    test_features = test_features.cpu()
    test_labels = test_labels.cpu()

    list_lora_layers = apply_lora(args, clip_model)
    clip_model = clip_model.cuda()

    if args.eval_only:
        load_lora(args, list_lora_layers)
        acc_test = evaluate_lora(args, clip_model, test_loader, dataset)
        print("**** Test accuracy: {:.2f}. ****\n".format(acc_test))
        return

    mark_only_lora_as_trainable(clip_model)
    total_iters = args.n_iters * args.shots

    optimizer = torch.optim.AdamW(get_lora_parameters(clip_model), weight_decay=1e-2, betas=(0.9, 0.999), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, total_iters, eta_min=1e-6)

    best_acc_val, best_acc_test = 0., 0.
    best_epoch_val = 0

    # training LoRA
    scaler = torch.cuda.amp.GradScaler()
    count_iters = 0
    finish = False
    while count_iters < total_iters:
        clip_model.train()
        acc_train = 0
        tot_samples = 0
        loss_epoch = 0.
        if args.encoder == 'vision':
            text_features = textual_features.t().half()
        for i, (images, target) in enumerate(tqdm(train_loader)):

            template = dataset.template[0]
            texts = [template.format(classname.replace('_', ' ')) for classname in dataset.classnames]
            images, target = images.cuda(), target.cuda()
            if args.encoder == 'text' or args.encoder == 'both':
                with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                    texts = clip.tokenize(texts).cuda()
                    class_embeddings = clip_model.encode_text(texts)
                text_features = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)

            if args.encoder == 'vision' or args.encoder == 'both':
                with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                    image_features = clip_model.encode_image(images)
            else:
                with torch.no_grad():
                    with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                        image_features = clip_model.encode_image(images)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            cosine_similarity = logit_scale * image_features @ text_features.t()
            loss = F.cross_entropy(cosine_similarity, target)
            acc_train += cls_acc(cosine_similarity, target) * target.shape[0]
            loss_epoch += loss.item() * target.shape[0]
            tot_samples += target.shape[0]
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)

            scaler.update()
            scheduler.step()

            count_iters += 1

            if count_iters == total_iters:
                break

        if count_iters < total_iters:
            acc_train /= tot_samples
            loss_epoch /= tot_samples
            current_lr = scheduler.get_last_lr()[0]
            print('LR: {:.6f}, Acc: {:.4f}, Loss: {:.4f}'.format(current_lr, acc_train, loss_epoch))
            # wandb.log({"lr":current_lr, "train_acc": acc_train, "train_loss": loss_epoch}, step=count_iters)

        # Eval
        if VALIDATION:
            clip_model.eval()
            acc_val = evaluate_lora(args, clip_model, val_loader, dataset)
            print("**** Val accuracy: {:.2f}. ****\n".format(acc_val))
            # wandb.log({"val_acc": acc_val}, step=count_iters)

    acc_test = evaluate_lora(args, clip_model, test_loader, dataset)
    print("**** Final test accuracy: {:.2f}. ****\n".format(acc_test))
    # wandb.log({"test_acc": acc_test})

    if args.save_path != None:
        save_lora(args, list_lora_layers)
    return

def run_lora_plus(args, clip_model, logit_scale, dataset, train_loader, val_loader, test_loader):
    VALIDATION = True

    # Textual features
    print("\nGetting textual features as CLIP's classifier.")
    textual_features = clip_classifier(dataset.classnames, dataset.template, clip_model)

    # Pre-load val features
    print("\nLoading visual features and labels from val set.")
    val_features, val_labels = pre_load_features(clip_model, val_loader)

    # Pre-load test features
    print("\nLoading visual features and labels from test set.")
    test_features, test_labels = pre_load_features(clip_model, test_loader)

    test_features = test_features.cuda()
    test_labels = test_labels.cuda()

    # Zero-shot CLIP
    clip_logits = logit_scale * test_features @ textual_features
    zs_acc = cls_acc(clip_logits, test_labels)
    print("\n**** Zero-shot CLIP's test accuracy: {:.2f}. ****\n".format(zs_acc))
    wandb.log({"zs_acc": zs_acc})

    test_features = test_features.cpu()
    test_labels = test_labels.cpu()

    list_lora_layers = apply_lora(args, clip_model)
    clip_model = clip_model.cuda()

    if args.eval_only:
        load_lora(args, list_lora_layers)
        acc_test = evaluate_lora(args, clip_model, test_loader, dataset)
        print("**** Test accuracy: {:.2f}. ****\n".format(acc_test))
        return

    mark_only_lora_as_trainable(clip_model)
    total_iters = args.n_iters * args.shots

    lora_lr_ratio = args.lr_ratio_AB  # A和B矩阵学习率的比例
    base_lr = args.lr  # 基础学习率
    weight_decay = args.weight_decay
    lora_a_params, lora_b_params = get_lora_plus_parameters(clip_model)  # 获取LoRA参数

    # 定义优化器
    optimizer = torch.optim.AdamW([
        {'params': lora_a_params, 'lr': base_lr},  # 设置LoRA矩阵A的学习率
        {'params': lora_b_params, 'lr': base_lr * lora_lr_ratio}  # 设置LoRA矩阵B的学习率
    ], weight_decay=weight_decay, betas=(0.9, 0.999))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, total_iters, eta_min=1e-6)

    best_acc_val, best_acc_test = 0., 0.
    best_epoch_val = 0

    # training LoRA
    scaler = torch.cuda.amp.GradScaler()
    count_iters = 0
    finish = False
    while count_iters < total_iters:
        clip_model.train()
        acc_train = 0
        tot_samples = 0
        loss_epoch = 0.
        if args.encoder == 'vision':
            text_features = textual_features.t().half()
        for i, (images, target) in enumerate(tqdm(train_loader)):

            template = dataset.template[0]
            texts = [template.format(classname.replace('_', ' ')) for classname in dataset.classnames]
            images, target = images.cuda(), target.cuda()
            if args.encoder == 'text' or args.encoder == 'both':
                with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                    texts = clip.tokenize(texts).cuda()
                    class_embeddings = clip_model.encode_text(texts)
                text_features = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)

            if args.encoder == 'vision' or args.encoder == 'both':
                with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                    image_features = clip_model.encode_image(images)
            else:
                with torch.no_grad():
                    with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                        image_features = clip_model.encode_image(images)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            cosine_similarity = logit_scale * image_features @ text_features.t()
            loss = F.cross_entropy(cosine_similarity, target)
            acc_train += cls_acc(cosine_similarity, target) * target.shape[0]
            loss_epoch += loss.item() * target.shape[0]
            tot_samples += target.shape[0]
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)

            scaler.update()
            scheduler.step()

            count_iters += 1

            if count_iters == total_iters:
                break

        if count_iters < total_iters:
            acc_train /= tot_samples
            loss_epoch /= tot_samples
            lr_A, lr_B = scheduler.get_last_lr()
            print('Epochs: {}/{}, LR_A: {:.6f}, LR_B: {:.6f}, Acc: {:.4f}, Loss: {:.4f}'.format(count_iters, total_iters, lr_A, lr_B, acc_train, loss_epoch))
            wandb.log({"LR_A": lr_A, "LR_B": lr_B, "train_acc": acc_train, "train_loss": loss_epoch}, step=count_iters)

        # Eval
        if VALIDATION:
            clip_model.eval()
            acc_val = evaluate_lora(args, clip_model, val_loader, dataset)
            print("**** Val accuracy: {:.2f}. ****\n".format(acc_val))
            wandb.log({"val_acc": acc_val}, step=count_iters)

    acc_test = evaluate_lora(args, clip_model, test_loader, dataset)
    print("**** Final test accuracy: {:.2f}. ****\n".format(acc_test))
    wandb.log({"test_acc": acc_test})

    if args.save_path != None:
        save_lora(args, list_lora_layers)
    return

def run_adalora(args, clip_model, logit_scale, dataset, train_loader, val_loader, test_loader):
    VALIDATION = True
    
    # Textual features
    print("\nGetting textual features as CLIP's classifier.")
    textual_features = clip_classifier(dataset.classnames, dataset.template, clip_model)

    # Pre-load val features
    print("\nLoading visual features and labels from val set.")
    val_features, val_labels = pre_load_features(clip_model, val_loader)

    # Pre-load test features
    print("\nLoading visual features and labels from test set.")
    test_features, test_labels = pre_load_features(clip_model, test_loader)

    test_features = test_features.cuda()
    test_labels = test_labels.cuda()

    # Zero-shot CLIP
    clip_logits = logit_scale * test_features @ textual_features
    zs_acc = cls_acc(clip_logits, test_labels)
    print("\n**** Zero-shot CLIP's test accuracy: {:.2f}. ****\n".format(zs_acc))

    test_features = test_features.cpu()
    test_labels = test_labels.cpu()

    list_lora_layers = apply_Adalora(args, clip_model)
    clip_model = clip_model.cuda()

    if args.eval_only:
        load_lora(args, list_lora_layers)
        acc_test = evaluate_lora(args, clip_model, test_loader, dataset)
        print("**** Test accuracy: {:.2f}. ****\n".format(acc_test))
        return

    mark_only_lora_as_trainable(clip_model)
    total_iters = args.n_iters * args.shots
    assert total_iters == args.total_step, 'total_iters should be equal to args.total_step!'

    # Initialize the RankAllocator
    rankallocator = RankAllocator(
    clip_model, lora_r=args.lora_init_rank, target_rank=args.target_rank,
    init_warmup=args.init_warmup, final_warmup=args.final_warmup, mask_interval=args.mask_interval, 
    total_step=args.total_step, beta1=args.beta1, beta2=args.beta2,
    wandb_writter=args.wandb_writter, wandb_writter_loginterval=args.log_interval
    )

    optimizer = torch.optim.AdamW(get_lora_parameters(clip_model), weight_decay=1e-2, betas=(0.9, 0.999), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, total_iters, eta_min=1e-6)

    best_acc_val, best_acc_test = 0., 0.
    best_epoch_val = 0

    # training LoRA
    scaler = torch.cuda.amp.GradScaler()
    count_iters = 0
    finish = False
    while count_iters < total_iters:
        clip_model.train()
        acc_train = 0
        tot_samples = 0
        loss_epoch = 0.
        loss_orth_epoch = 0
        if args.encoder == 'vision':
            text_features = textual_features.t().half()
        for i, (images, target) in enumerate(tqdm(train_loader)):

            template = dataset.template[0]
            texts = [template.format(classname.replace('_', ' ')) for classname in dataset.classnames]
            images, target = images.cuda(), target.cuda()
            if args.encoder == 'text' or args.encoder == 'both':
                with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                    texts = clip.tokenize(texts).cuda()
                    class_embeddings = clip_model.encode_text(texts)
                text_features = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)

            if args.encoder == 'vision' or args.encoder == 'both':
                with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                    image_features = clip_model.encode_image(images)
            else:
                with torch.no_grad():
                    with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                        image_features = clip_model.encode_image(images)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            cosine_similarity = logit_scale * image_features @ text_features.t()

            # Loss
            loss = F.cross_entropy(cosine_similarity, target)
            loss_orth = compute_orth_regu(model=clip_model, regu_weight=0.1)
            total_loss = loss + loss_orth
            loss_epoch += loss.item() * target.shape[0]
            loss_orth_epoch += loss_orth.item() * target.shape[0]
            # Accuracy
            acc_train += cls_acc(cosine_similarity, target) * target.shape[0]
            tot_samples += target.shape[0]
            # Backward
            optimizer.zero_grad()
            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            # RankAllocator
            rankallocator.update_and_mask(model=clip_model, global_step=count_iters)

            scaler.update()
            scheduler.step()

            count_iters += 1

            if count_iters == total_iters:
                break

        if count_iters < total_iters:
            acc_train /= tot_samples
            loss_epoch /= tot_samples
            loss_orth_epoch /= tot_samples
            current_lr = scheduler.get_last_lr()[0]
            print('Epochs: {}/{}, LR: {:.6f}, Acc: {:.4f}, Loss: {:.4f}, Loss_Orth: {:.4f}'.format(count_iters, total_iters, current_lr, acc_train, loss_epoch, loss_orth_epoch))
            wandb.log({"train_acc": acc_train, "train_loss": loss_epoch, "train_loss_orth": loss_orth_epoch}, step=count_iters)

        # Eval
        if VALIDATION:
            clip_model.eval()
            acc_val = evaluate_lora(args, clip_model, val_loader, dataset)
            if acc_val > best_acc_val:
                best_acc_val = acc_val
                print("**** Best val accuracy: {:.2f}. ****\n".format(best_acc_val))
                wandb.log({"Best_Val_Accuracy": best_acc_val}, step=count_iters)
            print("**** Val accuracy: {:.2f}. ****\n".format(acc_val))
            wandb.log({"val_acc": acc_val}, step=count_iters)

    acc_test = evaluate_lora(args, clip_model, test_loader, dataset)
    if acc_test > best_acc_test:
        best_acc_test = acc_test
        print("**** Best Test Accuracy: {:.2f}. ****\n".format(best_acc_test))
        wandb.log({"Best_Test_Accuracy": best_acc_test}, step=count_iters)
    print("**** Final Test Accuracy: {:.2f}. ****\n".format(acc_test))
    wandb.log({"test_acc": acc_test})

    if args.save_path != None:
        save_lora(args, list_lora_layers)
    return

def run_dylora(args, clip_model, logit_scale, dataset, train_loader, val_loader, test_loader):
    VALIDATION = True
    
    # Textual features
    print("\nGetting textual features as CLIP's classifier.")
    textual_features = clip_classifier(dataset.classnames, dataset.template, clip_model)

    # Pre-load val features
    print("\nLoading visual features and labels from val set.")
    val_features, val_labels = pre_load_features(clip_model, val_loader)

    # Pre-load test features
    print("\nLoading visual features and labels from test set.")
    test_features, test_labels = pre_load_features(clip_model, test_loader)

    test_features = test_features.cuda()
    test_labels = test_labels.cuda()

    # Zero-shot CLIP
    clip_logits = logit_scale * test_features @ textual_features
    zs_acc = cls_acc(clip_logits, test_labels)
    print("\n**** Zero-shot CLIP's test accuracy: {:.2f}. ****\n".format(zs_acc))

    test_features = test_features.cpu()
    test_labels = test_labels.cpu()

    list_lora_layers = apply_Dylora(args, clip_model)
    clip_model = clip_model.cuda()

    if args.eval_only:
        load_lora(args, list_lora_layers)
        acc_test = evaluate_lora(args, clip_model, test_loader, dataset)
        print("**** Test accuracy: {:.2f}. ****\n".format(acc_test))
        return

    mark_only_lora_as_trainable(clip_model)
    total_iters = args.n_iters * args.shots

    optimizer = torch.optim.AdamW(get_lora_parameters(clip_model), weight_decay=1e-2, betas=(0.9, 0.999), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, total_iters, eta_min=1e-6)

    best_acc_val, best_acc_test = 0., 0.
    best_epoch_val = 0

    # training LoRA
    scaler = torch.cuda.amp.GradScaler()
    count_iters = 0
    finish = False
    while count_iters < total_iters:
        # 在每次前向传播之前，随机选择一个秩值并设置
        rank = random.randint(args.min_rank, args.max_rank-1)
        for layer in clip_model.modules():
            if isinstance(layer, DyLinear):
                layer.set_rank(rank, frozen=args.frozen)

        clip_model.train()
        acc_train = 0
        tot_samples = 0
        loss_epoch = 0.
        if args.encoder == 'vision':
            text_features = textual_features.t().half()

        for i, (images, target) in enumerate(tqdm(train_loader)):

            template = dataset.template[0]
            texts = [template.format(classname.replace('_', ' ')) for classname in dataset.classnames]
            images, target = images.cuda(), target.cuda()
            if args.encoder == 'text' or args.encoder == 'both':
                with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                    texts = clip.tokenize(texts).cuda()
                    class_embeddings = clip_model.encode_text(texts)
                text_features = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)

            if args.encoder == 'vision' or args.encoder == 'both':
                with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                    image_features = clip_model.encode_image(images)
            else:
                with torch.no_grad():
                    with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                        image_features = clip_model.encode_image(images)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            cosine_similarity = logit_scale * image_features @ text_features.t()

            # Loss
            loss = F.cross_entropy(cosine_similarity, target)
            loss_epoch += loss.item() * target.shape[0]
            # Accuracy
            acc_train += cls_acc(cosine_similarity, target) * target.shape[0]
            tot_samples += target.shape[0]
            # Backward
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)

            scaler.update()
            scheduler.step()

            count_iters += 1

            if count_iters == total_iters:
                break

        if count_iters < total_iters:
            acc_train /= tot_samples
            loss_epoch /= tot_samples
            current_lr = scheduler.get_last_lr()[0]
            print('Epochs: {}/{}, LR: {:.6f}, Acc: {:.4f}, Loss: {:.4f}'.format(count_iters, total_iters, current_lr, acc_train, loss_epoch))
            wandb.log({"train_acc": acc_train, "train_loss": loss_epoch, "LR": current_lr}, step=count_iters)

        # Eval
        if VALIDATION:
            clip_model.eval()
            acc_val = evaluate_lora(args, clip_model, val_loader, dataset)
            if acc_val > best_acc_val:
                best_acc_val = acc_val
                wandb.log({"Best Val Accuracy": best_acc_val}, step=count_iters)
            print(f"Validation Accuracy: {acc_val:.4f}")
            wandb.log({"Val Accuracy": acc_val}, step=count_iters)

        acc_test = evaluate_lora(args, clip_model, test_loader, dataset)
        if acc_test > best_acc_test:
            best_acc_test = acc_test
            wandb.log({"Best Test Accuracy": best_acc_test}, step=count_iters)
        print(f"Test Accuracy: {acc_test:.4f}")
        wandb.log({"Test Accuracy": acc_test}, step=count_iters)

    if args.save_path != None:
        save_lora(args, list_lora_layers)
    return



def run_bilora(args, clip_model, logit_scale, dataset, train_loader, val_loader, test_loader):
    VALIDATION = True
    
    # Textual features
    print("\nGetting textual features as CLIP's classifier.")
    textual_features = clip_classifier(dataset.classnames, dataset.template, clip_model)

    # Pre-load val features
    print("\nLoading visual features and labels from val set.")
    val_features, val_labels = pre_load_features(clip_model, val_loader)

    # Pre-load test features
    print("\nLoading visual features and labels from test set.")
    test_features, test_labels = pre_load_features(clip_model, test_loader)

    test_features = test_features.cuda()
    test_labels = test_labels.cuda()

    # Zero-shot CLIP
    clip_logits = logit_scale * test_features @ textual_features
    zs_acc = cls_acc(clip_logits, test_labels)
    print("\n**** Zero-shot CLIP's test accuracy: {:.2f}. ****\n".format(zs_acc))

    test_features = test_features.cpu()
    test_labels = test_labels.cpu()

    list_lora_layers = apply_Bilora(args, clip_model)
    clip_model = clip_model.cuda()

    if args.eval_only:
        load_lora(args, list_lora_layers)
        acc_test = evaluate_lora(args, clip_model, test_loader, dataset)
        print("**** Test accuracy: {:.2f}. ****\n".format(acc_test))
        return

    mark_only_lora_as_trainable(clip_model)
    total_iters = args.n_iters * args.shots

    optimizer = torch.optim.AdamW(get_lora_parameters(clip_model), weight_decay=1e-2, betas=(0.9, 0.999), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, total_iters, eta_min=1e-6)

    best_acc_val, best_acc_test = 0., 0.
    best_epoch_val = 0

    # training LoRA
    scaler = torch.cuda.amp.GradScaler()
    count_iters = 0
    finish = False
    while count_iters < total_iters:
        # 在每次前向传播之前，随机选择一个秩值并设置
        # rank = random.randint(args.min_rank, args.max_rank-1)
        rank = args.max_rank-1
        # for layer in clip_model.modules():
        #     if isinstance(layer, DyLinear):
        #         layer.set_rank(rank, frozen=args.frozen)

        clip_model.train()
        acc_train = 0
        tot_samples = 0
        loss_epoch = 0.
        if args.encoder == 'vision':
            text_features = textual_features.t().half()

        for i, (images, target) in enumerate(tqdm(train_loader)):

            template = dataset.template[0]
            texts = [template.format(classname.replace('_', ' ')) for classname in dataset.classnames]
            images, target = images.cuda(), target.cuda()
            if args.encoder == 'text' or args.encoder == 'both':
                with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                    texts = clip.tokenize(texts).cuda()
                    class_embeddings = clip_model.encode_text(texts)
                text_features = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)

            if args.encoder == 'vision' or args.encoder == 'both':
                with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                    image_features = clip_model.encode_image(images)
            else:
                with torch.no_grad():
                    with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                        image_features = clip_model.encode_image(images)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            cosine_similarity = logit_scale * image_features @ text_features.t()

            # Loss
            loss = F.cross_entropy(cosine_similarity, target)
            loss_epoch += loss.item() * target.shape[0]
            # Accuracy
            acc_train += cls_acc(cosine_similarity, target) * target.shape[0]
            tot_samples += target.shape[0]
            # Backward
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)

            scaler.update()
            scheduler.step()

            count_iters += 1

            if count_iters == total_iters:
                break

        if count_iters < total_iters:
            acc_train /= tot_samples
            loss_epoch /= tot_samples
            current_lr = scheduler.get_last_lr()[0]
            print('Epochs: {}/{}, LR: {:.6f}, Acc: {:.4f}, Loss: {:.4f}'.format(count_iters, total_iters, current_lr, acc_train, loss_epoch))
            wandb.log({"train_acc": acc_train, "train_loss": loss_epoch, "LR": current_lr}, step=count_iters)

        # Eval
        if VALIDATION:
            clip_model.eval()
            acc_val = evaluate_lora(args, clip_model, val_loader, dataset)
            print(f"Val Accuracy/Rank {rank}, Validation Accuracy: {acc_val:.4f}")
            wandb.log({f"Val Accuracy/Rank {rank}": acc_val}, step=count_iters)

        acc_test = evaluate_lora(args, clip_model, test_loader, dataset)
        print(f"Test Accuracy/Rank {rank}, Validation Accuracy: {acc_test:.4f}")
        wandb.log({f"Test Accuracy/Rank {rank}": acc_test})    

    if args.save_path != None:
        save_lora(args, list_lora_layers)
    return

def run_lolora(args, clip_model, logit_scale, dataset, train_loader, val_loader, test_loader):
    VALIDATION = True
    
    # Textual features
    print("\nGetting textual features as CLIP's classifier.")
    textual_features = clip_classifier(dataset.classnames, dataset.template, clip_model)

    # Pre-load val features
    print("\nLoading visual features and labels from val set.")
    val_features, val_labels = pre_load_features(clip_model, val_loader)

    # Pre-load test features
    print("\nLoading visual features and labels from test set.")
    test_features, test_labels = pre_load_features(clip_model, test_loader)

    test_features = test_features.cuda()
    test_labels = test_labels.cuda()

    # Zero-shot CLIP
    clip_logits = logit_scale * test_features @ textual_features
    zs_acc = cls_acc(clip_logits, test_labels)
    print("\n**** Zero-shot CLIP's test accuracy: {:.2f}. ****\n".format(zs_acc))

    test_features = test_features.cpu()
    test_labels = test_labels.cpu()

    list_lora_layers = apply_Lolora(args, clip_model)
    clip_model = clip_model.cuda()

    if args.eval_only:
        load_lora(args, list_lora_layers)
        acc_test = evaluate_lora(args, clip_model, test_loader, dataset)
        print("**** Test accuracy: {:.2f}. ****\n".format(acc_test))
        return

    mark_only_lora_as_trainable(clip_model)
    total_iters = args.n_iters * args.shots

    # 计算额外的参数量和clip模型所有可训练的参数量
    num_lora_params = sum(p.numel() for p in get_lora_parameters(clip_model))
    num_trainable_params = sum(p.numel() for p in clip_model.parameters() if p.requires_grad)
    print(f"LoRA parameters: {num_lora_params / 1e6:.2f}M, Trainable parameters: {num_trainable_params / 1e6:.2f}M.")
    wandb.log({"LoRA parameters": num_lora_params / 1e6, "Trainable parameters": num_trainable_params / 1e6})

    optimizer = torch.optim.AdamW(get_lora_parameters(clip_model), weight_decay=1e-2, betas=(0.9, 0.999), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, total_iters, eta_min=1e-6)

    best_acc_val, best_acc_test = 0., 0.
    best_epoch_val = 0

    # training LoRA
    scaler = torch.cuda.amp.GradScaler()
    count_iters = 0
    finish = False
    while count_iters < total_iters:
        # 在每次前向传播之前，随机选择一个秩值并设置
        rank = random.randint(args.min_rank, args.max_rank-1)
        for layer in clip_model.modules():
            if isinstance(layer, LoLinear):
                layer.set_rank(rank, frozen=args.frozen)

        clip_model.train()
        acc_train = 0
        tot_samples = 0
        loss_epoch = 0.
        if args.encoder == 'vision':
            text_features = textual_features.t().half()

        for i, (images, target) in enumerate(tqdm(train_loader)):

            template = dataset.template[0]
            texts = [template.format(classname.replace('_', ' ')) for classname in dataset.classnames]
            images, target = images.cuda(), target.cuda()
            if args.encoder == 'text' or args.encoder == 'both':
                with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                    texts = clip.tokenize(texts).cuda()
                    class_embeddings = clip_model.encode_text(texts)
                text_features = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)

            if args.encoder == 'vision' or args.encoder == 'both':
                with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                    image_features = clip_model.encode_image(images)
            else:
                with torch.no_grad():
                    with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                        image_features = clip_model.encode_image(images)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            cosine_similarity = logit_scale * image_features @ text_features.t()

            # Loss
            loss = F.cross_entropy(cosine_similarity, target)
            loss_epoch += loss.item() * target.shape[0]
            # Accuracy
            acc_train += cls_acc(cosine_similarity, target) * target.shape[0]
            tot_samples += target.shape[0]
            # Backward
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)

            scaler.update()
            scheduler.step()

            count_iters += 1

            if count_iters == total_iters:
                break

        if count_iters < total_iters:
            acc_train /= tot_samples
            loss_epoch /= tot_samples
            current_lr = scheduler.get_last_lr()[0]
            print('Epochs: {}/{}, LR: {:.6f}, Acc: {:.4f}, Loss: {:.4f}'.format(count_iters, total_iters, current_lr, acc_train, loss_epoch))
            wandb.log({"train_acc": acc_train, "train_loss": loss_epoch, "LR": current_lr}, step=count_iters)

        # Eval
        if VALIDATION:
            clip_model.eval()
            acc_val = evaluate_lora(args, clip_model, val_loader, dataset)
            if acc_val > best_acc_val:
                best_acc_val = acc_val
                wandb.log({"Best Val Accuracy": best_acc_val}, step=count_iters)
            print(f"Validation Accuracy: {acc_val:.4f}")
            wandb.log({"Val Accuracy": acc_val}, step=count_iters)

        acc_test = evaluate_lora(args, clip_model, test_loader, dataset)
        if acc_test > best_acc_test:
            best_acc_test = acc_test
            wandb.log({"Best Test Accuracy": best_acc_test}, step=count_iters)
        print(f"Test Accuracy: {acc_test:.4f}")
        wandb.log({"Test Accuracy": acc_test}, step=count_iters)

    if args.save_path != None:
        save_lora(args, list_lora_layers)
    return



def run_lolora_v2(args, clip_model, logit_scale, dataset, train_loader, val_loader, test_loader):
    VALIDATION = True
    
    # Textual features
    print("\nGetting textual features as CLIP's classifier.")
    textual_features = clip_classifier(dataset.classnames, dataset.template, clip_model)

    # Pre-load val features
    print("\nLoading visual features and labels from val set.")
    val_features, val_labels = pre_load_features(clip_model, val_loader)

    # Pre-load test features
    print("\nLoading visual features and labels from test set.")
    test_features, test_labels = pre_load_features(clip_model, test_loader)

    test_features = test_features.cuda()
    test_labels = test_labels.cuda()

    # Zero-shot CLIP
    clip_logits = logit_scale * test_features @ textual_features
    zs_acc = cls_acc(clip_logits, test_labels)
    print("\n**** Zero-shot CLIP's test accuracy: {:.2f}. ****\n".format(zs_acc))

    test_features = test_features.cpu()
    test_labels = test_labels.cpu()

    list_lora_layers = apply_Lolora_v2(args, clip_model)
    clip_model = clip_model.cuda()

    if args.eval_only:
        load_lora(args, list_lora_layers)
        acc_test = evaluate_lora(args, clip_model, test_loader, dataset)
        print("**** Test accuracy: {:.2f}. ****\n".format(acc_test))
        return

    mark_only_lora_as_trainable(clip_model)
    total_iters = args.n_iters * args.shots

    # 计算额外的参数量和clip模型所有可训练的参数量
    num_lora_params = sum(p.numel() for p in get_lora_parameters(clip_model))
    num_trainable_params = sum(p.numel() for p in clip_model.parameters() if p.requires_grad)
    print(f"LoRA Params: {num_lora_params / 1e6:.2f}M, Trainable Params: {num_trainable_params / 1e6:.2f}M.")
    wandb.log({"Params/LoRA Params": num_lora_params / 1e6, "Params/Trainable Params": num_trainable_params / 1e6})

    optimizer = torch.optim.AdamW(get_lora_parameters(clip_model), weight_decay=1e-2, betas=(0.9, 0.999), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, total_iters, eta_min=1e-6)

    best_acc_val, best_acc_test = 0., 0.
    best_epoch_val = 0

    # training LoRA
    scaler = torch.cuda.amp.GradScaler()
    count_iters = 0
    epochs = 0
    finish = False
    while count_iters < total_iters:
        # 在每次前向传播之前，随机选择一个秩值并设置
        for n, layer in clip_model.named_modules():
            if isinstance(layer, LoLinear_v2):
                layer.set_rank(frozen=args.frozen)
                wandb.log({"Ranknum/%s"%n: layer.get_rank()}, step=count_iters)

        clip_model.train()
        acc_train = 0
        tot_samples = 0
        loss_epoch = 0.
        if args.encoder == 'vision':
            text_features = textual_features.t().half()

        for i, (images, target) in enumerate(tqdm(train_loader)):

            template = dataset.template[0]
            texts = [template.format(classname.replace('_', ' ')) for classname in dataset.classnames]
            images, target = images.cuda(), target.cuda()
            if args.encoder == 'text' or args.encoder == 'both':
                with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                    texts = clip.tokenize(texts).cuda()
                    class_embeddings = clip_model.encode_text(texts)
                text_features = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)

            if args.encoder == 'vision' or args.encoder == 'both':
                with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                    image_features = clip_model.encode_image(images)
            else:
                with torch.no_grad():
                    with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                        image_features = clip_model.encode_image(images)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            cosine_similarity = logit_scale * image_features @ text_features.t()

            # Loss
            loss = F.cross_entropy(cosine_similarity, target)
            loss_epoch += loss.item() * target.shape[0]
            # Accuracy
            acc_train += cls_acc(cosine_similarity, target) * target.shape[0]
            tot_samples += target.shape[0]
            # Backward
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)

            scaler.update()
            scheduler.step()

            count_iters += 1

            if count_iters == total_iters:
                break

        if count_iters < total_iters:
            acc_train /= tot_samples
            loss_epoch /= tot_samples
            current_lr = scheduler.get_last_lr()[0]
            print('Epochs: {}/{}, LR: {:.6f}, Acc: {:.4f}, Loss: {:.4f}'.format(count_iters, total_iters, current_lr, acc_train, loss_epoch))
            wandb.log({"train_acc": acc_train, "loss/train_loss": loss_epoch, "LR": current_lr}, step=count_iters)

        # Eval
        if VALIDATION:
            clip_model.eval()
            acc_val = evaluate_lora(args, clip_model, val_loader, dataset)
            if acc_val > best_acc_val:
                best_acc_val = acc_val
                wandb.log({"Optical Calculation/Best Val Accuracy": best_acc_val}, step=count_iters)
            print(f"Validation Accuracy: {acc_val:.4f}")
            wandb.log({"Val Accuracy": acc_val}, step=count_iters)

        acc_test = evaluate_lora(args, clip_model, test_loader, dataset)
        if acc_test > best_acc_test:
            best_acc_test = acc_test
            wandb.log({"Optical Calculation/Best Test Accuracy": best_acc_test}, step=count_iters)
        print(f"Test Accuracy: {acc_test:.4f}")
        wandb.log({"Test Accuracy": acc_test}, step=count_iters)

    epochs += 1
    # 设置interval
    if epochs % args.interval == 0:
        for module in clip_model.modules():
            if isinstance(module, LoLinear_v2):
                if module.if_prune:
                    module.prune()

    if args.save_path != None:
        save_lora(args, list_lora_layers)
    return


def find_layers(module, layers=[nn.Linear], name=''):
    """
    Recursively find the layers of a certain type in a module.

    Args:
        module (nn.Module): PyTorch module.
        layers (list): List of layer types to find.
        name (str): Name of the module.

    Returns:
        dict: Dictionary of layers of the given type(s) within the module.
    """
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_layers(
            child, layers=layers, name=name + '.' + name1 if name != '' else name1
        ))
    return res

def cal_layers_lora(args, clip_model, train_loader, dataset):
    '''
    先获取每一层的输入、输出和每一层的权重，然后计算每一层的lod值
    '''
    all_text_layers_inputs = {}
    all_text_layers_weights = {}
    all_vision_layers_inputs = {}
    all_vision_layers_weights = {}
    lod_values = {}

    def get_text_input_hook(name):
        def hook_fn(module, input, output):
            all_text_layers_inputs[name] = input[0].detach()
        return hook_fn
    
    def get_vision_input_hook(name):
        def hook_fn(module, input, output):
            all_vision_layers_inputs[name] = input[0].detach()
        return hook_fn
    
    # 为每一层的线性层添加钩子，并获取权重
    hooks = []
    for i in range(len(clip_model.transformer.resblocks)):
        text_resblock = clip_model.transformer.resblocks[i]
        vision_resblock = clip_model.visual.transformer.resblocks[i]
        text_subset_layers = find_layers(text_resblock, layers=[nn.Linear])
        vision_subset_layers = find_layers(vision_resblock, layers=[nn.Linear])
        
        for name, layer in text_subset_layers.items():
            hook = layer.register_forward_hook(get_text_input_hook(f"resblock_{i}_{name}"))
            hooks.append(hook)
            all_text_layers_weights[f"resblock_{i}_{name}"] = layer.weight.detach()
        for name, layer in vision_subset_layers.items():
            hook = layer.register_forward_hook(get_vision_input_hook(f"resblock_{i}_{name}"))
            hooks.append(hook)
            all_vision_layers_weights[f"resblock_{i}_{name}"] = layer.weight.detach()

    # 遍历数据集以执行前向传播并收集输入
    clip_model.eval()  # 确保模型在推理模式下
    for batch_idx, (images, target) in enumerate(tqdm(train_loader)):
        images, target = images.cuda(), target.cuda()
        with torch.no_grad():
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                if args.encoder == 'text' or args.encoder == 'both':
                    # 随机选择一个 classname
                    classnames = random.sample(dataset.classnames, k=1)
                    clip_classifier(classnames, dataset.template, clip_model)
                if args.encoder == 'vision' or args.encoder == 'both':
                    image_features = clip_model.encode_image(images)
        break  # 只执行一次前向传播

    # 分别计算text和vision的lod值
    def calculate_lod(all_layers_inputs, all_layers_weights):
        lod_values = {}
        for name, inputs in all_layers_inputs.items():
            weights = all_layers_weights[name]
            
            # 计算每个权重的异常值得分
            inputs = inputs.reshape(-1, inputs.shape[-1])
            A_ij = torch.abs(weights) * torch.linalg.vector_norm(inputs, ord=2, dim=0, keepdim=True)

            # 计算阈值 M * 平均异常值得分
            mean_A_l = A_ij.mean().item()
            M = args.lod_M  # 假设 args.threshold 为 LOD 的阈值因子

            # 计算 LOD 值
            lod_value = (A_ij > M * mean_A_l).float().mean().item()
            lod_values[name] = lod_value
        
        return lod_values

    # 计算 text 和 vision 的 LOD 值
    text_lod_values = calculate_lod(all_text_layers_inputs, all_text_layers_weights)
    vision_lod_values = calculate_lod(all_vision_layers_inputs, all_vision_layers_weights)

    # 将两者的结果存入 lod_values 中
    lod_values['text'] = text_lod_values
    lod_values['vision'] = vision_lod_values

    # 移除钩子
    for hook in hooks:
        hook.remove()

    # 合并layers的值
    def merge_resblocks(data_dict):
        merged_text_data = {}
        merged_vision_data = {}

        # 合并text数据
        for key, value in data_dict['text'].items():
            layer = '_'.join(key.split('_')[:2])  # 获取resblock层的名称，例如resblock_0_attn
            if layer not in merged_text_data:
                merged_text_data[layer] = 0
            merged_text_data[layer] += value

        # 合并vision数据
        for key, value in data_dict['vision'].items():
            layer = '_'.join(key.split('_')[:2])  # 获取resblock层的名称，例如resblock_0_attn
            if layer not in merged_vision_data:
                merged_vision_data[layer] = 0
            merged_vision_data[layer] += value

        data = {'text': merged_text_data, 'vision': merged_vision_data}
        # 将data保存到文件中
        import pickle
        with open('merged_data.pkl', 'wb') as f:
            pickle.dump(data, f)

        return data

    lod_values = merge_resblocks(lod_values)

    # 根据text和vision每层的lod分布计算稀疏度
    def standardize_array(data, S, lambda_value):
        """
        标准化数组，使其均值为 S 并且范围在 [S-lambda, S+lambda] 之间。

        参数:
        - data: 需要标准化的原始数组（列表或NumPy数组）
        - S: 目标均值
        - lambda_value: 范围的参数

        返回:
        - 标准化后的数组（列表形式）
        """
        # 将数据转换为Tensor
        data = torch.tensor(data, dtype=torch.float32)

        # 计算原始数组的均值和标准差
        mean = torch.mean(data)
        std = torch.std(data)

        # 标准化到零均值和单位标准差
        normalized_data = (data - mean) / std

        # 缩放到 [S-lambda, S+lambda] 范围
        scaling_factor = lambda_value / torch.max(torch.abs(normalized_data))
        scaled_data = normalized_data * scaling_factor + S

        # 返回为列表形式
        return scaled_data.tolist()
    
    # 计算每一层的稀疏度
    S = args.lod_S  # 目标稀疏度
    lambda_ = args.lod_lambda  # 超参数，用于控制层间稀疏性的波动范围
    text_lora_ratio = standardize_array(list(lod_values['text'].values()), S=S, lambda_value=lambda_)
    vision_lora_ratio = standardize_array(list(lod_values['vision'].values()), S=S, lambda_value=lambda_)

    return text_lora_ratio, vision_lora_ratio

def get_inps_ops_weights(args, clip_model, train_loader, dataset):
    '''
    获取clip每一个线性层的输入、输出和权重
    '''
    all_text_layers_inputs = {}
    all_text_layers_outputs = {}
    all_text_layers_weights = {}
    all_vision_layers_inputs = {}
    all_vision_layers_outputs = {}
    all_vision_layers_weights = {}

    def get_hook(name, is_text):
        def hook_fn(module, input, output):
            if is_text:
                all_text_layers_inputs[name] = input[0].detach()
                all_text_layers_outputs[name] = F.linear(input[0].detach(), module.weight.detach())
                all_text_layers_weights[name] = module.weight.detach()
            else:
                all_vision_layers_inputs[name] = input[0].detach()
                all_vision_layers_outputs[name] = F.linear(input[0].detach(), module.weight.detach())
                all_vision_layers_weights[name] = module.weight.detach()
        return hook_fn

    def add_hooks_to_model(model, prefix, is_text):
        hooks = []
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):  # 添加其他需要的层类型
                full_name = f"{prefix}.{name}"
                hook = module.register_forward_hook(get_hook(full_name, is_text))
                hooks.append(hook)
        return hooks

    # 为文本编码器添加hooks
    text_hooks = add_hooks_to_model(clip_model.transformer, "text", True)

    # 为视觉编码器添加hooks
    vision_hooks = add_hooks_to_model(clip_model.visual.transformer, "vision", False)

    hooks = text_hooks + vision_hooks

    # 遍历数据集以执行前向传播并收集输入
    clip_model.eval()  # 确保模型在推理模式下
    for batch_idx, (images, target) in enumerate(tqdm(train_loader)):
        images, target = images.cuda(), target.cuda()
        with torch.no_grad():
            with torch.amp.autocast(device_type="cuda", dtype=torch.float32):
                if args.encoder == 'text' or args.encoder == 'both':
                    # 随机选择一个 classname
                    classnames = random.sample(dataset.classnames, k=1)
                    clip_classifier(classnames, dataset.template, clip_model)
                if args.encoder == 'vision' or args.encoder == 'both':
                    clip_model.encode_image(images)
        break  # 只执行一次前向传播

    # 移除钩子
    for hook in hooks:
        hook.remove()
    
    def convert_to_float32(d):
        return {k: v.float() for k, v in d.items()}

    all_text_layers_inputs = convert_to_float32(all_text_layers_inputs)
    all_text_layers_outputs = convert_to_float32(all_text_layers_outputs)
    all_text_layers_weights = convert_to_float32(all_text_layers_weights)
    all_vision_layers_inputs = convert_to_float32(all_vision_layers_inputs)
    all_vision_layers_outputs = convert_to_float32(all_vision_layers_outputs)
    all_vision_layers_weights = convert_to_float32(all_vision_layers_weights)

    return all_text_layers_inputs, all_text_layers_outputs, all_text_layers_weights, all_vision_layers_inputs, all_vision_layers_outputs, all_vision_layers_weights

def cal_W_prime(args, inputs_dict, weights_dict, outputs_dict):
    '''
    通过asvd方法计算每一个线性层的W'和缩放矩阵S
    '''
    W_prime_dict = {}
    S_dict = {}
    alpha = args.alpha if args.alpha is not None else 0.5  # 假设alpha为0.5，可以根据需要调整

    for layer_name in weights_dict.keys():
        W = weights_dict[layer_name] # shape: out, in
        X = inputs_dict[layer_name]  # shape: ..., in

        # 计算缩放矩阵 S
        # in, in
        S = torch.diag(torch.mean(torch.abs(X.view(-1, W.size(1))), dim=0, dtype=W.dtype) ** alpha)

        # 缩放权重矩阵 W'
        W_prime = W @ S

        # 保留W_prime和S
        W_prime_dict[layer_name] = W_prime
        S_dict[layer_name] = S
        
    return W_prime_dict, S_dict

def cal_error(args, inputs_dict, W_prime_dict, S_dict, weights_dict):
    '''
    计算ASVD和SVD的输出误差
    '''
    error = 0
    count = 0
    for layer_name in W_prime_dict.keys():
        if any(param+'_proj' in layer_name for param in args.params):
            W_prime = W_prime_dict[layer_name] # shape: out, in
            S = S_dict[layer_name] # shape: in, in

            # 对缩放后的矩阵 W' 进行奇异值分解
            U, Sigma, Vt = torch.linalg.svd(W_prime, full_matrices=False)

            # 保留前 k 个奇异值
            k = args.r
            U_k = U[:, :k]
            Sigma_k = torch.diag(Sigma[:k])
            V_k = Vt[:k, :]

            # 计算通过W'重建的输出误差
            X = inputs_dict[layer_name]
            W = weights_dict[layer_name]
            delta_W = U_k @ Sigma_k @ V_k @ torch.inverse(S) - W
            delta_Y = F.linear(X, delta_W)
            error += torch.norm(delta_Y, p=2)
            count += 1
        else:
            continue

    error /= count

    # 计算通过svd分解原weight的误差
    original_error = 0
    count = 0
    for layer_name in W_prime_dict.keys():
        if any(param+'_proj' in layer_name for param in args.params):
            W = weights_dict[layer_name] # shape: out, in
            X = inputs_dict[layer_name] # shape: ..., in
            # 对原权重矩阵 W 进行奇异值分解
            U, Sigma, Vt = torch.linalg.svd(W, full_matrices=False)
            # 保留前 k 个奇异值
            k = args.r
            U_k = U[:, :k]
            Sigma_k = torch.diag(Sigma[:k])
            V_k = Vt[:k, :]
            # 计算输出误差
            delta_W = U_k @ Sigma_k @ V_k - W
            delta_Y = F.linear(X, delta_W)
            original_error += torch.norm(delta_Y, p=2)
            count += 1
        else:
            continue

    # 计算平均误差
    original_error /= count

    # 计算随机初始化相比原weight的误差
    random_error = 0
    count = 0
    import math
    for layer_name in W_prime_dict.keys():
        if any(param+'_proj' in layer_name for param in args.params):
            W = weights_dict[layer_name] # shape: out, in
            X = inputs_dict[layer_name] # shape: ..., in
            # 对原权重矩阵 W 进行奇异值分解
            U, Sigma, Vt = torch.linalg.svd(W, full_matrices=False)
            # 保留前 k 个奇异值
            k = args.r
            U_k = U[:, :k]
            Sigma_k = torch.diag(Sigma[:k])
            V_k = Vt[:k, :]
            # 对U, Sigma, Vt进行kaiming_uniform初始化
            nn.init.kaiming_uniform_(U_k, a=math.sqrt(5))
            nn.init.kaiming_uniform_(V_k, a=math.sqrt(5))
            nn.init.kaiming_uniform_(Sigma_k, a=math.sqrt(5))
            # 计算输出误差
            delta_W = U_k @ Sigma_k @ V_k - W
            delta_Y = F.linear(X, delta_W)
            random_error += torch.norm(delta_Y, p=2)
            count += 1
        else:
            continue

    random_error /= count


    return error, original_error, random_error
        
def run_warmup_(args, clip_model, dataset, train_loader):
    # 替换规范原有的attn层
    list_linear_layers = apply_attn(args, clip_model)
    
    clip_model = clip_model.cuda()
    if args.apply_owl:
        # 计算clip_lora每一层的lod
        text_ratios, vision_ratios = cal_layers_lora(args, clip_model, train_loader, dataset)
    
    return text_ratios, vision_ratios

def run_warmup_asvd(args, clip_model, dataset, train_loader):
    """
    run_warmup_asvd
    ----

    该函数用于在训练前对模型进行预计算，计算每一层的加权Weight和缩放矩阵S。

    返回:
    - text_W_prime_dict: 文本层的加权Weight字典
    - text_S_dict: 文本层的缩放矩阵S字典
    - vision_W_prime_dict: 视觉层的加权Weight字典
    - vision_S_dict: 视觉层的缩放矩阵S字典
    """
    # 规范原有的attn层
    list_linear_layers = apply_attn(args, clip_model)
    
    clip_model = clip_model.cuda()

    # 获取clip_lora每一线性层的输入、输出和权重。按一个batch进行计算
    all_text_layers_inputs, all_text_layers_outputs, all_text_layers_weights, all_vision_layers_inputs, all_vision_layers_outputs, all_vision_layers_weights = get_inps_ops_weights(args, clip_model, train_loader, dataset)

    # 通过asvd方法计算每一层的W'和缩放矩阵S
    text_W_prime_dict, text_S_dict = cal_W_prime(args, all_text_layers_inputs, all_text_layers_weights, all_text_layers_outputs)
    vision_W_prime_dict, vision_S_dict = cal_W_prime(args, all_vision_layers_inputs, all_vision_layers_weights, all_vision_layers_outputs)

    # 计算输出误差
    clip_text_error, text_original_error, text_random_error = cal_error(args, all_text_layers_inputs, text_W_prime_dict, text_S_dict, all_text_layers_weights)
    clip_vision_error, vision_original_error, vision_random_error = cal_error(args, all_vision_layers_inputs, vision_W_prime_dict, vision_S_dict, all_vision_layers_weights)

    print("可能影响误差的因素：r={}, batch_size={}, params={}".format(args.r, args.batch_size, args.params))
    print(f"asvd后，对比原输出CLIP-text的平均误差: {clip_text_error:.4f} svd后，对比原输出CLIP-Text的平均误差: {text_original_error:.4f} 随机初始化后，对比原输出CLIP-Text的平均误差: {text_random_error:.4f}")
    print(f"asvd后，对比原输出CLIP-vision的平均误差: {clip_vision_error:.4f} svd后，对比原输出CLIP-Vision的平均误差: {vision_original_error:.4f} 随机初始化后，对比原输出CLIP-Vision的平均误差: {vision_random_error:.4f}")
    wandb.log({
        "误差/CLIP-text平均误差(ASVD) delta_Y = norm(Y - Y_W')": clip_text_error, 
        "误差/CLIP-text平均误差(SVD) delta_Y = norm(Y - Y_W)": text_original_error,
        "误差/CLIP-text平均误差(随机初始化) delta_Y = norm(Y - Y_W)": text_random_error,
        "误差/CLIP-vision平均误差(ASVD) delta_Y = norm(Y - Y_W')": clip_vision_error,
        "误差/CLIP-vision平均误差(SVD) delta_Y = norm(Y - Y_W)": vision_original_error,
        "误差/CLIP-vision平均误差(随机初始化) delta_Y = norm(Y - Y_W)": vision_random_error,
    })
    clip_model = clip_model.cpu()

    return text_W_prime_dict, text_S_dict, vision_W_prime_dict, vision_S_dict

def run_lolora_v3(args, clip_model, logit_scale, dataset, train_loader, val_loader, test_loader, text_ratios, vision_ratios):
    VALIDATION = True
    
    # Textual features
    print("\nGetting textual features as CLIP's classifier.")
    textual_features = clip_classifier(dataset.classnames, dataset.template, clip_model)

    # Pre-load val features
    print("\nLoading visual features and labels from val set.")
    val_features, val_labels = pre_load_features(clip_model, val_loader)

    # Pre-load test features
    print("\nLoading visual features and labels from test set.")
    test_features, test_labels = pre_load_features(clip_model, test_loader)

    test_features = test_features.cuda()
    test_labels = test_labels.cuda()

    # Zero-shot CLIP
    clip_logits = logit_scale * test_features @ textual_features
    zs_acc = cls_acc(clip_logits, test_labels)
    print("\n**** Zero-shot CLIP's test accuracy: {:.2f}. ****\n".format(zs_acc))

    test_features = test_features.cpu()
    test_labels = test_labels.cpu()
    
    list_linear_layers = apply_Lolora_v3(args, clip_model, text_ratios, vision_ratios)

    clip_model = clip_model.cuda()

    if args.eval_only:
        load_lora(args, list_linear_layers)
        acc_test = evaluate_lora(args, clip_model, test_loader, dataset)
        print("**** Test accuracy: {:.2f}. ****\n".format(acc_test))
        return

    mark_only_lora_as_trainable(clip_model)
    total_iters = args.n_iters * args.shots

    # 计算额外的参数量和clip模型所有可训练的参数量
    num_lora_params = sum(p.numel() for p in get_lora_parameters(clip_model))
    num_trainable_params = sum(p.numel() for p in clip_model.parameters() if p.requires_grad)
    print(f"LoRA Params: {num_lora_params / 1e6:.2f}M, Trainable Params: {num_trainable_params / 1e6:.2f}M.")
    wandb.log({"LoRA Params": num_lora_params / 1e6, "Trainable Params": num_trainable_params / 1e6})

    optimizer = torch.optim.AdamW(get_lora_parameters(clip_model), weight_decay=1e-2, betas=(0.9, 0.999), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, total_iters, eta_min=1e-6)

    best_acc_val, best_acc_test = 0., 0.
    best_epoch_val = 0

    # training LoRA
    scaler = torch.cuda.amp.GradScaler()
    count_iters = 0
    epochs = 0
    finish = False
    while count_iters < total_iters:
        # 在每次前向传播之前，随机选择一个秩值并设置
        for n, layer in clip_model.named_modules():
            if isinstance(layer, LoLinear_v2):
                layer.set_rank(frozen=args.frozen)
                wandb.log({"Ranknum/%s"%n: layer.get_rank()}, step=count_iters)

        clip_model.train()
        acc_train = 0
        tot_samples = 0
        loss_epoch = 0.
        if args.encoder == 'vision':
            text_features = textual_features.t().half()

        for i, (images, target) in enumerate(tqdm(train_loader)):

            template = dataset.template[0]
            texts = [template.format(classname.replace('_', ' ')) for classname in dataset.classnames]
            images, target = images.cuda(), target.cuda()
            if args.encoder == 'text' or args.encoder == 'both':
                with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                    texts = clip.tokenize(texts).cuda()
                    class_embeddings = clip_model.encode_text(texts)
                text_features = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)

            if args.encoder == 'vision' or args.encoder == 'both':
                with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                    image_features = clip_model.encode_image(images)
            else:
                with torch.no_grad():
                    with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                        image_features = clip_model.encode_image(images)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            cosine_similarity = logit_scale * image_features @ text_features.t()

            # Loss
            loss = F.cross_entropy(cosine_similarity, target)
            loss_epoch += loss.item() * target.shape[0]
            # Accuracy
            acc_train += cls_acc(cosine_similarity, target) * target.shape[0]
            tot_samples += target.shape[0]
            # Backward
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)

            scaler.update()
            scheduler.step()

            count_iters += 1

            if count_iters == total_iters:
                break

        if count_iters < total_iters:
            acc_train /= tot_samples
            loss_epoch /= tot_samples
            current_lr = scheduler.get_last_lr()[0]
            print('Epochs: {}/{}, LR: {:.6f}, Acc: {:.4f}, Loss: {:.4f}'.format(count_iters, total_iters, current_lr, acc_train, loss_epoch))
            wandb.log({"train_acc": acc_train, "train_loss": loss_epoch, "LR": current_lr}, step=count_iters)

        # Eval
        if VALIDATION:
            clip_model.eval()
            acc_val = evaluate_lora(args, clip_model, val_loader, dataset)
            if acc_val > best_acc_val:
                best_acc_val = acc_val
                wandb.log({"Best Val Accuracy": best_acc_val}, step=count_iters)
            print(f"Validation Accuracy: {acc_val:.4f}")
            wandb.log({"Val Accuracy": acc_val}, step=count_iters)

        acc_test = evaluate_lora(args, clip_model, test_loader, dataset)
        if acc_test > best_acc_test:
            best_acc_test = acc_test
            wandb.log({"Best Test Accuracy": best_acc_test}, step=count_iters)
        print(f"Test Accuracy: {acc_test:.4f}")
        wandb.log({"Test Accuracy": acc_test}, step=count_iters)

        epochs += 1

    if args.save_path != None:
        save_lora(args, list_linear_layers)
    return

def run_dora(args, clip_model, logit_scale, dataset, train_loader, val_loader, test_loader):
    VALIDATION = True
    
    # Textual features
    print("\nGetting textual features as CLIP's classifier.")
    textual_features = clip_classifier(dataset.classnames, dataset.template, clip_model)

    # Pre-load val features
    print("\nLoading visual features and labels from val set.")
    val_features, val_labels = pre_load_features(clip_model, val_loader)

    # Pre-load test features
    print("\nLoading visual features and labels from test set.")
    test_features, test_labels = pre_load_features(clip_model, test_loader)

    test_features = test_features.cuda()
    test_labels = test_labels.cuda()

    # Zero-shot CLIP
    clip_logits = logit_scale * test_features @ textual_features
    zs_acc = cls_acc(clip_logits, test_labels)
    print("\n**** Zero-shot CLIP's test accuracy: {:.2f}. ****\n".format(zs_acc))

    test_features = test_features.cpu()
    test_labels = test_labels.cpu()

    list_lora_layers = apply_Dora(args, clip_model)
    clip_model = clip_model.cuda()

    if args.eval_only:
        load_lora(args, list_lora_layers)
        acc_test = evaluate_lora(args, clip_model, test_loader, dataset)
        print("**** Test accuracy: {:.2f}. ****\n".format(acc_test))
        return

    mark_only_lora_as_trainable(clip_model)
    total_iters = args.n_iters * args.shots

    optimizer = torch.optim.AdamW(get_lora_parameters(clip_model), weight_decay=args.weight_decay, betas=(0.9, 0.999), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, total_iters, eta_min=1e-6)

    # 计算可训练参数量
    trainable_params = sum(p.numel() for p in clip_model.parameters() if p.requires_grad)
    print(f"可训练参数量: {trainable_params/ 1e6:.2f}M")
    wandb.log({"Params/Trainable Params": trainable_params/ 1e6})

    best_acc_val, best_acc_test = 0., 0.
    best_epoch_val = 0

    # training LoRA
    scaler = torch.cuda.amp.GradScaler()
    count_iters = 0
    finish = False
    while count_iters < total_iters:

        clip_model.train()
        acc_train = 0
        tot_samples = 0
        loss_epoch = 0.
        if args.encoder == 'vision':
            text_features = textual_features.t().half()

        for i, (images, target) in enumerate(tqdm(train_loader)):

            template = dataset.template[0]
            texts = [template.format(classname.replace('_', ' ')) for classname in dataset.classnames]
            images, target = images.cuda(), target.cuda()
            if args.encoder == 'text' or args.encoder == 'both':
                with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                    texts = clip.tokenize(texts).cuda()
                    class_embeddings = clip_model.encode_text(texts)
                text_features = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)

            if args.encoder == 'vision' or args.encoder == 'both':
                with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                    image_features = clip_model.encode_image(images)
            else:
                with torch.no_grad():
                    with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                        image_features = clip_model.encode_image(images)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            cosine_similarity = logit_scale * image_features @ text_features.t()

            # Loss
            loss = F.cross_entropy(cosine_similarity, target)
            loss_epoch += loss.item() * target.shape[0]
            # Accuracy
            acc_train += cls_acc(cosine_similarity, target) * target.shape[0]
            tot_samples += target.shape[0]
            # Backward
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)

            scaler.update()
            scheduler.step()

            count_iters += 1

            if count_iters == total_iters:
                break

        if count_iters < total_iters:
            acc_train /= tot_samples
            loss_epoch /= tot_samples
            current_lr = scheduler.get_last_lr()[0]
            print('Epochs: {}/{}, LR: {:.6f}, Acc: {:.4f}, Loss: {:.4f}'.format(count_iters, total_iters, current_lr, acc_train, loss_epoch))
            wandb.log({"train_acc": acc_train, "loss/train_loss": loss_epoch, "LR": current_lr}, step=count_iters)

        # Eval
        if VALIDATION:
            acc_val = evaluate_lora(args, clip_model, val_loader, dataset)
            if acc_val > best_acc_val:
                best_acc_val = acc_val
                wandb.log({"Optical Calculation/Best Val Accuracy": best_acc_val}, step=count_iters)
            print(f"Validation Accuracy: {acc_val:.4f}")
            wandb.log({"Val Accuracy": acc_val}, step=count_iters)

        acc_test = evaluate_lora(args, clip_model, test_loader, dataset)
        if acc_test > best_acc_test:
            best_acc_test = acc_test
            wandb.log({"Optical Calculation/Best Test Accuracy": best_acc_test}, step=count_iters)
        print(f"Test Accuracy: {acc_test:.4f}")
        wandb.log({"Test Accuracy": acc_test}, step=count_iters)

    if args.save_path != None:
        save_lora(args, list_lora_layers)
    return

def run_dora_v2(args, clip_model, logit_scale, dataset, train_loader, val_loader, test_loader):
    VALIDATION = True
    
    # Textual features
    print("\nGetting textual features as CLIP's classifier.")
    textual_features = clip_classifier(dataset.classnames, dataset.template, clip_model)

    # Pre-load val features
    print("\nLoading visual features and labels from val set.")
    val_features, val_labels = pre_load_features(clip_model, val_loader)

    # Pre-load test features
    print("\nLoading visual features and labels from test set.")
    test_features, test_labels = pre_load_features(clip_model, test_loader)

    test_features = test_features.cuda()
    test_labels = test_labels.cuda()

    # Zero-shot CLIP
    clip_logits = logit_scale * test_features @ textual_features
    zs_acc = cls_acc(clip_logits, test_labels)
    print("\n**** Zero-shot CLIP's test accuracy: {:.2f}. ****\n".format(zs_acc))

    test_features = test_features.cpu()
    test_labels = test_labels.cpu()

    list_lora_layers = apply_Dora_v2(args, clip_model)
    clip_model = clip_model.cuda()

    if args.eval_only:
        load_lora(args, list_lora_layers)
        acc_test = evaluate_lora(args, clip_model, test_loader, dataset)
        print("**** Test accuracy: {:.2f}. ****\n".format(acc_test))
        return

    mark_only_lora_as_trainable(clip_model)

    num_trainable_params = sum(p.numel() for p in clip_model.parameters() if p.requires_grad)
    print(f"Trainable Params: {num_trainable_params / 1e6:.2f}M.")
    wandb.log({"Trainable Params": num_trainable_params / 1e6})

    total_iters = args.n_iters * args.shots

    optimizer = torch.optim.AdamW(get_lora_parameters(clip_model), weight_decay=1e-2, betas=(0.9, 0.999), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, total_iters, eta_min=1e-6)

    best_acc_val, best_acc_test = 0., 0.
    best_epoch_val = 0

    # training LoRA
    scaler = torch.cuda.amp.GradScaler()
    count_iters = 0
    finish = False
    while count_iters < total_iters:

        clip_model.train()
        acc_train = 0
        tot_samples = 0
        loss_epoch = 0.
        if args.encoder == 'vision':
            text_features = textual_features.t().half()

        for i, (images, target) in enumerate(tqdm(train_loader)):

            template = dataset.template[0]
            texts = [template.format(classname.replace('_', ' ')) for classname in dataset.classnames]
            images, target = images.cuda(), target.cuda()
            if args.encoder == 'text' or args.encoder == 'both':
                with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                    texts = clip.tokenize(texts).cuda()
                    class_embeddings = clip_model.encode_text(texts)
                text_features = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)

            if args.encoder == 'vision' or args.encoder == 'both':
                with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                    image_features = clip_model.encode_image(images)
            else:
                with torch.no_grad():
                    with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                        image_features = clip_model.encode_image(images)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            cosine_similarity = logit_scale * image_features @ text_features.t()

            # Loss
            loss = F.cross_entropy(cosine_similarity, target)
            loss_epoch += loss.item() * target.shape[0]
            # Accuracy
            acc_train += cls_acc(cosine_similarity, target) * target.shape[0]
            tot_samples += target.shape[0]
            # Backward
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)

            scaler.update()
            scheduler.step()

            count_iters += 1

            if count_iters == total_iters:
                break

        if count_iters < total_iters:
            acc_train /= tot_samples
            loss_epoch /= tot_samples
            current_lr = scheduler.get_last_lr()[0]
            print('Epochs: {}/{}, LR: {:.6f}, Acc: {:.4f}, Loss: {:.4f}'.format(count_iters, total_iters, current_lr, acc_train, loss_epoch))
            wandb.log({"train_acc": acc_train, "train_loss": loss_epoch, "LR": current_lr}, step=count_iters)

        # Eval
        if VALIDATION:
            acc_val = evaluate_lora(args, clip_model, val_loader, dataset)
            if acc_val > best_acc_val:
                best_acc_val = acc_val
                wandb.log({"Best Val Accuracy": best_acc_val}, step=count_iters)
            print(f"Validation Accuracy: {acc_val:.4f}")
            wandb.log({"Val Accuracy": acc_val}, step=count_iters)

        acc_test = evaluate_lora(args, clip_model, test_loader, dataset)
        if acc_test > best_acc_test:
            best_acc_test = acc_test
            wandb.log({"Best Test Accuracy": best_acc_test}, step=count_iters)
        print(f"Test Accuracy: {acc_test:.4f}")
        wandb.log({"Test Accuracy": acc_test}, step=count_iters)

    if args.save_path != None:
        save_lora(args, list_lora_layers)
    return

def run_dora_v3(args, clip_model, logit_scale, dataset, train_loader, val_loader, test_loader):
    VALIDATION = True
    
    # Textual features
    print("\nGetting textual features as CLIP's classifier.")
    textual_features = clip_classifier(dataset.classnames, dataset.template, clip_model)

    # Pre-load val features
    print("\nLoading visual features and labels from val set.")
    val_features, val_labels = pre_load_features(clip_model, val_loader)

    # Pre-load test features
    print("\nLoading visual features and labels from test set.")
    test_features, test_labels = pre_load_features(clip_model, test_loader)

    test_features = test_features.cuda()
    test_labels = test_labels.cuda()

    # Zero-shot CLIP
    clip_logits = logit_scale * test_features @ textual_features
    zs_acc = cls_acc(clip_logits, test_labels)
    print("\n**** Zero-shot CLIP's test accuracy: {:.2f}. ****\n".format(zs_acc))

    test_features = test_features.cpu()
    test_labels = test_labels.cpu()

    list_lora_layers = apply_Dora_v3(args, clip_model)
    clip_model = clip_model.cuda()

    if args.eval_only:
        load_lora(args, list_lora_layers)
        acc_test = evaluate_lora(args, clip_model, test_loader, dataset)
        print("**** Test accuracy: {:.2f}. ****\n".format(acc_test))
        return

    mark_only_lora_as_trainable(clip_model)

    num_trainable_params = sum(p.numel() for p in clip_model.parameters() if p.requires_grad)
    print(f"Trainable Params: {num_trainable_params / 1e6:.2f}M.")
    wandb.log({"Trainable Params": num_trainable_params / 1e6})

    total_iters = args.n_iters * args.shots

    optimizer = torch.optim.AdamW(get_lora_parameters(clip_model), weight_decay=1e-2, betas=(0.9, 0.999), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, total_iters, eta_min=1e-6)

    best_acc_val, best_acc_test = 0., 0.
    best_epoch_val = 0

    # training LoRA
    scaler = torch.cuda.amp.GradScaler()
    count_iters = 0
    finish = False
    while count_iters < total_iters:

        clip_model.train()
        acc_train = 0
        tot_samples = 0
        loss_epoch = 0.
        if args.encoder == 'vision':
            text_features = textual_features.t().half()

        for i, (images, target) in enumerate(tqdm(train_loader)):

            template = dataset.template[0]
            texts = [template.format(classname.replace('_', ' ')) for classname in dataset.classnames]
            images, target = images.cuda(), target.cuda()
            if args.encoder == 'text' or args.encoder == 'both':
                with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                    texts = clip.tokenize(texts).cuda()
                    class_embeddings = clip_model.encode_text(texts)
                text_features = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)

            if args.encoder == 'vision' or args.encoder == 'both':
                with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                    image_features = clip_model.encode_image(images)
            else:
                with torch.no_grad():
                    with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                        image_features = clip_model.encode_image(images)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            cosine_similarity = logit_scale * image_features @ text_features.t()

            # Loss
            loss = F.cross_entropy(cosine_similarity, target)
            loss_epoch += loss.item() * target.shape[0]
            # Accuracy
            acc_train += cls_acc(cosine_similarity, target) * target.shape[0]
            tot_samples += target.shape[0]
            # Backward
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)

            scaler.update()
            scheduler.step()

            count_iters += 1

            if count_iters == total_iters:
                break

        if count_iters < total_iters:
            acc_train /= tot_samples
            loss_epoch /= tot_samples
            current_lr = scheduler.get_last_lr()[0]
            print('Epochs: {}/{}, LR: {:.6f}, Acc: {:.4f}, Loss: {:.4f}'.format(count_iters, total_iters, current_lr, acc_train, loss_epoch))
            wandb.log({"train_acc": acc_train, "train_loss": loss_epoch, "LR": current_lr}, step=count_iters)

        # Eval
        if VALIDATION:
            acc_val = evaluate_lora(args, clip_model, val_loader, dataset)
            if acc_val > best_acc_val:
                best_acc_val = acc_val
                wandb.log({"Best Val Accuracy": best_acc_val}, step=count_iters)
            print(f"Validation Accuracy: {acc_val:.4f}")
            wandb.log({"Val Accuracy": acc_val}, step=count_iters)

        acc_test = evaluate_lora(args, clip_model, test_loader, dataset)
        if acc_test > best_acc_test:
            best_acc_test = acc_test
            wandb.log({"Best Test Accuracy": best_acc_test}, step=count_iters)
        print(f"Test Accuracy: {acc_test:.4f}")
        wandb.log({"Test Accuracy": acc_test}, step=count_iters)

    if args.save_path != None:
        save_lora(args, list_lora_layers)
    return

def run_dora_v4(args, clip_model, logit_scale, dataset, train_loader, val_loader, test_loader):
    VALIDATION = True
    
    # Textual features
    print("\nGetting textual features as CLIP's classifier.")
    textual_features = clip_classifier(dataset.classnames, dataset.template, clip_model)

    # Pre-load val features
    print("\nLoading visual features and labels from val set.")
    val_features, val_labels = pre_load_features(clip_model, val_loader)

    # Pre-load test features
    print("\nLoading visual features and labels from test set.")
    test_features, test_labels = pre_load_features(clip_model, test_loader)

    test_features = test_features.cuda()
    test_labels = test_labels.cuda()

    # Zero-shot CLIP
    clip_logits = logit_scale * test_features @ textual_features
    zs_acc = cls_acc(clip_logits, test_labels)
    print("\n**** Zero-shot CLIP's test accuracy: {:.2f}. ****\n".format(zs_acc))

    test_features = test_features.cpu()
    test_labels = test_labels.cpu()

    list_lora_layers = apply_Dora_v4(args, clip_model)
    clip_model = clip_model.cuda()

    if args.eval_only:
        load_lora(args, list_lora_layers)
        acc_test = evaluate_lora(args, clip_model, test_loader, dataset)
        print("**** Test accuracy: {:.2f}. ****\n".format(acc_test))
        return

    mark_only_lora_as_trainable(clip_model)

    num_trainable_params = sum(p.numel() for p in clip_model.parameters() if p.requires_grad)
    print(f"Trainable Params: {num_trainable_params / 1e6:.2f}M.")
    wandb.log({"Trainable Params": num_trainable_params / 1e6})

    total_iters = args.n_iters * args.shots

    optimizer = torch.optim.AdamW(get_lora_parameters(clip_model), weight_decay=1e-2, betas=(0.9, 0.999), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, total_iters, eta_min=1e-6)

    best_acc_val, best_acc_test = 0., 0.
    best_epoch_val = 0

    # training LoRA
    scaler = torch.cuda.amp.GradScaler()
    count_iters = 0
    finish = False
    while count_iters < total_iters:

        clip_model.train()
        acc_train = 0
        tot_samples = 0
        loss_epoch = 0.
        if args.encoder == 'vision':
            text_features = textual_features.t().half()

        for i, (images, target) in enumerate(tqdm(train_loader)):

            template = dataset.template[0]
            texts = [template.format(classname.replace('_', ' ')) for classname in dataset.classnames]
            images, target = images.cuda(), target.cuda()
            if args.encoder == 'text' or args.encoder == 'both':
                with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                    texts = clip.tokenize(texts).cuda()
                    class_embeddings = clip_model.encode_text(texts)
                text_features = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)

            if args.encoder == 'vision' or args.encoder == 'both':
                with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                    image_features = clip_model.encode_image(images)
            else:
                with torch.no_grad():
                    with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                        image_features = clip_model.encode_image(images)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            cosine_similarity = logit_scale * image_features @ text_features.t()

            # Loss
            loss = F.cross_entropy(cosine_similarity, target)
            loss_epoch += loss.item() * target.shape[0]
            # Accuracy
            acc_train += cls_acc(cosine_similarity, target) * target.shape[0]
            tot_samples += target.shape[0]
            # Backward
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)

            scaler.update()
            scheduler.step()

            count_iters += 1

            if count_iters == total_iters:
                break

        if count_iters < total_iters:
            acc_train /= tot_samples
            loss_epoch /= tot_samples
            current_lr = scheduler.get_last_lr()[0]
            print('Epochs: {}/{}, LR: {:.6f}, Acc: {:.4f}, Loss: {:.4f}'.format(count_iters, total_iters, current_lr, acc_train, loss_epoch))
            wandb.log({"train_acc": acc_train, "train_loss": loss_epoch, "LR": current_lr}, step=count_iters)

        # Eval
        if VALIDATION:
            acc_val = evaluate_lora(args, clip_model, val_loader, dataset)
            if acc_val > best_acc_val:
                best_acc_val = acc_val
                wandb.log({"Best Val Accuracy": best_acc_val}, step=count_iters)
            print(f"Validation Accuracy: {acc_val:.4f}")
            wandb.log({"Val Accuracy": acc_val}, step=count_iters)

        acc_test = evaluate_lora(args, clip_model, test_loader, dataset)
        if acc_test > best_acc_test:
            best_acc_test = acc_test
            wandb.log({"Best Test Accuracy": best_acc_test}, step=count_iters)
        print(f"Test Accuracy: {acc_test:.4f}")
        wandb.log({"Test Accuracy": acc_test}, step=count_iters)

    if args.save_path != None:
        save_lora(args, list_lora_layers)
    return

def run_dora_v5(args, clip_model, logit_scale, dataset, train_loader, val_loader, test_loader):
    VALIDATION = True
    
    # Textual features
    print("\nGetting textual features as CLIP's classifier.")
    textual_features = clip_classifier(dataset.classnames, dataset.template, clip_model)

    # Pre-load val features
    print("\nLoading visual features and labels from val set.")
    val_features, val_labels = pre_load_features(clip_model, val_loader)

    # Pre-load test features
    print("\nLoading visual features and labels from test set.")
    test_features, test_labels = pre_load_features(clip_model, test_loader)

    test_features = test_features.cuda()
    test_labels = test_labels.cuda()

    # Zero-shot CLIP
    clip_logits = logit_scale * test_features @ textual_features
    zs_acc = cls_acc(clip_logits, test_labels)
    print("\n**** Zero-shot CLIP's test accuracy: {:.2f}. ****\n".format(zs_acc))

    test_features = test_features.cpu()
    test_labels = test_labels.cpu()

    list_lora_layers = apply_Dora_v5(args, clip_model)
    clip_model = clip_model.cuda()

    if args.eval_only:
        load_lora(args, list_lora_layers)
        acc_test = evaluate_lora(args, clip_model, test_loader, dataset)
        print("**** Test accuracy: {:.2f}. ****\n".format(acc_test))
        return

    mark_only_lora_as_trainable(clip_model)

    num_trainable_params = sum(p.numel() for p in clip_model.parameters() if p.requires_grad)
    print(f"Trainable Params: {num_trainable_params / 1e6:.2f}M.")
    wandb.log({"Trainable Params": num_trainable_params / 1e6})

    total_iters = args.n_iters * args.shots

    optimizer = torch.optim.AdamW(get_lora_parameters(clip_model), weight_decay=1e-2, betas=(0.9, 0.999), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, total_iters, eta_min=1e-6)

    best_acc_val, best_acc_test = 0., 0.
    best_epoch_val = 0

    # training LoRA
    scaler = torch.cuda.amp.GradScaler()
    count_iters = 0
    finish = False
    while count_iters < total_iters:

        clip_model.train()
        acc_train = 0
        tot_samples = 0
        loss_epoch = 0.
        if args.encoder == 'vision':
            text_features = textual_features.t().half()

        for i, (images, target) in enumerate(tqdm(train_loader)):

            template = dataset.template[0]
            texts = [template.format(classname.replace('_', ' ')) for classname in dataset.classnames]
            images, target = images.cuda(), target.cuda()
            if args.encoder == 'text' or args.encoder == 'both':
                with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                    texts = clip.tokenize(texts).cuda()
                    class_embeddings = clip_model.encode_text(texts)
                text_features = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)

            if args.encoder == 'vision' or args.encoder == 'both':
                with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                    image_features = clip_model.encode_image(images)
            else:
                with torch.no_grad():
                    with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                        image_features = clip_model.encode_image(images)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            cosine_similarity = logit_scale * image_features @ text_features.t()

            # Loss
            loss = F.cross_entropy(cosine_similarity, target)
            loss_epoch += loss.item() * target.shape[0]
            # Accuracy
            acc_train += cls_acc(cosine_similarity, target) * target.shape[0]
            tot_samples += target.shape[0]
            # Backward
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)

            scaler.update()
            scheduler.step()

            count_iters += 1

            if count_iters == total_iters:
                break

        if count_iters < total_iters:
            acc_train /= tot_samples
            loss_epoch /= tot_samples
            current_lr = scheduler.get_last_lr()[0]
            print('Epochs: {}/{}, LR: {:.6f}, Acc: {:.4f}, Loss: {:.4f}'.format(count_iters, total_iters, current_lr, acc_train, loss_epoch))
            wandb.log({"train_acc": acc_train, "train_loss": loss_epoch, "LR": current_lr}, step=count_iters)

        # Eval
        if VALIDATION:
            acc_val = evaluate_lora(args, clip_model, val_loader, dataset)
            if acc_val > best_acc_val:
                best_acc_val = acc_val
                wandb.log({"Best Val Accuracy": best_acc_val}, step=count_iters)
            print(f"Validation Accuracy: {acc_val:.4f}")
            wandb.log({"Val Accuracy": acc_val}, step=count_iters)

        acc_test = evaluate_lora(args, clip_model, test_loader, dataset)
        if acc_test > best_acc_test:
            best_acc_test = acc_test
            wandb.log({"Best Test Accuracy": best_acc_test}, step=count_iters)
        print(f"Test Accuracy: {acc_test:.4f}")
        wandb.log({"Test Accuracy": acc_test}, step=count_iters)

    if args.save_path != None:
        save_lora(args, list_lora_layers)
    return

def run_dora_v7(args, clip_model, logit_scale, dataset, train_loader, val_loader, test_loader,
                text_W_prime_dict, text_S_dict, vision_W_prime_dict, vision_S_dict):
    VALIDATION = True
    
    # Textual features
    print("\nGetting textual features as CLIP's classifier.")
    textual_features = clip_classifier(dataset.classnames, dataset.template, clip_model)

    # Pre-load val features
    print("\nLoading visual features and labels from val set.")
    val_features, val_labels = pre_load_features(clip_model, val_loader)

    # Pre-load test features
    print("\nLoading visual features and labels from test set.")
    test_features, test_labels = pre_load_features(clip_model, test_loader)

    test_features = test_features.cuda()
    test_labels = test_labels.cuda()

    # Zero-shot CLIP
    clip_logits = logit_scale * test_features @ textual_features
    zs_acc = cls_acc(clip_logits, test_labels)
    print("\n**** Zero-shot CLIP's test accuracy: {:.2f}. ****\n".format(zs_acc))

    test_features = test_features.cpu()
    test_labels = test_labels.cpu()

    list_lora_layers = apply_Dora_v7(args, clip_model, text_W_prime_dict, text_S_dict, vision_W_prime_dict, vision_S_dict)
    
    # 清空cuda内的text和vision字典缓存
    del text_W_prime_dict, text_S_dict, vision_W_prime_dict, vision_S_dict
    torch.cuda.empty_cache()
    
    clip_model = clip_model.cuda()

    if args.eval_only:
        load_lora(args, list_lora_layers)
        acc_test = evaluate_lora(args, clip_model, test_loader, dataset)
        print("**** Test accuracy: {:.2f}. ****\n".format(acc_test))
        return

    mark_only_lora_as_trainable(clip_model)

    num_trainable_params = sum(p.numel() for p in clip_model.parameters() if p.requires_grad)
    print(f"Trainable Params: {num_trainable_params / 1e6:.2f}M.")
    wandb.log({"Trainable Params": num_trainable_params / 1e6})

    total_iters = args.n_iters * args.shots

    optimizer = torch.optim.AdamW(get_lora_parameters(clip_model), weight_decay=1e-2, betas=(0.9, 0.999), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, total_iters, eta_min=1e-6)

    best_acc_val, best_acc_test = 0., 0.
    best_epoch_val = 0

    # training LoRA
    scaler = torch.cuda.amp.GradScaler()
    count_iters = 0
    finish = False
    while count_iters < total_iters:

        clip_model.train()
        acc_train = 0
        tot_samples = 0
        loss_epoch = 0.
        if args.encoder == 'vision':
            text_features = textual_features.t().half()

        for i, (images, target) in enumerate(tqdm(train_loader)):

            template = dataset.template[0]
            texts = [template.format(classname.replace('_', ' ')) for classname in dataset.classnames]
            images, target = images.cuda(), target.cuda()
            if args.encoder == 'text' or args.encoder == 'both':
                with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                    texts = clip.tokenize(texts).cuda()
                    class_embeddings = clip_model.encode_text(texts)
                text_features = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)

            if args.encoder == 'vision' or args.encoder == 'both':
                with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                    image_features = clip_model.encode_image(images)
            else:
                with torch.no_grad():
                    with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                        image_features = clip_model.encode_image(images)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            cosine_similarity = logit_scale * image_features @ text_features.t()

            # Loss
            loss = F.cross_entropy(cosine_similarity, target)
            loss_epoch += loss.item() * target.shape[0]
            # Accuracy
            acc_train += cls_acc(cosine_similarity, target) * target.shape[0]
            tot_samples += target.shape[0]
            # Backward
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)

            scaler.update()
            scheduler.step()

            count_iters += 1

            if count_iters == total_iters:
                break

        if count_iters < total_iters:
            acc_train /= tot_samples
            loss_epoch /= tot_samples
            current_lr = scheduler.get_last_lr()[0]
            print('Epochs: {}/{}, LR: {:.6f}, Acc: {:.4f}, Loss: {:.4f}'.format(count_iters, total_iters, current_lr, acc_train, loss_epoch))
            wandb.log({"train_acc": acc_train, "train_loss": loss_epoch, "LR": current_lr}, step=count_iters)

        # Eval
        if VALIDATION:
            acc_val = evaluate_lora(args, clip_model, val_loader, dataset)
            if acc_val > best_acc_val:
                best_acc_val = acc_val
                wandb.log({"Best Val Accuracy": best_acc_val}, step=count_iters)
            print(f"Validation Accuracy: {acc_val:.4f}")
            wandb.log({"Val Accuracy": acc_val}, step=count_iters)

        acc_test = evaluate_lora(args, clip_model, test_loader, dataset)
        if acc_test > best_acc_test:
            best_acc_test = acc_test
            wandb.log({"Best Test Accuracy": best_acc_test}, step=count_iters)
        print(f"Test Accuracy: {acc_test:.4f}")
        wandb.log({"Test Accuracy": acc_test}, step=count_iters)

    if args.save_path != None:
        save_lora(args, list_lora_layers)
    return

def run_dora_v7_2(args, clip_model, logit_scale, dataset, train_loader, val_loader, test_loader,
                text_W_prime_dict, text_S_dict, vision_W_prime_dict, vision_S_dict):
    VALIDATION = True
    
    # Textual features
    print("\nGetting textual features as CLIP's classifier.")
    textual_features = clip_classifier(dataset.classnames, dataset.template, clip_model)

    # Pre-load val features
    print("\nLoading visual features and labels from val set.")
    val_features, val_labels = pre_load_features(clip_model, val_loader)

    # Pre-load test features
    print("\nLoading visual features and labels from test set.")
    test_features, test_labels = pre_load_features(clip_model, test_loader)

    test_features = test_features.cuda()
    test_labels = test_labels.cuda()

    # Zero-shot CLIP
    clip_logits = logit_scale * test_features @ textual_features
    zs_acc = cls_acc(clip_logits, test_labels)
    print("\n**** Zero-shot CLIP's test accuracy: {:.2f}. ****\n".format(zs_acc))

    test_features = test_features.cpu()
    test_labels = test_labels.cpu()

    list_lora_layers = apply_Dora_v7_2(args, clip_model, text_W_prime_dict, text_S_dict, vision_W_prime_dict, vision_S_dict)
    
    # 清空cuda内的text和vision字典缓存
    del text_W_prime_dict, text_S_dict, vision_W_prime_dict, vision_S_dict
    torch.cuda.empty_cache()
    
    clip_model = clip_model.cuda()

    if args.eval_only:
        load_lora(args, list_lora_layers)
        acc_test = evaluate_lora(args, clip_model, test_loader, dataset)
        print("**** Test accuracy: {:.2f}. ****\n".format(acc_test))
        return

    mark_only_lora_as_trainable(clip_model)

    num_trainable_params = sum(p.numel() for p in clip_model.parameters() if p.requires_grad)
    print(f"Trainable Params: {num_trainable_params / 1e6:.2f}M.")
    wandb.log({"Trainable Params": num_trainable_params / 1e6})

    total_iters = args.n_iters * args.shots

    optimizer = torch.optim.AdamW(get_lora_parameters(clip_model), weight_decay=1e-2, betas=(0.9, 0.999), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, total_iters, eta_min=1e-6)

    best_acc_val, best_acc_test = 0., 0.
    best_epoch_val = 0

    # training LoRA
    scaler = torch.cuda.amp.GradScaler()
    count_iters = 0
    finish = False
    while count_iters < total_iters:

        clip_model.train()
        acc_train = 0
        tot_samples = 0
        loss_epoch = 0.
        if args.encoder == 'vision':
            text_features = textual_features.t().half()

        for i, (images, target) in enumerate(tqdm(train_loader)):

            template = dataset.template[0]
            texts = [template.format(classname.replace('_', ' ')) for classname in dataset.classnames]
            images, target = images.cuda(), target.cuda()
            if args.encoder == 'text' or args.encoder == 'both':
                with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                    texts = clip.tokenize(texts).cuda()
                    class_embeddings, text_qkv_loss = clip_model.encode_text(texts)
                text_features = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)

            if args.encoder == 'vision' or args.encoder == 'both':
                with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                    image_features, vision_qkv_loss = clip_model.encode_image(images)
            else:
                with torch.no_grad():
                    with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                        image_features = clip_model.encode_image(images)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            cosine_similarity = logit_scale * image_features @ text_features.t()

            # Loss
            loss = F.cross_entropy(cosine_similarity, target)
            # 加入qkv_loss
            total_loss = loss + (vision_qkv_loss + text_qkv_loss) * args.qkv_loss_weight
            loss_epoch += total_loss.item() * target.shape[0]
            # Accuracy
            acc_train += cls_acc(cosine_similarity, target) * target.shape[0]
            tot_samples += target.shape[0]
            # Backward
            optimizer.zero_grad()
            scaler.scale(total_loss).backward()
            scaler.step(optimizer)

            scaler.update()
            scheduler.step()

            count_iters += 1

            if count_iters == total_iters:
                break

        if count_iters < total_iters:
            acc_train /= tot_samples
            loss_epoch /= tot_samples
            current_lr = scheduler.get_last_lr()[0]
            print('Epochs: {}/{}, LR: {:.6f}, Acc: {:.4f}, Total Loss: {:.4f}, Vision QKV Loss: {:.4f}, Text QKV Loss: {:.4f}'.format(count_iters, total_iters, current_lr, acc_train, loss_epoch, vision_qkv_loss * args.qkv_loss_weight, text_qkv_loss * args.qkv_loss_weight))
            wandb.log({"train_acc": acc_train, "loss/train_loss": loss_epoch, "LR": current_lr, "loss/Vision QKV Loss": vision_qkv_loss * args.qkv_loss_weight, "loss/Text QKV Loss": text_qkv_loss * args.qkv_loss_weight}, step=count_iters)

        # Eval
        if VALIDATION:
            acc_val = evaluate_lora(args, clip_model, val_loader, dataset)
            if acc_val > best_acc_val:
                best_acc_val = acc_val
                wandb.log({"Optical Calculation/Best Val Accuracy": best_acc_val}, step=count_iters)
            print(f"Validation Accuracy: {acc_val:.4f}")
            wandb.log({"Val Accuracy": acc_val}, step=count_iters)

        acc_test = evaluate_lora(args, clip_model, test_loader, dataset)
        if acc_test > best_acc_test:
            best_acc_test = acc_test
            wandb.log({"Optical Calculation/Best Test Accuracy": best_acc_test}, step=count_iters)
        print(f"Test Accuracy: {acc_test:.4f}")
        wandb.log({"Test Accuracy": acc_test}, step=count_iters)

    if args.save_path != None:
        save_lora(args, list_lora_layers)
    return

def run_dora_v7_3(args, clip_model, logit_scale, dataset, train_loader, val_loader, test_loader,
                text_W_prime_dict, text_S_dict, vision_W_prime_dict, vision_S_dict):
    VALIDATION = True
    
    # Textual features
    print("\nGetting textual features as CLIP's classifier.")
    textual_features = clip_classifier(dataset.classnames, dataset.template, clip_model)

    # Pre-load val features
    print("\nLoading visual features and labels from val set.")
    val_features, val_labels = pre_load_features(clip_model, val_loader)

    # Pre-load test features
    print("\nLoading visual features and labels from test set.")
    test_features, test_labels = pre_load_features(clip_model, test_loader)

    test_features = test_features.cuda()
    test_labels = test_labels.cuda()

    # Zero-shot CLIP
    clip_logits = logit_scale * test_features @ textual_features
    zs_acc = cls_acc(clip_logits, test_labels)
    print("\n**** Zero-shot CLIP's test accuracy: {:.2f}. ****\n".format(zs_acc))

    test_features = test_features.cpu()
    test_labels = test_labels.cpu()

    list_lora_layers = apply_Dora_v7_3(args, clip_model, text_W_prime_dict, text_S_dict, vision_W_prime_dict, vision_S_dict)
    
    # 清空cuda内的text和vision字典缓存
    del text_W_prime_dict, text_S_dict, vision_W_prime_dict, vision_S_dict
    torch.cuda.empty_cache()
    
    clip_model = clip_model.cuda()

    if args.eval_only:
        load_lora(args, list_lora_layers)
        acc_test = evaluate_lora(args, clip_model, test_loader, dataset)
        print("**** Test accuracy: {:.2f}. ****\n".format(acc_test))
        return

    mark_only_lora_as_trainable(clip_model)

    num_trainable_params = sum(p.numel() for p in clip_model.parameters() if p.requires_grad)
    print(f"Trainable Params: {num_trainable_params / 1e6:.2f}M.")
    wandb.log({"Trainable Params": num_trainable_params / 1e6})

    total_iters = args.n_iters * args.shots

    optimizer = torch.optim.AdamW(get_lora_parameters(clip_model), weight_decay=1e-2, betas=(0.9, 0.999), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, total_iters, eta_min=1e-6)

    best_acc_val, best_acc_test = 0., 0.
    best_epoch_val = 0

    # training LoRA
    scaler = torch.cuda.amp.GradScaler()
    count_iters = 0
    finish = False
    while count_iters < total_iters:

        clip_model.train()
        acc_train = 0
        tot_samples = 0
        loss_epoch = 0.
        if args.encoder == 'vision':
            text_features = textual_features.t().half()

        for i, (images, target) in enumerate(tqdm(train_loader)):

            template = dataset.template[0]
            texts = [template.format(classname.replace('_', ' ')) for classname in dataset.classnames]
            images, target = images.cuda(), target.cuda()
            if args.encoder == 'text' or args.encoder == 'both':
                with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                    texts = clip.tokenize(texts).cuda()
                    class_embeddings, text_qkv_loss = clip_model.encode_text(texts)
                text_features = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)

            if args.encoder == 'vision' or args.encoder == 'both':
                with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                    image_features, vision_qkv_loss = clip_model.encode_image(images)
            else:
                with torch.no_grad():
                    with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                        image_features = clip_model.encode_image(images)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            cosine_similarity = logit_scale * image_features @ text_features.t()

            # Loss
            loss = F.cross_entropy(cosine_similarity, target)
            # 加入qkv_loss
            total_loss = loss + (vision_qkv_loss + text_qkv_loss) * args.qkv_loss_weight
            loss_epoch += total_loss.item() * target.shape[0]
            # Accuracy
            acc_train += cls_acc(cosine_similarity, target) * target.shape[0]
            tot_samples += target.shape[0]
            # Backward
            optimizer.zero_grad()
            scaler.scale(total_loss).backward()
            scaler.step(optimizer)

            scaler.update()
            scheduler.step()

            count_iters += 1

            if count_iters == total_iters:
                break

        if count_iters < total_iters:
            acc_train /= tot_samples
            loss_epoch /= tot_samples
            current_lr = scheduler.get_last_lr()[0]
            print('Epochs: {}/{}, LR: {:.6f}, Acc: {:.4f}, Total Loss: {:.4f}, Vision QKV Loss: {:.4f}, Text QKV Loss: {:.4f}'.format(count_iters, total_iters, current_lr, acc_train, loss_epoch, vision_qkv_loss * args.qkv_loss_weight, text_qkv_loss * args.qkv_loss_weight))
            wandb.log({"train_acc": acc_train, "loss/train_loss": loss_epoch, "LR": current_lr, "loss/Vision QKV Loss": vision_qkv_loss * args.qkv_loss_weight, "loss/Text QKV Loss": text_qkv_loss * args.qkv_loss_weight}, step=count_iters)

        # Eval
        if VALIDATION:
            acc_val = evaluate_lora(args, clip_model, val_loader, dataset)
            if acc_val > best_acc_val:
                best_acc_val = acc_val
                wandb.log({"Optical Calculation/Best Val Accuracy": best_acc_val}, step=count_iters)
            print(f"Validation Accuracy: {acc_val:.4f}")
            wandb.log({"Val Accuracy": acc_val}, step=count_iters)

        acc_test = evaluate_lora(args, clip_model, test_loader, dataset)
        if acc_test > best_acc_test:
            best_acc_test = acc_test
            wandb.log({"Optical Calculation/Best Test Accuracy": best_acc_test}, step=count_iters)
        print(f"Test Accuracy: {acc_test:.4f}")
        wandb.log({"Test Accuracy": acc_test}, step=count_iters)

    if args.save_path != None:
        save_lora(args, list_lora_layers)
    return

def run_dora_v8(args, clip_model, logit_scale, dataset, train_loader, val_loader, test_loader,
                text_W_prime_dict, text_S_dict, vision_W_prime_dict, vision_S_dict):
    VALIDATION = True
    
    # Textual features
    print("\nGetting textual features as CLIP's classifier.")
    textual_features = clip_classifier(dataset.classnames, dataset.template, clip_model)

    # Pre-load val features
    print("\nLoading visual features and labels from val set.")
    val_features, val_labels = pre_load_features(clip_model, val_loader)

    # Pre-load test features
    print("\nLoading visual features and labels from test set.")
    test_features, test_labels = pre_load_features(clip_model, test_loader)

    test_features = test_features.cuda()
    test_labels = test_labels.cuda()

    # Zero-shot CLIP
    clip_logits = logit_scale * test_features @ textual_features
    zs_acc = cls_acc(clip_logits, test_labels)
    print("\n**** Zero-shot CLIP's test accuracy: {:.2f}. ****\n".format(zs_acc))

    test_features = test_features.cpu()
    test_labels = test_labels.cpu()

    list_lora_layers = apply_Dora_v8(args, clip_model, text_W_prime_dict, text_S_dict, vision_W_prime_dict, vision_S_dict)
    
    # 清空cuda内的text和vision字典缓存
    del text_W_prime_dict, text_S_dict, vision_W_prime_dict, vision_S_dict
    torch.cuda.empty_cache()
    
    clip_model = clip_model.cuda()

    if args.eval_only:
        load_lora(args, list_lora_layers)
        acc_test = evaluate_lora(args, clip_model, test_loader, dataset)
        print("**** Test accuracy: {:.2f}. ****\n".format(acc_test))
        return

    mark_only_lora_as_trainable(clip_model)

    num_trainable_params = sum(p.numel() for p in clip_model.parameters() if p.requires_grad)
    print(f"Trainable Params: {num_trainable_params / 1e6:.2f}M.")
    wandb.log({"Trainable Params": num_trainable_params / 1e6})

    total_iters = args.n_iters * args.shots

    optimizer = torch.optim.AdamW(get_lora_parameters(clip_model), weight_decay=1e-2, betas=(0.9, 0.999), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, total_iters, eta_min=1e-6)

    best_acc_val, best_acc_test = 0., 0.
    best_epoch_val = 0

    # training LoRA
    scaler = torch.cuda.amp.GradScaler()
    count_iters = 0
    finish = False
    while count_iters < total_iters:

        clip_model.train()
        acc_train = 0
        tot_samples = 0
        loss_epoch = 0.
        if args.encoder == 'vision':
            text_features = textual_features.t().half()

        for i, (images, target) in enumerate(tqdm(train_loader)):

            template = dataset.template[0]
            texts = [template.format(classname.replace('_', ' ')) for classname in dataset.classnames]
            images, target = images.cuda(), target.cuda()
            if args.encoder == 'text' or args.encoder == 'both':
                with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                    texts = clip.tokenize(texts).cuda()
                    class_embeddings = clip_model.encode_text(texts)
                text_features = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)

            if args.encoder == 'vision' or args.encoder == 'both':
                with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                    image_features = clip_model.encode_image(images)
            else:
                with torch.no_grad():
                    with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                        image_features = clip_model.encode_image(images)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            cosine_similarity = logit_scale * image_features @ text_features.t()

            # Loss
            loss = F.cross_entropy(cosine_similarity, target)
            loss_epoch += loss.item() * target.shape[0]
            # Accuracy
            acc_train += cls_acc(cosine_similarity, target) * target.shape[0]
            tot_samples += target.shape[0]
            # Backward
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)

            scaler.update()
            scheduler.step()

            count_iters += 1

            if count_iters == total_iters:
                break

        if count_iters < total_iters:
            acc_train /= tot_samples
            loss_epoch /= tot_samples
            current_lr = scheduler.get_last_lr()[0]
            print('Epochs: {}/{}, LR: {:.6f}, Acc: {:.4f}, Loss: {:.4f}'.format(count_iters, total_iters, current_lr, acc_train, loss_epoch))
            wandb.log({"train_acc": acc_train, "loss/train_loss": loss_epoch, "LR": current_lr}, step=count_iters)

        # Eval
        if VALIDATION:
            acc_val = evaluate_lora(args, clip_model, val_loader, dataset)
            if acc_val > best_acc_val:
                best_acc_val = acc_val
                wandb.log({"Optical Calculation/Best Val Accuracy": best_acc_val}, step=count_iters)
            print(f"Validation Accuracy: {acc_val:.4f}")
            wandb.log({"Val Accuracy": acc_val}, step=count_iters)

        acc_test = evaluate_lora(args, clip_model, test_loader, dataset)
        if acc_test > best_acc_test:
            best_acc_test = acc_test
            wandb.log({"Optical Calculation/Best Test Accuracy": best_acc_test}, step=count_iters)
        print(f"Test Accuracy: {acc_test:.4f}")
        wandb.log({"Test Accuracy": acc_test}, step=count_iters)

    if args.save_path != None:
        save_lora(args, list_lora_layers)
    return

def run_dora_v9(args, clip_model, logit_scale, dataset, train_loader, val_loader, test_loader,
                text_W_prime_dict, text_S_dict, vision_W_prime_dict, vision_S_dict):
    VALIDATION = True
    
    # Textual features
    print("\nGetting textual features as CLIP's classifier.")
    textual_features = clip_classifier(dataset.classnames, dataset.template, clip_model)

    # Pre-load val features
    print("\nLoading visual features and labels from val set.")
    val_features, val_labels = pre_load_features(clip_model, val_loader)

    # Pre-load test features
    print("\nLoading visual features and labels from test set.")
    test_features, test_labels = pre_load_features(clip_model, test_loader)

    test_features = test_features.cuda()
    test_labels = test_labels.cuda()

    # Zero-shot CLIP
    clip_logits = logit_scale * test_features @ textual_features
    zs_acc = cls_acc(clip_logits, test_labels)
    print("\n**** Zero-shot CLIP's test accuracy: {:.2f}. ****\n".format(zs_acc))

    test_features = test_features.cpu()
    test_labels = test_labels.cpu()

    list_lora_layers = apply_Dora_v9(args, clip_model, text_W_prime_dict, text_S_dict, vision_W_prime_dict, vision_S_dict)
    
    # 清空cuda内的text和vision字典缓存
    del text_W_prime_dict, text_S_dict, vision_W_prime_dict, vision_S_dict
    torch.cuda.empty_cache()
    
    clip_model = clip_model.cuda()

    if args.eval_only:
        load_lora(args, list_lora_layers)
        acc_test = evaluate_lora(args, clip_model, test_loader, dataset)
        print("**** Test accuracy: {:.2f}. ****\n".format(acc_test))
        return

    mark_only_lora_as_trainable(clip_model)

    num_trainable_params = sum(p.numel() for p in clip_model.parameters() if p.requires_grad)
    print(f"Trainable Params: {num_trainable_params / 1e6:.2f}M.")
    wandb.log({"Trainable Params": num_trainable_params / 1e6})

    total_iters = args.n_iters * args.shots

    optimizer = torch.optim.AdamW(get_lora_parameters(clip_model), weight_decay=1e-2, betas=(0.9, 0.999), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, total_iters, eta_min=1e-6)

    best_acc_val, best_acc_test = 0., 0.
    best_epoch_val = 0

    # training LoRA
    scaler = torch.cuda.amp.GradScaler()
    count_iters = 0
    finish = False
    while count_iters < total_iters:

        clip_model.train()
        acc_train = 0
        tot_samples = 0
        loss_epoch = 0.
        if args.encoder == 'vision':
            text_features = textual_features.t().half()

        for i, (images, target) in enumerate(tqdm(train_loader)):

            template = dataset.template[0]
            texts = [template.format(classname.replace('_', ' ')) for classname in dataset.classnames]
            images, target = images.cuda(), target.cuda()
            if args.encoder == 'text' or args.encoder == 'both':
                with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                    texts = clip.tokenize(texts).cuda()
                    class_embeddings = clip_model.encode_text(texts)
                text_features = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)

            if args.encoder == 'vision' or args.encoder == 'both':
                with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                    image_features = clip_model.encode_image(images)
            else:
                with torch.no_grad():
                    with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                        image_features = clip_model.encode_image(images)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            cosine_similarity = logit_scale * image_features @ text_features.t()
                
            # Loss
            loss = F.cross_entropy(cosine_similarity, target)

            # 跳过第一次迭代
            if count_iters != 0:
                # 计算ortho_loss
                ortho_loss = 0
                layer_num = 0
                for n, layer in clip_model.named_modules():
                    if isinstance(layer, DoraLinear_v9):
                        layer_num += 1
                        ortho_loss += layer.orthonormal_loss()
                ortho_loss *= args.ortho_loss_weight / layer_num
                loss += ortho_loss

            loss_epoch += loss.item() * target.shape[0]
            # Accuracy
            acc_train += cls_acc(cosine_similarity, target) * target.shape[0]
            tot_samples += target.shape[0]
            # Backward
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)

            scaler.update()
            scheduler.step()

            count_iters += 1

            if count_iters == total_iters:
                break

        if count_iters < total_iters:
            acc_train /= tot_samples
            loss_epoch /= tot_samples
            current_lr = scheduler.get_last_lr()[0]
            print('Epochs: {}/{}, LR: {:.6f}, Acc: {:.4f}, Loss: {:.4f}, Ortho Loss: {:.4f}'.format(count_iters, total_iters, current_lr, acc_train, loss_epoch, ortho_loss))
            wandb.log({"train_acc": acc_train, "loss/train_loss": loss_epoch, "LR": current_lr, "loss/ortho_loss": ortho_loss}, step=count_iters)

        # Eval
        if VALIDATION:
            acc_val = evaluate_lora(args, clip_model, val_loader, dataset)
            if acc_val > best_acc_val:
                best_acc_val = acc_val
                wandb.log({"Optical Calculation/Best Val Accuracy": best_acc_val}, step=count_iters)
            print(f"Validation Accuracy: {acc_val:.4f}")
            wandb.log({"Val Accuracy": acc_val}, step=count_iters)

        acc_test = evaluate_lora(args, clip_model, test_loader, dataset)
        if acc_test > best_acc_test:
            best_acc_test = acc_test
            wandb.log({"Optical Calculation/Best Test Accuracy": best_acc_test}, step=count_iters)
        print(f"Test Accuracy: {acc_test:.4f}")
        wandb.log({"Test Accuracy": acc_test}, step=count_iters)

    if args.save_path != None:
        save_lora(args, list_lora_layers)
    return

def run_nora_a(args, clip_model, logit_scale, dataset, train_loader, val_loader, test_loader, text_W_prime_dict, text_S_dict, vision_W_prime_dict, vision_S_dict):
    VALIDATION = True
    
    # Textual features
    print("\nGetting textual features as CLIP's classifier.")
    textual_features = clip_classifier(dataset.classnames, dataset.template, clip_model)

    # Pre-load val features
    print("\nLoading visual features and labels from val set.")
    val_features, val_labels = pre_load_features(clip_model, val_loader)

    # Pre-load test features
    print("\nLoading visual features and labels from test set.")
    test_features, test_labels = pre_load_features(clip_model, test_loader)

    test_features = test_features.cuda()
    test_labels = test_labels.cuda()

    # Zero-shot CLIP
    clip_logits = logit_scale * test_features @ textual_features
    zs_acc = cls_acc(clip_logits, test_labels)
    print("\n**** Zero-shot CLIP's test accuracy: {:.2f}. ****\n".format(zs_acc))

    test_features = test_features.cpu()
    test_labels = test_labels.cpu()

    list_lora_layers = apply_NoRA_A(args, clip_model, text_W_prime_dict, text_S_dict, vision_W_prime_dict, vision_S_dict)
    
    clip_model = clip_model.cuda()

    if args.eval_only:
        load_lora(args, list_lora_layers)
        acc_test = evaluate_lora(args, clip_model, test_loader, dataset)
        print("**** Test accuracy: {:.2f}. ****\n".format(acc_test))
        return

    mark_only_lora_as_trainable(clip_model)

    num_trainable_params = sum(p.numel() for p in clip_model.parameters() if p.requires_grad)
    print(f"Trainable Params: {num_trainable_params / 1e6:.2f}M.")
    wandb.log({"Trainable Params": num_trainable_params / 1e6})

    total_iters = args.n_iters * args.shots

    optimizer = torch.optim.AdamW(get_lora_parameters(clip_model), weight_decay=args.weight_decay, betas=(0.9, 0.999), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, total_iters, eta_min=1e-6)

    best_acc_val, best_acc_test = 0., 0.
    best_epoch_val = 0

    # training LoRA
    scaler = torch.cuda.amp.GradScaler()
    count_iters = 0
    finish = False
    while count_iters < total_iters:

        clip_model.train()
        acc_train = 0
        tot_samples = 0
        loss_epoch = 0.
        if args.encoder == 'vision':
            text_features = textual_features.t().half()

        for i, (images, target) in enumerate(tqdm(train_loader)):

            template = dataset.template[0]
            texts = [template.format(classname.replace('_', ' ')) for classname in dataset.classnames]
            images, target = images.cuda(), target.cuda()
            if args.encoder == 'text' or args.encoder == 'both':
                with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                    texts = clip.tokenize(texts).cuda()
                    class_embeddings = clip_model.encode_text(texts)
                text_features = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)

            if args.encoder == 'vision' or args.encoder == 'both':
                with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                    image_features = clip_model.encode_image(images)
            else:
                with torch.no_grad():
                    with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                        image_features = clip_model.encode_image(images)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            cosine_similarity = logit_scale * image_features @ text_features.t()
                
            # Loss
            loss = F.cross_entropy(cosine_similarity, target)
            loss_epoch += loss.item() * target.shape[0]
            # Accuracy
            acc_train += cls_acc(cosine_similarity, target) * target.shape[0]
            tot_samples += target.shape[0]
            # Backward
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)

            scaler.update()
            scheduler.step()

            count_iters += 1

            if count_iters == total_iters:
                break

        if count_iters < total_iters:
            acc_train /= tot_samples
            loss_epoch /= tot_samples
            current_lr = scheduler.get_last_lr()[0]
            print('Epochs: {}/{}, LR: {:.6f}, Acc: {:.4f}, Loss: {:.4f}'.format(count_iters, total_iters, current_lr, acc_train, loss_epoch))
            wandb.log({"train_acc": acc_train, "loss/train_loss": loss_epoch, "LR": current_lr}, step=count_iters)

        # Eval
        if VALIDATION:
            acc_val = evaluate_lora(args, clip_model, val_loader, dataset)
            if acc_val > best_acc_val:
                best_acc_val = acc_val
                wandb.log({"Optical Calculation/Best Val Accuracy": best_acc_val}, step=count_iters)
            print(f"Validation Accuracy: {acc_val:.4f}")
            wandb.log({"Val Accuracy": acc_val}, step=count_iters)

        acc_test = evaluate_lora(args, clip_model, test_loader, dataset)
        if acc_test > best_acc_test:
            best_acc_test = acc_test
            wandb.log({"Optical Calculation/Best Test Accuracy": best_acc_test}, step=count_iters)
        print(f"Test Accuracy: {acc_test:.4f}")
        wandb.log({"Test Accuracy": acc_test}, step=count_iters)

    if args.save_path != None:
        save_lora(args, list_lora_layers)
    return

def run_nora_b(args, clip_model, logit_scale, dataset, train_loader, val_loader, test_loader, text_W_prime_dict, text_S_dict, vision_W_prime_dict, vision_S_dict):
    VALIDATION = True
    
    # Textual features
    print("\nGetting textual features as CLIP's classifier.")
    textual_features = clip_classifier(dataset.classnames, dataset.template, clip_model)

    # Pre-load val features
    print("\nLoading visual features and labels from val set.")
    val_features, val_labels = pre_load_features(clip_model, val_loader)

    # Pre-load test features
    print("\nLoading visual features and labels from test set.")
    test_features, test_labels = pre_load_features(clip_model, test_loader)

    test_features = test_features.cuda()
    test_labels = test_labels.cuda()

    # Zero-shot CLIP
    clip_logits = logit_scale * test_features @ textual_features
    zs_acc = cls_acc(clip_logits, test_labels)
    print("\n**** Zero-shot CLIP's test accuracy: {:.2f}. ****\n".format(zs_acc))

    test_features = test_features.cpu()
    test_labels = test_labels.cpu()

    list_lora_layers = apply_NoRA_B(args, clip_model, text_W_prime_dict, text_S_dict, vision_W_prime_dict, vision_S_dict)
    
    clip_model = clip_model.cuda()

    if args.eval_only:
        load_lora(args, list_lora_layers)
        acc_test = evaluate_lora(args, clip_model, test_loader, dataset)
        print("**** Test accuracy: {:.2f}. ****\n".format(acc_test))
        return

    mark_only_lora_as_trainable(clip_model)

    num_trainable_params = sum(p.numel() for p in clip_model.parameters() if p.requires_grad)
    print(f"Trainable Params: {num_trainable_params / 1e6:.2f}M.")
    wandb.log({"Trainable Params": num_trainable_params / 1e6})

    total_iters = args.n_iters * args.shots

    optimizer = torch.optim.AdamW(get_lora_parameters(clip_model), weight_decay=args.weight_decay, betas=(0.9, 0.999), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, total_iters, eta_min=1e-6)

    best_acc_val, best_acc_test = 0., 0.
    best_epoch_val = 0

    # training LoRA
    scaler = torch.cuda.amp.GradScaler()
    count_iters = 0
    finish = False
    while count_iters < total_iters:

        clip_model.train()
        acc_train = 0
        tot_samples = 0
        loss_epoch = 0.
        if args.encoder == 'vision':
            text_features = textual_features.t().half()

        for i, (images, target) in enumerate(tqdm(train_loader)):

            template = dataset.template[0]
            texts = [template.format(classname.replace('_', ' ')) for classname in dataset.classnames]
            images, target = images.cuda(), target.cuda()
            if args.encoder == 'text' or args.encoder == 'both':
                with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                    texts = clip.tokenize(texts).cuda()
                    class_embeddings = clip_model.encode_text(texts)
                text_features = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)

            if args.encoder == 'vision' or args.encoder == 'both':
                with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                    image_features = clip_model.encode_image(images)
            else:
                with torch.no_grad():
                    with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                        image_features = clip_model.encode_image(images)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            cosine_similarity = logit_scale * image_features @ text_features.t()
                
            # Loss
            loss = F.cross_entropy(cosine_similarity, target)
            loss_epoch += loss.item() * target.shape[0]
            # Accuracy
            acc_train += cls_acc(cosine_similarity, target) * target.shape[0]
            tot_samples += target.shape[0]
            # Backward
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)

            scaler.update()
            scheduler.step()

            count_iters += 1

            if count_iters == total_iters:
                break

        if count_iters < total_iters:
            acc_train /= tot_samples
            loss_epoch /= tot_samples
            current_lr = scheduler.get_last_lr()[0]
            print('Epochs: {}/{}, LR: {:.6f}, Acc: {:.4f}, Loss: {:.4f}'.format(count_iters, total_iters, current_lr, acc_train, loss_epoch))
            wandb.log({"train_acc": acc_train, "loss/train_loss": loss_epoch, "LR": current_lr}, step=count_iters)

        # Eval
        if VALIDATION:
            acc_val = evaluate_lora(args, clip_model, val_loader, dataset)
            if acc_val > best_acc_val:
                best_acc_val = acc_val
                wandb.log({"Optical Calculation/Best Val Accuracy": best_acc_val}, step=count_iters)
            print(f"Validation Accuracy: {acc_val:.4f}")
            wandb.log({"Val Accuracy": acc_val}, step=count_iters)

        acc_test = evaluate_lora(args, clip_model, test_loader, dataset)
        if acc_test > best_acc_test:
            best_acc_test = acc_test
            wandb.log({"Optical Calculation/Best Test Accuracy": best_acc_test}, step=count_iters)
        print(f"Test Accuracy: {acc_test:.4f}")
        wandb.log({"Test Accuracy": acc_test}, step=count_iters)

    if args.save_path != None:
        save_lora(args, list_lora_layers)
    return

def run_nora_c(args, clip_model, logit_scale, dataset, train_loader, val_loader, test_loader, text_W_prime_dict, text_S_dict, vision_W_prime_dict, vision_S_dict):
    VALIDATION = True
    
    # Textual features
    print("\nGetting textual features as CLIP's classifier.")
    textual_features = clip_classifier(dataset.classnames, dataset.template, clip_model)

    # Pre-load val features
    print("\nLoading visual features and labels from val set.")
    val_features, val_labels = pre_load_features(clip_model, val_loader)

    # Pre-load test features
    print("\nLoading visual features and labels from test set.")
    test_features, test_labels = pre_load_features(clip_model, test_loader)

    test_features = test_features.cuda()
    test_labels = test_labels.cuda()

    # Zero-shot CLIP
    clip_logits = logit_scale * test_features @ textual_features
    zs_acc = cls_acc(clip_logits, test_labels)
    print("\n**** Zero-shot CLIP's test accuracy: {:.2f}. ****\n".format(zs_acc))

    test_features = test_features.cpu()
    test_labels = test_labels.cpu()

    list_lora_layers = apply_NoRA_C(args, clip_model, text_W_prime_dict, text_S_dict, vision_W_prime_dict, vision_S_dict)
    
    clip_model = clip_model.cuda()

    if args.eval_only:
        load_lora(args, list_lora_layers)
        acc_test = evaluate_lora(args, clip_model, test_loader, dataset)
        print("**** Test accuracy: {:.2f}. ****\n".format(acc_test))
        return

    mark_only_lora_as_trainable(clip_model)

    num_trainable_params = sum(p.numel() for p in clip_model.parameters() if p.requires_grad)
    print(f"Trainable Params: {num_trainable_params / 1e6:.2f}M.")
    wandb.log({"Trainable Params": num_trainable_params / 1e6})

    total_iters = args.n_iters * args.shots

    optimizer = torch.optim.AdamW(get_lora_parameters(clip_model), weight_decay=args.weight_decay, betas=(0.9, 0.999), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, total_iters, eta_min=1e-6)

    best_acc_val, best_acc_test = 0., 0.
    best_epoch_val = 0

    # training LoRA
    scaler = torch.cuda.amp.GradScaler()
    count_iters = 0
    finish = False
    while count_iters < total_iters:

        clip_model.train()
        acc_train = 0
        tot_samples = 0
        loss_epoch = 0.
        if args.encoder == 'vision':
            text_features = textual_features.t().half()

        for i, (images, target) in enumerate(tqdm(train_loader)):

            template = dataset.template[0]
            texts = [template.format(classname.replace('_', ' ')) for classname in dataset.classnames]
            images, target = images.cuda(), target.cuda()
            if args.encoder == 'text' or args.encoder == 'both':
                with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                    texts = clip.tokenize(texts).cuda()
                    class_embeddings = clip_model.encode_text(texts)
                text_features = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)

            if args.encoder == 'vision' or args.encoder == 'both':
                with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                    image_features = clip_model.encode_image(images)
            else:
                with torch.no_grad():
                    with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                        image_features = clip_model.encode_image(images)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            cosine_similarity = logit_scale * image_features @ text_features.t()
                
            # Loss
            loss = F.cross_entropy(cosine_similarity, target)
            loss_epoch += loss.item() * target.shape[0]
            # Accuracy
            acc_train += cls_acc(cosine_similarity, target) * target.shape[0]
            tot_samples += target.shape[0]
            # Backward
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)

            scaler.update()
            scheduler.step()

            count_iters += 1

            if count_iters == total_iters:
                break

        if count_iters < total_iters:
            acc_train /= tot_samples
            loss_epoch /= tot_samples
            current_lr = scheduler.get_last_lr()[0]
            print('Epochs: {}/{}, LR: {:.6f}, Acc: {:.4f}, Loss: {:.4f}'.format(count_iters, total_iters, current_lr, acc_train, loss_epoch))
            wandb.log({"train_acc": acc_train, "loss/train_loss": loss_epoch, "LR": current_lr}, step=count_iters)

        # Eval
        if VALIDATION:
            acc_val = evaluate_lora(args, clip_model, val_loader, dataset)
            if acc_val > best_acc_val:
                best_acc_val = acc_val
                wandb.log({"Optical Calculation/Best Val Accuracy": best_acc_val}, step=count_iters)
            print(f"Validation Accuracy: {acc_val:.4f}")
            wandb.log({"Val Accuracy": acc_val}, step=count_iters)

        acc_test = evaluate_lora(args, clip_model, test_loader, dataset)
        if acc_test > best_acc_test:
            best_acc_test = acc_test
            wandb.log({"Optical Calculation/Best Test Accuracy": best_acc_test}, step=count_iters)
        print(f"Test Accuracy: {acc_test:.4f}")
        wandb.log({"Test Accuracy": acc_test}, step=count_iters)

    if args.save_path != None:
        save_lora(args, list_lora_layers)
    return
