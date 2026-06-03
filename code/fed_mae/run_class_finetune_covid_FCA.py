# -*- coding: utf-8 -*-
"""
Federated fine-tuning on COVID-FL with FCA (Federated Classifier Anchoring):
- FedAvg server aggregation
- Personalized classifiers at each client (locally kept)
- Balanced softmax for classifier calibration
- Consistency regularization between federated and personalized classifiers
- Optional FedProx proximal term
"""

import argparse
import os
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

# ---------------------------------------------------------------------
# TensorBoard helper
# ---------------------------------------------------------------------
class TBWriter:
    def __init__(self, log_dir):
        self.log_dir = log_dir
        self.writer = SummaryWriter(log_dir)

    def add_scalar(self, tag, scalar_value, global_step=None):
        self.writer.add_scalar(tag, scalar_value, global_step)

    def add_scalars(self, head, x, step):
        if isinstance(x, dict):
            for k, v in x.items():
                self.writer.add_scalar(f"{head}/{k}", v, step)

    def flush(self):
        self.writer.flush()

    def close(self):
        self.writer.close()


# ---------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------
current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

import fed_mae.models_vit as models_vit  # noqa: E402
from util.FedAvg_utils import (        # noqa: E402
    Partial_Client_Selection,
    valid,
)
from util.data_utils import (          # noqa: E402
    DatasetFLFinetune,
    create_dataset_and_evalmetrix,
)
import util.misc as misc               # noqa: E402


# ---------------------------------------------------------------------
# FCA-specific Model with Dual Classifiers
# ---------------------------------------------------------------------
class FCAModel(nn.Module):
    """
    Wraps a ViT model with two classifier heads:
    - federated_head: shared across all clients (aggregated)
    - personalized_head: kept locally at each client
    """
    def __init__(self, backbone_model, num_classes):
        super().__init__()
        self.backbone = backbone_model
        
        # Remove original head
        if hasattr(self.backbone, 'head'):
            in_features = self.backbone.head.in_features
            self.backbone.head = nn.Identity()
        else:
            # Fallback: assume 768 for ViT-Base
            in_features = 768
        
        # Two classifier heads
        self.federated_head = nn.Linear(in_features, num_classes)
        self.personalized_head = nn.Linear(in_features, num_classes)
        
    def forward(self, x, use_personalized=False):
        """
        Args:
            x: input images
            use_personalized: if True, use personalized_head; else federated_head
        Returns:
            logits from selected head
        """
        features = self.backbone(x)
        if use_personalized:
            return self.personalized_head(features)
        else:
            return self.federated_head(features)
    
    def forward_both(self, x):
        """Returns logits from both heads"""
        features = self.backbone(x)
        fed_logits = self.federated_head(features)
        pers_logits = self.personalized_head(features)
        return fed_logits, pers_logits


# ---------------------------------------------------------------------
# Balanced Softmax Loss with Class Frequency Prior
# ---------------------------------------------------------------------
class BalancedSoftmaxLoss(nn.Module):
    """
    Balanced Softmax Loss from the FCA paper.
    Adjusts logits by adding log(class_frequency) before softmax.
    """
    def __init__(self, class_frequencies):
        super().__init__()
        self.register_buffer('log_prior', torch.log(class_frequencies + 1e-9))
    
    def forward(self, logits, targets):
        """
        Args:
            logits: [B, C] raw logits
            targets: [B] class indices
        """
        adjusted_logits = logits + self.log_prior
        return F.cross_entropy(adjusted_logits, targets)


# ---------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------
def get_args():
    parser = argparse.ArgumentParser(
        'FCA training on COVID-FL',
        add_help=False
    )

    # Model & optimization
    parser.add_argument('--model', default='vit_base_patch16', type=str)
    parser.add_argument('--input_size', default=224, type=int)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--accum_iter', default=1, type=int)
    parser.add_argument('--drop_path', type=float, default=0.1)

    parser.add_argument('--clip_grad', type=float, default=None)
    parser.add_argument('--weight_decay', type=float, default=0.05)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--blr', type=float, default=5e-4)
    parser.add_argument('--layer_decay', type=float, default=0.75)
    parser.add_argument('--min_lr', type=float, default=1e-6)
    parser.add_argument('--warmup_epochs', type=int, default=5)

    # Augmentations
    parser.add_argument('--color_jitter', default=None, type=float)
    parser.add_argument('--aa', default='rand-m9-mstd0.5-inc1', type=str)
    parser.add_argument('--smoothing', default=0.1, type=float)
    parser.add_argument('--mixup', default=0.0, type=float)
    parser.add_argument('--cutmix', default=0.0, type=float)
    parser.add_argument('--mixup_prob', default=1.0, type=float)
    parser.add_argument('--mixup_switch_prob', default=0.5, type=float)
    parser.add_argument('--mixup_mode', default='batch', type=str)
    parser.add_argument('--cutmix_minmax', type=float, nargs='+', default=None)

    # Pretrained MAE checkpoint
    parser.add_argument('--finetune', default='', type=str)
    parser.add_argument('--global_pool', action='store_true')
    parser.set_defaults(global_pool=True)
    parser.add_argument('--cls_token', action='store_false', dest='global_pool')

    # Dataset / FL setup
    parser.add_argument('--data_set', default='COVID-FL', type=str)
    parser.add_argument('--data_path', default='', type=str)
    parser.add_argument('--nb_classes', default=3, type=int)

    parser.add_argument('--n_clients', default=12, type=int)
    parser.add_argument('--num_local_clients', default=-1, type=int)
    parser.add_argument('--E_epoch', default=1, type=int)
    parser.add_argument('--max_communication_rounds', default=100, type=int)

    parser.add_argument('--split_type', default='split_real', type=str)
    parser.add_argument('--proxy_clients', default=None, type=str)

    parser.add_argument('--disable_eval_during_finetuning', action='store_true')
    parser.add_argument('--evaluate_only_last_round', action='store_true')

    # Distributed / device
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://')

    # Logging / I/O
    parser.add_argument('--log_dir', default='', type=str)
    parser.add_argument('--output_dir', default='', type=str)
    parser.add_argument('--resume', default='', type=str)
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--f_name', default='', type=str)

    parser.add_argument('--pin_mem', action='store_true')
    parser.set_defaults(pin_mem=True)

    # FCA-specific hyperparameters
    parser.add_argument('--lambda_fed', type=float, default=1.0,
                        help='Weight for federated classifier loss')
    parser.add_argument('--lambda_pers', type=float, default=3.0,
                        help='Weight for personalized classifier loss')
    parser.add_argument('--temperature', type=float, default=3.0,
                        help='Temperature for KL divergence in consistency loss')
    
    # FedProx coefficient
    parser.add_argument('--fedprox_mu', type=float, default=0.0,
                        help='FedProx proximal coefficient')

    return parser.parse_args()


# ---------------------------------------------------------------------
# Build FCA Model
# ---------------------------------------------------------------------
def build_fca_model(args):
    print(f"Building FCA model: {args.model}")

    # Build base ViT
    base_model = models_vit.__dict__[args.model](
        num_classes=args.nb_classes,
        drop_path_rate=args.drop_path,
        global_pool=args.global_pool,
    )

    # Load MAE pretrain
    if args.finetune and os.path.isfile(args.finetune):
        print(f"Loading MAE pretrain from {args.finetune}")
        checkpoint = torch.load(args.finetune, map_location='cpu')
        state_dict = checkpoint.get("model", checkpoint)
        new_state = state_dict.copy()

        # Adjust pos_embed if needed
        if "pos_embed" in new_state:
            ckpt_pos = new_state["pos_embed"]
            model_pos = base_model.pos_embed
            if ckpt_pos.shape != model_pos.shape:
                print(f"Adjusting pos_embed: ckpt={ckpt_pos.shape}, model={model_pos.shape}")
                grid = ckpt_pos[:, 1:]
                old_n = int(grid.shape[1] ** 0.5)
                new_n = int(base_model.patch_embed.num_patches ** 0.5)
                if old_n != new_n:
                    grid = grid.reshape(1, old_n, old_n, -1).permute(0, 3, 1, 2)
                    grid = torch.nn.functional.interpolate(
                        grid, size=(new_n, new_n),
                        mode='bicubic', align_corners=False
                    )
                    grid = grid.permute(0, 2, 3, 1).reshape(1, new_n * new_n, -1)
                new_state["pos_embed"] = grid

        # Keep encoder-related keys only
        allowed_prefixes = ["patch_embed", "pos_embed", "cls_token", "blocks", "norm"]
        filtered_state = {
            k: v for k, v in new_state.items()
            if any(k.startswith(p) for p in allowed_prefixes)
        }

        missing, unexpected = base_model.load_state_dict(filtered_state, strict=False)
        print(f"Loaded MAE encoder. Missing: {len(missing)}, Unexpected: {len(unexpected)}")
    else:
        print("WARNING: no MAE pretrain loaded!")

    # Freeze encoder, unfreeze last blocks
    print("Freezing encoder, unfreezing last 12 blocks + heads")
    for _, p in base_model.named_parameters():
        p.requires_grad = False

    num_blocks = len(base_model.blocks)
    UNFREEZE = 12
    start = max(0, num_blocks - UNFREEZE)

    for name, p in base_model.named_parameters():
        if name.startswith("blocks."):
            idx = int(name.split(".")[1])
            if idx >= start:
                p.requires_grad = True
        if name.startswith("head") or name.startswith("fc_norm") or name.startswith("norm"):
            p.requires_grad = True

    # Wrap in FCA model
    fca_model = FCAModel(base_model, args.nb_classes)
    
    total_params = sum(p.numel() for p in fca_model.parameters())
    trainable_params = sum(p.numel() for p in fca_model.parameters() if p.requires_grad)
    print(f"Trainable: {trainable_params/1e6:.2f}M / {total_params/1e6:.2f}M "
          f"({100.0*trainable_params/total_params:.2f}%)")

    return fca_model


# ---------------------------------------------------------------------
# Compute class frequencies per client
# ---------------------------------------------------------------------
def compute_class_frequencies(data_loader, num_classes):
    """Count class occurrences in dataset and return frequencies"""
    counts = torch.zeros(num_classes)
    total = 0
    
    for _, targets in data_loader:
        for t in targets:
            counts[t.item()] += 1
            total += 1
    
    if total == 0:
        return torch.ones(num_classes) / num_classes
    
    frequencies = counts / total
    # Ensure no zero frequencies
    frequencies = torch.clamp(frequencies, min=1e-6)
    return frequencies


# ---------------------------------------------------------------------
# FCA Training for one client
# ---------------------------------------------------------------------
def train_fca_client(args, model, optimizer, train_loader, device, 
                     class_frequencies, global_state=None, client_name=""):
    """
    Train one client with FCA loss:
    - L_fed: balanced softmax on federated head
    - L_pers: balanced softmax on personalized head
    - L_con: KL divergence between the two heads (consistency)
    """
    model.to(device)
    model.train()
    
    # Balanced softmax losses
    criterion_fed = BalancedSoftmaxLoss(class_frequencies).to(device)
    criterion_pers = BalancedSoftmaxLoss(class_frequencies).to(device)
    
    total_loss = 0.0
    num_batches = 0
    
    for local_ep in range(args.E_epoch):
        for batch_idx, (samples, targets) in enumerate(train_loader):
            samples = samples.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            
            # Forward through both heads
            fed_logits, pers_logits = model.forward_both(samples)
            
            # Loss 1: Federated classifier (balanced softmax)
            loss_fed = criterion_fed(fed_logits, targets)
            
            # Loss 2: Personalized classifier (balanced softmax)
            loss_pers = criterion_pers(pers_logits, targets)
            
            # Loss 3: Consistency regularization (KL divergence)
            # Guide federated classifier with personalized classifier
            # Stop gradient on personalized to prevent it from being affected
            with torch.no_grad():
                pers_soft = F.softmax(pers_logits / args.temperature, dim=1)
            
            fed_log_soft = F.log_softmax(fed_logits / args.temperature, dim=1)
            loss_con = F.kl_div(fed_log_soft, pers_soft, reduction='batchmean')
            loss_con = loss_con * (args.temperature ** 2)
            
            # Combined FCA loss
            loss = (args.lambda_fed * loss_fed + 
                    args.lambda_pers * loss_pers + 
                    loss_con)
            
            # FedProx term (optional)
            if args.fedprox_mu > 0 and global_state is not None:
                prox_term = 0.0
                for name, p in model.named_parameters():
                    if not p.requires_grad:
                        continue
                    if name in global_state:
                        prox_term += (p - global_state[name].to(device)).pow(2).sum()
                loss = loss + 0.5 * args.fedprox_mu * prox_term
            
            optimizer.zero_grad()
            loss.backward()
            
            if args.clip_grad is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
            
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            if batch_idx % 50 == 0:
                print(f"  [{client_name}] Batch {batch_idx}/{len(train_loader)} "
                      f"Loss={loss.item():.4f} (fed={loss_fed.item():.3f}, "
                      f"pers={loss_pers.item():.3f}, con={loss_con.item():.3f})")
    
    avg_loss = total_loss / max(num_batches, 1)
    model.to('cpu')
    return avg_loss


# ---------------------------------------------------------------------
# FCA Validation (use federated head for generalization)
# ---------------------------------------------------------------------
def validate_fca(args, model, data_loader, device, use_personalized=False):
    """
    Validate FCA model.
    - use_personalized=False: use federated head (for global test set)
    - use_personalized=True: use personalized head (for local specialization)
    """
    model.to(device)
    model.eval()
    
    correct = 0
    total = 0
    
    with torch.no_grad():
        for samples, targets in data_loader:
            samples = samples.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            
            logits = model(samples, use_personalized=use_personalized)
            preds = logits.argmax(dim=1)
            
            correct += (preds == targets).sum().item()
            total += targets.size(0)
    
    acc = 100.0 * correct / max(total, 1)
    model.to('cpu')
    return {"acc1": acc}


# ---------------------------------------------------------------------
# Federated Averaging for FCA (only aggregate federated head + backbone)
# ---------------------------------------------------------------------
def fca_average_model(args, global_model, client_models, selected_clients):
    """
    Average only the backbone and federated_head.
    Personalized heads remain local and are NOT aggregated.
    """
    global_state = global_model.state_dict()
    
    # Initialize aggregation dict
    aggregated = {}
    for key in global_state.keys():
        aggregated[key] = torch.zeros_like(global_state[key])
    
    # Weighted average
    for client in selected_clients:
        client_state = client_models[client].state_dict()
        weight = args.clients_weightes.get(client, 1.0 / len(selected_clients))
        
        for key in aggregated.keys():
            # Only aggregate backbone and federated_head
            if 'personalized_head' not in key:
                aggregated[key] += weight * client_state[key].cpu()
    
    # Update global model
    global_model.load_state_dict(aggregated, strict=False)


# ---------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------
def main(args, model):
    misc.init_distributed_mode(args)
    if not hasattr(args, "model_name"):
        args.model_name = args.model

    device = torch.device(args.device)
    misc.fix_random_seeds(args)
    cudnn.benchmark = True

    # Create dataset
    print("Creating dataset and evaluation metrics...")
    create_dataset_and_evalmetrix(args, mode='finetune')
    print("Clients:", args.dis_cvs_files)
    print("Client sample counts:", args.clients_with_len)

    # --- FIX: Initialize the weights dictionary ---
    args.clients_weightes = {}
    # ----------------------------------------------

    # Global validation loader
    if args.disable_eval_during_finetuning:
        data_loader_val = None
    else:
        dataset_val = DatasetFLFinetune(args=args, phase='test')
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)
        data_loader_val = torch.utils.data.DataLoader(
            dataset_val,
            sampler=sampler_val,
            batch_size=args.batch_size,
            num_workers=8,
            pin_memory=args.pin_mem,
            drop_last=False,
        )

    # Eval-only mode
    if args.eval:
        if not args.resume:
            print("ERROR: --eval requires --resume")
            sys.exit(1)
        model.to(device)
        test_stats = validate_fca(args, model, data_loader_val, device)
        print(f"Test Acc: {test_stats['acc1']:.2f}%")
        return

    # Setup LR
    if args.lr is None:
        args.lr = args.blr * args.batch_size / 256
        print(f"Effective LR = {args.lr:.6f}")

    # Initialize client models and optimizers
    print("Initializing FCA clients...")
    model_all = {}
    optimizer_all = {}
    
    for client in args.dis_cvs_files:
        model_all[client] = deepcopy(model)
        
        # Optimizer for all trainable parameters
        param_groups = [
            {"params": [p for p in model_all[client].parameters() if p.requires_grad]}
        ]
        optimizer_all[client] = torch.optim.AdamW(
            param_groups,
            lr=args.lr,
            betas=(0.9, 0.999),
            weight_decay=args.weight_decay
        )

    # Global model
    model_avg = deepcopy(model).cpu()
    log_writer = TBWriter(args.log_dir) if args.log_dir else None

    # Resume if provided
    start_epoch = 0
    if args.resume and os.path.isfile(args.resume):
        print(f"Loading checkpoint from {args.resume}")
        checkpoint = torch.load(args.resume, map_location='cpu')
        state = checkpoint.get('model', checkpoint)
        model_avg.load_state_dict(state, strict=False)
        start_epoch = checkpoint.get('epoch', 0) + 1

    # Sync initial model
    for c in model_all:
        model_all[c].load_state_dict(model_avg.state_dict())

    print("\n" + "="*60)
    print("FCA FEDERATED TRAINING LOOP")
    print("="*60)
    print(f"Max rounds: {args.max_communication_rounds}")
    print(f"Lambda_fed: {args.lambda_fed}, Lambda_pers: {args.lambda_pers}")
    print(f"Temperature: {args.temperature}, FedProx mu: {args.fedprox_mu}")
    print("="*60 + "\n")

    max_acc = 0.0
    epoch = start_epoch

    while epoch <= args.max_communication_rounds:
        print(f"\n{'='*60}")
        print(f"ROUND {epoch}")
        print(f"{'='*60}\n")

        # Select clients
        if args.num_local_clients == -1:
            selected_clients = args.dis_cvs_files
        else:
            selected_clients = np.random.choice(
                args.dis_cvs_files,
                args.num_local_clients,
                replace=False
            ).tolist()

        # Calculate client weights
        total_len = sum(args.clients_with_len[c] for c in selected_clients)
        for c in selected_clients:
            # This line caused the error because the dict wasn't initialized
            args.clients_weightes[c] = args.clients_with_len[c] / total_len

        print(f"Selected clients: {selected_clients}")

        # Store global state for FedProx
        global_state = None
        if args.fedprox_mu > 0:
            global_state = {
                k: v.detach().clone()
                for k, v in model_avg.state_dict().items()
            }

        # Local training per client
        for c in selected_clients:
            print(f"\n--- Training Client: {c} ---")
            args.single_client = c

            dataset_train = DatasetFLFinetune(args=args, phase='train')
            loader_train = torch.utils.data.DataLoader(
                dataset_train,
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=0,
                pin_memory=args.pin_mem,
                drop_last=True,
            )

            # Compute class frequencies for this client
            class_freq = compute_class_frequencies(loader_train, args.nb_classes)
            print(f"  Class frequencies for {c}: {class_freq.tolist()}")

            # Train with FCA
            avg_loss = train_fca_client(
                args, 
                model_all[c], 
                optimizer_all[c],
                loader_train,
                device,
                class_freq,
                global_state,
                client_name=c
            )
            print(f"  Client {c} avg loss: {avg_loss:.4f}")

        # Federated averaging
        print("\nAggregating global model (backbone + federated head only)...")
        fca_average_model(args, model_avg, model_all, selected_clients)

        # Sync global model to all clients (except personalized heads)
        for cid in model_all:
            client_state = model_all[cid].state_dict()
            global_state = model_avg.state_dict()
            
            # Update only backbone and federated_head
            for key in global_state.keys():
                if 'personalized_head' not in key:
                    client_state[key] = global_state[key]
            
            model_all[cid].load_state_dict(client_state)

        # Save checkpoint
        if args.output_dir:
            ckpt_path = os.path.join(args.output_dir, f"checkpoint-{epoch}.pth")
            torch.save(
                {"model": model_avg.state_dict(), "epoch": epoch},
                ckpt_path
            )

        # Validation (use federated head)
        if data_loader_val is not None:
            test_stats = validate_fca(args, model_avg, data_loader_val, device, 
                                     use_personalized=False)
            acc = test_stats["acc1"]
            print(f"\nValidation Accuracy (Federated Head): {acc:.2f}%")

            if log_writer:
                log_writer.add_scalar('val/acc', acc, epoch)

            if acc > max_acc:
                max_acc = acc
                best_path = os.path.join(args.output_dir, "checkpoint-best.pth")
                torch.save(
                    {"model": model_avg.state_dict(), "epoch": epoch},
                    best_path
                )
                print(f"*** NEW BEST MODEL: {acc:.2f}% ***")

        epoch += 1

    print("\n" + "="*60)
    print("TRAINING FINISHED")
    print(f"Best Accuracy: {max_acc:.2f}%")
    print("="*60)

    if log_writer:
        log_writer.close()


# ---------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------
if __name__ == "__main__":
    args = get_args()

    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    if args.log_dir:
        Path(args.log_dir).mkdir(parents=True, exist_ok=True)

    print("="*60)
    print("Building FCA Model")
    print("="*60)
    model = build_fca_model(args)

    print("\n" + "="*60)
    print("Starting FCA Training")
    print("="*60)
    main(args, model)