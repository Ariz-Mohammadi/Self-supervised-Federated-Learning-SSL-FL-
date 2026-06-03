# -*- coding: utf-8 -*-

import argparse
import datetime
import json
import numpy as np
import os
import sys
import time
from copy import deepcopy
from pathlib import Path
import re

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn

from torch.utils.tensorboard import SummaryWriter

# Simple TB wrapper so engine_for_finetuning can use log_writer.add_scalars(...)
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


# --------------------------------------------------------
# project imports
# --------------------------------------------------------
current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import fed_mae.models_vit as models_vit
from fed_mae.engine_for_finetuning import train_one_epoch
from util.FedAvg_utils import Partial_Client_Selection, valid, average_model
from util.data_utils import DatasetFLFinetune, create_dataset_and_evalmetrix
import util.misc as misc


# --------------------------------------------------------
# argument parser
# --------------------------------------------------------
def get_args():
    parser = argparse.ArgumentParser(
        'Fed-MAE finetuning on COVID-FL with FedAvg',
        add_help=False
    )

    # General
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--accum_iter', default=1, type=int)
    parser.add_argument('--model', default='vit_base_patch16', type=str)
    parser.add_argument('--input_size', default=224, type=int)
    parser.add_argument('--drop_path', type=float, default=0.1)

    parser.add_argument('--disable_eval_during_finetuning', action='store_true')
    parser.add_argument('--evaluate_only_last_round', action='store_true')

    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--seed', default=0, type=int)

    parser.add_argument('--log_dir', default='', type=str)
    parser.add_argument('--output_dir', default='', type=str)

    parser.add_argument('--resume', default='', type=str)
    parser.add_argument('--eval', action='store_true')

    # FedAvg specifics
    parser.add_argument('--n_clients', default=5, type=int)
    parser.add_argument('--num_local_clients', default=-1, type=int)
    parser.add_argument('--E_epoch', default=1, type=int)
    parser.add_argument('--max_communication_rounds', default=200, type=int)

    # Optimizer
    parser.add_argument('--clip_grad', type=float, default=None)
    parser.add_argument('--weight_decay', type=float, default=0.05)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--blr', type=float, default=5e-4)
    parser.add_argument('--layer_decay', type=float, default=0.75)
    parser.add_argument('--min_lr', type=float, default=1e-6)
    parser.add_argument('--warmup_epochs', type=int, default=5)

    # Augmentation
    parser.add_argument('--color_jitter', default=None, type=float)
    parser.add_argument('--aa', default='rand-m9-mstd0.5-inc1', type=str)
    parser.add_argument('--smoothing', default=0.1, type=float)
    parser.add_argument('--mixup', default=0.0, type=float)
    parser.add_argument('--cutmix', default=0.0, type=float)
    parser.add_argument('--mixup_prob', default=1.0, type=float)
    parser.add_argument('--mixup_switch_prob', default=0.5, type=float)
    parser.add_argument('--mixup_mode', default='batch', type=str)
    parser.add_argument('--cutmix_minmax', type=float, nargs='+', default=None)

    # Pretrain
    parser.add_argument('--finetune', default='', type=str)
    parser.add_argument('--global_pool', action='store_true')
    parser.set_defaults(global_pool=True)
    parser.add_argument('--cls_token', action='store_false', dest='global_pool')

    # Dataset
    parser.add_argument('--data_set', default='COVID-FL', type=str)
    parser.add_argument('--data_path', default='', type=str)
    parser.add_argument('--nb_classes', default=3, type=int)

    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://')

    parser.add_argument('--f_name', default='', type=str)
    parser.add_argument('--pin_mem', action='store_true')
    parser.set_defaults(pin_mem=True)

    # These exist for compatibility; COVID-FL uses data_path + split_real
    parser.add_argument('--data_path_train', default='', type=str)
    parser.add_argument('--data_path_test', default='', type=str)
    parser.add_argument('--split_type', default='split_real', type=str)
    parser.add_argument('--proxy_clients', default=None, type=str)

    return parser.parse_args()


# --------------------------------------------------------
# main
# --------------------------------------------------------
def main(args, model):
    misc.init_distributed_mode(args)
    device = torch.device(args.device)
    misc.fix_random_seeds(args)
    cudnn.benchmark = True

    # =====================================================
    # create dataset + client metadata
    # =====================================================
    print("Creating dataset and evaluation metrics...")
    create_dataset_and_evalmetrix(args, mode='finetune')

    print("Clients:", args.dis_cvs_files)
    print("Client sample counts:", args.clients_with_len)

    # Validation loader (global test set)
    dataset_val = None if args.disable_eval_during_finetuning else DatasetFLFinetune(args=args, phase='test')
    sampler_val = torch.utils.data.SequentialSampler(dataset_val) if dataset_val else None
    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=args.batch_size, num_workers=8,
        pin_memory=args.pin_mem, drop_last=False
    ) if dataset_val else None

    # =====================================================
    # eval-only mode
    # =====================================================
    if args.eval:
        if not args.resume:
            print("ERROR: --eval requires --resume checkpoint")
            sys.exit(1)
        print("Evaluation only mode.")
        model.to(device)
        test_stats = valid(args, model, data_loader_val)
        print(f"Test Acc: {test_stats['acc1']:.2f}%")
        return

    # =====================================================
    # FedAvg client initialization
    # =====================================================
    if args.lr is None:
        args.lr = args.blr * args.batch_size / 256
        print(f"Effective LR = {args.lr:.6f}")

    print("Initializing clients via Partial_Client_Selection...")
    model_all, optimizer_all, criterion_all, loss_scaler_all, mixup_fn_all = \
        Partial_Client_Selection(args, model, mode="finetune")
    print(f"Initialized {len(model_all)} client models.")

    # =====================================================
    # class-weighted loss for COVID-FL (3 classes)
    # =====================================================
    if args.data_set.lower() == "covid-fl" and args.nb_classes == 3:
        # train split counts (you reported):
        # trainset: 0: 7285, 1: 5237, 2: 3522
        total_counts = torch.tensor([7285.0, 5237.0, 3522.0], device=device)
        class_weights = total_counts.sum() / (3 * total_counts)
        print("Using class weights (COVID-FL train distribution):", class_weights.tolist())

        for client_id in model_all.keys():
            criterion_all[client_id] = nn.CrossEntropyLoss(weight=class_weights)

    # =====================================================
    # global average model (server model)
    # =====================================================
    model_avg = deepcopy(model).cpu()
    log_writer = TBWriter(args.log_dir) if args.log_dir else None

    # =====================================================
    # resume checkpoint (global model)
    # =====================================================
    start_epoch = 0
    if args.resume and os.path.isfile(args.resume):
        print(f"Loading checkpoint from {args.resume}")
        checkpoint = torch.load(args.resume, map_location='cpu')
        state = checkpoint.get('model', checkpoint)
        msg = model_avg.load_state_dict(state, strict=False)
        print("Resume load_state_dict:", msg)
        start_epoch = checkpoint.get('epoch', 0) + 1

    # sync initial global model to all clients
    for c in model_all:
        model_all[c].load_state_dict(model_avg.state_dict())

    # =====================================================
    # FedAvg training loop
    # =====================================================
    print("========== FEDAVG TRAINING LOOP ==========")
    print(f"Max communication rounds: {args.max_communication_rounds}")

    max_acc = 0.0
    epoch = start_epoch

    while epoch <= args.max_communication_rounds:
        print(f"\n===== ROUND {epoch} =====\n")

        # -------------------------------------------------
        # select clients for this round
        # -------------------------------------------------
        if args.num_local_clients == -1:
            selected_clients = args.dis_cvs_files
        else:
            selected_clients = np.random.choice(
                args.dis_cvs_files,
                args.num_local_clients,
                replace=False
            ).tolist()

        # compute FedAvg weights based on client data sizes
        total_len = sum(args.clients_with_len[c] for c in selected_clients)
        for c in selected_clients:
            args.clients_weightes[c] = args.clients_with_len[c] / total_len

        print("Selected clients:", selected_clients)
        print("Client weights:", args.clients_weightes)

        # -------------------------------------------------
        # local training per selected client
        # -------------------------------------------------
        for c in selected_clients:
            args.single_client = c

            dataset_train = DatasetFLFinetune(args=args, phase='train')
            loader_train = torch.utils.data.DataLoader(
                dataset_train,
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=0,      # keep 0 here to avoid worker encoding issues
                pin_memory=args.pin_mem,
                drop_last=True
            )

            model_c = model_all[c]
            optimizer_c = optimizer_all[c]
            criterion_c = criterion_all[c]
            loss_scaler_c = loss_scaler_all[c]
            mixup_fn_c = mixup_fn_all[c]

            for _ in range(args.E_epoch):
                train_stats = train_one_epoch(
                    model_c, criterion_c, loader_train,
                    optimizer_c, device, epoch, loss_scaler_c,
                    args.clip_grad, c, mixup_fn_c,
                    log_writer=log_writer, args=args
                )

        # -------------------------------------------------
        # federated averaging (server aggregation)
        # -------------------------------------------------
        print("Aggregating global model...")
        # IMPORTANT: average_model updates model_avg IN-PLACE (no assignment!)
        average_model(args, model_avg, model_all)

        # sync updated global model back to all clients
        for client_id in model_all:
            model_all[client_id].load_state_dict(model_avg.state_dict())

        # -------------------------------------------------
        # save global checkpoint
        # -------------------------------------------------
        if args.output_dir:
            ckpt_path = os.path.join(args.output_dir, f"checkpoint-{epoch}.pth")
            torch.save(
                {"model": model_avg.state_dict(), "epoch": epoch},
                ckpt_path
            )

        # -------------------------------------------------
        # validation on global model
        # -------------------------------------------------
        if data_loader_val is not None:
            model_avg.to(device)
            test_stats = valid(args, model_avg, data_loader_val)
            acc = test_stats["acc1"]
            print(f"Validation Accuracy: {acc:.2f}%")

            if acc > max_acc:
                max_acc = acc
                best_path = os.path.join(args.output_dir, "checkpoint-best.pth")
                torch.save(
                    {"model": model_avg.state_dict(), "epoch": epoch},
                    best_path
                )
                print(f"New BEST model saved: {acc:.2f}%")

            model_avg.to("cpu")

        epoch += 1

    print("\nTraining finished.")


# --------------------------------------------------------
# model builder (MAE → ViT finetuning)
# --------------------------------------------------------
def build_model(args):
    print(f"Building model: {args.model}")

    # build ViT backbone (classification head will be used)
    model = models_vit.__dict__[args.model](
        num_classes=args.nb_classes,
        drop_path_rate=args.drop_path,
        global_pool=args.global_pool,
    )

    # ----------------------------------------------------
    # load MAE pretraining (encoder) into ViT
    # ----------------------------------------------------
    if args.finetune and os.path.isfile(args.finetune):
        print(f"Loading MAE pretrain checkpoint from {args.finetune}")
        checkpoint = torch.load(args.finetune, map_location='cpu')
        state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint
        new_state = state_dict.copy()

        # fix pos_embed mismatch if needed
        if "pos_embed" in new_state:
            ckpt_pos = new_state["pos_embed"]
            model_pos = model.pos_embed
            if ckpt_pos.shape != model_pos.shape:
                print(f"Adjusting pos_embed: ckpt={ckpt_pos.shape}, model={model_pos.shape}")
                # drop cls token
                grid = ckpt_pos[:, 1:]  # [1, N, C]
                old_n = int(grid.shape[1] ** 0.5)
                new_n = int(model.patch_embed.num_patches ** 0.5)
                if old_n != new_n:
                    grid = grid.reshape(1, old_n, old_n, -1).permute(0, 3, 1, 2)
                    grid = torch.nn.functional.interpolate(
                        grid, size=(new_n, new_n),
                        mode='bicubic', align_corners=False
                    )
                    grid = grid.permute(0, 2, 3, 1).reshape(1, new_n * new_n, -1)
                new_state["pos_embed"] = grid
                print("Final pos_embed shape:", new_state["pos_embed"].shape)

        # keep only encoder-related keys
        allowed_prefixes = ["patch_embed", "pos_embed", "cls_token", "blocks", "norm"]
        filtered_state = {
            k: v for k, v in new_state.items()
            if any(k.startswith(p) for p in allowed_prefixes)
        }

        missing, unexpected = model.load_state_dict(filtered_state, strict=False)
        print(f"Loaded MAE encoder into ViT backbone.")
        print(f"Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}")
        if len(missing) > 0:
            print("  missing (first 10):", missing[:10])
        if len(unexpected) > 0:
            print("  unexpected (first 10):", unexpected[:10])
    else:
        print("WARNING: no MAE pretrain loaded, training from scratch!")

    # ----------------------------------------------------
    # freeze encoder, unfreeze last 12 blocks + head
    # ----------------------------------------------------
    print("Freezing encoder, then unfreezing last 12 blocks + head.")
    for name, p in model.named_parameters():
        p.requires_grad = False

    num_blocks = len(model.blocks)
    UNFREEZE = 12
    start = max(0, num_blocks - UNFREEZE)

    for name, p in model.named_parameters():
        if name.startswith("blocks."):
            idx = int(name.split(".")[1])
            if idx >= start:
                p.requires_grad = True
        if name.startswith("head") or name.startswith("fc_norm"):
            p.requires_grad = True

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable params: {trainable_params/1e6:.2f}M / {total_params/1e6:.2f}M "
          f"({100.0*trainable_params/total_params:.2f}%)")

    return model


# --------------------------------------------------------
# entry point
# --------------------------------------------------------
if __name__ == "__main__":
    args = get_args()
    args.model_name = args.model

    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    if args.log_dir:
        Path(args.log_dir).mkdir(parents=True, exist_ok=True)

    print("==== Building Model ====")
    model = build_model(args)

    print("==== Starting Training ====")
    main(args, model)
