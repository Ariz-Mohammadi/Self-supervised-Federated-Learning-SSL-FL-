# -*- coding: utf-8 -*-
"""
Centralized fine-tuning for COVID-FL dataset (ViT backbone, MAE-pretrained)
- 3-class head + class-weighted CrossEntropy
- Robust pos_embed resize
- CLS-token pooling if model outputs per-token logits
- Two LR param groups (head vs encoder)
- Grad clipping
"""

import os
import math
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

# Make fed_mae importable
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    import fed_mae.models_vit as models_vit
except ImportError:
    print("Error: Could not import 'fed_mae.models_vit'. Check PYTHONPATH.")
    sys.exit(1)


# =========================
# Args (edit here)
# =========================
class Args:
    # Paths (CHANGED for COVID-FL)
    finetune   = "/cta/users/undergrad2/SSL-FL/out_pretrain_covidfl/checkpoint-859.pth"
    data_path  = "/cta/users/undergrad2/SSL-FL/data/COVID-FL"
    output_dir = "/cta/users/undergrad2/SSL-FL/out_finetune_covidfl_central_from_retina_code"

    # Model (CHANGED nb_classes=3)
    model        = "vit_base_patch16"
    nb_classes   = 3            # 3-way classification for COVID-FL
    global_pool  = True
    drop_path    = 0.1
    unfreeze_last= 12           # unfreeze all 12 ViT blocks (+ head + norm)

    # Training
    epochs         = 100        # you can reduce for quick sanity run
    batch_size     = 64
    weight_decay   = 0.05
    base_head_lr   = 1e-3       # LR for head (logits)
    base_enc_lr    = 3e-4       # LR for encoder (ViT blocks)
    warmup_epochs  = 5

    # Sampling / imbalance
    USE_SAMPLER    = False      # False: no sampler, rely on class-weighted CE

    # Device
    device = "cuda"


# =========================
# Dataset
# =========================
class CovidCentralDataset(Dataset):
    """
    Central dataset for COVID-FL.

    Expects:
      - csv_path: CSV with at least 2 columns:
          col0 = relative image path (from img_root)
          col1 = integer label in {0,1,2}
      - img_root: directory where those relative paths live
    """
    def __init__(self, csv_path, img_root, train=True, img_size=224):
        self.df = pd.read_csv(csv_path)
        self.paths  = self.df.iloc[:, 0].astype(str).tolist()
        self.labels = self.df.iloc[:, 1].astype(int).tolist()
        self.img_root = img_root

        if train:
            self.tfm = transforms = __import__("torchvision").transforms.Compose([
                __import__("torchvision").transforms.RandomResizedCrop(img_size, scale=(0.7, 1.0)),
                __import__("torchvision").transforms.RandomHorizontalFlip(p=0.5),
                __import__("torchvision").transforms.RandomApply([
                    __import__("torchvision").transforms.ColorJitter(
                        brightness=0.2, contrast=0.2, saturation=0.2
                    )
                ], p=0.5),
                __import__("torchvision").transforms.RandomAffine(
                    degrees=5, translate=(0.02, 0.02), scale=(0.95, 1.05)
                ),
                __import__("torchvision").transforms.ToTensor(),
                __import__("torchvision").transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
                __import__("torchvision").transforms.RandomErasing(
                    p=0.1, scale=(0.02, 0.08), ratio=(0.3, 3.0), inplace=True
                ),
            ])
        else:
            self.tfm = transforms = __import__("torchvision").transforms.Compose([
                __import__("torchvision").transforms.Resize(256),
                __import__("torchvision").transforms.CenterCrop(img_size),
                __import__("torchvision").transforms.ToTensor(),
                __import__("torchvision").transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
            ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_root, self.paths[idx])
        img = Image.open(img_path).convert("RGB")
        return self.tfm(img), self.labels[idx]


# =========================
# Utility: CLS-pooling
# =========================
def _cls_pool_logits(logits: torch.Tensor) -> torch.Tensor:
    """
    For safety if model returns [B,T,C], use CLS token (index 0).
    Returns [B,C].
    """
    if logits.dim() == 3:       # [B, T, C]
        logits = logits[:, 0, :]  # CLS token
    return logits               # [B,C]


# =========================
# Train / Validate
# =========================
def train_one_epoch_simple(model, loader, optimizer, device, epoch, criterion):
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    for it, (imgs, labels) in enumerate(loader):
        imgs   = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        logits = model(imgs)
        logits = _cls_pool_logits(logits)          # [B,3]
        loss   = criterion(logits, labels.long())

        loss.backward()
        # grad clipping for stability (kept from original)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        running_loss += loss.item() * labels.size(0)
        total += labels.size(0)

        with torch.no_grad():
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()

        if (it + 1) % 50 == 0:
            acc = 100.0 * correct / max(1, total)
            avg_loss = running_loss / max(1, total)
            lr_head = optimizer.param_groups[0]["lr"]
            lr_enc  = optimizer.param_groups[1]["lr"]
            print("[Epoch %d | %d/%d] acc=%.2f%% loss=%.4f lr_head=%.6f lr_enc=%.6f" %
                  (epoch + 1, it + 1, len(loader), acc, avg_loss, lr_head, lr_enc),
                  flush=True)

    epoch_loss = running_loss / max(1, total)
    epoch_acc  = 100.0 * correct / max(1, total)
    print("Epoch %d: train loss=%.4f  train acc=%.2f%%" %
          (epoch + 1, epoch_loss, epoch_acc), flush=True)
    return {"loss": epoch_loss, "acc": epoch_acc}


@torch.no_grad()
def validate(model, loader, device, criterion=None):
    model.eval()
    total_loss, n = 0.0, 0
    correct = 0

    for imgs, labels in loader:
        imgs   = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(imgs)
        logits = _cls_pool_logits(logits)          # [B,3]

        if criterion is not None:
            loss = criterion(logits, labels.long())
            total_loss += loss.item() * labels.size(0)
        n += labels.size(0)

        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()

    val_loss = (total_loss / max(1, n)) if criterion is not None else float("nan")
    val_acc  = 100.0 * correct / max(1, n)
    return val_loss, val_acc


# =========================
# Main
# =========================
def main():
    args = Args()
    device = torch.device(args.device)
    print("Using device:", device)

    # ---- Build model ----
    model = models_vit.__dict__[args.model](
        num_classes=args.nb_classes,
        drop_path_rate=args.drop_path,
        global_pool=args.global_pool
    )
    in_features = model.head.in_features
    model.head = nn.Linear(in_features, args.nb_classes)  # 3 logits for 3 classes
    print("Installed 3-logit linear head for nb_classes=%d" % args.nb_classes)

    # ---- Load MAE checkpoint (encoder weights) ----
    assert isinstance(args.finetune, str) and len(args.finetune) > 0, "Args.finetune is empty."
    assert os.path.isfile(args.finetune), f"Finetune ckpt not found: {args.finetune}"
    print("=> Loading MAE pretrain from", args.finetune)

    ckpt = torch.load(args.finetune, map_location="cpu")
    state = ckpt.get("model", ckpt)
    enc_state = {k: v for k, v in state.items() if not k.startswith("decoder")}

    # --- pos_embed adjust (handles 197 -> 196 tokens etc.) ---
    if "pos_embed" in enc_state and hasattr(model, "pos_embed"):
        pe_ckpt = enc_state["pos_embed"]       # [1, N_ckpt, C]
        pe_curr = model.pos_embed              # [1, N_curr, C]
        if pe_ckpt.shape != pe_curr.shape:
            num_patches_curr = getattr(model.patch_embed, "num_patches", pe_curr.shape[1] - 1)
            num_extra_curr = pe_curr.shape[1] - num_patches_curr  # 0/1

            n_ckpt = pe_ckpt.shape[1]
            has_cls_ckpt = int((n_ckpt - 1) ** 0.5) ** 2 == (n_ckpt - 1)
            if has_cls_ckpt:
                cls_ckpt, patch_ckpt = pe_ckpt[:, :1], pe_ckpt[:, 1:]
            else:
                cls_ckpt, patch_ckpt = None, pe_ckpt

            C = patch_ckpt.shape[-1]
            hw_ckpt = int(patch_ckpt.shape[1] ** 0.5)
            patch_ckpt = patch_ckpt.reshape(1, hw_ckpt, hw_ckpt, C).permute(0, 3, 1, 2)

            hw_curr = int(num_patches_curr ** 0.5)
            patch_resized = F.interpolate(
                patch_ckpt, size=(hw_curr, hw_curr),
                mode="bicubic", align_corners=False
            )
            patch_resized = patch_resized.permute(0, 2, 3, 1).reshape(1, hw_curr * hw_curr, C)

            if num_extra_curr > 0:
                if cls_ckpt is None:
                    cls_ckpt = pe_curr[:, :num_extra_curr]
                pe_new = torch.cat([cls_ckpt[:, :num_extra_curr], patch_resized], dim=1)
            else:
                pe_new = patch_resized

            enc_state["pos_embed"] = pe_new
            print("=> Adjusted pos_embed from", tuple(pe_ckpt.shape), "to", tuple(pe_new.shape))

    msg = model.load_state_dict(enc_state, strict=False)
    try:
        print("[finetune load] missing=%d unexpected=%d" %
              (len(msg.missing_keys), len(msg.unexpected_keys)))
    except Exception:
        missing, unexpected = msg
        print("[finetune load] missing=%d unexpected=%d" %
              (len(missing), len(unexpected)))

    # ---- Freeze -> unfreeze last K + head + norm ----
    print("Freezing layers...")
    for p in model.parameters():
        p.requires_grad = False
    try:
        num_blocks = len(model.blocks)
    except AttributeError:
        num_blocks = 12
    unf_k = max(int(args.unfreeze_last), 0)
    print("Model has %d blocks. Unfreezing the last %d." % (num_blocks, unf_k))
    for name, p in model.named_parameters():
        if (any(name.startswith("blocks.%d" % i) for i in range(num_blocks - unf_k, num_blocks))
            or name.startswith("head") or name.startswith("norm")):
            p.requires_grad = True

    t_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    all_params = sum(p.numel() for p in model.parameters())
    print("=> Trainable params: %.2fM / %.2fM (%.2f%%)" %
          (t_params / 1e6, all_params / 1e6, 100.0 * t_params / max(1, all_params)))
    model.to(device)

    # ---- Data ----
    train_csv = os.path.join(args.data_path, "train.csv")
    test_csv  = os.path.join(args.data_path, "test.csv")  # used as "val" here
    train_ds = CovidCentralDataset(train_csv, os.path.join(args.data_path, "train"), train=True)
    val_ds   = CovidCentralDataset(test_csv,  os.path.join(args.data_path, "test"),  train=False)

    print("=== DEBUG: DATA ===")
    print("Train size=%d  Val size=%d" % (len(train_ds), len(val_ds)))

    # Labels stats for weighting / sampler
    try:
        labels_tensor = torch.as_tensor(train_ds.labels, dtype=torch.long)
    except Exception:
        tmp_loader = DataLoader(train_ds, batch_size=1024, shuffle=False, num_workers=4, pin_memory=True)
        lbls = []
        for _, y in tmp_loader:
            lbls.append(y)
        labels_tensor = torch.cat(lbls).long()

    class_count = torch.bincount(labels_tensor)
    print("class counts (train):", class_count.tolist())
    total_labels = class_count.sum().item()
    print("class ratios:",
          [f"{i}: {class_count[i].item()/max(1,total_labels):.3f}" for i in range(len(class_count))])
    print("====================")

    # ---- Dataloaders (sampler optional) ----
    if Args.USE_SAMPLER:
        class_weight_for_sampling = (class_count.sum().float() / class_count.float())
        sample_weight = class_weight_for_sampling[labels_tensor].cpu().numpy()
        sampler = WeightedRandomSampler(sample_weight, num_samples=len(sample_weight), replacement=True)
        dl_train = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler,
                              num_workers=4, pin_memory=True)
        print("=> Using WeightedRandomSampler")
    else:
        dl_train = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=4, pin_memory=True)

    dl_val = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=4, pin_memory=True)

    # ---- Criterion: class-weighted CrossEntropy ----
    # inverse-frequency style weighting: w_i = N / (K * n_i)
    num_classes = len(class_count)
    class_weights = class_count.sum().float() / (num_classes * class_count.float())
    print("Class weights for CE:", class_weights.tolist())
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))

    # ---- Optimizer (two groups: head vs encoder) ----
    head_params, enc_params = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if n.startswith("head"):
            head_params.append(p)
        else:
            enc_params.append(p)

    optimizer = torch.optim.AdamW(
        [{"params": head_params, "lr": args.base_head_lr},
         {"params": enc_params,  "lr": args.base_enc_lr}],
        weight_decay=args.weight_decay
    )
    print("=> AdamW LRs: head=%.3g, encoder=%.3g, wd=%.3g" %
          (args.base_head_lr, args.base_enc_lr, args.weight_decay))

    head_n = sum(p.numel() for n, p in model.named_parameters()
                 if n.startswith("head") and p.requires_grad)
    enc_n  = sum(p.numel() for n, p in model.named_parameters()
                 if (not n.startswith("head")) and p.requires_grad)
    print("=== DEBUG: OPT ===")
    print("Trainable head params=%s  encoder params=%s" %
          (format(head_n, ","), format(enc_n, ",")))
    print("===================")

    # ---- LR schedule: warmup -> cosine ----
    eta_min_head = args.base_head_lr * 0.1
    eta_min_enc  = args.base_enc_lr  * 0.1

    def set_epoch_lr(ep):
        if ep < args.warmup_epochs:
            f = float(ep + 1) / max(1, args.warmup_epochs)
            lr_h, lr_e = args.base_head_lr * f, args.base_enc_lr * f
        else:
            t = ep - args.warmup_epochs
            T = max(1, args.epochs - args.warmup_epochs)
            cos_t = 0.5 * (1.0 + math.cos(math.pi * t / T))
            lr_h = eta_min_head + (args.base_head_lr - eta_min_head) * cos_t
            lr_e = eta_min_enc  + (args.base_enc_lr  - eta_min_enc)  * cos_t
        optimizer.param_groups[0]["lr"] = lr_h
        optimizer.param_groups[1]["lr"] = lr_e
        return lr_h, lr_e

    # ---- Train ----
    best_acc = -1.0
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    for epoch in range(args.epochs):
        lr_h, lr_e = set_epoch_lr(epoch)
        print("\n===== Epoch %d / %d =====" % (epoch + 1, args.epochs))
        print("epoch %d LRs -> head=%.6g enc=%.6g" % (epoch, lr_h, lr_e))

        _ = train_one_epoch_simple(model, dl_train, optimizer, device, epoch, criterion)
        val_loss, val_acc = validate(model, dl_val, device, criterion)
        print("Val: acc=%.2f%%  loss=%.4f" % (val_acc, val_loss))

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch + 1,
                    "best_acc": best_acc,
                },
                os.path.join(args.output_dir, "checkpoint-best.pth")
            )
            print("? Saved new best checkpoint (val_acc=%.2f%%)" % best_acc)

    print("Training complete. Best val_acc=%.2f%%" % best_acc)


if __name__ == "__main__":
    main()
