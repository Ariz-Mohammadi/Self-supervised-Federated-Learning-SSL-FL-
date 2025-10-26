"""
Centralized TRAINING FROM SCRATCH for Retina dataset using ViT
- This is a DIAGNOSTIC script.
- It trains a ViT model from a random initialization (no pre-trained weights).
- ALL layers are trainable. There is NO layer freezing.
- Uses a weighted loss for class imbalance and an LR scheduler.
"""

import argparse, os, time, datetime, json, math, sys
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd

# Fix import path (go one level up)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import util.misc as misc


# ============================================================
# Dataset (No changes)
# ============================================================
class RetinaCentralDataset(Dataset):
    def __init__(self, csv_path, img_root, train=True, img_size=224):
        self.df = pd.read_csv(csv_path)
        self.paths  = self.df.iloc[:, 0].astype(str).tolist()
        self.labels = self.df.iloc[:, 1].astype(int).tolist()
        self.img_root = img_root
        if train:
            self.tfm = transforms.Compose([
                transforms.RandomResizedCrop(img_size, scale=(0.6, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
            ])
        else:
            self.tfm = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(img_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
            ])
    def __len__(self):
        return len(self.paths)
    def __getitem__(self, idx):
        img_path = os.path.join(self.img_root, self.paths[idx])
        img = Image.open(img_path).convert("RGB")
        return self.tfm(img), self.labels[idx]


# ============================================================
# Training and validation loops
# ============================================================
def train_one_epoch_simple(model, loader, optimizer, device, epoch, log_interval=50):
    model.train()
    # Weighted loss for your 73.48% / 26.52% class imbalance
    class_weights = torch.tensor([1/0.7348, 1/0.2652], device=device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    total_loss, correct, total = 0.0, 0, 0
    for it, (imgs, labels) in enumerate(loader):
        imgs, labels = imgs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        
        preds = model(imgs)
        loss = criterion(preds, labels)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        bs = labels.size(0)
        total_loss += loss.item() * bs
        correct += (preds.argmax(1) == labels).sum().item()
        total += bs

        if log_interval and (it + 1) % log_interval == 0:
            batch_acc = 100.0 * correct / total
            avg_loss  = total_loss / total
            lr = optimizer.param_groups[0]["lr"]
            print(f"[Epoch {epoch} | {it+1}/{len(loader)}] acc={batch_acc:.2f}% loss={avg_loss:.4f} lr={lr:.6f}", flush=True)

    avg_loss = total_loss / total
    train_acc = 100.0 * correct / total
    print(f"Epoch {epoch}: train loss={avg_loss:.4f}  train acc={train_acc:.2f}%", flush=True)
    return {"train_loss": avg_loss, "train_acc": train_acc}


@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    criterion = nn.CrossEntropyLoss() # Validation is unweighted
    total_loss, correct, total = 0.0, 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        preds = model(imgs)
        loss = criterion(preds, labels)
        total_loss += loss.item() * imgs.size(0)
        correct += (preds.argmax(1) == labels).sum().item()
        total += labels.size(0)
    avg_loss = total_loss / total
    acc = 100.0 * correct / total
    return avg_loss, acc


# ============================================================
# LR Scheduler and Warmup
# ============================================================
def adjust_learning_rate(optimizer, epoch, args):
    if epoch < args.warmup_epochs:
        lr = args.lr * (epoch + 1) / args.warmup_epochs
    else:
        lr = args.min_lr + (args.lr - args.min_lr) * 0.5 * \
            (1. + math.cos(math.pi * (epoch - args.warmup_epochs) / (args.epochs - args.warmup_epochs)))
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    return lr


# ============================================================
# Argument parser
# ============================================================
def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="vit_base_patch16")
    p.add_argument("--finetune", type=str, default="", help="THIS SHOULD BE EMPTY FOR THIS SCRIPT")
    p.add_argument("--data_path", default="/cta/users/undergrad2/SSL-FL/data/Retina")
    p.add_argument("--nb_classes", type=int, default=2)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--min_lr", type=float, default=1e-6)
    p.add_argument("--warmup_epochs", type=int, default=10)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--weight_decay", type=float, default=0.05)
    p.add_argument("--output_dir", default="")
    p.add_argument("--device", default="cuda")
    return p.parse_args()


# ============================================================
# Main
# ============================================================
def main(args):
    device = torch.device(args.device)
    print(f"Using device: {device}")

    # This import assumes the 'fed_mae' directory is in the parent path
    import fed_mae.models_vit as models_vit
    model = models_vit.__dict__[args.model](
        num_classes=args.nb_classes,
        drop_path_rate=0.1, # Using a default drop_path
        global_pool=True,   # Using a default global_pool
    )

    # === Replace head with 2-layer MLP ===
    in_features = model.head.in_features
    model.head = nn.Sequential(
        nn.Linear(in_features, 512),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(512, args.nb_classes)
    )
    print("? Replaced head with 2-layer MLP head")

    # === Check for --finetune argument ===
    if args.finetune and os.path.isfile(args.finetune):
        print("!! ERROR: This script is for training from scratch. Do not use the --finetune argument. !!")
        sys.exit(1)
    else:
        print("? Starting training from scratch with a randomly initialized model.")
    
    # =================================================================
    # CRITICAL CHANGE: The layer-freezing block has been completely REMOVED.
    # All layers of the model will be trained.
    # =================================================================
    
    t_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    all_params = sum(p.numel() for p in model.parameters())
    print(f"=> Trainable params: {t_params/1e6:.2f}M / {all_params/1e6:.2f}M ({100*t_params/all_params:.2f}%)")
    model.to(device)

    # === Dataset ===
    # Assumes a combined train.csv and test.csv exist in the data_path
    train_csv = os.path.join(args.data_path, "train.csv") 
    test_csv = os.path.join(args.data_path, "test.csv")
    train_ds = RetinaCentralDataset(train_csv, os.path.join(args.data_path, "train"), train=True)
    val_ds = RetinaCentralDataset(test_csv, os.path.join(args.data_path, "test"), train=False)
    dl_train = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    dl_val = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # === Optimizer ===
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    print(f"=> Using AdamW optimizer with base LR: {args.lr:.6f}")

    start_epoch, best_acc = 0, 0.0

    # === Train loop ===
    for epoch in range(start_epoch, args.epochs):
        print(f"\n===== Epoch {epoch+1}/{args.epochs} =====", flush=True)
        adjust_learning_rate(optimizer, epoch, args)
        
        train_stats = train_one_epoch_simple(model, dl_train, optimizer, device, epoch)
        val_loss, val_acc = validate(model, dl_val, device)
        print(f"Val: acc={val_acc:.2f}%  loss={val_loss:.4f}", flush=True)

        if val_acc > best_acc:
            best_acc = val_acc
            if args.output_dir:
                Path(args.output_dir).mkdir(parents=True, exist_ok=True)
                torch.save({"model": model.state_dict()}, os.path.join(args.output_dir, "checkpoint-best.pth"))
                print(f"? Saved new best checkpoint (acc={best_acc:.2f}%)", flush=True)

    print(f"\n?? Training complete. Best val acc={best_acc:.2f}%", flush=True)

# ============================================================
# Entry
# ============================================================
if __name__ == "__main__":
    args = get_args()
    main(args)