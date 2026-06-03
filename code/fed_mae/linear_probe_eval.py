# -*- coding: utf-8 -*-
"""
Linear probing for MAE-pretrained encoders (Retina/CXR/etc.)

- Loads MAE encoder weights, resizes pos_embed if needed
- Freezes encoder; trains only a linear head
- CSV can be:
    * 2 columns: path,label
    * 1 column:  path (label inferred from first directory in path)
- Supports RGB or grayscale, image files and .npy tensors
- Optional class-balanced sampler or pos_weight (binary)
"""

import os, sys, math, argparse
from pathlib import Path
from typing import List, Tuple
import numpy as np
import pandas as pd
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True  # tolerate truncated files

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms

from sklearn.metrics import roc_auc_score, f1_score, balanced_accuracy_score

# import backbone defs
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import fed_mae.models_vit as models_vit  # noqa: E402


def parse_args():
    ap = argparse.ArgumentParser("Linear probe for MAE encoders")
    ap.add_argument("--checkpoint", required=True, type=str)
    ap.add_argument("--data_path", required=True, type=str, help="Folder with train/, test/, and CSVs")
    ap.add_argument("--output_dir", required=True, type=str)

    ap.add_argument("--model", default="vit_base_patch16", type=str)
    ap.add_argument("--nb_classes", default=2, type=int)  # 2 = binary (1 logit), >=3 = multi-class
    ap.add_argument("--global_pool", default=True, action="store_true")
    ap.add_argument("--no_global_pool", dest="global_pool", action="store_false")
    ap.set_defaults(global_pool=True)
    ap.add_argument("--drop_path", default=0.1, type=float)

    ap.add_argument("--epochs", default=10, type=int)
    ap.add_argument("--batch_size", default=128, type=int)
    ap.add_argument("--lr", default=1e-3, type=float)
    ap.add_argument("--weight_decay", default=0.05, type=float)
    ap.add_argument("--device", default="cuda", type=str)
    ap.add_argument("--num_workers", default=4, type=int)
    ap.add_argument("--img_size", default=224, type=int)

    # data options
    ap.add_argument("--grayscale", default=False, action="store_true", help="1-channel pipeline (e.g., CXR)")
    ap.add_argument("--balance_sampler", default=False, action="store_true", help="Use class-balanced sampler")
    ap.add_argument("--pos_weight", default=None, type=float, help="Binary only: BCEWithLogits pos_weight")
    return ap.parse_args()


# ----------------------------
# Robust image/.npy loader
# ----------------------------
ALLOWED_IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
ALLOWED_NPY_EXTS = {".npy"}

def load_any(path: str, grayscale: bool) -> Image.Image:
    ext = os.path.splitext(path)[1].lower()
    if ext in ALLOWED_IMG_EXTS:
        img = Image.open(path)
        img = img.convert("L") if grayscale else img.convert("RGB")
        return img
    elif ext in ALLOWED_NPY_EXTS:
        arr = np.load(path)
        # expected shapes: (H,W), (H,W,1), (H,W,3), or (C,H,W)
        if arr.ndim == 2:
            # (H,W) grayscale
            if grayscale:
                arr2 = arr
            else:
                arr2 = np.stack([arr]*3, axis=-1)  # to RGB
        elif arr.ndim == 3:
            if arr.shape[0] in (1,3) and arr.shape[1] > 5 and arr.shape[2] > 5:
                # (C,H,W) -> (H,W,C)
                arr2 = np.transpose(arr, (1,2,0))
            else:
                # (H,W,C)
                arr2 = arr
            if grayscale and arr2.shape[-1] == 3:
                # convert RGB -> gray
                arr2 = (0.2989*arr2[...,0] + 0.5870*arr2[...,1] + 0.1140*arr2[...,2]).astype(arr2.dtype)
        else:
            raise ValueError(f"Unsupported npy shape {arr.shape} for {path}")

        # scale to uint8 if float
        if np.issubdtype(arr2.dtype, np.floating):
            mn, mx = np.nanmin(arr2), np.nanmax(arr2)
            if not np.isfinite(mn) or not np.isfinite(mx) or mx <= mn:
                arr2 = np.nan_to_num(arr2, nan=0.0)
                mn, mx = float(arr2.min()), float(arr2.max())
                if mx <= mn:  # fallback
                    arr2 = np.clip(arr2, 0.0, 1.0)
                    mn, mx = 0.0, 1.0
            arr2 = (255.0 * (arr2 - mn) / (mx - mn + 1e-8)).astype(np.uint8)
        elif arr2.dtype != np.uint8:
            # try to fit into uint8
            arr2 = np.clip(arr2, 0, 255).astype(np.uint8)

        if arr2.ndim == 2:
            mode = "L" if grayscale else "L"
        else:
            mode = "L" if (grayscale and arr2.shape[-1] == 1) else "RGB"
            if grayscale and arr2.ndim == 3 and arr2.shape[-1] > 1:
                # if still multi-channel in grayscale mode, collapse
                arr2 = (0.2989*arr2[...,0] + 0.5870*arr2[...,1] + 0.1140*arr2[...,2]).astype(np.uint8)
                mode = "L"
        return Image.fromarray(arr2, mode=mode)
    else:
        raise ValueError(f"Unsupported file extension: {ext} ({path})")


# ----------------------------
# Dataset
# ----------------------------
class RetinaCentralDataset(Dataset):
    """
    CSV:
      - 2 columns: path,label
      - 1 column : path (label inferred from first directory)
    """
    def __init__(self, csv_path: str, img_root: str, train: bool, img_size: int = 224, grayscale: bool = False):
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV not found: {csv_path}")
        
        df = pd.read_csv(csv_path, header=None, sep=None, engine="python")
            
        if df.shape[1] < 1:
            raise RuntimeError(f"CSV '{csv_path}' is empty.")

        paths = df.iloc[:, 0].astype(str).tolist()
        # keep only supported extensions
        keep = []
        for p in paths:
            ext = os.path.splitext(p)[1].lower()
            if (ext in ALLOWED_IMG_EXTS) or (ext in ALLOWED_NPY_EXTS):
                keep.append(p)
        dropped = len(paths) - len(keep)
        if dropped > 0:
            print(f"[dataset] Dropped {dropped} unsupported files from {csv_path} (kept {len(keep)}).")
        self.paths = keep

        # labels
        if df.shape[1] >= 2:
            labels_all = df.iloc[:, 1].astype(int).tolist()
            self.labels = [labels_all[i] for i, p in enumerate(paths) if p in self.paths]
        else:
            labs = []
            for p in self.paths:
                first = p.split("/")[0].split("\\")[0]
                try:
                    labs.append(int(first))
                except:
                    labs.append(0 if first.lower() in {"0","neg","normal","nodr"} else 1)
            self.labels = labs

        self.img_root = img_root
        self.grayscale = grayscale

        mean_rgb = [0.485, 0.456, 0.406]
        std_rgb  = [0.229, 0.224, 0.225]
        mean_gray = [0.5]
        std_gray  = [0.25]

        if train:
            aug = [
                transforms.RandomResizedCrop(img_size, scale=(0.7, 1.0)),
                transforms.RandomHorizontalFlip(0.5),
                transforms.RandomApply([transforms.ColorJitter(0.2,0.2,0.2)], p=0.5),
                transforms.RandomAffine(degrees=5, translate=(0.02,0.02), scale=(0.95,1.05)),
            ]
        else:
            aug = [transforms.Resize(max(256, img_size)), transforms.CenterCrop(img_size)]

        # to tensor + norm
        if grayscale:
            aug += [transforms.Grayscale(num_output_channels=1), transforms.ToTensor(),
                    transforms.Normalize(mean=mean_gray, std=std_gray)]
        else:
            aug += [transforms.ToTensor(), transforms.Normalize(mean=mean_rgb, std=std_rgb)]

        if train:
            aug.append(transforms.RandomErasing(p=0.1, scale=(0.02,0.08), ratio=(0.3,3.0), inplace=True))

        self.tfm = transforms.Compose(aug)

        print(f"[dataset] {csv_path}: {len(self.paths)} samples | grayscale={grayscale} | img_root={img_root}")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx: int):
        path_rel = self.paths[idx]
        full = os.path.join(self.img_root, path_rel)
        try:
            img = load_any(full, self.grayscale)
        except Exception as e:
            raise RuntimeError(f"Failed to load {full}: {e}")
        x = self.tfm(img)
        y = self.labels[idx]
        return x, y


# ----------------------------
# Positional embedding resize
# ----------------------------
def maybe_resize_pos_embed(enc_state, model):
    if "pos_embed" not in enc_state or not hasattr(model, "pos_embed"):
        return enc_state
    pe_ckpt = enc_state["pos_embed"]
    pe_curr = model.pos_embed
    if pe_ckpt.shape == pe_curr.shape:
        return enc_state

    n_ckpt = pe_ckpt.shape[1]
    has_cls_ckpt = int((n_ckpt - 1) ** 0.5) ** 2 == (n_ckpt - 1)
    if has_cls_ckpt:
        cls_ckpt, patch_ckpt = pe_ckpt[:, :1], pe_ckpt[:, 1:]
    else:
        cls_ckpt, patch_ckpt = None, pe_ckpt
    C = patch_ckpt.shape[-1]
    hw_ckpt = int(patch_ckpt.shape[1] ** 0.5)
    patch_ckpt = patch_ckpt.reshape(1, hw_ckpt, hw_ckpt, C).permute(0, 3, 1, 2)

    num_patches_curr = getattr(model.patch_embed, "num_patches", pe_curr.shape[1] - 1)
    hw_curr = int(num_patches_curr ** 0.5)
    patch_resized = F.interpolate(patch_ckpt, size=(hw_curr, hw_curr), mode="bicubic", align_corners=False)
    patch_resized = patch_resized.permute(0, 2, 3, 1).reshape(1, hw_curr * hw_curr, C)
    num_extra_curr = pe_curr.shape[1] - num_patches_curr  # 0 or 1

    if num_extra_curr > 0:
        if cls_ckpt is None:
            cls_ckpt = pe_curr[:, :num_extra_curr]
        pe_new = torch.cat([cls_ckpt[:, :num_extra_curr], patch_resized], dim=1)
    else:
        pe_new = patch_resized
    enc_state["pos_embed"] = pe_new
    print("=> Adjusted pos_embed from", tuple(pe_ckpt.shape), "to", tuple(pe_new.shape))
    return enc_state


# ----------------------------
# Train / Validate
# ----------------------------
def train_one_epoch(model, loader, optimizer, criterion, device, epoch, binary: bool):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for it, (imgs, labels) in enumerate(loader):
        imgs = imgs.to(device, non_blocking=True)
        labels = torch.as_tensor(labels, device=device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(imgs)

        if binary:
            logits = logits.squeeze(1)
            loss = criterion(logits, labels.float())
            preds = (torch.sigmoid(logits) > 0.5).long()
        else:
            loss = criterion(logits, labels.long())
            preds = torch.argmax(logits, dim=1)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * imgs.size(0)
        correct += (preds == labels).sum().item()
        total += imgs.size(0)

        if (it + 1) % 50 == 0:
            print(f"[Epoch {epoch} | {it+1}/{len(loader)}] loss={loss.item():.4f}", flush=True)

    acc = 100.0 * correct / max(1, total)
    return running_loss / max(1, total), acc


@torch.no_grad()
def validate(model, loader, criterion, device, binary: bool, nb_classes: int):
    model.eval()
    total_loss, n = 0.0, 0
    all_probs, all_targets = [], []
    all_preds = []

    for imgs, labels in loader:
        imgs = imgs.to(device, non_blocking=True)
        labels = torch.as_tensor(labels, device=device)
        logits = model(imgs)

        if binary:
            logits = logits.squeeze(1)
            loss = criterion(logits, labels.float())
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).long()
            all_probs.append(probs.detach().cpu())
        else:
            loss = criterion(logits, labels.long())
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)
            all_probs.append(probs.detach().cpu())

        total_loss += loss.item() * labels.size(0)
        n += labels.size(0)
        all_targets.append(labels.detach().cpu())
        all_preds.append(preds.detach().cpu())

    y_true = torch.cat(all_targets).numpy()
    y_pred = torch.cat(all_preds).numpy()
    val_loss = total_loss / max(1, n)

    if binary:
        y_prob = torch.cat(all_probs).numpy().astype(float)
        try:
            val_auc = roc_auc_score(y_true, y_prob)
        except Exception:
            val_auc = float("nan")
        val_f1 = f1_score(y_true, y_pred, zero_division=0)
        val_bacc = balanced_accuracy_score(y_true, y_pred)
        val_acc = (y_pred == y_true).mean() * 100.0
    else:
        y_prob = torch.cat(all_probs).numpy().astype(float)
        try:
            val_auc = roc_auc_score(y_true, y_prob, multi_class="ovr")
        except Exception:
            val_auc = float("nan")
        val_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
        val_bacc = balanced_accuracy_score(y_true, y_pred)
        val_acc = (y_pred == y_true).mean() * 100.0

    return val_loss, val_acc, val_auc, val_f1, val_bacc


def main():
    args = parse_args()
    device = torch.device(args.device)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    print("Using device:", device)

    binary = (args.nb_classes == 2)

    # Build model (timm head is our linear probe)
    model = models_vit.__dict__[args.model](
        num_classes=(1 if binary else args.nb_classes),
        drop_path_rate=args.drop_path,
        global_pool=args.global_pool,
    )

    # Load MAE encoder
    print("=> Loading pretrained encoder from", args.checkpoint)
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    state = ckpt.get("model", ckpt)
    enc_state = {k: v for k, v in state.items() if not k.startswith("decoder")}
    enc_state = maybe_resize_pos_embed(enc_state, model)
    msg = model.load_state_dict(enc_state, strict=False)
    try:
        print(f"[load] missing={len(msg.missing_keys)} unexpected={len(msg.unexpected_keys)}")
        if msg.missing_keys:   print("  missing:", msg.missing_keys[:8], ("..." if len(msg.missing_keys)>8 else ""))
        if msg.unexpected_keys:print("  unexpected:", msg.unexpected_keys[:8], ("..." if len(msg.unexpected_keys)>8 else ""))
    except Exception:
        pass

    # Freeze encoder
    for name, p in model.named_parameters():
        if not name.startswith("head"):
            p.requires_grad = False
    print("=> Encoder frozen. Only head is trainable.")
    model.to(device)

    # Data
    train_csv = os.path.join(args.data_path, "train.csv")
    test_csv  = os.path.join(args.data_path, "test.csv")
    train_root = os.path.join(args.data_path, "train")
    test_root  = os.path.join(args.data_path, "test")

    train_ds = RetinaCentralDataset(train_csv, train_root, train=True,  img_size=args.img_size, grayscale=args.grayscale)
    val_ds   = RetinaCentralDataset(test_csv,  test_root,  train=False, img_size=args.img_size, grayscale=args.grayscale)

    # Optional balanced sampling
    if args.balance_sampler:
        labels = np.array(train_ds.labels)
        class_counts = np.bincount(labels)
        class_counts = np.maximum(class_counts, 1)
        weights = 1.0 / class_counts[labels]
        sampler = WeightedRandomSampler(weights=weights, num_samples=len(weights), replacement=True)
        dl_train = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler,
                              num_workers=args.num_workers, pin_memory=True, drop_last=False)
    else:
        dl_train = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True, drop_last=False)

    dl_val = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True, drop_last=False)

    # Loss/opt
    if binary:
        if args.pos_weight is not None:
            criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([args.pos_weight], device=device))
        else:
            criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(model.head.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Train
    best_metric = -float("inf")
    best_path = os.path.join(args.output_dir, "checkpoint-best.pth")

    for epoch in range(1, args.epochs + 1):
        print(f"\n===== Epoch {epoch}/{args.epochs} =====", flush=True)
        train_loss, train_acc = train_one_epoch(model, dl_train, optimizer, criterion, device, epoch, binary)
        val_loss, val_acc, val_auc, val_f1, val_bacc = validate(model, dl_val, criterion, device, binary, args.nb_classes)

        if binary:
            print(f"Train: loss={train_loss:.4f}, acc={train_acc:.2f}% | "
                  f"Val: loss={val_loss:.4f}, acc={val_acc:.2f}% | "
                  f"AUC={val_auc:.3f} | F1={val_f1:.3f} | BalAcc={val_bacc:.3f}", flush=True)
            metric_for_ckpt = 0.0 if math.isnan(val_auc) else val_auc
        else:
            print(f"Train: loss={train_loss:.4f}, acc={train_acc:.2f}% | "
                  f"Val: loss={val_loss:.4f}, acc={val_acc:.2f}% | "
                  f"macroAUC={val_auc:.3f} | macroF1={val_f1:.3f} | BalAcc={val_bacc:.3f}", flush=True)
            metric_for_ckpt = 0.0 if math.isnan(val_auc) else val_auc

        if metric_for_ckpt > best_metric:
            best_metric = metric_for_ckpt
            torch.save({"model": model.state_dict(), "best_metric": best_metric,
                        "binary": binary, "nb_classes": args.nb_classes},
                       best_path)
            print(f"=> Saved new best model to {best_path} (metric={best_metric:.4f})", flush=True)

    print(f"\n✓ Linear probe done. Best metric = {best_metric:.4f}  |  saved: {best_path}")


if __name__ == "__main__":
    main()
