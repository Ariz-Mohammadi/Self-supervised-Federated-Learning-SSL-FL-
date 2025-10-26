# --------------------------------------------------------
# Centralized fine-tuning for MAE-pretrained ViT
# Based on Fed-MAE structure (single central training version)
# --------------------------------------------------------

import argparse, os, json, datetime, time
from pathlib import Path
from copy import deepcopy

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import fed_mae.models_vit as models_vit
import util.misc as misc



# ===================== Dataset =====================
class RetinaCentralDataset(Dataset):
    """
    Expects a CSV with two columns:
      col 0: image name (e.g. 27970_left.png)
      col 1: integer label (0 or 1)
    """
    def __init__(self, csv_path, root_dir, train=True, img_size=224):
        if not os.path.isfile(csv_path):
            raise FileNotFoundError(f"CSV not found: {csv_path}")
        self.df = pd.read_csv(csv_path)
        self.root_dir = root_dir

        if self.df.shape[1] < 2:
            raise ValueError(f"CSV must have at least 2 columns (path,label). Got shape={self.df.shape} @ {csv_path}")
        self.paths = self.df.iloc[:, 0].astype(str).tolist()
        self.labels = self.df.iloc[:, 1].astype(int).tolist()

        if train:
            self.tfm = transforms.Compose([
                transforms.RandomResizedCrop(img_size, scale=(0.6, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
            ])
        else:
            self.tfm = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(img_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
            ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.paths[idx])
        if not os.path.isfile(img_path):
            raise FileNotFoundError(f"? Image not found: {img_path}")
        img = Image.open(img_path).convert("RGB")
        return self.tfm(img), self.labels[idx]


# ===================== Training & Validation =====================
def train_one_epoch_simple(model, loader, optimizer, device, epoch, clip_grad=None):
    model.train()
    loss_fn = nn.CrossEntropyLoss()
    total_loss = 0
    for it, (imgs, labels) in enumerate(loader):
        imgs, labels = imgs.to(device, non_blocking=True), labels.to(device)
        logits = model(imgs)
        loss = loss_fn(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        if clip_grad:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        optimizer.step()

        total_loss += loss.item()
    avg_loss = total_loss / len(loader)
    print(f"Epoch {epoch}: train loss={avg_loss:.4f}")
    return {"loss": avg_loss}


@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    correct, total, total_loss = 0, 0, 0
    loss_fn = nn.CrossEntropyLoss()
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        logits = model(imgs)
        loss = loss_fn(logits, labels)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.numel()
        total_loss += loss.item()
    acc = 100.0 * correct / total
    avg_loss = total_loss / len(loader)
    print(f"Val: acc={acc:.2f}%  loss={avg_loss:.4f}")
    return {"acc": acc, "loss": avg_loss}


# ===================== Main =====================
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    misc.fix_random_seeds(args)
    print("Device:", device)

    # ====== Load model ======
    model = models_vit.__dict__[args.model](
        num_classes=args.nb_classes,
        drop_path_rate=args.drop_path,
        global_pool=args.global_pool,
    )

    # ====== Load pretrain ======
    if args.finetune and os.path.isfile(args.finetune):
        print(f"=> Loading MAE pretrain from {args.finetune}")
        ckpt = torch.load(args.finetune, map_location="cpu")
        state = ckpt.get("model", ckpt)
        new_state = state.copy()

        if "pos_embed" in new_state:
            ckpt_pos_embed = new_state["pos_embed"]
            model_pos_embed = model.pos_embed
            if ckpt_pos_embed.shape != model_pos_embed.shape:
                print(f"=> Adjusting pos_embed: checkpoint {tuple(ckpt_pos_embed.shape)} vs model {tuple(model_pos_embed.shape)}")
                grid_pos_embed = ckpt_pos_embed[:, 1:]  # remove CLS
                num_patches = model.patch_embed.num_patches
                old_size = int(grid_pos_embed.shape[1] ** 0.5)
                new_size = int(num_patches ** 0.5)

                def resize_pos_embed(posemb, gs_old, gs_new):
                    posemb_grid = posemb.reshape(1, gs_old, gs_old, -1).permute(0, 3, 1, 2)
                    posemb_grid = torch.nn.functional.interpolate(
                        posemb_grid, size=(gs_new, gs_new), mode='bicubic', align_corners=False
                    )
                    posemb_grid = posemb_grid.permute(0, 2, 3, 1).reshape(1, gs_new * gs_new, -1)
                    return posemb_grid

                grid_pos_embed = resize_pos_embed(grid_pos_embed, old_size, new_size)
                new_state["pos_embed"] = grid_pos_embed
                print("=> Final pos_embed set:", tuple(new_state["pos_embed"].shape))

        allowed_prefixes = ["patch_embed", "pos_embed", "cls_token", "blocks", "norm"]
        filtered_state = {k: v for k, v in new_state.items() if any(k.startswith(p) for p in allowed_prefixes)}
        missing, unexpected = model.load_state_dict(filtered_state, strict=False)
        print(f"[finetune load] missing={len(missing)} unexpected={len(unexpected)}")
        print("  missing (first 10):", missing[:10])
        print("  unexpected (first 10):", unexpected[:10])
    else:
        print("WARNING: no pretrain loaded, training from scratch!")

    # ====== Freeze/Unfreeze policy ======
    print("=> Freezing all, then unfreezing last 2 blocks + all LayerNorms + head")
    for p in model.parameters():
        p.requires_grad = False
    for name, p in model.named_parameters():
        if "blocks.22" in name or "blocks.23" in name or "norm" in name or "head" in name:
            p.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"=> Trainable params: {trainable/1e6:.2f}M / {total/1e6:.2f}M ({100*trainable/total:.2f}%)")

    model.to(device)

    # ====== Datasets ======
    train_csv = os.path.join(args.data_path, "train.csv")
    test_csv  = os.path.join(args.data_path, "test.csv")
    print(f"Using CSVs: {train_csv} | {test_csv}")

    ds_train = RetinaCentralDataset(train_csv, os.path.join(args.data_path, "train"), train=True)
    ds_test  = RetinaCentralDataset(test_csv, os.path.join(args.data_path, "test"), train=False)

    dl_train = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    dl_test  = DataLoader(ds_test, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # ====== Optimizer ======
    args.lr = args.blr * args.batch_size / 256
    print(f"=> Base LR (blr): {args.blr:.6f} | Effective LR: {args.lr:.6f}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # ====== Resume checkpoint ======
    start_epoch = 0
    best_acc = 0.0
    if hasattr(args, "resume") and args.resume and os.path.isfile(args.resume):
        print(f"=> Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(checkpoint["model"], strict=False)
        if "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = checkpoint.get("epoch", 0) + 1
        best_acc = checkpoint.get("best_acc", 0.0)
        print(f"=> Resumed from epoch {start_epoch} (best_acc={best_acc:.2f}%)")
    else:
        print("=> No resume checkpoint provided, starting fresh.")

    # ====== Training loop ======
    for epoch in range(start_epoch, args.epochs):
        print(f"\n===== Epoch {epoch} =====")
        train_stats = train_one_epoch_simple(model, dl_train, optimizer, device, epoch, clip_grad=args.clip_grad)
        val_stats = validate(model, dl_test, device)

        if val_stats["acc"] > best_acc:
            best_acc = val_stats["acc"]
            torch.save({
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "best_acc": best_acc,
            }, os.path.join(args.output_dir, "checkpoint-best.pth"))
            print(f"=> New best model saved ({best_acc:.2f}%)")

        # periodic checkpoint
        if (epoch + 1) % 10 == 0:
            torch.save({
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "best_acc": best_acc,
            }, os.path.join(args.output_dir, f"checkpoint-{epoch}.pth"))

    print(f"Training complete. Best accuracy: {best_acc:.2f}%")
    print("Total time:", str(datetime.timedelta(seconds=int(time.time()))))


# ===================== Arg Parser =====================
def get_args():
    parser = argparse.ArgumentParser("Central fine-tuning", add_help=False)
    parser.add_argument("--model", default="vit_base_patch16")
    parser.add_argument("--finetune", default="", type=str)
    parser.add_argument("--resume", default="", type=str)
    parser.add_argument("--data_set", default="Retina")
    parser.add_argument("--data_path", default="/cta/users/undergrad2/SSL-FL/data/Retina")
    parser.add_argument("--batch_size", default=64, type=int)
    parser.add_argument("--epochs", default=100, type=int)
    parser.add_argument("--blr", default=5e-4, type=float)
    parser.add_argument("--weight_decay", default=0.05, type=float)
    parser.add_argument("--drop_path", default=0.3, type=float)
    parser.add_argument("--nb_classes", default=2, type=int)
    parser.add_argument("--global_pool", action="store_true")
    parser.add_argument("--clip_grad", default=None, type=float)
    parser.add_argument("--output_dir", default="./out_finetune_retina_central_base")
    parser.add_argument("--log_dir", default="./out_finetune_retina_central_base/tb")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", default=0, type=int)
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
