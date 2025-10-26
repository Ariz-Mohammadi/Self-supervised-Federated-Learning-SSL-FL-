import argparse, os, time, datetime, torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd
import numpy as np

import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))  # add parent (SSL-FL/code)
import util.misc as misc
import fed_mae.models_vit as models_vit


# ============================================================
# Dataset
# ============================================================
class RetinaCentralDataset(Dataset):
    def __init__(self, csv_path, root_dir, train=True, img_size=224):
        self.root_dir = root_dir
        self.train = train
        self.df = pd.read_csv(csv_path)
        self.paths = self.df.iloc[:, 0].astype(str).tolist()
        self.labels = self.df.iloc[:, 1].astype(int).tolist()

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

    def __getitem__(self, idx):
        img_name = self.paths[idx]
        label = self.labels[idx]
        subfolder = "train" if self.train else "test"
        img_path = os.path.join(self.root_dir, subfolder, img_name)

        if not os.path.isfile(img_path):
            raise FileNotFoundError(f"? Image not found: {img_path}")

        img = Image.open(img_path).convert("RGB")
        img = self.tfm(img)
        return img, label

    def __len__(self):
        return len(self.paths)


# ============================================================
# Train / Eval
# ============================================================
@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total, correct, total_loss = 0, 0, 0.0
    criterion = nn.CrossEntropyLoss()
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        preds = model(imgs)
        loss = criterion(preds, labels)
        total_loss += loss.item() * imgs.size(0)
        correct += (preds.argmax(1) == labels).sum().item()
        total += imgs.size(0)
    return correct / total * 100.0, total_loss / total


def train_one_epoch_simple(model, loader, optimizer, device, epoch, clip_grad=None):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    criterion = nn.CrossEntropyLoss()
    for it, (imgs, labels) in enumerate(loader):
        imgs, labels = imgs.to(device), labels.to(device)
        preds = model(imgs)
        loss = criterion(preds, labels)
        optimizer.zero_grad()
        loss.backward()
        if clip_grad:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        optimizer.step()

        total_loss += loss.item() * imgs.size(0)
        correct += (preds.argmax(1) == labels).sum().item()
        total += imgs.size(0)

    avg_loss = total_loss / total
    acc = correct / total * 100.0
    print(f"Epoch {epoch}: train loss={avg_loss:.4f}  acc={acc:.2f}%")
    return {"train_loss": avg_loss, "train_acc": acc}


# ============================================================
# Warmup + Cosine LR scheduler
# ============================================================
def adjust_learning_rate(optimizer, epoch, args):
    """Warmup for first few epochs, then cosine decay."""
    if epoch < args.warmup_epochs:
        lr = args.lr * (epoch + 1) / args.warmup_epochs
    else:
        progress = (epoch - args.warmup_epochs) / max(1, args.epochs - args.warmup_epochs)
        lr = args.min_lr + 0.5 * (args.lr - args.min_lr) * (1 + np.cos(np.pi * progress))
    for pg in optimizer.param_groups:
        pg["lr"] = lr
    return lr


# ============================================================
# Main
# ============================================================
def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="vit_base_patch16", type=str)
    p.add_argument("--finetune", default="", type=str)
    p.add_argument("--data_path", default="/cta/users/undergrad2/SSL-FL/data/Retina", type=str)
    p.add_argument("--nb_classes", default=2, type=int)
    p.add_argument("--batch_size", default=128, type=int)
    p.add_argument("--lr", default=3e-4, type=float, help="Base learning rate")
    p.add_argument("--min_lr", default=1e-6, type=float)
    p.add_argument("--warmup_epochs", default=10, type=int)
    p.add_argument("--epochs", default=100, type=int)
    p.add_argument("--weight_decay", default=0.05, type=float)
    p.add_argument("--output_dir", default="./out_central", type=str)
    p.add_argument("--device", default="cuda", type=str)
    
    p.add_argument("--seed", default=0, type=int, help="Random seed for reproducibility")
    return p.parse_args()


def main(args):
    device = torch.device(args.device)
    misc.fix_random_seeds(args)

    print("Creating dataset...")
    csv_train = os.path.join(args.data_path, "train.csv")
    csv_test = os.path.join(args.data_path, "test.csv")
    ds_train = RetinaCentralDataset(csv_train, root_dir=args.data_path, train=True)
    ds_val = RetinaCentralDataset(csv_test, root_dir=args.data_path, train=False)
    dl_train = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True, num_workers=4)
    dl_val = DataLoader(ds_val, batch_size=args.batch_size, shuffle=False, num_workers=4)
    print(f"Using CSVs: {csv_train} | {csv_test}")

    # === Build model ===
    model = models_vit.__dict__[args.model](
        num_classes=args.nb_classes, drop_path_rate=0.3, global_pool=True
    )

    # --- Replace classification head with 2-layer MLP ---
    in_features = model.head.in_features
    model.head = nn.Sequential(
        nn.Linear(in_features, 512),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(512, args.nb_classes),
    )
    print("=> Replaced head with 2-layer MLP.")

    # --- Load pretrain if available ---
    if args.finetune and os.path.isfile(args.finetune):
        print(f"=> Loading MAE pretrain from {args.finetune}")
        ckpt = torch.load(args.finetune, map_location="cpu")
        state = ckpt.get("model", ckpt)
        if "pos_embed" in state and model.pos_embed.shape != state["pos_embed"].shape:
            print(f"=> Adjusting pos_embed: checkpoint {state['pos_embed'].shape} vs model {model.pos_embed.shape}")
            pe = state["pos_embed"]
            pe = pe[:, 1:, :]
            num_patches = model.patch_embed.num_patches
            old_size = int(pe.shape[1] ** 0.5)
            new_size = int(num_patches ** 0.5)
            pe = pe.reshape(1, old_size, old_size, -1).permute(0, 3, 1, 2)
            pe = torch.nn.functional.interpolate(pe, size=(new_size, new_size), mode="bicubic", align_corners=False)
            pe = pe.permute(0, 2, 3, 1).reshape(1, new_size * new_size, -1)
            state["pos_embed"] = pe
            print(f"=> Final pos_embed set: {pe.shape}")
        allowed = ["patch_embed", "pos_embed", "cls_token", "blocks", "norm"]
        filtered = {k: v for k, v in state.items() if any(k.startswith(a) for a in allowed)}
        missing, unexpected = model.load_state_dict(filtered, strict=False)
        print(f"[finetune load] missing={len(missing)} unexpected={len(unexpected)}")

    # --- Unfreeze logic ---
    for name, param in model.named_parameters():
        param.requires_grad = False
    for name, param in model.named_parameters():
        if any(name.startswith(f"blocks.{i}") for i in range(10, 12)) or "norm" in name or "head" in name:
            param.requires_grad = True
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"=> Freezing all, then unfreezing last 2 blocks + norms + head ({trainable/1e6:.2f}M / {total/1e6:.2f}M)")

    model.to(device)

    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                                  lr=args.lr, weight_decay=args.weight_decay)
    print(f"=> Using learning rate: {args.lr:.6f}")

    best_acc, start_time = 0.0, time.time()
    for epoch in range(args.epochs):
        print(f"\n===== Epoch {epoch+1} =====", flush=True)
        lr = adjust_learning_rate(optimizer, epoch, args)
        print(f"Current LR: {lr:.6f}", flush=True)

        train_stats = train_one_epoch_simple(model, dl_train, optimizer, device, epoch)
        val_acc, val_loss = evaluate(model, dl_val, device)
        print(f"Val: acc={val_acc:.2f}%  loss={val_loss:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({"model": model.state_dict(), "best_acc": best_acc, "epoch": epoch},
                       os.path.join(args.output_dir, "checkpoint-best.pth"))
            print(f"=> New best acc={best_acc:.2f}%")

    total_time = str(datetime.timedelta(seconds=int(time.time()-start_time)))
    print(f"Training done in {total_time} | Best acc={best_acc:.2f}%")


# ============================================================
if __name__ == "__main__":
    args = get_args()
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
    main(args)
