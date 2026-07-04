#!/usr/bin/env python3
"""
train_bg_detector.py — бинарный детектор «Фон vs руда».

Архитектура каскада:
  изображение → frozen_encoder → 2048-эмбеддинг
                     ├─ frozen_grade_head  → [Отал / Рядов / Трудн]
                     └─ trainable_bg_head  → [is_background]

  Финал: если sigmoid(bg_logit) > threshold → класс 3 (Фон),
         иначе → argmax(grade_head).

Тренируем ТОЛЬКО bg_head: Linear(2048 → 1).
Всё остальное заморожено.

Файл checkpoint: bg_head_best.pth (только веса bg_head: 2049 params).
"""
import os, sys, random, argparse
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import segmentation_models_pytorch as smp

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE = 224

DEFAULT_DATA_ROOT = os.path.expanduser("~/egor/Nornikel_Hac/shlif_data")
DEFAULT_ENCODER_CKPT = os.path.expanduser(
    "~/egor/Nornikel_Hac/pretrained-microscopy-models/grade_unfreeze_best.pth"
)
DEFAULT_BG_CKPT_OUT = os.path.expanduser(
    "~/egor/Nornikel_Hac/pretrained-microscopy-models/bg_head_best.pth"
)

TASK_DIR = "Задача 3. Скажи мне, кто твой шлиф"
ORE_SUBDIRS = [
    ("Фото руд по сортам. ч1", "Оталькованные руды"),
    ("Фото руд по сортам. ч2", "оталькованные"),
    ("Фото руд по сортам. ч1", "Рядовые руды"),
    ("Фото руд по сортам. ч2", "рядовые"),
    ("Фото руд по сортам. ч1", "Труднообогатимые руды"),
    ("Фото руд по сортам. ч2", "тонкие"),
]


# ── Encoder (frozen) ────────────────────────────────────────────────────────
class GradeClassifier(nn.Module):
    def __init__(self, n_classes=3):
        super().__init__()
        dummy = smp.Unet(encoder_name="se_resnext50_32x4d",
                         encoder_weights=None, in_channels=3, classes=2)
        self.encoder = dummy.encoder
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Linear(2048, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, n_classes),
        )

    def get_embedding(self, x):
        return self.pool(self.encoder(x)[-1]).view(x.size(0), -1)

    def forward(self, x):
        return self.head(self.get_embedding(x))


# ── Data ────────────────────────────────────────────────────────────────────
def gather_paths(data_root):
    ore_paths = []
    task = os.path.join(data_root, TASK_DIR)
    for parts in ORE_SUBDIRS:
        d = os.path.join(task, *parts)
        if not os.path.isdir(d): continue
        for f in os.listdir(d):
            if f.lower().endswith((".jpg", ".jpeg", ".png")):
                ore_paths.append(os.path.join(d, f))

    bg_dir = os.path.join(data_root, "background")
    bg_paths = [os.path.join(bg_dir, f) for f in os.listdir(bg_dir)
                if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    return ore_paths, bg_paths


class BinaryDataset(Dataset):
    def __init__(self, paths, labels, augment=False):
        self.paths = paths
        self.labels = labels
        if augment:
            self.tf = transforms.Compose([
                transforms.RandomResizedCrop(IMG_SIZE, scale=(0.6, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.ColorJitter(0.3, 0.3, 0.2, 0.05),
                transforms.RandomRotation(30),
                transforms.ToTensor(),
                transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
            ])
        else:
            self.tf = transforms.Compose([
                transforms.Resize(256), transforms.CenterCrop(IMG_SIZE),
                transforms.ToTensor(),
                transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
            ])

    def __len__(self): return len(self.paths)

    def __getitem__(self, i):
        img = Image.open(self.paths[i]).convert("RGB")
        return self.tf(img), torch.tensor(self.labels[i], dtype=torch.float32)


# ── Train ────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root",   default=DEFAULT_DATA_ROOT)
    ap.add_argument("--encoder-ckpt", default=DEFAULT_ENCODER_CKPT)
    ap.add_argument("--bg-ckpt-out", default=DEFAULT_BG_CKPT_OUT)
    ap.add_argument("--epochs",  type=int,   default=40)
    ap.add_argument("--batch",   type=int,   default=32)
    ap.add_argument("--lr",      type=float, default=1e-2)
    ap.add_argument("--neg-ratio", type=int, default=5,
                    help="ore:background ratio in training batches")
    ap.add_argument("--seed",    type=int,   default=42)
    ap.add_argument("--threshold", type=float, default=0.5,
                    help="sigmoid threshold for background prediction")
    args = ap.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    print(f"Device: {DEVICE}")

    # ── Load frozen encoder ──────────────────────────────────────────────
    print(f"Loading frozen encoder from {args.encoder_ckpt}...")
    enc_model = GradeClassifier(n_classes=3).to(DEVICE)
    enc_model.load_state_dict(
        torch.load(args.encoder_ckpt, map_location=DEVICE, weights_only=True)
    )
    enc_model.eval()
    for p in enc_model.parameters():
        p.requires_grad_(False)

    # ── Trainable background head ────────────────────────────────────────
    bg_head = nn.Linear(2048, 1).to(DEVICE)
    nn.init.normal_(bg_head.weight, std=0.01)
    nn.init.zeros_(bg_head.bias)
    print(f"BG head params: {sum(p.numel() for p in bg_head.parameters()):,}")

    # ── Data ─────────────────────────────────────────────────────────────
    print("Gathering data...")
    ore_paths, bg_paths = gather_paths(os.path.expanduser(args.data_root))
    print(f"  Ore: {len(ore_paths)} | Background: {len(bg_paths)}")

    # Stratified train/val split (80/20) — preserve all bg in train mostly
    rng = random.Random(args.seed)
    rng.shuffle(bg_paths)
    bg_cut = max(1, int(len(bg_paths) * 0.8))
    bg_train, bg_val = bg_paths[:bg_cut], bg_paths[bg_cut:]

    rng.shuffle(ore_paths)
    ore_cut = int(len(ore_paths) * 0.8)
    ore_train, ore_val = ore_paths[:ore_cut], ore_paths[ore_cut:]

    # Oversample background in training so ratio = 1:neg_ratio
    bg_rep = bg_train * (args.neg_ratio * len(ore_train) // len(bg_train) + 1)
    bg_rep = bg_rep[:args.neg_ratio * len(ore_train)]  # cap

    train_paths = ore_train + bg_rep
    train_labels = [0] * len(ore_train) + [1] * len(bg_rep)
    val_paths   = ore_val   + bg_val
    val_labels  = [0] * len(ore_val)   + [1] * len(bg_val)

    # Shuffle train
    c = list(zip(train_paths, train_labels)); rng.shuffle(c)
    train_paths, train_labels = zip(*c)

    print(f"  Train: {len(train_paths)} (ore {len(ore_train)} + bg_rep {len(bg_rep)})")
    print(f"  Val:   {len(val_paths)} (ore {len(ore_val)} + bg {len(bg_val)})")

    train_ds = BinaryDataset(list(train_paths), list(train_labels), augment=True)
    val_ds   = BinaryDataset(val_paths, val_labels, augment=False)
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                              num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch, shuffle=False,
                              num_workers=4, pin_memory=True)

    # ── Optimizer + loss ─────────────────────────────────────────────────
    pos_weight = torch.tensor([args.neg_ratio], dtype=torch.float32).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.Adam(bg_head.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=args.epochs, eta_min=args.lr/50
    )

    best_f1, best_sd = 0.0, None

    for epoch in range(1, args.epochs + 1):
        # ── Train step ───────────────────────────────────────────────────
        bg_head.train()
        losses = []
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            with torch.no_grad():
                emb = enc_model.get_embedding(x)
            logit = bg_head(emb).squeeze(1)
            loss = criterion(logit, y)
            opt.zero_grad(); loss.backward(); opt.step()
            losses.append(loss.item())
        sched.step()

        # ── Eval ─────────────────────────────────────────────────────────
        bg_head.eval()
        tp = fp = fn = tn = 0
        with torch.no_grad():
            for x, y in val_loader:
                emb = enc_model.get_embedding(x.to(DEVICE))
                pred = (torch.sigmoid(bg_head(emb).squeeze(1)) > args.threshold).cpu()
                y = y.bool()
                tp += (pred & y).sum().item()
                fp += (pred & ~y).sum().item()
                fn += (~pred & y).sum().item()
                tn += (~pred & ~y).sum().item()

        prec = tp/(tp+fp) if tp+fp else 0
        rec  = tp/(tp+fn) if tp+fn else 0
        f1   = 2*prec*rec/(prec+rec) if prec+rec else 0
        acc  = (tp+tn)/(tp+fp+fn+tn)

        flag = " ★" if f1 > best_f1 else ""
        if f1 > best_f1:
            best_f1 = f1
            best_sd = {k: v.cpu().clone() for k,v in bg_head.state_dict().items()}
            torch.save(best_sd, args.bg_ckpt_out)

        print(f"ep{epoch:3d}  loss={np.mean(losses):.4f}  acc={acc:.3f}  "
              f"P={prec:.3f} R={rec:.3f} F1={f1:.3f}  "
              f"tp={tp} fp={fp} fn={fn} tn={tn}{flag}", flush=True)

    print(f"\nBest bg F1: {best_f1:.3f}")
    print(f"Checkpoint: {args.bg_ckpt_out}")

    # ── Quick sanity: cascade eval on val set ─────────────────────────────
    print("\n=== Cascade eval on val set ===")
    if best_sd:
        bg_head.load_state_dict(best_sd)
    bg_head.eval()
    correct_bg = wrong_as_ore = wrong_as_bg = correct_ore = 0
    with torch.no_grad():
        for x, y in val_loader:
            emb = enc_model.get_embedding(x.to(DEVICE))
            is_bg = (torch.sigmoid(bg_head(emb).squeeze(1)) > args.threshold).cpu()
            grade = enc_model.head(emb).argmax(1).cpu()
            for bg_pred, gt, gr in zip(is_bg, y.bool(), grade):
                if gt:   # true background
                    if bg_pred: correct_bg += 1
                    else: wrong_as_ore += 1
                else:  # true ore
                    if bg_pred: wrong_as_bg += 1
                    else: correct_ore += 1

    total_bg  = correct_bg + wrong_as_ore
    total_ore = correct_ore + wrong_as_bg
    print(f"  Background: {correct_bg}/{total_bg} correct ({100*correct_bg/max(1,total_bg):.0f}%)")
    print(f"  Ore:        {correct_ore}/{total_ore} correct ({100*correct_ore/max(1,total_ore):.0f}%)")
    print("Done.")


if __name__ == "__main__":
    main()
