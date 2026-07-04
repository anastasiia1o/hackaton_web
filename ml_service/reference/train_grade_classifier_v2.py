#!/usr/bin/env python3
"""
Grade classifier v2 — unfreeze fine-tuning on top of frozen checkpoint.
Starts from grade_best.pth, unfreezes encoder with encoder_lr = LR * 0.1.
"""
import os, sys, random, argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import ssl

ssl._create_default_https_context = ssl._create_unverified_context
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import segmentation_models_pytorch as smp

SEED       = 42
BATCH_SIZE = 8    # encoder backprop needs more mem; 8×512 fits 11GB
IMG_SIZE   = 512
EPOCHS     = 40
LR         = 1e-4   # slightly higher for faster convergence
ENC_LR_MULT = 0.1  # encoder gets 10% of head LR
PATIENCE   = 10
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

DEFAULT_BASE = os.path.expanduser(
    "~/egor/Nornikel_Hac/shlif_data/Задача 3. Скажи мне, кто твой шлиф"
)

CLASS_NAMES = ["Оталькованная", "Рядовая", "Труднообогатимая"]

SUBDIRS = {
    0: [("Фото руд по сортам. ч1", "Оталькованные руды"),
        ("Фото руд по сортам. ч2", "оталькованные")],
    1: [("Фото руд по сортам. ч1", "Рядовые руды"),
        ("Фото руд по сортам. ч2", "рядовые")],
    2: [("Фото руд по сортам. ч1", "Труднообогатимые руды"),
        ("Фото руд по сортам. ч2", "тонкие")],
}


def gather_paths(base):
    paths, labels = [], []
    for label, parts_list in SUBDIRS.items():
        for parts in parts_list:
            d = os.path.join(base, *parts)
            if not os.path.isdir(d): continue
            for f in os.listdir(d):
                if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                    paths.append(os.path.join(d, f)); labels.append(label)
    return paths, labels


class GradeDataset(Dataset):
    def __init__(self, paths, labels, augment=True):
        self.paths, self.labels, self.augment = paths, labels, augment

    def __len__(self): return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        W, H = img.size; s = min(H, W)
        if self.augment:
            y0 = random.randint(0, H-s); x0 = random.randint(0, W-s)
        else:
            y0, x0 = (H-s)//2, (W-s)//2
        img = img.crop((x0, y0, x0+s, y0+s)).resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
        arr = np.array(img, dtype=np.float32)
        if self.augment:
            if random.random() > 0.5: arr = arr[:, ::-1].copy()
            if random.random() > 0.5: arr = arr[::-1].copy()
            if random.random() > 0.5: arr = np.rot90(arr, random.randint(1,3)).copy()
            if random.random() > 0.3: arr = np.clip(arr * random.uniform(0.75, 1.25), 0, 255)
        arr = (arr/255.0 - IMAGENET_MEAN) / IMAGENET_STD
        return torch.from_numpy(arr.transpose(2,0,1)), self.labels[idx]


class GradeClassifier(nn.Module):
    def __init__(self, n_classes=3):
        super().__init__()
        dummy = smp.Unet(encoder_name="se_resnext50_32x4d", encoder_weights=None,
                         in_channels=3, classes=2)
        self.encoder = dummy.encoder
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Linear(2048, 256), nn.ReLU(), nn.Dropout(0.3), nn.Linear(256, n_classes),
        )

    def forward(self, x):
        z = self.pool(self.encoder(x)[-1]).view(-1, 2048)
        return self.head(z)


def macro_f1(preds, labels, n=3):
    preds, labels = np.array(preds), np.array(labels)
    f1s = []
    for c in range(n):
        tp = ((preds==c)&(labels==c)).sum(); fp = ((preds==c)&(labels!=c)).sum()
        fn = ((preds!=c)&(labels==c)).sum()
        p = tp/(tp+fp+1e-8); r = tp/(tp+fn+1e-8)
        f1s.append(2*p*r/(p+r+1e-8))
    return np.mean(f1s), f1s


def evaluate(model, loader):
    model.eval(); all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in loader:
            all_preds.extend(model(imgs.to(DEVICE)).argmax(1).cpu().tolist())
            all_labels.extend(labels.tolist())
    acc = np.mean(np.array(all_preds)==np.array(all_labels))
    f1, f1s = macro_f1(all_preds, all_labels)
    return acc, f1, f1s, all_preds, all_labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--from-ckpt", required=True, help="start from frozen checkpoint")
    parser.add_argument("--save-ckpt", default="grade_unfreeze_best.pth")
    parser.add_argument("--data-root", default=None,
                        help="root dir of the dataset (parent of 'Фото руд по сортам. ч1' etc.)")
    args = parser.parse_args()

    save_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), args.save_ckpt
    )
    base = os.path.expanduser(args.data_root) if args.data_root else DEFAULT_BASE

    paths, labels = gather_paths(base)
    print(f"Total: {len(paths)}")
    for c, name in enumerate(CLASS_NAMES): print(f"  {name}: {labels.count(c)}")

    idx_by_class = {c: [i for i,l in enumerate(labels) if l==c] for c in range(3)}
    train_idx, test_idx = [], []
    for c in range(3):
        idxs = idx_by_class[c]; random.shuffle(idxs); n = max(1, int(len(idxs)*0.20))
        test_idx.extend(idxs[:n]); train_idx.extend(idxs[n:])
    random.shuffle(train_idx); random.shuffle(test_idx)

    train_paths = [paths[i] for i in train_idx]; train_labels = [labels[i] for i in train_idx]
    test_paths  = [paths[i] for i in test_idx];  test_labels  = [labels[i] for i in test_idx]
    print(f"\nTrain: {len(train_paths)}  Test: {len(test_paths)}")

    train_dl = DataLoader(GradeDataset(train_paths, train_labels), batch_size=BATCH_SIZE,
                          shuffle=True, num_workers=4, pin_memory=True)
    test_dl  = DataLoader(GradeDataset(test_paths,  test_labels,  augment=False),
                          batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    model = GradeClassifier(n_classes=3).to(DEVICE)
    sd = torch.load(args.from_ckpt, map_location=DEVICE, weights_only=True)
    model.load_state_dict(sd)
    print(f"Loaded: {args.from_ckpt}")

    # Unfreeze encoder
    for p in model.encoder.parameters(): p.requires_grad = True
    print(f"Encoder unfrozen. LR: head={LR}, encoder={LR*ENC_LR_MULT}")

    optimizer = torch.optim.Adam([
        {"params": model.encoder.parameters(), "lr": LR * ENC_LR_MULT},
        {"params": model.head.parameters(),    "lr": LR},
    ])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    class_counts = [train_labels.count(c) for c in range(3)]
    w = torch.tensor([1.0/c for c in class_counts], dtype=torch.float32)
    w = w / w.mean()
    criterion = nn.CrossEntropyLoss(weight=w.to(DEVICE))
    print(f"Class weights: {[f'{v:.2f}' for v in w.tolist()]}")

    best_f1 = 0.0; patience_cnt = 0
    for epoch in range(1, EPOCHS+1):
        model.train()
        total_loss = 0.0
        for imgs, lbls in train_dl:
            imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
            loss = criterion(model(imgs), lbls)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            total_loss += loss.item()
        scheduler.step()

        acc, f1, f1s, _, _ = evaluate(model, test_dl)
        f1s_str = " ".join(f"{n[:4]}={v:.3f}" for n, v in zip(CLASS_NAMES, f1s))
        print(f"[{epoch:3d}/{EPOCHS}] loss={total_loss/len(train_dl):.4f}  "
              f"acc={acc:.4f}  macro-F1={f1:.4f}  [{f1s_str}]")

        if f1 > best_f1:
            best_f1 = f1
            torch.save(model.state_dict(), save_path)
            print(f"  ✓ saved {save_path}  (best F1={best_f1:.4f})")
            patience_cnt = 0
        else:
            patience_cnt += 1
            if patience_cnt >= PATIENCE:
                print(f"Early stop at epoch {epoch}")
                break

    print(f"\nBest macro-F1: {best_f1:.4f}")

    model.load_state_dict(torch.load(save_path, map_location=DEVICE, weights_only=True))
    acc, f1, f1s, preds, true_labels = evaluate(model, test_dl)
    print(f"\n=== Final test results ===")
    print(f"  Accuracy:   {acc:.4f}")
    print(f"  Macro-F1:   {f1:.4f}")
    for c, name in enumerate(CLASS_NAMES):
        correct = sum(p==c and t==c for p,t in zip(preds,true_labels))
        total   = sum(t==c for t in true_labels)
        pred_c  = sum(p==c for p in preds)
        tp=correct; fp=pred_c-tp; fn=total-tp
        prec=tp/(tp+fp+1e-8); rec=tp/(tp+fn+1e-8); f1c=2*prec*rec/(prec+rec+1e-8)
        print(f"  {name}: {correct}/{total}  P={prec:.3f}  R={rec:.3f}  F1={f1c:.3f}")


if __name__ == "__main__":
    main()
