#!/usr/bin/env python3
"""
Grade classification: Оталькованная / Рядовая / Труднообогатимая
Strategy: frozen MicroNet encoder (se_resnext50_32x4d) → GlobalAvgPool → Linear(2048, 3)
Data: ~1220 images from shlif_data, folder=label, stratified 80/20 split.
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
BATCH_SIZE = 16
IMG_SIZE   = 512
EPOCHS     = 40
LR         = 3e-4
PATIENCE   = 12
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
            if not os.path.isdir(d):
                print(f"[WARN] missing dir: {d}"); continue
            for f in os.listdir(d):
                if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                    paths.append(os.path.join(d, f))
                    labels.append(label)
    return paths, labels


class GradeDataset(Dataset):
    def __init__(self, paths, labels, augment=True):
        self.paths, self.labels, self.augment = paths, labels, augment

    def __len__(self): return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        W, H = img.size
        # Random crop or center crop
        s = min(H, W)
        if self.augment:
            y0 = random.randint(0, H - s)
            x0 = random.randint(0, W - s)
        else:
            y0, x0 = (H - s) // 2, (W - s) // 2
        img = img.crop((x0, y0, x0 + s, y0 + s))
        img = img.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
        arr = np.array(img, dtype=np.float32)
        if self.augment:
            if random.random() > 0.5: arr = arr[:, ::-1].copy()
            if random.random() > 0.5: arr = arr[::-1].copy()
            if random.random() > 0.5:
                k = random.randint(1, 3)
                arr = np.rot90(arr, k).copy()
            if random.random() > 0.3:
                arr = np.clip(arr * random.uniform(0.75, 1.25), 0, 255)
        arr = (arr / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
        return torch.from_numpy(arr.transpose(2, 0, 1)), self.labels[idx]


def load_micronet(encoder):
    cache_dir = os.path.join(torch.hub.get_dir(), "checkpoints")
    path = os.path.join(cache_dir,
                        "se_resnext50_32x4d_pretrained_microscopynet_v1.0.pth.tar")
    if not os.path.exists(path):
        print("[WARN] MicroNet not found, using random init"); return
    sd = torch.load(path, map_location="cpu", weights_only=True)
    if "state_dict" in sd: sd = sd["state_dict"]
    enc_sd = encoder.state_dict()
    to_load = {k: sd[k] for k in enc_sd if k in sd and enc_sd[k].shape == sd[k].shape}
    enc_sd.update(to_load); encoder.load_state_dict(enc_sd)
    print(f"MicroNet encoder: {len(to_load)}/{len(enc_sd)} keys")


class GradeClassifier(nn.Module):
    def __init__(self, n_classes=3):
        super().__init__()
        # Borrow SMP's encoder (se_resnext50_32x4d) and strip decoder
        dummy = smp.Unet(encoder_name="se_resnext50_32x4d",
                         encoder_weights=None, in_channels=3, classes=2)
        self.encoder = dummy.encoder
        load_micronet(self.encoder)
        for p in self.encoder.parameters(): p.requires_grad = False

        self.pool = nn.AdaptiveAvgPool2d(1)
        enc_out_ch = 2048  # se_resnext50 last stage channels
        self.head = nn.Sequential(
            nn.Linear(enc_out_ch, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, n_classes),
        )

    def forward(self, x):
        feats = self.encoder(x)   # list of feature maps
        z = self.pool(feats[-1])  # last stage: (B, 2048, 1, 1)
        z = z.view(z.size(0), -1)
        return self.head(z)


def macro_f1(preds, labels, n=3):
    preds, labels = np.array(preds), np.array(labels)
    f1s = []
    for c in range(n):
        tp = ((preds == c) & (labels == c)).sum()
        fp = ((preds == c) & (labels != c)).sum()
        fn = ((preds != c) & (labels == c)).sum()
        p = tp / (tp + fp + 1e-8)
        r = tp / (tp + fn + 1e-8)
        f1s.append(2 * p * r / (p + r + 1e-8))
    return np.mean(f1s)


def evaluate(model, loader):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(DEVICE)
            logits = model(imgs)
            preds = logits.argmax(1).cpu().tolist()
            all_preds.extend(preds); all_labels.extend(labels.tolist())
    acc = np.mean(np.array(all_preds) == np.array(all_labels))
    f1  = macro_f1(all_preds, all_labels)
    return acc, f1, all_preds, all_labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="grade_best.pth")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--data-root", default=None,
                        help="root dir of the dataset (parent of 'Фото руд по сортам. ч1' etc.); "
                             "defaults to the cluster path in BASE")
    args = parser.parse_args()
    ckpt = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), args.checkpoint
    )
    base = os.path.expanduser(args.data_root) if args.data_root else DEFAULT_BASE

    paths, labels = gather_paths(base)
    print(f"Total images: {len(paths)}")
    for c, name in enumerate(CLASS_NAMES):
        print(f"  {name}: {labels.count(c)}")

    # Stratified split 80/20
    idx_by_class = {c: [i for i, l in enumerate(labels) if l == c] for c in range(3)}
    train_idx, test_idx = [], []
    for c in range(3):
        idxs = idx_by_class[c]
        random.shuffle(idxs)
        n_test = max(1, int(len(idxs) * 0.20))
        test_idx.extend(idxs[:n_test]); train_idx.extend(idxs[n_test:])
    random.shuffle(train_idx); random.shuffle(test_idx)

    train_paths = [paths[i] for i in train_idx]; train_labels = [labels[i] for i in train_idx]
    test_paths  = [paths[i] for i in test_idx];  test_labels  = [labels[i] for i in test_idx]
    print(f"\nTrain: {len(train_paths)}  Test: {len(test_paths)}")
    for c, name in enumerate(CLASS_NAMES):
        print(f"  {name}: train={train_labels.count(c)}  test={test_labels.count(c)}")

    train_ds = GradeDataset(train_paths, train_labels, augment=True)
    test_ds  = GradeDataset(test_paths,  test_labels,  augment=False)
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=4, pin_memory=True)
    test_dl  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=4, pin_memory=True)

    model = GradeClassifier(n_classes=3).to(DEVICE)

    if args.eval_only:
        model.load_state_dict(torch.load(ckpt, map_location=DEVICE, weights_only=True))
        acc, f1, preds, true_labels = evaluate(model, test_dl)
        print(f"\nTest  acc={acc:.4f}  macro-F1={f1:.4f}")
        for c, name in enumerate(CLASS_NAMES):
            correct = sum(p == c and t == c for p, t in zip(preds, true_labels))
            total   = sum(t == c for t in true_labels)
            print(f"  {name}: {correct}/{total} correct")
        return

    # Class weights (inverse frequency)
    class_counts = [train_labels.count(c) for c in range(3)]
    w = torch.tensor([1.0/c for c in class_counts], dtype=torch.float32)
    w = w / w.mean()
    print(f"\nClass weights: {[f'{v:.2f}' for v in w.tolist()]}")
    criterion = nn.CrossEntropyLoss(weight=w.to(DEVICE))

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=LR
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_f1 = 0.0; patience_cnt = 0
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for imgs, lbls in train_dl:
            imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
            loss = criterion(model(imgs), lbls)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            total_loss += loss.item()
        scheduler.step()

        acc, f1, _, _ = evaluate(model, test_dl)
        print(f"[{epoch:3d}/{EPOCHS}] loss={total_loss/len(train_dl):.4f}  "
              f"acc={acc:.4f}  macro-F1={f1:.4f}")

        if f1 > best_f1:
            best_f1 = f1
            torch.save(model.state_dict(), ckpt)
            print(f"  ✓ saved {ckpt}  (best F1={best_f1:.4f})")
            patience_cnt = 0
        else:
            patience_cnt += 1
            if patience_cnt >= PATIENCE:
                print(f"Early stop at epoch {epoch}")
                break

    print(f"\nBest macro-F1: {best_f1:.4f}")

    # Final detailed evaluation
    model.load_state_dict(torch.load(ckpt, map_location=DEVICE, weights_only=True))
    acc, f1, preds, true_labels = evaluate(model, test_dl)
    print(f"\n=== Final test results ===")
    print(f"  Accuracy:    {acc:.4f}")
    print(f"  Macro-F1:    {f1:.4f}")
    for c, name in enumerate(CLASS_NAMES):
        correct = sum(p == c and t == c for p, t in zip(preds, true_labels))
        total   = sum(t == c for t in true_labels)
        pred_c  = sum(p == c for p in preds)
        tp = correct
        fp = pred_c - tp
        fn = total - tp
        prec = tp / (tp + fp + 1e-8); rec = tp / (tp + fn + 1e-8)
        f1c  = 2 * prec * rec / (prec + rec + 1e-8)
        print(f"  {name}: {correct}/{total}  P={prec:.3f}  R={rec:.3f}  F1={f1c:.3f}")


if __name__ == "__main__":
    main()
