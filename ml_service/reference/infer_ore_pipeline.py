#!/usr/bin/env python3
"""
infer_ore_pipeline.py — ore shlifь analysis: talc segmentation + grade classification.

Outputs per image:
  <stem>_mask.png      binary talc mask
  <stem>_overlay.jpg   original + blue talc overlay
  <stem>_heatmap.png   talc probability heatmap
  report.csv           talc_percent, grade, confidence per image

Checkpoints:
  --talc-ckpt   talc_v2_unfreeze_best.pth  (UNet, MicroNet encoder, IoU=0.50)
  --grade-ckpt  grade_unfreeze_best.pth    (classifier, macro-F1=0.944)
  --bg-ckpt     bg_head_best.pth           (background detector, F1=1.000)  [optional]

4-class cascade (when --bg-ckpt provided):
  image → frozen encoder → bg_head  → if bg: class "Фон"
                         ↘ grade_head → Оталькованная / Рядовая / Труднообогатимая

Usage:
  python infer_ore_pipeline.py \
      --talc-ckpt  talc_v2_unfreeze_best.pth \
      --grade-ckpt grade_unfreeze_best.pth \
      --bg-ckpt    bg_head_best.pth \
      --input /path/to/image_or_dir \
      --out results/
"""
import os, sys, glob, csv, argparse
import numpy as np
from PIL import Image
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp

ENCODER    = "se_resnext50_32x4d"
CROP_SIZE  = 512
STRIDE     = 256
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

CLASS_NAMES = ["Оталькованная", "Рядовая", "Труднообогатимая", "Фон"]

# Training-set mean talc fraction (used for quantile calibration)
# From 42 razmentka images, 34-train split: mean_gt = 30.3%
TRAIN_MEAN_GT = 0.303


# ── Grade classifier definition (mirrors train_grade_classifier.py) ──────
class GradeClassifier(nn.Module):
    def __init__(self, n_classes=3):
        super().__init__()
        dummy = smp.Unet(encoder_name=ENCODER, encoder_weights=None,
                         in_channels=3, classes=2)
        self.encoder = dummy.encoder
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Linear(2048, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, n_classes),
        )

    def forward(self, x):
        feats = self.encoder(x)
        z = self.pool(feats[-1]).view(feats[-1].size(0), -1)
        return self.head(z)


def load_talc_model(ckpt):
    model = smp.Unet(encoder_name=ENCODER, encoder_weights=None, in_channels=3, classes=2)
    model.load_state_dict(torch.load(ckpt, map_location=DEVICE, weights_only=True))
    return model.to(DEVICE).eval()


def load_grade_model(ckpt):
    model = GradeClassifier(n_classes=3)
    model.load_state_dict(torch.load(ckpt, map_location=DEVICE, weights_only=True))
    return model.to(DEVICE).eval()


def load_bg_head(ckpt):
    head = nn.Linear(2048, 1)
    head.load_state_dict(torch.load(ckpt, map_location=DEVICE, weights_only=True))
    return head.to(DEVICE).eval()


# ── Talc inference ────────────────────────────────────────────────────────
def _infer_crops(model, img_np):
    H, W = img_np.shape[:2]; cs, st = CROP_SIZE, STRIDE
    acc = np.zeros((H, W), np.float32); cnt = np.zeros_like(acc)
    ys = list(range(0, max(1, H-cs+1), st))
    if not ys or ys[-1]+cs < H: ys.append(max(0, H-cs))
    xs = list(range(0, max(1, W-cs+1), st))
    if not xs or xs[-1]+cs < W: xs.append(max(0, W-cs))
    for y0 in ys:
        for x0 in xs:
            patch = img_np[y0:y0+cs, x0:x0+cs].astype(np.float32)
            ph = cs-patch.shape[0]; pw = cs-patch.shape[1]
            if ph or pw: patch = np.pad(patch, ((0,ph),(0,pw),(0,0)), "reflect")
            t = torch.from_numpy(
                ((patch/255-IMAGENET_MEAN)/IMAGENET_STD).transpose(2,0,1)
            ).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                p = F.softmax(model(t), dim=1)[0,1].cpu().numpy()
            h2=cs-ph; w2=cs-pw
            acc[y0:y0+h2, x0:x0+w2] += p[:h2,:w2]
            cnt[y0:y0+h2, x0:x0+w2] += 1.0
    return acc / np.maximum(cnt, 1e-6)


def predict_talc_tta(model, img_np):
    p0 = _infer_crops(model, img_np)
    p1 = _infer_crops(model, img_np[:, ::-1].copy())[:, ::-1]
    p2 = _infer_crops(model, img_np[::-1].copy())[::-1]
    return (p0+p1+p2) / 3.0


def calibrate_threshold(probs, mean_gt_frac=TRAIN_MEAN_GT):
    """Per-image T* such that pred_frac ≈ mean_gt_frac (fixed-quantile)."""
    return float(np.percentile(probs, (1.0 - mean_gt_frac) * 100))


# ── Grade inference ───────────────────────────────────────────────────────
def predict_grade(grade_model, img_np, bg_head=None, bg_threshold=0.5, n_crops=5):
    """Multi-crop cascade: bg_head (optional) → if background → class 3 (Фон),
    else → grade_model → class 0/1/2."""
    H, W = img_np.shape[:2]
    crop_size = 512
    size = min(H, W, crop_size)
    grade_probs_list = []
    bg_logits = []

    positions = [(0, 0), (H-size, 0), (0, W-size), (H-size, W-size),
                 ((H-size)//2, (W-size)//2)]
    for y0, x0 in positions[:n_crops]:
        y0 = max(0, min(y0, H-size)); x0 = max(0, min(x0, W-size))
        crop = img_np[y0:y0+size, x0:x0+size]
        if size < crop_size:
            crop = np.array(Image.fromarray(crop).resize((crop_size, crop_size), Image.BILINEAR))
        arr = (crop.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
        t = torch.from_numpy(arr.transpose(2, 0, 1)).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            emb = grade_model.pool(grade_model.encoder(t)[-1]).view(1, -1)
            grade_probs_list.append(
                F.softmax(grade_model.head(emb), dim=1)[0].cpu().numpy()
            )
            if bg_head is not None:
                bg_logits.append(bg_head(emb).item())

    avg_grade = np.mean(grade_probs_list, axis=0)  # shape (3,)

    if bg_head is not None:
        bg_prob = float(torch.sigmoid(torch.tensor(np.mean(bg_logits))))
        if bg_prob > bg_threshold:
            # Blend: background confidence into a 4-element vector
            all_probs = np.append(avg_grade * (1 - bg_prob), bg_prob)
            return 3, all_probs  # class 3 = Фон

    all_probs = np.append(avg_grade, 0.0)  # Фон prob = 0 when bg_head absent
    return int(avg_grade.argmax()), all_probs


# ── Visualization ─────────────────────────────────────────────────────────
def make_overlay(img, mask, alpha=0.45, color=(40, 90, 230)):
    out = img.astype(np.float32).copy()
    for c in range(3):
        out[..., c][mask] = (1-alpha)*out[...,c][mask] + alpha*color[c]
    return np.clip(out, 0, 255).astype(np.uint8)


def save_heatmap(path, prob):
    d = np.clip(prob, 0, 1)
    Image.fromarray(np.stack([d*255, d*255, 255*(1-d)], -1).astype(np.uint8)).save(path)


def collect(inp):
    if os.path.isdir(inp):
        return sorted(p for p in glob.glob(os.path.join(inp, "*"))
                      if p.lower().endswith((".jpg",".jpeg",".png",".tif",".tiff")))
    return [inp]


def main(args):
    os.makedirs(args.out, exist_ok=True)

    print(f"Loading talc model:  {args.talc_ckpt}")
    talc_model = load_talc_model(args.talc_ckpt)

    grade_model = None
    if args.grade_ckpt and os.path.exists(args.grade_ckpt):
        print(f"Loading grade model: {args.grade_ckpt}")
        grade_model = load_grade_model(args.grade_ckpt)
    else:
        print("[WARN] Grade model not found — grade from talc fraction rule only")

    bg_head = None
    if args.bg_ckpt and os.path.exists(args.bg_ckpt):
        print(f"Loading bg detector: {args.bg_ckpt}  (4-class cascade enabled)")
        bg_head = load_bg_head(args.bg_ckpt)
    else:
        print("BG detector: not used (3-class mode)")

    print(f"Device: {DEVICE}\n")
    rows = []
    for i, path in enumerate(collect(args.input)):
        stem = os.path.splitext(os.path.basename(path))[0]
        img  = np.array(Image.open(path).convert("RGB"))

        # ── Talc segmentation ──
        probs = predict_talc_tta(talc_model, img)
        T = calibrate_threshold(probs) if args.calibrate else args.thr
        mask = (probs > T).astype(bool)
        talc_frac = mask.mean() * 100.0

        # ── Grade classification ──
        if grade_model is not None:
            grade_idx, grade_probs = predict_grade(grade_model, img, bg_head=bg_head)
            grade      = CLASS_NAMES[grade_idx]
            grade_conf = float(grade_probs[grade_idx])
        else:
            # Fallback: rule-based
            grade      = "Оталькованная" if talc_frac > 10.0 else "Рядовая"
            grade_conf = None

        # ── Artifacts ──
        Image.fromarray((mask*255).astype(np.uint8)).save(
            os.path.join(args.out, stem+"_mask.png"))
        Image.fromarray(make_overlay(img, mask)).save(
            os.path.join(args.out, stem+"_overlay.jpg"), quality=88)
        save_heatmap(os.path.join(args.out, stem+"_heatmap.png"), probs)

        row = {
            "file":          stem,
            "talc_percent":  f"{talc_frac:.2f}",
            "threshold":     f"{T:.3f}",
            "grade":         grade,
            "grade_conf":    f"{grade_conf:.3f}" if grade_conf is not None else "—",
        }
        rows.append(row)
        conf_str = f"  grade={grade}({grade_conf*100:.0f}%)" if grade_conf else ""
        print(f"[{i+1}] {stem[:32]:32s}  talc={talc_frac:.1f}%  T={T:.3f}{conf_str}", flush=True)

    with open(os.path.join(args.out, "report.csv"), "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        wr.writeheader(); wr.writerows(rows)
    print(f"\nDone → {args.out}/report.csv  ({len(rows)} images)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Ore shlifь analysis pipeline")
    ap.add_argument("--talc-ckpt",  required=True, help="talc segmentation checkpoint")
    ap.add_argument("--grade-ckpt", default=None,  help="grade classifier checkpoint")
    ap.add_argument("--bg-ckpt",    default=None,
                    help="background detector head (bg_head_best.pth); enables 4-class mode")
    ap.add_argument("--input",      required=True, help="image or directory")
    ap.add_argument("--out",        default="ore_results")
    ap.add_argument("--thr",       type=float, default=0.79,
                    help="talc threshold (default 0.79, from held-out calibration)")
    ap.add_argument("--calibrate", action="store_true", default=False,
                    help="per-image fixed-quantile calibration (for оталькованная images only)")
    main(ap.parse_args())
