#!/usr/bin/env python3
"""
panorama_infer.py — grid + sliding-window inference on gigapixel ore panoramas.

Modes:
  --mode grid   : non-overlapping tiles; each tile → one class label
  --mode slide  : 50 % overlap; on overlap zones soft-vote (sum probs) → argmax
  --mode both   : run both, save results side-by-side for comparison

Outputs (per image, in --out dir):
  <name>_grid_map.png       colour tile map (upscaled to original res)
  <name>_grid_overlay.jpg   semi-transparent overlay on original
  <name>_grid_tiles.csv     per-tile: row,col,label,conf,time_ms
  <name>_slide_map.png
  <name>_slide_overlay.jpg
  <name>_slide_tiles.csv
  <name>_compare.jpg        grid | slide side-by-side (downscaled to 2000px wide)
  <name>_speed.txt          timing report

Usage:
  python panorama_infer.py \
    --ckpt grade_unfreeze_best.pth \
    --input panoramas/ \
    --out pano_results/ \
    --mode both --tile 512 --workers 4
"""
import argparse, csv, os, sys, time, glob
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

# Allow giant images
Image.MAX_IMAGE_SIZE = 300_000_000

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import segmentation_models_pytorch as smp

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

CLASS_NAMES = ["talc", "ordinary", "fine"]
CLASS_LABELS_RU = ["Оталькованная", "Рядовая", "Труднообогатимая"]
# Colors: talc=red, ordinary=green, fine=blue
COLORS = np.array([
    [220,  50,  50],   # 0 talc
    [ 50, 190,  50],   # 1 ordinary
    [ 50, 100, 220],   # 2 fine
], dtype=np.uint8)


# ── Model ─────────────────────────────────────────────────────────────────────
class GradeClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        dummy = smp.Unet(encoder_name="se_resnext50_32x4d",
                         encoder_weights=None, in_channels=3, classes=2)
        self.encoder = dummy.encoder
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Linear(2048, 256), nn.ReLU(), nn.Dropout(0.3), nn.Linear(256, 3),
        )

    def forward(self, x):
        return self.head(self.pool(self.encoder(x)[-1]).view(-1, 2048))


def load_model(ckpt_path):
    model = GradeClassifier().to(DEVICE)
    model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE, weights_only=True))
    return model.eval()


# ── Tile inference (batched) ──────────────────────────────────────────────────
def preprocess(crop_np, tile_size):
    h, w = crop_np.shape[:2]
    if h != tile_size or w != tile_size:
        crop_np = np.array(Image.fromarray(crop_np).resize((tile_size, tile_size), Image.BILINEAR))
    arr = (crop_np.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(arr.transpose(2, 0, 1))


def infer_batch(model, tensors):
    """tensors: list of (C,H,W) → returns probs array (N,3)"""
    batch = torch.stack(tensors).to(DEVICE)
    with torch.no_grad():
        return F.softmax(model(batch), dim=1).cpu().numpy()


# ── Grid inference ────────────────────────────────────────────────────────────
def run_grid(model, img_np, tile_size, batch_size=32):
    H, W = img_np.shape[:2]
    rows = (H + tile_size - 1) // tile_size
    cols = (W + tile_size - 1) // tile_size

    labels = np.full((rows, cols), -1, dtype=np.int8)
    confs  = np.zeros((rows, cols), dtype=np.float32)
    probs  = np.zeros((rows, cols, 3), dtype=np.float32)
    times  = []

    positions, tensors = [], []

    def flush(pos, tens):
        t0 = time.perf_counter()
        p = infer_batch(model, tens)
        dt = (time.perf_counter() - t0) * 1000
        times.append(dt)
        for (r, c), pv in zip(pos, p):
            labels[r, c] = pv.argmax()
            confs[r, c]  = pv.max()
            probs[r, c]  = pv

    for r in range(rows):
        for c in range(cols):
            y0 = r * tile_size; x0 = c * tile_size
            crop = img_np[y0:min(y0+tile_size, H), x0:min(x0+tile_size, W)]
            positions.append((r, c))
            tensors.append(preprocess(crop, tile_size))
            if len(tensors) == batch_size:
                flush(positions, tensors)
                positions, tensors = [], []

    if tensors:
        flush(positions, tensors)

    return labels, confs, probs, times


# ── Sliding-window inference ──────────────────────────────────────────────────
def run_slide(model, img_np, tile_size, stride, batch_size=32):
    H, W = img_np.shape[:2]
    # Ensure full coverage
    ys = list(range(0, H - tile_size + 1, stride))
    if not ys or ys[-1] + tile_size < H: ys.append(max(0, H - tile_size))
    xs = list(range(0, W - tile_size + 1, stride))
    if not xs or xs[-1] + tile_size < W: xs.append(max(0, W - tile_size))

    # Accumulate at stride-grid resolution
    nrows = len(ys); ncols = len(xs)
    prob_acc = np.zeros((nrows, ncols, 3), dtype=np.float32)
    times = []

    positions, tensors = [], []

    def flush(pos, tens):
        t0 = time.perf_counter()
        p = infer_batch(model, tens)
        dt = (time.perf_counter() - t0) * 1000
        times.append(dt)
        for (ri, ci), pv in zip(pos, p):
            prob_acc[ri, ci] += pv   # soft-vote accumulation

    for ri, y0 in enumerate(ys):
        for ci, x0 in enumerate(xs):
            crop = img_np[y0:y0+tile_size, x0:x0+tile_size]
            positions.append((ri, ci))
            tensors.append(preprocess(crop, tile_size))
            if len(tensors) == batch_size:
                flush(positions, tensors)
                positions, tensors = [], []

    if tensors:
        flush(positions, tensors)

    # For each grid cell, argmax of accumulated (summed) probabilities
    labels = prob_acc.argmax(axis=2).astype(np.int8)
    confs  = prob_acc.max(axis=2) / np.maximum(prob_acc.sum(axis=2), 1e-6)
    probs  = prob_acc / np.maximum(prob_acc.sum(axis=2, keepdims=True), 1e-6)

    return labels, confs, probs, times, ys, xs


# ── Visualisation ─────────────────────────────────────────────────────────────
def smooth_probs(probs, sigma=1.2):
    """Gaussian blur on prob map before argmax → softer boundaries."""
    from scipy.ndimage import gaussian_filter
    smoothed = np.stack([gaussian_filter(probs[..., c], sigma=sigma) for c in range(3)], axis=2)
    return smoothed / np.maximum(smoothed.sum(axis=2, keepdims=True), 1e-6)


def probs_to_colormap(probs, sigma=1.2):
    """Smooth probs → soft-boundary colour map."""
    p = smooth_probs(probs, sigma) if sigma > 0 else probs
    labels = p.argmax(axis=2).astype(np.int8)
    h, w = labels.shape
    out = np.zeros((h, w, 3), dtype=np.uint8)
    for cls in range(3):
        out[labels == cls] = COLORS[cls]
    return out, labels


def labels_to_colormap(labels):
    """labels (R,C) int8 → RGB image (R,C,3)"""
    h, w = labels.shape
    out = np.zeros((h, w, 3), dtype=np.uint8)
    for cls in range(3):
        out[labels == cls] = COLORS[cls]
    out[labels == -1] = [128, 128, 128]
    return out


def upscale_map(color_map, target_h, target_w, smooth_px=0):
    """Upscale colour map; optionally blur before nearest-neighbour to soften tile edges."""
    if smooth_px > 0:
        from scipy.ndimage import gaussian_filter
        color_map = gaussian_filter(color_map.astype(np.float32), sigma=[smooth_px, smooth_px, 0]).astype(np.uint8)
    return np.array(Image.fromarray(color_map).resize(
        (target_w, target_h), Image.NEAREST))


def make_overlay(img_np, color_map_full, alpha=0.45):
    out = img_np.astype(np.float32)
    mask = np.any(color_map_full != 128, axis=2)  # not gray = classified
    for c in range(3):
        out[..., c][mask] = (1-alpha)*out[..., c][mask] + alpha*color_map_full[..., c][mask]
    return np.clip(out, 0, 255).astype(np.uint8)


def add_legend(img):
    """Paste a small legend in the top-left corner."""
    from PIL import ImageDraw
    pil = Image.fromarray(img)
    draw = ImageDraw.Draw(pil)
    box_h = 24; pad = 8; x0 = pad
    for i, (name, color) in enumerate(zip(CLASS_LABELS_RU, COLORS.tolist())):
        y0 = pad + i * (box_h + 4)
        draw.rectangle([x0, y0, x0+box_h, y0+box_h], fill=tuple(color))
        draw.text((x0+box_h+6, y0+4), name, fill=(255, 255, 255))
    return np.array(pil)


def save_tiles_csv(path, labels, confs, ys_or_rows, xs_or_cols, tile_size):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["row", "col", "y0", "x0", "label", "label_ru", "conf"])
        for ri in range(labels.shape[0]):
            for ci in range(labels.shape[1]):
                lbl = int(labels[ri, ci])
                y0 = ys_or_rows[ri] if isinstance(ys_or_rows, list) else ri * tile_size
                x0 = xs_or_cols[ci] if isinstance(xs_or_cols, list) else ci * tile_size
                w.writerow([ri, ci, y0, x0,
                             CLASS_NAMES[lbl] if lbl >= 0 else "unknown",
                             CLASS_LABELS_RU[lbl] if lbl >= 0 else "—",
                             f"{confs[ri,ci]:.4f}"])


# ── Speed report ──────────────────────────────────────────────────────────────
def speed_report(mode, H, W, tile_size, stride, all_times_ms, total_s, n_tiles):
    n_batches = len(all_times_ms)
    per_tile = sum(all_times_ms) / n_batches if n_batches else 0
    mpix = H * W / 1e6
    lines = [
        f"=== Speed: {mode} ===",
        f"  Image       : {W}×{H} ({mpix:.1f} Mpix)",
        f"  Tile size   : {tile_size}",
        f"  Stride      : {stride}",
        f"  Tiles total : {n_tiles}  ({n_batches} batches of ≤32)",
        f"  GPU ms/batch: {per_tile:.1f} ms",
        f"  Wall time   : {total_s:.1f} s  ({mpix/total_s:.2f} Mpix/s)",
        f"  10k×10k est : {100 / (mpix/total_s):.0f} s",
    ]
    return "\n".join(lines)


# ── Main ──────────────────────────────────────────────────────────────────────
def process_image(model, img_path, out_dir, tile_size, batch_size, modes, smooth=False):
    stem = os.path.splitext(os.path.basename(img_path))[0]
    print(f"\n{'='*60}")
    print(f"  {os.path.basename(img_path)}")

    t_load = time.perf_counter()
    img = Image.open(img_path).convert("RGB")
    W, H = img.size
    img_np = np.array(img)
    print(f"  Loaded: {W}×{H} in {time.perf_counter()-t_load:.1f}s")

    speed_lines = [f"Image: {os.path.basename(img_path)} ({W}×{H})"]
    compare_imgs = []

    stride_slide = tile_size // 2

    for mode in modes:
        print(f"\n  [{mode}] running...", flush=True)
        t0 = time.perf_counter()

        if mode == "grid":
            stride = tile_size
            labels, confs, probs, times = run_grid(model, img_np, tile_size, batch_size)
            ys_list = list(range(0, H, tile_size))
            xs_list = list(range(0, W, tile_size))
        else:
            stride = stride_slide
            labels, confs, probs, times, ys_list, xs_list = run_slide(
                model, img_np, tile_size, stride_slide, batch_size)

        total_s = time.perf_counter() - t0
        n_tiles = len(ys_list) * len(xs_list)
        print(f"  [{mode}] {n_tiles} tiles in {total_s:.1f}s "
              f"({W*H/1e6/total_s:.2f} Mpix/s)", flush=True)

        sp = speed_report(mode, H, W, tile_size, stride, times, total_s, n_tiles)
        speed_lines.append(sp)
        print(sp)

        # Colour map: hard squares by default, Gaussian blur if --smooth
        sigma_prob = (0.6 if mode == "slide" else 0.4) if smooth else 0
        cmap, _ = probs_to_colormap(probs, sigma=sigma_prob)
        cmap_full = upscale_map(cmap, H, W, smooth_px=1 if smooth else 0)

        # Save maps
        Image.fromarray(cmap).save(os.path.join(out_dir, f"{stem}_{mode}_map_small.png"))
        Image.fromarray(cmap_full).save(os.path.join(out_dir, f"{stem}_{mode}_map.png"))

        # Confidence map
        conf_vis = (confs * 255).astype(np.uint8)
        conf_full = np.array(Image.fromarray(conf_vis).resize((W, H), Image.NEAREST))
        conf_rgb = np.stack([conf_full, conf_full, np.zeros_like(conf_full)], axis=2)
        Image.fromarray(conf_rgb.astype(np.uint8)).save(
            os.path.join(out_dir, f"{stem}_{mode}_conf.png"))

        # Overlay
        overlay = add_legend(make_overlay(img_np, cmap_full))
        Image.fromarray(overlay).save(
            os.path.join(out_dir, f"{stem}_{mode}_overlay.jpg"), quality=85)

        # Per-tile CSV
        save_tiles_csv(
            os.path.join(out_dir, f"{stem}_{mode}_tiles.csv"),
            labels, confs, ys_list, xs_list, tile_size)

        # Class distribution
        total_cells = labels.size
        for c in range(3):
            pct = (labels == c).sum() / total_cells * 100
            print(f"    {CLASS_LABELS_RU[c]:20s}: {pct:5.1f}%")

        compare_imgs.append((mode, overlay))

    # Side-by-side compare (downscale to ≤2000px wide each)
    if len(compare_imgs) == 2:
        MAX_W = 2000
        scale = min(1.0, MAX_W / W)
        th, tw = int(H * scale), int(W * scale)
        panels = [np.array(Image.fromarray(img).resize((tw, th), Image.BILINEAR))
                  for _, img in compare_imgs]
        sep = np.ones((th, 8, 3), dtype=np.uint8) * 200
        combined = np.concatenate([panels[0], sep, panels[1]], axis=1)
        Image.fromarray(combined).save(
            os.path.join(out_dir, f"{stem}_compare.jpg"), quality=85)
        print(f"\n  Saved compare: {stem}_compare.jpg")

    with open(os.path.join(out_dir, f"{stem}_speed.txt"), "w") as f:
        f.write("\n\n".join(speed_lines))

    print(f"\n  Done → {out_dir}/{stem}_*")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt",    required=True)
    ap.add_argument("--input",   required=True, help="image or directory")
    ap.add_argument("--out",     default="pano_results")
    ap.add_argument("--mode",    default="both", choices=["grid","slide","both"])
    ap.add_argument("--tile",    type=int, default=512)
    ap.add_argument("--batch",   type=int, default=32)
    ap.add_argument("--smooth",  action="store_true", default=False,
                    help="Gaussian smoothing on tile boundaries (default: hard squares)")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    modes = ["grid", "slide"] if args.mode == "both" else [args.mode]

    print(f"Loading model: {args.ckpt}")
    model = load_model(args.ckpt)
    print(f"Device: {DEVICE}  tile={args.tile}  batch={args.batch}")

    if os.path.isdir(args.input):
        paths = sorted(p for p in glob.glob(os.path.join(args.input, "*"))
                       if p.lower().endswith((".jpg",".jpeg",".png",".tif",".tiff")))
    else:
        paths = [args.input]

    for path in paths:
        process_image(model, path, args.out, args.tile, args.batch, modes, smooth=args.smooth)

    print(f"\nAll done → {args.out}/")


if __name__ == "__main__":
    main()
