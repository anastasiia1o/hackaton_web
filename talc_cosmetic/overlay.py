# -*- coding: utf-8 -*-
"""
talc_cosmetic.overlay — КОСМЕТИЧЕСКАЯ подсветка талька поверх снимка.

Опциональный визуальный слой ПОСЛЕ основного блочного анализа: чистый CV
(numpy + scipy), никак не влияет на предсказание модели, метрики или класс руды.

Пайплайн (вендор ../deploy_segment + ../talc_red_zones, всё внутри репозитория):
    оригинал → палитровая сегментация (mean-field Potts, seglib) →
    контрастная перекраска (тальк = тёмный «чёрный» класс) →
    density-сборка «области оталькования» (talc_region) →
    красный полупрозрачный оверлей на оригинале + heatmap уверенности.

Палитра микроскопа («панорамная»/«жёлтая») выбирается АВТОМАТИЧЕСКИ по минимальной
ошибке квантования. Сегментация считается на уменьшённой копии (seg_max_side) —
слой косметический, поэтому разрешение снижаем ради скорости, а маску региона
возвращаем в размер оригинала (NEAREST).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

from . import seglib
from .talc_region import extract_talc, region_density

PALETTES_DIR = Path(__file__).with_name("palettes")

# Финальные параметры талько-подсветки (talc_red_zones/README.md §3).
DEFAULTS = dict(dark=50, gray_tol=20, sigma=17.0, dens_thr=0.17, min_area_frac=2e-3)


@dataclass
class TalcOverlayResult:
    overlay: np.ndarray                 # RGB uint8 — красная зона на оригинале
    region: np.ndarray                  # bool-маска области оталькования (размер оригинала)
    density: np.ndarray                 # [0..1] карта плотности (размер сегментации)
    palette_name: str                   # какая палитра микроскопа выбрана
    talc_raw_pct: float                 # доля пикселей талька (сырой тёмный класс)
    region_pct: float                   # доля площади красной зоны
    contrast: np.ndarray = field(repr=False, default=None)  # RGB uint8 — палитровая сегментация


def _red_overlay(base_rgb, region, alpha=0.45, color=(230, 40, 40)):
    """Полупрозрачная красная заливка области `region` поверх base_rgb (uint8)."""
    out = base_rgb.astype(np.float32).copy()
    m = region.astype(bool)
    for c in range(3):
        out[..., c][m] = (1 - alpha) * out[..., c][m] + alpha * color[c]
    return out.clip(0, 255).astype(np.uint8)


def _load_palettes():
    return seglib.load_all_palettes(PALETTES_DIR)


def compute_talc_overlay(
    rgb,
    *,
    seg_max_side: int = 1400,
    alpha: float = 0.45,
    palette: str | None = None,
    iters: int = 3,
    tau: float = 12.0,
    lam: float = 0.9,
    radius: int = 3,
    **params,
) -> TalcOverlayResult:
    """Косметическая талько-подсветка для RGB-массива (uint8, H×W×3).

    rgb           — снимок (обычно уже уменьшённый до разрешения показа);
    seg_max_side  — на этой длинной стороне считается палитровая сегментация
                    (маска региона потом растягивается обратно в размер rgb);
    palette       — имя палитры принудительно, иначе автоопределение;
    params        — переопределить dark/gray_tol/sigma/dens_thr/min_area_frac.
    """
    p = {**DEFAULTS, **params}
    rgb = np.asarray(rgb)[..., :3].astype(np.uint8)
    H, W = rgb.shape[:2]

    # Считаем сегментацию на уменьшённой копии — слой косметический.
    long_side = max(H, W)
    scale = min(1.0, seg_max_side / long_side) if long_side > 0 else 1.0
    if scale < 1.0:
        small = np.asarray(
            Image.fromarray(rgb).resize(
                (max(1, int(W * scale)), max(1, int(H * scale))), Image.BILINEAR
            )
        )
    else:
        small = rgb

    palettes = _load_palettes()
    if not palettes:
        raise RuntimeError(f"Нет палитр в {PALETTES_DIR}")
    if palette and palette in palettes:
        pal_name, pal = palette, palettes[palette]
    else:
        sample = small.reshape(-1, 3).astype(np.float32)
        if len(sample) > 4000:
            idx = np.linspace(0, len(sample) - 1, 4000).astype(int)
            sample = sample[idx]
        pal_name, _errs = seglib.classify_palette(sample, palettes)
        pal = palettes[pal_name]

    xp, uf1d, _device = seglib.get_backend(prefer_gpu=True)
    labels = seglib.nearest_reference_labels_spatial(
        small, centers=pal["palette"], xp=xp, uniform_filter1d=uf1d,
        tau=tau, lam=lam, iters=iters, radius=radius,
    )
    contrast_small = np.clip(pal["contrast"][labels] * 255, 0, 255).astype(np.uint8)

    talc = extract_talc(contrast_small, dark=p["dark"], gray_tol=p["gray_tol"])
    talc_raw_pct = float(talc.mean() * 100.0)
    region_small, density = region_density(
        talc.astype(np.float32), sigma=p["sigma"], dens_thr=p["dens_thr"],
        min_area_frac=p["min_area_frac"],
    )

    # Маску региона возвращаем в разрешение оригинала (NEAREST — граница резкая).
    if region_small.shape != (H, W):
        region = np.asarray(
            Image.fromarray((region_small.astype(np.uint8) * 255)).resize(
                (W, H), Image.NEAREST
            )
        ) > 127
        contrast = np.asarray(
            Image.fromarray(contrast_small).resize((W, H), Image.NEAREST)
        )
    else:
        region = region_small
        contrast = contrast_small

    overlay = _red_overlay(rgb, region, alpha=alpha)
    region_pct = float(region.mean() * 100.0)

    return TalcOverlayResult(
        overlay=overlay, region=region, density=density, palette_name=pal_name,
        talc_raw_pct=talc_raw_pct, region_pct=region_pct, contrast=contrast,
    )


def density_heatmap(density) -> np.ndarray:
    """Плотность [0..1] → сине-жёлтая карта уверенности (uint8 RGB)."""
    d = np.clip(density, 0, 1)
    return np.stack([d * 255, d * 255, 255 * (1 - d)], axis=-1).astype(np.uint8)
