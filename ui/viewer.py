"""
VIEWER — построение цветного overlay поверх исходного изображения.

Поток B (UI) отвечает за отрисовку, но саму цветную маску удобно собирать
здесь одной функцией, чтобы цвета классов были едиными (берём из config).

Для MVP делаем статичный overlay (наложение полупрозрачной маски).
Zoom/pan/minimap для больших панорам — задача потока B поверх этого
(см. PLAN_AGENT_B.md): tiled-загрузка и streamlit-image-zoom/навигатор.
"""

from __future__ import annotations

import numpy as np
from PIL import Image

from src import config


def colorize_mask(mask: np.ndarray) -> Image.Image:
    """Превратить маску кодов классов в цветное RGBA-изображение."""
    h, w = mask.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    for cls, color in config.CLASS_COLORS.items():
        rgba[mask == cls] = color
    return Image.fromarray(rgba, mode="RGBA")


def make_overlay(
    base_image: Image.Image,
    mask: np.ndarray,
    show_classes: set[int] | None = None,
    opacity: float = 1.0,
) -> Image.Image:
    """
    Наложить цветную маску на исходное изображение.

    show_classes — какие классы показывать (для вкл/выкл слоёв в UI).
    opacity      — общий множитель прозрачности слоя маски (0..1).
    """
    base = base_image.convert("RGBA")
    if base.size != (mask.shape[1], mask.shape[0]):
        # Приводим маску к размеру картинки (на случай масштабирования превью).
        mask_img = Image.fromarray(mask, mode="L").resize(base.size, Image.NEAREST)
        mask = np.array(mask_img, dtype=np.uint8)

    if show_classes is not None:
        # Обнуляем скрытые классы, чтобы они не подсвечивались.
        filtered = np.where(np.isin(mask, list(show_classes)), mask, 0).astype(np.uint8)
        mask = filtered

    overlay = colorize_mask(mask)
    if opacity < 1.0:
        alpha = np.array(overlay.split()[-1], dtype=np.float32) * float(opacity)
        overlay.putalpha(Image.fromarray(alpha.astype(np.uint8), mode="L"))

    return Image.alpha_composite(base, overlay)


def legend_items() -> list[tuple[str, str]]:
    """Легенда для UI: (название класса, hex-цвет)."""
    items = []
    for cls, name in config.CLASS_NAMES.items():
        if cls == config.CLASS_BACKGROUND:
            continue
        r, g, b, _a = config.CLASS_COLORS[cls]
        items.append((name, f"#{r:02x}{g:02x}{b:02x}"))
    return items
