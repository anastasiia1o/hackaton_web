"""
DATASET EXPORT — экспорт результатов в формате «как в S2_v2» (эталонный
пример стороннего датасета, который используют другие люди в своей работе):

    imgs/<split>/<name>.jpg            — исходное изображение
    masks/<split>/<name>.png           — маска классов, R=G=B=id (как читают PIL.convert("L"))
    masks_colored/<split>/<name>.png   — непрозрачная цветная маска (для быстрого визуального контроля)
    masks_human/<split>/<name>.jpg     — триптих source|overlay|annotation + легенда (для проверки человеком)

Это ОДИН из вариантов сохранения результатов — наравне с CSV/JSON/PDF/GeoJSON
(src/reports.py) и форматом data/datasets/.../exports/active_learning
(src/dataset_storage.py). Ничего из существующих форматов не заменяет.
`split` — необязательная подпапка (например "train"/"test"); по умолчанию
плоская структура без подпапок, т.к. train/test-разбиение — не наша концепция,
это только пример структуры папок.
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def mask_to_id_image(mask: np.ndarray) -> Image.Image:
    """R=G=B=id класса — как masks/*.png эталонного датасета (id класса можно
    восстановить и через .convert("L"), и просто читая R-канал)."""
    v = mask.astype(np.uint8)
    rgb = np.stack([v, v, v], axis=-1)
    return Image.fromarray(rgb, mode="RGB")


def colorize_opaque(
    mask: np.ndarray, class_colors: dict[int, tuple[int, int, int, int]],
) -> Image.Image:
    """Непрозрачная цветная маска (без наложения на исходник) — как masks_colored/*.png."""
    h, w = mask.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for cls, color in class_colors.items():
        rgb[mask == cls] = color[:3]
    return Image.fromarray(rgb, mode="RGB")


def _blend(
    image: Image.Image, mask: np.ndarray,
    class_colors: dict[int, tuple[int, int, int, int]], opacity: float = 0.55,
) -> Image.Image:
    base = image.convert("RGBA")
    if base.size != (mask.shape[1], mask.shape[0]):
        mask_img = Image.fromarray(mask, mode="L").resize(base.size, Image.NEAREST)
        mask = np.array(mask_img, dtype=np.uint8)
    h, w = mask.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    for cls, color in class_colors.items():
        r, g, b = color[:3]
        a = int(color[3]) if len(color) > 3 else 255
        rgba[mask == cls] = (r, g, b, int(a * opacity))
    overlay = Image.fromarray(rgba, mode="RGBA")
    return Image.alpha_composite(base, overlay)


def build_human_triptych(
    image: Image.Image,
    mask: np.ndarray,
    class_colors: dict[int, tuple[int, int, int, int]],
    class_names: dict[int, str],
    panel_height: int = 420,
) -> Image.Image:
    """
    source | overlay | annotation в одной картинке + легенда снизу — для
    быстрой визуальной проверки человеком (как masks_human/*.jpg в эталоне).
    """
    src = image.convert("RGB")
    overlay = _blend(src, mask, class_colors, opacity=0.55).convert("RGB")
    colored = colorize_opaque(mask, class_colors)

    def fit(im: Image.Image) -> Image.Image:
        w, h = im.size
        scale = panel_height / h
        return im.resize((max(1, int(w * scale)), panel_height))

    panels = [fit(src), fit(overlay), fit(colored)]
    titles = ["source", "overlay", "annotation"]

    gap = 6
    title_h = 26
    legend_items = [
        (class_names.get(cid, str(cid)), color)
        for cid, color in sorted(class_colors.items()) if cid != 0
    ]
    legend_h = 30 if legend_items else 0

    total_w = sum(p.width for p in panels) + gap * (len(panels) - 1)
    total_h = title_h + panel_height + legend_h
    canvas = Image.new("RGB", (total_w, total_h), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()

    x = 0
    for panel, title in zip(panels, titles):
        tw = draw.textlength(title, font=font)
        draw.text((x + panel.width / 2 - tw / 2, 5), title, fill=(30, 30, 30), font=font)
        canvas.paste(panel, (x, title_h))
        x += panel.width + gap

    if legend_items:
        lx, ly = 8, title_h + panel_height + 6
        for name, color in legend_items:
            draw.rectangle([lx, ly + 2, lx + 12, ly + 14], fill=tuple(color[:3]))
            draw.text((lx + 16, ly), name, fill=(30, 30, 30), font=font)
            lx += int(16 + draw.textlength(name, font=font) + 18)

    return canvas


def export_s2_bundle(
    out_dir: Path,
    items: Iterable[dict[str, Any]],
    class_colors: dict[int, tuple[int, int, int, int]],
    class_names: dict[int, str],
    split: str = "",
    include_human: bool = True,
) -> dict:
    """
    items: итерируемое [{"name": str (без расширения), "image": PIL.Image,
    "mask": np.ndarray}, ...] — может быть генератором, элементы обрабатываются
    ОДИН ЗА РАЗ (открыл/записал/закрыл), поэтому для батча из многих панорам
    не нужно держать все декодированные изображения в памяти одновременно.

    Пишет imgs/masks/masks_colored(/masks_human) под out_dir, с необязательной
    подпапкой split (например "train"). Возвращает {"dir", "num_items"}.
    """
    out_dir = Path(out_dir)
    sub = split.strip("/") if split else ""

    def _d(name: str) -> Path:
        return (out_dir / name / sub) if sub else (out_dir / name)

    imgs_dir, masks_dir, colored_dir = _d("imgs"), _d("masks"), _d("masks_colored")
    human_dir = _d("masks_human") if include_human else None
    for d in (imgs_dir, masks_dir, colored_dir, *([human_dir] if human_dir else [])):
        d.mkdir(parents=True, exist_ok=True)

    count = 0
    for it in items:
        name, image, mask = it["name"], it["image"], it["mask"]
        image.convert("RGB").save(imgs_dir / f"{name}.jpg", quality=92)
        mask_to_id_image(mask).save(masks_dir / f"{name}.png")
        colorize_opaque(mask, class_colors).save(colored_dir / f"{name}.png")
        if human_dir is not None:
            build_human_triptych(image, mask, class_colors, class_names).save(
                human_dir / f"{name}.jpg", quality=90
            )
        count += 1

    return {"dir": str(out_dir), "num_items": count}


def zip_directory(dir_path: Path) -> bytes:
    """Упаковать папку в zip и вернуть байты (для st.download_button)."""
    dir_path = Path(dir_path)
    with tempfile.TemporaryDirectory() as tmp:
        base = str(Path(tmp) / dir_path.name)
        zip_path = shutil.make_archive(base, "zip", str(dir_path))
        return Path(zip_path).read_bytes()
