"""
infer.py — тайлинг панорамы РЕАЛЬНОЙ моделью -> JSON строго по API_CONTRACT.md.

Это «сырые факты о пикселях» из контракта: блочная маска, сетка патчей
(patch_grid), карта уверенности и objects[]. Проценты и класс руды считает САЙТ
(src/metrics.py + src/classification.py) — здесь мы их НЕ трогаем.

Как это соответствует контракту v2 (patch-classification):
  - модель классифицирует квадратные ТАЙЛЫ (train-FOV 512 px) -> сетка патчей
    rows×cols с кодом класса и уверенностью на ячейку (это `patch_grid`);
  - полноразмерная `mask` = nearest-апскейл этой сетки до image_size (поэтому
    metrics/classification читают её как обычную пиксельную маску);
  - objects[] = связные компоненты одноклассовых ячеек, bbox/площадь — в
    координатах исходника.

Форма ответа ДОЛЖНА совпадать с mock_ml/generator.py — тогда переключение
mock<->real проходит без правок в остальном сайте.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np
from PIL import Image

try:                        # работает и как пакет (python -m ml_service.server),
    from . import model as M  # и как скрипт (python ml_service/server.py)
except ImportError:
    import model as M

# Гигапиксельные панорамы: снимаем защиту PIL от «decompression bomb».
Image.MAX_IMAGE_PIXELS = None

# Легенда классов КОНТРАКТА (0..4). Дублирует src/config.py.CLASS_NAMES —
# сервис намеренно самодостаточен и не импортирует код сайта.
CLASS_LEGEND = {
    0: "Фон / нерудная матрица",
    1: "Обычные срастания",
    2: "Тонкие срастания",
    3: "Тальк",
    4: "Артефакт / исключено",
}
CLASS_ARTIFACT = 4

# Размер тайла (одно train-FOV модели) и параметры по умолчанию.
DEFAULT_TILE = int(os.getenv("ORE_ML_TILE", "512"))
DEFAULT_BATCH = int(os.getenv("ORE_ML_BATCH", "32"))
DEFAULT_MODE = os.getenv("ORE_ML_MODE_INFER", "grid")  # "grid" | "slide"
# Порог уверенности: ячейку с conf < τ помечаем кодом 4 (артефакт/неуверенно),
# как оговорено в API_CONTRACT.md. 0.0 -> отключено (модель уверена почти всегда).
CONF_THRESHOLD = float(os.getenv("ORE_ML_CONF_THRESHOLD", "0.0"))

MODEL_VERSION = "grade-se_resnext50-micronet-bg-0.2.0"


# ── Тайлинг (адаптировано из ../ore_classification/panorama_infer.py) ─────────
# Возвращают уже КОДЫ КОНТРАКТА (0 фон / 1 обычные / 2 тонкие / 3 тальк) и
# уверенность 0..1 на ячейку. Детекция фона идёт ПО ТАЙЛАМ на любых снимках,
# включая панорамы: тайл-фон (sigmoid(bg_head) > bg_thr) -> код 0, уверенность =
# вероятность фона; иначе argmax головы сорта. bg_head=None -> режим «3 класса».
def _run_grid(model, bg_head, img_np, tile, batch_size, bg_thr):
    """Непересекающаяся сетка тайлов; каждый тайл -> один класс. stride == tile."""
    H, W = img_np.shape[:2]
    rows = (H + tile - 1) // tile
    cols = (W + tile - 1) // tile
    grid_labels = np.zeros((rows, cols), dtype=np.uint8)   # коды контракта
    confs = np.zeros((rows, cols), dtype=np.float32)

    positions, tensors = [], []

    def flush(pos, tens):
        grade, bg = M.infer_batch_cascade(model, bg_head, tens)
        for (r, c), gv, bv in zip(pos, grade, bg):
            if bv > bg_thr:
                grid_labels[r, c] = M.CONTRACT_BACKGROUND
                confs[r, c] = float(bv)
            else:
                grid_labels[r, c] = M.MODEL_TO_CONTRACT[int(gv.argmax())]
                confs[r, c] = float(gv.max())

    for r in range(rows):
        for c in range(cols):
            y0, x0 = r * tile, c * tile
            crop = img_np[y0 : min(y0 + tile, H), x0 : min(x0 + tile, W)]
            positions.append((r, c))
            tensors.append(M.preprocess(crop, tile))
            if len(tensors) == batch_size:
                flush(positions, tensors)
                positions, tensors = [], []
    if tensors:
        flush(positions, tensors)

    return grid_labels, confs, rows, cols, tile


def _run_slide(model, bg_head, img_np, tile, batch_size, bg_thr):
    """Скользящее окно, шаг tile//2, soft-vote (сумма вероятностей) в перекрытиях."""
    H, W = img_np.shape[:2]
    stride = tile // 2
    ys = list(range(0, max(1, H - tile + 1), stride)) or [0]
    if ys[-1] + tile < H:
        ys.append(max(0, H - tile))
    xs = list(range(0, max(1, W - tile + 1), stride)) or [0]
    if xs[-1] + tile < W:
        xs.append(max(0, W - tile))

    rows, cols = len(ys), len(xs)
    prob_acc = np.zeros((rows, cols, 3), dtype=np.float32)
    bg_grid = np.zeros((rows, cols), dtype=np.float32)

    positions, tensors = [], []

    def flush(pos, tens):
        grade, bg = M.infer_batch_cascade(model, bg_head, tens)
        for (ri, ci), gv, bv in zip(pos, grade, bg):
            prob_acc[ri, ci] += gv
            bg_grid[ri, ci] = float(bv)

    for ri, y0 in enumerate(ys):
        for ci, x0 in enumerate(xs):
            crop = img_np[y0 : y0 + tile, x0 : x0 + tile]
            positions.append((ri, ci))
            tensors.append(M.preprocess(crop, tile))
            if len(tensors) == batch_size:
                flush(positions, tensors)
                positions, tensors = [], []
    if tensors:
        flush(positions, tensors)

    idx_labels = prob_acc.argmax(axis=2).astype(np.uint8)
    confs = prob_acc.max(axis=2) / np.maximum(prob_acc.sum(axis=2), 1e-6)
    grid_labels = M.MODEL_TO_CONTRACT[idx_labels].astype(np.uint8)
    is_bg = bg_grid > bg_thr
    grid_labels[is_bg] = M.CONTRACT_BACKGROUND
    confs = confs.astype(np.float32)
    confs[is_bg] = bg_grid[is_bg]
    return grid_labels, confs, rows, cols, stride


# ── Сборка объектов из сетки ─────────────────────────────────────────────────
def _objects_from_grid(grid_labels, grid_conf, W, H):
    """Связные компоненты одноклассовых ячеек -> objects[] в координатах исходника.

    bbox/площадь берём в координатах ПОЛНОГО кадра: nearest-апскейл делит W на
    cols, H на rows равными ячейками, поэтому ячейка (r,c) занимает прямоугольник
    примерно [c*cw, r*ch, cw, ch]. Так bbox совпадает с тем, что реально в маске.
    """
    from scipy import ndimage  # noqa: PLC0415

    rows, cols = grid_labels.shape
    cw, ch = W / cols, H / rows
    objects: list[dict] = []
    oid = 0
    # Только «рудные»/значимые коды (фон 0 не выделяем в объекты).
    for cls in (1, 2, 3, CLASS_ARTIFACT):
        labeled, n = ndimage.label(grid_labels == cls)
        for comp in range(1, n + 1):
            rs, cs = np.where(labeled == comp)
            if rs.size == 0:
                continue
            c0, c1 = int(cs.min()), int(cs.max())
            r0, r1 = int(rs.min()), int(rs.max())
            x0 = int(round(c0 * cw))
            y0 = int(round(r0 * ch))
            x1 = int(round((c1 + 1) * cw))
            y1 = int(round((r1 + 1) * ch))
            objects.append(
                {
                    "id": oid,
                    "class": int(cls),
                    "bbox": [x0, y0, min(x1, W) - x0, min(y1, H) - y0],
                    "area_px": int(round(rs.size * cw * ch)),
                    "confidence": float(round(grid_conf[rs, cs].mean() / 255.0, 3)),
                }
            )
            oid += 1
    return objects


def _upsample_nearest(grid: np.ndarray, w: int, h: int) -> np.ndarray:
    """Растянуть сетку rows×cols до (h, w) методом ближайшего соседа."""
    return np.array(
        Image.fromarray(grid.astype(np.uint8), mode="L").resize((w, h), Image.NEAREST),
        dtype=np.uint8,
    )


# ── Точка входа сервиса ───────────────────────────────────────────────────────
def analyze_image(
    image_path: str,
    out_dir: str,
    params: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Прогнать модель по изображению и вернуть dict строго по API_CONTRACT.md.

    Тяжёлые данные (маска, confidence, сетки) сохраняются PNG-файлами в out_dir,
    в JSON уходят АБСОЛЮТНЫЕ пути (сайт и сервис на одной ФС — см. контракт §Docker).
    """
    params = params or {}
    tile = int(params.get("tile", DEFAULT_TILE))
    batch = int(params.get("batch", DEFAULT_BATCH))
    mode = str(params.get("mode", DEFAULT_MODE))
    conf_thr = float(params.get("conf_threshold", CONF_THRESHOLD))
    bg_thr = float(params.get("bg_threshold", M.BG_THRESHOLD))

    t0 = time.time()
    img = Image.open(image_path).convert("RGB")
    W, H = img.size
    img_np = np.array(img)

    # `ckpt` — опциональный путь к весам (активное обучение переинференсит
    # дообученной моделью). load_model кешируется по пути, поэтому базовая модель
    # и дообученные версии сосуществуют в памяти без коллизий.
    ckpt = params.get("ckpt")
    model = M.load_model(ckpt) if ckpt else M.load_model()

    # Голова-детектор фона (общий энкодер). `bg_ckpt=""` в params отключает фон.
    # Энкодер при активном обучении заморожен -> bg_head совместим с дообученными
    # чекпоинтами и грузится независимо от пути grade-весов. Детекция фона идёт
    # ПО ТАЙЛАМ на любых снимках, включая панорамы (см. _run_grid/_run_slide).
    bg_ckpt = params.get("bg_ckpt", M.DEFAULT_BG_CKPT)
    bg_head = M.load_bg_head(bg_ckpt) if bg_ckpt else None

    if mode == "slide":
        grid_labels, confs, rows, cols, stride = _run_slide(
            model, bg_head, img_np, tile, batch, bg_thr
        )
    else:
        mode = "grid"
        grid_labels, confs, rows, cols, stride = _run_grid(
            model, bg_head, img_np, tile, batch, bg_thr
        )

    # grid_labels уже в кодах контракта (0 фон / 1 / 2 / 3). confs — 0..1.
    grid_labels = grid_labels.astype(np.uint8)
    grid_conf = np.clip(confs * 255.0, 0, 255).astype(np.uint8)

    warnings: list[str] = []
    n_bg = int((grid_labels == M.CONTRACT_BACKGROUND).sum())
    if bg_head is None:
        warnings.append(
            "[warning] Детектор фона отключён (bg_head_best.pth не найден) — "
            "фон не выделяется, режим «3 класса»."
        )
    # Ячейки с низкой уверенностью -> код 4 (артефакт/неуверенно), как в контракте.
    # Фон (код 0) не трогаем: у него «уверенность» = вероятность фона, а не сорта.
    if conf_thr > 0:
        uncertain = (confs < conf_thr) & (grid_labels != M.CONTRACT_BACKGROUND)
        n_unc = int(uncertain.sum())
        if n_unc:
            grid_labels[uncertain] = CLASS_ARTIFACT
            warnings.append(
                f"{n_unc} патч(ей) с уверенностью < {conf_thr:.2f} помечены как "
                f"неуверенные (код 4)."
            )

    # Полноразмерная блочная маска/уверенность = nearest-апскейл сетки.
    block_mask = _upsample_nearest(grid_labels, W, H)
    block_conf = _upsample_nearest(grid_conf, W, H)

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stem = Path(image_path).stem
    mask_path = out / f"{stem}__mask.png"
    conf_path = out / f"{stem}__confidence.png"
    grid_labels_path = out / f"{stem}__grid_labels.png"
    grid_conf_path = out / f"{stem}__grid_conf.png"
    Image.fromarray(block_mask, mode="L").save(mask_path)
    Image.fromarray(block_conf, mode="L").save(conf_path)
    Image.fromarray(grid_labels, mode="L").save(grid_labels_path)
    Image.fromarray(grid_conf, mode="L").save(grid_conf_path)

    objects = _objects_from_grid(grid_labels, grid_conf, W, H)

    return {
        "model_version": MODEL_VERSION,
        "inference_time_ms": int((time.time() - t0) * 1000),
        "inference_params": {
            "mode": mode, "tile": tile, "stride": stride,
            "batch": batch, "grid": [rows, cols], "device": M.device(),
            "bg_detector": bg_head is not None,
            "bg_threshold": bg_thr, "background_cells": n_bg,
        },
        "image_size": {"width": W, "height": H},
        "mask": str(mask_path.resolve()),
        "confidence_map": str(conf_path.resolve()),
        "class_legend": {int(k): v for k, v in CLASS_LEGEND.items()},
        "patch_grid": {
            "tile": tile, "stride": stride,
            "rows": rows, "cols": cols,
            "origin": [0, 0],
            "labels": str(grid_labels_path.resolve()),
            "conf": str(grid_conf_path.resolve()),
        },
        "objects": objects,
        "warnings": warnings,
    }
