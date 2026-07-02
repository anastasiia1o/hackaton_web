"""
MOCK ML — генератор синтетических результатов "как будто от нейросети".

Зачем: чтобы сайт можно было запустить и показать ПОЛНЫЙ сквозной сценарий
ещё до того, как ML-команда поднимет реальный сервис на :8001.

Что генерируем (строго по API_CONTRACT.md):
  - PNG-маску, где значение пикселя = код класса (0..4);
  - grayscale-PNG "карту уверенности";
  - список объектов objects с bbox/area/confidence;
  - метаданные (model_version, время, warnings).

Форма ответа ИДЕНТИЧНА реальному ML — поэтому позже mock просто заменяется
на real без изменений в остальном коде сайта.
"""

from __future__ import annotations

import time
import hashlib
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from src import config


def _rng_for(image_path: str, seed_params: dict[str, Any] | None) -> np.random.Generator:
    """
    Детерминированный генератор: одно и то же изображение -> одна и та же маска.
    Так демо стабильно и воспроизводимо (важно для жюри).
    """
    key = image_path + str(sorted((seed_params or {}).items()))
    h = int(hashlib.sha256(key.encode()).hexdigest(), 16) % (2**32)
    return np.random.default_rng(h)


def _blob(mask: np.ndarray, rng, cls: int, n: int, rmin: int, rmax: int) -> list[dict]:
    """Нарисовать n круглых "включений" класса cls и вернуть их как объекты."""
    h, w = mask.shape
    objects: list[dict] = []
    yy, xx = np.ogrid[:h, :w]
    for _ in range(n):
        r = int(rng.integers(rmin, rmax))
        cx = int(rng.integers(r, w - r))
        cy = int(rng.integers(r, h - r))
        disk = (xx - cx) ** 2 + (yy - cy) ** 2 <= r * r
        mask[disk] = cls
        area = int(disk.sum())
        objects.append(
            {
                "cls": cls,
                "bbox": [cx - r, cy - r, 2 * r, 2 * r],
                "area_px": area,
                "confidence": float(round(rng.uniform(0.7, 0.98), 3)),
            }
        )
    return objects


def generate(
    image_path: str,
    out_dir: Path,
    params: dict[str, Any] | None = None,
    size: tuple[int, int] = (900, 700),
) -> dict[str, Any]:
    """
    Сгенерировать mock-ответ. Возвращает dict строго в формате API_CONTRACT.md.
    Маска и confidence сохраняются файлами в out_dir (тяжёлые данные — не в JSON).

    params поддерживает "сценарии" для демо:
      scenario = "talc"        -> много талька (>10%) -> Оталькованная
      scenario = "refractory"  -> преобладают тонкие  -> Труднообогатимая
      scenario = "ordinary"    -> преобладают обычные -> Рядовая
      scenario = "review"      -> пограничный тальк    -> Экспертная проверка
    """
    t0 = time.time()
    out_dir.mkdir(parents=True, exist_ok=True)
    params = params or {}
    scenario = params.get("scenario", "refractory")
    rng = _rng_for(image_path, params)

    w, h = size
    mask = np.zeros((h, w), dtype=np.uint8)  # 0 = фон

    warnings: list[str] = []
    objects_raw: list[dict] = []

    # Немного артефактов почти всегда (царапины/грязь) — узкие полосы.
    for _ in range(int(rng.integers(1, 4))):
        y = int(rng.integers(0, h))
        thick = int(rng.integers(2, 6))
        mask[max(0, y - thick):min(h, y + thick), :] = config.CLASS_ARTIFACT

    # Наборы включений под сценарий.
    if scenario == "talc":
        objects_raw += _blob(mask, rng, config.CLASS_ORDINARY, 6, 25, 45)
        objects_raw += _blob(mask, rng, config.CLASS_FINE, 6, 20, 40)
        # Тальк — крупные рассеянные тёмные зоны (>10% валидной площади).
        objects_raw += _blob(mask, rng, config.CLASS_TALC, 10, 55, 90)
    elif scenario == "ordinary":
        objects_raw += _blob(mask, rng, config.CLASS_ORDINARY, 14, 30, 55)
        objects_raw += _blob(mask, rng, config.CLASS_FINE, 4, 15, 30)
        objects_raw += _blob(mask, rng, config.CLASS_TALC, 2, 20, 35)
    elif scenario == "review":
        objects_raw += _blob(mask, rng, config.CLASS_ORDINARY, 8, 25, 45)
        objects_raw += _blob(mask, rng, config.CLASS_FINE, 8, 25, 45)
        # Тальк ровно около порога (~10%) -> пограничный случай.
        objects_raw += _blob(mask, rng, config.CLASS_TALC, 7, 46, 66)
        warnings.append("Доля талька близка к порогу 10% — возможна неоднозначность.")
    else:  # "refractory" по умолчанию
        objects_raw += _blob(mask, rng, config.CLASS_ORDINARY, 5, 20, 40)
        objects_raw += _blob(mask, rng, config.CLASS_FINE, 16, 25, 50)
        objects_raw += _blob(mask, rng, config.CLASS_TALC, 2, 15, 30)

    # --- Сохраняем маску PNG (значение пикселя = код класса) ---------------
    stem = Path(image_path).stem
    mask_path = out_dir / f"{stem}__mask.png"
    Image.fromarray(mask, mode="L").save(mask_path)

    # --- Карта уверенности (grayscale): ярче = увереннее --------------------
    conf = np.full((h, w), 235, dtype=np.uint8)
    conf[mask == config.CLASS_FINE] = int(rng.integers(150, 210))     # тонкие спорнее
    conf[mask == config.CLASS_ARTIFACT] = int(rng.integers(60, 110))  # артефакты
    conf_path = out_dir / f"{stem}__confidence.png"
    Image.fromarray(conf, mode="L").save(conf_path)

    # --- Присвоим id объектам и приведём к контракту -----------------------
    objects = []
    for i, o in enumerate(objects_raw):
        objects.append(
            {
                "id": i,
                "class": o["cls"],
                "bbox": o["bbox"],
                "area_px": o["area_px"],
                "confidence": o["confidence"],
            }
        )

    inference_ms = int((time.time() - t0) * 1000)

    return {
        "model_version": "mock-0.1.0",
        "inference_time_ms": inference_ms,
        "inference_params": {"scenario": scenario, **params},
        "image_size": {"width": w, "height": h},
        "mask": str(mask_path),
        "class_legend": {int(k): v for k, v in config.CLASS_NAMES.items()},
        "confidence_map": str(conf_path),
        "objects": objects,
        "warnings": warnings,
    }
