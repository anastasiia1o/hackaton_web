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


def _apply_talc_target(
    mask: np.ndarray, rng, objects_raw: list[dict], target_fraction: float
) -> float:
    """
    Перерисовать тальк так, чтобы его доля от ВСЕЙ площади маски была близка
    к target_fraction (0..1). Сценарные тальковые пятна стираются и заменяются
    пятнами, добавляемыми, пока не будет достигнута цель (или лимит попыток).
    Возвращает фактически достигнутую долю.
    """
    target_fraction = float(np.clip(target_fraction, 0.0, 0.95))
    h, w = mask.shape
    total = mask.size
    target_px = int(round(target_fraction * total))

    objects_raw[:] = [o for o in objects_raw if o["cls"] != config.CLASS_TALC]
    mask[mask == config.CLASS_TALC] = config.CLASS_BACKGROUND

    rmax = max(10, min(h, w) // 6)
    rmin = max(5, rmax // 2)
    attempts = 0
    max_attempts = 400
    while (
        int(np.count_nonzero(mask == config.CLASS_TALC)) < target_px
        and attempts < max_attempts
    ):
        objects_raw.extend(_blob(mask, rng, config.CLASS_TALC, 1, rmin, rmax + 1))
        attempts += 1

    return _safe_div(int(np.count_nonzero(mask == config.CLASS_TALC)), total)


def _apply_noise(mask: np.ndarray, rng, noise_level: float) -> float:
    """
    Имитация шума скана/грязи объектива: разбрасывает мелкие артефактные
    вкрапления по маске. noise_level в [0, 1] — доля площади, отдаваемая под
    вкрапления, растёт примерно линейно. Возвращает фактически добавленную
    долю площади (для warning геологу).
    """
    noise_level = float(np.clip(noise_level, 0.0, 1.0))
    if noise_level <= 0:
        return 0.0
    h, w = mask.shape
    total = mask.size
    # При noise_level=1.0 вкрапления покрывают до ~2% площади.
    n_specks = min(int(noise_level * 0.02 * total / 6), 20000)
    before = int(np.count_nonzero(mask == config.CLASS_ARTIFACT))
    for _ in range(n_specks):
        s = int(rng.integers(1, 4))
        y = int(rng.integers(0, max(1, h - s)))
        x = int(rng.integers(0, max(1, w - s)))
        mask[y:y + s, x:x + s] = config.CLASS_ARTIFACT
    after = int(np.count_nonzero(mask == config.CLASS_ARTIFACT))
    return _safe_div(after - before, total)


def _build_confidence_map(
    h: int, w: int, mask: np.ndarray, rng, illumination: str
) -> np.ndarray:
    """Grayscale-карта уверенности; illumination="uneven" добавляет виньетку
    (уверенность модели ниже к краям кадра — имитация плохого освещения)."""
    conf = np.full((h, w), 235, dtype=np.uint8)
    conf[mask == config.CLASS_FINE] = int(rng.integers(150, 210))     # тонкие спорнее
    conf[mask == config.CLASS_ARTIFACT] = int(rng.integers(60, 110))  # артефакты

    if illumination == "uneven":
        yy, xx = np.mgrid[0:h, 0:w]
        cy, cx = h / 2.0, w / 2.0
        dist = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
        max_dist = float(np.sqrt(cy ** 2 + cx ** 2)) or 1.0
        vignette = 1.0 - 0.55 * (dist / max_dist)  # 1.0 в центре -> ~0.45 по углам
        conf = np.clip(conf.astype(np.float64) * vignette, 20, 255).astype(np.uint8)

    return conf


def _safe_div(a: float, b: float) -> float:
    return float(a) / float(b) if b else 0.0


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

    Дополнительные (необязательные) параметры сцены поверх сценария:
      talc_fraction = 0.0..1.0  -> задать долю талька вручную (точнее сценария,
                                    перерисовывает тальковые пятна под цель)
      noise_level   = 0.0..1.0  -> добавить имитацию шума скана/грязи (мелкие
                                    артефактные вкрапления, до ~2% площади)
      illumination  = "uneven"  -> неравномерная освещённость: виньетка на
                                    карте уверенности (ниже к краям кадра)
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

    # --- Ручная доводка сцены поверх сценария (необязательно) --------------
    talc_fraction_param = params.get("talc_fraction")
    if talc_fraction_param is not None:
        achieved = _apply_talc_target(mask, rng, objects_raw, float(talc_fraction_param))
        warnings.append(
            f"Доля талька задана вручную через params.talc_fraction: "
            f"цель {float(talc_fraction_param) * 100:.1f}%, достигнуто {achieved * 100:.1f}%."
        )

    noise_level = float(params.get("noise_level", 0.0) or 0.0)
    if noise_level > 0:
        added_fraction = _apply_noise(mask, rng, noise_level)
        warnings.append(
            f"Добавлен имитационный шум скана (уровень {noise_level:.2f}): "
            f"+{added_fraction * 100:.2f}% артефактных вкраплений."
        )

    illumination = params.get("illumination", "flat")
    if illumination == "uneven":
        warnings.append(
            "Неравномерное освещение кадра — уверенность модели снижена к краям панорамы."
        )

    # --- Сохраняем маску PNG (значение пикселя = код класса) ---------------
    stem = Path(image_path).stem
    mask_path = out_dir / f"{stem}__mask.png"
    Image.fromarray(mask, mode="L").save(mask_path)

    # --- Карта уверенности (grayscale): ярче = увереннее --------------------
    conf = _build_confidence_map(h, w, mask, rng, illumination)
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
