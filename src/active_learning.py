"""
ACTIVE LEARNING — интерактивный цикл прямо на странице OreVision.

Настоящее активное обучение (в отличие от ручного аннотатора «Разметка эксперта»):
геолог правит предсказание на снимке → модель ДООБУЧАЕТСЯ на этом исправлении →
результат СРАЗУ переинференсится и показывается на той же картинке («было/стало»).

Оркестрация тонкая, тяжёлое — в ml_service:
  - `ml_service/train.py:quick_finetune_multihead` — быстрое дообучение ОБЕИХ
    голов (сорт + фон, энкодер заморожен, признаки кешируются) на патчах
    эксперта + якорях; сохраняет ОДИН мультиголовый чекпоинт (encoder+head+
    bg_head в одном .pth), поэтому `ORE_ML_CKPT` подключает обе головы разом;
  - `ml_service/infer.py:analyze_image(params={"ckpt": ...})` — переинференс
    дообученной моделью (load_model кешируется по пути весов).

Патчи собираются из двух источников:
  - ИСПРАВЛЕНИЯ эксперта (высокий вес) — кроп области + верный класс;
  - ЯКОРЯ (низкий вес) — случайные тайлы снимка с ТЕКУЩИМ предсказанием модели,
    чтобы дообучение не «схлопнуло» все классы в исправленный (rehearsal).
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Optional

import numpy as np
from PIL import Image

from . import config, contract
from .pipeline import load_mask


def _al_dir() -> Path:
    d = config.DATA_DIR / "active_learning"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _contract_to_model() -> dict[int, int]:
    """Код класса контракта (1/2/3) -> индекс выхода модели (talc0/ord1/fine2)."""
    from ml_service.train import CONTRACT_TO_MODEL

    return CONTRACT_TO_MODEL


def build_correction_items(
    image_path: str, corrections: list[dict[str, Any]], *, weight: float = 1.0,
) -> list[tuple[Image.Image, int, float]]:
    """
    Патчи из сохранённых исправлений эксперта: кроп области (в исходном
    разрешении) + верный класс. Нетренируемые классы (фон/артефакт) пропускаем.
    """
    from ui import viewer

    c2m = _contract_to_model()
    items: list[tuple[Image.Image, int, float]] = []
    for c in corrections:
        rf = c.get("region_fraction")
        cid = int(c.get("correct_class", 0))
        if not rf or cid not in c2m:
            continue
        crop, _meta = viewer.crop_region_highres(
            str(image_path), rf["x0"], rf["y0"], rf["x1"], rf["y1"],
        )
        items.append((crop, c2m[cid], weight))
    return items


def build_bg_items(
    image_path: str,
    corrections: list[dict[str, Any]],
    *,
    weight: float = 1.0,
) -> list[tuple[Image.Image, int, float]]:
    """
    Патчи для дообучения ГОЛОВЫ ФОНА из исправлений эксперта. bg_label:
      - исправление на ФОН (код 0)        -> 1 (это фон),
      - исправление на руду (коды 1/2/3)  -> 0 (это НЕ фон),
      - артефакт (код 4)                  -> пропускаем (неоднозначно для фона).
    Именно исправления «фон → руда» вытаскивают тайлы из-под маски фона.
    """
    from ui import viewer

    items: list[tuple[Image.Image, int, float]] = []
    for c in corrections:
        rf = c.get("region_fraction")
        cid = int(c.get("correct_class", 0))
        if not rf or cid == config.CLASS_ARTIFACT:
            continue
        bg_label = 1 if cid == config.CLASS_BACKGROUND else 0
        crop, _meta = viewer.crop_region_highres(
            str(image_path), rf["x0"], rf["y0"], rf["x1"], rf["y1"],
        )
        items.append((crop, bg_label, weight))
    return items


def build_bg_anchor_items(
    base_img: Image.Image,
    base_mask: np.ndarray,
    corrections: list[dict[str, Any]],
    *,
    n: int = 8,
    tile_frac: float = 0.18,
    weight: float = 0.3,
    seed: int = 43,
) -> list[tuple[Image.Image, int, float]]:
    """
    Якоря для головы фона: случайные тайлы, размеченные ТЕКУЩИМ предсказанием
    (bg_label: 1 если мажоритарный класс тайла — фон, иначе 0), малый вес. Держат
    в дообучении оба класса (фон/руда), чтобы правки не «схлопнули» всё в руду.
    Тайлы внутри исправленных областей исключаются.
    """
    rng = random.Random(seed)
    W, H = base_img.size
    side = max(32, int(min(W, H) * tile_frac))

    excl = []
    for c in corrections:
        rf = c.get("region_fraction")
        if rf:
            excl.append((rf["x0"] * W, rf["y0"] * H, rf["x1"] * W, rf["y1"] * H))

    if base_mask.shape[:2] != (H, W):
        base_mask = np.array(
            Image.fromarray(base_mask.astype(np.uint8), mode="L").resize((W, H), Image.NEAREST),
            dtype=np.uint8,
        )

    items: list[tuple[Image.Image, int, float]] = []
    tries = 0
    while len(items) < n and tries < n * 10:
        tries += 1
        x0 = rng.randint(0, max(0, W - side))
        y0 = rng.randint(0, max(0, H - side))
        cx, cy = x0 + side / 2, y0 + side / 2
        if any(ex0 <= cx <= ex1 and ey0 <= cy <= ey1 for ex0, ey0, ex1, ey1 in excl):
            continue
        sub = base_mask[y0:y0 + side, x0:x0 + side]
        if sub.size == 0:
            continue
        vals, counts = np.unique(sub, return_counts=True)
        maj = int(vals[counts.argmax()])
        if maj == config.CLASS_ARTIFACT:
            continue
        bg_label = 1 if maj == config.CLASS_BACKGROUND else 0
        crop = base_img.crop((x0, y0, x0 + side, y0 + side))
        items.append((crop, bg_label, weight))
    return items


def build_anchor_items(
    base_img: Image.Image,
    base_mask: np.ndarray,
    corrections: list[dict[str, Any]],
    *,
    n: int = 8,
    tile_frac: float = 0.18,
    weight: float = 0.3,
    seed: int = 42,
) -> list[tuple[Image.Image, int, float]]:
    """
    Якорные патчи: случайные тайлы снимка, размеченные ТЕКУЩИМ предсказанием
    модели (мажоритарный класс в тайле), с малым весом. Тайлы, центр которых
    попадает в исправленную область, и тайлы с нетренируемым мажор-классом
    (фон/артефакт) пропускаются. Стабилизируют дообучение (rehearsal).
    """
    c2m = _contract_to_model()
    rng = random.Random(seed)
    W, H = base_img.size
    side = max(32, int(min(W, H) * tile_frac))

    excl = []
    for c in corrections:
        rf = c.get("region_fraction")
        if rf:
            excl.append((rf["x0"] * W, rf["y0"] * H, rf["x1"] * W, rf["y1"] * H))

    # маску приводим к размеру base (координаты тайлов — в пикселях base)
    if base_mask.shape[:2] != (H, W):
        base_mask = np.array(
            Image.fromarray(base_mask.astype(np.uint8), mode="L").resize((W, H), Image.NEAREST),
            dtype=np.uint8,
        )

    items: list[tuple[Image.Image, int, float]] = []
    tries = 0
    while len(items) < n and tries < n * 10:
        tries += 1
        x0 = rng.randint(0, max(0, W - side))
        y0 = rng.randint(0, max(0, H - side))
        cx, cy = x0 + side / 2, y0 + side / 2
        if any(ex0 <= cx <= ex1 and ey0 <= cy <= ey1 for ex0, ey0, ex1, ey1 in excl):
            continue
        sub = base_mask[y0:y0 + side, x0:x0 + side]
        if sub.size == 0:
            continue
        vals, counts = np.unique(sub, return_counts=True)
        maj = int(vals[counts.argmax()])
        if maj not in c2m:
            continue
        crop = base_img.crop((x0, y0, x0 + side, y0 + side))
        items.append((crop, c2m[maj], weight))
    return items


def retrain_and_save(
    image_path: str,
    items: list[tuple[Image.Image, int, float]],
    *,
    version: int,
    from_ckpt: Optional[str] = None,
    epochs: int = 60,
    bg_items: Optional[list[tuple[Image.Image, int, float]]] = None,
    bg_epochs: int = 150,
) -> tuple[str, dict]:
    """
    Дообучить голову СОРТА и (если есть bg_items) голову ФОНА поверх ОДНОГО
    замороженного энкодера и сохранить ОДИН версионированный мультиголовый
    чекпоинт (encoder+head+bg_head) под data/active_learning/.

    from_ckpt=None → стартуем с вшитой модели; передать предыдущую AL-версию
    для накопительного дообучения (обеих голов сразу — это один и тот же файл).
    Возвращает (ckpt_path, report).
    """
    from ml_service import train as T
    from ml_service.model import DEFAULT_CKPT

    base_ckpt = from_ckpt or DEFAULT_CKPT
    stem = Path(image_path).stem
    save = _al_dir() / f"{stem}__al_v{version}.pth"
    report = T.quick_finetune_multihead(
        items, bg_items or [], from_ckpt=base_ckpt, save_ckpt=str(save),
        epochs=epochs, bg_epochs=bg_epochs,
    )
    report["version"] = version
    return str(save), report


def reanalyze(image_path: str, ckpt_path: str):
    """
    Переинференс изображения дообученной моделью → AnalysisResult (маска +
    метрики + класс руды), как в pipeline.run_analysis, но с явным чекпоинтом и
    ОТДЕЛЬНОЙ выходной папкой (чтобы не затирать исходный результат «до»).
    ckpt_path — мультиголовый чекпоинт (encoder+head+bg_head в одном файле).
    """
    from ml_service import infer

    from . import classification as clf
    from . import metrics as metrics_mod
    from .schemas import AnalysisResult, MLResponse

    out_dir = _al_dir() / "results" / Path(ckpt_path).stem
    params: dict[str, Any] = {"ckpt": str(ckpt_path)}
    raw = infer.analyze_image(str(image_path), out_dir=str(out_dir), params=params)
    if config.VALIDATE_ML_RESPONSE:
        contract.assert_valid(raw)
    ml = MLResponse.from_json(raw)

    max_side = max(ml.image_size.get("width", 0), ml.image_size.get("height", 0))
    if max_side > config.MAX_DIMENSION_WARN:
        m = metrics_mod.compute_metrics_from_mask_path(ml.mask_path, ml)
    else:
        m = metrics_mod.compute_metrics(load_mask(ml.mask_path), ml)
    classification = clf.classify(m)
    return AnalysisResult(
        image_name=Path(image_path).name,
        image_path=str(image_path),
        ml=ml,
        metrics=m,
        classification=classification,
    )
