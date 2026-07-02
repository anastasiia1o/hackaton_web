"""
PIPELINE — склейка потока A в одну функцию.

Это фасад, которым пользуется UI (поток B) и batch-обработка.
Один вызов run_analysis(image_path) делает всё:

    ML (маска, объекты)  →  metrics (проценты)  →  classification (класс руды)

и возвращает единый AnalysisResult (см. schemas.py), готовый к отрисовке.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import numpy as np
from PIL import Image

from . import config, ml_client, metrics as metrics_mod, classification as clf
from .schemas import AnalysisResult


def load_mask(mask_path: str) -> np.ndarray:
    """Загрузить PNG-маску как 2D-массив кодов классов."""
    return np.array(Image.open(mask_path).convert("L"), dtype=np.uint8)


def run_analysis(
    image_path: str,
    params: Optional[dict[str, Any]] = None,
    mode: Optional[str] = None,
) -> AnalysisResult:
    """Полный сквозной анализ одного изображения."""
    ml = ml_client.analyze(image_path, params=params, mode=mode)
    mask = load_mask(ml.mask_path)
    m = metrics_mod.compute_metrics(mask, ml)
    classification = clf.classify(m)
    return AnalysisResult(
        image_name=Path(image_path).name,
        image_path=image_path,
        ml=ml,
        metrics=m,
        classification=classification,
    )
