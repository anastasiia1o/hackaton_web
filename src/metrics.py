"""
Расчёт количественных метрик по маске.

ВАЖНО: проценты считает САЙТ, а не ML. ML отдаёт только маску (пиксель = код
класса) и площади в пикселях. Здесь мы превращаем пиксели в доли/проценты по
геологически правильной формуле:

    Валидная площадь = вся площадь − артефакты (класс 4)
    Доля талька      = площадь(класс 3) / валидная площадь
    Доля тонких среди сульфидов = площадь(класс 2) / (площадь(1) + площадь(2))

Мы считаем площади ДВУМЯ путями и это ок:
- по самой маске (точный попиксельный подсчёт) — основной путь;
- список objects от ML используется для средней уверенности.
"""

from __future__ import annotations

import numpy as np

from . import config
from .schemas import MLResponse, Metrics


def _safe_div(a: float, b: float) -> float:
    """Деление, которое не падает при нуле в знаменателе."""
    return float(a) / float(b) if b else 0.0


def compute_metrics(mask: np.ndarray, ml: MLResponse) -> Metrics:
    """
    mask — 2D-массив uint8, где значение пикселя = код класса (0..4).
    ml   — ответ ML-сервиса (нужен для средней уверенности по объектам).
    """
    total_px = int(mask.size)

    # Площадь каждого класса — попиксельно по маске.
    class_area_px: dict[int, int] = {}
    for cls in (
        config.CLASS_BACKGROUND,
        config.CLASS_ORDINARY,
        config.CLASS_FINE,
        config.CLASS_TALC,
        config.CLASS_ARTIFACT,
    ):
        class_area_px[cls] = int(np.count_nonzero(mask == cls))

    artifact_px = class_area_px[config.CLASS_ARTIFACT]
    valid_px = total_px - artifact_px  # ключевая формула: убираем артефакты

    talc_px = class_area_px[config.CLASS_TALC]
    ordinary_px = class_area_px[config.CLASS_ORDINARY]
    fine_px = class_area_px[config.CLASS_FINE]
    sulphide_px = ordinary_px + fine_px

    metrics = Metrics(
        total_px=total_px,
        valid_px=valid_px,
        artifact_px=artifact_px,
        class_area_px=class_area_px,
        talc_fraction=_safe_div(talc_px, valid_px),
        sulphide_fraction=_safe_div(sulphide_px, valid_px),
        ordinary_fraction=_safe_div(ordinary_px, valid_px),
        fine_fraction=_safe_div(fine_px, valid_px),
        fine_of_sulphides=_safe_div(fine_px, sulphide_px),
        artifact_fraction=_safe_div(artifact_px, total_px),
        mean_confidence=_mean_confidence(ml),
    )
    return metrics


def _mean_confidence(ml: MLResponse) -> float:
    """Средняя уверенность, взвешенная по площади объектов."""
    if not ml.objects:
        return 1.0
    total_area = sum(o.area_px for o in ml.objects)
    if total_area == 0:
        return float(np.mean([o.confidence for o in ml.objects]))
    weighted = sum(o.confidence * o.area_px for o in ml.objects)
    return _safe_div(weighted, total_area)


def as_percent_rows(m: Metrics) -> list[dict[str, str]]:
    """
    Готовые строки для таблицы в UI и для CSV.
    Поток B может отрисовать это напрямую, ничего не пересчитывая.
    """
    def pct(x: float) -> str:
        return f"{x * 100:.1f}%"

    return [
        {"Метрика": "Валидная площадь (без артефактов)", "Значение": pct(_safe_div(m.valid_px, m.total_px))},
        {"Метрика": "Доля всех сульфидов", "Значение": pct(m.sulphide_fraction)},
        {"Метрика": "  — обычные срастания", "Значение": pct(m.ordinary_fraction)},
        {"Метрика": "  — тонкие срастания", "Значение": pct(m.fine_fraction)},
        {"Метрика": "Доля тонких среди сульфидов", "Значение": pct(m.fine_of_sulphides)},
        {"Метрика": "Доля талька (от валидной площади)", "Значение": pct(m.talc_fraction)},
        {"Метрика": "Доля артефактов", "Значение": pct(m.artifact_fraction)},
        {"Метрика": "Средняя уверенность модели", "Значение": pct(m.mean_confidence)},
    ]
