"""
ACTIVE QUERY — политика отбора для active learning (см. docs/PATCH_AL_REDESIGN.md, §7).

Недостающий «active» кусок: из `patch_grid.conf`/`labels` (уверенность и границы
классов) ранжируем панорамы/регионы — наверх кластеры низкой уверенности и стыки
классов, чтобы эксперт размечал САМОЕ ИНФОРМАТИВНОЕ первым. Патчи с
`conf < τ_conf` авто-помечаются как требующие проверки.

Модуль чистый (numpy), без Streamlit — легко тестировать и звать из пайплайна.
`conf` подаётся как сетка яркостей 0..255 (как в patch_grid.conf) ИЛИ как доли
0..1 — нормализуем автоматически.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Optional

import numpy as np

from . import config


def _as_conf01(conf: np.ndarray) -> np.ndarray:
    """Привести уверенность к 0..1 (принимаем и 0..255, и уже 0..1)."""
    conf = np.asarray(conf, dtype=np.float64)
    if conf.size and conf.max() > 1.0:
        conf = conf / 255.0
    return np.clip(conf, 0.0, 1.0)


def patch_uncertainty(conf: np.ndarray) -> np.ndarray:
    """Неопределённость патча = 1 − уверенность (сетка той же формы)."""
    return 1.0 - _as_conf01(conf)


def boundary_mask(labels: np.ndarray) -> np.ndarray:
    """
    Патчи на СТЫКЕ классов: хоть один 4-сосед имеет другой код класса.
    Границы классов информативны для дообучения (там модель чаще ошибается).
    """
    labels = np.asarray(labels)
    b = np.zeros(labels.shape, dtype=bool)
    if labels.size == 0:
        return b
    b[:-1, :] |= labels[:-1, :] != labels[1:, :]
    b[1:, :] |= labels[:-1, :] != labels[1:, :]
    b[:, :-1] |= labels[:, :-1] != labels[:, 1:]
    b[:, 1:] |= labels[:, :-1] != labels[:, 1:]
    return b


@dataclass
class GridScore:
    """Оценка информативности одной панорамы/региона по её patch_grid."""
    n_patches: int
    n_low_conf: int          # патчей с conf < τ_conf
    low_conf_fraction: float
    mean_uncertainty: float  # средняя (1−conf)
    boundary_fraction: float # доля патчей на стыке классов
    priority: float          # итоговый приоритет для worklist (больше — важнее)


def score_grid(
    labels: np.ndarray,
    conf: np.ndarray,
    tau_conf: float = config.PATCH_CONF_THRESHOLD,
    background_id: int = config.CLASS_BACKGROUND,
) -> GridScore:
    """
    Посчитать информативность сетки патчей. Фон (background_id) в статистику
    уверенности НЕ включаем — интересуют рудные/спорные патчи, а не пустая
    матрица. Приоритет = смесь доли неуверенных, средней неопределённости и
    доли граничных патчей.
    """
    labels = np.asarray(labels)
    conf01 = _as_conf01(conf)
    fg = labels != background_id
    n_fg = int(fg.sum())
    if n_fg == 0:
        return GridScore(0, 0, 0.0, 0.0, 0.0, 0.0)

    unc = 1.0 - conf01
    low = (conf01 < tau_conf) & fg
    n_low = int(low.sum())
    mean_unc = float(unc[fg].mean())
    bfrac = float((boundary_mask(labels) & fg).sum()) / float(n_fg)
    low_frac = n_low / float(n_fg)

    # Приоритет: доля неуверенных доминирует, средняя неопределённость и границы
    # — вспомогательные слагаемые. Всё в 0..1, priority примерно в 0..1.
    priority = 0.6 * low_frac + 0.25 * mean_unc + 0.15 * bfrac
    return GridScore(
        n_patches=n_fg, n_low_conf=n_low, low_conf_fraction=low_frac,
        mean_uncertainty=mean_unc, boundary_fraction=bfrac, priority=priority,
    )


@dataclass
class WorklistItem:
    id: str
    score: GridScore

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "priority": self.score.priority, **asdict(self.score)}


def build_worklist(
    items: list[dict[str, Any]],
    tau_conf: float = config.PATCH_CONF_THRESHOLD,
) -> list[WorklistItem]:
    """
    Ранжировать список панорам/регионов по информативности (по убыванию).

    items: [{"id": str, "labels": 2D-array, "conf": 2D-array}, ...] — сетки
    patch_grid.labels/conf. Возвращает worklist: самое информативное — первым.
    """
    scored = [
        WorklistItem(id=str(it["id"]), score=score_grid(it["labels"], it["conf"], tau_conf))
        for it in items
    ]
    scored.sort(key=lambda w: w.score.priority, reverse=True)
    return scored


def is_low_confidence(
    conf: np.ndarray,
    labels: Optional[np.ndarray] = None,
    tau_conf: float = config.PATCH_CONF_THRESHOLD,
    min_fraction: float = 0.5,
    background_id: int = config.CLASS_BACKGROUND,
) -> bool:
    """
    Считать ли регион «неуверенным» (→ авто-статус needs_expert_review): если
    ≥ min_fraction его РУДНЫХ патчей имеют conf < τ_conf. Если labels не заданы —
    считаем по всем патчам.
    """
    conf01 = _as_conf01(conf)
    if labels is not None:
        fg = np.asarray(labels) != background_id
        vals = conf01[fg]
    else:
        vals = conf01.reshape(-1)
    if vals.size == 0:
        return False
    return float(np.mean(vals < tau_conf)) >= min_fraction
