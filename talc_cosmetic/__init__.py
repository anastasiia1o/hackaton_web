# -*- coding: utf-8 -*-
"""
talc_cosmetic — опциональная КОСМЕТИЧЕСКАЯ подсветка талька (палитровая
сегментация + density-сборка «области оталькования»), включаемая галочкой
ПОСЛЕ основного блочного анализа. Чистый CV (numpy + scipy), на предсказание
модели/метрики/класс руды НЕ влияет.

Вендор-пакет: код из ../deploy_segment (seglib) и ../talc_red_zones (talc_region),
внесён внутрь репозитория ради самодостаточности (см. CLAUDE.md).
"""
from .overlay import (
    TalcOverlayResult,
    compute_talc_overlay,
    density_heatmap,
    DEFAULTS,
)

__all__ = [
    "TalcOverlayResult",
    "compute_talc_overlay",
    "density_heatmap",
    "DEFAULTS",
]
