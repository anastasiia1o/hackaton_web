# -*- coding: utf-8 -*-
"""
talc_region — сборка сплошной «области оталькования» талька из рассеянных точек.

Вендор-копия ядра из ../talc_red_zones/scripts/talc_region.py + talc_from_segmented.py:
тальк на палитровой сегментации — самый тёмный «чёрный» класс RGB≈(30,30,30),
рассеянный мелкими крапинами. density-процедура (гаусс-размытие → порог плотности)
собирает крапины в гладкие сплошные зоны — это и есть «область оталькования».

Чистый CV поверх готовой сегментации: только numpy + scipy, без обучения.
Параметры по умолчанию — финальные, подобранные визуально в тюнере
(talc_red_zones/README.md §3): dark=50, gray_tol=20, sigma=17, dens_thr=0.17.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi


def extract_talc(rgb, dark=50, gray_tol=20):
    """Тальк = тёмный серый («чёрный») класс: все каналы < dark и почти-серый."""
    r = rgb[..., 0].astype(np.int16)
    g = rgb[..., 1].astype(np.int16)
    b = rgb[..., 2].astype(np.int16)
    dark_m = (r < dark) & (g < dark) & (b < dark)
    gray_m = (np.abs(r - g) < gray_tol) & (np.abs(g - b) < gray_tol)
    return dark_m & gray_m


def _remove_small(mask, min_area_frac):
    """Убрать связные компоненты меньше доли площади кадра (выбросы-одиночки)."""
    if min_area_frac <= 0:
        return mask
    lab, n = ndi.label(mask)
    if n == 0:
        return mask
    counts = np.bincount(lab.ravel())
    keep_lab = counts >= (min_area_frac * mask.size)
    keep_lab[0] = False                              # фон
    return keep_lab[lab]


def region_density(prob, sigma=17.0, dens_thr=0.17, min_area_frac=2e-3, fill=True):
    """Density-map: гаусс-сглаживание точек → нормировка → порог → чистка.

    dens_thr задаётся в долях от максимума плотности (устойчиво к масштабу).
    Возвращает (region_bool, density_normed[0..1]).
    """
    dens = ndi.gaussian_filter(prob.astype(np.float32), sigma=sigma)
    peak = float(dens.max())
    dens_n = dens / peak if peak > 1e-9 else dens
    region = dens_n >= dens_thr
    if fill:
        region = ndi.binary_fill_holes(region)
    region = _remove_small(region, min_area_frac)
    return region, dens_n
