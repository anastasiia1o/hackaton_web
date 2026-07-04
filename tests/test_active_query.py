"""
Тесты политики active learning (src/active_query.py).

Запуск:  python scratchpad/runtests.py tests.test_active_query
"""

import numpy as np

from src import active_query as aq
from src import config


def test_patch_uncertainty_normalizes_0_255():
    conf = np.array([[0, 255], [128, 255]], dtype=np.uint8)
    unc = aq.patch_uncertainty(conf)
    assert unc[0, 0] == 1.0        # conf 0 → неопределённость 1
    assert unc[0, 1] == 0.0        # conf 255 → неопределённость 0
    assert abs(unc[1, 0] - 0.498) < 0.01


def test_boundary_mask_marks_class_edges():
    labels = np.array([[1, 1, 2], [1, 1, 2], [1, 1, 2]])
    b = aq.boundary_mask(labels)
    assert b[0, 1]      # рядом со сменой класса 1→2
    assert b[0, 2]
    assert not b[0, 0]  # внутри однородной зоны — не граница


def test_score_grid_ignores_background():
    labels = np.zeros((10, 10), dtype=np.uint8)   # весь фон
    conf = np.zeros((10, 10), dtype=np.uint8)
    s = aq.score_grid(labels, conf)
    assert s.n_patches == 0
    assert s.priority == 0.0


def test_low_confidence_region_scores_higher():
    labels = np.full((10, 10), config.CLASS_FINE, dtype=np.uint8)
    high = np.full((10, 10), 240, dtype=np.uint8)   # уверенно
    low = np.full((10, 10), 60, dtype=np.uint8)     # неуверенно
    s_high = aq.score_grid(labels, high)
    s_low = aq.score_grid(labels, low)
    assert s_low.priority > s_high.priority
    assert s_low.n_low_conf > s_high.n_low_conf


def test_build_worklist_sorts_most_informative_first():
    labels = np.full((8, 8), config.CLASS_TALC, dtype=np.uint8)
    items = [
        {"id": "confident", "labels": labels, "conf": np.full((8, 8), 250, np.uint8)},
        {"id": "uncertain", "labels": labels, "conf": np.full((8, 8), 40, np.uint8)},
        {"id": "mixed", "labels": labels, "conf": np.full((8, 8), 130, np.uint8)},
    ]
    wl = aq.build_worklist(items)
    assert wl[0].id == "uncertain"
    assert wl[-1].id == "confident"


def test_is_low_confidence_threshold():
    labels = np.full((10, 10), config.CLASS_FINE, dtype=np.uint8)
    assert aq.is_low_confidence(np.full((10, 10), 40, np.uint8), labels)
    assert not aq.is_low_confidence(np.full((10, 10), 240, np.uint8), labels)
