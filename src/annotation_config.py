"""
ANNOTATION CONFIG — классы разметки для второго этапа (active learning).

Требование ТЗ: классы разметки настраиваются через конфигурационный файл,
а не хардкодятся в интерфейсе. Источник правды — configs/annotation_classes.json
(id, машинное имя, русское название, цвет RGBA). Если файл отсутствует или
повреждён — используются встроенные значения по умолчанию (те же самые), чтобы
редактор разметки не падал на "чистой" машине.

Это НЕ то же самое, что src/config.py:CLASS_NAMES/CLASS_COLORS — те коды
описывают классы ПЕРВОГО этапа (пиксельная сегментация фаз/выход ML), а классы
здесь — итоговые категории ВТОРОГО этапа для ручной/active-learning разметки
(тальк, обычные/тонкие срастания, неопределённая, неразмеченная область).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import config

DEFAULT_CONFIG_PATH = config.BASE_DIR / "configs" / "annotation_classes.json"

_DEFAULT_CLASSES: list[dict[str, Any]] = [
    {"id": 0, "name": "unlabeled", "name_ru": "Неразмеченная область", "color": [0, 0, 0, 0]},
    {"id": 1, "name": "talc", "name_ru": "Тальк", "color": [30, 90, 230, 190]},
    {"id": 2, "name": "ordinary_intergrowth", "name_ru": "Обычные срастания", "color": [0, 200, 0, 190]},
    {"id": 3, "name": "fine_intergrowth", "name_ru": "Тонкие срастания", "color": [220, 30, 30, 190]},
    {"id": 4, "name": "uncertain", "name_ru": "Неопределённая область / требует проверки", "color": [255, 165, 0, 190]},
]

UNLABELED_ID = 0

# Статусы жизненного цикла разметки одного региона (ROI).
STATUS_DRAFT = "draft"
STATUS_REVIEWED = "reviewed"
STATUS_ACCEPTED = "accepted_for_training"
STATUS_NEEDS_REVIEW = "needs_expert_review"
ALL_STATUSES = (STATUS_DRAFT, STATUS_REVIEWED, STATUS_ACCEPTED, STATUS_NEEDS_REVIEW)
STATUS_LABELS_RU = {
    STATUS_DRAFT: "Черновик",
    STATUS_REVIEWED: "Проверено",
    STATUS_ACCEPTED: "Принято для обучения",
    STATUS_NEEDS_REVIEW: "Требует экспертной проверки",
}
# По умолчанию в экспорт для дообучения попадают только эти статусы.
EXPORTABLE_STATUSES = (STATUS_ACCEPTED,)


@dataclass(frozen=True)
class AnnotationClass:
    id: int
    name: str
    name_ru: str
    color: tuple[int, int, int, int]


def _parse(raw: list[dict[str, Any]]) -> list[AnnotationClass]:
    out = []
    for c in raw:
        color = c.get("color", [255, 0, 255, 190])
        color = tuple(int(v) for v in (color + [255] * 4)[:4])
        out.append(AnnotationClass(
            id=int(c["id"]), name=str(c["name"]),
            name_ru=str(c["name_ru"]), color=color,
        ))
    return out


def load_classes(path: Path | None = None) -> list[AnnotationClass]:
    """Загрузить конфигурируемые классы разметки из JSON (или fallback на default)."""
    p = path or DEFAULT_CONFIG_PATH
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))["classes"]
        return _parse(raw)
    except Exception:  # noqa: BLE001 — конфиг не обязателен, работаем на дефолте
        return _parse(_DEFAULT_CLASSES)


def classes_by_id(path: Path | None = None) -> dict[int, AnnotationClass]:
    return {c.id: c for c in load_classes(path)}


def classes_json_dict(path: Path | None = None) -> dict[str, Any]:
    """Представление классов для classes.json в экспорте (id -> метаданные)."""
    return {
        str(c.id): {"name": c.name, "name_ru": c.name_ru, "color": list(c.color)}
        for c in load_classes(path)
    }
