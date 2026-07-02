"""
SHARED SCHEMAS — "шов" (seam) между двумя потоками работы.

Это САМЫЙ ВАЖНЫЙ файл для командной работы двух агентов.

Поток A (Core & Analysis) ПРОИЗВОДИТ эти объекты (ml_client -> metrics ->
classification). Поток B (UI & Viewer) ПОТРЕБЛЯЕТ их и рисует.

Правило: любое изменение структур в этом файле — это изменение контракта
между агентами. Его НЕЛЬЗЯ делать молча. Сначала запись в
docs/coordination/HANDOFF.md, потом изменение здесь, потом уведомление второго
агента. Тогда потоки A и B никогда не "разъезжаются".

Мы используем обычные dataclasses (без Pydantic), чтобы не тянуть лишних
зависимостей и чтобы код читался геологом почти как текст.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Optional


# ---------------------------------------------------------------------------
# 1. СЫРЫЕ ФАКТЫ ОТ ML (то, что возвращает ML-сервис; см. API_CONTRACT.md)
#    ML отдаёт ТОЛЬКО пиксели и площади. Никаких процентов и класса руды.
# ---------------------------------------------------------------------------

@dataclass
class MLObject:
    """Один найденный объект (включение) на изображении."""
    id: int
    cls: int                 # код класса из class_legend (1..4)
    bbox: list[int]          # [x, y, w, h] в пикселях
    area_px: int             # площадь объекта в пикселях
    confidence: float        # уверенность модели 0..1


@dataclass
class MLResponse:
    """
    Полный ответ ML-сервиса на POST /analyze.
    Одинаков для mock и real режимов — в этом весь смысл контракта.
    """
    model_version: str
    inference_time_ms: int
    inference_params: dict[str, Any]
    image_size: dict[str, int]           # {"width": W, "height": H}
    mask_path: str                       # путь к PNG-маске (пиксель = код класса)
    class_legend: dict[int, str]         # {0: "...", 1: "...", ...}
    confidence_map_path: Optional[str]   # путь к grayscale-PNG уверенности
    objects: list[MLObject] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @staticmethod
    def from_json(d: dict[str, Any]) -> "MLResponse":
        """Собрать MLResponse из "сырого" JSON (ключи как в API_CONTRACT.md)."""
        objects = [
            MLObject(
                id=o["id"],
                cls=o["class"],
                bbox=o["bbox"],
                area_px=o["area_px"],
                confidence=o["confidence"],
            )
            for o in d.get("objects", [])
        ]
        return MLResponse(
            model_version=d["model_version"],
            inference_time_ms=d["inference_time_ms"],
            inference_params=d.get("inference_params", {}),
            image_size=d["image_size"],
            mask_path=d["mask"],
            class_legend={int(k): v for k, v in d["class_legend"].items()},
            confidence_map_path=d.get("confidence_map"),
            objects=objects,
            warnings=d.get("warnings", []),
        )


# ---------------------------------------------------------------------------
# 2. ПОСЧИТАННЫЕ МЕТРИКИ (это уже считает САЙТ в src/metrics.py, не ML)
# ---------------------------------------------------------------------------

@dataclass
class Metrics:
    """
    Количественные метрики по площадям.
    Валидная площадь = вся площадь МИНУС артефакты (класс 4).
    Все доли — в диапазоне 0..1 (проценты = *100 при показе).
    """
    total_px: int                 # все пиксели изображения
    valid_px: int                 # валидная площадь (без артефактов)
    artifact_px: int              # площадь артефактов
    class_area_px: dict[int, int] # площадь каждого класса в пикселях

    talc_fraction: float          # доля талька от валидной площади
    sulphide_fraction: float      # доля всех сульфидов (1+2) от валидной площади
    ordinary_fraction: float      # доля обычных срастаний от валидной площади
    fine_fraction: float          # доля тонких срастаний от валидной площади
    fine_of_sulphides: float      # доля тонких СРЕДИ сульфидов (2 / (1+2))
    artifact_fraction: float      # доля артефактов от всей площади

    mean_confidence: float        # средняя уверенность по объектам 0..1


# ---------------------------------------------------------------------------
# 3. РЕЗУЛЬТАТ КЛАССИФИКАЦИИ (rule-based, src/classification.py)
# ---------------------------------------------------------------------------

# Возможные итоговые классы руды (строки — то, что видит геолог).
ORE_TALC = "Оталькованная руда"
ORE_ORDINARY = "Рядовая руда"
ORE_REFRACTORY = "Труднообогатимая руда"
ORE_REVIEW = "Требуется экспертная проверка"


@dataclass
class Classification:
    """Итог применения прозрачных геологических правил."""
    ore_class: str            # один из ORE_* выше
    reason: str               # человекочитаемое объяснение "почему"
    needs_review: bool        # true -> геологу стоит перепроверить вручную
    rule_trace: list[str]     # пошаговый след сработавших правил (для аудита)


# ---------------------------------------------------------------------------
# 4. ИТОГОВЫЙ ПАКЕТ ДЛЯ UI (что поток B получает и рисует)
# ---------------------------------------------------------------------------

@dataclass
class AnalysisResult:
    """
    Единый объект, который UI кладёт на экран.
    Собирается пайплайном потока A: ml -> metrics -> classification.
    """
    image_name: str
    image_path: str
    ml: MLResponse
    metrics: Metrics
    classification: Classification

    def to_dict(self) -> dict[str, Any]:
        """Плоский словарь для экспорта в JSON/лог."""
        return {
            "image_name": self.image_name,
            "image_path": self.image_path,
            "model_version": self.ml.model_version,
            "inference_time_ms": self.ml.inference_time_ms,
            "image_size": self.ml.image_size,
            "metrics": asdict(self.metrics),
            "classification": asdict(self.classification),
            "warnings": self.ml.warnings,
        }
