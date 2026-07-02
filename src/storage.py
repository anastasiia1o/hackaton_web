"""
STORAGE — локальное хранение файлов. Никакого облака и внешних БД.

Раскладка на диске:
  data/uploads/<image>              — исходные изображения
  data/results/<stem>/mask.png      — маска
  data/results/<stem>/confidence.png
  data/results/<stem>/result.json   — метрики + классификация
  data/results/<stem>/metrics.csv   — таблица метрик
  data/results/<stem>/corrections/  — экспертные исправления
  data/results/analysis_log.jsonl   — общий лог всех анализов (воспроизводимость)
"""

from __future__ import annotations

import json
import platform
import time
from dataclasses import asdict
from pathlib import Path

from . import config


def save_upload(file_bytes: bytes, filename: str) -> Path:
    """Сохранить загруженное изображение в data/uploads и вернуть путь."""
    config.ensure_dirs()
    dest = config.UPLOADS_DIR / filename
    dest.write_bytes(file_bytes)
    return dest


def result_dir(image_path: str) -> Path:
    """Папка результатов для конкретного изображения."""
    d = config.RESULTS_DIR / Path(image_path).stem
    d.mkdir(parents=True, exist_ok=True)
    return d


def append_log(record: dict) -> None:
    """Дописать строку в общий JSONL-лог (для воспроизводимости)."""
    config.ensure_dirs()
    record = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), **record}
    log_path = config.RESULTS_DIR / "analysis_log.jsonl"
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def save_correction(image_path: str, correction: dict) -> Path:
    """
    Сохранить экспертное исправление (для будущего дообучения ML).
    correction — произвольный dict: {class, bbox/polygon, comment, author}.
    """
    d = result_dir(image_path) / "corrections"
    d.mkdir(parents=True, exist_ok=True)
    fname = f"corr_{int(time.time()*1000)}.json"
    path = d / fname
    path.write_text(json.dumps(correction, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def list_corrections(image_path: str) -> list[dict]:
    """Прочитать все сохранённые исправления по изображению."""
    d = result_dir(image_path) / "corrections"
    if not d.exists():
        return []
    out = []
    for p in sorted(d.glob("corr_*.json")):
        try:
            out.append(json.loads(p.read_text(encoding="utf-8")))
        except Exception:  # noqa: BLE001
            continue
    return out


# --------------------------------------------------------------------------- #
# Воспроизводимость: снимки настроек и манифест прогона
# --------------------------------------------------------------------------- #

def thresholds_snapshot() -> dict:
    """Все пороги классификации на момент прогона (для повторяемости)."""
    return {
        "talc_threshold": config.TALC_THRESHOLD,
        "talc_borderline_low": config.TALC_BORDERLINE_LOW,
        "talc_borderline_high": config.TALC_BORDERLINE_HIGH,
        "low_confidence_threshold": config.LOW_CONFIDENCE_THRESHOLD,
        "artifact_warn_fraction": config.ARTIFACT_WARN_FRACTION,
        "tie_margin": config.TIE_MARGIN,
        "min_sulphide_fraction": config.MIN_SULPHIDE_FRACTION,
    }


def environment_snapshot() -> dict:
    """Версии и режим — чтобы можно было воспроизвести условия анализа."""
    return {
        "app_version": config.APP_VERSION,
        "contract_version": config.CONTRACT_VERSION,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "ml_mode": config.ML_MODE,
        "ml_service_url": config.ML_SERVICE_URL,
    }


def save_run_manifest(result) -> Path:
    """
    Сохранить полный «паспорт прогона» (run_manifest.json): что за изображение,
    какая модель и параметры, какие пороги применялись, что получилось.
    По этому файлу любой прогон можно объяснить жюри и повторить.
    """
    d = result_dir(result.image_path)
    manifest = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "image_name": result.image_name,
        "image_path": result.image_path,
        "image_size": result.ml.image_size,
        "model_version": result.ml.model_version,
        "inference_time_ms": result.ml.inference_time_ms,
        "inference_params": result.ml.inference_params,
        "thresholds": thresholds_snapshot(),
        "environment": environment_snapshot(),
        "metrics": asdict(result.metrics),
        "classification": asdict(result.classification),
        "warnings": result.ml.warnings,
    }
    path = d / "run_manifest.json"
    path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return path
