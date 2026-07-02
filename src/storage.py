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
import shutil
import time
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
