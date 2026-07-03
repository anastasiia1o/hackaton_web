"""
EVENT LOG — журналы событий приложения (импорт, batch, ошибки, разметка, экспорт).

Хранение: data/logs/app_events.jsonl (импорт/ошибки/сохранения разметки/экспорт)
и data/logs/batch_events.jsonl (пакетная обработка). Отдельно от
data/results/analysis_log.jsonl (журнал самого ML-анализа — см. src/storage.py),
который эти функции НЕ трогают.

clear_logs() — единственный способ удалять записи логов, и он гарантированно
не касается файлов вне data/logs/ (изображения, ROI, маски, экспорты, модели,
конфиги остаются нетронутыми).
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional

from . import config

LOG_DIR = config.DATA_DIR / "logs"
APP_LOG = LOG_DIR / "app_events.jsonl"
BATCH_LOG = LOG_DIR / "batch_events.jsonl"

KIND_IMPORT = "import"
KIND_ERROR = "error"
KIND_ANNOTATION_SAVE = "annotation_save"
KIND_EXPORT = "export"
KIND_BATCH = "batch"


def _ensure_dir() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)


def _append(path: Path, record: dict) -> None:
    _ensure_dir()
    record = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), **record}
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def log_import(dataset_id: str, event: str, **payload) -> None:
    _append(APP_LOG, {"kind": KIND_IMPORT, "dataset_id": dataset_id, "event": event, **payload})


def log_error(dataset_id: Optional[str], where: str, message: str) -> None:
    _append(APP_LOG, {"kind": KIND_ERROR, "dataset_id": dataset_id, "where": where, "message": message})


def log_annotation_save(dataset_id: str, image_id: str, region_id: str, status: str, revision: int) -> None:
    _append(APP_LOG, {
        "kind": KIND_ANNOTATION_SAVE, "dataset_id": dataset_id,
        "image_id": image_id, "region_id": region_id, "status": status, "revision": revision,
    })


def log_export(dataset_id: str, export_id: str, num_samples: int) -> None:
    _append(APP_LOG, {
        "kind": KIND_EXPORT, "dataset_id": dataset_id,
        "export_id": export_id, "num_samples": num_samples,
    })


def log_batch(dataset_id: Optional[str], event: str, **payload) -> None:
    _append(BATCH_LOG, {"kind": KIND_BATCH, "dataset_id": dataset_id, "event": event, **payload})


def read_events(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def read_all_events() -> list[dict]:
    """Оба журнала вместе, отсортированы по времени (для страницы «Логи»)."""
    events = read_events(APP_LOG) + read_events(BATCH_LOG)
    return sorted(events, key=lambda e: e.get("ts", ""))


SCOPE_DATASET = "dataset"
SCOPE_ALL = "all"


def clear_logs(scope: str, dataset_id: Optional[str] = None) -> dict:
    """
    Безопасно очистить журналы событий. НИКОГДА не трогает ничего, кроме
    data/logs/*.jsonl (изображения/ROI/маски/экспорты/отчёты/модели/конфиги
    не затрагиваются, т.к. эта функция даже не знает их путей).

    scope="dataset" — удалить только записи с dataset_id == dataset_id.
    scope="all"      — удалить вообще все записи логов (сами файлы остаются,
                        просто пустыми).
    Возвращает {"app_deleted": N, "batch_deleted": M, "total_deleted": N+M}.
    """
    if scope not in (SCOPE_DATASET, SCOPE_ALL):
        raise ValueError(f"Неизвестный scope: {scope}")
    if scope == SCOPE_DATASET and not dataset_id:
        raise ValueError("scope='dataset' требует dataset_id")

    deleted = {"app_deleted": 0, "batch_deleted": 0}
    for label, path in (("app_deleted", APP_LOG), ("batch_deleted", BATCH_LOG)):
        events = read_events(path)
        if scope == SCOPE_ALL:
            keep = []
            removed = len(events)
        else:
            keep = [e for e in events if e.get("dataset_id") != dataset_id]
            removed = len(events) - len(keep)
        deleted[label] = removed
        _ensure_dir()
        with open(path, "w", encoding="utf-8") as f:
            for e in keep:
                f.write(json.dumps(e, ensure_ascii=False) + "\n")

    deleted["total_deleted"] = deleted["app_deleted"] + deleted["batch_deleted"]
    return deleted
