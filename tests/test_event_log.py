"""Тесты журналов событий и безопасной очистки (src/event_log.py)."""

import pytest

from src import event_log as el


@pytest.fixture(autouse=True)
def isolated_log_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(el, "LOG_DIR", tmp_path / "logs")
    monkeypatch.setattr(el, "APP_LOG", tmp_path / "logs" / "app_events.jsonl")
    monkeypatch.setattr(el, "BATCH_LOG", tmp_path / "logs" / "batch_events.jsonl")
    yield


def test_log_and_read_events():
    el.log_import("dsA", "queue_built", count=3)
    el.log_error("dsA", "batch", "boom")
    el.log_annotation_save("dsA", "img1", "roi_0001", "draft", 1)
    el.log_export("dsA", "export_1", 5)
    el.log_batch("dsA", "started", total=10)

    app_events = el.read_events(el.APP_LOG)
    batch_events = el.read_events(el.BATCH_LOG)
    assert len(app_events) == 4
    assert len(batch_events) == 1
    assert all("ts" in e for e in app_events)


def test_clear_logs_scope_dataset_keeps_other_datasets():
    el.log_import("dsA", "queue_built", count=1)
    el.log_import("dsB", "queue_built", count=2)
    el.log_batch("dsA", "started")
    el.log_batch("dsB", "started")

    result = el.clear_logs(el.SCOPE_DATASET, dataset_id="dsA")
    assert result["app_deleted"] == 1
    assert result["batch_deleted"] == 1
    assert result["total_deleted"] == 2

    remaining_app = el.read_events(el.APP_LOG)
    remaining_batch = el.read_events(el.BATCH_LOG)
    assert all(e["dataset_id"] == "dsB" for e in remaining_app)
    assert all(e["dataset_id"] == "dsB" for e in remaining_batch)


def test_clear_logs_scope_all_removes_everything():
    el.log_import("dsA", "queue_built", count=1)
    el.log_batch("dsB", "started")
    result = el.clear_logs(el.SCOPE_ALL)
    assert result["total_deleted"] == 2
    assert el.read_events(el.APP_LOG) == []
    assert el.read_events(el.BATCH_LOG) == []


def test_clear_logs_requires_dataset_id_for_dataset_scope():
    with pytest.raises(ValueError):
        el.clear_logs(el.SCOPE_DATASET)


def test_clear_logs_invalid_scope():
    with pytest.raises(ValueError):
        el.clear_logs("bogus")
