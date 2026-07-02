# PLAN — Поток A (Core & Analysis)

Ты отвечаешь за данные и логику: ML-клиент, метрики, классификацию, отчёты.
UI (`app.py`, `ui/`) — не твой файл; трогай только через SEAM и HANDOFF.

## Что уже готово (фундамент)
- `config.py`, `schemas.py` (SEAM), `ml_client.py`, `mock_ml/generator.py`
- `metrics.py`, `classification.py`, `pipeline.py`, `reports.py`, `storage.py`
- `src/contract.py` — валидатор контракта + `docs/ML_INTEGRATION_GUIDE.md`
- `src/gis_export.py` — экспорт GeoJSON/Shapefile
- Воспроизводимость: `run_manifest.json` + расширенный лог
- `tests/` — 18 тестов (классификация + edge-кейсы + контракт + GIS), все зелёные

## Твой backlog (по приоритету)

1. **Держать тесты зелёными.** Любая правка логики → сначала тест.
   `pytest -q` обязана проходить перед каждым PR.

2. **Реальный ML-режим.** Когда команда `orevision-ml` поднимет `:8001`:
   - проверить `_analyze_real` (multipart, разбор JSON);
   - сверить поля ответа с `API_CONTRACT.md`; расхождения — в `HANDOFF.md`;
   - решить вопрос передачи маски (общий volume vs endpoint) — см. контракт §«v2».

3. **Метрики для панорам >10000px.** Если ML вернёт маску по тайлам —
   агрегировать площади классов суммированием по тайлам (не грузить всё в RAM).
   Договориться с потоком B о формате прогресса через `HANDOFF.md`.

4. **Экспорт GeoJSON/Shapefile** объектов (доп. пожелание ТЗ, для ГИС).
   Взять `objects[*].bbox`, отдать как полигоны. Добавить в `reports.py`.

5. **Edge-кейсы в тестах:** пустая маска, 100% артефактов (valid_px=0 —
   деление уже защищено `_safe_div`, но нужен явный тест и warning).

## Границы (не делай)
- Не пиши обучение нейросети.
- Не меняй `app.py` / `ui/*` — это поток B.
- Меняешь `schemas.py`/`config.py` → сначала запись в `HANDOFF.md`.

## Definition of Done для задачи потока A
- `pytest -q` зелёный; `py_compile` без ошибок;
- если изменился контракт — запись в `HANDOFF.md`;
- PR в `main`, строка в `BOARD.md` переведена в DONE.
