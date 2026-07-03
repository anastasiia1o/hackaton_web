# BOARD — доска задач OreVision

Обновляют **оба агента**. Пиши в конец нужной секции, чужие строки не трогай.
Статусы: `TODO` → `DOING` → `DONE`. Владелец: `A` (Core) или `B` (UI).

Формат строки: `- [СТАТУС] (владелец) описание — ветка/PR`

---

## Общий фундамент (готов до старта параллельной работы)

- [DONE] (A) Каркас репозитория, config, schemas (SEAM) — на main
- [DONE] (A) API_CONTRACT.md зафиксирован (v1) — на main
- [DONE] (A) mock ML + ml_client (mock/real) — на main
- [DONE] (A) metrics + classification + тесты (все зелёные) — на main
- [DONE] (A) reports CSV/JSON/PDF, storage — на main
- [DONE] (B) Базовый Streamlit-экран: загрузка→overlay→метрики→экспорт — на main
- [DONE] (A/B) batch_process.py — на main

## Поток A — Core & Analysis

- [DONE] (A) Валидатор контракта `src/contract.py` + гайд `docs/ML_INTEGRATION_GUIDE.md` (5 тестов)
- [DONE] (A) Закалка логики: valid=0, нет сульфидов, ничья 50/50, тальк на границе (4 теста)
- [DONE] (A) Экспорт GeoJSON/Shapefile объектов (`src/gis_export.py`, 3 теста)
- [DONE] (A) Воспроизводимость: run_manifest.json + расширенный analysis_log.jsonl
- [TODO] (A) Локальный ML-стаб на :8001 для теста реального пути (отложено по решению команды)
- [TODO] (A) Реальный режим: проверить `_analyze_real` против настоящего ML (когда поднимут :8001)
- [TODO] (A) Тайловый расчёт метрик для панорам >10000px (агрегация по тайлам)
- [TODO] (A) Обогатить mock-генератор: регулируемый % талька, шум, неравномерное освещение

## Поток B — UI & Viewer

- [DONE] (B) Zoom/pan для больших изображений (самодостаточный inline-вьюер) — stream-b/viewer-zoom
- [DONE] (B) Minimap/навигатор по панораме (встроен в вьюер) — stream-b/viewer-zoom
- [DONE] (B) Инспектор участка в высоком разрешении (ленивый decode по кнопке, лимит памяти) — stream-b/region-inspector; полноценный DeepZoom-пирамидный вьюер не делаем (нужна vendored JS-либа/статик-сервинг — против clean-machine)
- [DONE] (B) Многостраничность: страница «Batch», страница «История/лог» — stream-b/multipage
- [DONE] (B) Редактор экспертной коррекции: выделение области + правильный класс, save_correction — stream-b/mask-editor (dependency-free, т.к. drawable-canvas несовместим со Streamlit 1.58)
- [DONE] (B) Тумблеры слоёв + карта уверенности как переключаемый слой overlay — stream-b/confidence-layer
- [DONE] (B) Интеграция выходов A: кнопка GeoJSON, обработка ContractError, история под обогащённый лог — stream-b/integrate-a-outputs
- [DONE] (A, по прямой просьбе — см. HANDOFF) UX: ползунки "Область по X/Y" в
  инспекторе участка и экспертной коррекции заменены на drag/resize мышью
  (кастомный vanilla-JS компонент, без npm) — stream-a/region-picker-ux;
  browser-тест перетаскивания не проводился (нет браузера у агента), нужна
  ручная проверка

## Интеграция / упаковка

- [DONE] (A) Docker/чистая машина: пофикшены баг `.gitignore` (папки `data/*` не
  создавались на чистом клоне), отсутствующий `pandas` в requirements.txt,
  телеметрия Streamlit (нарушала "всё локально"), headless-режим для контейнера
  — stream-a/docker-readme-cleanup. Секция ML в docker-compose.yml остаётся
  TODO — появится, когда `orevision-ml` даст образ.
- [TODO] (A/B) README: финальные скриншоты и видео-демо для сдачи (руками, не
  автоматизировано)
- [DONE] (A) Проверка запуска на «чистой» машине: свежий venv + `pip install -r
  requirements.txt` + `pytest` + поднятие `streamlit run app.py` (все страницы
  отвечают 200) — проверено без Docker (Docker Desktop недоступен в среде
  агента); сам `docker build`/`docker compose up` руками не гонялся, статически
  вычитан
