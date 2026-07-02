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

- [TODO] (A) Реальный режим: проверить `_analyze_real` против настоящего ML (когда поднимут :8001)
- [TODO] (A) Тайловый расчёт метрик для панорам >10000px (агрегация по тайлам)
- [TODO] (A) Экспорт GeoJSON/Shapefile объектов (доп. пожелание из ТЗ)
- [TODO] (A) Расширить тесты: edge-кейсы (пустая маска, 100% артефактов)

## Поток B — UI & Viewer

- [DONE] (B) Zoom/pan для больших изображений (самодостаточный inline-вьюер) — stream-b/viewer-zoom
- [DONE] (B) Minimap/навигатор по панораме (встроен в вьюер) — stream-b/viewer-zoom
- [TODO] (B) Тайловая ленивая загрузка больших TIFF (без зависаний) — сейчас decode-downscale до 2600px
- [DONE] (B) Многостраничность: страница «Batch», страница «История/лог» — stream-b/multipage
- [TODO] (B) Полноценный редактор экспертной коррекции маски (рисование области)
- [DONE] (B) Тумблеры слоёв + карта уверенности как переключаемый слой overlay — stream-b/confidence-layer

## Интеграция / упаковка

- [TODO] (A/B) Docker Compose: сайт + (позже) ML в одной сети — черновик готов
- [TODO] (A/B) README: финальные скриншоты и видео-демо для сдачи
- [TODO] (A/B) Проверка запуска на «чистой» машине по инструкции
