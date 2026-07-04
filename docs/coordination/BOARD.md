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
- [DONE] (A) Тайловый расчёт метрик для панорам >10000px (агрегация по тайлам) — stream-a/tiled-metrics
- [DONE] (A) Обогатить mock-генератор: регулируемый % талька, шум, неравномерное освещение — stream-a/mock-generator-enrich
- [DONE] (A) Второй вариант экспорта «как S2_v2» (imgs/masks/masks_colored/
  masks_human, `src/dataset_export.py`, 13 тестов) на главной/batch/AL-
  странице + фикс зависающих кнопок экспорта на главной странице (sticky
  demo-режим, кэш `run_analysis`, PDF в одну кнопку, batch/AL-результаты
  вынесены в session_state) — stream-a/annotation-suite

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
- [DONE] (A, по прямой просьбе — см. HANDOFF) Очередь импорта (путь/папка/
  файлы), полноценный инспектор панорамы (pan/zoom/fit/1:1/ROI/несколько
  сохранённых участков), редактор разметки active learning (кисть/полигон/
  ластик/undo-redo/присвоение класса/экспорт), безопасная очистка логов —
  stream-a/annotation-suite; новые src/-модули (dataset_storage,
  annotation_config, batch_import, event_log) + 26 тестов, зелёные; нужна
  ручная проверка canvas-инструментов в браузере (нет браузера у агента)
- [DONE] (A, по прямой просьбе — см. HANDOFF) Фидбэк-правки: лассо
  (произвольная замкнутая линия) вместо прямоугольника на главной странице и
  в «Разметке эксперта»; страница разметки упрощена (без кисти/полигон-клика/
  undo-redo, только выделение+подпись); «Инспектор панорамы» удалён (разметка
  работает на кадре напрямую); «История и лог»/«Логи» сдвинуты вниз в меню;
  «Ручной путь к папке» убран из «Пакетной обработки» — stream-a/annotation-
  suite; +2 теста (get_or_create_whole_image_roi); нужна ручная проверка
  обводки мышью в браузере (нет браузера у агента)
- [DONE] (A, по прямой просьбе — см. HANDOFF) «Инспектор участка» на главной
  странице объединён с быстрой коррекцией (одно лассо вместо двух, каждая
  кнопка — одно действие); таблица «было → стало» на главной и в «Разметке
  эксперта» (`majority_class_in_polygon` в `src/dataset_export.py`, 4 теста)
  — stream-a/inspector-quick-correction
- [DONE] (A, по прямой просьбе — см. HANDOFF) «Разметка эксперта»: иконка
  проводника рядом с полем пути, множественная загрузка файлов, удаление
  шлифа/панорамы из датасета (`dataset_storage.remove_image()`, не трогает
  исходники и уже сохранённую разметку, 4 теста) — stream-a/al-dataset-management
- [DONE] (A, по прямой просьбе — см. HANDOFF) «Разметка эксперта»: загрузка
  без отдельной кнопки-подтверждения, удаление крестиком рядом с выбором
  шлифа, ПКМ отменяет лассо-выделение (общий компонент), таблица и экспорт
  переименованы по-человечески, читаемые имена экспорта — stream-a/al-page-ux-polish
- [DONE] (A, по прямой просьбе — см. HANDOFF) Переход на patch-classification
  (docs/PATCH_AL_REDESIGN.md): контракт v2 (блочная маска + `patch_grid`),
  `schemas.PatchGrid`, валидация `patch_grid`, mock отдаёт patch_grid, единое
  пространство id разметки (= коды контракта), новые `src/quantizer.py` и
  `src/active_query.py`, ImageFolder-экспорт патчей
  (`dataset_export.write_imagefolder` + `dataset_storage.export_active_learning_patch`),
  третий вариант экспорта на странице разметки. 88 тестов зелёных (+21) —
  stream-a/patch-al-redesign

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

- [DONE] (A) Подключение РЕАЛЬНОЙ модели: `ml_service/` — HTTP-обёртка
  (Flask `/health`+`/analyze`) поверх `grade_unfreeze_best.pth`
  (se_resnext50_32x4d/MicroNet). Ответ строго по contract v2 (блочная mask +
  patch_grid), классы 3-сорта→коды контракта (talc→3, ordinary→1, fine→2).
  Сайт не меняется — переключение `OREVISION_ML_MODE=real`. Форма ответа
  проверена без torch (test_contract_shape.py: валидатор + пайплайн зелёные) —
  stream-a/real-model-service
