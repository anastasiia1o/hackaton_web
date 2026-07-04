# OreVision — локальный анализ полированных шлифов руды

Локальное веб-приложение для геологической лаборатории. Геолог загружает
OM-изображение шлифа и получает цветовую маску фаз, количественные метрики и
итоговую классификацию руды по прозрачным геологическим правилам. Всё работает
**локально**, данные не уходят в интернет.

```
Браузер ──▶ OreVision (Streamlit, localhost:8501) ──▶ ML-клиент ──▶ ML-сервис (FastAPI, localhost:8001)
                     │                                                     ▲
                     └── сам считает проценты и класс руды (rule-based) ───┘  (mask/objects/confidence)
```

> **Кто что считает.** ML отдаёт только «сырые факты о пикселях» (маска,
> объекты, площади, уверенность). Проценты и класс руды считает **сайт** —
> открытым кодом, который может прочитать геолог (`src/classification.py`).

---

## 1. Быстрый старт (Windows, режим разработки)

Нужен установленный **Python 3.11+** и **VS Code** (по желанию).

```powershell
# 1. Клонировать репозиторий и войти в папку
git clone <URL-приватного-репозитория> orevision-app
cd orevision-app

# 2. Создать виртуальное окружение (изолированные библиотеки проекта)
python -m venv .venv
.venv\Scripts\activate            # активация venv в PowerShell

# 3. Установить зависимости
pip install -r requirements.txt

# 4. Запустить сайт
streamlit run OreVision.py
```

Откроется браузер на `http://localhost:8501`. Загрузите изображение шлифа —
его проанализирует **встроенная модель** классификации сортов руды
(`ml_service/grade_unfreeze_best.pth`, вшита в репозиторий).

MOCK-режима больше нет: по умолчанию модель считает **в процессе сайта**
(`OREVISION_ML_MODE=local`), отдельный сервер не нужен. Нужен только torch-стек
из `requirements.txt` (тяжёлый; можно поставить CPU-сборку torch).

### Тесты логики
```powershell
pytest -q
```

---

## 2. Вынести инференс в отдельный сервис (опционально)

По умолчанию всё считается локально. Если инференс хочется вынести на отдельную
(например, GPU) машину — поднимите сервис из этого же репозитория:

```powershell
python ml_service/server.py               # слушает :8001
$env:OREVISION_ML_MODE = "real"           # сайт ходит в сервис по HTTP
streamlit run OreVision.py
```

Контракт запроса/ответа описан в [`API_CONTRACT.md`](API_CONTRACT.md).
Он одинаков для local и real — поэтому больше ничего менять не нужно.

---

## 3. Демо-режим через Docker (финальная упаковка)

Не обязателен для баллов, но удобно показать «одной командой» на любой машине.
Нужен **Docker Desktop**.

```powershell
docker compose up --build
```

Сайт будет на `http://localhost:8501`. Остановить — `Ctrl+C`, затем
`docker compose down`.

**Если Docker не работает или не успеваете** — это не блокер. Показывайте
demo обычным `streamlit run OreVision.py` из раздела 1. Он не хуже для жюри.

---

## 4. Структура репозитория

```
orevision-app/
├── OreVision.py            # Streamlit: главный экран (анализ → экспорт → active learning)
├── batch_process.py        # пакетная обработка серии изображений (CLI)
├── pages/                  # многостраничность Streamlit
│   ├── 1_Пакетная_обработка.py     # очередь импорта (папка/файлы) + batch
│   ├── 2_Разметка_эксперта.py     # ручной аннотатор (зум+лассо) + экспорт патчей для дообучения
│   ├── 3_История_образцов.py       # журнал ML-анализов
│   └── 4_Логи.py                   # события импорта/batch/ошибок/разметки + очистка
├── ml_service/             # ВСТРОЁННАЯ модель классификации сортов руды
│   ├── grade_unfreeze_best.pth     # веса (99.7 МБ, вшиты в репозиторий)
│   ├── model.py            # загрузка весов + батч-инференс тайлов (torch лениво)
│   ├── infer.py            # тайлинг панорамы → JSON по API_CONTRACT.md (contract v2)
│   ├── server.py           # опциональный Flask-сервис (/health, /analyze) для режима real
│   └── reference/          # исходные скрипты модели от ML-команды (для аудита)
├── API_CONTRACT.md         # формат ответа модели (JSON)
├── requirements.txt        # ядро сайта + torch-стек встроенной модели
├── Dockerfile / docker-compose.yml
├── .streamlit/config.toml  # телеметрия выключена; maxUploadSize=5 ГБ
├── src/                    # ЛОГИКА
│   ├── config.py           # пороги, цвета, пути, ML_MODE (local|real)
│   ├── schemas.py          # общие структуры данных (MLResponse/Metrics/…)
│   ├── ml_client.py        # local (инференс в процессе) | real (HTTP на :8001)
│   ├── contract.py         # валидатор ответа модели по API_CONTRACT.md
│   ├── metrics.py          # площади → проценты
│   ├── classification.py   # rule-based геологика (класс руды)
│   ├── pipeline.py         # склейка ML→метрики→классификация
│   ├── quantizer.py        # область эксперта → квадратные патчи train-размера
│   ├── reports.py          # экспорт CSV/JSON/PDF + run_manifest.json
│   ├── gis_export.py       # экспорт GeoJSON/Shapefile
│   ├── storage.py          # локальное хранение, экспертные коррекции
│   ├── batch_import.py     # очередь импорта: скан папки/проба файла
│   ├── dataset_storage.py  # датасеты/ROI/маски/ревизии/экспорт для active learning
│   ├── annotation_config.py# классы разметки (configs/annotation_classes.json)
│   └── event_log.py        # data/logs/*.jsonl + безопасная очистка
├── ui/                     # ИНТЕРФЕЙС
│   ├── viewer.py           # overlay масок, zoom/pan, lasso_picker, annotator
│   ├── components.py       # карточки, таблицы, легенда
│   ├── file_pickers.py + folder_picker_frontend/    # выбор папки/файлов (проводник)
│   ├── annotator_frontend/ # аннотатор с зумом (прямоуг. зум + лассо + оригинал/оверлей)
│   └── lasso_picker_frontend/    # свободное лассо (используется в экспорте коррекций)
├── mock_ml/generator.py    # ТЕСТ-ФИКСТУРА (контракт-валидный ответ без torch); не рантайм
├── tests/                  # тесты логики (pytest, 91)
├── docs/                   # ML_INTEGRATION_GUIDE + coordination (BOARD/HANDOFF)
└── data/                   # ЛОКАЛЬНЫЕ данные (в Git не попадают, кроме .gitkeep)
    ├── uploads/  results/  samples/  datasets/  logs/
```

---

## 5. GitHub (кратко)

**Что храним в Git:** код, документацию, конфиги, тесты И **веса встроенной
модели** (`ml_service/grade_unfreeze_best.pth`, 99.7 МБ — вшиты, чтобы репозиторий
был самодостаточным; это единственный крупный бинарник в исключении `.gitignore`).

**Что НЕ коммитим:** конфиденциальные изображения (TIFF/PNG), результаты
инференса, `data/`, `.venv`, `.env`, прочие веса (`*.pt/*.pth/*.onnx`).

**Ветки:** `main` — стабильная, только через Pull Request; прямо в `main` не
коммитим. Рабочие ветки — `stream-*/<задача>`.

> Примечание: репозиторий раньше вёлся двумя агентами (потоки A/B, см.
> `AGENTS.md`); сейчас это единый самодостаточный проект с вшитой моделью, и
> разделение на потоки неактуально.

---

## 6a. Хранилище датасетов и разметки (active learning)

«Пакетная обработка» (очередь импорта) и «Разметка эксперта» (лассо-выделение
+ подпись участков) используют общий формат хранения на диске
(`src/dataset_storage.py`). «Разметка эксперта» размечает показанный кадр
целиком одним ROI (`roi.json` с `"kind": "whole_image"`, x=0,y=0 на весь
показанный — уменьшенный для больших панорам, как и везде в приложении —
кадр), без отдельного шага вырезания прямоугольника:

```
data/datasets/<dataset_id>/
  images/<image_id>.<ext>            — управляемая КОПИЯ исходника (picker-источники;
                                        для "путь к папке" копии нет — храним ссылку,
                                        исходный файл никогда не изменяется)
  manifest.jsonl                     — реестр изображений (формат/размер/разрешение/
                                        источник/валидность)
  annotations/<image_id>/
    image_meta.json
    regions/<region_id>/
      roi.json                — x, y, width, height ОТНОСИТЕЛЬНО исходной панорамы
                                 + coordinate_system + created_at + статус + ревизия
      roi_image.png            — вырезанный участок (копия пикселей, lossless)
      semantic_mask.png        — 8-bit ОДНОКАНАЛЬНАЯ PNG, значение пикселя = id класса;
                                  размеры точно равны roi_image.png
      annotation_state.json    — status, revision, updated_at, author, class_pixel_counts
      shapes.geojson           — подписанные лассо-контуры (класс + свободная подпись
                                  в properties каждой фигуры)
      revisions/rev_NNNN_.../   — снимок ПРЕДЫДУЩЕЙ маски+состояния перед КАЖДЫМ
                                   явным сохранением (принятая разметка не теряется)
  exports/active_learning/<export_id>/
    images/<sample_id>.png  masks/<sample_id>.png
    manifest.csv  manifest.jsonl  classes.json

data/logs/
  app_events.jsonl     — импорт, ошибки, сохранения разметки, экспорт
  batch_events.jsonl   — старт/прогресс/завершение пакетной обработки
```

Классы разметки — **не хардкод**, а `configs/annotation_classes.json`
(id, машинное имя, русское название, цвет). По умолчанию: тальк (синий),
обычные срастания (зелёный), тонкие срастания (красный), неопределённая
область (жёлто-оранжевый), неразмеченная область (прозрачный).

Статусы разметки: `draft` → `reviewed` → `accepted_for_training` /
`needs_expert_review`. Экспорт для дообучения (кнопка «📦 Экспортировать
подтверждённые разметки») берёт **только** `accepted_for_training`.

Ничего из `data/` (включая `data/datasets` и `data/logs`) в Git не попадает —
см. `.gitignore`.

### Второй вариант сохранения: «как S2_v2»

По отдельному запросу добавлен ещё один вариант сохранения результатов —
структура папок приблизительно как в примере стороннего датасета `S2_v2`
(лежит только локально в `S2_v2/`, в Git не коммитится — см. `.gitignore`):

```
imgs/<name>.jpg            — исходное изображение
masks/<name>.png           — маска классов, R=G=B=id (как masks/*.png в S2_v2)
masks_colored/<name>.png   — непрозрачная цветная маска (для быстрого визуального контроля)
masks_human/<name>.jpg     — триптих source|overlay|annotation + легенда (проверка человеком)
```

Реализовано в `src/dataset_export.py` (`export_s2_bundle`, `zip_directory`,
принимает элементы ленивым генератором — не держит все изображения батча в
памяти одновременно) и подключено как дополнительная кнопка «📦 …(формат
S2_v2)» в трёх местах:
- главная страница — экспорт текущего результата и экспертных исправлений;
- «Пакетная обработка» — экспорт всех обработанных изображений одним zip;
- «Разметка эксперта» — второй вариант экспорта active learning датасета
  (`dataset_storage.export_active_learning_s2_style`, отдельная ветка
  `exports/active_learning_s2v2/`, не пересекается с классическим форматом).

---

## 6. Частые проблемы

- *«streamlit не найден»* — не активировано venv (`\.venv\Scripts\activate`)
  или не установлены зависимости (`pip install -r requirements.txt`).
- *PowerShell не даёт активировать venv* — выполните один раз:
  `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`.
- *Красный индикатор модели* — в local-режиме не установлен torch
  (`pip install -r requirements.txt`) или нет файла весов
  `ml_service/grade_unfreeze_best.pth`; в real-режиме не поднят
  `python ml_service/server.py`.
- *Большая панорама (проверено до ~570 Мп)* — сайт грузит уменьшенное превью
  (≤2600px по стороне) с zoom/pan и minimap; детальный просмотр участка в
  исходном разрешении — кнопка «🔬 Инспектор участка». Расчёт метрик по маске
  для панорам >10000px идёт по тайлам, не грузя всё в память.
