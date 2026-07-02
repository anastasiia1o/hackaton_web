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
streamlit run app.py
```

Откроется браузер на `http://localhost:8501`. Нажмите
**«Показать на демо-образце»** — увидите весь сценарий без загрузки файлов.

По умолчанию включён **MOCK-режим**: ML имитируется локально, реальный сервис
не нужен. Это позволяет показать продукт до готовности нейросети.

### Пакетная обработка серии изображений
```powershell
python batch_process.py data\uploads --scenario refractory
```

### Тесты логики
```powershell
pytest -q
```

---

## 2. Переключение на реальный ML

Когда команда `orevision-ml` поднимет сервис на `http://localhost:8001`:

```powershell
$env:OREVISION_ML_MODE = "real"      # включить реальный режим
streamlit run app.py
```

Контракт запроса/ответа описан в [`API_CONTRACT.md`](API_CONTRACT.md).
Он одинаков для mock и real — поэтому больше ничего менять не нужно.

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
demo обычным `streamlit run app.py` из раздела 1. Он не хуже для жюри.

---

## 4. Структура репозитория

```
orevision-app/
├── app.py                  # Streamlit: главный экран (поток B)
├── batch_process.py        # пакетная обработка серии изображений
├── API_CONTRACT.md         # договор с ML-сервисом
├── AGENTS.md               # как работают два агента в одном репо
├── requirements.txt
├── Dockerfile / docker-compose.yml
├── src/                    # ЛОГИКА (поток A)
│   ├── config.py           # пороги, цвета, пути
│   ├── schemas.py          # SEAM — общие структуры данных
│   ├── ml_client.py        # HTTP-клиент к ML (mock/real)
│   ├── metrics.py          # площади → проценты
│   ├── classification.py   # rule-based геологика
│   ├── pipeline.py         # склейка ML→метрики→классификация
│   ├── reports.py          # экспорт CSV/JSON/PDF
│   └── storage.py          # локальное хранение
├── ui/                     # ИНТЕРФЕЙС (поток B)
│   ├── viewer.py           # overlay масок, слои
│   └── components.py        # карточки, таблицы, легенда
├── mock_ml/                # локальная имитация ML по контракту
│   └── generator.py
├── tests/                  # тесты логики
├── docs/
│   ├── AGENT_EXECUTION_PLAN.md
│   └── coordination/       # BOARD / HANDOFF / планы потоков
└── data/                   # ЛОКАЛЬНЫЕ данные (в Git НЕ попадают)
    ├── uploads/  results/  samples/
```

---

## 5. GitHub и командная работа (кратко)

Полные правила — в [`AGENTS.md`](AGENTS.md). Главное:

**Что храним в Git:** код, документацию, конфиги, тесты.
**Что НЕ коммитим:** конфиденциальные изображения (TIFF/PNG), результаты,
веса моделей, `.venv`, `.env`. За это отвечает `.gitignore`.

**Private-репозиторий:** на GitHub → New repository → Private → пригласить
второго участника (Settings → Collaborators).

**Ветки:** `main` — стабильная, только через Pull Request.
`stream-a/<задача>` — ветки логики, `stream-b/<задача>` — ветки UI.
Прямо в `main` не коммитим.

**Как двое не мешают друг другу:** поток A правит `src/`, поток B — `app.py`/`ui/`.
Общая точка — `src/schemas.py` (менять только через запись в `HANDOFF.md`).
Задачи и статусы — в `docs/coordination/BOARD.md`, решения — в `HANDOFF.md`.

**Большие/секретные файлы:** держим в локальной `data/` (вне Git). Для обмена
образцами — облачный диск или Git LFS, но не обычный commit.

---

## 6. Частые проблемы

- *«streamlit не найден»* — не активировано venv (`\.venv\Scripts\activate`)
  или не установлены зависимости (`pip install -r requirements.txt`).
- *PowerShell не даёт активировать venv* — выполните один раз:
  `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`.
- *ML-сервис недоступен в real-режиме* — сайт покажет красный индикатор;
  вернитесь в mock (`$env:OREVISION_ML_MODE="mock"`).
- *Большое изображение тормозит* — в MVP показывается превью; полноценный
  tiled-вьюер в разработке (поток B).
