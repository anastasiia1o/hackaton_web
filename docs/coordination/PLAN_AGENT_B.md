# PLAN — Поток B (UI & Viewer)

Ты отвечаешь за то, что видит геолог: экран Streamlit, overlay, вьюер больших
панорам, таблицы, экспортные кнопки, экспертный редактор. Логику не трогаешь —
берёшь готовый `AnalysisResult` из `src.pipeline.run_analysis(...)`.

## Что уже готово (фундамент)
- `app.py` — сквозной экран: загрузка → анализ → overlay+слои → метрики →
  классификация → экспорт CSV/JSON/PNG/PDF → заглушка экспертной коррекции
- `ui/viewer.py` — `make_overlay`, `colorize_mask`, `legend_items`
- `ui/components.py` — карточка класса, таблица метрик, легенда, след правил

## Как пользоваться результатом потока A (не пересчитывай сам!)
```python
from src.pipeline import run_analysis, load_mask
result = run_analysis(image_path)          # AnalysisResult
result.classification.ore_class            # строка класса руды
result.classification.reason               # готовый текст вывода
result.metrics.talc_fraction               # доли уже посчитаны (0..1)
mask = load_mask(result.ml.mask_path)      # 2D-маска кодов классов
```
Цвета/названия классов — `src.config.CLASS_COLORS/CLASS_NAMES`, не хардкодь.

## Твой backlog (по приоритету)

1. **Zoom/pan для больших изображений.** Сейчас `st.image` показывает превью.
   Подключить интерактивный вьюер (варианты: `streamlit-image-coordinates`,
   `pydeck`/deck.gl BitmapLayer, или tiled-подход через `openseadragon` в
   `components.html`). Начать с одного изображения, overlay поверх.

2. **Тайловая загрузка больших TIFF.** Не грузить 10000×10000 целиком —
   читать по плиткам (Pillow `Image.crop` / `tifffile`), показывать текущую
   область + minimap. Согласовать с потоком A формат тайлов через `HANDOFF.md`.

3. **Навигатор/minimap** — маленькая карта всей панорамы с рамкой видимой зоны.

4. **Многостраничность** (`pages/`): страница «Пакетная обработка» (запуск
   `batch_process` из UI + прогресс-бар) и «История/лог» (чтение
   `data/results/analysis_log.jsonl`).

5. **Слои и карта уверенности как переключаемый overlay** (сейчас confidence
   в expander — сделать полноценным слоем с прозрачностью).

6. **Экспертный редактор маски** — рисование области поверх изображения
   (`streamlit-drawable-canvas`), сохранение через `storage.save_correction`.

## Границы (не делай)
- Не считай проценты и не применяй геологические правила в `app.py` —
  это поток A (`metrics.py`/`classification.py`).
- Не меняй `schemas.py`. Нужно новое поле — попроси A через `HANDOFF.md`.

## Definition of Done для задачи потока B
- `streamlit run app.py` открывается, сценарий проходит без ошибок;
- `py_compile` без ошибок;
- PR в `main`, строка в `BOARD.md` переведена в DONE.
