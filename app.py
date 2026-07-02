"""
OreVision — локальный веб-интерфейс (Streamlit).

Запуск (Windows PowerShell, из корня репозитория, при активном .venv):
    streamlit run app.py
Затем открыть в браузере:  http://localhost:8501

Это ГЛАВНЫЙ экран сквозного сценария:
    загрузка изображения → анализ (ML) → overlay-маска + слои →
    метрики → rule-based классификация → экспорт (CSV/JSON/PDF).

Работает в MOCK-режиме без ML-сервиса (config.ML_MODE = "mock").
Когда ML-команда поднимет сервис — переключаемся на "real" одной настройкой.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image

from src import config, ml_client, reports, storage, gis_export
from src.contract import ContractError
from src.pipeline import run_analysis, load_mask
from ui import viewer, components

# --- Общая настройка страницы ----------------------------------------------
st.set_page_config(page_title="OreVision", page_icon="⛏️", layout="wide")
config.ensure_dirs()

# Панорамы бывают гигапиксельными (сотни Мп) — снимаем лимит PIL на размер,
# сами уменьшаем изображение при показе (viewer.load_display_image).
Image.MAX_IMAGE_PIXELS = None

# --- Боковая панель: режим ML, статус, сценарий (для демо) ------------------
with st.sidebar:
    st.title("⛏️ OreVision")
    st.caption("Локальный анализ полированных шлифов руды")

    ok, msg = ml_client.health_check()
    (st.success if ok else st.error)(msg)

    st.divider()
    st.subheader("Режим ML")
    st.write(f"Текущий режим: **{config.ML_MODE.upper()}**")
    st.caption(
        "Смените режим переменной окружения `OREVISION_ML_MODE=real`, "
        "чтобы подключить реальный ML-сервис на :8001."
    )

    scenario = None
    if config.ML_MODE == "mock":
        st.subheader("Демо-сценарий (mock)")
        scenario = st.selectbox(
            "Какой тип руды имитировать",
            options=["refractory", "ordinary", "talc", "review"],
            format_func=lambda s: {
                "refractory": "Труднообогатимая (тонкие)",
                "ordinary": "Рядовая (обычные)",
                "talc": "Оталькованная (тальк >10%)",
                "review": "Пограничный (проверка)",
            }[s],
        )

    st.divider()
    st.subheader("Слои маски")
    show_ordinary = st.checkbox("Обычные срастания (зелёный)", value=True)
    show_fine = st.checkbox("Тонкие срастания (красный)", value=True)
    show_talc = st.checkbox("Тальк (синий)", value=True)
    show_artifact = st.checkbox("Артефакты (серый)", value=False)
    opacity = st.slider("Прозрачность маски", 0.0, 1.0, 0.55, 0.05)

    st.divider()
    st.subheader("Слой уверенности")
    show_confidence = st.checkbox(
        "Показать карту уверенности", value=False,
        help="Тепловой слой: красный — модель не уверена, зелёный — уверена.",
    )
    conf_opacity = st.slider("Прозрачность слоя уверенности", 0.0, 1.0, 0.5, 0.05)

    st.divider()
    interactive = st.checkbox(
        "Интерактивный вьюер (zoom/pan + minimap)", value=True,
        help="Колесо — приблизить/отдалить, перетаскивание — панорама, "
             "мини-карта справа внизу — навигатор по всей панораме.",
    )

# --- Заголовок и легенда ----------------------------------------------------
st.header("Анализ изображения шлифа")
components.legend_bar()

uploaded = st.file_uploader(
    "Загрузите OM-изображение шлифа (TIFF / PNG / JPEG)",
    type=[e.lstrip(".") for e in config.SUPPORTED_FORMATS],
)

# Кнопка "демо без файла" — удобно показывать жюри без загрузки.
demo = st.button("Показать на демо-образце (без загрузки файла)")


def _resolve_input() -> Path | None:
    """Определить, какое изображение анализировать: загруженное или демо."""
    if uploaded is not None:
        path = storage.save_upload(uploaded.getvalue(), uploaded.name)
        return path
    if demo:
        # Создаём простой серый холст как "исходник" под mock-маску.
        config.ensure_dirs()
        demo_path = config.SAMPLES_DIR / "demo_slide.png"
        if not demo_path.exists():
            Image.new("RGB", (900, 700), (60, 60, 66)).save(demo_path)
        return demo_path
    return None


image_path = _resolve_input()

if image_path is None:
    st.info("Загрузите изображение или нажмите «Показать на демо-образце».")
    st.stop()

# --- Предупреждение о размере панорамы -------------------------------------
with Image.open(image_path) as im:
    w, h = im.size
if max(w, h) > config.MAX_DIMENSION_WARN:
    st.warning(
        f"Большое панорамное изображение ({w}×{h}). Для показа оно уменьшается "
        f"до {viewer.DISPLAY_MAX_DIM}px по большей стороне, но zoom/pan во вьюере "
        f"работают на стороне браузера. Полностайловая загрузка исходного "
        f"разрешения — в разработке (поток B)."
    )

# --- Запуск анализа ---------------------------------------------------------
with st.spinner("Анализируем изображение…"):
    params = {"scenario": scenario} if scenario else None
    try:
        result = run_analysis(str(image_path), params=params)
    except ContractError as e:
        st.error(f"Ответ ML-сервиса не соответствует контракту: {e}")
        st.caption(
            "Проверьте ML-сервис (см. docs/ML_INTEGRATION_GUIDE.md) или "
            "переключитесь в MOCK-режим."
        )
        st.stop()
    except Exception as e:  # noqa: BLE001
        st.error(f"Не удалось выполнить анализ: {e}")
        st.stop()

# --- Итоговая классификация -------------------------------------------------
components.classification_card(result)
components.rule_trace(result)

# --- Overlay + метрики бок о бок -------------------------------------------
col_img, col_metrics = st.columns([3, 2])

with col_img:
    st.subheader("Изображение с цветовой маской")
    show_classes = set()
    if show_ordinary:
        show_classes.add(config.CLASS_ORDINARY)
    if show_fine:
        show_classes.add(config.CLASS_FINE)
    if show_talc:
        show_classes.add(config.CLASS_TALC)
    if show_artifact:
        show_classes.add(config.CLASS_ARTIFACT)

    # Уменьшаем панораму на этапе декодирования — иначе гигапиксельный JPEG
    # съест память. make_overlay сам подгонит маску под этот размер.
    base = viewer.load_display_image(str(image_path))
    mask = load_mask(result.ml.mask_path)
    overlay = viewer.make_overlay(base, mask, show_classes=show_classes, opacity=opacity)

    # Переключаемый слой уверенности поверх маски (работает в обоих режимах показа).
    has_conf = bool(result.ml.confidence_map_path and Path(result.ml.confidence_map_path).exists())
    if show_confidence and has_conf:
        conf = viewer.load_confidence(result.ml.confidence_map_path)
        overlay = viewer.add_confidence_layer(overlay, conf, opacity=conf_opacity)

    if interactive:
        st.components.v1.html(
            viewer.interactive_viewer_html(overlay, height=660), height=680
        )
    else:
        st.image(overlay, use_container_width=True)

    if show_confidence and has_conf:
        chips = " ".join(
            f'<span style="display:inline-block;margin:2px 10px 2px 0;">'
            f'<span style="display:inline-block;width:12px;height:12px;'
            f'background:{hexcol};border-radius:2px;margin-right:5px;"></span>{name}</span>'
            for name, hexcol in viewer.confidence_legend()
        )
        st.caption("Слой уверенности:")
        st.markdown(chips, unsafe_allow_html=True)
    elif show_confidence and not has_conf:
        st.info("Карта уверенности недоступна для этого изображения.")

with col_metrics:
    st.subheader("Количественные метрики")
    components.metrics_table(result)
    if result.ml.warnings:
        for wmsg in result.ml.warnings:
            st.warning(wmsg)

# --- Экспорт ----------------------------------------------------------------
st.divider()
st.subheader("Экспорт результатов")

# Сохраняем overlay в PNG (для PDF и для скачивания).
overlay_png = storage.result_dir(str(image_path)) / "overlay.png"
overlay.convert("RGB").save(overlay_png)

exp_col1, exp_col2, exp_col3, exp_col4, exp_col5 = st.columns(5)

with exp_col1:
    st.download_button(
        "Скачать CSV",
        data=reports.csv_bytes(result),
        file_name=f"{image_path.stem}_metrics.csv",
        mime="text/csv",
    )
with exp_col2:
    import json
    st.download_button(
        "Скачать JSON",
        data=json.dumps(result.to_dict(), ensure_ascii=False, indent=2).encode("utf-8"),
        file_name=f"{image_path.stem}_result.json",
        mime="application/json",
    )
with exp_col3:
    with open(overlay_png, "rb") as f:
        st.download_button(
            "Скачать маску (PNG)",
            data=f.read(),
            file_name=f"{image_path.stem}_overlay.png",
            mime="image/png",
        )
with exp_col4:
    st.download_button(
        "Скачать GeoJSON",
        data=gis_export.geojson_bytes(result),
        file_name=f"{image_path.stem}_objects.geojson",
        mime="application/geo+json",
        help="Объекты (включения) как GeoJSON — для ГИС/QGIS.",
    )
with exp_col5:
    if st.button("Сформировать PDF-отчёт"):
        with st.spinner("Собираем PDF…"):
            pdf_path = reports.export_pdf(result, overlay_png=overlay_png)
        with open(pdf_path, "rb") as f:
            st.download_button(
                "Скачать PDF",
                data=f.read(),
                file_name=f"{image_path.stem}_report.pdf",
                mime="application/pdf",
            )

# --- Экспертная коррекция: выделить участок и указать правильный класс -------
st.divider()
with st.expander("Экспертная проверка: отметить и исправить участок (active learning)"):
    st.caption(
        "Выделите прямоугольную область, где модель ошиблась, и укажите "
        "правильный класс. Исправление сохраняется локально (для будущего "
        "дообучения ML). Координаты — в долях изображения, поэтому не зависят "
        "от масштаба превью."
    )
    ed_ctrl, ed_prev = st.columns([2, 3])
    with ed_ctrl:
        x_range = st.slider("Область по X (доля ширины)", 0.0, 1.0, (0.30, 0.60), 0.01)
        y_range = st.slider("Область по Y (доля высоты)", 0.0, 1.0, (0.30, 0.60), 0.01)
        corr_class = st.selectbox(
            "Правильный класс участка",
            options=[config.CLASS_ORDINARY, config.CLASS_FINE,
                     config.CLASS_TALC, config.CLASS_ARTIFACT],
            format_func=lambda c: config.CLASS_NAMES[c],
        )
        comment = st.text_input("Комментарий геолога")
        author = st.text_input("Автор", value="geolog")
        save_corr = st.button("Сохранить исправление")

    with ed_prev:
        region_color = config.CLASS_COLORS.get(corr_class, (255, 210, 60, 90))
        region_color = (region_color[0], region_color[1], region_color[2], 90)
        preview = viewer.preview_region(
            base, x_range[0], y_range[0], x_range[1], y_range[1], color=region_color
        )
        st.image(preview, use_container_width=True,
                 caption="Предпросмотр выделенной области")

    if save_corr:
        # Доли -> пиксели ОРИГИНАЛЬНОГО изображения (w, h — исходный размер).
        px0, py0 = int(min(x_range) * w), int(min(y_range) * h)
        px1, py1 = int(max(x_range) * w), int(max(y_range) * h)
        if px1 <= px0 or py1 <= py0:
            st.warning("Область пустая — сдвиньте ползунки, чтобы задать прямоугольник.")
        else:
            saved = storage.save_correction(str(image_path), {
                "correct_class": corr_class,
                "correct_class_name": config.CLASS_NAMES[corr_class],
                "bbox_px": [px0, py0, px1 - px0, py1 - py0],
                "region_fraction": {
                    "x0": round(min(x_range), 4), "y0": round(min(y_range), 4),
                    "x1": round(max(x_range), 4), "y1": round(max(y_range), 4),
                },
                "image_size": {"width": w, "height": h},
                "comment": comment,
                "author": author or "geolog",
            })
            st.success(f"Исправление сохранено: {saved.name}")

    existing = storage.list_corrections(str(image_path))
    if existing:
        st.caption(f"Сохранённых исправлений по этому изображению: {len(existing)}")
        st.dataframe(
            [{
                "класс": c.get("correct_class_name", c.get("correct_class")),
                "bbox_px": c.get("bbox_px"),
                "комментарий": c.get("comment", ""),
                "автор": c.get("author", ""),
            } for c in existing],
            use_container_width=True, hide_index=True,
        )
