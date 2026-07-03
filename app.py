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

import tempfile
from pathlib import Path
from typing import Any, Optional

import numpy as np
import streamlit as st
from PIL import Image

from src import config, dataset_export, ml_client, reports, storage, gis_export
from src.contract import ContractError
from src.pipeline import run_analysis, load_mask
from ui import viewer, components


@st.cache_data(show_spinner=False)
def _cached_run_analysis(
    image_path: str, params: Optional[dict[str, Any]], mode: Optional[str], _file_sig: str,
):
    """
    Кэш вокруг run_analysis: без него ЛЮБОЙ клик где угодно на странице
    (включая независимые кнопки скачивания) перезапускал бы весь анализ заново
    при каждом rerun — на реальном ML это могло занимать минуты и создавало
    впечатление, что кнопки "зависают"/работают только по очереди. Кэш-ключ
    включает params/mode и подпись файла (размер+mtime), поэтому смена
    сценария или загрузка нового файла с тем же именем всё равно пересчитает.
    """
    return run_analysis(image_path, params=params, mode=mode)

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
# ВАЖНО: st.button() возвращает True только на ОДИН rerun сразу после клика.
# Раньше от этого зависела вся страница результатов — стоило нажать любую
# другую кнопку (например, "Скачать CSV"), новый rerun видел demo=False,
# uploaded=None, и вся страница с результатами и остальными кнопками экспорта
# пропадала (нужно было заново нажимать демо-кнопку). Поэтому активацию демо
# сохраняем в session_state — один раз включили, и она остаётся до тех пор,
# пока не загрузят настоящий файл.
if st.button("Показать на демо-образце (без загрузки файла)"):
    st.session_state["demo_active"] = True


def _resolve_input() -> Path | None:
    """Определить, какое изображение анализировать: загруженное или демо."""
    if uploaded is not None:
        st.session_state["demo_active"] = False
        path = storage.save_upload(uploaded.getvalue(), uploaded.name)
        return path
    if st.session_state.get("demo_active"):
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

# --- Запуск анализа (кэшируется — см. _cached_run_analysis) -----------------
params = {"scenario": scenario} if scenario else None
_stat = Path(image_path).stat()
_file_sig = f"{_stat.st_size}:{_stat.st_mtime_ns}"
with st.spinner("Анализируем изображение…"):
    try:
        result = _cached_run_analysis(str(image_path), params, None, _file_sig)
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

# --- Инспектор участка (высокое разрешение, ленивый декод по кнопке) ---------
st.divider()
with st.expander("🔬 Инспектор участка (высокое разрешение)"):
    st.caption(
        "Обзор ниже уменьшен для скорости. Обведите область мышью замкнутой "
        "линией (не обязательно прямоугольником) и посмотрите её в максимально "
        "возможном разрешении. Тяжёлый исходник декодируется только по кнопке "
        "(не при каждом действии). Для гигапиксельных панорам детализация "
        "ограничена лимитом памяти."
    )
    insp_lasso = viewer.lasso_picker(
        base, key="insp_lasso",
        color="rgba(255, 210, 60, .28)", border_color="#ffcf33",
    )
    do_inspect = st.button("Показать участок в высоком разрешении")

    if do_inspect:
        if insp_lasso is None:
            st.warning("Сначала обведите область мышью замкнутой линией на изображении выше.")
        else:
            ins_x0, ins_y0, ins_x1, ins_y1 = insp_lasso["bbox"]
            with st.spinner("Готовим участок в высоком разрешении…"):
                crop, meta = viewer.crop_region_highres(
                    str(image_path), ins_x0, ins_y0, ins_x1, ins_y1
                )
            rx = meta["region_px_orig"]
            st.image(
                crop, use_container_width=True,
                caption=(f"Участок оригинала {rx[2]}×{rx[3]} px @ ({rx[0]},{rx[1]}) → "
                         f"показано {meta['shown_size'][0]}×{meta['shown_size'][1]} "
                         f"(область — bounding box обведённой линии)"),
            )
            if meta["capped"]:
                st.info(
                    f"Исходник очень большой ({meta['orig_size'][0]}×{meta['orig_size'][1]}) — "
                    f"для инспекции понижен (масштаб {meta['native_scale']}), чтобы уложиться "
                    f"в память. Участок всё равно детальнее обзора."
                )
            else:
                st.success("Участок показан в нативном разрешении.")

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
    # Формируем PDF сразу же (как остальные кнопки), а не по отдельному клику:
    # раньше здесь была цепочка "нажми Сформировать -> нажми Скачать", и на
    # каждый клик ЛЮБОЙ кнопки на странице анализ пересчитывался заново, из-за
    # чего казалось, что следующая кнопка не срабатывает, пока не нажмёшь
    # предыдущую. Теперь результат кэширован (_cached_run_analysis) и PDF
    # готов сразу — одна кнопка, как у CSV/JSON/PNG/GeoJSON.
    pdf_path = reports.export_pdf(result, overlay_png=overlay_png)
    with open(pdf_path, "rb") as f:
        st.download_button(
            "Скачать PDF",
            data=f.read(),
            file_name=f"{image_path.stem}_report.pdf",
            mime="application/pdf",
        )

# --- Второй вариант сохранения: как датасет (структура папок как в S2_v2) ---
st.markdown("**Вариант сохранения «как датасет» (imgs / masks / masks_colored / masks_human)**")
st.caption(
    "Тот же результат, но в виде папок imgs/masks/masks_colored/masks_human — "
    "структура приблизительно как в примере датасета S2_v2."
)
mask_for_export = mask
if mask_for_export.shape[:2] != (base.size[1], base.size[0]):
    mask_for_export = np.array(
        Image.fromarray(mask_for_export, mode="L").resize(base.size, Image.NEAREST), dtype=np.uint8
    )
with tempfile.TemporaryDirectory() as _tmp:
    _bundle_dir = Path(_tmp) / "bundle"
    dataset_export.export_s2_bundle(
        _bundle_dir,
        items=[{"name": image_path.stem, "image": base, "mask": mask_for_export}],
        class_colors=config.CLASS_COLORS, class_names=config.CLASS_NAMES,
    )
    _s2_zip_bytes = dataset_export.zip_directory(_bundle_dir)
st.download_button(
    "📦 Скачать в формате S2_v2 (ZIP)",
    data=_s2_zip_bytes,
    file_name=f"{image_path.stem}_s2v2.zip",
    mime="application/zip",
)

# --- Экспертная коррекция: выделить участок и указать правильный класс -------
st.divider()
with st.expander("Экспертная проверка: отметить и исправить участок (active learning)"):
    st.caption(
        "Обведите область мышью замкнутой линией прямо на изображении (не "
        "обязательно прямоугольником) и укажите правильный класс. Исправление "
        "сохраняется локально (для будущего дообучения ML)."
    )
    ed_ctrl, ed_prev = st.columns([2, 3])
    with ed_ctrl:
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
        r, g, b, _a = config.CLASS_COLORS.get(corr_class, (255, 210, 60, 90))
        corr_lasso = viewer.lasso_picker(
            base, key="corr_lasso",
            color=f"rgba({r}, {g}, {b}, .35)", border_color=f"#{r:02x}{g:02x}{b:02x}",
        )

    if save_corr:
        if corr_lasso is None:
            st.warning("Сначала обведите область мышью замкнутой линией на изображении.")
        else:
            corr_x0, corr_y0, corr_x1, corr_y1 = corr_lasso["bbox"]
            # Доли -> пиксели ОРИГИНАЛЬНОГО изображения (w, h — исходный размер).
            px0, py0 = int(corr_x0 * w), int(corr_y0 * h)
            px1, py1 = int(corr_x1 * w), int(corr_y1 * h)
            polygon_px = [[round(x * w), round(y * h)] for x, y in corr_lasso["points"]]
            saved = storage.save_correction(str(image_path), {
                "correct_class": corr_class,
                "correct_class_name": config.CLASS_NAMES[corr_class],
                "bbox_px": [px0, py0, px1 - px0, py1 - py0],
                "polygon_px": polygon_px,
                "region_fraction": {
                    "x0": round(corr_x0, 4), "y0": round(corr_y0, 4),
                    "x1": round(corr_x1, 4), "y1": round(corr_y1, 4),
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

        st.markdown("**Экспорт исправлений «как датасет» (S2_v2)**")
        if st.button("📦 Собрать экспорт исправлений (ZIP, формат S2_v2)"):
            corr_items = []
            for i, c in enumerate(existing):
                rf = c.get("region_fraction")
                if not rf:
                    continue
                crop, _meta = viewer.crop_region_highres(
                    str(image_path), rf["x0"], rf["y0"], rf["x1"], rf["y1"],
                )
                fill_cls = int(c.get("correct_class", 0))
                mask_arr = np.full((crop.height, crop.width), fill_cls, dtype=np.uint8)
                corr_items.append({
                    "name": f"{image_path.stem}_corr_{i + 1:03d}", "image": crop, "mask": mask_arr,
                })
            if not corr_items:
                st.warning("Нет исправлений с сохранёнными координатами для экспорта.")
                st.session_state.pop("corr_s2_zip", None)
            else:
                with tempfile.TemporaryDirectory() as _tmp:
                    _bundle_dir = Path(_tmp) / "bundle"
                    dataset_export.export_s2_bundle(
                        _bundle_dir, corr_items, config.CLASS_COLORS, config.CLASS_NAMES,
                    )
                    st.session_state["corr_s2_zip"] = dataset_export.zip_directory(_bundle_dir)

        _corr_zip = st.session_state.get("corr_s2_zip")
        if _corr_zip:
            st.download_button(
                "⬇️ Скачать исправления (ZIP, формат S2_v2)",
                data=_corr_zip, file_name=f"{image_path.stem}_corrections_s2v2.zip",
                mime="application/zip",
            )
