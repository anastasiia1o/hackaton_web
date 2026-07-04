"""
OreVision — локальный веб-интерфейс (Streamlit).

Запуск (из корня репозитория, при активном окружении):
    streamlit run OreVision.py
Затем открыть в браузере:  http://localhost:8501

Это ГЛАВНЫЙ экран сквозного сценария:
    загрузка изображения → анализ (ML) → overlay-маска + слои → метрики →
    rule-based классификация → ЭКСПОРТ ДАННЫХ → АКТИВНОЕ ОБУЧЕНИЕ (правки).

Модель ВШИТА в приложение (ml_service/grade_unfreeze_best.pth) и по умолчанию
считает В ПРОЦЕССЕ сайта (config.ML_MODE = "local") — MOCK-режима больше нет.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any, Optional

import numpy as np
import streamlit as st
from PIL import Image

from src import active_learning as al
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


@st.cache_data(show_spinner=False)
def _cached_talc_overlay(
    image_path: str, _file_sig: str, seg_max_side: int, alpha: float, palette: str | None,
):
    """Кэш косметической талько-подсветки (палитровая сегментация + density-зоны).

    Чистый CV (talc_cosmetic), НИКАК не влияет на предсказание/метрики/класс руды —
    это опциональный визуальный слой поверх снимка. Считается на разрешении показа
    (viewer.load_display_image), поэтому быстро и без риска OOM на гигапикселях.
    Возвращает picklable-словарь (overlay/region — np.ndarray, а не dataclass).
    """
    from talc_cosmetic import compute_talc_overlay

    disp = viewer.load_display_image(image_path).convert("RGB")
    rgb = np.asarray(disp)
    res = compute_talc_overlay(
        rgb, seg_max_side=seg_max_side, alpha=alpha,
        palette=(palette or None),
    )
    return {
        "overlay": res.overlay,
        "region": res.region,
        "palette_name": res.palette_name,
        "talc_raw_pct": res.talc_raw_pct,
        "region_pct": res.region_pct,
    }

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
    st.subheader("Модель")
    st.write(f"Режим: **{config.ML_MODE.upper()}**")
    st.caption(
        "Классификатор сортов руды ВШИТ в приложение "
        "(grade_unfreeze_best.pth) и считает локально. "
        "`OREVISION_ML_MODE=real` — вынести инференс в отдельный сервис "
        "ml_service/ на :8001 (напр. на GPU-машину)."
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


def _resolve_input() -> Path | None:
    """Изображение для анализа — загруженный файл (демо-режима больше нет)."""
    if uploaded is not None:
        return storage.save_upload(uploaded.getvalue(), uploaded.name)
    return None


image_path = _resolve_input()

if image_path is None:
    st.info("Загрузите изображение шлифа/панораму для анализа встроенной моделью.")
    st.stop()

# Сменили изображение → сбрасываем состояние активного обучения предыдущего
# снимка (иначе «стало»/дообученный чекпоинт показались бы для чужой картинки).
if st.session_state.get("al_image_stem") != image_path.stem:
    for _k in ("al_after", "al_ckpt", "al_version", "corr_s2_zip"):
        st.session_state.pop(_k, None)
    st.session_state["al_image_stem"] = image_path.stem

# Размер оригинала нужен дальше (координаты исправлений считаем в его пикселях).
# Ограничения на размер фотографии НЕТ — гигапиксельные панорамы поддерживаются
# (для показа кадр уменьшается во вьювере, разметка идёт в долях 0..1).
with Image.open(image_path) as im:
    w, h = im.size

# --- Запуск анализа (кэшируется — см. _cached_run_analysis) -----------------
_stat = Path(image_path).stat()
_file_sig = f"{_stat.st_size}:{_stat.st_mtime_ns}"
with st.spinner("Анализируем изображение встроенной моделью…"):
    try:
        base_result = _cached_run_analysis(str(image_path), None, None, _file_sig)
    except ContractError as e:
        st.error(f"Ответ модели не соответствует контракту: {e}")
        st.caption("См. docs/ML_INTEGRATION_GUIDE.md.")
        st.stop()
    except Exception as e:  # noqa: BLE001
        st.error(f"Не удалось выполнить анализ: {e}")
        st.stop()

# Если в этой сессии модель ДООБУЧЕНА на исправлениях ЭТОГО снимка — вся страница
# (первая фотка, карточка класса, метрики, экспорт) показывает результат
# ДООБУЧЕННОЙ модели; базовый результат храним для панели «до».
_al_after = st.session_state.get("al_after")
al_active = bool(_al_after) and st.session_state.get("al_image_stem") == image_path.stem
result = _al_after["result"] if al_active else base_result

# --- Итоговая классификация -------------------------------------------------
if al_active:
    st.info(
        f"🔁 Ниже — результаты **ДООБУЧЕННОЙ** модели (версия "
        f"v{st.session_state.get('al_version')}): первая фотка, класс руды и все "
        "проценты пересчитаны. Сравнение «до/стало» и сброс — в блоке "
        "«Активное обучение» ниже."
    )
components.classification_card(result)

# --- Overlay + метрики бок о бок -------------------------------------------
col_img, col_metrics = st.columns([3, 2])

# Метрики рисуем ПЕРВЫМИ (хотя колонка правая) — здесь же, под блоком процентов,
# живёт галочка косметической подсветки: её значение нужно ниже, при отрисовке
# оверлея в левой колонке. Порядок колонок в коде на вёрстку не влияет.
with col_metrics:
    st.subheader("Количественные метрики")
    components.metrics_table(result)
    if result.ml.warnings:
        for wmsg in result.ml.warnings:
            st.warning(wmsg)

    st.divider()
    st.markdown("**🔴 Косметическая подсветка талька**")
    show_talc_cosmetic = st.checkbox(
        "Подсветить область оталькования", value=False,
        help="Опциональный ВИЗУАЛЬНЫЙ слой поверх снимка: палитровая сегментация "
             "(панорамная/жёлтая — авто) + density-сборка сплошной «области "
             "оталькования» красной зоной. Чистый CV, на класс руды и метрики "
             "НЕ влияет. Результат — под изображением с маской слева.",
    )
    talc_cosmetic_alpha = st.slider(
        "Прозрачность красной зоны", 0.0, 1.0, 0.45, 0.05,
    )
    talc_cosmetic_palette = st.selectbox(
        "Палитра микроскопа",
        options=["авто", "панорамная", "жёлтая"],
        help="«авто» — палитра выбирается по минимальной ошибке квантования.",
    )

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

    # Базовый overlay (модель ДО дообучения) — для панели «до» активного обучения.
    # Считается из ОТДЕЛЬНОЙ маски base_result, поэтому «до» не затирается «стало».
    if al_active:
        base_overlay = viewer.make_overlay(
            base, load_mask(base_result.ml.mask_path),
            show_classes=show_classes, opacity=opacity,
        )
    else:
        base_overlay = overlay

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

    # --- Косметическая подсветка талька (опционально) -----------------------
    if show_talc_cosmetic:
        st.markdown("**🔴 Косметическая подсветка талька (область оталькования)**")
        with st.spinner("Палитровая сегментация + сборка талько-зон…"):
            try:
                _pal = None if talc_cosmetic_palette == "авто" else talc_cosmetic_palette
                talc_cos = _cached_talc_overlay(
                    str(image_path), _file_sig, 1400, talc_cosmetic_alpha, _pal,
                )
            except Exception as e:  # noqa: BLE001 — косметика не должна ронять анализ
                talc_cos = None
                st.warning(f"Не удалось построить талько-подсветку: {e}")
        if talc_cos is not None:
            if interactive:
                st.components.v1.html(
                    viewer.interactive_viewer_html(
                        Image.fromarray(talc_cos["overlay"]), height=660
                    ),
                    height=680,
                )
            else:
                st.image(talc_cos["overlay"], use_container_width=True)
            st.caption(
                f"Палитра: **{talc_cos['palette_name']}** · "
                f"тальк (сырой класс): **{talc_cos['talc_raw_pct']:.1f}%** · "
                f"площадь красной зоны: **{talc_cos['region_pct']:.1f}%**. "
                "Слой чисто визуальный — на класс руды и метрики не влияет."
            )

# Маска в разрешении показа — нужна и для экспорта, и для активного обучения.
mask_for_export = mask
if mask_for_export.shape[:2] != (base.size[1], base.size[0]):
    mask_for_export = np.array(
        Image.fromarray(mask_for_export, mode="L").resize(base.size, Image.NEAREST), dtype=np.uint8
    )

# ===========================================================================
# 1) ЭКСПОРТ ДАННЫХ  (сразу после инференса — первое, что видит геолог)
# ===========================================================================
st.divider()
st.subheader("📤 Экспорт данных")

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
    # PDF формируем сразу (как остальные кнопки): анализ кэширован
    # (_cached_run_analysis), поэтому одна кнопка — как у CSV/JSON/PNG/GeoJSON.
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

# ===========================================================================
# 2) АКТИВНОЕ ОБУЧЕНИЕ  (правка предсказания: зум + лассо в одном окне)
# ===========================================================================
st.divider()
st.subheader("🎯 Активное обучение")
st.caption(
    "🔍 **Лупа** — выделите прямоугольник, область растянется на весь кадр; "
    "**Сброс зума** возвращает полный вид. Переключатель **Оригинал/Оверлей** "
    "показывает снимок или маску. ✎ **Лассо** — обведите неверно распознанную "
    "область прямо на приближении и укажите верный класс: исправление уйдёт в "
    "дообучение, и фронтенд засэмплит из него обучающие патчи (src/quantizer.py)."
)

corr_class = st.selectbox(
    "Верный класс участка (для сохранения исправления)",
    options=[config.CLASS_ORDINARY, config.CLASS_FINE,
             config.CLASS_TALC, config.CLASS_ARTIFACT],
    format_func=lambda c: config.CLASS_NAMES[c],
)
r, g, b, _a = config.CLASS_COLORS.get(corr_class, (255, 210, 60, 90))
insp_lasso = viewer.annotator(
    overlay, key="al_annotator", original_image=base,
    color=f"rgba({r}, {g}, {b}, .30)", border_color=f"#{r:02x}{g:02x}{b:02x}",
)

was_name = "—"
if insp_lasso is not None:
    was_cls = dataset_export.majority_class_in_polygon(
        mask_for_export, insp_lasso["points"], base.width, base.height,
    )
    was_name = config.CLASS_NAMES.get(was_cls, "—") if was_cls is not None else "—"
    st.caption(f"Текущее предсказание модели для этой области: **{was_name}**")

comment = st.text_input("Комментарий (необязательно)", key="insp_comment")
author = st.text_input("Автор", value="geolog", key="insp_author")
save_corr = st.button("💾 Сохранить исправление", type="primary")

if save_corr:
    if insp_lasso is None:
        st.warning("Сначала обведите область лассо на изображении выше.")
    else:
        corr_x0, corr_y0, corr_x1, corr_y1 = insp_lasso["bbox"]
        px0, py0 = int(corr_x0 * w), int(corr_y0 * h)
        px1, py1 = int(corr_x1 * w), int(corr_y1 * h)
        polygon_px = [[round(x * w), round(y * h)] for x, y in insp_lasso["points"]]
        saved = storage.save_correction(str(image_path), {
            "was_class_name": was_name,
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

# --- История исправлений (персистентно, не зависит от кликов) ---------------
existing_corrections = storage.list_corrections(str(image_path))
if existing_corrections:
    st.markdown(f"**История исправлений этого изображения ({len(existing_corrections)})**")
    st.dataframe(
        [{
            "Исходный класс": c.get("was_class_name", "—"),
            "Класс эксперта": c.get("correct_class_name", c.get("correct_class")),
            "Комментарий": c.get("comment", ""),
            "Автор": c.get("author", ""),
            "Когда": c.get("created_at", ""),
        } for c in existing_corrections],
        use_container_width=True, hide_index=True,
    )

    if st.button("📦 Собрать экспорт исправлений (ZIP, формат S2_v2)"):
        corr_items = []
        for i, c in enumerate(existing_corrections):
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

# --- Дообучение на исправлениях и показ результата на этой же картинке --------
# Это и есть активное обучение в узком смысле: правка эксперта → модель
# дообучается (быстро, только «голова») → снимок сразу переанализируется
# дообученной моделью, показываем «до/стало».
st.divider()
st.markdown("### 🔁 Дообучить модель на исправлениях и показать результат")
st.caption(
    "Модель дообучается на сохранённых исправлениях этого снимка (быстро — учится "
    "только «голова» классификатора, энкодер заморожен), затем снимок сразу "
    "переанализируется дообученной моделью. Слева «до», справа «стало». "
    "Дообучение накопительное: каждый запуск стартует с предыдущей версии."
)

_TRAINABLE_CORR = {config.CLASS_ORDINARY, config.CLASS_FINE, config.CLASS_TALC}
trainable_corr = [
    c for c in existing_corrections
    if c.get("region_fraction") and int(c.get("correct_class", 0)) in _TRAINABLE_CORR
]
if not trainable_corr:
    st.info(
        "Сохраните хотя бы одно исправление с классом руды (обычные / тонкие / "
        "тальк, не «артефакт») — тогда появится кнопка дообучения."
    )
else:
    n_anchors = st.slider(
        "Якорных тайлов (стабилизируют дообучение, размечены текущим предсказанием)",
        0, 24, 8, help="Случайные тайлы снимка с текущим предсказанием модели и "
                       "малым весом — не дают дообучению «схлопнуть» все классы в исправленный.",
    )
    if st.button(f"🔁 Дообучить на {len(trainable_corr)} исправл. и показать «стало»",
                 type="primary"):
        try:
            with st.spinner("Дообучаем модель на исправлениях и переанализируем снимок…"):
                corr_items = al.build_correction_items(str(image_path), trainable_corr)
                anchor_items = al.build_anchor_items(
                    base, mask_for_export, trainable_corr, n=n_anchors,
                ) if n_anchors else []
                version = st.session_state.get("al_version", 0) + 1
                prev_ckpt = st.session_state.get("al_ckpt")  # накопительно
                ckpt, report = al.retrain_and_save(
                    str(image_path), corr_items + anchor_items,
                    version=version, from_ckpt=prev_ckpt,
                )
                new_result = al.reanalyze(str(image_path), ckpt)
            st.session_state["al_version"] = version
            st.session_state["al_ckpt"] = ckpt
            st.session_state["al_after"] = {"result": new_result, "report": report}
            # Перерисовываем ВСЮ страницу дообученной моделью: первая фотка,
            # класс руды и проценты вверху пересчитаются (al_active → result).
            st.rerun()
        except Exception as e:  # noqa: BLE001
            st.error(f"Не удалось дообучить/переанализировать: {e}")

if al_active:
    _rep = _al_after["report"]
    st.success(
        f"Дообучено (v{st.session_state.get('al_version')}) на {_rep['num_patches']} "
        f"патчах ({_rep['num_feature_vectors']} примеров), train-acc {_rep['train_acc']}. "
        "Первая фотка вверху страницы, класс руды и все проценты уже пересчитаны."
    )
    _bcol, _acol = st.columns(2)
    with _bcol:
        st.markdown("**До — базовая модель**")
        st.image(base_overlay, use_container_width=True)
    with _acol:
        st.markdown(f"**Стало — дообучено (v{st.session_state.get('al_version')})**")
        st.image(overlay, use_container_width=True)

    # (а) Сохранённый чекпоинт активного обучения: путь на диске + скачивание.
    _ckpt_path = st.session_state.get("al_ckpt")
    if _ckpt_path and Path(_ckpt_path).exists():
        _sz = Path(_ckpt_path).stat().st_size / 1e6
        st.caption(
            f"💾 Чекпоинт сохранён на диск: `{_ckpt_path}` ({_sz:.0f} МБ). "
            "Подключить как модель по умолчанию: переменная окружения "
            "`ORE_ML_CKPT=<путь>`. Дообучение накопительное — каждый запуск "
            "стартует с этой версии."
        )
        # Чтение .pth (~сотня МБ) — только по явному запросу, не на каждый rerun.
        if st.checkbox("Подготовить дообученный чекпоинт к скачиванию (.pth)"):
            with open(_ckpt_path, "rb") as _f:
                st.download_button(
                    "💾 Скачать дообученный чекпоинт",
                    data=_f.read(),
                    file_name=Path(_ckpt_path).name,
                    mime="application/octet-stream",
                )

    if st.button("↩️ Сбросить дообучение (вернуться к базовой модели)"):
        for _k in ("al_after", "al_ckpt", "al_version"):
            st.session_state.pop(_k, None)
        st.rerun()
