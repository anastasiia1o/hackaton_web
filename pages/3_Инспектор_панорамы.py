"""
СТРАНИЦА «Инспектор панорамы» (поток B).

Полный цикл выбора участка на большой панораме:
  - открыть панораму (из датасета или добавить новую тут же);
  - pan/zoom/fit-to-screen/1:1 в интерактивном вьюере (ui/viewer.py);
  - выделить прямоугольный ROI мышью (region_picker), увидеть рамку и точные
    координаты x, y, width, height ОТНОСИТЕЛЬНО исходной панорамы;
  - сохранить ROI (координаты + вырезанный участок) в датасете
    (src/dataset_storage.py) — можно сохранить несколько ROI на одну панораму
    и повторно открыть любой из них;
  - открыть сохранённый участок в «Активном обучении / Разметке эксперта».

Логику анализа/классификации (поток A) не трогаем — это чистый viewer + ROI.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image

from src import config, dataset_storage as ds, event_log as ev
from ui import viewer

st.set_page_config(page_title="OreVision — Инспектор панорамы", page_icon="🔬", layout="wide")
config.ensure_dirs()
Image.MAX_IMAGE_PIXELS = None

st.title("🔬 Инспектор панорамы")
st.caption(
    "Откройте панораму, найдите нужный фрагмент (pan/zoom), выделите участок "
    "мышью и сохраните его координаты для разметки."
)

dataset_id = st.text_input(
    "ID датасета", value=st.session_state.get("dataset_id", "default"),
    help="Тот же датасет, что и на странице «Пакетная обработка».",
)
st.session_state["dataset_id"] = dataset_id

existing_ids = ds.list_dataset_ids()
if existing_ids:
    st.caption("Существующие датасеты: " + ", ".join(f"`{d}`" for d in existing_ids))

images = ds.list_images(dataset_id)
valid_images = [im for im in images if im.get("valid")]

with st.expander("➕ Добавить новую панораму в этот датасет", expanded=not valid_images):
    add_tab_path, add_tab_upload = st.tabs(["Путь к файлу", "Загрузить файл"])
    with add_tab_path:
        p_str = st.text_input("Путь к изображению панорамы")
        if st.button("Зарегистрировать по пути") and p_str.strip():
            p = Path(p_str.strip())
            if not p.is_file():
                st.error(f"Файл не найден: {p}")
            else:
                row = ds.register_image(dataset_id, filename=p.name, source="manual_path", source_path=str(p))
                ev.log_import(dataset_id, "inspector_register_path", filename=p.name, valid=row["valid"])
                if row["valid"]:
                    st.success(f"Зарегистрировано: {p.name} ({row['width']}×{row['height']})")
                    st.rerun()
                else:
                    st.error(f"Не удалось прочитать изображение: {row['validation_error']}")
    with add_tab_upload:
        up = st.file_uploader(
            "Изображение панорамы", type=[e.lstrip(".") for e in config.SUPPORTED_FORMATS],
            key="inspector_upload",
        )
        if up is not None and st.button("Зарегистрировать загруженный файл"):
            data = up.getvalue()
            row = ds.register_image(dataset_id, filename=up.name, source="file_picker", file_bytes=data)
            ev.log_import(dataset_id, "inspector_register_upload", filename=up.name, valid=row["valid"])
            if row["valid"]:
                st.success(f"Зарегистрировано: {up.name} ({row['width']}×{row['height']})")
                st.rerun()
            else:
                st.error(f"Не удалось прочитать изображение: {row['validation_error']}")

images = ds.list_images(dataset_id)
valid_images = [im for im in images if im.get("valid")]

if not valid_images:
    st.info(
        "В этом датасете пока нет изображений. Добавьте панораму выше или "
        "зарегистрируйте её на странице «Пакетная обработка»."
    )
    st.stop()

img_options = {f"{im['original_filename']} ({im['width']}×{im['height']}) — {im['image_id']}": im["image_id"]
               for im in valid_images}
picked_label = st.selectbox("Панорама", options=list(img_options.keys()))
image_id = img_options[picked_label]
image_row = ds.get_image(dataset_id, image_id)
image_path = ds.resolve_image_path(dataset_id, image_id)

if image_path is None or not Path(image_path).exists():
    st.error("Файл изображения недоступен на диске (перемещён/удалён?).")
    st.stop()

W, H = image_row["width"], image_row["height"]
if max(W, H) > config.MAX_DIMENSION_WARN:
    st.warning(
        f"Большая панорама ({W}×{H}). Обзор ниже уменьшен для скорости, но "
        "координаты ROI пересчитываются в пиксели ОРИГИНАЛА точно."
    )

base = viewer.load_display_image(str(image_path))

st.subheader("Обзор панорамы (pan / zoom / fit / 1:1)")
st.components.v1.html(viewer.interactive_viewer_html(base, height=560), height=580)

st.subheader("Выделение участка (ROI)")
st.caption("Тяните мышью новую область, за уголок/край — изменить размер, за середину — сдвинуть.")
x0f, y0f, x1f, y1f = viewer.region_picker(
    base, key=f"inspector_roi_{image_id}", bbox=(0.40, 0.40, 0.55, 0.55),
    color="rgba(255, 210, 60, .28)", border_color="#ffcf33",
)
rx, ry = int(round(x0f * W)), int(round(y0f * H))
rw, rh = int(round(x1f * W)) - rx, int(round(y1f * H)) - ry

coord_col, action_col = st.columns([2, 1])
with coord_col:
    st.markdown(
        f"**Координаты ROI (пиксели исходной панорамы):** "
        f"x=`{rx}` · y=`{ry}` · width=`{rw}` · height=`{rh}`"
    )
with action_col:
    save_roi = st.button("💾 Сохранить ROI", type="primary")

if save_roi:
    with st.spinner("Вырезаю участок в высоком разрешении…"):
        crop, meta = viewer.crop_region_highres(str(image_path), x0f, y0f, x1f, y1f, out_max=2200)
    px, py, pw, ph = meta["region_px_orig"]
    roi = ds.create_roi(
        dataset_id, image_id, x=px, y=py, width=pw, height=ph,
        source_image_width=W, source_image_height=H, roi_image=crop,
    )
    ev.log_import(dataset_id, "roi_saved", image_id=image_id, region_id=roi["region_id"])
    if meta["capped"]:
        st.info(
            f"Исходник очень большой — участок сохранён с понижением "
            f"(масштаб {meta['native_scale']}), сохранённый ROI {crop.size[0]}×{crop.size[1]}px."
        )
    st.success(f"ROI сохранён: {roi['region_id']}")
    st.rerun()

st.divider()
st.subheader("Сохранённые ROI этой панорамы")
rois = ds.list_rois(dataset_id, image_id)
if not rois:
    st.info("Пока нет сохранённых ROI для этой панорамы.")
else:
    from src import annotation_config as ac
    table = [{
        "ROI": r["region_id"],
        "x": r["x"], "y": r["y"], "width": r["width"], "height": r["height"],
        "Статус": ac.STATUS_LABELS_RU.get(r.get("status"), r.get("status")),
        "Ревизия": r.get("revision", 0),
        "Создан": r.get("created_at", ""),
    } for r in rois]
    st.dataframe(pd.DataFrame(table), use_container_width=True, hide_index=True)

    roi_ids = [r["region_id"] for r in rois]
    picked_roi = st.selectbox("Выбрать ROI для просмотра / открытия в редакторе", options=roi_ids)

    prev_col, open_col = st.columns([1, 1])
    with prev_col:
        if st.button("👁️ Показать сохранённый участок"):
            roi_img = ds.load_roi_image(dataset_id, image_id, picked_roi)
            if roi_img is not None:
                st.image(roi_img, caption=f"{picked_roi} ({roi_img.size[0]}×{roi_img.size[1]}px)",
                         use_container_width=True)
    with open_col:
        if st.button("✏️ Открыть в редакторе разметки", type="primary"):
            st.session_state["al_selected"] = {
                "dataset_id": dataset_id, "image_id": image_id, "region_id": picked_roi,
            }
            st.switch_page("pages/4_Активное_обучение_Разметка_эксперта.py")
