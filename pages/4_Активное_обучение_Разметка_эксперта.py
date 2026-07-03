"""
СТРАНИЦА «Активное обучение / Разметка эксперта» (поток B).

Упрощённый CVAT-подобный редактор для второго этапа (морфология срастаний):
эксперт открывает ROI, сохранённый в «Инспекторе панорамы», исправляет или
создаёт маску (кисть/полигон/ластик), может присвоить один класс всему
участку целиком, и сохраняет результат с ревизией и статусом — для
последующего дообучения. Первый этап (пиксельная сегментация фаз силами ML)
и текущий rule-based классификатор (поток A) этой страницей не затрагиваются.
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

from src import annotation_config as ac
from src import config, dataset_storage as ds, event_log as ev
from ui import annotation_editor as ae

st.set_page_config(
    page_title="OreVision — Активное обучение / Разметка эксперта", page_icon="🖌️", layout="wide",
)
config.ensure_dirs()

st.title("🖌️ Активное обучение / Разметка эксперта")
st.caption(
    "Коррекция и создание масок ВТОРОГО этапа (морфология срастаний: тальк, "
    "обычные/тонкие срастания) для дообучения. Классы настраиваются в "
    "`configs/annotation_classes.json`, не хардкожены в интерфейсе."
)

classes = ac.load_classes()
legend_html = " ".join(
    f'<span style="display:inline-block;margin:2px 10px 2px 0;">'
    f'<span style="display:inline-block;width:12px;height:12px;border:1px solid #888;'
    f'background:rgba({c.color[0]},{c.color[1]},{c.color[2]},{c.color[3]/255:.2f});'
    f'border-radius:2px;margin-right:5px;"></span>{c.name_ru}</span>'
    for c in classes
)
st.markdown(legend_html, unsafe_allow_html=True)

# --- Выбор датасета / панорамы / ROI ----------------------------------------
selected = st.session_state.get("al_selected", {})
dataset_id = st.text_input(
    "ID датасета", value=selected.get("dataset_id") or st.session_state.get("dataset_id", "default"),
)
st.session_state["dataset_id"] = dataset_id

images = [im for im in ds.list_images(dataset_id) if im.get("valid")]
if not images:
    st.info(
        "В датасете нет панорам. Зарегистрируйте их на странице «Пакетная "
        "обработка» или «Инспектор панорамы»."
    )
    st.stop()

img_ids = [im["image_id"] for im in images]
default_img_idx = img_ids.index(selected["image_id"]) if selected.get("image_id") in img_ids else 0
img_labels = {im["image_id"]: f"{im['original_filename']} ({im['image_id']})" for im in images}
image_id = st.selectbox(
    "Панорама", options=img_ids, index=default_img_idx, format_func=lambda i: img_labels[i],
)

rois = ds.list_rois(dataset_id, image_id)
if not rois:
    st.info(
        "У этой панорамы пока нет сохранённых ROI. Выделите и сохраните участок "
        "в «Инспекторе панорамы»."
    )
    st.stop()

roi_ids = [r["region_id"] for r in rois]
default_roi_idx = roi_ids.index(selected["region_id"]) if selected.get("region_id") in roi_ids else 0
region_id = st.selectbox("Участок (ROI)", options=roi_ids, index=default_roi_idx)
roi = ds.load_roi(dataset_id, image_id, region_id)

st.caption(
    f"Координаты в исходной панораме: x={roi['x']} · y={roi['y']} · "
    f"width={roi['width']} · height={roi['height']}. "
    f"Ревизия: {roi.get('revision', 0)} · статус: "
    f"{ac.STATUS_LABELS_RU.get(roi.get('status'), roi.get('status'))}."
)

roi_image = ds.load_roi_image(dataset_id, image_id, region_id)
mask, state, shapes_geojson = ds.load_annotation(dataset_id, image_id, region_id)

# --- Слои / отображение ------------------------------------------------------
st.session_state.setdefault("al_reload_nonce", 0)
ctrl1, ctrl2, ctrl3, ctrl4 = st.columns([1, 1, 2, 1])
with ctrl1:
    show_image = st.checkbox("Показать изображение", value=True)
with ctrl2:
    show_mask = st.checkbox("Показать маску", value=True)
with ctrl3:
    mask_opacity = st.slider("Прозрачность маски", 0.0, 1.0, 0.6, 0.05)
with ctrl4:
    if st.button("↺ Перезагрузить сохранённую версию"):
        st.session_state["al_reload_nonce"] += 1
        st.rerun()

region_key = f"{dataset_id}|{image_id}|{region_id}|{st.session_state['al_reload_nonce']}"

value = ae.annotation_canvas(
    roi_image, mask, classes, region_key,
    shapes_geojson=shapes_geojson, show_image=show_image, show_mask=show_mask,
    mask_opacity=mask_opacity,
)

# --- Сохранение ---------------------------------------------------------------
st.divider()
save_col, status_col, author_col = st.columns([1, 2, 1])
with status_col:
    status_options = list(ac.ALL_STATUSES)
    current_status = roi.get("status", ac.STATUS_DRAFT)
    status = st.selectbox(
        "Статус разметки", options=status_options,
        index=status_options.index(current_status) if current_status in status_options else 0,
        format_func=lambda s: ac.STATUS_LABELS_RU.get(s, s),
    )
with author_col:
    author = st.text_input("Автор", value="geolog")
with save_col:
    save_clicked = st.button("💾 Сохранить", type="primary")

if save_clicked:
    if value is None:
        st.warning("Редактор ещё не готов — попробуйте ещё раз через секунду.")
    else:
        new_mask = ae.decode_mask_from_value(value)
        if new_mask is None or new_mask.shape != (roi_image.height, roi_image.width):
            st.error("Не удалось прочитать маску из редактора (несовпадение размера).")
        else:
            new_state = ds.save_annotation(
                dataset_id, image_id, region_id,
                mask=new_mask, shapes=value.get("shapes"), status=status, author=author,
            )
            ev.log_annotation_save(dataset_id, image_id, region_id, status, new_state["revision"])
            st.success(f"Сохранено: ревизия {new_state['revision']}, статус «{ac.STATUS_LABELS_RU.get(status, status)}».")
            counts = new_state.get("class_pixel_counts", {})
            classes_map = ac.classes_by_id()
            by_name = {
                classes_map[int(k)].name_ru if int(k) in classes_map else k: v
                for k, v in counts.items()
            }
            st.caption("Пиксели по классам: " + ", ".join(f"{k}={v}" for k, v in by_name.items()))

# --- Экспорт для дообучения ---------------------------------------------------
st.divider()
st.subheader("Экспорт для дообучения")
st.caption(
    "В экспорт попадают ТОЛЬКО разметки со статусом "
    "«Принято для обучения» (accepted_for_training)."
)
if st.button("📦 Экспортировать подтверждённые разметки для дообучения"):
    result = ds.export_active_learning(dataset_id)
    ev.log_export(dataset_id, result["export_id"], result["num_samples"])
    if result["num_samples"] == 0:
        st.warning("Нет разметок со статусом accepted_for_training — экспортировать нечего.")
    else:
        st.success(f"Экспортировано {result['num_samples']} пар image/mask в `{result['dir']}`.")
        with tempfile.TemporaryDirectory() as tmp:
            zip_base = str(Path(tmp) / result["export_id"])
            zip_path = shutil.make_archive(zip_base, "zip", result["dir"])
            with open(zip_path, "rb") as f:
                st.download_button(
                    "Скачать экспорт (ZIP)", data=f.read(),
                    file_name=f"{result['export_id']}.zip", mime="application/zip",
                )

exports = ds.list_exports(dataset_id)
if exports:
    st.caption(f"Ранее выполненные экспорты этого датасета: {', '.join(exports)}")
