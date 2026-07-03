"""
СТРАНИЦА «Разметка эксперта» (поток B).

Простое окно: геолог открывает шлиф/панораму, обводит мышью произвольную
ЗАМКНУТУЮ область (не только прямоугольником — свободное лассо), выбирает
класс (или пишет свою пометку) и добавляет подписанный участок. Так можно
набрать несколько подписанных областей на одном изображении, затем сохранить
результат для дообучения (active learning). Второй этап (морфология
срастаний) и текущий rule-based классификатор (поток A) не затрагиваются.
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw

from src import annotation_config as ac
from src import config, dataset_storage as ds, event_log as ev
from ui import viewer

st.set_page_config(page_title="OreVision — Разметка эксперта", page_icon="🖌️", layout="wide")
config.ensure_dirs()
Image.MAX_IMAGE_PIXELS = None

st.title("🖌️ Разметка эксперта")
st.caption(
    "Обведите мышью произвольную область на шлифе (замкнутая линия, не "
    "обязательно прямоугольник), подпишите её и добавьте. Можно отметить "
    "несколько участков на одном изображении, затем сохранить для дообучения."
)

classes = ac.load_classes()
labeled_classes = [c for c in classes if c.id != ac.UNLABELED_ID]
legend_html = " ".join(
    f'<span style="display:inline-block;margin:2px 10px 2px 0;">'
    f'<span style="display:inline-block;width:12px;height:12px;border:1px solid #888;'
    f'background:rgba({c.color[0]},{c.color[1]},{c.color[2]},{c.color[3]/255:.2f});'
    f'border-radius:2px;margin-right:5px;"></span>{c.name_ru}</span>'
    for c in classes
)
st.markdown(legend_html, unsafe_allow_html=True)

# --- Датасет и панорама -------------------------------------------------------
dataset_id = st.text_input("ID датасета", value=st.session_state.get("dataset_id", "default"))
st.session_state["dataset_id"] = dataset_id

with st.expander("➕ Добавить новый шлиф/панораму в этот датасет"):
    add_tab_path, add_tab_upload = st.tabs(["Путь к файлу", "Загрузить файл"])
    with add_tab_path:
        p_str = st.text_input("Путь к изображению", key="al_add_path")
        if st.button("Зарегистрировать по пути", key="al_add_path_btn") and p_str.strip():
            p = Path(p_str.strip())
            if not p.is_file():
                st.error(f"Файл не найден: {p}")
            else:
                row = ds.register_image(dataset_id, filename=p.name, source="manual_path", source_path=str(p))
                ev.log_import(dataset_id, "al_register_path", filename=p.name, valid=row["valid"])
                if row["valid"]:
                    st.success(f"Зарегистрировано: {p.name}")
                    st.rerun()
                else:
                    st.error(f"Не удалось прочитать изображение: {row['validation_error']}")
    with add_tab_upload:
        up = st.file_uploader(
            "Изображение шлифа/панорамы", type=[e.lstrip(".") for e in config.SUPPORTED_FORMATS],
            key="al_upload",
        )
        if up is not None and st.button("Зарегистрировать загруженный файл", key="al_upload_btn"):
            data = up.getvalue()
            row = ds.register_image(dataset_id, filename=up.name, source="file_picker", file_bytes=data)
            ev.log_import(dataset_id, "al_register_upload", filename=up.name, valid=row["valid"])
            if row["valid"]:
                st.success(f"Зарегистрировано: {up.name}")
                st.rerun()
            else:
                st.error(f"Не удалось прочитать изображение: {row['validation_error']}")

images = [im for im in ds.list_images(dataset_id) if im.get("valid")]
if not images:
    st.info("В датасете нет изображений. Добавьте шлиф/панораму выше или на странице «Пакетная обработка».")
    st.stop()

img_labels = {im["image_id"]: f"{im['original_filename']} ({im['image_id']})" for im in images}
image_id = st.selectbox("Шлиф / панорама", options=list(img_labels.keys()), format_func=lambda i: img_labels[i])
image_path = ds.resolve_image_path(dataset_id, image_id)
if image_path is None or not Path(image_path).exists():
    st.error("Файл изображения недоступен на диске (перемещён/удалён?).")
    st.stop()

disp = viewer.load_display_image(str(image_path))
roi = ds.get_or_create_whole_image_roi(dataset_id, image_id, disp)
region_id = roi["region_id"]

_, saved_state, saved_shapes = ds.load_annotation(dataset_id, image_id, region_id)

# --- Состояние черновика (набранные, но ещё не сохранённые/уже сохранённые подписанные участки) ---
draft_key = f"al_draft_{dataset_id}_{image_id}"
if draft_key not in st.session_state:
    loaded = []
    if saved_shapes and saved_shapes.get("features"):
        for f in saved_shapes["features"]:
            ring = f["geometry"]["coordinates"][0]
            pts = ring[:-1] if len(ring) > 1 and ring[0] == ring[-1] else ring
            props = f.get("properties", {})
            loaded.append({
                "points": [[float(x), float(y)] for x, y in pts],
                "class_id": int(props.get("class_id", ac.UNLABELED_ID)),
                "note": props.get("note", ""),
            })
    st.session_state[draft_key] = loaded

st.session_state.setdefault(f"al_lasso_nonce_{draft_key}", 0)


def _class_by_id(cid: int) -> ac.AnnotationClass | None:
    return next((c for c in classes if c.id == cid), None)


def _shape_label(shape: dict) -> str:
    c = _class_by_id(shape["class_id"])
    name = c.name_ru if c else str(shape["class_id"])
    return f"{name} — {shape['note']}" if shape.get("note") else name


def _shape_color(shape: dict) -> tuple[str, str]:
    c = _class_by_id(shape["class_id"])
    if c is None:
        return "rgba(150,150,150,.35)", "#969696"
    r, g, b, a = c.color
    return f"rgba({r},{g},{b},{max(a, 120) / 255:.2f})", f"#{r:02x}{g:02x}{b:02x}"


committed = []
for s in st.session_state[draft_key]:
    fill, border = _shape_color(s)
    committed.append({"points": s["points"], "label": _shape_label(s), "color": fill, "border_color": border})

# --- Выделение + подпись ------------------------------------------------------
st.subheader("Выделение участка")
lasso_key = f"al_lasso_{draft_key}_{st.session_state[f'al_lasso_nonce_{draft_key}']}"
pending = viewer.lasso_picker(
    disp, key=lasso_key, committed=committed,
    color="rgba(255, 210, 60, .30)", border_color="#ffcf33",
)

pick_col, note_col, add_col = st.columns([2, 3, 1])
with pick_col:
    class_options = [c.id for c in labeled_classes]
    pick_class = st.selectbox(
        "Класс участка", options=class_options,
        format_func=lambda cid: _class_by_id(cid).name_ru,
    )
with note_col:
    note = st.text_input("Своя подпись / комментарий (необязательно)")
with add_col:
    st.write("")
    add_clicked = st.button("➕ Добавить", disabled=pending is None)

if add_clicked and pending is not None:
    st.session_state[draft_key].append({
        "points": [list(p) for p in pending["points"]],
        "class_id": pick_class, "note": note,
    })
    st.session_state[f"al_lasso_nonce_{draft_key}"] += 1
    st.rerun()
elif pending is None:
    st.caption("Обведите область мышью на изображении выше, затем нажмите «Добавить».")

# --- Список добавленных участков ---------------------------------------------
st.subheader(f"Подписанные участки ({len(st.session_state[draft_key])})")
if st.session_state[draft_key]:
    table = [{
        "№": i + 1, "Класс/подпись": _shape_label(s), "Точек контура": len(s["points"]),
    } for i, s in enumerate(st.session_state[draft_key])]
    st.dataframe(pd.DataFrame(table), use_container_width=True, hide_index=True)

    remove_opts = [f"{i + 1}. {_shape_label(s)}" for i, s in enumerate(st.session_state[draft_key])]
    to_remove = st.multiselect("Убрать участок(и)", options=remove_opts)
    if st.button("🗑️ Убрать выбранные", disabled=not to_remove):
        idx_remove = {int(x.split(".", 1)[0]) - 1 for x in to_remove}
        st.session_state[draft_key] = [
            s for i, s in enumerate(st.session_state[draft_key]) if i not in idx_remove
        ]
        st.rerun()
else:
    st.info("Пока нет ни одного подписанного участка.")

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
    st.write("")
    save_clicked = st.button("💾 Сохранить разметку", type="primary")

if save_clicked:
    shapes = st.session_state[draft_key]
    W, H = disp.width, disp.height
    mask_img = Image.new("L", (W, H), ac.UNLABELED_ID)
    draw = ImageDraw.Draw(mask_img)
    for s in shapes:
        if len(s["points"]) < 3:
            continue
        poly_px = [(x * W, y * H) for x, y in s["points"]]
        draw.polygon(poly_px, fill=s["class_id"])
    mask_array = np.array(mask_img, dtype=np.uint8)

    geojson = {
        "type": "FeatureCollection",
        "features": [{
            "type": "Feature",
            "properties": {
                "class_id": s["class_id"],
                "class_name_ru": (_class_by_id(s["class_id"]) or ac.AnnotationClass(0, "", "", (0, 0, 0, 0))).name_ru,
                "note": s.get("note", ""),
            },
            "geometry": {"type": "Polygon", "coordinates": [
                [[x, y] for x, y in s["points"]] + [list(s["points"][0])]
            ]},
        } for s in shapes if len(s["points"]) >= 3],
    }

    new_state = ds.save_annotation(
        dataset_id, image_id, region_id,
        mask=mask_array, shapes=geojson, status=status, author=author,
    )
    ev.log_annotation_save(dataset_id, image_id, region_id, status, new_state["revision"])
    st.success(
        f"Сохранено: ревизия {new_state['revision']}, статус "
        f"«{ac.STATUS_LABELS_RU.get(status, status)}», участков: {len(shapes)}."
    )

# --- Экспорт для дообучения ---------------------------------------------------
st.divider()
st.subheader("Экспорт для дообучения")
st.caption(
    "В экспорт попадают ТОЛЬКО разметки со статусом «Принято для обучения» "
    "(accepted_for_training)."
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
