"""
СТРАНИЦА «Пакетная обработка» (поток B).

Два способа наполнить очередь импорта ДО запуска обработки:
  - выбор папки через проводник (рекурсивно, кастомный компонент);
  - выбор нескольких отдельных файлов через проводник.

Очередь показывает файл/формат/размер файла/размер изображения (если
доступен)/статус валидации/источник и позволяет убрать отдельные элементы
перед запуском. Обработка (ML → метрики → классификация → экспорт CSV/JSON/
PDF) — тот же самый run_analysis/reports.export_all, что и раньше; логику
потока A не трогаем. Дополнительно каждое изображение можно (по умолчанию —
да) зарегистрировать в датасете для разметки (src/dataset_storage.py), чтобы
оно стало доступно на странице «Разметка эксперта».
"""

from __future__ import annotations

import tempfile
import uuid
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from src import batch_import as bi
from src import config, dataset_export, dataset_storage as ds, event_log as ev, reports, storage
from src.pipeline import run_analysis, load_mask
from ui import file_pickers, viewer


def _iter_s2_items(pairs: list[tuple[str, str, str]]):
    """
    Лениво (по одному) открыть пары (изображение, маска) для экспорта в
    формате S2_v2 — не держим все декодированные картинки батча в памяти
    одновременно, а собираем каждую запись прямо перед записью в архив.
    """
    for name, img_path, mask_path in pairs:
        img = viewer.load_display_image(img_path)
        mask = load_mask(mask_path)
        if mask.shape[:2] != (img.size[1], img.size[0]):
            mask = np.array(
                Image.fromarray(mask, mode="L").resize(img.size, Image.NEAREST), dtype=np.uint8
            )
        yield {"name": name, "image": img, "mask": mask}

st.set_page_config(page_title="OreVision — Пакетная обработка", page_icon="🗂️", layout="wide")
config.ensure_dirs()
Image.MAX_IMAGE_PIXELS = None  # панорамы бывают гигапиксельными

st.title("🗂️ Пакетная обработка")
st.caption(
    "Соберите очередь изображений (папка / отдельные файлы), проверьте её, "
    "затем запустите обработку. Результаты попадают в «Историю и лог»."
)

st.session_state.setdefault("import_queue", [])   # list[dict]
st.session_state.setdefault("_folder_pick_nonce", None)
st.session_state.setdefault("_files_pick_nonce", None)

dataset_id = st.text_input(
    "ID датасета (для «Разметки эксперта»)",
    value=st.session_state.get("dataset_id", "default"),
    help="Изображения из очереди можно дополнительно зарегистрировать в этом "
         "датасете, чтобы потом разметить их.",
)
st.session_state["dataset_id"] = dataset_id


def _queue_row(item: bi.QueueItem) -> dict:
    return {
        "id": uuid.uuid4().hex,
        "filename": item.filename,
        "source": item.source,
        "format": item.format,
        "size": item.file_size_bytes,
        "width": item.width,
        "height": item.height,
        "valid": item.valid,
        "error": item.validation_error,
        "source_path": item.source_path,
        "file_bytes": item.file_bytes,
    }


def _add_items(items: list[bi.QueueItem]) -> None:
    existing_keys = {(r["filename"], r["source_path"]) for r in st.session_state["import_queue"]}
    added = 0
    for it in items:
        key = (it.filename, it.source_path)
        if key in existing_keys:
            continue
        st.session_state["import_queue"].append(_queue_row(it))
        added += 1
    return added


# --- Наполнение очереди: два источника (вкладки) -----------------------------
tab_folder, tab_files = st.tabs([
    "🗂️ Выбрать папку (проводник)", "🖼️ Выбрать файлы (проводник)",
])

with tab_folder:
    st.caption(
        "Откроется системный диалог выбора папки. Поддерживаемые форматы "
        f"ищутся рекурсивно: {', '.join(config.SUPPORTED_FORMATS)}."
    )
    folder_value = file_pickers.folder_or_files_picker(
        key="folder_pick", mode="folder", label="Выбрать папку…",
    )
    if folder_value and folder_value.get("nonce") != st.session_state["_folder_pick_nonce"]:
        st.session_state["_folder_pick_nonce"] = folder_value.get("nonce")
        pairs = file_pickers.decode_picked_files(folder_value)
        items = [bi.queue_item_from_bytes(name, data, source=bi.SOURCE_FOLDER_PICKER) for name, data in pairs]
        added = _add_items(items)
        ev.log_import(dataset_id, "queue_scan_folder_picker", found=len(items), added=added)
        st.success(f"Прочитано {len(items)} файлов из папки, добавлено в очередь: {added}.")

with tab_files:
    st.caption("Откроется системный диалог выбора файлов (можно выделить несколько).")
    files_value = file_pickers.folder_or_files_picker(
        key="files_pick", mode="files", label="Выбрать файлы…",
    )
    if files_value and files_value.get("nonce") != st.session_state["_files_pick_nonce"]:
        st.session_state["_files_pick_nonce"] = files_value.get("nonce")
        pairs = file_pickers.decode_picked_files(files_value)
        items = [bi.queue_item_from_bytes(name, data, source=bi.SOURCE_FILE_PICKER) for name, data in pairs]
        added = _add_items(items)
        ev.log_import(dataset_id, "queue_pick_files", found=len(items), added=added)
        st.success(f"Выбрано {len(items)} файлов, добавлено в очередь: {added}.")

# --- Очередь импорта ---------------------------------------------------------
st.divider()
st.subheader(f"Очередь импорта ({len(st.session_state['import_queue'])})")

queue = st.session_state["import_queue"]
if not queue:
    st.info("Очередь пуста. Добавьте изображения одним из способов выше.")
else:
    display_rows = []
    for i, r in enumerate(queue):
        display_rows.append({
            "№": i + 1,
            "Файл": r["filename"],
            "Формат": r["format"],
            "Размер файла": bi.human_size(r["size"]),
            "Размер изображения": f"{r['width']}×{r['height']}" if r["width"] and r["height"] else "—",
            "Источник": bi.SOURCE_LABELS_RU.get(r["source"], r["source"]),
            "Статус": "✅ OK" if r["valid"] else f"❌ {r['error']}",
        })
    st.dataframe(pd.DataFrame(display_rows), use_container_width=True, hide_index=True)

    n_valid = sum(1 for r in queue if r["valid"])
    n_invalid = len(queue) - n_valid
    if n_invalid:
        st.warning(f"Невалидных файлов в очереди: {n_invalid} — они будут пропущены при обработке.")

    remove_col, clear_col = st.columns([3, 1])
    with remove_col:
        options = [f"{i + 1}. {r['filename']}" for i, r in enumerate(queue)]
        to_remove = st.multiselect("Убрать из очереди", options=options, key="remove_from_queue")
        if st.button("🗑️ Удалить выбранные", disabled=not to_remove):
            remove_idx = {int(s.split(".", 1)[0]) - 1 for s in to_remove}
            st.session_state["import_queue"] = [
                r for i, r in enumerate(queue) if i not in remove_idx
            ]
            ev.log_import(dataset_id, "queue_remove_items", removed=len(remove_idx))
            st.rerun()
    with clear_col:
        if st.button("Очистить всю очередь"):
            st.session_state["import_queue"] = []
            ev.log_import(dataset_id, "queue_clear", removed=len(queue))
            st.rerun()

register_in_dataset = st.checkbox(
    "Также зарегистрировать изображения в датасете (для «Разметки эксперта»)",
    value=True,
)

scenario = None
if config.ML_MODE == "mock":
    scenario = st.selectbox(
        "Демо-сценарий (mock применяется ко всем изображениям)",
        options=["refractory", "ordinary", "talc", "review"],
        format_func=lambda s: {
            "refractory": "Труднообогатимая (тонкие)",
            "ordinary": "Рядовая (обычные)",
            "talc": "Оталькованная (тальк >10%)",
            "review": "Пограничный (проверка)",
        }[s],
    )

run = st.button("▶️ Запустить обработку", type="primary", disabled=not any(r["valid"] for r in queue))

# --- Выполнение -------------------------------------------------------------
# Результаты кладём в session_state и рендерим НИЖЕ безусловно (не внутри
# "if run:"), иначе после клика на любую другую кнопку (например, скачать
# сводку CSV) следующий rerun видел бы run=False и вся сводка/кнопки экспорта
# пропадали бы — то же "зависание", что было на главной странице.
if run:
    valid_items = [r for r in queue if r["valid"]]
    params = {"scenario": scenario} if scenario else None
    progress = st.progress(0.0, text="Старт…")
    status = st.empty()
    rows: list[dict] = []
    errors: list[str] = []
    s2_pairs: list[tuple[str, str, str]] = []   # (name, image_path, mask_path) — для экспорта S2_v2

    ev.log_batch(dataset_id, "started", total=len(valid_items))

    for i, r in enumerate(valid_items, start=1):
        status.write(f"Обрабатываю **{r['filename']}** ({i}/{len(valid_items)})…")
        try:
            if r["source_path"]:
                img_path = Path(r["source_path"])
            else:
                # у picker-источников нет пути на диске — материализуем в uploads,
                # чтобы существующий пайплайн (run_analysis) мог прочитать файл.
                img_path = storage.save_upload(r["file_bytes"], r["filename"])

            if register_in_dataset:
                ds.register_image(
                    dataset_id, filename=r["filename"], source=r["source"],
                    file_bytes=r["file_bytes"], source_path=r["source_path"],
                )

            result = run_analysis(str(img_path), params=params)
            base = viewer.load_display_image(str(img_path))
            mask = load_mask(result.ml.mask_path)
            overlay = viewer.make_overlay(base, mask, opacity=0.55)
            overlay_png = config.RESULTS_DIR / img_path.stem / "overlay.png"
            overlay_png.parent.mkdir(parents=True, exist_ok=True)
            overlay.convert("RGB").save(overlay_png)

            reports.export_all(result, overlay_png=overlay_png)  # пишет и в лог
            rows.append({
                "Изображение": r["filename"],
                "Класс руды": result.classification.ore_class,
                "Тальк, %": round(result.metrics.talc_fraction * 100, 1),
                "Тонкие/сульфиды, %": round(result.metrics.fine_of_sulphides * 100, 1),
                "Проверка": "да" if result.classification.needs_review else "",
            })
            # Пути (не декодированные картинки!) — бандл S2_v2 собирается лениво,
            # по одному изображению за раз, только если пользователь его закажет.
            s2_pairs.append((f"{i:04d}_{img_path.stem}", str(img_path), result.ml.mask_path))
            ev.log_batch(dataset_id, "item_done", filename=r["filename"], ore_class=result.classification.ore_class)
        except Exception as e:  # noqa: BLE001
            msg = f"{r['filename']}: {e}"
            errors.append(msg)
            ev.log_error(dataset_id, "batch_item", msg)
        progress.progress(i / len(valid_items), text=f"{i}/{len(valid_items)}")

    status.empty()
    progress.empty()
    ev.log_batch(dataset_id, "finished", done=len(rows), errors=len(errors))
    st.session_state["batch_result"] = {
        "rows": rows, "errors": errors, "s2_pairs": s2_pairs,
        "total": len(valid_items), "register_in_dataset": register_in_dataset,
        "dataset_id": dataset_id,
    }
    st.session_state.pop("batch_s2_zip", None)

batch_result = st.session_state.get("batch_result")
if batch_result:
    rows, errors = batch_result["rows"], batch_result["errors"]
    st.success(f"Готово: обработано {len(rows)} из {batch_result['total']}.")

    if rows:
        df = pd.DataFrame(rows)
        st.subheader("Сводка")
        st.dataframe(df, use_container_width=True, hide_index=True)

        st.subheader("Распределение по классам руды")
        st.bar_chart(df["Класс руды"].value_counts())

        dl_col1, dl_col2 = st.columns(2)
        with dl_col1:
            st.download_button(
                "Скачать сводку (CSV)",
                data=df.to_csv(index=False).encode("utf-8-sig"),
                file_name="batch_summary.csv",
                mime="text/csv",
            )
        with dl_col2:
            if st.button("📦 Собрать экспорт результатов (ZIP, формат S2_v2)"):
                with tempfile.TemporaryDirectory() as _tmp:
                    _bundle_dir = Path(_tmp) / "bundle"
                    dataset_export.export_s2_bundle(
                        _bundle_dir, _iter_s2_items(batch_result["s2_pairs"]),
                        config.CLASS_COLORS, config.CLASS_NAMES,
                    )
                    st.session_state["batch_s2_zip"] = dataset_export.zip_directory(_bundle_dir)

            _zip = st.session_state.get("batch_s2_zip")
            if _zip:
                st.download_button(
                    "⬇️ Скачать (ZIP, формат S2_v2)", data=_zip,
                    file_name="batch_s2v2.zip", mime="application/zip",
                )

        st.caption(f"Отчёты по каждому образцу сохранены в `{config.RESULTS_DIR}`.")
        if batch_result["register_in_dataset"]:
            st.caption(
                f"Изображения также зарегистрированы в датасете "
                f"**{batch_result['dataset_id']}** — откройте «Разметка эксперта», "
                "чтобы их подписать."
            )

    if errors:
        with st.expander(f"Ошибки ({len(errors)})"):
            for e in errors:
                st.error(e)
