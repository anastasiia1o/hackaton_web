"""
REPORTS — экспорт результатов: CSV, JSON и PDF.

Всё сохраняется ЛОКАЛЬНО в папку результатов изображения.
PDF собираем через reportlab (чистый Python, без системных зависимостей —
это важно, чтобы легко завелось на Windows и в Docker).
"""

from __future__ import annotations

import csv
import io
import json
from pathlib import Path

from . import storage
from .metrics import as_percent_rows
from .schemas import AnalysisResult


def export_json(result: AnalysisResult) -> Path:
    """Сохранить полный результат в result.json."""
    d = storage.result_dir(result.image_path)
    path = d / "result.json"
    path.write_text(
        json.dumps(result.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return path


def export_csv(result: AnalysisResult) -> Path:
    """Сохранить таблицу метрик в metrics.csv."""
    d = storage.result_dir(result.image_path)
    path = d / "metrics.csv"
    rows = as_percent_rows(result.metrics)
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["Метрика", "Значение"])
        writer.writeheader()
        writer.writerows(rows)
    return path


def csv_bytes(result: AnalysisResult) -> bytes:
    """CSV в память (для кнопки download в Streamlit без записи на диск)."""
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=["Метрика", "Значение"])
    writer.writeheader()
    writer.writerows(as_percent_rows(result.metrics))
    return buf.getvalue().encode("utf-8-sig")


def export_pdf(result: AnalysisResult, overlay_png: Path | None = None) -> Path:
    """
    Собрать краткий PDF-отчёт: заголовок, вывод классификации, таблица метрик,
    (опц.) картинка с overlay. Возвращает путь к PDF.
    """
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import cm
    from reportlab.lib import colors
    from reportlab.platypus import (
        SimpleDocTemplate,
        Paragraph,
        Spacer,
        Table,
        TableStyle,
        Image as RLImage,
    )
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

    d = storage.result_dir(result.image_path)
    path = d / "report.pdf"

    styles = getSampleStyleSheet()
    title_style = styles["Title"]
    h2 = ParagraphStyle("h2", parent=styles["Heading2"], spaceBefore=10)
    normal = styles["BodyText"]

    doc = SimpleDocTemplate(str(path), pagesize=A4,
                            leftMargin=2*cm, rightMargin=2*cm,
                            topMargin=1.5*cm, bottomMargin=1.5*cm)
    story = []

    story.append(Paragraph("OreVision — отчёт по анализу шлифа", title_style))
    story.append(Spacer(1, 6))
    story.append(Paragraph(f"Изображение: <b>{result.image_name}</b>", normal))
    story.append(Paragraph(
        f"Модель: {result.ml.model_version} · "
        f"Время инференса: {result.ml.inference_time_ms} мс · "
        f"Размер: {result.ml.image_size.get('width')}×{result.ml.image_size.get('height')} px",
        normal,
    ))

    story.append(Paragraph("Итоговая классификация", h2))
    story.append(Paragraph(f"<b>{result.classification.ore_class}</b>", normal))
    story.append(Paragraph(result.classification.reason, normal))

    if overlay_png and Path(overlay_png).exists():
        story.append(Paragraph("Цветовая маска", h2))
        img = RLImage(str(overlay_png))
        max_w = 16 * cm
        if img.drawWidth > max_w:
            ratio = max_w / img.drawWidth
            img.drawWidth *= ratio
            img.drawHeight *= ratio
        story.append(img)

    story.append(Paragraph("Количественные метрики", h2))
    data = [["Метрика", "Значение"]] + [
        [r["Метрика"], r["Значение"]] for r in as_percent_rows(result.metrics)
    ]
    table = Table(data, colWidths=[11*cm, 4*cm])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f3b57")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#eef3f8")]),
    ]))
    story.append(table)

    if result.ml.warnings:
        story.append(Paragraph("Предупреждения", h2))
        for w in result.ml.warnings:
            story.append(Paragraph(f"• {w}", normal))

    story.append(Spacer(1, 10))
    story.append(Paragraph(
        "<i>Классификация выполнена прозрачными правилами (rule-based) на основе "
        "площадей фаз. Метрики рассчитаны сайтом OreVision, сегментация — ML-модулем.</i>",
        normal,
    ))

    doc.build(story)
    return path


def export_all(result: AnalysisResult, overlay_png: Path | None = None) -> dict[str, Path]:
    """Экспортировать всё сразу и записать в лог. Удобно для batch."""
    paths = {
        "json": export_json(result),
        "csv": export_csv(result),
        "pdf": export_pdf(result, overlay_png=overlay_png),
    }
    storage.append_log({
        "image": result.image_name,
        "ore_class": result.classification.ore_class,
        "talc_fraction": round(result.metrics.talc_fraction, 4),
        "fine_of_sulphides": round(result.metrics.fine_of_sulphides, 4),
        "model_version": result.ml.model_version,
    })
    return paths
