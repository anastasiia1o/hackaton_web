"""
BATCH — пакетная обработка серии изображений без участия пользователя.

Запуск (Windows PowerShell):
    python batch_process.py data\\uploads

Для каждого изображения из папки:
    ML → метрики → классификация → сохранение CSV/JSON/PDF + строка в лог.
Анализ делает встроенная модель (config.ML_MODE, по умолчанию local).
В конце печатает сводную таблицу по всем образцам.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src import config, reports
from src.pipeline import run_analysis
from ui import viewer
from src.pipeline import load_mask
from PIL import Image


def find_images(folder: Path) -> list[Path]:
    files: list[Path] = []
    for ext in config.SUPPORTED_FORMATS:
        files.extend(folder.glob(f"*{ext}"))
        files.extend(folder.glob(f"*{ext.upper()}"))
    return sorted(set(files))


def main() -> int:
    parser = argparse.ArgumentParser(description="OreVision batch processing")
    parser.add_argument("folder", help="Папка с изображениями")
    parser.add_argument("--mode", default=None, help="local|real (по умолчанию из config)")
    args = parser.parse_args()

    config.ensure_dirs()
    folder = Path(args.folder)
    images = find_images(folder)
    if not images:
        print(f"В папке {folder} не найдено изображений {config.SUPPORTED_FORMATS}")
        return 1

    params = None
    print(f"Найдено изображений: {len(images)}\n")
    summary = []

    for img in images:
        try:
            result = run_analysis(str(img), params=params, mode=args.mode)
            # overlay для PDF
            base = Image.open(img)
            mask = load_mask(result.ml.mask_path)
            overlay = viewer.make_overlay(base, mask, opacity=0.55)
            overlay_png = config.RESULTS_DIR / img.stem / "overlay.png"
            overlay_png.parent.mkdir(parents=True, exist_ok=True)
            overlay.convert("RGB").save(overlay_png)

            reports.export_all(result, overlay_png=overlay_png)
            summary.append((
                img.name,
                result.classification.ore_class,
                f"{result.metrics.talc_fraction*100:.1f}%",
                f"{result.metrics.fine_of_sulphides*100:.1f}%",
            ))
            print(f"[OK]  {img.name:30s} → {result.classification.ore_class}")
        except Exception as e:  # noqa: BLE001
            print(f"[ERR] {img.name}: {e}", file=sys.stderr)

    print("\n=== СВОДКА ===")
    print(f"{'Изображение':32s}{'Класс руды':28s}{'Тальк':8s}{'Тонкие/сульф.'}")
    for name, cls, talc, fine in summary:
        print(f"{name:32s}{cls:28s}{talc:8s}{fine}")
    print(f"\nОтчёты сохранены в: {config.RESULTS_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
