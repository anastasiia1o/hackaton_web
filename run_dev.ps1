# run_dev.ps1 — быстрый запуск в режиме разработки (Windows PowerShell).
# Использование:  .\run_dev.ps1
# Создаёт venv при первом запуске, ставит зависимости и поднимает сайт.

$ErrorActionPreference = "Stop"

if (-not (Test-Path ".venv")) {
    Write-Host "Создаю виртуальное окружение .venv ..." -ForegroundColor Cyan
    python -m venv .venv
}

Write-Host "Активирую .venv ..." -ForegroundColor Cyan
. .\.venv\Scripts\Activate.ps1

Write-Host "Устанавливаю зависимости ..." -ForegroundColor Cyan
pip install -r requirements.txt

Write-Host "Запускаю OreVision на http://localhost:8501 ..." -ForegroundColor Green
streamlit run app.py
