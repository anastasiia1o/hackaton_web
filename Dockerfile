# OreVision — образ сайта (Streamlit). Финальная упаковка demo-режима.
FROM python:3.11-slim

# Системные зависимости для работы с изображениями (Pillow и т.п.).
RUN apt-get update && apt-get install -y --no-install-recommends \
        libjpeg62-turbo zlib1g \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Сначала зависимости — так слой кэшируется и пересборка быстрее.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Затем код проекта.
COPY . .

# Данные остаются локальными (монтируются volume-ом из docker-compose).
# STREAMLIT_BROWSER_GATHERUSAGESTATS=false и HEADLESS=true — без них Streamlit
# в контейнере без TTY может ждать интерактивного ввода при первом запуске
# и отправлять анонимную телеметрию (нарушая "всё работает локально").
ENV OREVISION_ML_MODE=mock \
    PYTHONUNBUFFERED=1 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHERUSAGESTATS=false

EXPOSE 8501

# Запуск сайта, доступного снаружи контейнера.
CMD ["streamlit", "run", "app.py", \
     "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true"]
