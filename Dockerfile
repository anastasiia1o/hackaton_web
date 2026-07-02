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
ENV OREVISION_ML_MODE=mock \
    PYTHONUNBUFFERED=1

EXPOSE 8501

# Запуск сайта, доступного снаружи контейнера.
CMD ["streamlit", "run", "app.py", \
     "--server.address=0.0.0.0", "--server.port=8501"]
