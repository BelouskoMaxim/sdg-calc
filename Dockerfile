FROM python:3.12-slim

RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . .

# Обновление pip, setuptools и wheel
RUN pip install --upgrade pip setuptools==68.0.0 wheel

# Установка зависимостей
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8050
CMD ["python", "app.py"]