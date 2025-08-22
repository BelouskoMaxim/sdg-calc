# Используем официальное Python 3.12 slim-образ
FROM python:3.12-slim

# Обновляем системные зависимости и ставим компилятор
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Устанавливаем рабочую директорию
WORKDIR /app

# Копируем файлы проекта
COPY . .

# Обновляем pip, setuptools и wheel
RUN pip install --upgrade pip setuptools wheel

# Устанавливаем зависимости проекта
RUN pip install --no-cache-dir -r requirements.txt

# Экспонируем порт Dash
EXPOSE 8050

# Команда запуска приложения
CMD ["python", "app.py"]