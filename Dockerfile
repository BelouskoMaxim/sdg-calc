FROM python:3.13-slim

# Устанавливаем системные зависимости (если нужны pandas/numpy/scipy)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    libffi-dev \
    libssl-dev \
    libatlas-base-dev \
    libblas-dev \
    liblapack-dev \
    && rm -rf /var/lib/apt/lists/*

# Обновляем pip/setuptools/wheel — критично для Python 3.13
RUN pip install --upgrade pip setuptools wheel

# Копируем зависимости
COPY requirements.txt .

# Ставим зависимости
RUN pip install --no-cache-dir -r requirements.txt

# Копируем код
COPY . .

# Запуск
CMD ["python", "app.py"]