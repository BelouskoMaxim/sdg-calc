FROM python:3.13

# Обновляем репозитории и ставим зависимости
RUN sed -i 's|deb.debian.org|deb.debian.org|g' /etc/apt/sources.list && \
    apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        gcc \
        g++ \
        libffi-dev \
        libssl-dev \
        libatlas-base-dev \
        libblas-dev \
        liblapack-dev \
    && rm -rf /var/lib/apt/lists/*

# Обновляем pip/setuptools/wheel
RUN pip install --upgrade pip setuptools wheel

# Копируем зависимости
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Копируем код
COPY . .

# Запуск
CMD ["python", "app.py"]