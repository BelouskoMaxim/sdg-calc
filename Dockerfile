# Используем Python 3.12 с Debian (prebuilt wheels для Pandas 2.0.3)
FROM python:3.12-bullseye

# Рабочая директория
WORKDIR /app

# Копируем файлы проекта
COPY . .

# Обновляем pip
RUN pip install --upgrade pip

# Устанавливаем зависимости
RUN pip install --no-cache-dir -r requirements.txt

# Открываем порт для Dash
EXPOSE 8050
ENV PORT=8050
ENV HOST=0.0.0.0

# Запуск приложения
CMD ["python", "app.py"]