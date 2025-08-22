FROM python:3.12-bullseye
WORKDIR /app
COPY . .

# Обновляем pip и устанавливаем build tools
RUN pip install --upgrade pip setuptools wheel

# Ставим зависимости
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8050
CMD ["python", "app.py"]