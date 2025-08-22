# Use full Debian Python image (includes build tools)
FROM python:3.12-bullseye

WORKDIR /app
COPY . .

RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8050
ENV PORT=8050
ENV HOST=0.0.0.0

CMD ["python", "app.py"]
