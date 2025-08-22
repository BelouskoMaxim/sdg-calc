# Use Python 3.12 instead of 3.13
FROM python:3.12-slim

WORKDIR /app
COPY . .

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8050

ENV PORT=8050
ENV HOST=0.0.0.0

CMD ["python", "app.py"]
