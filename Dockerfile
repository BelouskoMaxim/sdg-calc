# Use Python 3.12 with Debian (prebuilt wheels for Pandas 2.0.3)
FROM python:3.12-bullseye

WORKDIR /app
COPY . .

# Upgrade pip
RUN pip install --upgrade pip

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8050
ENV PORT=8050
ENV HOST=0.0.0.0

CMD ["python", "app.py"]