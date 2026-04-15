FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY . .

# Pre-train models and save artifacts — baked into image so startup is instant
RUN python train.py

ENV PORT=8080
EXPOSE 8080

CMD ["sh", "-c", "streamlit run app.py --server.port=${PORT} --server.address=0.0.0.0 --server.headless=true"]
