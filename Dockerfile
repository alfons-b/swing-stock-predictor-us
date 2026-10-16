# Image untuk job (daily/retrain/backtest) — konsisten di cloud mana pun. TIDAK berisi secrets:
# credentials masuk lewat environment saat container dijalankan.
FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 TZ=America/New_York
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app/ app/
COPY config/ config/
COPY sql/ sql/
COPY main.py .
RUN useradd -m runner && mkdir -p data/cache data/output reports models && chown -R runner /app
USER runner
ENTRYPOINT ["python", "main.py"]
CMD ["daily"]
