FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc g++ curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project
COPY . .

# Pre-download FinBERT model into image so container runs offline
RUN python -c "from transformers import AutoTokenizer, AutoModelForSequenceClassification; \
    AutoTokenizer.from_pretrained('yiyanghkust/finbert-tone'); \
    AutoModelForSequenceClassification.from_pretrained('yiyanghkust/finbert-tone')"

ENV TRANSFORMERS_OFFLINE=1
ENV HF_HUB_OFFLINE=1
ENV PYTHONUNBUFFERED=1

EXPOSE 8080

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8080"]
