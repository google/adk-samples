FROM python:3.13-slim

WORKDIR /app

# Install system deps for pdfplumber (poppler)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (cache layer)
COPY invoice_processing/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt uvicorn fastapi requests

# Copy application code
COPY webhook_server.py /app/webhook_server.py
COPY invoice_processing/ /app/invoice_processing/

# Create empty exemplary_data and data dirs (runtime use)
RUN mkdir -p /app/invoice_processing/exemplary_data \
    && mkdir -p /app/invoice_processing/data/agent_output \
    && mkdir -p /app/invoice_processing/data/alf_output

# Cloud Run sets PORT env var
ENV PORT=8080
EXPOSE 8080

# Run webhook server — Cloud Run requires 0.0.0.0
CMD ["python", "webhook_server.py", "--host", "0.0.0.0", "--port", "8080", "--skip-investigation"]
