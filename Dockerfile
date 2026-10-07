FROM python:3.10-slim

WORKDIR /app

# Create non-root application user
RUN useradd -m -u 1000 appuser

# Copy and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code, templates, and essential model artifacts
COPY src/ ./src/
COPY templates/ ./templates/
COPY artifacts/preprocessor.pkl ./artifacts/
COPY artifacts/model.pkl ./artifacts/
COPY artifacts/threshold.json ./artifacts/
COPY artifacts/metrics.json ./artifacts/
COPY app.py .

# Permissions
RUN chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
