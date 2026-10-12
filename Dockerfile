FROM python:3.12-slim

WORKDIR /app

COPY . /app

RUN pip install --no-cache-dir . 2>/dev/null || true

EXPOSE 8000

ENV PYTHONUNBUFFERED=1

CMD ["python", "-m", "msaforge", "serve", "--host", "0.0.0.0", "--port", "8000"]
