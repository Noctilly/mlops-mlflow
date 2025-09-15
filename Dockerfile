FROM python:3.11-slim

WORKDIR /app

COPY web_service.py /app/web_service.py

RUN pip install --no-cache-dir \
    fastapi==0.111.0 \
    uvicorn[standard]==0.30.0 \
    mlflow==2.14.1 \
    pandas==2.2.2

EXPOSE 8000

CMD ["uvicorn", "web_service:app", "--host", "0.0.0.0", "--port", "8000"]
