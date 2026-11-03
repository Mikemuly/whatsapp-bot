FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .

RUN --mount=type=cache,target=/root/.cache/pip \
       pip install --upgrade pip && \
       pip install --default-timeout=120 --retries 10 -r requirements.txt

COPY app/ ./app/
COPY data/ ./data/
COPY rag.py ./rag.py

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]