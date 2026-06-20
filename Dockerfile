# Bluffing by DQN and CFR in Leduc Hold'em - Thesis Environment
FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libxrender1 libsm6 libxext6 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PYTHONPATH=/app
ENV WANDB_MODE=disabled

VOLUME ["/app/results"]

CMD ["python", "train_parallel_suite.py", "--all", "--seed", "42"]
