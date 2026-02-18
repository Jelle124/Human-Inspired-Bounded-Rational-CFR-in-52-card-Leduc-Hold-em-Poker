# Bluffing by DQN and CFR in Leduc Hold'em - Replication Environment
# Run on CPU (suitable for laptops) - no CUDA required

FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for matplotlib
RUN apt-get update && apt-get install -y --no-install-recommends \
    libxrender1 \
    libsm6 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies first (better layer caching)
# Use CPU-only PyTorch for laptops (~200MB vs ~2GB with CUDA)
COPY requirements.txt .
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# Copy project code
COPY . .

# Ensure custom_leduc_rlcard is importable
ENV PYTHONPATH=/app

# Default: disable wandb in container (override with -e WANDB_MODE=online if desired)
ENV WANDB_MODE=disabled

# Volume for persisting results
VOLUME ["/app/results"]

# Default command: run full pipeline (training -> evaluation -> analysis)
# Override to run individual steps, e.g.:
#   docker run bluffing python simultaneous_training.py
#   docker run bluffing python evaluate_simultaneous.py
#   docker run bluffing python analyze_bluff_ReactionCFR_DQNBluff.py
CMD ["python", "run_pipeline.py"]
