# Base image with CUDA support for GPU acceleration
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04 as base

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3-pip \
    python3-dev \
    git \
    wget \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Create a working directory
WORKDIR /app

# Create Python virtual environment
RUN python3 -m pip install --upgrade pip setuptools wheel

# Install DeepSeek-specific dependencies
RUN pip install --no-cache-dir \
    torch==2.0.1 \
    transformers==4.34.0 \
    accelerate==0.23.0 \
    bitsandbytes==0.41.1 \
    deepseek-ai==0.1.0 \
    peft==0.5.0

# Install HIM core dependencies
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Stage for development
FROM base as development
RUN pip install --no-cache-dir \
    pytest \
    black \
    isort \
    flake8 \
    mypy \
    jupyter \
    notebook

# Stage for training
FROM base as training
ENV TRAINING_MODE=1
RUN pip install --no-cache-dir \
    wandb \
    tensorboard \
    pytorch-lightning==2.0.8

# Stage for web interface
FROM base as web
RUN pip install --no-cache-dir \
    streamlit==1.25.0 \
    gradio==3.35.0 \
    plotly==5.16.0

# Copy application code
COPY . /app/

# Set Python path
ENV PYTHONPATH=/app

# Entry point script
COPY docker-entrypoint.sh /app/
RUN chmod +x /app/docker-entrypoint.sh

ENTRYPOINT ["/app/docker-entrypoint.sh"]

