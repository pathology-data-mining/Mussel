FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

ARG BACKEND=torch-gpu
ENV BACKEND=$BACKEND

ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies in a single layer and clean up to reduce size
RUN apt-get update && apt-get install -y \
  build-essential \
  libgdal-dev \
  liblapack-dev \
  libblas-dev \
  gfortran \
  libgl1 \
  libgl1-mesa-dev \
  ffmpeg \
  libsm6 \
  libxext6 \
  curl \
  zip \
  git \
  ca-certificates \
  && rm -rf /var/lib/apt/lists/*

# Install uv package manager system-wide
RUN curl -LsSf https://astral.sh/uv/install.sh | sh && \
    mv /root/.cargo/bin/uv /usr/local/bin/ || mv /root/.local/bin/uv /usr/local/bin/

# Install AWS CLI
RUN curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip" && \
    unzip awscliv2.zip && \
    ./aws/install && \
    rm -rf awscliv2.zip aws

# Set working directory
WORKDIR /code/mussel

# Copy all application code
COPY . .

# Install dependencies and package in one step
# This ensures the virtual environment is created with all dependencies including the editable package
RUN uv sync --frozen --extra $BACKEND

# Create cache directory with wide permissions so any user can write to it
RUN mkdir -p /.cache && chmod 777 /.cache
