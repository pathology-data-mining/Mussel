FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

ARG BACKEND=torch-gpu
ENV BACKEND=$BACKEND

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install \
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
  git -y

# Install uv package manager
# Download and install uv with proper SSL handling
RUN apt-get update && apt-get install -y ca-certificates && \
    curl -LsSf https://astral.sh/uv/install.sh -o /tmp/install.sh && \
    sh /tmp/install.sh && rm /tmp/install.sh

# Ensure the installed binary is on the `PATH`
ENV PATH="/root/.local/bin/:$PATH"

RUN curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
RUN unzip awscliv2.zip
RUN ./aws/install

# Set working directory
WORKDIR /code/mussel

# Copy only dependency files first to leverage Docker cache
COPY pyproject.toml uv.lock ./

# Install dependencies (this layer will be cached if dependencies don't change)
RUN uv sync --frozen --extra $BACKEND

# Copy the rest of the application code
COPY . .
