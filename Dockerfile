#FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04
FROM python:3.11-slim
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

ARG BACKEND=torch-gpu
ENV BACKEND=$BACKEND

ENV UV_SYSTEM_PYTHON=1

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
  sudo \
  vim-tiny \
  && rm -rf /var/lib/apt/lists/*


# Install AWS CLI
RUN curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip" && \
  unzip awscliv2.zip && \
  ./aws/install && \
  rm -rf awscliv2.zip aws

RUN curl -fsSL "https://github.com/tianon/gosu/releases/download/1.17/gosu-$(dpkg --print-architecture)" -o /usr/local/bin/gosu && \
  chmod +x /usr/local/bin/gosu && \
  gosu nobody true

# Set working directory
WORKDIR /app

# Install dependencies
RUN --mount=type=cache,target=/root/.cache/uv \
  --mount=type=bind,source=uv.lock,target=uv.lock \
  --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
  uv pip install --system -r pyproject.toml --extra $BACKEND

# Copy the project into the image
ADD . /app

# Sync the project
RUN --mount=type=cache,target=/root/.cache/uv \
  uv pip install --system . --no-deps --force-reinstall 

COPY entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

# Set entrypoint to handle user permissions
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]

# Default command
CMD ["python", "-c", "print('Mussel container ready. Use mussel-docker <command>')"]
