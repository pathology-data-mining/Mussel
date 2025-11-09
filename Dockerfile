FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

# Install uv directly
RUN apt-get update && apt-get install -y curl && \
  curl -LsSf https://astral.sh/uv/install.sh | sh && \
  mv /root/.local/bin/uv /usr/local/bin/ && \
  mv /root/.local/bin/uvx /usr/local/bin/ && \
  rm -rf /var/lib/apt/lists/*

ARG BACKEND=torch-gpu
ENV BACKEND=$BACKEND

ENV UV_SYSTEM_PYTHON=1

ENV DEBIAN_FRONTEND=noninteractive

# Install Python 3.10 (default for Ubuntu 22.04) and system dependencies
# Skip openssh-client post-install errors (known issue on network file systems)
RUN apt-get update && \
  (apt-get install -y --no-install-recommends \
  python3 \
  python3-dev \
  python3-pip \
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
  vim-tiny || true) && \
  dpkg --configure -a && \
  rm -rf /var/lib/apt/lists/*


# Install AWS CLI (need unzip first, may have been skipped)
RUN apt-get update && apt-get install -y unzip && rm -rf /var/lib/apt/lists/* && \
  curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip" && \
  unzip awscliv2.zip && \
  ./aws/install && \
  rm -rf awscliv2.zip aws

RUN curl -fsSL "https://github.com/tianon/gosu/releases/download/1.17/gosu-$(dpkg --print-architecture)" -o /usr/local/bin/gosu && \
  chmod +x /usr/local/bin/gosu || true

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
