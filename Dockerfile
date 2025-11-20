# Stage 1: Builder
FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04 AS builder

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=America/New_York

# Install Python 3.11
RUN apt-get update && apt-get install -y \
  software-properties-common \
  && add-apt-repository ppa:deadsnakes/ppa \
  && apt-get update && apt-get install -y \
  python3.11 \
  python3.11-dev \
  python3.11-distutils \
  curl \
  && rm -rf /var/lib/apt/lists/*

# Set Python 3.11 as default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 \
  && update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

ARG BACKEND=torch-gpu
ENV BACKEND=$BACKEND
ENV UV_SYSTEM_PYTHON=1

# Install build dependencies
RUN apt-get update && apt-get install -y \
  build-essential \
  libgdal-dev \
  liblapack-dev \
  libblas-dev \
  gfortran \
  git \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
RUN --mount=type=cache,target=/root/.cache/uv \
  --mount=type=bind,source=uv.lock,target=uv.lock \
  --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
  uv pip install --system -r pyproject.toml --extra $BACKEND --extra distributed

# Copy and install the project
COPY . /app
RUN --mount=type=cache,target=/root/.cache/uv \
  uv pip install --system . --no-deps --force-reinstall

# Stage 2: Runtime
FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04
# Set TMPDIR to use more spacious location (not /tmp which is small)
ENV TMPDIR=/mnt/batch/tasks/workitems/tmp

ENV DEBIAN_FRONTEND=noninteractive

# Install Python 3.11
RUN apt-get update && apt-get install -y \
  software-properties-common \
  && add-apt-repository ppa:deadsnakes/ppa \
  && apt-get update && apt-get install -y \
  python3.11 \
  python3.11-distutils \
  curl \
  && rm -rf /var/lib/apt/lists/*

# Set Python 3.11 as default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 \
  && update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1

# Install only runtime dependencies
RUN apt-get update && apt-get install -y \
  libgdal30 \
  liblapack3 \
  libblas3 \
  libgfortran5 \
  libgl1 \
  ffmpeg \
  libsm6 \
  libxext6 \
  curl \
  ca-certificates \
  sudo \
  unzip \
  && rm -rf /var/lib/apt/lists/*

# Install AWS CLI (slim version)
RUN curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip" && \
  unzip awscliv2.zip && \
  ./aws/install && \
  rm -rf awscliv2.zip aws

# Install Azure CLI
RUN curl -sL https://aka.ms/InstallAzureCLIDeb | bash

# Install gosu
RUN curl -fsSL "https://github.com/tianon/gosu/releases/download/1.17/gosu-$(dpkg --print-architecture)" -o /usr/local/bin/gosu && \
  chmod +x /usr/local/bin/gosu && \
  gosu nobody true

# Copy Python packages from builder (Ubuntu packages install to /usr/lib)
COPY --from=builder /usr/lib/python3.11 /usr/lib/python3.11
COPY --from=builder /usr/local/lib/python3.11 /usr/local/lib/python3.11
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy only necessary application files (not the entire /app directory)
WORKDIR /app
# Copy only the installed package from site-packages, not all source files
RUN mkdir -p /app

# Copy scripts directory for Azure Batch
COPY scripts /app/scripts
RUN chmod +x /app/scripts/azure_batch/*.sh

COPY entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["python", "-c", "print('Mussel container ready. Use mussel-docker <command>')"]
