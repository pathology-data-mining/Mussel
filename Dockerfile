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

# Download the latest installer
ADD https://astral.sh/uv/0.6.10/install.sh /uv-installer.sh

# Run the installer then remove it
RUN sh /uv-installer.sh && rm /uv-installer.sh

# Ensure the installed binary is on the `PATH`
ENV PATH="/root/.local/bin/:$PATH"

RUN curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
RUN unzip awscliv2.zip
RUN ./aws/install

COPY . /code/mussel
WORKDIR /code/mussel

RUN uv sync --frozen --extra $BACKEND
