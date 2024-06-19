FROM mambaorg/micromamba:1-focal-cuda-11.7.1

ARG YOUR_ENV

ENV YOUR_ENV=${YOUR_ENV} \
             PYTHONFAULTHANDLER=1 \
             PYTHONUNBUFFERED=1 \
             PYTHONHASHSEED=random \
             PIP_NO_CACHE_DIR=off \
             PIP_DISABLE_PIP_VERSION_CHECK=on \
             PIP_DEFAULT_TIMEOUT=100

USER root

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
  libxext6 -y

USER $MAMBA_USER

COPY --chown=$MAMBA_USER:$MAMBA_USER conda-lock.yml /code/
WORKDIR /code
RUN micromamba install -y -n base -c conda-forge conda-lock && \
    micromamba clean --all --yes
ARG MAMBA_DOCKERFILE_ACTIVATE=1  # (otherwise python will not be found)

RUN conda-lock install -p /opt/conda/

COPY --chown=$MAMBA_USER:$MAMBA_USER . /code

RUN pip install --no-deps .
