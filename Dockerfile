# NOTE: To build this you will need a docker version >= 19.03 and DOCKER_BUILDKIT=1
#
#       If you do not use buildkit you are not going to have a good time
#
#       For reference:
#           https://docs.docker.com/develop/develop-images/build_enhancements/

ARG UBUNTU_VERSION=22.04

FROM ubuntu:${UBUNTU_VERSION}

# See http://bugs.python.org/issue19846
ENV LANG C.UTF-8
ARG PYTHON=python3

RUN apt-get update -y && \
    apt-get upgrade -y && \
    apt-get install -y --no-install-recommends --fix-missing \
    sudo \
    numactl \
    wget \
    libjemalloc-dev \
    ${PYTHON} \
    ${PYTHON}-pip

RUN ${PYTHON} -m pip --no-cache-dir install --upgrade \
    pip \
    setuptools \
    psutil

# Some TF tools expect a "python" binary
RUN ln -s $(which ${PYTHON}) /usr/local/bin/python

ARG IPEX_VERSION=2.1.100
ARG PYTORCH_VERSION=2.1.1
ARG TORCHAUDIO_VERSION=2.1.1
ARG TORCHVISION_VERSION=0.16.1
ARG TORCH_CPU_URL=https://download.pytorch.org/whl/cpu/torch_stable.html

RUN \
    python -m pip install --no-cache-dir \
    torch==${PYTORCH_VERSION}+cpu torchvision==${TORCHVISION_VERSION}+cpu torchaudio==${TORCHAUDIO_VERSION}+cpu -f ${TORCH_CPU_URL} && \
    python -m pip install --no-cache-dir \
    intel_extension_for_pytorch==${IPEX_VERSION}

RUN python -m pip --no-cache-dir install --upgrade \
    diffusers \
    transformers \
    accelerate \
    mkl \
    intel-openmp

RUN useradd -m ubuntu
RUN echo 'ubuntu ALL=(ALL) NOPASSWD: ALL' >> /etc/sudoers
USER ubuntu
WORKDIR /home/ubuntu
RUN mkdir /home/ubuntu/sdxlturbo
COPY sdxlturbo /home/ubuntu/sdxlturbo
ENV KMP_BLOCKTIME=1
ENV KMP_AFFINITY=granularity=fine,compact,1,0
ENV LD_PRELOAD=$LD_PRELOAD:/usr/local/lib/libiomp5.so
ENV LD_PRELOAD=$LD_PRELOAD:/usr/lib/x86_64-linux-gnu/libjemalloc.so.2
ENV MALLOC_CONF="oversize_threshold:1,background_thread:true,metadata_thp:auto,dirty_decay_ms:9000000000,muzzy_decay_ms:9000000000"
