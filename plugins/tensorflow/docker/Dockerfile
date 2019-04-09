ARG BASE_CUDA_VERSION=nvidia/cuda:10.0-cudnn7-devel-ubuntu16.04
FROM $BASE_CUDA_VERSION

LABEL maintainer "domen.tabernik@fri.uni-lj.si"

# TF/PY version argument must be after FROM statement
ARG TF_VER=1.13.1
ARG PY_VER=3.5
ARG PY_VER_MAJOR=3

ENV PYTHON "python$PY_VER"
ENV PYTHON_MAJOR "python$PY_VER_MAJOR"

ENV LD_LIBRARY_PATH "/usr/local/nvidia/lib:/usr/local/nvidia/lib64"
ENV DAU_CONVNET_HOME /opt/dau-convnet

WORKDIR $DAU_CONVNET_HOME

RUN echo "Using TensorFlow==$TF_VER"
RUN echo "Using python binary path=$PYTHON"

# Install general packages for building 
RUN apt-get update && \
    apt-get install -y software-properties-common \
		       sudo \
		       build-essential \
	cmake \
        build-essential \
        curl \
        git \
        libcurl3-dev \
        libfreetype6-dev \
        libpng12-dev \
        libzmq3-dev \
        pkg-config \
        rsync \
        software-properties-common \
        unzip \
        zip \
        zlib1g-dev \
	libopenblas-dev

# Install specific python and tensorflow versions  
RUN apt-get install -y $PYTHON \
                       $PYTHON-dev \
                       $PYTHON_MAJOR-pip && \
    apt-get clean

RUN $PYTHON -m pip --no-cache-dir install numpy
RUN $PYTHON -m pip install tensorflow==$TF_VER

# NOTE: since docker build does not provide nvidia drivers we cannot run "import tensorflow"
# using tensorflow-gpu so we only use CPU tensorflow during build-time and then install 
# tensorflow-gpu after DAU-ConvNet is compiled

# Download and build DAU-ConvNet plugin
RUN git clone https://github.com/skokec/DAU-ConvNet . &&  \
    git submodule update --init --recursive
   
RUN mkdir build && cd build && \
    cmake -DBLAS=Open -DBUILD_TENSORFLOW_PLUGIN=on -DPYTHON_EXECUTABLE="/usr/bin/$PYTHON" .. && \
    make -j install

# We need to install back GPU support for tensorflow 
RUN $PYTHON -m pip install tensorflow-gpu==$TF_VER


# Install two scripts that will verify integrity of build with tests
COPY verify_dau_import.py /opt/verify_dau_import.py
COPY test_dau.sh /opt/test_dau.sh

