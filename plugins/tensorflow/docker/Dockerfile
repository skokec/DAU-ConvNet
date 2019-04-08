FROM nvidia/cuda:10.0-cudnn7-devel-ubuntu16.04

LABEL maintainer "domen.tabernik@fri.uni-lj.si"

ENV TF_VER 1.13.1

ENV LD_LIBRARY_PATH "/usr/local/nvidia/lib:/usr/local/nvidia/lib64"
ENV DAU_CONVNET_HOME /opt/dau-convnet

WORKDIR $DAU_CONVNET_HOME

RUN apt-get update && \
    apt-get install -y software-properties-common \
		       sudo \
		       build-essential \
		       python \
		       python-pip \
        build-essential \
        curl \
        git \
        libcurl3-dev \
        libfreetype6-dev \
        libpng12-dev \
        libzmq3-dev \
        pkg-config \
        python-dev \
        rsync \
        software-properties-common \
        unzip \
        zip \
        zlib1g-dev && \
    apt-get clean

RUN pip --no-cache-dir install \
        ipykernel \        
        matplotlib \
        numpy \
        scipy \
        sklearn \
        pandas \
        libopenblas-dev && \
    python -m ipykernel.kernelspec

RUN pip install tensorflow==$TF_VER

# HACK: since docker build does not provide nvidia drivers we cannot run "import tensorflow"
# instead we remove GPU support for tensorflow and install only CPU support 
# then we reinstall it after DAU-ConvNet is compiled
RUN pip uninstall -y tensorflow tensorflow-gpu
RUN pip install tensorflow==$TF_VER

# Download and build DAU-ConvNet plugin
RUN git clone https://github.com/skokec/DAU-ConvNet . &&  \
    git submodule update --init --recursive
   
RUN mkdir build && cd build && \
    cmake -DBLAS=Open -DBUILD_TENSORFLOW_PLUGIN=on .. && \
    make -j install

# We need to install back GPU support for tensorflow 
RUN pip install tensorflow-gpu==$TF_VER
    
