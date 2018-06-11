#!/bin/bash

BUILD_DIR=~/Documents/5.letnik/gauss_conv/new_impl/VitjanZ/DAU-ConvNet/build/
if [ ! -d $BUILD_DIR ]; then
 mkdir $BUILD_DIR
fi

cd $BUILD_DIR
cmake ..
make
cp ./plugins/tensorflow/*.so ../plugins/tensorflow/bin/
