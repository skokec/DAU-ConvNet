#!/bin/bash

cd ~/Documents/5.letnik/gauss_conv/new_impl/DAU-ConvNet/build
cmake ..
make
cp ./plugins/tensorflow/*.so ../plugins/tensorflow/bin/
