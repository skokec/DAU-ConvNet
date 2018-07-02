# DAU-ConvNet
Displaced Aggregation Units for Convolutional Networks from CVPR 2018 paper titled "Spatially-Adaptive Filter Units for Deep Neural Networks"

Self-contained DAU layer implementation (C++ and CUDA). Use this library to implement DAU layers in any deep learning frameworks.

## Caffe ##
We currently provide a Caffe implementation based on this code in https://github.com/skokec/DAU-ConvNet-caffe

Pretrained models for Caffe from CVPR 2018 papers can be found here:
* [ AlexNet-DAU-ConvNet (default)](https://gist.github.com/skokec/d7e1b81b8c2426d411e0b491941b4ef2) (56.9% top-1 accuracy, 0.7 mio DAU units)
* [AlexNet-DAU-ConvNet-small](https://gist.github.com/skokec/c9748b5d7ff99fcce7a20b9a2806004f) (56.4% top-1 accuracy, 0.3 mio DAU units)
* [AlexNet-DAU-ConvNet-large](https://gist.github.com/skokec/d3b97367af569524fb85cf026cf5dcb8) (57.3% top-1 accuracy, 1.5 mio DAU units)

## TensorFlow ## 
TensorFlow implementation is coming soon. Work in progress version can be found in https://github.com/VitjanZ/DAU-ConvNet
