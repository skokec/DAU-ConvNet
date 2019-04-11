# DAU-ConvNet #
Displaced Aggregation Units for Convolutional Networks from CVPR 2018 paper titled "Spatially-Adaptive Filter Units for Deep Neural Networks" developed as part of [Deep Compositional Networks](http://www.vicos.si/Research/DeepCompositionalNet).

Self-contained DAU layer implementation (C++ and CUDA). Use this library to implement DAU layers in any deep learning frameworks. More details on DAUs are available on our [ViCoS research page](http://www.vicos.si/Research/DeepCompositionalNet) .

Available implementations :
 * TensorFlow
 * [Caffe](https://github.com/skokec/DAU-ConvNet-caffe)
 
See below for more details on each implementation.

## Citation ##
Please cite our CVPR 2018 paper when using DAU code:

```
@inproceedings{Tabernik2018,
	title = {{Spatially-Adaptive Filter Units for Deep Neural Networks}},
	author = {Tabernik, Domen and Kristan, Matej and Leonardis, Ale{\v{s}}},
	booktitle = {2018 IEEE/CVF Conference on Computer Vision and Pattern Recognition},
	year = {2018}
	pages = {9388--9396}
}
```

## Acknowledgment ##

We thank Vitjan Zavrtanik (VitjanZ) for TensorFlow C++/Python wrapper. 

## Requirements and dependency libraries for DAU-ConvNet ##
 * Ubuntu 16.04 (not tested on other OS and other versions)
 * C++11
 * CMake 2.8 or newer (tested on version 3.5)
 * CUDA SDK Toolkit (tested on version 8.0 and 9.0)
 * BLAS (ATLAS or OpenBLAS)
 * cuBlas
 
# Caffe #
A Caffe implementation based on this library is available in [DAU-ConvNet-caffe](https://github.com/skokec/DAU-ConvNet-caffe) repository. 

Pretrained models for Caffe from CVPR 2018 papers are available:
* [AlexNet-DAU-ConvNet (default)](https://gist.github.com/skokec/d7e1b81b8c2426d411e0b491941b4ef2) (56.9% top-1 accuracy, 0.7 mio DAU units)
* [AlexNet-DAU-ConvNet-small](https://gist.github.com/skokec/c9748b5d7ff99fcce7a20b9a2806004f) (56.4% top-1 accuracy, 0.3 mio DAU units)
* [AlexNet-DAU-ConvNet-large](https://gist.github.com/skokec/d3b97367af569524fb85cf026cf5dcb8) (57.3% top-1 accuracy, 1.5 mio DAU units)


# TensorFlow #

We provide TensorFlow plugin and appropriate Python wrappers that can be used to directly replace the `tf.contrib.layers.conv2d` function. Note, our C++/CUDA code natively supports only NCHW format for input, please update your TensorFlow models to use this format.  

## Additional requirements and dependency libraries for TensorFlow plugin ##
 * Requirements for DAU-ConvNet library (see above)
 * Python (tested on Python2.7 and Python3.5)
 * TensorFlow 1.6 or newer
 * Numpy
 * (optional) Scipy, matplotlib and python-tk  for running unit test in `dau_conv_test.py`
 

## Build and installation ##

On Ubuntu 16.04 with pre-installed CUDA and cuBLAS (e.g. using nvidia/cuda:9.0-cudnn7-devel-ubuntu16.04 or nvidia/cuda:8.0-cudnn5-devel-ubuntu16.04 docker) install dependencies first:

```bash
apt-get update
apt-get install cmake python python-pip libopenblas-dev
 
pip install tensorflow-gpu>=1.6
# Note: during instalation tensorflow package is sufficent, but during running the tensorflow-gpu is required.
```

Then clone the repository and build from source:
```bash
git clone https://github.com/skokec/DAU-ConvNet
git submodule update --init --recursive

mkdir DAU-ConvNet/build
cd DAU-ConvNet/build

cmake -DBLAS=Open -DBUILD_TENSORFLOW_PLUGIN=on ..

make -j # creates whl file in build/plugin/tensorflow/wheelhouse
make install # will install whl package (with .so files) into python dist-packages folder 

```

### Unit test ###

To validate installation using unit tests also install scipy, matplotlib and python-tk, and then run `dau_conv_test.py`:

```bash
apt-get install python-tk                
                
pip install scipy matplotlib

python DAU-ConvNet/plugins/tensorflow/tests/dau_conv_test.py
```

### Common issues ###

__I got `undefined symbol: _ZN9perftools8gputools4cuda17AsCUDAStreamValueEPNS0_6StreamE` when running the code.__

Please make sure that your TensorFlow is compiled against GPU/CUDA. In pip the `tensroflow` and `tensorflow-gpu` packages provide the same libtensorflow_framework.so in the same folder but only `tensorflow-gpu` has the the .so that is compiled against the CUDA. If `tensroflow` gets installed _after_ the `tensorflow-gpu` then .so with CUDA support will be overriden by the .so without it. Make sure to install `tensorflow-gpu` the last or not to install `tensroflow` at all.

## Usage ##

There are two available methods to use our DAU convolution. Using `dau_conv.DAUConv2d` class based on `base.Layer` or using wrapper `dau_conv.dau_conv2d` functions. See below for example on using `dau_conv2d` method.  


Method `dau_conv.dau_conv2d`: 
```python
dau_conv2d(inputs,
             filters, # number of output filters
             dau_units, # number of DAU units per image axis, e.g, (2,2) for 4 DAUs per filter 
             max_kernel_size, # maximal possible size of kernel that limits the offset of DAUs (highest value that can be used=17)  
             stride=1, # only stride=1 supported 
             mu_learning_rate_factor=500, # additional factor for gradients of mu1 and mu2
             data_format=None,
             activation_fn=tf.nn.relu,
             normalizer_fn=None,
             normalizer_params=None,
             weights_initializer=tf.random_normal_initializer(stddev=0.1), 
             weights_regularizer=None,
             mu1_initializer=None, # see below for default initialization values
             mu1_regularizer=None, # see below for default initialization values
             mu2_initializer=None,
             mu2_regularizer=None,
             sigma_initializer=None,
             sigma_regularizer=None,
             biases_initializer=tf.zeros_initializer(),
             biases_regularizer=None,
             reuse=None,
             variables_collections=None,
             outputs_collections=None,
             trainable=True,
             scope=None)
```

Class `dau_conv.DAUConv2d`: 
```python

DAUConv2d(filters, # number of output filters
           dau_units, # number of DAU units per image axis, e.g, (2,2) for 4 DAUs total per one filter
           max_kernel_size, # maximal possible size of kernel that limits the offset of DAUs (highest value that can be used=17)
           strides=1, # only stride=1 supported
           data_format='channels_first', # supports only 'channels_last' 
           activation=None,
           use_bias=True,
           weight_initializer=tf.random_normal_initializer(stddev=0.1),
           mu1_initializer=None, # see below for default initialization values
           mu2_initializer=None, # see below for default initialization values
           sigma_initializer=None,
           bias_initializer=tf.zeros_initializer(),
           weight_regularizer=None,
           mu1_regularizer=None,
           mu2_regularizer=None,
           sigma_regularizer=None,
           bias_regularizer=None,
           activity_regularizer=None,
           weight_constraint=None,
           mu1_constraint=None,
           mu2_constraint=None,
           sigma_constraint=None,
           bias_constraint=None,
           trainable=True,
           mu_learning_rate_factor=500, # additional factor for gradients of mu1 and mu2 
           unit_testing=False, # for competability between CPU and GPU version (where gradients of last edge need to be ignored) during unit testing
           name=None)
```

### Mean initialization (mu1 and mu2 initializers) ### 

Mean values (e.g. learned offsets) of DAU units are always based on (0,0) being at the center of the kernel. Default initialization (when passing None) is to arrange units equally over the available space using `dau_conv.DAUGridMean` initializer class:

```python
    if self.mu1_initializer is None:
        self.mu1_initializer = DAUGridMean(dau_units=self.dau_units, max_value=np.floor(self.max_kernel_size[1]/2.0)-1, dau_unit_axis=2)
        
    if self.mu2_initializer is None:
        self.mu2_initializer = DAUGridMean(dau_units=self.dau_units, max_value=np.floor(self.max_kernel_size[0]/2.0)-1, dau_unit_axis=1)
```

Other TensorFlow initializer classes can be used. For instance distributing them uniformly over the center of the kernel is accomplished by:
```python
dau_conv2d(...
          mu1_initializer = tf.random_uniform_initializer(minval=-np.floor(max_kernel_size/2.0), 
                                                          maxval=np.floor(max_kernel_size/2.0),dtype=tf.float32),
          mu2_initializer = tf.random_uniform_initializer(minval=-np.floor(max_kernel_size/2.0), 
                                                          maxval=np.floor(max_kernel_size/2.0),dtype=tf.float32), 
          ...)
```

Initializer `dau_conv.DAUGridMean` class:
```python
dau_conv.DAUGridMean(dau_units, # number of DAU units per image axis e.g. (2,2) for 4 DAUs total 
                     max_value, # max offset 
                     dau_unit_axis=2) # axis for DAU units in input tensor where 2 => mu1, 1 => mu2, (default=2) 
```

### Limtations and restrictions ###

Current implementation is limited to using only the following settings:
 * `data_format = 'NCHW'`: only 'NCHW' format available in our C++/CUDA implementation
 * `stride = 1`: striding not implemented yet
 * `max_kernel_size <= 65`: due to pre-defined CUDA kernels max offsets are restricted to specific values:
   * `max_kernel_size <= 9` and `max_kernel_size <= 17`: most optimal kernel implementations
   * `max_kernel_size <= 33` and `max_kernel_size <= 65`: less optimal  implementation that have additional computational penalty due to larger memory utilization
   * NOTE: selection of which CUDA kernel is used is performed based on actual offset values at each call so even setting large kernel sizes can be fast if all offset values (in each layer) are smaller than 8 pixels.

### Example of code usage ###

CIFAR-10 example is available [here](https://github.com/skokec/DAU-ConvNet-cifar10-example).

Example of three DAU convolutional layer and one fully connected using batch norm and L2 regularization on weights:

```python
import tensorflow as tf

from tensorflow.contrib.framework import arg_scope

from dau_conv import dau_conv2d

with arg_scope([dau_conv2d, tf.contrib.layers.fully_connected],
                weights_regularizer=tf.contrib.layers.l2_regularizer(0.0005),
                weights_initializer=tf.contrib.layers.xavier_initializer(uniform=False),
                biases_initializer=None,
                normalizer_fn=tf.layers.batch_normalization,
                normalizer_params=dict(center=True,
                                       scale=True,
                                       momentum=0.9999, 
                                       epsilon=0.001, 
                                       axis=1, # NOTE: use axis=1 for NCHW format !!
                                       training=in_training)):
            
            inputs = ...
            
            # convert from NHWC to NCHW format
            inputs = tf.transpose(inputs, [0,3,1,2])
            
            net = dau_conv2d(inputs, 96, dau_units=(2,2), max_kernel_size=9,
                                    mu_learning_rate_factor=500, data_format='NCHW', scope='dau_conv1')
            net = tf.contrib.layers.max_pool2d(net, [2, 2], scope='pool1', data_format="NCHW")

            net = dau_conv2d(net, 96, dau_units=(2,2), max_kernel_size=9,
                                    mu_learning_rate_factor=500, data_format='NCHW', scope='dau_conv2')
            net = tf.contrib.layers(net, [2, 2], scope='pool2', data_format="NCHW")

            net = dau_conv2d(net, 192, dau_units=(2,2), max_kernel_size=9,
                                    mu_learning_rate_factor=500, data_format='NCHW', scope='dau_conv3')
            net = tf.contrib.layers.max_pool2d(net, [2, 2], scope='pool3', data_format="NCHW")
            net = tf.reshape(net, [net.shape[0], -1])

            net = tf.contrib.layers.fully_connected(net, NUM_CLASSES, scope='fc4',
                                                    activation_fn=None,
                                                    normalizer_fn=None,
                                                    biases_initializer=tf.constant_initializer(0))

```

