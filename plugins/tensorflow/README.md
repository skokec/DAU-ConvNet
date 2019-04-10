# DAU-ConvNet #
Displaced Aggregation Units for Convolutional Networks from CVPR 2018 paper titled "Spatially-Adaptive Filter Units for Deep Neural Networks" developed as part of [Deep Compositional Networks](http://www.vicos.si/Research/DeepCompositionalNet).

More details on DAUs are available on our [ViCoS research page](http://www.vicos.si/Research/DeepCompositionalNet) .

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

## Installation ##

You can install dau-conv package from our local PyPi repository:

```bash
pip install dau_conv
```

## Requirements and dependency libraries for DAU-ConvNet ##
 * Ubuntu 16.04 (not tested on other OS and other versions)
 * CUDA SDK Toolkit (tested on version 8.0, 9.0 and 10)
 * BLAS (ATLAS or OpenBLAS)
 * cuBlas
 * TensorFlow

```bash
apt-get install open
```


## Example of code usage ##

See our [Github repository](https://github.com/skokec/DAU-ConvNet) for more detailed information.

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
