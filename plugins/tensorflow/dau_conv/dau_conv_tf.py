import os
import numpy as np
import tensorflow as tf

from tensorflow.python.layers import base
from tensorflow.python.layers import utils

from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape

from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import init_ops

class DAUConv2dTF(base.Layer):

    # C++/CUDA implementation will compute N units at the same time - enforce this constraint !!
    DAU_UNITS_GROUP = 2

    def __init__(self, filters,
                 dau_units,
                 max_kernel_size,
                 strides=1,
                 data_format='channels_first',
                 activation=None,
                 use_bias=True,
                 weight_initializer=init_ops.random_normal_initializer(stddev=0.1),
                 mu1_initializer=None,
                 mu2_initializer=None,
                 sigma_initializer=None,
                 bias_initializer=init_ops.zeros_initializer(),
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
                 mu_learning_rate_factor=500,
                 dau_unit_border_bound=0.01,
                 dau_unit_single_dim=False,
                 dau_aggregation_forbid_positive_dim1=False,
                 dau_sigma_trainable=False,
                 dau_mu_interpolation=True,
                 unit_testing=False, # for compatibility between CPU and GPU version (where gradients of last edge need to be ignored) during unit testing
                 name=None,
                 **kwargs):
        super(DAUConv2dTF, self).__init__(trainable=trainable, name=name,
                                    activity_regularizer=activity_regularizer,
                                    **kwargs)
        self.rank = 2
        self.filters = filters
        self.dau_units = utils.normalize_tuple(dau_units, self.rank, 'dau_components')
        self.max_kernel_size = max_kernel_size
        self.padding = np.floor(self.max_kernel_size/2.0)
        self.strides = strides
        self.data_format = utils.normalize_data_format(data_format)
        self.activation = activation
        self.use_bias = use_bias
        self.bias_initializer = bias_initializer
        self.bias_regularizer = bias_regularizer
        self.bias_constraint = bias_constraint

        self.weight_initializer = weight_initializer
        self.weight_regularizer = weight_regularizer
        self.weight_constraint = weight_constraint

        self.mu1_initializer = mu1_initializer
        self.mu1_regularizer = mu1_regularizer
        self.mu1_constraint = mu1_constraint

        self.mu2_initializer = mu2_initializer
        self.mu2_regularizer = mu2_regularizer
        self.mu2_constraint = mu2_constraint

        self.sigma_initializer = sigma_initializer
        self.sigma_regularizer = sigma_regularizer
        self.sigma_constraint = sigma_constraint

        if self.mu1_initializer is None:
            raise Exception("Must initialize MU1")
        if self.mu2_initializer is None:
            raise Exception("Must initialize MU2")

        if self.sigma_initializer is None:
            self.sigma_initializer=init_ops.constant_initializer(0.5)

        self.mu_learning_rate_factor = mu_learning_rate_factor

        self.input_spec = base.InputSpec(ndim=self.rank + 2)

        self.dau_unit_border_bound = dau_unit_border_bound
        self.num_dau_units_all = np.int32(np.prod(self.dau_units))

        self.dau_weights = None
        self.dau_mu1 = None
        self.dau_mu2 = None
        self.dau_sigma = None

        self.dau_sigma_trainable = dau_sigma_trainable

    def set_dau_variables_manually(self, w = None, mu1 = None, mu2 = None, sigma = None):
        """ Manually set w,mu1,mu2 and/or sigma variables with custom tensor. Call before build() or __call__().
        The shape must match the expecated shape as returned by the get_dau_variable_shape(input_shape)
        otherwise the build() function will fail."""

        if w is not None:
            self.dau_weights = w

        if mu1 is not None:
            self.dau_mu1 = mu1

        if mu2 is not None:
            self.dau_mu2 = mu2

        if sigma is not None:
            self.dau_sigma = sigma

    def _get_input_channel_axis(self):
        if self.data_format == 'channels_first':
            channel_axis = 1
        else:
            channel_axis = -1

        return channel_axis

    def _get_input_channels(self, input_shape):
        channel_axis = self._get_input_channel_axis()

        if input_shape[channel_axis].value is None:
            raise ValueError('The channel dimension of the inputs '
                             'should be defined. Found `None`.')

        return input_shape[channel_axis].value

    def get_dau_variable_shape(self, input_shape):
        # get input
        num_input_channels = self._get_input_channels(input_shape)

        dau_params_shape_ = (num_input_channels, self.dau_units[0], self.dau_units[1], self.filters)
        dau_params_shape = (1, num_input_channels, self.num_dau_units_all, self.filters)

        return dau_params_shape

    def add_dau_weights_var(self, input_shape):
        dau_params_shape = self.get_dau_variable_shape(input_shape)
        return self.add_variable(name='weights',
                                 shape=dau_params_shape,
                                 initializer=self.weight_initializer,
                                 regularizer=self.weight_regularizer,
                                 constraint=self.weight_constraint,
                                 trainable=True,
                                 dtype=self.dtype)

    def add_dau_mu1_var(self, input_shape):
        dau_params_shape = self.get_dau_variable_shape(input_shape)
        mu1_var = self.add_variable(name='mu1',
                                 shape=dau_params_shape,
                                 initializer=self.mu1_initializer,
                                 regularizer=self.mu1_regularizer,
                                 constraint=self.mu1_constraint,
                                 trainable=True,
                                 dtype=self.dtype)

        # limit max offset based on self.dau_unit_border_bound and kernel size
        mu1_var = tf.minimum(tf.maximum(mu1_var,
                                        -(self.max_kernel_size - self.dau_unit_border_bound)),
                             self.max_kernel_size - self.dau_unit_border_bound)

        return mu1_var



    def add_dau_mu2_var(self, input_shape):
        dau_params_shape = self.get_dau_variable_shape(input_shape)
        mu2_var = self.add_variable(name='mu2',
                                   shape=dau_params_shape,
                                   initializer=self.mu2_initializer,
                                   regularizer=self.mu2_regularizer,
                                   constraint=self.mu2_constraint,
                                   trainable=True,
                                   dtype=self.dtype)


        # limit max offset based on self.dau_unit_border_bound and kernel size
        mu2_var = tf.minimum(tf.maximum(mu2_var,
                                        -(self.max_kernel_size - self.dau_unit_border_bound)),
                             self.max_kernel_size - self.dau_unit_border_bound)

        return mu2_var
    def add_dau_sigma_var(self, input_shape, trainable=False):
        dau_params_shape = self.get_dau_variable_shape(input_shape)

        # create single sigma variable
        sigma_var = self.add_variable(name='sigma',
                                      shape=dau_params_shape,
                                      initializer=self.sigma_initializer,
                                      regularizer=self.sigma_regularizer,
                                      constraint=self.sigma_constraint,
                                      trainable=self.dau_sigma_trainable,
                                      dtype=self.dtype)

        # but make variable shared across all channels as required for the efficient DAU implementation
        return sigma_var


    def add_bias_var(self):
        return self.add_variable(name='bias',
                                 shape=(self.filters,),
                                 initializer=self.bias_initializer,
                                 regularizer=self.bias_regularizer,
                                 constraint=self.bias_constraint,
                                 trainable=True,
                                 dtype=self.dtype)

    def build(self, input_shape):
        input_shape = tensor_shape.TensorShape(input_shape)

        dau_params_shape = self.get_dau_variable_shape(input_shape)
        if self.dau_weights is None:
            self.dau_weights = self.add_dau_weights_var(input_shape)
        elif np.any(self.dau_weights.shape != dau_params_shape):
            raise ValueError('Shape mismatch for variable `dau_weights`')
        if self.dau_mu1 is None:
            self.dau_mu1 = self.add_dau_mu1_var(input_shape)
        elif np.any(self.dau_mu1.shape != dau_params_shape):
            raise ValueError('Shape mismatch for variable `dau_mu1`')

        if self.dau_mu2 is None:
            self.dau_mu2 = self.add_dau_mu2_var(input_shape)
        elif np.any(self.dau_mu2.shape != dau_params_shape):
            raise ValueError('Shape mismatch for variable `dau_mu2`')
        if self.dau_sigma is None:
            self.dau_sigma = self.add_dau_sigma_var(input_shape, trainable=self.dau_sigma_trainable)
        elif np.any(self.dau_sigma.shape != dau_params_shape):
            raise ValueError('Shape mismatch for variable `dau_sigma`')

        if self.use_bias:
            self.bias = self.add_bias_var()
        else:
            self.bias = None

        input_channel_axis = self._get_input_channel_axis()
        num_input_channels = self._get_input_channels(input_shape)

        self.input_spec = base.InputSpec(ndim=self.rank + 2,
                                         axes={input_channel_axis: num_input_channels})

        kernel_shape = tf.TensorShape((self.max_kernel_size, self.max_kernel_size, num_input_channels, self.filters))

        self._convolution_op = nn_ops.Convolution(
            input_shape,
            filter_shape=kernel_shape,
            dilation_rate=(1,1),
            strides=(self.strides,self.strides),
            padding="SAME",
            data_format=utils.convert_data_format(self.data_format,
                                                  self.rank + 2))
        self.built = True

    def call(self, inputs):

        def get_kernel_fn(dau_w, dau_mu1, dau_mu2, dau_sigma, max_kernel_size):

            [X,Y] = np.meshgrid(np.arange(max_kernel_size),np.arange(max_kernel_size))

            X = np.reshape(X,(max_kernel_size*max_kernel_size,1,1,1)) - int(max_kernel_size/2)
            Y = np.reshape(Y,(max_kernel_size*max_kernel_size,1,1,1)) - int(max_kernel_size/2)

            X = X.astype(np.float32)
            Y = Y.astype(np.float32)

            # Gaussian kernel

            X = tf.convert_to_tensor(X,name='X',dtype=tf.float32)
            Y = tf.convert_to_tensor(Y,name='Y',dtype=tf.float32)

            gauss_kernel = tf.exp(-1* (tf.pow(X - dau_mu1,2.0) + tf.pow(Y - dau_mu2,2.0)) / (2.0*tf.pow(dau_sigma,2.0)),name='gauss_kernel')

            gauss_kernel_sum = tf.reduce_sum(gauss_kernel,axis=0, keep_dims=True,name='guass_kernel_sum')

            gauss_kernel_norm = tf.divide(gauss_kernel, gauss_kernel_sum ,name='gauss_kernel_norm')

            # normalize to sum of 1 and add weight
            gauss_kernel_norm = tf.multiply(dau_w, gauss_kernel_norm,name='gauss_kernel_weight')

            # sum over Gaussian units
            gauss_kernel_norm = tf.reduce_sum(gauss_kernel_norm, axis=2, keep_dims=True,name='gauss_kernel_sum_units')

            # convert to [Kw,Kh,S,F] shape
            gauss_kernel_norm = tf.reshape(gauss_kernel_norm, (max_kernel_size, max_kernel_size, gauss_kernel_norm.shape[1], gauss_kernel_norm.shape[3]),name='gauss_kernel_reshape')

            return gauss_kernel_norm

        try:
            # try with XLA if exists
            from tensorflow.contrib.compiler import xla

            gauss_kernel_norm = xla.compile(computation=get_kernel_fn, inputs=(self.dau_weights, self.dau_mu1, self.dau_mu2, self.dau_sigma, self.max_kernel_size))[0]

        except:
            # otherwise revert to direct method call
            gauss_kernel_norm = get_kernel_fn(self.dau_weights, self.dau_mu1, self.dau_mu2, self.dau_sigma, self.max_kernel_size)

        outputs = self._convolution_op(inputs, gauss_kernel_norm)

        if self.use_bias:
            if self.data_format == 'channels_first':
                if self.rank == 1:
                    # nn.bias_add does not accept a 1D input tensor.
                    bias = array_ops.reshape(self.bias, (1, self.filters, 1))
                    outputs += bias
                if self.rank == 2:
                    outputs = nn.bias_add(outputs, self.bias, data_format='NCHW')
                if self.rank == 3:
                    # As of Mar 2017, direct addition is significantly slower than
                    # bias_add when computing gradients. To use bias_add, we collapse Z
                    # and Y into a single dimension to obtain a 4D input tensor.
                    outputs_shape = outputs.shape.as_list()
                    if outputs_shape[0] is None:
                        outputs_shape[0] = -1
                    outputs_4d = array_ops.reshape(outputs,
                                                   [outputs_shape[0], outputs_shape[1],
                                                    outputs_shape[2] * outputs_shape[3],
                                                    outputs_shape[4]])
                    outputs_4d = nn.bias_add(outputs_4d, self.bias, data_format='NCHW')
                    outputs = array_ops.reshape(outputs_4d, outputs_shape)
            else:
                outputs = nn.bias_add(outputs, self.bias, data_format='NHWC')

        if self.activation is not None:
            return self.activation(outputs)
        return outputs

    def compute_output_shape(self, input_shape):
        input_shape = tensor_shape.TensorShape(input_shape).as_list()
        if self.data_format == 'channels_last':
            space = input_shape[1:-1]
            new_space = []
            for i in range(len(space)):
                new_dim = utils.conv_output_length(
                    space[i],
                    self.max_kernel_size[i],
                    padding=self.padding,
                    stride=self.strides[i],
                    dilation=1)
                new_space.append(new_dim)
            return tensor_shape.TensorShape([input_shape[0]] + new_space +
                                            [self.filters])
        else:
            space = input_shape[2:]
            new_space = []
            for i in range(len(space)):
                new_dim = utils.conv_output_length(
                    space[i],
                    self.kernel_size[i],
                    padding=self.padding,
                    stride=self.strides,
                    dilation=1)
                new_space.append(new_dim)
            return tensor_shape.TensorShape([input_shape[0], self.filters] +
                                            new_space)


from tensorflow.contrib.framework.python.ops import add_arg_scope
from tensorflow.python.ops import variable_scope
from tensorflow.contrib.layers.python.layers import layers as layers_contrib
from tensorflow.contrib.layers.python.layers import utils as utils_contrib

@add_arg_scope
def dau_conv2d_tf(inputs,
             filters,
             dau_units,
             max_kernel_size,
             stride=1,
             mu_learning_rate_factor=500,
             data_format=None,
             activation_fn=nn.relu,
             normalizer_fn=None,
             normalizer_params=None,
             weights_initializer=init_ops.random_normal_initializer(stddev=0.1), #init_ops.glorot_uniform_initializer(),
             weights_regularizer=None,
             weights_constraint=None,
             mu1_initializer=None,
             mu1_regularizer=None,
             mu1_constraint=None,
             mu2_initializer=None,
             mu2_regularizer=None,
             mu2_constraint=None,
             sigma_initializer=None,
             sigma_regularizer=None,
             sigma_constraint=None,
             biases_initializer=init_ops.zeros_initializer(),
             biases_regularizer=None,
             biases_constraint=None,
             dau_unit_border_bound=0.01,
             dau_sigma_trainable=False,
             dau_mu_interpolation=True,
             reuse=None,
             variables_collections=None,
             outputs_collections=None,
             trainable=True,
             scope=None):

    if data_format not in [None, 'NWC', 'NCW', 'NHWC', 'NCHW', 'NDHWC', 'NCDHW']:
        raise ValueError('Invalid data_format: %r' % (data_format,))

    layer_variable_getter = layers_contrib._build_variable_getter({
        'bias': 'biases',
        'weight': 'weights',
        'mu1': 'mu1',
        'mu2': 'mu2',
        'sigma': 'sigma'
    })

    with variable_scope.variable_scope(
            scope, 'DAUConv', [inputs], reuse=reuse,
            custom_getter=layer_variable_getter) as sc:
        inputs = ops.convert_to_tensor(inputs)
        input_rank = inputs.get_shape().ndims

        if input_rank != 4:
            raise ValueError('DAU convolution not supported for input with rank',
                             input_rank)

        df = ('channels_first'
              if data_format and data_format.startswith('NC') else 'channels_last')

        layer = DAUConv2dTF(filters,
                          dau_units,
                          max_kernel_size,
                          strides=stride,
                          data_format=df,
                          activation=None,
                          use_bias=not normalizer_fn and biases_initializer,
                          mu_learning_rate_factor=mu_learning_rate_factor,
                          weight_initializer=weights_initializer,
                          mu1_initializer=mu1_initializer,
                          mu2_initializer=mu2_initializer,
                          sigma_initializer=sigma_initializer,
                          bias_initializer=biases_initializer,
                          weight_regularizer=weights_regularizer,
                          mu1_regularizer=mu1_regularizer,
                          mu2_regularizer=mu2_regularizer,
                          sigma_regularizer=sigma_regularizer,
                          bias_regularizer=biases_regularizer,
                          activity_regularizer=None,
                          weight_constraint=weights_constraint,
                          mu1_constraint=mu1_constraint,
                          mu2_constraint=mu2_constraint,
                          sigma_constraint=sigma_constraint,
                          bias_constraint=biases_constraint,
                          dau_unit_border_bound=dau_unit_border_bound,
                          dau_sigma_trainable=dau_sigma_trainable,
                          dau_mu_interpolation=dau_mu_interpolation,
                          trainable=trainable,
                          unit_testing=False,
                          name=sc.name,
                          _scope=sc,
                          _reuse=reuse)

        outputs = layer.apply(inputs)

        # Add variables to collections.
        layers_contrib._add_variable_to_collections(layer.dau_weights, variables_collections, 'weights')
        layers_contrib._add_variable_to_collections(layer.dau_mu1, variables_collections, 'mu1')
        layers_contrib._add_variable_to_collections(layer.dau_mu2, variables_collections, 'mu2')
        layers_contrib._add_variable_to_collections(layer.dau_sigma, variables_collections, 'sigma')

        if layer.use_bias:
            layers_contrib._add_variable_to_collections(layer.bias, variables_collections, 'biases')

        if normalizer_fn is not None:
            normalizer_params = normalizer_params or {}
            outputs = normalizer_fn(outputs, **normalizer_params)

        if activation_fn is not None:
            outputs = activation_fn(outputs)
        return utils_contrib.collect_named_outputs(outputs_collections, sc.name, outputs)
