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

import _base_op_grad

base_op_module = tf.load_op_library('./bin/libbase_op.so')

class DAUGridMean(init_ops.Initializer):
    """Initializer for DAU means/offsets to follow the grid-based pattern, equaly spaced in each dimension.
    This is for mu1 in x axis.
    Args:
      dau_units: list of size 2. number of units in each direction (mu1,mu2)
      max_value: float. Limit of dau unit positions relative from the center
      use_centered_values: boolean. Output values that correspond to 0,0 at center (default=True), if false then 0,0 is at top-right corner
      dau_unit_axis: Integer. Axis for DAU units in input tensor (default=2)
    """
    def __init__(self, dau_units, max_value, dau_unit_axis=2):
        self.dau_units = dau_units
        self.dau_unit_axis = dau_unit_axis
        self.max_value = max_value

    def __call__(self, shape, dtype=None, partition_info=None):

        num_units = shape[self.dau_unit_axis]

        # create values
        vals = np.arange(num_units) * (2*self.max_value+1) / float(num_units) + (- 0.5+(2*self.max_value+1)/float(2*num_units)) - self.max_value

        # reshape to the same rank as requested shape
        shape_vals = np.ones(len(shape),dtype=np.int32)
        shape_vals[self.dau_unit_axis] = num_units

        # reshape and so convert it to tensor
        vals = np.reshape(vals,shape_vals)
        vals = tf.convert_to_tensor(vals, np.float32)

        # replicate in all remaining dims
        tile_rep_shape = list(shape)
        tile_rep_shape[num_units] = 1

        vals = tf.tile(input=vals,
                       multiples=tile_rep_shape)

        # TODO: this is temporary until C++ code is fixed to accept [S,mu1,mu2,F] values as well
        return tf.reshape(vals,[1,shape[0],shape[1]*shape[2],shape[3]])

    def get_config(self):
        return {
            "dau_units": self.dau_units,
            "dau_unit_axis": self.dau_unit_axis,
            "max_value": self.max_value
        }

class ZeroNLast(init_ops.Initializer):
    """Wrapper initializer that zeros N number of last values in specific axis.
    Args:
      base_init: tf.Initializer. Base Initializer operation
      last_num_to_zero: Integer. number of last N values in axis to zero-out
      axis: Integer. Which axis to zero.
    """

    def __init__(self, base_init, last_num_to_zero, axis):
        self.base_init = base_init
        self.last_num_to_zero = last_num_to_zero
        self.axis = axis

    def __call__(self, shape, dtype=None, partition_info=None):
        all_vals = self.base_init(shape, dtype, partition_info)

        shape_ones = all_vals.shape.as_list()
        shape_ones[self.axis] = all_vals.shape[self.axis] - self.last_num_to_zero

        shape_zeros = all_vals.shape.as_list()
        shape_zeros[self.axis] = self.last_num_to_zero

        ones_vals = tf.ones(shape_ones,dtype=tf.float32)
        zero_vals = tf.zeros(shape_zeros,dtype=tf.float32)

        valid_vals = tf.multiply(all_vals,tf.concat([ones_vals, zero_vals], axis=self.axis))

        return valid_vals

    def get_config(self):
        return {
            "last_num_to_zero": self.last_num_to_zero,
            "axis": self.axis,
            "base_init": self.base_init.get_config()
        }


class _DAUConvolution(object):
    """Helper class for _dau_convolution.
    Note that this class assumes that shapes of input and filter passed to
    __call__ are compatible with input_shape and filter_shape passed to the
    constructor.
    Arguments:
      input_shape: static input shape, i.e. input.get_shape().
      padding: see _non_atrous_convolution.
      data_format: see _non_atrous_convolution.
      strides: see _non_atrous_convolution.
      name: see _non_atrous_convolution.
    """
    def __init__(
            self,
            input_shape,
            num_output,
            dau_units,
            max_kernel_size,
            padding,
            data_format=None,
            strides=None,
            num_dau_units_ignore=0,
            unit_testing=False,
            name=None):
        self.num_output = num_output
        self.padding = padding
        self.name = name
        self.dau_units = dau_units
        self.max_kernel_size = max_kernel_size
        self.unit_testing = unit_testing
        input_shape = input_shape
        if input_shape.ndims is None:
            raise ValueError("Rank of convolution must be known")
        if input_shape.ndims < 3 or input_shape.ndims > 5:
            raise ValueError(
                "`input` and `filter` must have rank at least 3 and at most 5")
        conv_dims = input_shape.ndims - 2
        if strides is None:
            strides = [1] * conv_dims
        elif len(strides) != conv_dims:
            raise ValueError("len(strides)=%d, but should be %d" % (len(strides),
                                                                    conv_dims))
        if conv_dims == 1:
            # not supported
            raise ValueError("One dimensional DAUConv not supported - only two dimensions supported.")
        elif conv_dims == 2:
            if data_format is None or data_format == "NHWC":
                raise ValueError("data_format \"NHWC\" not supported - TODO: manually convert to NHWC.")
            elif data_format == "NCHW":
                strides = [1, 1] + list(strides)
            else:
                raise ValueError("data_format must be \"NHWC\" or \"NCHW\".")
            self.strides = strides
            self.data_format = data_format
            self.dau_conv_op = base_op_module.base_op
        elif conv_dims == 3:
            # not supported
            raise ValueError("Three dimensional DAUConv not supported - only two dimensions supported.")

        # currently supporting only strides=1
        if strides is not None and any(map(lambda x: x != 1, strides)):
            raise ValueError("Only strides=1 supported.")

    # pylint: enable=redefined-builtin

    def __call__(self, inp, w, mu1, mu2, sigma):  # pylint: disable=redefined-builtin

        # TODO: number_units should be infereed from W, but we need to fix to have size of W,mu1,mu2,sigma in [S, Gy, Gx, F] format
        settings = dict(num_output=self.num_output,
                        number_units_x=self.dau_units[0],
                        number_units_y=self.dau_units[1],
                        kernel_size=self.max_kernel_size[0],
                        pad=self.padding[0],
                        component_border_bound=1,
                        sigma_lower_bound=0.01,
                        unit_testing=self.unit_testing)
        return self.dau_conv_op(
            input=inp,
            weights=w,
            mu1=mu1,
            mu2=mu2,
            sigma=sigma,
            **settings
            #strides=self.strides,
            #padding=self.padding,
            #data_format=self.data_format,
            #name=self.name
            )

class DAUConv(base.Layer):

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
                 mu1_initializer=init_ops.zeros_initializer(),
                 mu2_initializer=init_ops.zeros_initializer(),
                 sigma_initializer=init_ops.constant_initializer(0.5),
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
                 unit_testing=False, # for competability between CPU and GPU version (where gradients of last edge need to be ignored) during unit testing
                 name=None,
                 **kwargs):
        super(DAUConv, self).__init__(trainable=trainable, name=name,
                                    activity_regularizer=activity_regularizer,
                                    **kwargs)
        self.rank = 2
        self.filters = filters
        self.dau_units = utils.normalize_tuple(dau_units, self.rank, 'dau_components')
        self.max_kernel_size = utils.normalize_tuple(max_kernel_size, self.rank, 'max_kernel_size')
        self.padding = list(map(lambda x: np.floor(x/2), self.max_kernel_size))
        self.strides = utils.normalize_tuple(strides, self.rank, 'strides')
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

        self.unit_testing = unit_testing

        self.input_spec = base.InputSpec(ndim=self.rank + 2)

        self.num_dau_units_all = np.prod(self.dau_units)
        self.num_dau_units_ignore = 0

        # if we have less then 2 units per channel then or have odd number of them then add one more dummy unit
        # since computation is always done with 2 units at the same time (effecitvly set weight=0 for those dummy units)

        # make sure we have at least ALLOWED_UNITS_GROUP (this is requested so for fast version that can handle only factor of 2)
        if  self.num_dau_units_all % self.DAU_UNITS_GROUP != 0:
            new_num_units = np.ceil(self.num_dau_units_all / self.DAU_UNITS_GROUP) * self.DAU_UNITS_GROUP

            self.num_dau_units_ignore = new_num_units - self.num_dau_units_all

            if self.dau_units[0] < self.dau_units[1]:
                self.dau_units = (self.dau_units[0] + self.num_dau_units_ignore, self.dau_units[1])
            else:
                self.dau_units = (self.dau_units[0], self.dau_units[1] + self.num_dau_units_ignore)

            self.num_dau_units_all = new_num_units

            self.weight_initializer = ZeroNLast(self.weight_initializer, last_num_to_zero=self.num_dau_units_ignore, axis=2)


        self.dau_weights = None
        self.dau_mu1 = None
        self.dau_mu2 = None
        self.dau_sigma = None

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
            raise ValueError('Only `channels_first` supported, i.e., NCHW format.')

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

    def build(self, input_shape):
        input_shape = tensor_shape.TensorShape(input_shape)

        dau_params_shape = self.get_dau_variable_shape(input_shape)
        if self.dau_weights is None:
            self.dau_weights = self.add_variable(name='weights',
                                                 shape=dau_params_shape,
                                                 initializer=self.weight_initializer,
                                                 regularizer=self.weight_regularizer,
                                                 constraint=self.weight_constraint,
                                                 trainable=True,
                                                 dtype=self.dtype)
        elif np.any(self.dau_weights != dau_params_shape):
            raise ValueError('Shape mismatch for variable `dau_weights`')
        if self.dau_mu1 is None:
            self.dau_mu1 = self.add_variable(name='mu1',
                                                 shape=dau_params_shape,
                                                 initializer=self.mu1_initializer,
                                                 regularizer=self.mu1_regularizer,
                                                 constraint=self.mu1_constraint,
                                                 trainable=True,
                                                 dtype=self.dtype)
        elif np.any(self.dau_mu1 != dau_params_shape):
            raise ValueError('Shape mismatch for variable `dau_mu1`')

        if self.dau_mu2 is None:
            self.dau_mu2 = self.add_variable(name='mu2',
                                                 shape=dau_params_shape,
                                                 initializer=self.mu2_initializer,
                                                 regularizer=self.mu2_regularizer,
                                                 constraint=self.mu2_constraint,
                                                 trainable=True,
                                                 dtype=self.dtype)
        elif np.any(self.dau_mu2 != dau_params_shape):
            raise ValueError('Shape mismatch for variable `dau_mu2`')
        if self.dau_sigma is None:
            self.dau_sigma = self.add_variable(name='sigma',
                                                 shape=dau_params_shape,
                                                 initializer=self.sigma_initializer,
                                                 regularizer=self.sigma_regularizer,
                                                 constraint=self.sigma_constraint,
                                                 trainable=False,
                                                 dtype=self.dtype)
        elif np.any(self.dau_sigma != dau_params_shape):
            raise ValueError('Shape mismatch for variable `dau_sigma`')

        if self.use_bias:
            self.bias = self.add_variable(name='bias',
                                          shape=(self.filters,),
                                          initializer=self.bias_initializer,
                                          regularizer=self.bias_regularizer,
                                          constraint=self.bias_constraint,
                                          trainable=True,
                                          dtype=self.dtype)
        else:
            self.bias = None

        input_channel_axis = self._get_input_channel_axis()
        num_input_channels = self._get_input_channels(input_shape)

        self.input_spec = base.InputSpec(ndim=self.rank + 2,
                                         axes={input_channel_axis: num_input_channels})

        self._dau_convolution_op = _DAUConvolution(
            input_shape,
            num_output=self.filters,
            dau_units=self.dau_units,
            max_kernel_size=self.max_kernel_size,
            padding=self.padding,
            strides=self.strides,
            num_dau_units_ignore=self.num_dau_units_ignore,
            unit_testing=self.unit_testing,
            data_format=utils.convert_data_format(self.data_format,
                                                  self.rank + 2))
        self.built = True

    def call(self, inputs):
        outputs = self._dau_convolution_op(inputs, self.dau_weights, self.dau_mu1, self.dau_mu2, self.dau_sigma)

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
                    dilation=self.dilation_rate[i])
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
                    stride=self.strides[i],
                    dilation=self.dilation_rate[i])
                new_space.append(new_dim)
            return tensor_shape.TensorShape([input_shape[0], self.filters] +
                                            new_space)