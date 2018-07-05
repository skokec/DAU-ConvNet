#!/usr/bin/env python3

import tensorflow as tf
import os
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import sparse_ops

print(os.getcwd())
base_op_grad_module = tf.load_op_library('../../cmake-build-debug/plugins/tensorflow/libbase_op_grad.so')


@ops.RegisterGradient("BaseOp")
def _base_op_grad_cc(op, grad):
    # Op is the Op object - get all the inputs
    # Grad is the gradient with respect to the first input
    number_units_x = op.get_attr("number_units_x")
    number_units_y = op.get_attr("number_units_y")
    num_output = op.get_attr("num_output")
    kernel_size = op.get_attr("kernel_size")
    pad = op.get_attr("pad")
    stride = op.get_attr("stride")
    unit_normalization = op.get_attr("unit_normalization")
    square_unit_normalization = op.get_attr("square_unit_normalization")
    mean_iteration_step = op.get_attr("mean_iteration_step")
    sigma_iteration_step = op.get_attr("sigma_iteration_step")
    component_border_bound = op.get_attr("component_border_bound")
    sigma_lower_bound = op.get_attr("sigma_lower_bound")
    merge_iteration_step = op.get_attr("merge_iteration_step")
    merge_threshold = op.get_attr("merge_threshold")

    return base_op_grad_module.base_op_grad(grad, op.inputs[0], op.inputs[1], op.inputs[2], op.inputs[3], op.inputs[4],
                                            number_units_x=number_units_x,
                                            number_units_y=number_units_y,
                                            num_output=num_output,
                                            kernel_size=kernel_size,
                                            pad=pad,
                                            stride=stride,
                                            unit_normalization=unit_normalization,
                                            square_unit_normalization=square_unit_normalization,
                                            mean_iteration_step=mean_iteration_step,
                                            sigma_iteration_step=sigma_iteration_step,
                                            component_border_bound=component_border_bound,
                                            sigma_lower_bound=sigma_lower_bound,
                                            merge_iteration_step=merge_iteration_step,
                                            merge_threshold=merge_threshold)


# python impl.
# @ops.RegisterGradient("BaseOp")
def _base_op_grad(op, grad):
    input_tensor = op.inputs[0]
    weight_tensor = op.inputs[1]
    input_rows = array_ops.shape(input_tensor)[0]
    output_rows = array_ops.shape(weight_tensor)[0]

    grad_input = tf.matmul(tf.transpose(grad), weight_tensor)
    grad_weights = tf.multiply(tf.transpose(grad),
                               tf.reshape(tf.tile(tf.reshape(input_tensor, [input_rows]), [output_rows]),
                                          [output_rows, -1]))

    return [tf.transpose(grad_input), grad_weights]
