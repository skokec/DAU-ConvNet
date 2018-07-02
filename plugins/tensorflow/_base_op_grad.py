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
    print("test")
    """
        .Attr("number_units_x : int  = 2")
        .Attr("number_units_y : int = 2")
        .Attr("bias_term: bool = true")
        .Attr("kernel_size: int = 9")
        .Attr("pad: int = 4")
        .Attr("stride: int = 1")
        .Attr("unit_normalization: bool = true")
        .Attr("square_unit_normalization: bool = true")
        .Attr("mean_iteration_step: int = 1")
        .Attr("sigma_iteration_step: int = 1")
        .Attr("component_border_bound: int = 4")
        .Attr("sigma_lower_bound: float = 0.3")
        .Attr("merge_iteration_step: int = 0")
        .Attr("merge_threshold: int = 1");
    """
    offsets_already_centered = op.get_attr("offsets_already_centered")
    number_units_x = op.get_attr("number_units_x")
    print(number_units_x)
    number_units_y = op.get_attr("number_units_y")
    print(number_units_y)
    bias_term = op.get_attr("bias_term")
    print(bias_term)
    kernel_size = op.get_attr("kernel_size")
    print(kernel_size)
    pad = op.get_attr("pad")
    print(pad)
    stride = op.get_attr("stride")
    print(stride)
    unit_normalization = op.get_attr("unit_normalization")
    print(unit_normalization)
    square_unit_normalization = op.get_attr("square_unit_normalization")
    print(square_unit_normalization)
    mean_iteration_step = op.get_attr("mean_iteration_step")
    print(mean_iteration_step)
    sigma_iteration_step = op.get_attr("sigma_iteration_step")
    print(sigma_iteration_step)
    component_border_bound = op.get_attr("component_border_bound")
    print(component_border_bound)
    sigma_lower_bound = op.get_attr("sigma_lower_bound")
    print(sigma_lower_bound)
    merge_iteration_step = op.get_attr("merge_iteration_step")
    print(merge_iteration_step)
    merge_threshold = op.get_attr("merge_threshold")
    print(merge_threshold)

    return base_op_grad_module.base_op_grad(grad, op.inputs[0], op.inputs[1], op.inputs[2], op.inputs[3], op.inputs[4],
                                            offsets_already_centered=offsets_already_centered,
                                            number_units_x=number_units_x,
                                            number_units_y=number_units_y,
                                            bias_term=bias_term,
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
