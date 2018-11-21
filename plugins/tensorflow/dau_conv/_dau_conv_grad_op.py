#!/usr/bin/env python3

import tensorflow as tf
from tensorflow.python.framework import ops

dau_conv_grad_module = tf.load_op_library('libdau_conv_grad_op.so')


@ops.RegisterGradient("DAUConv")
def _dau_conv_op_grad_cc(op, grad):
    # Op is the Op object - get all the inputs
    # Grad is the gradient with respect to the first input
    number_units_x = op.get_attr("number_units_x")
    number_units_y = op.get_attr("number_units_y")
    number_units_ignore = op.get_attr("number_units_ignore")
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
    unit_testing = op.get_attr("unit_testing")
    mu_learning_rate_factor = op.get_attr("mu_learning_rate_factor")
    single_dim_kernel = op.get_attr("single_dim_kernel")
    forbid_positive_dim1 = op.get_attr("forbid_positive_dim1")


    return dau_conv_grad_module.dau_conv_grad(grad, op.inputs[0], op.inputs[1], op.inputs[2], op.inputs[3], op.inputs[4],
                                            number_units_x=number_units_x,
                                            number_units_y=number_units_y,
                                            number_units_ignore=number_units_ignore,
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
                                            merge_threshold=merge_threshold,
                                            mu_learning_rate_factor=mu_learning_rate_factor,
                                            single_dim_kernel=single_dim_kernel,
                                            forbid_positive_dim1=forbid_positive_dim1,
                                            unit_testing=unit_testing)
