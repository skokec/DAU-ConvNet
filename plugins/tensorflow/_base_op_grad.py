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

    print("test")
    return base_op_grad_module.base_op_grad(grad, op.inputs[0], op.inputs[1], op.inputs[2], op.inputs[3], op.inputs[4])

# python impl.
#@ops.RegisterGradient("BaseOp")
def _base_op_grad(op, grad):
  
    input_tensor = op.inputs[0]
    weight_tensor = op.inputs[1]
    input_rows = array_ops.shape(input_tensor)[0]
    output_rows = array_ops.shape(weight_tensor)[0]
    
    grad_input = tf.matmul(tf.transpose(grad), weight_tensor)
    grad_weights = tf.multiply(tf.transpose(grad), tf.reshape(tf.tile(tf.reshape(input_tensor, [input_rows]), [output_rows]), [output_rows, -1]))
    
    return [tf.transpose(grad_input), grad_weights]
  
