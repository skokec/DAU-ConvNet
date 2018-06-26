#!/usr/bin/env python3

import unittest
import time
import numpy as np
import tensorflow as tf
import _base_op_grad
base_op_module = tf.load_op_library('../../cmake-build-debug/plugins/tensorflow/libbase_op.so')
#base_op_module = tf.load_op_library('./bin/libbase_op.so')

class baseOpOpTest(unittest.TestCase):
            
    """
    def test_baseOpHardCoded(self):
        with tf.Session(''):
            result = base_op_module.base_op([[1], [2]], [[1, 2], [3, 4]]).eval()
            self.assertEqual(result.shape[0], 2)
            self.assertEqual(result[0], 5)
            self.assertEqual(result[1], 11)
    """

    def test_baseOpRandom(self):
        with tf.Session(''):
            n = 4
            m = 5
            
            for i in range(30):
                #x_rand = np.random.randint(10, size = (n, 1))
                #W_rand = np.random.randint(10, size = (m, n))
                #result_rand = np.dot(W_rand, x_rand)
                #  .Input("weights: float")
                #  .Input("mu1: float")
                #  .Input("mu2: float")
                #  .Input("sigma: float")
                #nchw
                x_rand = np.random.rand(16,32,64,64)

                #1,conv_in_channels_ == from input, units_per_channel, conv_out_channels_==num filters
                W_rand = np.random.rand(1,32,4,64)
                mu1_rand = np.random.rand(1,32,4,64)
                mu2_rand = np.random.rand(1,32,4,64)
                sigma_rand = np.random.rand(1,32,4,64)
                t_start = time.time()
                result = base_op_module.base_op(x_rand, W_rand, mu1_rand, mu2_rand, sigma_rand).eval()
                t_end = time.time()
                print(t_end-t_start)
                #np.testing.assert_array_equal(result, result_rand)


if __name__ == '__main__':
    unittest.main()
