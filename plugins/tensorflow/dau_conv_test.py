#!/usr/bin/env python3

import unittest
import time
import numpy as np
import tensorflow as tf

from dau_conv import DAUConv
from dau_conv import ZeroNLast
from dau_conv import DAUGridMean

class DAUConvTest(unittest.TestCase):

    def test_DAUConvFixed(self):

        results = []
        for i in range(1):
            #nchw
            #x_rand = tf.random_normal([16,32,64,64],dtype=tf.float32)
            x_rand = tf.ones([16,32,32,32],dtype=tf.float32)

            op = DAUConv(filters=64,
                         dau_units=(2,2),
                         max_kernel_size=7,
                         use_bias=False,
                         weight_initializer=tf.constant_initializer(1),
                         mu1_initializer=tf.constant_initializer(0),
                         mu2_initializer=tf.constant_initializer(0),
                         sigma_initializer=tf.constant_initializer(0.1))

            result_i = op(x_rand)

            results.append(result_i)

        init = tf.global_variables_initializer()

        c = tf.ConfigProto(allow_soft_placement=True,
                           log_device_placement=True)
        c.gpu_options.visible_device_list = '0'
        c.gpu_options.allow_growth = True

        with tf.Session(config=c) as s:
            s.run(init)
            t_start = time.time()

            r = s.run(results)

            t_end = time.time()
            print(t_end-t_start)

        gt_vals = np.ones(r[0].shape,dtype=np.float32)*32*4

        self.assertTrue(np.all(r[0] == gt_vals))

        print(r[0][0,0,0:5,:])
        print(r[0].shape)



    def test_DAUConvFwdRandom(self):

        results = []
        for i in range(1):
            #nchw
            x_rand = tf.random_normal([16,32,64,64],dtype=tf.float32)

            op = DAUConv(filters=64,
                         dau_units=(2,2),
                         max_kernel_size=7,
                         use_bias=False,
                         weight_initializer=tf.random_normal_initializer(stddev=0.1),
                         #mu1_initializer=DAUGridMean(dau_units=(2,2), max_value=3, dau_unit_axis=2),
                         #mu2_initializer=DAUGridMean(dau_units=(2,2), max_value=3, dau_unit_axis=1),
                         mu1_initializer=tf.random_uniform_initializer(minval=-3, maxval=3),
                         mu2_initializer=tf.random_uniform_initializer(minval=-3, maxval=3),
                         sigma_initializer=tf.constant_initializer(0.5))

            result_i = op(x_rand)

            results.append(result_i)

        init = tf.global_variables_initializer()

        c = tf.ConfigProto(allow_soft_placement=True,
                           log_device_placement=True)
        c.gpu_options.visible_device_list = '0'
        c.gpu_options.allow_growth = True

        with tf.Session(config=c) as s:
            s.run(init)
            t_start = time.time()

            r = s.run(results)

            t_end = time.time()
            print(t_end-t_start)

if __name__ == '__main__':
    unittest.main()
