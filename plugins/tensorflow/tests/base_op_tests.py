#!/usr/bin/env python3

import unittest
import numpy as np
import tensorflow

if tensorflow.__version__.startswith('1'):
        import tensorflow as tf
else:
        import tensorflow.compat.v1 as tf
        tf.disable_v2_behavior()

import _base_op_grad
base_op_module = tf.load_op_library('bin/libbase_op.so')

class baseOpOpTest(unittest.TestCase):
    def test_raisesExceptionWithIncompatibleDimensions(self):
        with tf.Session(''):
            with self.assertRaises(ValueError):
                base_op_module.base_op([1, 2], [[1, 2], [3, 4]]).eval()
            with self.assertRaises(ValueError):
                self.assertRaises(base_op_module.base_op([1, 2], [1, 2, 3, 4]).eval(), ValueError)
            with self.assertRaises(ValueError):
                self.assertRaises(base_op_module.base_op([1, 2, 3], [[1, 2], [3, 4]]).eval(), ValueError)
            
    def test_baseOpHardCoded(self):
        with tf.Session(''):
            result = base_op_module.base_op([[1], [2]], [[1, 2], [3, 4]]).eval()
            self.assertEqual(result.shape[0], 2)
            self.assertEqual(result[0], 5)
            self.assertEqual(result[1], 11)
    
    def test_baseOpGradientXHardCoded(self):
        with tf.Session('') as sess:
            x = tf.placeholder(tf.float32, shape = (2))
            W = tf.constant(np.asarray([[1, 2], [3, 4]]).astype(np.float32))
            
            Wx_tf = tf.matmul(W, tf.reshape(x, [-1, 1]))
            Wx_base_op = base_op_module.base_op(tf.reshape(x, [-1, 1]), W)
            
            grad_x_tf = tf.gradients(Wx_tf, x)
            grad_x_base_op = tf.gradients(Wx_base_op, x)
            
            gradient_tf = sess.run(grad_x_tf, feed_dict = {x: np.asarray([1, 2]).astype(np.float32)})
            gradient_base_op = sess.run(grad_x_base_op, feed_dict = {x: np.asarray([1, 2]).astype(np.float32)})
            
            self.assertEqual(gradient_tf[0][0], gradient_base_op[0][0])
            self.assertEqual(gradient_tf[0][1], gradient_base_op[0][1])
    
    def test_baseOpGradientWHardCoded(self):
        with tf.Session('') as sess:
            x = tf.constant(np.asarray([1, 2]).astype(np.float32))
            W = tf.placeholder(tf.float32, shape = (2, 2))
            
            Wx_tf = tf.matmul(W, tf.reshape(x, [-1, 1]))
            Wx_base_op = base_op_module.base_op(tf.reshape(x, [-1, 1]), W)
            
            grad_W_tf = tf.gradients(Wx_tf, W)
            grad_W_base_op = tf.gradients(Wx_base_op, W)
            
            gradient_tf = sess.run(grad_W_tf, feed_dict = {W: np.asarray([[1, 2], [3, 4]]).astype(np.float32)})
            gradient_base_op = sess.run(grad_W_base_op, feed_dict = {W: np.asarray([[1, 2], [3, 4]]).astype(np.float32)})
            
            self.assertEqual(gradient_tf[0][0][0], gradient_base_op[0][0][0])
            self.assertEqual(gradient_tf[0][0][1], gradient_base_op[0][0][1])
            self.assertEqual(gradient_tf[0][1][0], gradient_base_op[0][1][0])
            self.assertEqual(gradient_tf[0][1][1], gradient_base_op[0][1][1])
    
    def test_baseOpRandom(self):
        with tf.Session(''):
            n = 4
            m = 5
            
            for i in range(100):
                x_rand = np.random.randint(10, size = (n, 1))
                W_rand = np.random.randint(10, size = (m, n))
                result_rand = np.dot(W_rand, x_rand)
                
                result = base_op_module.base_op(x_rand, W_rand).eval()
                np.testing.assert_array_equal(result, result_rand)
    
    def test_baseOpGradientXRandom(self):
        with tf.Session('') as sess:
            n = 4
            m = 5
            
            x = tf.placeholder(tf.float32, shape = (n))
            W = tf.placeholder(tf.float32, shape = (m, n))
            
            Wx_tf = tf.matmul(W, tf.reshape(x, [-1, 1]))
            Wx_base_op = base_op_module.base_op(tf.reshape(x, [-1, 1]), W)
            
            grad_x_tf = tf.gradients(Wx_tf, x)
            grad_x_base_op = tf.gradients(Wx_base_op, x)
            
            for i in range(100):
                x_rand = np.random.randint(10, size = (n))
                W_rand = np.random.randint(10, size = (m, n))
                
                gradient_tf = sess.run(grad_x_tf, feed_dict = {x: x_rand, W: W_rand})
                gradient_base_op = sess.run(grad_x_base_op, feed_dict = {x: x_rand, W: W_rand})
                
                np.testing.assert_array_equal(gradient_tf, gradient_base_op)
                
    def test_baseOpGradientWRandom(self):
        with tf.Session('') as sess:
            n = 4
            m = 5
            
            x = tf.placeholder(tf.float32, shape = (n))
            W = tf.placeholder(tf.float32, shape = (m, n))
            
            Wx_tf = tf.matmul(W, tf.reshape(x, [-1, 1]))
            Wx_base_op = base_op_module.base_op(tf.reshape(x, [-1, 1]), W)
            
            grad_W_tf = tf.gradients(Wx_tf, W)
            grad_W_base_op = tf.gradients(Wx_base_op, W)
            
            for i in range(100):
                x_rand = np.random.randint(10, size = (n))
                W_rand = np.random.randint(10, size = (m, n))
                
                gradient_tf = sess.run(grad_W_tf, feed_dict = {x: x_rand, W: W_rand})
                gradient_base_op = sess.run(grad_W_base_op, feed_dict = {x: x_rand, W: W_rand})
                
                np.testing.assert_array_equal(gradient_tf, gradient_base_op)
                  
                
if __name__ == '__main__':
    unittest.main()
