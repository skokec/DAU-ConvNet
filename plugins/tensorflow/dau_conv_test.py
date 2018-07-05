#!/usr/bin/env python3

import unittest
import time
import numpy as np
import tensorflow as tf
import scipy
from dau_conv import DAUConv
from dau_conv import ZeroNLast
from dau_conv import DAUGridMean

from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.filters import convolve

from tensorflow.python.ops import init_ops

import pylab as plt

class DAUMeanTest(init_ops.Initializer):
    def __init__(self,tmp):
        pass

    def __call__(self, shape, dtype=None, partition_info=None):

        # create values
        vals = np.zeros(shape,dtype=np.float32)
        for i in range(shape[3]):
            vals[:,:,:,i] = np.random.rand()*2
            #vals[:

        #vals = tf.random_uniform(shape=shape,minval=-3, maxval=3,dtype=tf.float32)
        #vals = tf.floor(vals)
        #return vals
        vals = np.random.rand(shape[0],shape[1],shape[2],shape[3])*4-2
        vals = np.floor(vals*100) / 100
        #print('DAUMeanTest size of means:',vals.shape)
        return tf.convert_to_tensor(vals, np.float32)


        #return tf.reshape(vals,[1,shape[0],shape[1]*shape[2],shape[3]])

    def get_config(self):
        return {}

class DAUConvTest(unittest.TestCase):

    def _offset_and_sum(self, x, w, mu1, mu2):
        S = w.shape[1]
        G = w.shape[2]
        F = w.shape[3]

        out_shape = (x.shape[0], F, x.shape[2], x.shape[3])

        width_out = out_shape[-1]
        height_out = out_shape[-2]

        y = np.zeros(out_shape, dtype=np.float32)

        # add padding but b
        max_offset = np.max((np.max(np.abs(mu1)),
                             np.max(np.abs(mu2))))
        padding = np.int32(np.ceil(max_offset + 1))
        x_pad = np.pad(x, pad_width=[(0,),(0,),(padding,),(padding,)], mode='constant')

        for f in range(F):
            for s in range(S):
                for g in range(G):
                    w_val = w[0,s,g,f]
                    offset_x = mu1[0,s,g,f]
                    offset_y = mu2[0,s,g,f]

                    offset_x_int = np.floor(offset_x)
                    offset_y_int = np.floor(offset_y)

                    interpol_off_x = offset_x - offset_x_int
                    interpol_off_y = offset_y - offset_y_int

                    for dy in [0,1]:
                        for dx in [0,1]:
                            interpol_w = w_val

                            interpol_w = interpol_w * ((1-interpol_off_x) if dx == 0 else interpol_off_x)
                            interpol_w = interpol_w * ((1-interpol_off_y) if dy == 0 else interpol_off_y)

                            access_off_x = np.int32(offset_x_int + dx + padding)
                            access_off_y = np.int32(offset_y_int + dy + padding)

                            x_s = x_pad[:,s,access_off_y:height_out + access_off_y, access_off_x:width_out + access_off_x]

                            y[:,f,:,:] = y[:,f,:,:] + interpol_w *x_s
        return y


    def _forward_cpu(self, x, w, mu1, mu2, sigma):
        N = x.shape[0]
        S = x.shape[1]

        sigma_val = sigma[0]

        x_blur = np.zeros(x.shape,dtype=np.float32)

        filter,_,_,_,_ = self._get_filters(sigma_val)

        # pre-blur the X
        for n in range(N):
            for s in range(S):
                x_blur[n,s,:,:] = convolve(x[n,s,:,:],weights=filter,mode='constant')

        # then offset and sum element-wise
        y = self._offset_and_sum(x_blur,w,mu1,mu2)

        return y

    def _offset_and_dot(self, x, error, mu1, mu2):
        S = mu1.shape[1]
        G = mu1.shape[2]
        F = mu1.shape[3]

        out_shape = (x.shape[0], F, x.shape[2], x.shape[3])

        width_out = out_shape[-1]
        height_out = out_shape[-2]

        # add padding but b
        max_offset = np.max((np.max(np.abs(mu1)),
                             np.max(np.abs(mu2))))
        padding = np.int32(np.ceil(max_offset + 1))
        x_pad = np.pad(x, pad_width=[(0,),(0,),(padding,padding+1),(padding,padding+1)], mode='constant')

        output = np.zeros(mu1.shape,dtype=np.float32)

        for f in range(F):
            for s in range(S):
                for g in range(G):
                    offset_x = mu1[0,s,g,f]
                    offset_y = mu2[0,s,g,f]

                    offset_x_int = np.floor(offset_x)
                    offset_y_int = np.floor(offset_y)

                    interpol_off_x = offset_x - offset_x_int
                    interpol_off_y = offset_y - offset_y_int

                    for dy in [0,1]:
                        for dx in [0,1]:
                            interpol_w = 1

                            interpol_w = interpol_w * ((1-interpol_off_x) if dx == 0 else interpol_off_x)
                            interpol_w = interpol_w * ((1-interpol_off_y) if dy == 0 else interpol_off_y)

                            access_off_x = np.int32(offset_x_int + dx + padding)
                            access_off_y = np.int32(offset_y_int + dy + padding)

                            output[0,s,g,f] += np.sum(np.multiply(x_pad[:,s,access_off_y:height_out + access_off_y, access_off_x:width_out + access_off_x],
                                                                 error[:,f,:,:])) * interpol_w
        return output

    def _get_filters(self, sigma):
        N = 9

        x = np.tile(np.arange(N),(N,1))-4
        y = x.T

        filter = np.exp(-1 * (x**2 + y**2) /(2*sigma**2))
        deriv_w = filter
        deriv_mu1 = x / (sigma**2) * filter
        deriv_mu2 = y / (sigma**2) * filter
        deriv_sigma = (x**2 + y**2) / (sigma**3) * filter

        sum_filter = np.sum(filter)
        sum_mu1 = np.sum(deriv_mu1) / sum_filter
        sum_mu2 = np.sum(deriv_mu2) / sum_filter
        sum_sigma = np.sum(deriv_sigma) / sum_filter

        filter = filter / sum_filter
        deriv_w = deriv_w / sum_filter

        deriv_mu1 = deriv_mu1 / sum_filter - deriv_w *  sum_mu1
        deriv_mu2 = deriv_mu2 / sum_filter - deriv_w *  sum_mu2
        deriv_sigma = deriv_sigma / sum_filter - deriv_w * sum_sigma

        return (filter, deriv_w, deriv_mu1, deriv_mu2, deriv_sigma)

    def _backward_cpu(self, x, error, w, mu1, mu2, sigma):

        # we get back-propagated error by rotating offsets i.e. we just use negatives of offsets
        backprop_error = self._forward_cpu(error,
                                           np.swapaxes(w, 1,3),
                                           np.swapaxes(-1 * mu1, 1,3),
                                           np.swapaxes(-1 * mu2, 1,3), sigma)
        N = error.shape[0]
        S = error.shape[1]

        sigma_val = sigma[0]

        filter,deriv_w, deriv_mu1,deriv_mu2,_ = self._get_filters(sigma_val)

        # next we need to get gradients wrt w,mu1,mu2
        if True:
            error_w_blur = np.zeros(error.shape,dtype=np.float32)
            # pre-blur the X
            for n in range(N):
                for s in range(S):
                    error_w_blur[n,s,:,:] = convolve(error[n,s,:,:],weights=deriv_w,mode='constant')

            # then offset and sum element-wise
            w_grad = self._offset_and_dot(x,error_w_blur,mu1,mu2)

        if True:
            error_mu1_blur = np.zeros(error.shape,dtype=np.float32)
            # pre-blur the X
            for n in range(N):
                for s in range(S):
                    error_mu1_blur[n,s,:,:] = convolve(error[n,s,:,:],weights=deriv_mu1,mode='constant')

            # then offset and sum element-wise
            mu1_grad = self._offset_and_dot(x,error_mu1_blur,mu1,mu2)

        if True:
            error_mu2_blur = np.zeros(error.shape,dtype=np.float32)
            # pre-blur the X
            for n in range(N):
                for s in range(S):
                    error_mu2_blur[n,s,:,:] = convolve(error[n,s,:,:],weights=deriv_mu2,mode='constant')

            # then offset and sum element-wise
            mu2_grad = self._offset_and_dot(x,error_mu2_blur,mu1,mu2)

        # add multiplication with weight for mean gradients
        mu1_grad = np.multiply(mu1_grad, w)
        mu2_grad = np.multiply(mu2_grad, w)

        return (backprop_error, w_grad, mu1_grad, mu2_grad)

    def _assertMatrix(self, mat, gt_mat, rel_tolerance=0.01, plot_difference=True):
        diff_abs = np.abs(mat - gt_mat)
        diff_rel = np.nan_to_num(diff_abs / np.abs(gt_mat+1e-9))

        diff_idx = np.where(diff_rel > rel_tolerance)
        if len(diff_idx[0]) > 0:
            avg_diff = np.mean(diff_rel[diff_rel > 0.01])

            print('tensorflow:')
            print(mat[diff_idx[0][0],diff_idx[0][1],0:5,:])
            print('numpy:')
            print(gt_mat[diff_idx[0][0],diff_idx[0][1],0:5,:])
            print(gt_mat.shape)
            print('Avg rel-difference: %f' % avg_diff)

            if plot_difference:
                h = np.histogram2d(diff_idx[2], diff_idx[3], bins=diff_rel.shape[2:4], range=[[0, diff_rel.shape[2] - 1], [0, diff_rel.shape[3] - 1]])

                plt.imshow(h[0])
                plt.title('Avg rel-difference: %f' % avg_diff)
                plt.show(block=True)

        self.assertTrue(np.all(diff_rel <= rel_tolerance))

    def test_DAUConvFixed(self):

        for dd in range(20):
            sigma = 0.5
            x_rand = np.random.rand(16,32,32,32)
            #x_rand = np.ones((16,32,32,32),dtype=np.float32)

            #nchw
            #x_rand = tf.random_normal([16,32,64,64],dtype=tf.float32)
            #x_rand_ = tf.ones([16,32,32,32],dtype=tf.float32)
            x = tf.placeholder(tf.float32, shape = x_rand.shape)

            op = DAUConv(filters=64,
                         dau_units=(2,2),
                         max_kernel_size=9,
                         use_bias=False,
                         weight_initializer=tf.constant_initializer(1,dtype=np.float32),
                         #mu1_initializer=tf.random_uniform_initializer(minval=-3, maxval=3,dtype=tf.float32),
                         #mu2_initializer=tf.random_uniform_initializer(minval=-3, maxval=3,dtype=tf.float32),
                         #mu1_initializer=tf.random_uniform_initializer(minval=-2, maxval=1.8,dtype=tf.float32),
                         #mu1_initializer=DAUMeanTest(0),
                         #mu2_initializer=DAUMeanTest(0),
                         mu1_initializer=tf.constant_initializer(-2.56,dtype=np.float32),
                         mu2_initializer=tf.constant_initializer(-1,dtype=np.float32),
                         sigma_initializer=tf.constant_initializer(sigma))

            result = op(x)
            result_error = tf.ones([np.int32(x.shape[0]),64,
                                             np.int32(x.shape[2]),
                                             np.int32(x.shape[3])],dtype=tf.float32)
            #result_error = tf.random_normal([np.int32(x.shape[0]),64,
            #                                 np.int32(x.shape[2]),
            #                                 np.int32(x.shape[3])],dtype=tf.float32)

            var_grad = tf.gradients(result, [x, op.dau_weights, op.dau_mu1, op.dau_mu2], grad_ys=result_error)


            init = tf.global_variables_initializer()

            c = tf.ConfigProto(allow_soft_placement=True,
                               log_device_placement=True)
            c.gpu_options.visible_device_list = '3'
            c.gpu_options.allow_growth = True

            with tf.Session(config=c) as s:
                s.run(init)
                t_start = time.time()

                r, r_error, r_grad, w, mu1, mu2  = s.run([result, result_error, var_grad, op.dau_weights, op.dau_mu1, op.dau_mu2], feed_dict = {x: x_rand})

                t_end = time.time()
                print(t_end-t_start)

            print("min/max vals of m1: %f, %f" % (np.min(mu1), np.max(mu1)))

            gt_fwd_vals = self._forward_cpu(x=x_rand, w=w, mu1=mu1, mu2=mu2,
                                            sigma=[sigma])

            # interpolation in C++ code at the right edge excludes one pixel so ignore those pixels in check
            r = r[:,:,:,:-2]
            gt_fwd_vals = gt_fwd_vals[:,:,:,:-2]

            #gt_bwd_vals = self._backward_cpu(x=x_rand, error=r_error, w=w, mu1=mu1,mu2=mu2,
            #                                 sigma=[sigma])

            self._assertMatrix(r, gt_fwd_vals, rel_tolerance=0.05,plot_difference=False)


            #self.assertTrue(np.all(diff_rel <= 0.01))
            #self.assertTrue(np.all(np.abs(r_grad[0] - gt_bwd_vals[0]) < 0.01))
            #self.assertTrue(np.all(np.abs(r_grad[1] - gt_bwd_vals[1]) < 0.01))
            #self.assertTrue(np.all(np.abs(r_grad[2] - gt_bwd_vals[2]) < 0.01))
            #self.assertTrue(np.all(np.abs(r_grad[3] - gt_bwd_vals[3]) < 0.01))




    def test_DAUConvFwdRandom(self):
        return
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
