#!/usr/bin/env python3

import unittest
import time
import numpy as np
import tensorflow as tf
import scipy
from dau_conv import DAUConv2d,DAUConv1d
from dau_conv import ZeroNLast
from dau_conv import DAUGridMean

from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.filters import convolve
from scipy.ndimage.filters import correlate

from tensorflow.python.ops import init_ops

import pylab as plt

class DAUConvPython:
    def _offset_and_sum(self, x, w, mu1, mu2, num_dau_units_ignore=0):
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
                for g in range(G-num_dau_units_ignore):
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


    def forward_cpu(self, x, w, mu1, mu2, sigma, num_dau_units_ignore=0, single_dim_kernel=False):
        N = x.shape[0]
        S = x.shape[1]

        sigma_val = sigma[0]

        x_blur = np.zeros(x.shape,dtype=np.float32)

        filter,_,_,_,_ = self._get_filters(sigma_val, single_dim_kernel=single_dim_kernel)

        # pre-blur the X
        for n in range(N):
            for s in range(S):
                x_blur[n,s,:,:] = correlate(x[n,s,:,:],weights=filter,mode='constant')

        # then offset and sum element-wise
        y = self._offset_and_sum(x_blur,w,mu1,mu2, num_dau_units_ignore=num_dau_units_ignore)

        return y

    def _offset_and_dot(self, x, error_, mu1, mu2, num_dau_units_ignore=0, ignore_edge_gradients=True):
        S = mu1.shape[1]
        G = mu1.shape[2]
        F = mu1.shape[3]

        out_shape = (x.shape[0], F, x.shape[2], x.shape[3])

        width_out = out_shape[-1]
        height_out = out_shape[-2]

        error = error_

        # set right/bottom edges to zero if we should ignore them (for GPU compatability)
        # this must be done for competability with GPU version since by one last pixel will not be accounted accurately
        # in GPU mode
        if ignore_edge_gradients:

            disable_last_column = False
            disable_last_row = False

            if width_out >= 64:
                disable_last_column = width_out % 64 == 0
            elif width_out >= 32:
                disable_last_column = width_out % 32 == 0
            elif width_out >= 16:
                disable_last_column = width_out % 16 == 0
            elif width_out >= 8:
                disable_last_column = width_out % 8 == 0

            if height_out >= 64:
                disable_last_row = height_out % 64 == 0
            elif height_out >= 32:
                disable_last_row = height_out % 32 == 0
            elif height_out >= 16:
                disable_last_row = height_out % 16 == 0
            elif height_out >= 8:
                disable_last_row = height_out % 8 == 0

            if disable_last_column:
                error[:,:,:,width_out-1] = 0.0
            if disable_last_row:
                error[:,:,height_out-1,:] = 0.0



        # add padding but b
        max_offset = np.max((np.max(np.abs(mu1)),
                             np.max(np.abs(mu2))))
        padding = np.int32(np.ceil(max_offset + 1))
        x_pad = np.pad(x, pad_width=[(0,),(0,),(padding,),(padding,)], mode='constant')

        output = np.zeros(mu1.shape,dtype=np.float32)

        for f in range(F):
            for s in range(S):
                for g in range(G-num_dau_units_ignore):
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

    def _get_filters(self, sigma, single_dim_kernel=False):
        N = 9

        x = np.tile(np.arange(N),(N,1))-4
        y = x.T

        filter = np.exp(-1 * (x**2 + y**2) /(2*sigma**2))

        # set values to zeros in second dimension if reqested only one dimension
        if single_dim_kernel:
            valid_filter = np.zeros_like(filter)

            valid_filter[valid_filter.shape[0]/2,:] = 1

            filter = filter * valid_filter

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

    def backward_cpu(self, x, error, w, mu1, mu2, sigma, num_dau_units_ignore=0, unit_testing=True, single_dim_kernel=False):

        # we get back-propagated error by rotating offsets i.e. we just use negatives of offsets
        backprop_error = self.forward_cpu(error,
                                           np.swapaxes(w, 1,3),
                                           np.swapaxes(-1 * mu1, 1,3),
                                           np.swapaxes(-1 * mu2, 1,3), sigma, single_dim_kernel=single_dim_kernel)
        N = x.shape[0]
        F = x.shape[1]

        sigma_val = sigma[0]

        filter,deriv_w, deriv_mu1,deriv_mu2,_ = self._get_filters(sigma_val, single_dim_kernel=single_dim_kernel)


        # next we need to get gradients wrt w,mu1,mu2
        if True:
            x_w_blur = np.zeros(x.shape,dtype=np.float32)
            # pre-blur the X
            for n in range(N):
                for f in range(F):
                    x_w_blur[n,f,:,:] = correlate(x[n,f,:,:],weights=deriv_w,mode='constant')

            # then offset and sum element-wise
            w_grad = self._offset_and_dot(x_w_blur,error,mu1,mu2, num_dau_units_ignore=num_dau_units_ignore, ignore_edge_gradients=unit_testing)

        if True:
            x_mu1_blur = np.zeros(x.shape,dtype=np.float32)
            # pre-blur the X
            for n in range(N):
                for f in range(F):
                    x_mu1_blur[n,f,:,:] = correlate(x[n,f,:,:],weights=deriv_mu1,mode='constant')

            # then offset and sum element-wise
            mu1_grad = self._offset_and_dot(x_mu1_blur,error,mu1,mu2, num_dau_units_ignore=num_dau_units_ignore, ignore_edge_gradients=unit_testing)

        if True:
            x_mu2_blur = np.zeros(x.shape,dtype=np.float32)
            # pre-blur the X
            for n in range(N):
                for f in range(F):
                    x_mu2_blur[n,f,:,:] = correlate(x[n,f,:,:],weights=deriv_mu2,mode='constant')

            # then offset and sum element-wise
            mu2_grad = self._offset_and_dot(x_mu2_blur,error,mu1,mu2, num_dau_units_ignore=num_dau_units_ignore, ignore_edge_gradients=unit_testing)

        # add multiplication with weight for mean gradients
        mu1_grad = np.multiply(mu1_grad, w)
        mu2_grad = np.multiply(mu2_grad, w)

        return (backprop_error, w_grad, mu1_grad, mu2_grad)

class DAUConvTest(unittest.TestCase):


    def _assertMatrix(self, mat, gt_mat, variable_name, rel_tolerance=0.01, plot_difference=True):
        diff_abs = np.abs(mat - gt_mat)
        diff_rel = np.nan_to_num(diff_abs / np.abs(gt_mat+1e-9))

        avg_diff = np.mean(diff_rel)

        #diff_idx = np.where(diff_rel > rel_tolerance)

        invalid_mask = np.logical_and(diff_rel > 1e-4, diff_abs > 1e-7)
        diff_idx = np.where(invalid_mask)

        num_diff_rate = len(diff_idx[0])/float(diff_rel.size)

        avg_diff = np.mean(diff_rel[invalid_mask]) if np.sum(invalid_mask) > 0 else 0

        if avg_diff > rel_tolerance and num_diff_rate > 1e-2:

            print('tensorflow:')
            print(mat[diff_idx[0][0],diff_idx[0][1],0:5,:])
            print('numpy:')
            print(gt_mat[diff_idx[0][0],diff_idx[0][1],0:5,:])
            print(gt_mat.shape)
            print('Avg rel-difference: %f for \'%s\' variable' % (avg_diff,variable_name))

            if plot_difference:
                plot_diff_idx = np.where(np.logical_and(invalid_mask, diff_rel > rel_tolerance))

                h = np.histogram2d(plot_diff_idx[2], plot_diff_idx[3], bins=diff_rel.shape[2:4], range=[[0, diff_rel.shape[2] - 1], [0, diff_rel.shape[3] - 1]])

                plt.imshow(h[0])
                plt.title('Avg rel-difference: %f for %s' % (avg_diff,variable_name))
                plt.show(block=True)

        self.assertTrue(avg_diff <= rel_tolerance or num_diff_rate <= 1e-2)

    def _run_DAUConv_forward_and_backward(self, repeat, N, W, H, S, F, dau_uints, max_kernel_size, max_offset_init):

        for i in range(repeat):
            mu_learning_rate_factor = 1000
            input_channels = S
            num_output = F
            sigma = 0.5
            x_rand = np.random.rand(N,input_channels,H,W)
            #x_rand = np.ones((16,num_output,32,32),dtype=np.float32)

            x = tf.placeholder(tf.float32, shape = x_rand.shape)

            op = DAUConv2d(filters=num_output,
                           dau_units=dau_uints,
                           max_kernel_size=max_kernel_size,
                           use_bias=False,
                           weight_initializer=tf.random_normal_initializer(stddev=0.1, dtype=np.float32),
                           mu1_initializer=tf.random_uniform_initializer(minval=-max_offset_init, maxval=max_offset_init,dtype=tf.float32),
                           mu2_initializer=tf.random_uniform_initializer(minval=-max_offset_init, maxval=max_offset_init,dtype=tf.float32),
                           #weight_initializer=tf.constant_initializer(1,dtype=np.float32),
                           #mu1_initializer=tf.constant_initializer(0,dtype=np.float32),
                           #mu2_initializer=tf.constant_initializer(0,dtype=np.float32),
                           sigma_initializer=tf.constant_initializer(sigma),
                           mu_learning_rate_factor=mu_learning_rate_factor,
                           unit_testing=True)

            result = op(x)
            #result_error = tf.ones([np.int32(x.shape[0]),num_output,
            #                                 np.int32(x.shape[2]),
            #                                 np.int32(x.shape[3])],dtype=tf.float32)
            result_error = tf.random_normal([np.int32(x.shape[0]),num_output,
                                             np.int32(x.shape[2]),
                                             np.int32(x.shape[3])],dtype=tf.float32)

            var_grad = tf.gradients(result, [x, op.dau_weights, op.dau_mu1, op.dau_mu2], grad_ys=result_error)


            init = tf.global_variables_initializer()

            c = tf.ConfigProto(allow_soft_placement=True,
                               log_device_placement=True)
            c.gpu_options.visible_device_list = '2'
            c.gpu_options.allow_growth = True

            with tf.Session(config=c) as s:

                s.run(init)
                t_start = time.time()

                r, r_error, r_grad, w, mu1, mu2  = s.run([result, result_error, var_grad, op.dau_weights, op.dau_mu1, op.dau_mu2], feed_dict = {x: x_rand})

                t_end = time.time()
                print(t_end-t_start)

            gt_fwd_vals = DAUConvPython().forward_cpu(x=x_rand, w=w, mu1=mu1, mu2=mu2,
                                                      sigma=[sigma], num_dau_units_ignore=op.num_dau_units_ignore)

            gt_bwd_vals = DAUConvPython().backward_cpu(x=x_rand, error=r_error, w=w, mu1=mu1,mu2=mu2,
                                                       sigma=[sigma], num_dau_units_ignore=op.num_dau_units_ignore, unit_testing=True)

            # interpolation in C++ code at the right edge excludes one pixel so ignore those pixels in check
            r = r[:,:,:,:-2]
            r_grad[0] = r_grad[0][:,:,:,:-2]
            gt_fwd_vals = gt_fwd_vals[:,:,:,:-2]
            gt_bwd_vals = (gt_bwd_vals[0][:,:,:,:-2],
                           gt_bwd_vals[1],
                           gt_bwd_vals[2]* mu_learning_rate_factor,
                           gt_bwd_vals[3]* mu_learning_rate_factor)

            self._assertMatrix(r, gt_fwd_vals, 'fwd_output', rel_tolerance=0.01,plot_difference=True)

            self._assertMatrix(r_grad[0], gt_bwd_vals[0], 'bwd_error', rel_tolerance=0.01,plot_difference=True)
            self._assertMatrix(r_grad[1], gt_bwd_vals[1], 'bwd_w_grad', rel_tolerance=0.01, plot_difference=True)
            self._assertMatrix(r_grad[2], gt_bwd_vals[2], 'bwd_mu1_grad', rel_tolerance=0.01, plot_difference=True)
            self._assertMatrix(r_grad[3], gt_bwd_vals[3], 'bwd_mu2_grad', rel_tolerance=0.01, plot_difference=True)

    def test_DAUConv(self):

        # test small kernels (9 and 17)
        self._run_DAUConv_forward_and_backward(repeat=5, N=16, W=32, H=32, S=32, F=32, dau_uints=(2,2), max_kernel_size=9, max_offset_init=3)
        self._run_DAUConv_forward_and_backward(repeat=5, N=16, W=32, H=32, S=32, F=32, dau_uints=(2,2), max_kernel_size=17, max_offset_init=6)

        self._run_DAUConv_forward_and_backward(repeat=5, N=16, W=6, H=6, S=64, F=256, dau_uints=(2,1), max_kernel_size=17, max_offset_init=8)

        # test with dynamic kernel size optimization (using smaller kernel dispite large allowed kernel)
        self._run_DAUConv_forward_and_backward(repeat=5, N=16, W=32, H=32, S=32, F=32, dau_uints=(2,2), max_kernel_size=17, max_offset_init=3)

        # test with uneven number of sub-features
        self._run_DAUConv_forward_and_backward(repeat=2, N=16, W=32, H=32, S=3, F=32, dau_uints=(2,2), max_kernel_size=17, max_offset_init=3)
        self._run_DAUConv_forward_and_backward(repeat=2, N=16, W=64, H=64, S=3, F=32, dau_uints=(2,2), max_kernel_size=33, max_offset_init=10)

        # test large kernels (33 and 65)
        self._run_DAUConv_forward_and_backward(repeat=2, N=16, W=64, H=64, S=32, F=32, dau_uints=(2,2), max_kernel_size=33, max_offset_init=10)
        self._run_DAUConv_forward_and_backward(repeat=2, N=16, W=64, H=64, S=32, F=32, dau_uints=(2,2), max_kernel_size=65, max_offset_init=20)

    def test_DAUConvSpeedTest(self):
        repeat=5
        N=32
        W=16
        H=16
        S=128
        F=32
        dau_uints=(2,1)
        max_kernel_size=9
        max_offset_init=3

        dau_times = []


        conv_times = []
        if True:
            input_channels = S
            num_output = F
            sigma = 0.5
            x_rand = np.random.rand(N,input_channels,H,W)
            #x = tf.placeholder(tf.float32, shape = x_rand.shape)
            x = tf.constant(0, shape = x_rand.shape, dtype=tf.float32)
            tmp = []

            op = tf.layers.Conv2D(filters=num_output,
                                  kernel_size=3,
                                  use_bias=False,
                                  padding='same',
                                  data_format='channels_first',
                                  kernel_initializer=tf.random_normal_initializer(stddev=0.1, dtype=np.float32))
            result = op.apply(x)

            tmp.append(tf.reduce_max(result))
            #tmp.append(tf.reduce_max(x))

            result_error = tf.random_normal([np.int32(x.shape[0]),num_output,
                                             np.int32(x.shape[2]),
                                             np.int32(x.shape[3])],dtype=tf.float32)

            var_grad = tf.gradients(result, [x]+op.weights, grad_ys=result_error)
            #var_grad = [x]+op.weights

            tmp.append(tf.reduce_max(var_grad[0]))
            tmp.append(tf.reduce_max(var_grad[1:]))

            init = tf.global_variables_initializer()

            c = tf.ConfigProto(allow_soft_placement=True,
                               log_device_placement=True)
            c.gpu_options.visible_device_list = '2'
            c.gpu_options.allow_growth = True

            with tf.Session(config=c) as s:

                s.run(init)
                for i in range(100):
                    t_start = time.time()
                    #s.run([result, result_error, var_grad], feed_dict = {x: x_rand})
                    s.run(tmp)
                    t_end = time.time()
                    t = t_end-t_start
                    conv_times.append(t)

        if True:
            mu_learning_rate_factor = 1000
            input_channels = S
            num_output = F
            sigma = 0.5
            x_rand = np.random.rand(N,input_channels,H,W)
            #x = tf.placeholder(tf.float32, shape = x_rand.shape)
            x = tf.constant(0, shape = x_rand.shape, dtype=tf.float32)
            tmp = []

            op = DAUConv2d(filters=num_output,
                           dau_units=dau_uints,
                           max_kernel_size=max_kernel_size,
                           use_bias=False,
                           weight_initializer=tf.random_normal_initializer(stddev=0.1, dtype=np.float32),
                           mu1_initializer=tf.random_uniform_initializer(minval=-max_offset_init, maxval=max_offset_init,dtype=tf.float32),
                           mu2_initializer=tf.random_uniform_initializer(minval=-max_offset_init, maxval=max_offset_init,dtype=tf.float32),
                           sigma_initializer=tf.constant_initializer(sigma),
                           mu_learning_rate_factor=mu_learning_rate_factor,
                           unit_testing=False)

            result = op(x)

            tmp.append(tf.reduce_max(result))
            #tmp.append(tf.reduce_max(x))

            result_error = tf.random_normal([np.int32(x.shape[0]),num_output,
                                             np.int32(x.shape[2]),
                                             np.int32(x.shape[3])],dtype=tf.float32)

            var_grad = tf.gradients(result, [x, op.dau_weights, op.dau_mu1, op.dau_mu2], grad_ys=result_error)
            #var_grad = [x, op.dau_weights, op.dau_mu1, op.dau_mu2]

            tmp.append(tf.reduce_max(var_grad[0]))
            tmp.append(tf.reduce_max(var_grad[1:]))

            init = tf.global_variables_initializer()

            c = tf.ConfigProto(allow_soft_placement=True,
                               log_device_placement=True)
            c.gpu_options.visible_device_list = '2'
            c.gpu_options.allow_growth = True

            with tf.Session(config=c) as s:

                s.run(init)

                for i in range(100):
                    t_start = time.time()
                    #s.run([result, result_error, var_grad], feed_dict = {x: x_rand})
                    #s.run([result], feed_dict = {x: x_rand})
                    s.run(tmp)
                    t_end = time.time()
                    t = t_end-t_start

                    dau_times.append(t)

                print("dau times: ", dau_times)
                print("conv times: ", conv_times)

                print("dau avg time: %f\n" % np.mean(dau_times[20:]))
                print("conv avg time: %f\n" % np.mean(conv_times[20:]))


    def test_DAUConvSingleUnit(self):
        # test with single DAU unit
        self._run_DAUConv_forward_and_backward(repeat=3, N=16, W=32, H=32, S=32, F=32, dau_uints=(1,1), max_kernel_size=9, max_offset_init=3)

    def test_DAUConvMemtest(self):

        N = 32
        W = 6
        H = 6
        input_channels = 128
        num_output = 256
        sigma = 0.5
        x_rand = np.random.rand(N,input_channels,H,W)

        x = tf.placeholder(tf.float32, shape = x_rand.shape)

        op = DAUConv2d(filters=num_output,
                       dau_units=(2,1),
                       max_kernel_size=9,
                       use_bias=False,
                       weight_initializer=tf.random_normal_initializer(stddev=0.1, dtype=np.float32),
                       mu1_initializer=tf.random_uniform_initializer(minval=-10, maxval=10,dtype=tf.float32),
                       mu2_initializer=tf.random_uniform_initializer(minval=-10, maxval=10,dtype=tf.float32),
                       sigma_initializer=tf.constant_initializer(sigma),
                       dau_unit_border_bound=0.1,
                       unit_testing=False)

        result = op(x)
        result_error = tf.random_normal([np.int32(x.shape[0]),num_output,
                                         np.int32(x.shape[2]),
                                         np.int32(x.shape[3])],dtype=tf.float32)

        var_grad = tf.gradients(result, [x, op.dau_weights, op.dau_mu1, op.dau_mu2], grad_ys=result_error)


        init = tf.global_variables_initializer()

        c = tf.ConfigProto(allow_soft_placement=True,
                           log_device_placement=True)
        c.gpu_options.visible_device_list = '2'
        c.gpu_options.allow_growth = True

        with tf.Session(config=c) as s:

            for nn in range(10000):
                s.run(init)
                t_start = time.time()

                r, r_error, r_grad, w, mu1, mu2  = s.run([result, result_error, var_grad, op.dau_weights, op.dau_mu1, op.dau_mu2], feed_dict = {x: x_rand})

                t_end = time.time()
                print(t_end-t_start)

    def _run_DAUConv1d_forward_and_backward(self, repeat, N, W, H, S, F, dau_uints, max_kernel_size, max_offset_init):

        for i in range(repeat):
            mu_learning_rate_factor = 1000
            input_channels = S
            num_output = F
            sigma = 0.5
            x_rand = np.random.rand(N,input_channels,H,W)
            #x_rand = np.ones((16,num_output,32,32),dtype=np.float32)

            x = tf.placeholder(tf.float32, shape = x_rand.shape)

            op = DAUConv1d(filters=num_output,
                           dau_units=dau_uints,
                           max_kernel_size=max_kernel_size,
                           use_bias=False,
                           weight_initializer=tf.random_normal_initializer(stddev=0.1, dtype=np.float32),
                           mu1_initializer=tf.random_uniform_initializer(minval=-max_offset_init, maxval=max_offset_init,dtype=tf.float32),
                           #weight_initializer=tf.constant_initializer(1,dtype=np.float32),
                           #mu1_initializer=tf.constant_initializer(0,dtype=np.float32),
                           sigma_initializer=tf.constant_initializer(sigma),
                           mu_learning_rate_factor=mu_learning_rate_factor,
                           unit_testing=True)

            result = op(x)
            #result_error = tf.ones([np.int32(x.shape[0]),num_output,
            #                                 np.int32(x.shape[2]),
            #                                 np.int32(x.shape[3])],dtype=tf.float32)
            result_error = tf.random_normal([np.int32(x.shape[0]),num_output,
                                             np.int32(x.shape[2]),
                                             np.int32(x.shape[3])],dtype=tf.float32)

            var_grad = tf.gradients(result, [x, op.dau_weights, op.dau_mu1, op.dau_mu2], grad_ys=result_error)


            init = tf.global_variables_initializer()

            c = tf.ConfigProto(allow_soft_placement=True,
                               log_device_placement=True)
            c.gpu_options.visible_device_list = '2'
            c.gpu_options.allow_growth = True

            with tf.Session(config=c) as s:

                s.run(init)
                t_start = time.time()

                r, r_error, r_grad, w, mu1, mu2  = s.run([result, result_error, var_grad, op.dau_weights, op.dau_mu1, op.dau_mu2], feed_dict = {x: x_rand})

                t_end = time.time()
                print(t_end-t_start)

            gt_fwd_vals = DAUConvPython().forward_cpu(x=x_rand, w=w, mu1=mu1, mu2=mu2, single_dim_kernel=True,
                                                      sigma=[sigma], num_dau_units_ignore=op.num_dau_units_ignore)

            gt_bwd_vals = DAUConvPython().backward_cpu(x=x_rand, error=r_error, w=w, mu1=mu1,mu2=mu2, single_dim_kernel=True,
                                                       sigma=[sigma], num_dau_units_ignore=op.num_dau_units_ignore, unit_testing=True)

            # interpolation in C++ code at the right edge excludes one pixel so ignore those pixels in check
            r = r[:,:,:,:-2]
            r_grad[0] = r_grad[0][:,:,:,:-2]
            gt_fwd_vals = gt_fwd_vals[:,:,:,:-2]
            gt_bwd_vals = (gt_bwd_vals[0][:,:,:,:-2],
                           gt_bwd_vals[1],
                           gt_bwd_vals[2]* mu_learning_rate_factor,
                           gt_bwd_vals[3]* mu_learning_rate_factor)

            self._assertMatrix(r, gt_fwd_vals, 'fwd_output', rel_tolerance=0.01,plot_difference=True)

            self._assertMatrix(r_grad[0], gt_bwd_vals[0], 'bwd_error', rel_tolerance=0.01,plot_difference=True)
            self._assertMatrix(r_grad[1], gt_bwd_vals[1], 'bwd_w_grad', rel_tolerance=0.01, plot_difference=True)
            self._assertMatrix(r_grad[2], gt_bwd_vals[2], 'bwd_mu1_grad', rel_tolerance=0.01, plot_difference=True)
            self._assertMatrix(r_grad[3], gt_bwd_vals[3], 'bwd_mu2_grad', rel_tolerance=0.01, plot_difference=True)

    def test_DAUConv1d(self):

        # test small kernels (9 and 17)
        self._run_DAUConv1d_forward_and_backward(repeat=5, N=4, W=32, H=8, S=32, F=32, dau_uints=(2,2), max_kernel_size=9, max_offset_init=3)
        self._run_DAUConv1d_forward_and_backward(repeat=5, N=16, W=32, H=32, S=32, F=32, dau_uints=(2,2), max_kernel_size=17, max_offset_init=6)


if __name__ == '__main__':
    unittest.main()
