/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include "dau_conv/util/common.hpp"

namespace DAUConvNet
{
    typedef enum {
        AF_BATCH_UNSUPPORTED = -1, /* invalid inputs */
        AF_BATCH_NONE,             /* one signal, one filter   */
        AF_BATCH_LHS,              /* many signal, one filter  */
        AF_BATCH_RHS,              /* one signal, many filter  */
        AF_BATCH_SAME,             /* signal and filter have same batch size */
        AF_BATCH_DIFF,             /* signal and filter have different batch size */
    } AF_BATCH_KIND;

    struct conv2_data_desc {
        conv2_data_desc() {}
        conv2_data_desc(int n, int c, int h, int w, int s_n, int s_c, int s_h, int s_w)  {
            dims[0] = n; dims[1] = c; dims[2] = h; dims[3] = w;
            strides[0] = s_n; strides[1] = s_c; strides[2] = s_h; strides[3] = s_w;
        }
        int dims[4];
        int strides[4];
    };

    template<typename Dtype>
    void caffe_gpu_convolve2(Dtype* out, const conv2_data_desc& out_desc,
                             const Dtype* signal, const conv2_data_desc& signal_desc,
                             const Dtype* filter, const conv2_data_desc& filter_desc, cudaStream_t streamId = 0);

namespace kernel
{



template<typename Dtype, int baseDim, bool expand>
void convolve_nd(Dtype* out, const conv2_data_desc& out_desc,
                 const Dtype* signal, const conv2_data_desc& signal_desc,
                 const Dtype* filter, const conv2_data_desc& filter_desc,
                 AF_BATCH_KIND kind, cudaStream_t streamId = 0);

}

}
