/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include "dau_conv/util/convolve.hpp"

#include <cassert>

namespace DAUConvNet
{


    template<typename Dtype, size_t baseDim, bool expand>
    void convolve(Dtype* out, const conv2_data_desc& out_desc,
                  const Dtype* signal, const conv2_data_desc& signal_desc,
                  const Dtype* filter, const conv2_data_desc& filter_desc,
                  cudaStream_t streamId ) {

        AF_BATCH_KIND kind;
        size_t sn = sizeof(signal_desc.dims) / sizeof(int);
        size_t fn = sizeof(filter_desc.dims) / sizeof(int);

        bool sn_stop = false, fn_stop = false;
        for (int i = 0; i < 4; ++i) {
            if (signal_desc.dims[i] <= 1 && !sn_stop)
                sn--;
            else
                sn_stop = true;

            if (filter_desc.dims[i] <= 1 && !fn_stop)
                fn--;
            else
                fn_stop = true;
        }

        if (sn == baseDim && fn == baseDim)
            kind = AF_BATCH_NONE;
        else if (sn == baseDim && (fn > baseDim && fn <= 4))
            kind = AF_BATCH_RHS;
        else if ((sn > baseDim && sn <= 4) && fn == baseDim)
            kind = AF_BATCH_LHS;
        else if ((sn > baseDim && sn <= 4) && (fn > baseDim && fn <= 4)) {
            bool doesDimensionsMatch = true;
            bool isInterleaved = true;
            for (int i = 3-baseDim; i >= 0; i--) {
                doesDimensionsMatch &= (signal_desc.dims[i] == filter_desc.dims[i]);
                isInterleaved &= (signal_desc.dims[i] == 1 || filter_desc.dims[i] == 1 || signal_desc.dims[i] == filter_desc.dims[i]);
            }
            if (doesDimensionsMatch) kind = AF_BATCH_SAME;
            else kind = (isInterleaved ? AF_BATCH_DIFF : AF_BATCH_UNSUPPORTED);
        } else
            kind = AF_BATCH_UNSUPPORTED;

        assert(kind != AF_BATCH_UNSUPPORTED && !(kind == AF_BATCH_DIFF && fn == 4));



        conv2_data_desc out_new_desc;
        if (expand) {
            for(size_t d=0; d<4; ++d) {
                if (kind==AF_BATCH_NONE || kind==AF_BATCH_RHS) {
                    out_new_desc.dims[d] = signal_desc.dims[d]+filter_desc.dims[d]-1;
                } else {
                    out_new_desc.dims[d] = (d>=baseDim ? signal_desc.dims[d]+filter_desc.dims[d]-1 : signal_desc.dims[d]);
                }
            }
        } else {
            out_new_desc = signal_desc;
            if (kind==AF_BATCH_RHS) {
                for (size_t i=0; i<4- baseDim; ++i) {
                    out_new_desc.dims[i] = filter_desc.dims[i];
                }
            } else if (kind == AF_BATCH_DIFF) {
                for (size_t i=0; i<4- baseDim; ++i) {
                    out_new_desc.dims[i] = signal_desc.dims[i] != 1 ? signal_desc.dims[i] : filter_desc.dims[i];
                }
            }
        }

        // ensure output of correct size or reshape
        bool reshape = false;
        for (size_t i=0; i<4; ++i) {
            reshape = reshape || out_new_desc.dims[i] != out_desc.dims[i];

        }
        if (reshape) {
            // out shape not consistent !!
            printf("Invalid output shape size, expetced shape size: %d,%d,%d,%d.\n", out_new_desc.dims[0], out_new_desc.dims[1], out_new_desc.dims[2], out_new_desc.dims[3]);
            throw std::exception();
        }


        kernel::convolve_nd<Dtype, baseDim, expand>(out, out_desc,
                                                    signal, signal_desc,
                                                    filter, filter_desc, kind, streamId);
    }


    template<typename Dtype>
    void caffe_gpu_convolve2(Dtype* out, const conv2_data_desc& out_desc,
                             const Dtype* signal, const conv2_data_desc& signal_desc,
                             const Dtype* filter, const conv2_data_desc& filter_desc, cudaStream_t streamId ) {
        return convolve<Dtype, 2, false>(out, out_desc,
                                         signal, signal_desc,
                                         filter, filter_desc,  streamId);
    }

    template void caffe_gpu_convolve2<float>(float * out, const conv2_data_desc& out_desc,
                                             const float* signal, const conv2_data_desc& signal_desc,
                                             const float* filter, const conv2_data_desc& filter_desc, cudaStream_t streamId );

    template<>
    void caffe_gpu_convolve2<double>(double* out, const conv2_data_desc& out_desc,
                                     const double* signal, const conv2_data_desc& signal_desc,
                                     const double* filter, const conv2_data_desc& filter_desc, cudaStream_t streamId ) {
        printf("Disabled compiling of caffe_gpu_convolve2 for double to speed-up compile.");
        throw std::exception();
    }
}
