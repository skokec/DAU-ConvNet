/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include "dau_conv/util/convolve.hpp"

#include "dau_conv/util/common.hpp"

#include <math_functions.h>
#include <cstdio>

namespace DAUConvNet
{

namespace kernel
{

#define divup(a, b) (((a)+(b)-1)/(b))

// we do not use CUDA_NUM_THREADS as 256 is more optimal for this function
static const int THREADS   = 256;

static const int THREADS_X = 16;
static const int THREADS_Y = 16;

//static const int CUBE_X    =  8;
//static const int CUBE_Y    =  8;
//static const int CUBE_Z    =  4;

// below shared MAX_*_LEN's are calculated based on
// a maximum shared memory configuration of 48KB per block
// considering complex types as well
static const int MAX_CONV1_FILTER_LEN = 129;
static const int MAX_CONV2_FILTER_LEN = 17;


// we shall declare the maximum size required of above all three cases
// and re-use the same constant memory locations for every case
__constant__ char cFilter[2*(2*(MAX_CONV1_FILTER_LEN-1)+CUDA_NUM_THREADS)*sizeof(double)];


template<typename T, bool expand, int fLen0, int fLen1, int fLen2, int fStr2>
__global__
void convolve2(T* out, const conv2_data_desc out_desc,
               const T* signal, const conv2_data_desc signal_desc,
               int nBBS0, int nBBS1, int o2, int o3, int s2, int s3)
{
    const size_t C_SIZE  = (THREADS_X+2*(fLen0-1))* (THREADS_Y+2*(fLen1-1));
    __shared__ T shrdMem[C_SIZE];

    const int radius0  = fLen0-1;
    const int radius1  = fLen1-1;
    const int padding0 = 2*radius0;
    const int padding1 = 2*radius1;
    const int shrdLen0 = THREADS_X + padding0;
    const int shrdLen1 = THREADS_Y + padding1;

    unsigned b0  = blockIdx.x / nBBS0;
    unsigned b1  = (blockIdx.y + blockIdx.z * gridDim.y) / nBBS1;
    T *dst = (T *)out+ (b0 * out_desc.strides[3-2] + /* activated with batched input signal */
                             o2 * out_desc.strides[3-2] + /* activated with batched input filter */
                             b1 * out_desc.strides[3-3] + /* activated with batched input signal */
                             o3 * out_desc.strides[3-3]); /* activated with batched input filter */

    const T *src = (const T *)signal + (b0 * signal_desc.strides[3-2] + /* activated with batched input signal */
                                            s2 * signal_desc.strides[3-2] + /* activated with batched input filter */
                                            b1 * signal_desc.strides[3-3] + /* activated with batched input signal */
                                            s3 * signal_desc.strides[3-3]); /* activated with batched input filter */

    const T *impulse  = (const T *)cFilter;

    int lx  = threadIdx.x;
    int ly  = threadIdx.y;
    int gx  = THREADS_X * (blockIdx.x-b0*nBBS0) + lx;
    int gy  = THREADS_Y * ((blockIdx.y + blockIdx.z * gridDim.y) -b1*nBBS1) + ly;

    if(b1 >= out_desc.dims[3-3])
        return;

    int s0 = signal_desc.strides[3-0];
    int s1 = signal_desc.strides[3-1];
    int d0 = signal_desc.dims[3-0];
    int d1 = signal_desc.dims[3-1];
    // below loops are traditional loops, they only run multiple
    // times filter length is more than launch size
#pragma unroll
    for (int b=ly, gy2=gy; b<shrdLen1; b+=THREADS_Y, gy2+=THREADS_Y) {
        int j = gy2-radius1;
        bool is_j  = j>=0 && j<d1;
        // move row_set THREADS_Y along coloumns
#pragma unroll
        for (int a=lx, gx2=gx; a<shrdLen0; a+=THREADS_X, gx2+=THREADS_X) {
            int i = gx2-radius0;
            bool is_i  = i>=0 && i<d0;
            shrdMem[b*shrdLen0+a] = (is_i && is_j ? src[i*s0+j*s1] : (T)0);
        }
    }
    __syncthreads();

    if (gx<out_desc.dims[3-0] && gy<out_desc.dims[3-1]) {
        int ci = lx + radius0 + (expand ? 0 : fLen0>>1);
        int cj = ly + radius1 + (expand ? 0 : fLen1>>1);

        T accum[fLen2];
        for (int fk = 0; fk < fLen2; ++fk) accum[fk] = T(0);
#pragma unroll
        for(int fj=0; fj<fLen1; ++fj) {
#pragma unroll
            for(int fi=0; fi<fLen0; ++fi) {
                T s_val = shrdMem[(cj-fj)*shrdLen0 + (ci-fi)];

                for (int fk = 0; fk < fLen2; ++fk) {
                    // CONVOLUTION
                    //T f_val = impulse[fj * fLen0 + fi + fk * (fStr2)];
                    // CORRELATION
                    T f_val = impulse[(fLen0-1-fj) * fLen0 + (fLen1-1 - fi) + fk * (fStr2)];

                    accum[fk] = accum[fk] + s_val * f_val;
                }
            }
        }
        for (int fk = 0; fk < fLen2; ++fk)
            dst[gy*out_desc.strides[3-1]+gx + fk * out_desc.strides[3-2] ] = (T)accum[fk];
    }
}

__inline__ __device__
int index(int i, int j, int k, int jstride, int kstride)
{
    return i+j*jstride+k*kstride;
}

struct conv_kparam_t {
    dim3              mBlocks;
    dim3             mThreads;
    size_t        mSharedSize;
    int           mBlk_x;
    int           mBlk_y;
    bool       outHasNoOffset;
    bool        inHasNoOffset;
    bool     launchMoreBlocks;
    int             o[3];
    int             s[3];
};

template<typename T>
void prepareKernelArgs(conv_kparam_t &params, const int* oDims, const int* fDims, int baseDim)
{
    int batchDims[4] = {1, 1, 1, 1};
    for(int i=0; i<4-baseDim; ++i) {
        batchDims[i] = (params.launchMoreBlocks ? 1 : oDims[i]);
    }

    const int maxBlocksY   = 64*1024-1; //cuda::getDeviceProp(cuda::getActiveDeviceId()).maxGridSize[1];
    if (baseDim==1) {
        // unsupported
    } else if (baseDim==2) {
        params.mThreads    = dim3(THREADS_X, THREADS_Y);
        params.mBlk_x      = divup(oDims[3-0], params.mThreads.x);
        params.mBlk_y      = divup(oDims[3-1], params.mThreads.y);
        params.mBlocks     = dim3(params.mBlk_x * batchDims[3-2], params.mBlk_y * batchDims[3-3]);
        params.mBlocks.z = divup(params.mBlocks.y, maxBlocksY);
        params.mBlocks.y = divup(params.mBlocks.y, params.mBlocks.z);
    } else if (baseDim==3) {
        // unsupported
    }
}


template<typename Dtype, bool expand, int f0, int f1, int f2>
void conv2Helper(const conv_kparam_t &p,
                 Dtype* out, const conv2_data_desc& out_desc,
                 const Dtype* sig, const conv2_data_desc& sig_desc,
                 cudaStream_t streamId)
{
    convolve2<Dtype, expand, f0, f1, f2, f0*f1><<<p.mBlocks, p.mThreads, 0, streamId>>>(out, out_desc, sig, sig_desc,
            p.mBlk_x, p.mBlk_y, p.o[1], p.o[2], p.s[1], p.s[2]);

    CUDA_POST_KERNEL_CHECK;
}

template<typename Dtype, bool expand, int f0, int f1>
void conv2Helper(const conv_kparam_t &p,
                 Dtype* out, const conv2_data_desc& out_desc,
                 const Dtype* sig, const conv2_data_desc& sig_desc,
                 int f2, cudaStream_t streamId)
{
    switch(f2) {
        case  1: conv2Helper<Dtype, expand, f0,  f1, 1>(p, out, out_desc, sig, sig_desc, streamId); break;
        case  3: conv2Helper<Dtype, expand, f0,  f1, 3>(p, out, out_desc, sig, sig_desc, streamId); break;
        case  4: conv2Helper<Dtype, expand, f0,  f1, 4>(p, out, out_desc, sig, sig_desc, streamId); break;
        default: printf("Unsupported filter batched filter third-dimention. Supported only [1 x K x K], [3 x K x K] and [4 x K x K].\n"); throw std::exception();
    }
}

template<typename Dtype, bool expand, int f0>
void conv2Helper(const conv_kparam_t &p,
                 Dtype* out, const conv2_data_desc& out_desc,
                 const Dtype* sig, const conv2_data_desc& sig_desc,
                 int f1, int f2, cudaStream_t streamId)
{
    switch(f1) {
        case  1: conv2Helper<Dtype, expand, f0,  1>(p, out, out_desc, sig, sig_desc, f2, streamId); break;
        case  2: conv2Helper<Dtype, expand, f0,  2>(p, out, out_desc, sig, sig_desc, f2, streamId); break;
        case  3: conv2Helper<Dtype, expand, f0,  3>(p, out, out_desc, sig, sig_desc, f2, streamId); break;
        case  4: conv2Helper<Dtype, expand, f0,  4>(p, out, out_desc, sig, sig_desc, f2, streamId); break;
        case  5: conv2Helper<Dtype, expand, f0,  5>(p, out, out_desc, sig, sig_desc, f2, streamId); break;
        default: printf("Unsupported filter size in caffe_gpu_convolve2. Supported up to 5x5 when unmatched witdh/height sizes.\n"); throw std::exception();
    }
}

template<typename Dtype, bool expand>
void conv2Helper(const conv_kparam_t &p,
                 Dtype* out, const conv2_data_desc& out_desc,
                 const Dtype* sig, const conv2_data_desc& sig_desc,
                 int f0, int f1, int f2, cudaStream_t streamId)
{
    switch(f0) {
        case  1: conv2Helper<Dtype, expand,  1>(p, out, out_desc, sig, sig_desc, f1, f2, streamId); break;
        case  2: conv2Helper<Dtype, expand,  2>(p, out, out_desc, sig, sig_desc, f1, f2, streamId); break;
        case  3: conv2Helper<Dtype, expand,  3>(p, out, out_desc, sig, sig_desc, f1, f2, streamId); break;
        case  4: conv2Helper<Dtype, expand,  4>(p, out, out_desc, sig, sig_desc, f1, f2, streamId); break;
        case  5: conv2Helper<Dtype, expand,  5>(p, out, out_desc, sig, sig_desc, f1, f2, streamId); break;
        default: {
                     if (f0==f1) {
                         switch(f1) {
                             case  6: conv2Helper<Dtype, expand,  6,  6>(p, out, out_desc, sig, sig_desc, f2, streamId); break;
                             case  7: conv2Helper<Dtype, expand,  7,  7>(p, out, out_desc, sig, sig_desc, f2, streamId); break;
                             case  8: conv2Helper<Dtype, expand,  8,  8>(p, out, out_desc, sig, sig_desc, f2, streamId); break;
                             case  9: conv2Helper<Dtype, expand,  9,  9>(p, out, out_desc, sig, sig_desc, f2, streamId); break;
                             case 10: conv2Helper<Dtype, expand, 10, 10>(p, out, out_desc, sig, sig_desc, f2, streamId); break;
                             case 11: conv2Helper<Dtype, expand, 11, 11>(p, out, out_desc, sig, sig_desc, f2, streamId); break;
                             case 12: conv2Helper<Dtype, expand, 12, 12>(p, out, out_desc, sig, sig_desc, f2, streamId); break;
                             case 13: conv2Helper<Dtype, expand, 13, 13>(p, out, out_desc, sig, sig_desc, f2, streamId); break;
                             case 14: conv2Helper<Dtype, expand, 14, 14>(p, out, out_desc, sig, sig_desc, f2, streamId); break;
                             case 15: conv2Helper<Dtype, expand, 15, 15>(p, out, out_desc, sig, sig_desc, f2, streamId); break;
                             case 16: conv2Helper<Dtype, expand, 16, 16>(p, out, out_desc, sig, sig_desc, f2, streamId); break;
                             case 17: conv2Helper<Dtype, expand, 17, 17>(p, out, out_desc, sig, sig_desc, f2, streamId); break;
                             default: printf("Unsupported filter size in caffe_gpu_convolve2. Supported up to 17x17.\n"); throw std::exception();
                         }
                     } else {
                         printf("Unsupported filter size in caffe_gpu_convolve2. Supported up to 5x5 when unmatched witdh/height sizes.\n"); throw std::exception();
                     }
                 } break;
    }
}

template<typename Dtype, bool expand>
void convolve_2d(conv_kparam_t &p,
                 Dtype* out, const conv2_data_desc& out_desc,
                 const Dtype* signal, const conv2_data_desc& signal_desc,
                 const Dtype* filt, const conv2_data_desc& filt_desc, cudaStream_t streamId)
{
    prepareKernelArgs<Dtype>(p, signal_desc.dims, filt_desc.dims, 2);

    int filterLen = filt_desc.dims[3-0] * filt_desc.dims[3-1];

    for (int b3=0; b3<filt_desc.dims[3-3]; ++b3) {
        int f3Off = b3 * filt_desc.strides[3-3];

        if (filt_desc.strides[3-2] == filt_desc.dims[3-0] *  filt_desc.dims[3-1]) {
            // if filter is not strrided we can use single kernel

            CUDA_CHECK(cudaMemcpyToSymbolAsync(kernel::cFilter,
                                               filt+(f3Off),
                                               filterLen * filt_desc.dims[3-2] *sizeof(Dtype),
                                               0, cudaMemcpyDeviceToDevice, streamId));

            p.o[1] = 0;
            p.o[2] = (p.outHasNoOffset ? 0 : b3);
            p.s[1] = 0;
            p.s[2] = (p.inHasNoOffset ? 0 : b3);

            conv2Helper<Dtype, expand>(p, out, out_desc, signal, signal_desc, filt_desc.dims[3-0], filt_desc.dims[3-1], filt_desc.dims[3-2], streamId);

        } else {
            for (int b2=0; b2<filt_desc.dims[3-2]; ++b2) {
                int f2Off = b2 * filt_desc.strides[3-2];

                // FIXME: if the filter array is strided, direct copy of symbols
                // might cause issues
                CUDA_CHECK(cudaMemcpyToSymbolAsync(kernel::cFilter,
                                                   filt+(f2Off+f3Off),
                                                   filterLen*sizeof(Dtype),
                                                   0, cudaMemcpyDeviceToDevice, streamId));

                p.o[1] = (p.outHasNoOffset ? 0 : b2);
                p.o[2] = (p.outHasNoOffset ? 0 : b3);
                p.s[1] = (p.inHasNoOffset ? 0 : b2);
                p.s[2] = (p.inHasNoOffset ? 0 : b3);

                conv2Helper<Dtype, expand>(p, out, out_desc, signal, signal_desc, filt_desc.dims[3-0], filt_desc.dims[3-1], 1, streamId);
            }
        }


    }
}


template<typename Dtype, int baseDim, bool expand>
void convolve_nd(Dtype* out, const conv2_data_desc& out_desc,
                 const Dtype* signal, const conv2_data_desc& signal_desc,
                 const Dtype* filt, const conv2_data_desc& filt_desc,
                 AF_BATCH_KIND kind, cudaStream_t streamId)
{
    bool callKernel = true;


    int MCFL2 = kernel::MAX_CONV2_FILTER_LEN;
    switch(baseDim) {
        case 2: if ((filt_desc.dims[3]*filt_desc.dims[2]) > (MCFL2 * MCFL2)) callKernel = false; break;
    }

    if (!callKernel) {
        printf("Unsupported filter dimension. Supported only 2-dimensional filter with third dimension as batch.\n"); throw std::exception();
    }

    conv_kparam_t param;
    for (int i=0; i<3; ++i) {
        param.o[i] = 0;
        param.s[i] = 0;
    }
    param.launchMoreBlocks = kind==AF_BATCH_SAME || kind==AF_BATCH_RHS;
    param.outHasNoOffset   = kind==AF_BATCH_LHS  || kind==AF_BATCH_NONE;
    param.inHasNoOffset    = kind!=AF_BATCH_SAME;

    switch(baseDim) {
        case 2: convolve_2d<Dtype, expand>(param, out, out_desc, signal, signal_desc, filt, filt_desc, streamId); break;
    }

}

#define INSTANTIATE(T)  \
    template void convolve_nd<T, 2, true >(T* out, const conv2_data_desc& out_desc,\
                                           const T* signal, const conv2_data_desc& signal_desc,\
                                           const T* filt, const conv2_data_desc& filt_desc, AF_BATCH_KIND kind, cudaStream_t streamId); \
    template void convolve_nd<T, 2, false>(T* out, const conv2_data_desc& out_desc, \
                                           const T* signal, const conv2_data_desc& signal_desc, \
                                           const T* filt, const conv2_data_desc& filt_desc, AF_BATCH_KIND kind, cudaStream_t streamId);\


INSTANTIATE(float)

}

}