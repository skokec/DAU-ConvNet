#include <math_functions.h>  // CUDA's, not dau_conv_impl's, for fabs, signbit

#include <cmath>

#include "dau_conv/util/common.hpp"
#include "dau_conv/util/math_functions.hpp"

#include <cub/cub/cub.cuh>

namespace DAUConvNet {

template <>
void caffe_gpu_gemm<float>(const CBLAS_TRANSPOSE TransA,
    const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
    const float alpha, const float* A, const float* B, const float beta,
    float* C, cublasHandle_t cublas_handle) {
  // Note that cublas follows fortran order.
  int lda = (TransA == CblasNoTrans) ? K : M;
  int ldb = (TransB == CblasNoTrans) ? N : K;
  cublasOperation_t cuTransA =
      (TransA == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
  cublasOperation_t cuTransB =
      (TransB == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
  CUBLAS_CHECK(cublasSgemm(cublas_handle, cuTransB, cuTransA,
      N, M, K, &alpha, B, ldb, A, lda, &beta, C, N));
}

template <>
void caffe_gpu_gemm<double>(const CBLAS_TRANSPOSE TransA,
    const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
    const double alpha, const double* A, const double* B, const double beta,
    double* C, cublasHandle_t cublas_handle) {
  // Note that cublas follows fortran order.
  int lda = (TransA == CblasNoTrans) ? K : M;
  int ldb = (TransB == CblasNoTrans) ? N : K;
  cublasOperation_t cuTransA =
      (TransA == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
  cublasOperation_t cuTransB =
      (TransB == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
  CUBLAS_CHECK(cublasDgemm(cublas_handle, cuTransB, cuTransA,
      N, M, K, &alpha, B, ldb, A, lda, &beta, C, N));
}

template <>
void caffe_gpu_gemv<float>(const CBLAS_TRANSPOSE TransA, const int M,
    const int N, const float alpha, const float* A, const float* x,
    const float beta, float* y, cublasHandle_t cublas_handle) {
  cublasOperation_t cuTransA =
      (TransA == CblasNoTrans) ? CUBLAS_OP_T : CUBLAS_OP_N;
  CUBLAS_CHECK(cublasSgemv(cublas_handle, cuTransA, N, M, &alpha,
      A, N, x, 1, &beta, y, 1));
}

template <>
void caffe_gpu_gemv<double>(const CBLAS_TRANSPOSE TransA, const int M,
    const int N, const double alpha, const double* A, const double* x,
    const double beta, double* y, cublasHandle_t cublas_handle) {
  cublasOperation_t cuTransA =
      (TransA == CblasNoTrans) ? CUBLAS_OP_T : CUBLAS_OP_N;
  CUBLAS_CHECK(cublasDgemv(cublas_handle, cuTransA, N, M, &alpha,
      A, N, x, 1, &beta, y, 1));
}

template <>
void caffe_gpu_axpy<float>(const int N, const float alpha, const float* X,
    float* Y, cublasHandle_t cublas_handle) {
  CUBLAS_CHECK(cublasSaxpy(cublas_handle, N, &alpha, X, 1, Y, 1));
}

template <>
void caffe_gpu_axpy<double>(const int N, const double alpha, const double* X,
    double* Y, cublasHandle_t cublas_handle) {
  CUBLAS_CHECK(cublasDaxpy(cublas_handle, N, &alpha, X, 1, Y, 1));
}

void caffe_gpu_memcpy(const size_t N, const void* X, void* Y) {
  if (X != Y) {
    CUDA_CHECK(cudaMemcpy(Y, X, N, cudaMemcpyDefault));  // NOLINT(dau_conv_impl/alt_fn)
  }
}

template <>
void caffe_gpu_scal<float>(const int N, const float alpha, float *X, cublasHandle_t cublas_handle) {
  CUBLAS_CHECK(cublasSscal(cublas_handle, N, &alpha, X, 1));
}

template <>
void caffe_gpu_scal<double>(const int N, const double alpha, double *X, cublasHandle_t cublas_handle) {
  CUBLAS_CHECK(cublasDscal(cublas_handle, N, &alpha, X, 1));
}

template <>
void caffe_gpu_scal<float>(const int N, const float alpha, float* X,
                           cublasHandle_t cublas_handle, cudaStream_t str) {
  cudaStream_t initial_stream;
  CUBLAS_CHECK(cublasGetStream(cublas_handle, &initial_stream));
  CUBLAS_CHECK(cublasSetStream(cublas_handle, str));
  CUBLAS_CHECK(cublasSscal(cublas_handle, N, &alpha, X, 1));
  CUBLAS_CHECK(cublasSetStream(cublas_handle, initial_stream));
}

template <>
void caffe_gpu_scal<double>(const int N, const double alpha, double* X,
                            cublasHandle_t cublas_handle, cudaStream_t str) {
  cudaStream_t initial_stream;
  CUBLAS_CHECK(cublasGetStream(cublas_handle, &initial_stream));
  CUBLAS_CHECK(cublasSetStream(cublas_handle, str));
  CUBLAS_CHECK(cublasDscal(cublas_handle, N, &alpha, X, 1));
  CUBLAS_CHECK(cublasSetStream(cublas_handle, initial_stream));
}

template <>
void caffe_gpu_axpby<float>(const int N, const float alpha, const float* X,
    const float beta, float* Y, cublasHandle_t cublas_handle) {
  caffe_gpu_scal<float>(N, beta, Y, cublas_handle);
  caffe_gpu_axpy<float>(N, alpha, X, Y, cublas_handle);
}

template <>
void caffe_gpu_axpby<double>(const int N, const double alpha, const double* X,
    const double beta, double* Y, cublasHandle_t cublas_handle) {
  caffe_gpu_scal<double>(N, beta, Y, cublas_handle);
  caffe_gpu_axpy<double>(N, alpha, X, Y, cublas_handle);
}

template <>
void caffe_gpu_dot<float>(const int n, const float* x, const float* y,
    float* out, cublasHandle_t cublas_handle) {
  CUBLAS_CHECK(cublasSdot(cublas_handle, n, x, 1, y, 1, out));
}

template <>
void caffe_gpu_dot<double>(const int n, const double* x, const double* y,
    double * out, cublasHandle_t cublas_handle) {
  CUBLAS_CHECK(cublasDdot(cublas_handle, n, x, 1, y, 1, out));
}



template <>
void caffe_gpu_asum<float>(const int n, const float* x, float* y, cublasHandle_t cublas_handle) {
  CUBLAS_CHECK(cublasSasum(cublas_handle, n, x, 1, y));
}

template <>
void caffe_gpu_asum<double>(const int n, const double* x, double* y, cublasHandle_t cublas_handle) {
  CUBLAS_CHECK(cublasDasum(cublas_handle, n, x, 1, y));
}

template <typename Dtype>
__global__ void set_kernel(const int n, const Dtype alpha, Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = alpha;
  }
}

template <typename Dtype>
void caffe_gpu_set(const int N, const Dtype alpha, Dtype* Y) {
  if (alpha == 0) {
    CUDA_CHECK(cudaMemset(Y, 0, sizeof(Dtype) * N));  // NOLINT(dau_conv_impl/alt_fn)
    return;
  }
  // NOLINT_NEXT_LINE(whitespace/operators)
  set_kernel<Dtype><<<CUDA_GET_BLOCKS(N), CUDA_NUM_THREADS>>>(
      N, alpha, Y);
}

template void caffe_gpu_set<int>(const int N, const int alpha, int* Y);
template void caffe_gpu_set<float>(const int N, const float alpha, float* Y);
template void caffe_gpu_set<double>(const int N, const double alpha, double* Y);

template <typename Dtype>
__global__ void add_scalar_kernel(const int n, const Dtype alpha, Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] += alpha;
  }
}

template <>
void caffe_gpu_add_scalar(const int N, const float alpha, float* Y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  add_scalar_kernel<float><<<CUDA_GET_BLOCKS(N), CUDA_NUM_THREADS>>>(
      N, alpha, Y);
}

template <>
void caffe_gpu_add_scalar(const int N, const double alpha, double* Y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  add_scalar_kernel<double><<<CUDA_GET_BLOCKS(N), CUDA_NUM_THREADS>>>(
      N, alpha, Y);
}

    template <typename Dtype>
__global__ void mul_kernel(const int n, const Dtype* a,
    const Dtype* b, Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = a[index] * b[index];
  }
}

template <>
void caffe_gpu_mul<float>(const int N, const float* a,
    const float* b, float* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  mul_kernel<float><<<CUDA_GET_BLOCKS(N), CUDA_NUM_THREADS>>>(N, a, b, y);
}

template <>
void caffe_gpu_mul<double>(const int N, const double* a,
    const double* b, double* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  mul_kernel<double><<<CUDA_GET_BLOCKS(N), CUDA_NUM_THREADS>>>(N, a, b, y);
}



void caffe_gpu_memcpy_async(const size_t N, const void* X, void* Y, cudaStream_t streamId) {
  if (X != Y) {
    CUDA_CHECK(cudaMemcpyAsync(Y, X, N, cudaMemcpyDefault));  // NOLINT(dau_conv_impl/alt_fn)
  }
}

template <typename Dtype>
void caffe_gpu_set_async(const int N, const Dtype alpha, Dtype* Y, cudaStream_t streamId) {
  if (alpha == 0) {
    //CUDA_CHECK(cudaMemsetAsync(Y, 0, sizeof(Dtype) * N, streamId));  // NOLINT(dau_conv_impl/alt_fn)
    //return;
  }
  // NOLINT_NEXT_LINE(whitespace/operators)
  set_kernel<Dtype><<<CUDA_GET_BLOCKS(N), CUDA_NUM_THREADS, 0, streamId>>>(
          N, alpha, Y);
  CUDA_POST_KERNEL_CHECK;
}

template void caffe_gpu_set_async<int>(const int N, const int alpha, int* Y, cudaStream_t streamId);
template void caffe_gpu_set_async<float>(const int N, const float alpha, float* Y, cudaStream_t streamId);
template void caffe_gpu_set_async<double>(const int N, const double alpha, double* Y, cudaStream_t streamId);


#include <stdio.h>

__global__ void clip_lower_kernel_double(const int n, const double lower_bound, const double* x, double* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = fmax(x[index], lower_bound);
  }
}

__global__ void clip_lower_kernel_float(const int n, const float lower_bound, const float* x, float* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = fmaxf(x[index], lower_bound);
  }
}

template <>
void caffe_gpu_clip_lower<float>(const int N, const float lower_bound, const float* x, float* y, cudaStream_t streamId) {
  clip_lower_kernel_float<<<CUDA_GET_BLOCKS(N), CUDA_NUM_THREADS, 0, streamId>>>(N, lower_bound, x, y);
  CUDA_POST_KERNEL_CHECK;
}

template <>
void caffe_gpu_clip_lower<double>(const int N, const double lower_bound, const double* x, double* y, cudaStream_t streamId) {
  clip_lower_kernel_double<<<CUDA_GET_BLOCKS(N), CUDA_NUM_THREADS, 0, streamId>>>(N, lower_bound, x, y);
  CUDA_POST_KERNEL_CHECK;
}


__global__ void clip_upper_kernel_double(const int n, const double lower_bound, const double* x, double* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = fmin(x[index], lower_bound);
  }
}

__global__ void clip_upper_kernel_float(const int n, const float lower_bound, const float* x, float* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = fminf(x[index], lower_bound);
  }
}

template <>
void caffe_gpu_clip_upper<float>(const int N, const float upper_bound, const float* x, float* y, cudaStream_t streamId) {
  clip_upper_kernel_float<<<CUDA_GET_BLOCKS(N), CUDA_NUM_THREADS, 0, streamId>>>(N, upper_bound, x, y);
  CUDA_POST_KERNEL_CHECK;
}

template <>
void caffe_gpu_clip_upper<double>(const int N, const double upper_bound, const double* x, double* y, cudaStream_t streamId) {
  clip_upper_kernel_double<<<CUDA_GET_BLOCKS(N), CUDA_NUM_THREADS, 0, streamId>>>(N, upper_bound, x, y);
  CUDA_POST_KERNEL_CHECK;
}


template <typename Dtype>
__global__ void clip_eps_kernel(const int n, const Dtype eps_bound, const Dtype* x, Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
    Dtype val = x[index];
    y[index] = abs(val) > eps_bound ? val : 0;
  }
}


template <>
void caffe_gpu_clip_eps<float>(const int N, const float eps_bound, const float* x, float* y, cudaStream_t streamId) {
  clip_eps_kernel<float><<<CUDA_GET_BLOCKS(N), CUDA_NUM_THREADS, 0, streamId>>>(N, eps_bound, x, y);
  CUDA_POST_KERNEL_CHECK;
}
template <>
void caffe_gpu_clip_eps<double>(const int N, const double eps_bound, const double* x, double* y, cudaStream_t streamId) {
  clip_eps_kernel<double><<<CUDA_GET_BLOCKS(N), CUDA_NUM_THREADS, 0, streamId>>>(N, eps_bound, x, y);
  CUDA_POST_KERNEL_CHECK;
}



template <typename Dtype>
void caffe_gpu_sum(const int n, const Dtype* x, Dtype* y, const int m, cudaStream_t streamId) {
  M_Assert(n % m == 0, "Invalid size in caffe_gpu_sum: n should be a multiple of m");
  int num_segments = n/m;

  int* offsets = new int[num_segments + 1];

  offsets[0] = 0;

  for (int i = 0; i < num_segments; i++) offsets[i+1] = m*(i+1);

  int* offsets_d;
  CUDA_CHECK(cudaMalloc(&offsets_d, sizeof(int)*(num_segments+1)));

  caffe_gpu_memcpy_async(sizeof(int)*(num_segments + 1), offsets, offsets_d);

  caffe_gpu_sum(n, x, y, num_segments, offsets_d, false);

  CUDA_CHECK(cudaFree(offsets_d));

  delete offsets;
}



template <typename Dtype>
void caffe_gpu_sum(const int n, const Dtype* x, Dtype* y, const int num_segments, int* offsets_gpu, bool with_add, cudaStream_t streamId) {

  // DeviceSegmentedReduce in version 1.5.1 always returns temp_storage_bytes=1 and never actually uses allocated storage
  // so we can just use non-zero value for temp storage and avoid getting temp_storage_bytes size
  size_t temp_storage_bytes = 0;
  void* temp_storage_d = (void*)1;

  if (with_add)
    M_Assert(false, "caffe_gpu_sum does not support with_add=true any more ");
  else
    CUDA_CHECK(cub::DeviceSegmentedReduce::Sum(temp_storage_d, temp_storage_bytes, x, y,  num_segments, offsets_gpu, offsets_gpu + 1, streamId, false));
}


template void caffe_gpu_sum<float>(const int n, const float* x, float* y, const int num_segments, int* offsets_gpu, bool with_add, cudaStream_t streamId);
template void caffe_gpu_sum<double>(const int n, const double* x, double* y, const int num_segments, int* offsets_gpu, bool with_add, cudaStream_t streamId);

template void caffe_gpu_sum<float>(const int n, const float* x, float* y, const int m, cudaStream_t streamId);
template void caffe_gpu_sum<double>(const int n, const double* x, double* y, const int m, cudaStream_t streamId);



#define OFFSET3(k,j,i, num_k, num_j, num_i) ((((k)) * (num_j) + (j))*(num_i) + (i) )
#define OFFSET4(l,k,j,i, num_l, num_k, num_j, num_i) ((( (l)*(num_k) + (k)) * (num_j) + (j))*(num_i) + (i) )

template <typename Dtype>
__global__ void pad2d_kernel(const int N, const int H, const int W, const int pad, const Dtype* x, Dtype* y) {
  CUDA_KERNEL_LOOP(index, N*H*W) {
    int w = (index % W) ;
    int nh = index / W;
    int h = (nh % H);
    int n = nh / H;

    y[OFFSET4(0,n,h + pad,w+pad, 1, N, H+2*pad,W+2*pad)] = x[index];
  }
}


template <typename Dtype>
void caffe_gpu_pad2d(const int I, const int H, const int W, int pad_size, const Dtype* X, Dtype* Y, cudaStream_t streamId) {
  pad2d_kernel<Dtype><<<CUDA_GET_BLOCKS(I*H*W), CUDA_NUM_THREADS, 0, streamId>>>(I, H, W, pad_size, X, Y);
}

template void caffe_gpu_pad2d(const int I, const int H, const int W, int pad_size, const float* X, float* Y, cudaStream_t streamId);
template void caffe_gpu_pad2d(const int I, const int H, const int W, int pad_size, const double* X, double* Y, cudaStream_t streamId);


}  // namespace dau_conv_impl
