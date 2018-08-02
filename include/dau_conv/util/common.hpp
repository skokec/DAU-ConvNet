//
// Created by domen on 3/23/18.
//

#ifndef DAUCONVNET_COMMON_H
#define DAUCONVNET_COMMON_H


#include <climits>
#include <cmath>
#include <fstream>  // NOLINT(readability/streams)
#include <iostream>  // NOLINT(readability/streams)
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <utility>  // pair
#include <vector>
#include <memory.h>
#include <memory>
#include <iostream>
#include <cstdio>

#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <driver_types.h>  // cuda driver types

#ifndef NDEBUG
#   define M_Assert(Expr, Msg) \
    __M_Assert(#Expr, Expr, __FILE__, __LINE__, Msg)

#else
#   define M_Assert(Expr, Msg)
#endif

void __M_Assert(const char* expr_str, bool expr, const char* file, int line, const char* msg);


//
// CUDA macros
//

// CUDA: various checks for different function calls.
#ifndef CUDA_CHECK
#define CUDA_CHECK(condition) \
  /* Code block avoids redefinition of cudaError_t error */ \
  do { \
    cudaError_t error = condition; \
    M_Assert(error == cudaSuccess, cudaGetErrorString(error)); \
  } while (0)
#endif

#ifndef CUBLAS_CHECK
#define CUBLAS_CHECK(condition) \
  do { \
    cublasStatus_t status = condition; \
    M_Assert(status == CUBLAS_STATUS_SUCCESS, DAUConvNet::cublasGetErrorString(status)); \
  } while (0)
#endif

// CUDA: grid stride looping
#ifndef CUDA_KERNEL_LOOP
#define CUDA_KERNEL_LOOP(i, n) \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; \
       i < (n); \
       i += blockDim.x * gridDim.x)
#endif

// CUDA: check for error after kernel execution and exit loudly if there is one.
#ifndef CUDA_POST_KERNEL_CHECK
#define CUDA_POST_KERNEL_CHECK CUDA_CHECK(cudaPeekAtLastError())
#endif


namespace DAUConvNet {

// Common functions and classes from std that dau_conv_impl often uses.
    using std::vector;

// CUDA: library error reporting.
    const char* cublasGetErrorString(cublasStatus_t error);

// CUDA: use 512 threads per block
    const int CUDA_NUM_THREADS = 512;

// CUDA: number of blocks for threads.
    inline int CUDA_GET_BLOCKS(const int N) {
        return (N + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS;
    }

    class DAUException : public std::runtime_error {
    public:
        DAUException(const std::string& what_arg ) : std::runtime_error(what_arg) {
        }
    };

}  // namespace DAUConvNet

template<typename ... Args>
std::string string_format( const std::string& format, Args ... args )
{
    size_t size = std::snprintf( nullptr, 0, format.c_str(), args ... ) + 1; // Extra space for '\0'
    std::unique_ptr<char[]> buf( new char[ size ] );
    std::snprintf( buf.get(), size, format.c_str(), args ... );
    return std::string( buf.get(), size - 1 ); // We don't want the '\0' inside
}


#ifndef CHECK
#define CHECK(Expr,Msg ) \
 if ((Expr) == false) { throw DAUConvNet::DAUException(string_format("ASSERT ERROR: %s\n", Msg)); }

#endif




#endif //DAUCONVNET_COMMON_H
