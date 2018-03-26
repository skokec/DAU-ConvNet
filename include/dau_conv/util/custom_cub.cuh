/******************************************************************************
 * Copyright (c) 2011, Duane Merrill.  All rights reserved.
 * Copyright (c) 2011-2015, NVIDIA CORPORATION.  All rights reserved.
 * 
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 * 
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 ******************************************************************************/

#pragma once

#include <iterator>
#include <iostream>

#include <cub/cub/cub.cuh>

#include <cub/cub/thread/thread_load.cuh>
#include <cub/cub/thread/thread_store.cuh>
#include <cub/cub/util_device.cuh>
#include <cub/cub/util_namespace.cuh>

#if (THRUST_VERSION >= 100700)
    // This iterator is compatible with Thrust API 1.7 and newer
    #include <thrust/iterator/iterator_facade.h>
    #include <thrust/iterator/iterator_traits.h>
#endif // THRUST_VERSION


/// Optional outer namespace(s)
namespace DAUConvNet {

/// CUB namespace
namespace custom_cub {

struct Mul
{
    /// Boolean sum operator, returns <tt>a + b</tt>
    template <typename T>
    __host__ __device__ __forceinline__ T operator()(const T &a, const T &b) const
    {
        return a * b;
    }
};

/**
 * \brief A random-access input wrapper for transforming dereferenced values.
 *
 * \par Overview
 * - TransformInputIteratorTwraps a unary conversion functor of type \p
 *   ConversionOp and a random-access input iterator of type <tt>InputIteratorT</tt>,
 *   using the former to produce references of type \p ValueType from the latter.
 * - Can be used with any data type.
 * - Can be constructed, manipulated, and exchanged within and between host and device
 *   functions.  Wrapped host memory can only be dereferenced on the host, and wrapped
 *   device memory can only be dereferenced on the device.
 * - Compatible with Thrust API v1.7 or newer.
 *
 * \par Snippet
 * The code snippet below illustrates the use of \p TransformInputIteratorTto
 * dereference an array of integers, tripling the values and converting them to doubles.
 * \par
 * \code
 * #include <cub/cub.cuh>   // or equivalently <cub/iterator/transform_input_iterator.cuh>
 *
 * // Functor for tripling integer values and converting to doubles
 * struct TripleDoubler
 * {
 *     __host__ __device__ __forceinline__
 *     double operator()(const int &a) const {
 *         return double(a * 2);
 *     }
 * };
 *
 * // Declare, allocate, and initialize a device array
 * int *d_in;                   // e.g., [8, 6, 7, 5, 3, 0, 9]
 * TripleDoubler conversion_op;
 *
 * // Create an iterator wrapper
 * cub::BinaryTransformInputIterator<double, TripleDoubler, int*> itr(d_in, conversion_op);
 *
 * // Within device code:
 * printf("%f\n", itr[0]);  // 24.0
 * printf("%f\n", itr[1]);  // 18.0
 * printf("%f\n", itr[6]);  // 27.0
 *
 * \endcode
 *
 * \tparam ValueType            The value type of this iterator
 * \tparam ConversionOp         Unary functor type for mapping objects of type \p InputType to type \p ValueType.  Must have member <tt>ValueType operator()(const InputType &datum)</tt>.
 * \tparam InputIteratorT       The type of the wrapped input iterator
 * \tparam OffsetT              The difference type of this iterator (Default: \p ptrdiff_t)
 *
 */
template <
    typename ValueType,
    typename ConversionOp,
    typename InputIteratorT1,
	typename InputIteratorT2,
    typename OffsetT = ptrdiff_t>
class BinaryTransformInputIterator
{
public:

    // Required iterator traits
    typedef BinaryTransformInputIterator              self_type;              ///< My own type
    typedef OffsetT                             difference_type;        ///< Type to express the result of subtracting one iterator from another
    typedef ValueType                           value_type;             ///< The type of the element the iterator can point to
    typedef ValueType*                          pointer;                ///< The type of a pointer to an element the iterator can point to
    typedef ValueType                           reference;              ///< The type of a reference to an element the iterator can point to

#if (THRUST_VERSION >= 100700)
    // Use Thrust's iterator categories so we can use these iterators in Thrust 1.7 (or newer) methods
    typedef typename thrust::detail::iterator_facade_category<
        thrust::any_system_tag,
        thrust::random_access_traversal_tag,
        value_type,
        reference
      >::type iterator_category;                                        ///< The iterator category
#else
    typedef std::random_access_iterator_tag     iterator_category;      ///< The iterator category
#endif  // THRUST_VERSION

private:

    ConversionOp    conversion_op;
    InputIteratorT1  input_itr_a;
    InputIteratorT2  input_itr_b;

public:

    /// Constructor
    __host__ __device__ __forceinline__ BinaryTransformInputIterator(
    InputIteratorT1      input_itr_a,          ///< Input iterator to wrap
	InputIteratorT2      input_itr_b,          ///< Input iterator to wrap
        ConversionOp        conversion_op)      ///< Conversion functor to wrap
    :
        conversion_op(conversion_op),
        input_itr_a(input_itr_a),
		input_itr_b(input_itr_b)
    {}

    /// Postfix increment
    __host__ __device__ __forceinline__ self_type operator++(int)
    {
        self_type retval = *this;
        input_itr_a++;
        input_itr_b++;
        return retval;
    }

    /// Prefix increment
    __host__ __device__ __forceinline__ self_type operator++()
    {
        input_itr_a++;
        input_itr_b++;
        return *this;
    }

    /// Indirection
    __host__ __device__ __forceinline__ reference operator*() const
    {
        return conversion_op(*input_itr_a, *input_itr_b);
    }

    /// Addition
    template <typename Distance>
    __host__ __device__ __forceinline__ self_type operator+(Distance n) const
    {
        self_type retval(input_itr_a + n, input_itr_b + n, conversion_op);
        return retval;
    }

    /// Addition assignment
    template <typename Distance>
    __host__ __device__ __forceinline__ self_type& operator+=(Distance n)
    {
        input_itr_a += n;
        input_itr_b += n;
        return *this;
    }

    /// Subtraction
    template <typename Distance>
    __host__ __device__ __forceinline__ self_type operator-(Distance n) const
    {
        self_type retval(input_itr_a - n, input_itr_b - n, conversion_op);
        return retval;
    }

    /// Subtraction assignment
    template <typename Distance>
    __host__ __device__ __forceinline__ self_type& operator-=(Distance n)
    {
        input_itr_a -= n;
        input_itr_b -= n;
        return *this;
    }

    /// Distance
    __host__ __device__ __forceinline__ difference_type operator-(self_type other) const
    {
        return input_itr_a - other.input_itr_a;
    }

    /// Array subscript
    template <typename Distance>
    __host__ __device__ __forceinline__ reference operator[](Distance n) const
    {
        return conversion_op(input_itr_a[n], input_itr_b[n]);
    }

    /// Structure dereference
    __host__ __device__ __forceinline__ pointer operator->()
    {
        return &conversion_op(*input_itr_a, *input_itr_b);
    }

    /// Equal to
    __host__ __device__ __forceinline__ bool operator==(const self_type& rhs)
    {
        return (input_itr_a == rhs.input_itr_a && input_itr_b == rhs.input_itr_b);
    }

    /// Not equal to
    __host__ __device__ __forceinline__ bool operator!=(const self_type& rhs)
    {
        return (input_itr_a != rhs.input_itr_a && input_itr_b != rhs.input_itr_b);
    }

    /// ostream operator
    friend std::ostream& operator<<(std::ostream& os, const self_type& itr)
    {
        return os;
    }
};


/**
 * Mapping of input based on index array. It will rearange indexes based on mapping table.
 */
template <
    typename ValueType,
    typename InputIteratorT,
	typename MappingIteratorT,
    typename OffsetT = ptrdiff_t>
class InputMappingIterator
{
public:

    // Required iterator traits
    typedef InputMappingIterator              self_type;              ///< My own type
    typedef OffsetT                             difference_type;        ///< Type to express the result of subtracting one iterator from another
    typedef ValueType                           value_type;             ///< The type of the element the iterator can point to
    typedef ValueType*                          pointer;                ///< The type of a pointer to an element the iterator can point to
    typedef ValueType                           reference;              ///< The type of a reference to an element the iterator can point to

#if (THRUST_VERSION >= 100700)
    // Use Thrust's iterator categories so we can use these iterators in Thrust 1.7 (or newer) methods
    typedef typename thrust::detail::iterator_facade_category<
        thrust::any_system_tag,
        thrust::random_access_traversal_tag,
        value_type,
        reference
      >::type iterator_category;                                        ///< The iterator category
#else
    typedef std::random_access_iterator_tag     iterator_category;      ///< The iterator category
#endif  // THRUST_VERSION

private:

    InputIteratorT  input_itr;
    MappingIteratorT  mapping_itr;

public:

    /// Constructor
    __host__ __device__ __forceinline__ InputMappingIterator(
    InputIteratorT      input_itr,          ///< Input iterator to wrap
	MappingIteratorT  	mapping_itr)      ///< Mapping for input interator

    :
        input_itr(input_itr),
		mapping_itr(mapping_itr)
    {}

    /// Postfix increment
    __host__ __device__ __forceinline__ self_type operator++(int)
    {
        self_type retval = *this;
        mapping_itr++;
        return retval;
    }

    /// Prefix increment
    __host__ __device__ __forceinline__ self_type operator++()
    {
    	mapping_itr++;
        return *this;
    }

    /// Indirection
    __host__ __device__ __forceinline__ reference operator*() const
    {
    	return input_itr[*mapping_itr];
    }

    /// Addition
    template <typename Distance>
    __host__ __device__ __forceinline__ self_type operator+(Distance n) const
    {
        self_type retval(input_itr, mapping_itr + n);
        return retval;
    }

    /// Addition assignment
    template <typename Distance>
    __host__ __device__ __forceinline__ self_type& operator+=(Distance n)
    {
    	mapping_itr += n;
        return *this;
    }

    /// Subtraction
    template <typename Distance>
    __host__ __device__ __forceinline__ self_type operator-(Distance n) const
    {
        self_type retval(input_itr, mapping_itr - n);
        return retval;
    }

    /// Subtraction assignment
    template <typename Distance>
    __host__ __device__ __forceinline__ self_type& operator-=(Distance n)
    {
    	mapping_itr -= n;
        return *this;
    }

    /// Distance
    __host__ __device__ __forceinline__ difference_type operator-(self_type other) const
    {
        return mapping_itr - other.mapping_itr;
    }

    /// Array subscript
    template <typename Distance>
    __host__ __device__ __forceinline__ reference operator[](Distance n) const
    {
    	return input_itr[mapping_itr[n]];
    }

    /// Structure dereference
    __host__ __device__ __forceinline__ pointer operator->()
    {
        return &input_itr[*mapping_itr];
    }

    /// Equal to
    __host__ __device__ __forceinline__ bool operator==(const self_type& rhs)
    {
        return (input_itr == rhs.input_itr && mapping_itr == rhs.mapping_itr);
    }

    /// Not equal to
    __host__ __device__ __forceinline__ bool operator!=(const self_type& rhs)
    {
        return (input_itr != rhs.input_itr && mapping_itr != rhs.mapping_itr);
    }

    /// ostream operator
    friend std::ostream& operator<<(std::ostream& os, const self_type& itr)
    {
        return os;
    }

};

/**
 * Segmented reduction that uses d_out values as intialization values (one block per segment)
 */
template <
    typename                ChainedPolicyT,             ///< Chained tuning policy
    typename                InputIteratorT,             ///< Random-access input iterator type for reading input items \iterator
    typename                OutputIteratorT,            ///< Output iterator type for recording the reduced aggregate \iterator
    typename                OffsetT,                    ///< Signed integer type for global offsets
    typename                ReductionOpT,               ///< Binary reduction functor type having member <tt>T operator()(const T &a, const T &b)</tt>
    typename                T>                          ///< Data element type that is convertible to the \p value type of \p InputIteratorT
__launch_bounds__ (int(ChainedPolicyT::ActivePolicy::ReducePolicy::BLOCK_THREADS))
__global__ void CaffeDeviceSegmentedReduceWithInitKernel(
    InputIteratorT          d_in,                       ///< [in] Pointer to the input sequence of data items
    OutputIteratorT         d_out,                      ///< [out] Pointer to the output aggregate
    int                     *d_begin_offsets,           ///< [in] %Devic-accessible pointer to the sequence of beginning offsets of length \p num_segments, such that <tt>d_begin_offsets[i]</tt> is the first element of the <em>i</em><sup>th</sup> data segment in <tt>d_keys_*</tt> and <tt>d_values_*</tt>
    int                     *d_end_offsets,             ///< [in] %Device-accessible pointer to the sequence of ending offsets of length \p num_segments, such that <tt>d_end_offsets[i]-1</tt> is the last element of the <em>i</em><sup>th</sup> data segment in <tt>d_keys_*</tt> and <tt>d_values_*</tt>.  If <tt>d_end_offsets[i]-1</tt> <= <tt>d_begin_offsets[i]</tt>, the <em>i</em><sup>th</sup> is considered empty.
    int                     num_segments,               ///< [in] The number of segments that comprise the sorting data
    ReductionOpT            reduction_op)               ///< [in] Binary reduction functor

{
    // Thread block type for reducing input tiles
    typedef cub::AgentReduce<
            typename ChainedPolicyT::ActivePolicy::ReducePolicy,
            InputIteratorT,
            OffsetT,
            ReductionOpT>
        AgentReduceT;

    // Shared memory storage
    __shared__ typename AgentReduceT::TempStorage temp_storage;

    OffsetT segment_begin   = d_begin_offsets[blockIdx.x];
    OffsetT segment_end     = d_end_offsets[blockIdx.x];

    // Check if empty problem
    if (segment_begin == segment_end)
    {
        return;
    }

    // Consume input tiles
    T block_aggregate = AgentReduceT(temp_storage, d_in, reduction_op).ConsumeRange(
        segment_begin,
        segment_end);

    // Normalize as needed
    cub::NormalizeReductionOutput(block_aggregate, segment_begin, d_in);

    if (threadIdx.x == 0)
        d_out[blockIdx.x] = reduction_op(d_out[blockIdx.x], block_aggregate);
    	//d_out[blockIdx.x] = reduction_op((T)0, block_aggregate);
}

/**
 * Utility class for dispatching the appropriately-tuned kernels for device-wide reduction
 */
template <
    typename InputIteratorT,    ///< Random-access input iterator type for reading input items \iterator
    typename OutputIteratorT,   ///< Output iterator type for recording the reduced aggregate \iterator
    typename OffsetT,           ///< Signed integer type for global offsets
    typename ReductionOpT>      ///< Binary reduction functor type having member <tt>T operator()(const T &a, const T &b)</tt>
struct CaffeDispatchSegmentedReduce :
    cub::DeviceReducePolicy<
        typename std::iterator_traits<InputIteratorT>::value_type,
        OffsetT,
        ReductionOpT>
{
    //------------------------------------------------------------------------------
    // Constants
    //------------------------------------------------------------------------------

    // Data type of input iterator
    typedef typename std::iterator_traits<InputIteratorT>::value_type T;


    //------------------------------------------------------------------------------
    // Problem state
    //------------------------------------------------------------------------------

    void                *d_temp_storage;        ///< [in] %Device-accessible allocation of temporary storage.  When NULL, the required allocation size is written to \p temp_storage_bytes and no work is done.
    size_t              &temp_storage_bytes;    ///< [in,out] Reference to size in bytes of \p d_temp_storage allocation
    InputIteratorT      d_in;                   ///< [in] Pointer to the input sequence of data items
    OutputIteratorT     d_out;                  ///< [out] Pointer to the output aggregate
    OffsetT             num_segments;           ///< [in] The number of segments that comprise the sorting data
    OffsetT             *d_begin_offsets;       ///< [in] %Device-accessible pointer to the sequence of beginning offsets of length \p num_segments, such that <tt>d_begin_offsets[i]</tt> is the first element of the <em>i</em><sup>th</sup> data segment in <tt>d_keys_*</tt> and <tt>d_values_*</tt>
    OffsetT             *d_end_offsets;         ///< [in] %Device-accessible pointer to the sequence of ending offsets of length \p num_segments, such that <tt>d_end_offsets[i]-1</tt> is the last element of the <em>i</em><sup>th</sup> data segment in <tt>d_keys_*</tt> and <tt>d_values_*</tt>.  If <tt>d_end_offsets[i]-1</tt> <= <tt>d_begin_offsets[i]</tt>, the <em>i</em><sup>th</sup> is considered empty.
    ReductionOpT        reduction_op;           ///< [in] Binary reduction functor
    cudaStream_t        stream;                 ///< [in] CUDA stream to launch kernels within.  Default is stream<sub>0</sub>.
    bool                debug_synchronous;      ///< [in] Whether or not to synchronize the stream after every kernel launch to check for errors.  Also causes launch configurations to be printed to the console.  Default is \p false.
    int                 ptx_version;            ///< [in] PTX version

    //------------------------------------------------------------------------------
    // Constructor
    //------------------------------------------------------------------------------

    /// Constructor
    CUB_RUNTIME_FUNCTION __forceinline__
    CaffeDispatchSegmentedReduce(
        void*                   d_temp_storage,
        size_t                  &temp_storage_bytes,
        InputIteratorT          d_in,
        OutputIteratorT         d_out,
        OffsetT                 num_segments,
        OffsetT                 *d_begin_offsets,
        OffsetT                 *d_end_offsets,
        ReductionOpT            reduction_op,
        cudaStream_t            stream,
        bool                    debug_synchronous,
        int                     ptx_version)
    :
        d_temp_storage(d_temp_storage),
        temp_storage_bytes(temp_storage_bytes),
        d_in(d_in),
        d_out(d_out),
        num_segments(num_segments),
        d_begin_offsets(d_begin_offsets),
        d_end_offsets(d_end_offsets),
        reduction_op(reduction_op),
        stream(stream),
        debug_synchronous(debug_synchronous),
        ptx_version(ptx_version)
    {}



    //------------------------------------------------------------------------------
    // Chained policy invocation
    //------------------------------------------------------------------------------

    /// Invocation
    template <
        typename                        ActivePolicyT,                  ///< Umbrella policy active for the target device
        typename                        DeviceSegmentedReduceKernelT>   ///< Function type of cub::DeviceSegmentedReduceKernel
    CUB_RUNTIME_FUNCTION __forceinline__
    cudaError_t InvokePasses(
        DeviceSegmentedReduceKernelT    segmented_reduce_kernel)        ///< [in] Kernel function pointer to parameterization of cub::DeviceSegmentedReduceKernel
    {
#ifndef CUB_RUNTIME_ENABLED

        // Kernel launch not supported from this device
        return CubDebug(cudaErrorNotSupported );
#else
        cudaError error = cudaSuccess;
        do
        {
            // Return if the caller is simply requesting the size of the storage allocation
            if (d_temp_storage == NULL)
            {
                temp_storage_bytes = 1;
                return cudaSuccess;
            }

            // Init kernel configuration
            cub::KernelConfig segmented_reduce_config;
            if (CubDebug(error = segmented_reduce_config.Init<typename ActivePolicyT::SegmentedReducePolicy>(segmented_reduce_kernel))) break;

            // Log device_reduce_sweep_kernel configuration
            if (debug_synchronous) _CubLog("Invoking MySegmentedDeviceReduceKernel<<<%d, %d, 0, %lld>>>(), %d items per thread, %d SM occupancy\n",
                num_segments,
                ActivePolicyT::SegmentedReducePolicy::BLOCK_THREADS,
                (long long) stream,
                ActivePolicyT::SegmentedReducePolicy::ITEMS_PER_THREAD,
                segmented_reduce_config.sm_occupancy);

            // Invoke DeviceReduceKernel
            segmented_reduce_kernel<<<num_segments, ActivePolicyT::SegmentedReducePolicy::BLOCK_THREADS, 0, stream>>>(
                d_in,
                d_out,
                d_begin_offsets,
                d_end_offsets,
                num_segments,
                reduction_op);

            // Check for failure to launch
            if (CubDebug(error = cudaPeekAtLastError())) break;

            // Sync the stream if specified to flush runtime errors
            if (debug_synchronous && (CubDebug(error = cub::SyncStream(stream)))) break;
        }
        while (0);

        return error;

#endif // CUB_RUNTIME_ENABLED

    }


    /// Invocation
    template <typename ActivePolicyT>
    CUB_RUNTIME_FUNCTION __forceinline__
    cudaError_t Invoke()
    {
        typedef typename CaffeDispatchSegmentedReduce::MaxPolicy MaxPolicyT;

        // Force kernel code-generation in all compiler passes
        return InvokePasses<ActivePolicyT>(
        	CaffeDeviceSegmentedReduceWithInitKernel<MaxPolicyT, InputIteratorT, OutputIteratorT, OffsetT, ReductionOpT, T>);
    }


    //------------------------------------------------------------------------------
    // Dispatch entrypoints
    //------------------------------------------------------------------------------

    /**
     * Internal dispatch routine for computing a device-wide reduction
     */
    CUB_RUNTIME_FUNCTION __forceinline__
    static cudaError_t Dispatch(
        void            *d_temp_storage,                    ///< [in] %Device-accessible allocation of temporary storage.  When NULL, the required allocation size is written to \p temp_storage_bytes and no work is done.
        size_t          &temp_storage_bytes,                ///< [in,out] Reference to size in bytes of \p d_temp_storage allocation
        InputIteratorT  d_in,                               ///< [in] Pointer to the input sequence of data items
        OutputIteratorT d_out,                              ///< [out] Pointer to the output aggregate
        int             num_segments,                       ///< [in] The number of segments that comprise the sorting data
        int             *d_begin_offsets,                   ///< [in] %Device-accessible pointer to the sequence of beginning offsets of length \p num_segments, such that <tt>d_begin_offsets[i]</tt> is the first element of the <em>i</em><sup>th</sup> data segment in <tt>d_keys_*</tt> and <tt>d_values_*</tt>
        int             *d_end_offsets,                     ///< [in] %Device-accessible pointer to the sequence of ending offsets of length \p num_segments, such that <tt>d_end_offsets[i]-1</tt> is the last element of the <em>i</em><sup>th</sup> data segment in <tt>d_keys_*</tt> and <tt>d_values_*</tt>.  If <tt>d_end_offsets[i]-1</tt> <= <tt>d_begin_offsets[i]</tt>, the <em>i</em><sup>th</sup> is considered empty.
        ReductionOpT    reduction_op,                       ///< [in] Binary reduction functor
        cudaStream_t    stream,                             ///< [in] <b>[optional]</b> CUDA stream to launch kernels within.  Default is stream<sub>0</sub>.
        bool            debug_synchronous)                  ///< [in] <b>[optional]</b> Whether or not to synchronize the stream after every kernel launch to check for errors.  Also causes launch configurations to be printed to the console.  Default is \p false.
    {
        typedef typename CaffeDispatchSegmentedReduce::MaxPolicy MaxPolicyT;

        if (num_segments <= 0)
            return cudaSuccess;

        cudaError error = cudaSuccess;
        do
        {
            // Get PTX version
            int ptx_version;
            if (CubDebug(error = cub::PtxVersion(ptx_version))) break;

            // Create dispatch functor
            CaffeDispatchSegmentedReduce dispatch(
                d_temp_storage, temp_storage_bytes,
                d_in, d_out,
                num_segments, d_begin_offsets, d_end_offsets,
                reduction_op,
                stream, debug_synchronous, ptx_version);

            // Dispatch to chained policy
            if (CubDebug(error = MaxPolicyT::Invoke(ptx_version, dispatch))) break;
        }
        while (0);

        return error;
    }
};

template <
	typename            InputIteratorT,
	typename            OutputIteratorT>
CUB_RUNTIME_FUNCTION
static cudaError_t segmentedSumWithAdd(
	void                *d_temp_storage,                    ///< [in] %Device-accessible allocation of temporary storage.  When NULL, the required allocation size is written to \p temp_storage_bytes and no work is done.
	size_t              &temp_storage_bytes,                ///< [in,out] Reference to size in bytes of \p d_temp_storage allocation
	InputIteratorT      d_in,                               ///< [in] Pointer to the input sequence of data items
	OutputIteratorT     d_out,                              ///< [out] Pointer to the output aggregate
	int                 num_segments,                       ///< [in] The number of segments that comprise the sorting data
	int                 *d_begin_offsets,                   ///< [in] %Device-accessible pointer to the sequence of beginning offsets of length \p num_segments, such that <tt>d_begin_offsets[i]</tt> is the first element of the <em>i</em><sup>th</sup> data segment in <tt>d_keys_*</tt> and <tt>d_values_*</tt>
	int                 *d_end_offsets,                     ///< [in] %Device-accessible pointer to the sequence of ending offsets of length \p num_segments, such that <tt>d_end_offsets[i]-1</tt> is the last element of the <em>i</em><sup>th</sup> data segment in <tt>d_keys_*</tt> and <tt>d_values_*</tt>.  If <tt>d_end_offsets[i]-1</tt> <= <tt>d_begin_offsets[i]</tt>, the <em>i</em><sup>th</sup> is considered empty.
	cudaStream_t        stream              = 0,            ///< [in] <b>[optional]</b> CUDA stream to launch kernels within.  Default is stream<sub>0</sub>.
	bool                debug_synchronous   = false)        ///< [in] <b>[optional]</b> Whether or not to synchronize the stream after every kernel launch to check for errors.  Also causes launch configurations to be printed to the console.  Default is \p false.
{
	typedef int OffsetT;                                                    // Signed integer type for global offsets
	typedef typename std::iterator_traits<InputIteratorT>::value_type T;    // Data element type

	return CaffeDispatchSegmentedReduce<InputIteratorT, OutputIteratorT, OffsetT, cub::Sum>::Dispatch(
		d_temp_storage,
		temp_storage_bytes,
		d_in,
		d_out,
		num_segments,
		d_begin_offsets,
		d_end_offsets,
		cub::Sum(),
		stream,
		debug_synchronous);
}


}               // CUB namespace

}		// Caffe namespace
