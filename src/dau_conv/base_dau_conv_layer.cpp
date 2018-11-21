#include <algorithm>
#include <vector>
#include <numeric>

#include "dau_conv/util/math_functions.hpp"
#include "dau_conv/util/im2col.hpp"

#include "dau_conv/dau_conv_impl/dau_conv_forward.hpp"
#include "dau_conv/dau_conv_impl/dau_conv_backward.hpp"

//#include <opencv2/opencv.hpp>
//#include <opencv2/gpu/gpu.hpp>
#include <dau_conv/base_dau_conv_layer.hpp>

namespace DAUConvNet {


template <typename Dtype>
void BaseDAUConvLayer<Dtype>::compute_output_shape() {
    this->height_out_ = (this->height_ + 2 * this->pad_h_ - this->kernel_h_)
                        / this->stride_h_ + 1;
    this->width_out_ = (this->width_ + 2 * this->pad_w_ - this->kernel_w_)
                       / this->stride_w_ + 1;
}



template <typename Dtype>
void BaseDAUConvLayer<Dtype>::LayerSetUp(const DAUConvSettings& settings,
                                         const BaseDAUComponentInitializer<Dtype>& param_initializer,
                                         BaseDAUKernelCompute<Dtype>* kernel_compute,
                                         BaseDAUKernelParams<Dtype>* kernel_param,
                                         BaseDAUKernelOutput<Dtype>* kernel_output,
                                         const vector<int>& bottom_shape, bool in_train) {


    DAU_CHECK(settings.number_units.size() > 0, "Missing at least one number_gauss parameter.");

    int NUM_UNITS_PER_AXIS_X = settings.number_units[0];
    int NUM_UNITS_PER_AXIS_Y = settings.number_units.size() > 1? settings.number_units[1] : NUM_UNITS_PER_AXIS_X;
    this->units_per_channel =  NUM_UNITS_PER_AXIS_X * NUM_UNITS_PER_AXIS_Y;

    this->num_units_ignore = 0;

    // make sure we have at least ALLOWED_UNITS_GROUP (this is requested so for fast version that can handle only factor of 2)
    if (this->units_per_channel % ALLOWED_UNITS_GROUP != 0) {
        int new_num_gauss = ceil(this->units_per_channel / (float)ALLOWED_UNITS_GROUP) * ALLOWED_UNITS_GROUP;
        this->num_units_ignore = new_num_gauss - units_per_channel;

        if (NUM_UNITS_PER_AXIS_X < NUM_UNITS_PER_AXIS_Y) {
            NUM_UNITS_PER_AXIS_X += this->num_units_ignore;
        } else {
            NUM_UNITS_PER_AXIS_Y += this->num_units_ignore;
        }

        this->units_per_channel = new_num_gauss;
    }

    DAU_CHECK(bottom_shape.size() == 4, "Input must have 4 axes, corresponding to (num, channels, height, width)");

    this->kernel_h_ = this->kernel_w_ = settings.kernel_size;
    this->pad_h_ = this->pad_w_ = settings.pad;
    this->stride_h_ = this->stride_w_ = settings.stride;

    DAU_CHECK(this->kernel_h_ > 0, "Filter dimensions cannot be zero.");
    DAU_CHECK(this->kernel_w_ > 0, "Filter dimensions cannot be zero.");

    DAU_CHECK(this->stride_h_ == 1, "BaseDAUConvLayer does not support stride>1 parameter at the moment");
    DAU_CHECK(this->stride_w_ == 1, "BaseDAUConvLayer does not support stride>1 parameter at the moment");

    this->max_kernel_w_ = this->kernel_w_;
    this->max_kernel_h_ = this->kernel_h_;

    // curently we support only NCHW format !! so channel axis is indexed as 1
    this->channel_axis_ = 1;
    const int first_spatial_axis = this->channel_axis_ + 1;
    const int num_axes = bottom_shape.size();
    this->num_spatial_axes_ = num_axes - first_spatial_axis;
    DAU_CHECK(this->num_spatial_axes_ >= 0, "Only positive num_spatial_axes allowed");

    // Configure output channels and groups.
    this->conv_in_channels_ = bottom_shape[this->channel_axis_];
    this->conv_out_channels_ = settings.num_output;
    DAU_CHECK(this->conv_out_channels_ > 0, "Only positive number of output channels allowed");

    this->bias_term_ = settings.bias_term;

    // initialize parameter shapes as soon as we know all the sizes
    this->reshape_params((vector<int>){1, this->conv_in_channels_, this->units_per_channel, this->conv_out_channels_});

    // initialize [w,mu1,mu2,sigma] params
    param_initializer.InitializeParameters(settings,
                                           this->param_w(), this->param_mu1(), this->param_mu2(), this->param_sigma(), this->is_data_on_gpu(),
                                           NUM_UNITS_PER_AXIS_X, NUM_UNITS_PER_AXIS_Y, this->num_units_ignore,
                                           this->conv_in_channels_, this->conv_out_channels_, this->kernel_h_, this->kernel_w_);

    this->offsets_already_centered_ = settings.offsets_already_centered;

    this->unit_border_bound = settings.component_border_bound;
    this->unit_sigma_lower_bound = settings.sigma_lower_bound;

    // decide if needed to perform gmm gauss normalization
    this->use_unit_normalization = settings.unit_normalization;
    this->use_square_unit_normalization = settings.square_unit_normalization;
    this->mean_iteration_step = settings.mean_iteration_step;
    this->sigma_iteration_step = settings.sigma_iteration_step;

    // make sure component merging is done only at training step
    if (in_train)
        this->unit_merge_iteration_step = settings.merge_iteration_step;
    else
        this->unit_merge_iteration_step = 0;

    this->unit_merge_threshold = settings.merge_threshold;

    this->current_iteration_index = 0;

    this->kernel_compute = kernel_compute;

    // setup for precomputed kernels
    this->kernel_compute->setup(this->use_unit_normalization,
                                this->use_square_unit_normalization,
                                this->unit_sigma_lower_bound,0, // unit_border_bound = 0 (do not need border bound for this implementation)
                                this->offsets_already_centered_,
                                this->single_dimension_kernel,
                                this->forbid_positive_dim1);

    // setup default bottom/top dimensions to zero
    this->bottom_dim_ = 0;
    this->top_dim_ = 0;
    this->batch_num_ = 0;

    // workspace data
    workspaceSizeInBytes = 0;

    // zero-out all values in buffer_fwd_ and in buffer_bwd_ (is safe as long as all values are int or ptr)
    memset(&buffer_fwd_, 0, sizeof(buffer_fwd_));
    memset(&buffer_bwd_, 0, sizeof(buffer_fwd_));

    // Create filter descriptor.
    Dtype sigma = this->get_sigma_val();

    DAU_CHECK(sigma > 0, "Must use sigma > 0 - initialize it with appropriate value");

    // define pre-filtering kernel size based on 5*sigma - NOTE: currently this is fixed and cannot be changed if sigma increases !!
    aggregation.kernel_h_ = 2 * (int)ceil(5 * sigma) + 1;
    aggregation.kernel_w_ = 2 * (int)ceil(5 * sigma) + 1;

    DAU_CHECK(aggregation.kernel_h_ > 1, "Sigma too small; must have gaussian kernel size > 1 - increase sigma value");
    DAU_CHECK(aggregation.kernel_w_ > 1, "Sigma too small; must have gaussian kernel size > 1 - increase sigma value");

    // we need to ensure to get the same intermediate size as input size so pad accordingly
    aggregation.pad_h_ = (int)floor(aggregation.kernel_h_ / 2);
    aggregation.pad_w_ = (int)floor(aggregation.kernel_w_ / 2);

    // we allow only stride of 1 for pre-filtering
    aggregation.stride_h_ = 1;
    aggregation.stride_w_ = 1;

    aggregation.current_sigma = 0;

    aggregation.kernels = kernel_output;

    // create buffers used to generate kernels (i.e. we need only one kernel
    aggregation.param = kernel_param;
    aggregation.param->reshape(1, 1, 1);

    paralel_streams = new cudaStream_t[4];
    for (int g = 0; g < 4; ++g) {
        CUDA_CHECK(cudaStreamCreate(&paralel_streams[g]));
    }

    // create default cuda stream if not using extrenal ones
    if (this->own_cuda_stream == true)
        CUDA_CHECK(cudaStreamCreate(&stream_));

    handles_setup_ = true;

    // by default we generate kernels with w=1, mu=(0,0) so fill buffers with them
    // NOTE: mu=(0,0) is center of kernel so use that value
    caffe_gpu_set_async(1, (Dtype)1.0f, aggregation.param->weight(), stream_);
    caffe_gpu_set_async(1, (Dtype)(this->offsets_already_centered_ == false ? (int)(aggregation.kernel_w_/2) : 0), aggregation.param->mu1(), stream_);
    caffe_gpu_set_async(1, (Dtype)(this->offsets_already_centered_ == false ? (int)(aggregation.kernel_h_/2) : 0), aggregation.param->mu2(), stream_);

    this->use_interpolation_ = true;

}

template <typename Dtype>
vector<int> BaseDAUConvLayer<Dtype>::Reshape(const vector<int>& bottom_shape, const vector<int>& top_shape) {

    const int batch_axis = this->channel_axis_ - 1;
    const int height_axis = this->channel_axis_ + 1;
    const int width_axis = this->channel_axis_ + 2;

    int top_count = top_shape.empty() == false ? std::accumulate(top_shape.begin(), top_shape.end(), 1, std::multiplies<int>()) : 0;
    int top_dim = top_shape.empty() == false ? std::accumulate(top_shape.begin() + this->channel_axis_, top_shape.end(), 1, std::multiplies<int>()) : 0;
    int bottom_dim = bottom_shape.empty() == false ? std::accumulate(bottom_shape.begin() + this->channel_axis_, bottom_shape.end(), 1, std::multiplies<int>()) : 0;

    if (this->bottom_dim_ == bottom_dim && top_count > 0 && this->top_dim_ == top_dim &&
        this->batch_num_ == bottom_shape[batch_axis] &&
        this->height_ == bottom_shape[height_axis] &&
        this->width_ == bottom_shape[width_axis] ) {
        return top_shape;
    }
    const int first_spatial_axis = this->channel_axis_ + 1;

    DAU_CHECK(bottom_shape.size() == 4, "Input must have 4 axes, corresponding to (num, channels, height, width)");
    this->batch_num_ = bottom_shape[batch_axis];
    this->height_ = bottom_shape[height_axis];
    this->width_ = bottom_shape[width_axis];
    DAU_CHECK(bottom_shape[this->channel_axis_] == this->conv_in_channels_, "Input size incompatible with convolution kernel.");

    // Shape the tops.
    this->compute_output_shape();

    vector<int> new_top_shape = {this->batch_num_, this->conv_out_channels_, this->height_out_, this->width_out_ };


    this->out_spatial_dim_ = std::accumulate(new_top_shape.begin() + first_spatial_axis, new_top_shape.end(), 1, std::multiplies<int>());
    this->bottom_dim_ = std::accumulate(bottom_shape.begin() + this->channel_axis_, bottom_shape.end(), 1, std::multiplies<int>());
    this->top_dim_ = std::accumulate(new_top_shape.begin() + this->channel_axis_, new_top_shape.end(), 1, std::multiplies<int>());


    DAU_CHECK(this->num_spatial_axes_ == 2, "BaseDAUConvLayer input must have 2 spatial axes (e.g., height and width). ");

    const int max_width = std::max(this->width_out_,this->width_);
    const int max_height = std::max(this->height_out_,this->height_);


    // prepare intermedite buffers for pre-computing kernels
    this->kernel_compute->reshape(1, 1, 1, this->aggregation.kernel_h_, this->aggregation.kernel_w_);

    // prepare output buffer used in kernel pre-computing
    this->aggregation.kernels->reshape(1, 1, 1, this->aggregation.kernel_h_, this->aggregation.kernel_w_);

    if (enabled_fwd_op) {
        forward_obj.reset(
                new DAUConvNet::DAUConvForward<Dtype>(this->width_, this->height_, this->width_out_, this->height_out_,
                                                      this->batch_num_, this->conv_in_channels_,
                                                      this->conv_out_channels_, this->units_per_channel,
                                                      this->use_interpolation_));
        forward_obj->get_allocation_sizes(this->kernel_w_, this->kernel_h_, this->offsets_already_centered_,
                                          &buffer_fwd_.filtered_images_sizes_,
                                          &buffer_fwd_.filter_weights_sizes_,
                                          &buffer_fwd_.filter_offsets_sizes_);
    } else {
        buffer_fwd_.filtered_images_sizes_ = 0;
        buffer_fwd_.filter_weights_sizes_ = 0;
        buffer_fwd_.filter_offsets_sizes_ = 0;
    }

    if (enabled_bwd_op) {
        // DAU_CHECK how much memory do we need for our custom kernels
        backward_grad_obj.reset(
                new DAUConvNet::DAUConvBackward<Dtype>(this->width_, this->height_, this->width_out_, this->height_out_,
                                                       this->batch_num_, this->conv_in_channels_,
                                                       this->conv_out_channels_, this->units_per_channel, this->NUM_K,
                                                       this->last_k_optional, this->use_interpolation_));

        // WARNING: if this->kernel_w_ or this->kernel_h_ changes then memory will not be allocated properly so we should use here
        //          maximum kernel_w_ and kernel_h_ allowed
        backward_grad_obj->get_allocation_sizes(this->kernel_w_, this->kernel_h_, this->offsets_already_centered_,
                                                &buffer_bwd_.filtered_images_sizes_,
                                                &buffer_bwd_.error_image_sizes_,
                                                &buffer_bwd_.filter_weights_sizes_,
                                                &buffer_bwd_.filter_offsets_sizes_);
        // for gradient aggregation

        // for error back-propagation
        // we use the same buffer as for forward pass but can be shared, just ensure buffer can accomodate both sizes
        size_t filtered_error_sizes_, filter_error_weights_sizes_, filter_error_offsets_sizes_;

        backward_backporp_obj.reset(
                new DAUConvNet::DAUConvForward<Dtype>(max_width, max_height, max_width, max_height, this->batch_num_,
                                                      this->conv_out_channels_, this->conv_in_channels_,
                                                      this->units_per_channel, this->use_interpolation_));
        backward_backporp_obj->get_allocation_sizes(this->kernel_w_, this->kernel_h_, this->offsets_already_centered_,
                                                    &filtered_error_sizes_,
                                                    &filter_error_weights_sizes_,
                                                    &filter_error_offsets_sizes_);

        buffer_bwd_.resized_top_for_bwd_sizes_ = 0;

        if (this->width_out_ != this->width_ || this->height_out_ != this->height_) {
            buffer_bwd_.resized_top_for_bwd_sizes_ =
                    this->batch_num_ * this->conv_out_channels_ * max_height * max_width * sizeof(Dtype);
        }

        // this ensures that buffers will accomodate both dau_conv_forward functions one used for forward pass and the second one used of error back-propagation
        buffer_fwd_.filtered_images_sizes_ = std::max(buffer_fwd_.filtered_images_sizes_, filtered_error_sizes_);
        buffer_fwd_.filter_weights_sizes_ = std::max(buffer_fwd_.filter_weights_sizes_, filter_error_weights_sizes_);
        buffer_fwd_.filter_offsets_sizes_ = std::max(buffer_fwd_.filter_offsets_sizes_, filter_error_offsets_sizes_);
    } else {
        buffer_bwd_.filtered_images_sizes_ = 0;
        buffer_bwd_.error_image_sizes_ = 0;
        buffer_bwd_.filter_weights_sizes_ = 0;
        buffer_bwd_.filter_offsets_sizes_ = 0;
        buffer_bwd_.resized_top_for_bwd_sizes_ = 0;
    }

    // reduce over all workspace sizes to get a maximum to allocate / reallocate
    size_t total_workspace_fwd = 0;
    size_t total_workspace_bwd_data = 0;

    total_workspace_fwd         = std::max(total_workspace_fwd,
                                           buffer_fwd_.filtered_images_sizes_ +
                                           buffer_fwd_.filter_weights_sizes_ +
                                           buffer_fwd_.filter_offsets_sizes_);
    total_workspace_bwd_data    = std::max(total_workspace_bwd_data,
                                           buffer_bwd_.filtered_images_sizes_ +
                                           buffer_bwd_.error_image_sizes_ +
                                           buffer_bwd_.filter_weights_sizes_ +
                                           buffer_bwd_.filter_offsets_sizes_ );

    total_workspace_bwd_data    = std::max(total_workspace_bwd_data,
                                           buffer_bwd_.resized_top_for_bwd_sizes_);

    // get max over all operations
    size_t total_max_workspace = std::max(total_workspace_fwd,
                                    total_workspace_bwd_data);

    // this is the total amount of storage needed over all groups + streams
    if (total_max_workspace > workspaceSizeInBytes) {
        if (this->enabled_memalloc_info) {
            std::string units;
            float newly_allocated = total_max_workspace - workspaceSizeInBytes;
            if (newly_allocated > 1024*1024*1024) {
                units = "GB";
                newly_allocated = newly_allocated / (1024*1024*1024);
            } else if (newly_allocated > 1024*1024) {
                units = "MB";
                newly_allocated = newly_allocated / (1024*1024);
            } else if (newly_allocated > 1024) {
                units = "kB";
                newly_allocated = newly_allocated / (1024);
            } else {
                units = "B";
            }
            std::cout << "Reallocating workspace storage with extra mem allocation: " << newly_allocated << " " << units << std::endl;
        }
        workspaceSizeInBytes = total_max_workspace;

        // allocate workspace data but we do not own it so do not store original pointer and do not do any cleanups
        void* workspaceData = this->allocate_workspace_mem(workspaceSizeInBytes);

        if (workspaceData == NULL) {
            workspaceSizeInBytes = 0;
        }

        // NOTE: buffer_ is not ready for multiple groups so modify this if you want to have multiple groups
        // we reuse the buffer allocated for convolution, however that buffer will bi small
        // TODO:
        //    - we need to implement a mechanizem to share workspace buffer between layers
        //    - this would work for layer that are executed in sequence but will fail for parallel layers
        //    - also make sure to count on multi-GPU implementations so do explicitly use only static buffers !!!

        // We should implement memory as:
        // - for each GPU-device we use seperate class for handling memory
        // - we use static array within this class to store memory storage classes and save them with GPU-id
        // - we can get GPU-id of current active device since that is setup for us by dau_conv_impl
        // - memory storage class needs to take care of:
        //   - allocating memory when requested
        //   - ensuring the same memory is used across all alocations
        //   - if more memory is requested then currently allocated then we need to allocat new memory
        //   - in case of allocating additional memory we NEED to ensure that existing instances with pointers to that memory are updated
        //     (this should be done by wraping allocated pointer around new class, possibly std:shared_ptr and using that one instead of raw pointers)
        // - we need to ensure to deallocate memory when all not used any more
        //   - when each BaseDAUConvLayer is destroyed it should indicate to memory manegement class to destroy its memory
        //   - memory manegment should keep an index if valid pointer that are still using it and it should deallocate CUDA memory when there are no more valid pointers!!



        // TODO: make all memory align to 4x 32bit values
        if (enabled_fwd_op || enabled_bwd_op) {
            // buffer_fwd_ is used for backward pass (for backproped error) so allocate for enabled_bwd_op==True
            buffer_fwd_.filtered_images = reinterpret_cast<Dtype*>(reinterpret_cast<char *>(workspaceData));
            buffer_fwd_.filter_offsets_and_weights = reinterpret_cast<Dtype*>(reinterpret_cast<char *>(workspaceData) + buffer_fwd_.filtered_images_sizes_);
            buffer_fwd_.filter_weights = reinterpret_cast<Dtype*>(reinterpret_cast<char *>(workspaceData) + buffer_fwd_.filtered_images_sizes_);
            buffer_fwd_.filter_offsets = reinterpret_cast<int*>(reinterpret_cast<char *>(workspaceData) + buffer_fwd_.filtered_images_sizes_ + buffer_fwd_.filter_weights_sizes_);
        }
        if (enabled_bwd_op) {
            buffer_bwd_.filtered_images = reinterpret_cast<Dtype*>(reinterpret_cast<char *>(workspaceData));
            buffer_bwd_.error_images = reinterpret_cast<Dtype*>(reinterpret_cast<char *>(workspaceData) + buffer_bwd_.filtered_images_sizes_);
            buffer_bwd_.filter_weights = reinterpret_cast<Dtype*>(reinterpret_cast<char *>(workspaceData) + buffer_bwd_.filtered_images_sizes_ + buffer_bwd_.error_image_sizes_);
            buffer_bwd_.filter_offsets = reinterpret_cast<int*>(reinterpret_cast<char *>(workspaceData) + buffer_bwd_.filtered_images_sizes_ + buffer_bwd_.error_image_sizes_ + buffer_bwd_.filter_weights_sizes_);

            // we can reuse workspace data since it will not be used at the same time
            buffer_bwd_.resized_top_for_bwd = reinterpret_cast<Dtype*>(reinterpret_cast<char *>(workspaceData));
        }
    }

    return new_top_shape;
}

template <typename Dtype>
BaseDAUConvLayer<Dtype>::~BaseDAUConvLayer() {
    // Check that handles have been setup before destroying.
    if (!handles_setup_) { return; }

    if (this->own_cuda_stream == true && stream_ != NULL)
        CUDA_CHECK(cudaStreamDestroy(stream_));

    for (int g = 0; g < 4; ++g) {
        CUDA_CHECK(cudaStreamDestroy(paralel_streams[g]));
    }

    delete [] paralel_streams;
}

__global__ void sync_fast_gauss_conv_groups() { }


template <typename Dtype>
bool BaseDAUConvLayer<Dtype>::update_prefiltering_kernels(cudaStream_t stream) {

    // if we changed the variance then re-compute the gaussian kernel
    // we assume there is only one sigma for the whole layer !!!
    Dtype sigma = this->get_sigma_val();

    if (std::fabs(aggregation.current_sigma - sigma) > 1e-5) {

        // we compute kernels for blur using the same code as in std-implementation but we compute only for a single
        // component i.e., num_in_channels = 1, num_out_channels = 1, num_gauss = 1, and we use weight=1, mu = [0,0]

        this->kernel_compute->get_kernels(*this->aggregation.param, *this->aggregation.kernels, this->enable_unit_bounds_guard_,
                                          cublas_handle, stream);

        this->aggregation.current_sigma = sigma;

        return true;
    }
    return false;
}

template <typename Dtype>
Dtype* BaseDAUConvLayer<Dtype>::get_gaussian_kernel(cudaStream_t stream) {

    update_prefiltering_kernels(stream);

    return this->aggregation.kernels->weight();
}

template <typename Dtype>
Dtype* BaseDAUConvLayer<Dtype>::get_deriv_kernel_params(cudaStream_t stream) {

    update_prefiltering_kernels(stream);

    return this->aggregation.kernels->d_params();
}

template <typename Dtype>
Dtype* BaseDAUConvLayer<Dtype>::get_deriv_kernel_error(cudaStream_t stream) {

    update_prefiltering_kernels(stream);

    return this->aggregation.kernels->d_error();
}

#define OFFSET(l,k,j,i, num_l, num_k, num_j, num_i) ((( (l)*(num_k) + (k)) * (num_j) + (j))*(num_i) + (i) )

template <typename Dtype>
Dtype cpu_dot_elementwise_skip(const Dtype* X, const int X_width, const int X_height, const int src_offset_x, const int src_offset_y,
                              const Dtype* Y, const int Y_width, const int Y_height, const int dst_offset_x, const int dst_offset_y,
                              const int copy_width, const int copy_height) {

    Dtype const* src_ptr = X + OFFSET(0, 0,src_offset_y,src_offset_x, 1, 1, X_height, X_width);
    Dtype const* dst_ptr = Y + OFFSET(0, 0,dst_offset_y,dst_offset_x, 1, 1, Y_height, Y_width);

    Dtype result = 0;

    for (int j = 0; j < copy_height; ++j) {
        for (int i = 0; i < copy_width; ++i) {
            result += dst_ptr[0] * src_ptr[0];

            // move to next element
            src_ptr++;
            dst_ptr++;
        }
        // if copy_width does not equalt to size of arrays then we need to advance for missing elements
        src_ptr += Y_width - copy_width;
        dst_ptr += X_width - copy_width;
    }

    return result;
}



template <typename Dtype>
void cpu_sum_elementwise_skip(const float alpha, const Dtype* X, const int X_width, const int X_height, const int src_offset_x, const int src_offset_y,
                              Dtype* Y, const int Y_width, const int Y_height, const int dst_offset_x, const int dst_offset_y,
                              const int copy_width, const int copy_height) {

    Dtype const* src_ptr = X + OFFSET(0, 0,src_offset_y,src_offset_x, 1, 1, X_height, X_width);
    Dtype* dst_ptr = Y + OFFSET(0, 0,dst_offset_y,dst_offset_x, 1, 1, Y_height, Y_width);

    for (int j = 0; j < copy_height; ++j) {
        for (int i = 0; i < copy_width; ++i) {
            dst_ptr[0] = dst_ptr[0] * src_ptr[0] * alpha;

            // move to next element
            src_ptr++;
            dst_ptr++;
        }
        // if copy_width does not equalt to size of arrays then we need to advance for missing elements
        src_ptr += Y_width - copy_width;
        dst_ptr += X_width - copy_width;
    }
}

template <typename Dtype>
void offset_and_sum_opencv(const Dtype* input_data,
                    const Dtype* filter_weights, const Dtype* filter_offsets_float_mu1, const Dtype* filter_offsets_float_mu2,
                    Dtype* output_data,
                    const int num_, const int conv_in_channels_, const int NUM_GAUSS, const int conv_out_channels_,
                    const int width_, const int height_,
                    const int width_out_, const int height_out_, const int kernel_width, const int kernel_height,
                    const bool offsets_already_centered, const DAUConvForward<float>::PARAM_FORMAT INPUT_FORMAT = DAUConvForward<float>::SGF) {

    // perform offset and sum over individual outputs
#define OFFSET(l,k,j,i, num_l, num_k, num_j, num_i) ((( (l)*(num_k) + (k)) * (num_j) + (j))*(num_i) + (i) )

    const int INTERPOlATION_Dx = 2;
    const int INTERPOlATION_Dy = 2;

    const int F_BATCH = 8;
    const int S_BATCH = 1;

    for (int n = 0; n < num_; ++n) {
        //printf("n=%d\n",n);

        //cv::Mat interm_mat(conv_in_channels_ * height_,width_, CV_32F, (Dtype*)input_data + n * conv_in_channels_ * width_ * height_);
        //cv::Mat top_mat(conv_out_channels_ * height_out_, width_out_, CV_32F, output_data + n * conv_out_channels_ * width_out_  * height_out_);

        int src_width = width_;
        int src_height = conv_in_channels_ * height_;
        Dtype const* src =  input_data + n * conv_in_channels_ * width_ * height_;

        int dst_width = width_out_;
        int dst_height = conv_out_channels_ * height_out_;
        Dtype* dst = output_data + n * conv_out_channels_ * width_out_  * height_out_;

        int border_x = width_/2 - width_out_/2;
        int border_y = height_/2 - height_out_/2;

        border_x = border_x > 0 ? border_x : 0;
        border_y = border_y > 0 ? border_y : 0;

        for (int i = 0; i <  dst_width*dst_height; ++i)
            dst[i] = 0;

        //top_mat.setTo(0);

        for (int f_offset = 0; f_offset < conv_out_channels_; f_offset+=F_BATCH) {
            for (int s_offset = 0; s_offset < conv_in_channels_; s_offset+=S_BATCH) {
                for (int ff = 0; ff < F_BATCH; ff++) {
                    for (int ss = 0; ss < S_BATCH; ss++) {
                        int f = f_offset + ff;
                        int s = s_offset + ss;

                        int access_f_offset = f * height_out_;
                        int access_s_offset = s * height_;

                        for (int g = 0; g < NUM_GAUSS; ++g) {
                            int param_offset = -1;
                            if (INPUT_FORMAT == DAUConvForward<float>::SGF)
                                param_offset = OFFSET(0, s,g,f, 1, conv_in_channels_, NUM_GAUSS, conv_out_channels_);
                            else if (INPUT_FORMAT == DAUConvForward<float>::FGS)
                                param_offset = OFFSET(0, f,g,s, 1, conv_out_channels_, NUM_GAUSS, conv_in_channels_);

                            float w = filter_weights[param_offset];

                            float offset_x = filter_offsets_float_mu1[param_offset] - (offsets_already_centered == false ? kernel_width/2 : 0);
                            float offset_y = filter_offsets_float_mu2[param_offset] - (offsets_already_centered == false ? kernel_height/2 : 0);

                            int offset_x_int = floor(offset_x);
                            int offset_y_int = floor(offset_y);

                            float interpol_off_x = offset_x - offset_x_int;
                            float interpol_off_y = offset_y - offset_y_int;


                            for (int dy = 0; dy < INTERPOlATION_Dy; ++dy) {
                                for (int dx = 0; dx < INTERPOlATION_Dx; ++dx) {

                                    int access_x_off = offset_x_int + dx;
                                    int access_y_off = offset_y_int + dy;

                                    float interpol_w = w;

                                    interpol_w *= (dx == 0 ? (1-interpol_off_x) : interpol_off_x);
                                    interpol_w *= (dy == 0 ? (1-interpol_off_y) : interpol_off_y);

                                    int copy_width = std::min(width_out_ + access_x_off, width_out_ - access_x_off);
                                    int copy_height = std::min(height_out_ + access_y_off, height_out_ - access_y_off);

                                    int src_offset_x = border_x+std::max(0, access_x_off);
                                    int src_offset_y =  border_y+std::max(0, access_y_off) + access_s_offset;

                                    int dst_offset_x = std::max(0, -access_x_off);
                                    int dst_offset_y = std::max(0, -access_y_off) + access_f_offset;
                                    /*cv::Rect interm_roi(border_x+std::max(0, access_x_off),
                                                        border_y+std::max(0, access_y_off) + access_s_offset,
                                                        std::min(width_out_ + access_x_off, width_out_ - access_x_off),
                                                        std::min(height_out_ + access_y_off, height_out_ - access_y_off));

                                    cv::Rect top_roi(std::max(0, -access_x_off),
                                                     std::max(0, -access_y_off) + access_f_offset,
                                                     std::min(width_out_ + access_x_off, width_out_ - access_x_off),
                                                     std::min(height_out_ + access_y_off, height_out_ - access_y_off));*/

                                    //std::cout << "top_roi: " << top_roi << " interm_roi: " << interm_roi  << std::endl;
                                    //if (top_roi.width > 0 && top_roi.height > 0 && interm_roi.width > 0 && interm_roi.height > 0) {
                                    if (copy_width > 0 && copy_height > 0) {
                                        //top_mat(top_roi) += interpol_w * interm_mat(interm_roi);

                                        cpu_sum_elementwise_skip(interpol_w, src, src_width, src_height, src_offset_x, src_offset_y,
                                                                 dst, dst_width, dst_height, dst_offset_x, dst_offset_y,
                                                                 copy_width, copy_height);


                                        //if (f == 0) {
                                        //    printf("sum of f,s,g=%d,%d,%d is val: ", f,s,g);
                                        //    std::cout << top_mat(top_roi) << " with top roi: " << top_roi  << " and inter roi: " << interm_roi << " and iter val " << interm_mat(interm_roi) << std::endl;
                                        //}
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}
    template <typename Dtype>
void BaseDAUConvLayer<Dtype>::Forward_cpu(const Dtype* bottom_data, const vector<int>& bottom_shape,
                                          Dtype* top_data, const vector<int>& top_shape) {

        // - first perform gaussian bluring based on variance that is fixed over the whole layer (use CuDNN for that)
    // - then perform forward pass with our custom kernel
    // - optionally add bias
    DAU_CHECK(this->is_data_on_gpu() == false, "Forward_cpu requires data on CPU, but is_data_on_gpu() returned true !");

    clock_t start_t = clock();

    // check if we need to do merging of components;
    // make sure we check based on steps done in backpropagation and we should avoid merging if only forward is called (by default current_iteration_index=0 so start at second iter
    bool do_merginig_optmization = this->unit_merge_iteration_step > 0 && (this->current_iteration_index + 1) % this->unit_merge_iteration_step == 0 ? true : false;

    // if during training then merge components if needed
    if (do_merginig_optmization) {
        //merge_components();
    }

    const int height_out = top_shape[this->channel_axis_ + 1];
    const int width_out = top_shape[this->channel_axis_ + 2];

    // get filter for gaussian blur step
    const Dtype* gauss_kernel = this->get_gaussian_kernel(stream_);

    // get buffers for all parameters that we learn
    const Dtype* filter_weights = this->param_w();
    const Dtype* filter_offsets_float_mu1 = this->param_mu1();
    const Dtype* filter_offsets_float_mu2 = this->param_mu2();

    {

        //const Dtype* bottom_data = bottom[i]->mutable_cpu_data();
        //Dtype* top_data = top[i]->mutable_cpu_data();

        Dtype* interm_data = this->temp_interm_buffer();


        // first perform convolutions with gaussian filter (i.e. gaussian blur)

        Dtype* col_buff = this->temp_col_buffer();

        for (int n = 0; n < this->batch_num_ * this->conv_in_channels_; ++n) {

            im2col_cpu(bottom_data + n * (this->height_* this->width_), 1, this->height_, this->width_,
                       this->aggregation.kernel_h_, this->aggregation.kernel_w_,
                       this->aggregation.pad_h_, this->aggregation.pad_w_,
                       this->aggregation.stride_h_, this->aggregation.stride_w_,
                       1,1, col_buff);

            caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, 1 , this->height_ * this->width_, this->aggregation.kernel_h_ * this->aggregation.kernel_w_,
                                  (Dtype)1., gauss_kernel , col_buff,
                                  (Dtype)0., interm_data + n * this->width_ * this->height_);
        }

        //Dtype* interm_data = bottom[i]->mutable_cpu_data();

        // now we take the blured input data and perform sum over shifted input data with our custom kernel
        offset_and_sum_opencv(interm_data,filter_weights, filter_offsets_float_mu1, filter_offsets_float_mu2,
                            top_data,
                            this->batch_num_, this->conv_in_channels_, this->units_per_channel, this->conv_out_channels_,
                            this->width_, this->height_,
                            this->width_out_, this->height_out_,
                            this->kernel_w_, this->kernel_h_, this->offsets_already_centered_);

        // add bias if needed
        if (this->bias_term_) {
            const Dtype *bias = this->param_bias();
            for (int n = 0; n < this->batch_num_ ; ++n) {
                this->forward_cpu_bias(top_data + n * this->top_dim_, bias);
            }
        }
    }
}


template <typename Dtype>
void offset_and_dot_opencv(const Dtype* input_data, const Dtype* error_data,
                           const Dtype* filter_weights, const Dtype* filter_offsets_float_mu1, const Dtype* filter_offsets_float_mu2,
                           Dtype* output_data,
                           const int num_, const int conv_in_channels_, const int NUM_GAUSS, const int conv_out_channels_,
                           const int width_, const int height_,
                           const int width_out_, const int height_out_, const int kernel_width, const int kernel_height,
                           const bool ignore_edge_gradients, const bool offsets_already_centered,
                           const DAUConvForward<float>::PARAM_FORMAT INPUT_FORMAT = DAUConvForward<float>::SGF) {

    // perform offset and sum over individual outputs
#define OFFSET(l,k,j,i, num_l, num_k, num_j, num_i) ((( (l)*(num_k) + (k)) * (num_j) + (j))*(num_i) + (i) )

    const int INTERPOlATION_Dx = 2;
    const int INTERPOlATION_Dy = 2;

    const int F_BATCH = 8;
    const int S_BATCH = 1;

    for (int n = 0; n < num_; ++n) {
        //printf("n=%d\n",n);

        // X == interm_data
        int X_width = width_;
        int X_height = conv_in_channels_ * height_;
        const Dtype* X_ptr = input_data + n * conv_in_channels_ * width_ * height_;

        // Y == top_data
        int Y_width = width_out_;
        int Y_height = conv_out_channels_ * height_out_;
        Dtype* Y_ptr = (Dtype*)error_data + n * conv_out_channels_ * width_out_  * height_out_;

        //cv::Mat interm_mat(conv_in_channels_ * height_,width_, CV_32F, (Dtype*)input_data + n * conv_in_channels_ * width_ * height_);
        //cv::Mat top_mat_org(conv_out_channels_ * height_out_, width_out_, CV_32F, (Dtype*)error_data + n * conv_out_channels_ * width_out_  * height_out_);

        // copy top matrix to another buffer so that we do not modify original data
        //cv::Mat top_mat;
        //top_mat_org.copyTo(top_mat);

        // set right/bottom edges to zero if we should ignore them (for GPU compatability)
        if (ignore_edge_gradients) {

            Dtype* Y_ptr_new = (Dtype*)malloc(Y_width*Y_height*sizeof(Dtype));
            memcpy(Y_ptr_new, Y_ptr, Y_width*Y_height*sizeof(Dtype));

            for (int f = 0; f< conv_out_channels_; ++f) {

                int access_f_offset = f * height_out_;

                bool disable_last_column = false;
                bool disable_last_row = false;

                if (width_out_ >= 64) disable_last_column = width_out_ % 64 == 0 ? true : false;
                else if (width_out_ >= 32) disable_last_column = width_out_ % 32 == 0 ? true : false;
                else if (width_out_ >= 16) disable_last_column = width_out_ % 16 == 0 ? true : false;
                else if (width_out_ >= 8) disable_last_column = width_out_ % 8 == 0 ? true : false;

                if (height_out_ >= 64) disable_last_row = height_out_ % 64 == 0 ? true : false;
                else if (height_out_ >= 32) disable_last_row = height_out_ % 32 == 0 ? true : false;
                else if (height_out_ >= 16) disable_last_row = height_out_ % 16 == 0 ? true : false;
                else if (height_out_ >= 8) disable_last_row = height_out_ % 8 == 0 ? true : false;

                //if (disable_last_column) top_mat(cv::Rect(width_out_-1, access_f_offset, 1, height_out_ )) = 0.0f;
                //if (disable_last_row) top_mat(cv::Rect(0, height_out_-1 + access_f_offset , width_out_, 1)) = 0.0f;

                if (disable_last_column) {
                    for (int i = 0; i < Y_height; ++i) {
                        Y_ptr_new[OFFSET(0,0,access_f_offset + i,  width_out_-1,
                                         1,1, Y_height,Y_width)] = 0;
                    }
                }
                if (disable_last_row) {
                    for (int i = 0; i < Y_width; ++i) {
                        Y_ptr_new[OFFSET(0,0,height_out_-1 + access_f_offset, i,
                                         1,1, Y_height,Y_width)] = 0;
                    }
                }
            }

            Y_ptr = Y_ptr_new;
        }
        for (int f_offset = 0; f_offset < conv_out_channels_; f_offset+=F_BATCH) {
            for (int s_offset = 0; s_offset < conv_in_channels_; s_offset+=S_BATCH) {
                for (int ff = 0; ff < F_BATCH; ff++) {
                    for (int ss = 0; ss < S_BATCH; ss++) {
                        int f = f_offset + ff;
                        int s = s_offset + ss;

                        int access_f_offset = f * height_out_;
                        int access_s_offset = s * height_;

                        for (int g = 0; g < NUM_GAUSS; ++g) {

                            int param_output_offset = OFFSET(0, s,g,f, 1, conv_in_channels_, NUM_GAUSS, conv_out_channels_);

                            int param_offset = -1;
                            if (INPUT_FORMAT == DAUConvForward<float>::SGF)
                                param_offset = OFFSET(0, s,g,f, 1, conv_in_channels_, NUM_GAUSS, conv_out_channels_);
                            else if (INPUT_FORMAT == DAUConvForward<float>::FGS)
                                param_offset = OFFSET(0, f,g,s, 1, conv_out_channels_, NUM_GAUSS, conv_in_channels_);

                            float w = filter_weights[param_offset];

                            float offset_x = filter_offsets_float_mu1[param_offset] - (offsets_already_centered == false ? kernel_width/2 : 0);
                            float offset_y = filter_offsets_float_mu2[param_offset] - (offsets_already_centered == false ? kernel_height/2 : 0);

                            int offset_x_int = floor(offset_x);
                            int offset_y_int = floor(offset_y);

                            float interpol_off_x = offset_x - offset_x_int;
                            float interpol_off_y = offset_y - offset_y_int;

                            for (int dy = 0; dy < INTERPOlATION_Dy; ++dy) {
                                for (int dx = 0; dx < INTERPOlATION_Dx; ++dx) {

                                    int access_x_off = offset_x_int + dx;
                                    int access_y_off = offset_y_int + dy;

                                    float interpol_w = 1;

                                    interpol_w *= (dx == 0 ? (1-interpol_off_x) : interpol_off_x);
                                    interpol_w *= (dy == 0 ? (1-interpol_off_y) : interpol_off_y);

                                    int copy_width = std::min(width_out_ + access_x_off, width_out_ - access_x_off);
                                    int copy_height = std::min(height_out_ + access_y_off, height_out_ - access_y_off);

                                    // X == interm_data
                                    int X_offset_x = std::max(0, access_x_off);
                                    int X_offset_y = std::max(0, access_y_off) + access_s_offset;

                                    // Y == top_data
                                    int Y_offset_x = std::max(0, -access_x_off);
                                    int Y_offset_y = std::max(0, -access_y_off) + access_f_offset;
                                    /*
                                    cv::Rect interm_roi(std::max(0, access_x_off),
                                                        std::max(0, access_y_off) + access_s_offset,
                                                        std::min(width_out_ + access_x_off, width_out_ - access_x_off),
                                                        std::min(height_out_ + access_y_off, height_out_ - access_y_off));

                                    cv::Rect top_roi(std::max(0, -access_x_off),
                                                     std::max(0, -access_y_off) + access_f_offset,
                                                     std::min(width_out_ + access_x_off, width_out_ - access_x_off),
                                                     std::min(height_out_ + access_y_off, height_out_ - access_y_off));
                                    */

                                    //if (top_roi.width > 0 && top_roi.height > 0 && interm_roi.width > 0 && interm_roi.height > 0) {
                                    if (copy_width > 0 && copy_height > 0) {

                                        Dtype tmp = cpu_dot_elementwise_skip(X_ptr, X_width, X_height, X_offset_x, X_offset_y,
                                                                             Y_ptr, Y_width, Y_height, Y_offset_x, Y_offset_y,
                                                                             copy_width, copy_height);

                                        //output_data[param_output_offset] += interpol_w * top_mat(top_roi).dot(interm_mat(interm_roi));
                                        output_data[param_output_offset] += interpol_w * tmp;


                                        /*if (f == 0 && s == 0 && g == 0)
                                        {
                                            printf("sum of f,s,g=%d,%d,%d, n=%d from ", f,s,g, n);
                                            std::cout << "sum " << output_data[param_output_offset] << " and top " << top_mat(top_roi) << " with top roi: " << top_roi  << " and inter roi: " << interm_roi << " and inter val " << interm_mat(interm_roi) << " and w: " << interpol_w << std::endl;
                                        }*/
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        if (ignore_edge_gradients)
            delete Y_ptr;
    }
}
template <typename Dtype>
void BaseDAUConvLayer<Dtype>::Backward_cpu(const Dtype* top_data, const Dtype* top_error, const vector<int>& top_shape, bool propagate_down,
                                           const Dtype* bottom_data, Dtype* bottom_error, const vector<int>& bottom_shape, const vector<bool>& params_propagate_down ) {

    DAU_CHECK(this->is_data_on_gpu() == false, "Backward_cpu requires data on CPU, but is_data_on_gpu() returned true !");
// get buffers for all parameters that we learn
    const Dtype* filter_weights = this->param_w();
    const Dtype* filter_offsets_float_mu1 = this->param_mu1();
    const Dtype* filter_offsets_float_mu2 = this->param_mu2();


    Dtype* param_weights_diff = this->param_w_grad();
    Dtype* param_mu1_diff = this->param_mu1_grad();
    Dtype* param_mu2_diff = this->param_mu2_grad();
    Dtype* param_sigma_diff = this->param_sigma_grad();

    Dtype* bias_diff = this->param_bias_grad();

    Dtype* bwd_gradients_data = this->temp_bwd_gradients();

    // get filters for back-propagation
    const Dtype* deriv_error_kernel = this->get_deriv_kernel_error(stream_);

    // get filters for param gradients
    const Dtype* deriv_kernels_data  = this->get_deriv_kernel_params(stream_);

    // intermediate data for blurred input
    Dtype* interm_data = this->temp_interm_buffer();

    // transform all four accumulated gradients into seperabe buffers of size [S x G x F]
    int param_size = this->units_per_channel * this->conv_in_channels_ * this->conv_out_channels_;

    {
        // input data
        //const Dtype* bottom_data = bottom[i]->cpu_data();
        //Dtype* bottom_error = bottom[i]->mutable_cpu_diff();

        // actual output data
        //const Dtype* top_data = top[i]->cpu_data();
        //const Dtype* top_error = top[i]->cpu_diff();


        // perform back-propagation of the error values first (we may override errors at boundries to make compatible with GPU version)
        if (propagate_down) {
            // we need to do pre-filtering of the error values

            // make sure col_buffer is big enough

            Dtype* col_buff = this->temp_col_buffer();

            int border_x = this->width_/2 - this->width_out_/2;
            int border_y = this->height_/2 - this->height_out_/2;

            border_x = border_x > 0 ? border_x : 0;
            border_y = border_y > 0 ? border_y : 0;

            // over all top errors where each output channel is considered individual sample as well
            for (int n = 0; n < this->batch_num_ * this->conv_out_channels_; ++n) {

                im2col_cpu(top_error + n * (this->height_out_* this->width_out_), 1, this->height_out_, this->width_out_,
                           this->aggregation.kernel_h_, this->aggregation.kernel_w_,
                           this->aggregation.pad_h_ + border_y, this->aggregation.pad_w_ + border_x,
                           this->aggregation.stride_h_, this->aggregation.stride_w_,
                           1,1, col_buff);

                caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, 1 , this->height_ * this->width_, this->aggregation.kernel_h_ * this->aggregation.kernel_w_,
                                      (Dtype)1., deriv_error_kernel, col_buff,
                                      (Dtype)0., interm_data + n * this->width_ * this->height_ );
            }

            // then use our custom kernel for forwarding, however we need to transpose kernels, which in our case means
            // that we need to rotate mu1,mu2 locations

            // we can re-use bwd_gradients_data buffer for mu1 and mu2 that are rotated
            Dtype *param_mu1_backprop = this->temp_param_buffer() + 0 * param_size;
            Dtype *param_mu2_backprop = this->temp_param_buffer() + 1 * param_size;

            // rot(mu) = (kernel_w-1) - mu
            {
                memcpy(param_mu1_backprop, filter_offsets_float_mu1, sizeof(Dtype) * param_size);
                memcpy(param_mu2_backprop, filter_offsets_float_mu2, sizeof(Dtype) * param_size);

                caffe_scal(param_size, (Dtype)-1, param_mu1_backprop);
                caffe_scal(param_size, (Dtype)-1, param_mu2_backprop);

                // if params are already centered then nothing else needed
                if (this->offsets_already_centered_ == false) {
                    caffe_add_scalar(param_size, (Dtype) (this->kernel_w_ - 1), param_mu1_backprop);
                    caffe_add_scalar(param_size, (Dtype) (this->kernel_h_ - 1), param_mu2_backprop);
                }
            }

            // now we take the blured error data and perform sum over shifted input data with our custom kernel i.e. forward pass
            offset_and_sum_opencv(interm_data,
                                  filter_weights, param_mu1_backprop, param_mu2_backprop,
                                  bottom_error,
                                  this->batch_num_, this->conv_out_channels_, this->units_per_channel, this->conv_in_channels_,
                                  this->width_, this->height_,
                                  this->width_, this->height_, this->kernel_w_, this->kernel_h_,
                                  this->offsets_already_centered_, DAUConvForward<float>::FGS);


        }
        // Gradient w.r.t. bias.
        if (this->bias_term_ && params_propagate_down[4]) {

            Dtype* bias_diff = this->param_bias_grad();
            for (int n = 0; n < this->batch_num_; ++n) {
                this->backward_cpu_bias(bias_diff, top_error + n * this->top_dim_);
            }
        }

        // Gradient w.r.t w,mu1,mu2 and sigma
        if (params_propagate_down[0]) {

            // first pre-filter input data with appropriate derivative filters
            int size_batch_k = this->batch_num_ * this->conv_in_channels_ * this->width_ * this->height_;

            Dtype* col_buff = this->temp_col_buffer();

            for (int n = 0; n < this->batch_num_ * this->conv_in_channels_; ++n) {

                im2col_cpu(bottom_data + n * (this->height_* this->width_), 1, this->height_, this->width_,
                           this->aggregation.kernel_h_, this->aggregation.kernel_w_,
                           this->aggregation.pad_h_, this->aggregation.pad_w_,
                           this->aggregation.stride_h_, this->aggregation.stride_w_,
                           1,1, col_buff);

                for (int k = 0; k < this->NUM_K; ++k) {
                    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, 1 , this->height_ * this->width_, this->aggregation.kernel_h_ * this->aggregation.kernel_w_,
                                          (Dtype)1., deriv_kernels_data + k * (this->aggregation.kernel_h_ * this->aggregation.kernel_w_) , col_buff,
                                          (Dtype)0., interm_data + n * this->width_ * this->height_ + k * size_batch_k);
                }
            }
            Dtype* top_error_expended = NULL;
            Dtype* top_error_ex = (Dtype*)top_error;

            if (this->width_out_ != this->width_ && this->height_out_ != this->height_) {
                // extend top data if we have top data of not the same size

                int border_x = this->width_/2 - this->width_out_/2;
                int border_y = this->height_/2 - this->height_out_/2;

                border_x = border_x > 0 ? border_x : 0;
                border_y = border_y > 0 ? border_y : 0;

                vector<int> new_shape = {top_shape[this->channel_axis_-1], top_shape[this->channel_axis_], top_shape[this->channel_axis_+1] + 2*border_y, top_shape[this->channel_axis_+2] + 2*border_x};
                int new_count = std::accumulate(new_shape.begin(), new_shape.end(), 1, std::multiplies<int>());
                top_error_expended = new Dtype[new_count];
                //top_error_expended.Reshape(top_shape[this->channel_axis_-1], top_shape[this->channel_axis_], top_shape[this->channel_axis_+1] + 2*border_y, top_shape[this->channel_axis_+2] + 2*border_x);

                top_error_ex = top_error_expended;

                memset(top_error_ex, 0, sizeof(Dtype) * new_count);

                for (int n = 0; n < new_shape[0]; ++n) {
                    for (int c = 0; c < new_shape[1]; ++c) {
                        for (int h = 0; h < top_shape[this->channel_axis_+1]; ++h) {
                            for (int w = 0; w < top_shape[this->channel_axis_+2]; ++w) {
                                int in_offset = OFFSET(n,c,h,w, top_shape[0], top_shape[1], top_shape[2], top_shape[3]);
                                int out_offset = OFFSET(n,c,border_y+h,border_x+w, new_shape[0], new_shape[1], new_shape[2], new_shape[3]);

                                top_error_ex[out_offset] = top_error[in_offset];
                            }
                        }
                    }
                }
            }

            // then collect gradients by shifting convolved bottom input data and multiplying it with the top error data
            for (int k = 0; k < this->NUM_K; ++k) {
                //printf("k=%d\n",k);

                offset_and_dot_opencv(interm_data + k * size_batch_k,
                                      top_error_ex,
                                      filter_weights, filter_offsets_float_mu1, filter_offsets_float_mu2,
                                      bwd_gradients_data + k * param_size,
                                      this->batch_num_, this->conv_in_channels_, this->units_per_channel, this->conv_out_channels_,
                                      this->width_, this->height_,
                                      this->width_, this->height_, this->kernel_w_, this->kernel_h_,
                                      this->ignore_edge_gradients_, this->offsets_already_centered_);

            }
            if (top_error_expended != NULL)
                delete[] top_error_expended;

        }
    }

    // we need accumulate gradients them to the final buffer and add weights to some derivates
    if (params_propagate_down[0] || params_propagate_down[1] ||
        params_propagate_down[2] || params_propagate_down[3]) {
        // multiply gradients with appropriate weights
        /// add add weight multiplyer as specifed by derivative formula only for mu1,mu2 and sigma
        if (NUM_K > 1 && params_propagate_down[1]) caffe_mul(param_size, bwd_gradients_data + 1 * param_size, filter_weights, bwd_gradients_data + 1 * param_size); // mu1
        if (NUM_K > 2 && params_propagate_down[2]) caffe_mul(param_size, bwd_gradients_data + 2 * param_size, filter_weights, bwd_gradients_data + 2 * param_size); // mu2
        if (NUM_K > 3 && params_propagate_down[3]) caffe_mul(param_size, bwd_gradients_data + 3 * param_size, filter_weights, bwd_gradients_data + 3 * param_size); // sigma

        // for weight gradient we only accumulate to final buffer
        if (NUM_K > 0 && params_propagate_down[0]) caffe_axpy(param_size, (Dtype)1, bwd_gradients_data + 0 * param_size, param_weights_diff); // w
        if (NUM_K > 1 && params_propagate_down[1]) caffe_axpy(param_size, (Dtype)1, bwd_gradients_data + 1 * param_size, param_mu1_diff); // mu1
        if (NUM_K > 2 && params_propagate_down[2]) caffe_axpy(param_size, (Dtype)1, bwd_gradients_data + 2 * param_size, param_mu2_diff); // mu2
        if (NUM_K > 3 && params_propagate_down[3]) caffe_axpy(param_size, (Dtype)1, bwd_gradients_data + 3 * param_size, param_sigma_diff); // sigma

        // if we need to ignore last few gauss then make sure we do not update their parameters
        if (this->num_units_ignore > 0) {
            this->set_last_n_gauss_to_zero(param_weights_diff, this->num_units_ignore);
            this->set_last_n_gauss_to_zero(param_mu1_diff, this->num_units_ignore);
            this->set_last_n_gauss_to_zero(param_mu2_diff, this->num_units_ignore);
            this->set_last_n_gauss_to_zero(param_sigma_diff, this->num_units_ignore);
        }
    }
}

template <typename Dtype>
void BaseDAUConvLayer<Dtype>::forward_cpu_bias(Dtype* output, const Dtype* bias){
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, this->conv_out_channels_,
                          this->out_spatial_dim_, 1, (Dtype)1., bias, this->temp_bias_multiplier(),
                          (Dtype)1., output);
}

template <typename Dtype>
void BaseDAUConvLayer<Dtype>::backward_cpu_bias(Dtype* bias, const Dtype* input) {
    caffe_cpu_gemv<Dtype>(CblasNoTrans, this->conv_out_channels_, this->out_spatial_dim_, 1.,
                          input, this->temp_bias_multiplier(), 1., bias);
}

template <typename Dtype>
void BaseDAUConvLayer<Dtype>::forward_gpu_bias(Dtype* output, const Dtype* bias) {
    caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, this->conv_out_channels_,
                          out_spatial_dim_, 1, (Dtype)1., bias, temp_bias_multiplier(),
                          (Dtype)1., output, cublas_handle);
}

template <typename Dtype>
void BaseDAUConvLayer<Dtype>::backward_gpu_bias(Dtype* bias, const Dtype* input) {
    caffe_gpu_gemv<Dtype>(CblasNoTrans, this->conv_out_channels_, out_spatial_dim_, 1.,
                          input, temp_bias_multiplier(), 1., bias, cublas_handle);
}

template class BaseDAUConvLayer<float>;
template class BaseDAUConvLayer<double>;

}   // namespace dau_conv_impl
