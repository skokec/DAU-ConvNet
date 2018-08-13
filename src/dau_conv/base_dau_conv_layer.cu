#include <vector>
#include <memory>
#include <cmath>

#include "dau_conv/base_dau_conv_layer.hpp"

#include "dau_conv/dau_conv_impl/dau_conv_forward.hpp"
#include "dau_conv/dau_conv_impl/dau_conv_backward.hpp"

#include "dau_conv/util/math_functions.hpp"
#include "dau_conv/util/convolve.hpp"

namespace DAUConvNet {

template <typename Dtype>
void BaseDAUConvLayer<Dtype>::Forward_gpu(const Dtype* bottom_data, const vector<int>& bottom_shape,
                                          Dtype* top_data, const vector<int>& top_shape) {
	// - first perform gaussian bluring based on variance that is fixed over the whole layer (use CuDNN for that)
	// - then perform forward pass with our custom kernel
	// - optionally add bias
    CHECK(this->is_data_on_gpu() == true, "Forward_gpu requires data on GPU, but is_data_on_gpu() returned false !");

	// check if we need to do merging of components;
	// make sure we check based on steps done in backpropagation and we should avoid merging if only forward is called (by default current_iteration_index=0 so start at second iter
	bool do_merginig_optmization = this->unit_merge_iteration_step > 0 && (this->current_iteration_index + 1) % this->unit_merge_iteration_step == 0 ? true : false;

	// if during training then merge components if needed
	if (do_merginig_optmization) {
		//merge_components();
	}

    // before we get params we need to ensure params are within valid bounds
    {
        // we still need to ensure our values are within valid bounds
        // clip sigma, mu1 and mu2 to within bounds
        caffe_gpu_clip_lower(this->conv_in_channels_*this->units_per_channel*this->conv_out_channels_, this->unit_sigma_lower_bound, this->param_sigma(), this->param_sigma());

		Dtype mu1_lower_limit = this->offsets_already_centered_ == false ? (Dtype)unit_border_bound : (-1* (int)(this->max_kernel_w_/2) + (Dtype)unit_border_bound);
		Dtype mu2_lower_limit = this->offsets_already_centered_ == false ? (Dtype)unit_border_bound : (-1* (int)(this->max_kernel_h_/2) + (Dtype)unit_border_bound);

		Dtype mu1_upper_limit = this->offsets_already_centered_  == false ? this->kernel_w_-1 - (Dtype)unit_border_bound : ((int)(this->max_kernel_w_/2) - (Dtype)unit_border_bound);
		Dtype mu2_upper_limit = this->offsets_already_centered_  == false ? this->kernel_h_-1 - (Dtype)unit_border_bound : ((int)(this->max_kernel_h_/2) - (Dtype)unit_border_bound);

        caffe_gpu_clip_lower(this->conv_in_channels_*this->units_per_channel*this->conv_out_channels_, mu1_lower_limit, this->param_mu1(), this->param_mu1());
        caffe_gpu_clip_lower(this->conv_in_channels_*this->units_per_channel*this->conv_out_channels_, mu2_lower_limit, this->param_mu2(), this->param_mu2());

        caffe_gpu_clip_upper(this->conv_in_channels_*this->units_per_channel*this->conv_out_channels_, mu1_upper_limit, this->param_mu1(), this->param_mu1());
        caffe_gpu_clip_upper(this->conv_in_channels_*this->units_per_channel*this->conv_out_channels_, mu2_upper_limit, this->param_mu2(), this->param_mu2());
    }

	const int height_out = top_shape[this->channel_axis_ + 1];
	const int width_out = top_shape[this->channel_axis_ + 2];

	// get filter for gaussian blur step
	const Dtype* gauss_kernel = this->get_gaussian_kernel(stream_[0]);

    // get buffers for all parameters that we learn
	const Dtype* filter_weights = this->param_w();
	const Dtype* filter_offsets_float_mu1 = this->param_mu1();
	const Dtype* filter_offsets_float_mu2 = this->param_mu2();

    // number of all parameters
    int param_size = this->units_per_channel * this->conv_in_channels_ * this->conv_out_channels_;

    cudaEvent_t memset_top, memset_filter;
    CUDA_CHECK(cudaEventCreate(&memset_top));
    CUDA_CHECK(cudaEventCreate(&memset_filter));

	{
		// intermediate data for blurred input
		Dtype* interm_data = this->temp_interm_buffer();

		// convolve with kernel
		{
			caffe_gpu_set_async<Dtype>(this->conv_out_channels_* this->batch_num_* this->height_out_* this->width_out_, (Dtype)0, top_data, paralel_streams[0]);
			caffe_gpu_set_async<Dtype>(buffer_fwd_.filtered_images_sizes_ / sizeof(float), (Dtype)0, buffer_fwd_.filtered_images, paralel_streams[1]);

			CUDA_CHECK(cudaEventRecord(memset_top, paralel_streams[0]));
			CUDA_CHECK(cudaEventRecord(memset_filter, paralel_streams[1]));

			conv2_data_desc sig_desc(1, this->conv_in_channels_* this->batch_num_, this->height_, this->width_,
									 this->conv_in_channels_* this->batch_num_*this->height_*this->width_, this->height_*this->width_, this->width_, 1);
			conv2_data_desc filt_desc(1,1,this->aggregation.kernel_h_,this->aggregation.kernel_w_,
									  this->aggregation.kernel_h_ * this->aggregation.kernel_w_, this->aggregation.kernel_h_ * this->aggregation.kernel_w_, this->aggregation.kernel_w_, 1);

			conv2_data_desc out_desc = sig_desc;

			caffe_gpu_convolve2(interm_data, out_desc,
								bottom_data, sig_desc,
								gauss_kernel, filt_desc, stream_[0]);

			CUDA_CHECK(cudaStreamWaitEvent(stream_[0], memset_top, 0));
			CUDA_CHECK(cudaStreamWaitEvent(stream_[0], memset_filter, 0));
		}

        // Get maximum offset for optimizing CUDA kernel
        Dtype actual_max_offset = (std::max<int>(this->kernel_w_, this->kernel_h_) / 2);

        // optimize only when offsets are already centered and we can quickly get max abs offset value
        if (this->offsets_already_centered_ && this->dynamic_kernel_size_) {
            Dtype max_mu1 = 0, max_mu2 = 0;

            caffe_gpu_amax(param_size, filter_offsets_float_mu1, &max_mu1, cublas_handle);
            caffe_gpu_amax(param_size, filter_offsets_float_mu2, &max_mu2, cublas_handle);

            actual_max_offset = std::max<Dtype>(std::abs(max_mu1), std::abs(max_mu2));
        }

        this->forward_obj->forward_pass(interm_data,
										filter_offsets_float_mu1, filter_offsets_float_mu2, filter_weights, DAUConvForward<Dtype>::SGF,
										this->kernel_w_, this->kernel_h_, actual_max_offset, this->offsets_already_centered_,
										top_data,
										buffer_fwd_.filtered_images,
										NULL,
										NULL,
                                        buffer_fwd_.filter_offsets_and_weights, stream_[0]);

		// add bias if needed
		if (this->bias_term_) {
            const Dtype* bias_data = this->param_bias();

            this->forward_gpu_bias(top_data, bias_data);
		}
	}
    CUDA_CHECK(cudaEventDestroy(memset_top));
    CUDA_CHECK(cudaEventDestroy(memset_filter));
}


template <typename Dtype>
void BaseDAUConvLayer<Dtype>::Backward_gpu(const Dtype* top_data, const Dtype* top_error, const vector<int>& top_shape, bool propagate_down,
                                           const Dtype* bottom_data, Dtype* bottom_error, const vector<int>& bottom_shape, const vector<bool>& params_propagate_down ) {

    //  - first convolve bottom input data with kernels for individual parameters (w, mu1, mu2, sigma)
	//  - then compute and collect gradients by shifting convolved bottom input data and multiplying it with the top error data
	//  - finally back-propagade the error by convolving top error with the rotated filters (we can use the same function as for forward-pass, but need to transpose mu1 and mu2 values)

    CHECK(this->is_data_on_gpu() == true, "Backward_gpu requires data on GPU, but is_data_on_gpu() returned false !");

    this->current_iteration_index++;
    //return;

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
	const Dtype* deriv_error_kernel = this->get_deriv_kernel_error(stream_[0]);

    // get filters for param gradients
	const Dtype* deriv_kernels_data  = this->get_deriv_kernel_params(stream_[0]);

	// intermediate data for blurred input
	Dtype* interm_data = this->temp_interm_buffer();

	// transform all four accumulated gradients into seperabe buffers of size [S x G x F]
	int param_size = this->units_per_channel * this->conv_in_channels_ * this->conv_out_channels_;

	// make sure gradient aggregation buffer is zeroed
	caffe_gpu_memset(param_size * NUM_K * sizeof(Dtype), 0, bwd_gradients_data);

	cudaEvent_t memset_top, memset_filter, memset_error;
	CUDA_CHECK(cudaEventCreate(&memset_top));
	CUDA_CHECK(cudaEventCreate(&memset_filter));
	CUDA_CHECK(cudaEventCreate(&memset_error));

	{
		// Gradient w.r.t. bias.
		if (this->bias_term_ && params_propagate_down[4]) {
            this->backward_gpu_bias(bias_diff, top_error);
		}

		// Get maximum offset for optimizing CUDA kernel
		Dtype actual_max_offset = (std::max<int>(this->kernel_w_, this->kernel_h_) / 2);

		// optimize only when offsets are already centered and we can quickly get max abs offset value
		if (this->offsets_already_centered_ && this->dynamic_kernel_size_) {
			Dtype max_mu1 = 0, max_mu2 = 0;

			caffe_gpu_amax(param_size, filter_offsets_float_mu1, &max_mu1, cublas_handle);
			caffe_gpu_amax(param_size, filter_offsets_float_mu2, &max_mu2, cublas_handle);

            actual_max_offset = std::max<Dtype>(std::abs(max_mu1), std::abs(max_mu2));
		}

		// Gradient w.r.t w,mu1,mu2 and sigma
		if (params_propagate_down[0]) {

			// TODO: if it is faster we should add zeroing to input prepare functions !!

			// convolve with kernel
			{
                caffe_gpu_set_async(this->buffer_bwd_.filtered_images_sizes_/sizeof(Dtype), (Dtype)0, this->buffer_bwd_.filtered_images, paralel_streams[0]);
                caffe_gpu_set_async(this->buffer_bwd_.error_image_sizes_/sizeof(Dtype), (Dtype)0, this->buffer_bwd_.error_images, paralel_streams[1]);

                CUDA_CHECK(cudaEventRecord(memset_filter, paralel_streams[0]));
                CUDA_CHECK(cudaEventRecord(memset_error, paralel_streams[1]));

                conv2_data_desc sig_desc(this->conv_in_channels_* this->batch_num_, 1, this->height_, this->width_,
										 this->height_*this->width_,  this->height_*this->width_, this->width_, 1);

				conv2_data_desc filt_desc(1,this->NUM_K,this->aggregation.kernel_h_,this->aggregation.kernel_w_,
										  this->NUM_K * this->aggregation.kernel_h_ * this->aggregation.kernel_w_, this->aggregation.kernel_h_ * this->aggregation.kernel_w_, this->aggregation.kernel_w_, 1);

				conv2_data_desc out_desc(this->conv_in_channels_* this->batch_num_, this->NUM_K, this->height_, this->width_,
										 this->height_*this->width_ * this->NUM_K,  this->height_*this->width_, this->width_, 1);

				caffe_gpu_convolve2(interm_data, out_desc,
									bottom_data, sig_desc,
									deriv_kernels_data, filt_desc, stream_[0]);


                CUDA_CHECK(cudaStreamWaitEvent(stream_[0], memset_filter, 0));
                CUDA_CHECK(cudaStreamWaitEvent(stream_[0], memset_error, 0));

			}

			// collect gradients by shifting convolved bottom input data and multiplying it with the top error data

            // WARNING: if this->kernel_w_ or this->kernel_h_ changes then memory will not be allocated properly
			backward_grad_obj->backward_pass(interm_data, top_error,
									   filter_offsets_float_mu1, filter_offsets_float_mu2,
									   filter_weights, this->kernel_w_, this->kernel_h_, actual_max_offset,
   									   this->offsets_already_centered_,
									   bwd_gradients_data,
									   this->buffer_bwd_.filtered_images,
									   this->buffer_bwd_.error_images,
									   this->buffer_bwd_.filter_weights,
									   this->buffer_bwd_.filter_offsets,
                                       this->ignore_edge_gradients_, 0);
									   //this->ignore_edge_gradients_, stream_[0]);

		}


		// finally perform back-propagation of the error values
		if (propagate_down) {

            Dtype const* top_error_for_bwd = top_error;

            // if size top_error (input) is smaller then interm_data (output)  (i.e. expected input should be the same size as output)
			// then we need to copy top_error to bigger buffer i.e. with padded zeros
			if (buffer_bwd_.resized_top_for_bwd_sizes_ > 0) {
				// set zeros
				caffe_gpu_set_async<Dtype>(buffer_bwd_.resized_top_for_bwd_sizes_ / sizeof(float), (Dtype)0, buffer_bwd_.resized_top_for_bwd, stream_[0]);

				// then copy but with appropriate padding
				caffe_gpu_pad2d(this->batch_num_ * this->conv_out_channels_, this->height_out_, this->width_out_, this->width_/2 - this->width_out_/2, top_error, buffer_bwd_.resized_top_for_bwd, stream_[0]);

                top_error_for_bwd = buffer_bwd_.resized_top_for_bwd;
			}
			// convolve with kernels
			{

                // NOTE: memory buffer is shared with gradient compute so make sure not to zero it before backward_grad_obj->backward_pass is done

                caffe_gpu_set_async<Dtype>(this->conv_in_channels_* this->batch_num_* this->height_* this->width_, (Dtype)0, bottom_error, paralel_streams[0]);
                caffe_gpu_set_async<Dtype>(buffer_fwd_.filtered_images_sizes_ / sizeof(float), (Dtype)0, buffer_fwd_.filtered_images, paralel_streams[1]);

                CUDA_CHECK(cudaEventRecord(memset_top, paralel_streams[0]));
                CUDA_CHECK(cudaEventRecord(memset_filter, paralel_streams[1]));

				int max_width = std::max(this->width_out_,this->width_);
				int max_height = std::max(this->height_out_,this->height_);

				conv2_data_desc sig_desc(1, this->conv_out_channels_* this->batch_num_, max_height, max_width,
										 this->conv_out_channels_* this->batch_num_*max_height*max_width, max_height*max_width, max_width, 1);

				conv2_data_desc filt_desc(1,1,this->aggregation.kernel_h_,this->aggregation.kernel_w_,
										  this->aggregation.kernel_h_ * this->aggregation.kernel_w_, this->aggregation.kernel_h_ * this->aggregation.kernel_w_, this->aggregation.kernel_w_, 1);

				conv2_data_desc out_desc = sig_desc;


				caffe_gpu_convolve2(interm_data, out_desc,
                                    top_error_for_bwd, sig_desc,
									deriv_error_kernel, filt_desc, stream_[0]);

                CUDA_CHECK(cudaStreamWaitEvent(stream_[0], memset_top, 0));
                CUDA_CHECK(cudaStreamWaitEvent(stream_[0], memset_filter, 0));

			}
			// then use our custom kernel for forwarding, however we need to transpose kernels, which in our case means
			// that we need to rotate mu1,mu2 locations


			// get param buffer for mu1 and mu2 that will be rotated
			Dtype *param_mu1_backprop = this->temp_param_buffer() + 0 * param_size;
			Dtype *param_mu2_backprop = this->temp_param_buffer() + 1 * param_size;

			// rot(mu) = (kernel_w-1) - mu
			{
				caffe_gpu_memcpy_async(param_size * sizeof(float), filter_offsets_float_mu1, param_mu1_backprop, 0);
				caffe_gpu_memcpy_async(param_size * sizeof(float), filter_offsets_float_mu2, param_mu2_backprop, 0);

				caffe_gpu_scal(param_size, (Dtype)-1, param_mu1_backprop, cublas_handle);
				caffe_gpu_scal(param_size, (Dtype)-1, param_mu2_backprop, cublas_handle);

				// if params are already centered then nothing else needed
				if (this->offsets_already_centered_ == false) {
					caffe_gpu_add_scalar(param_size, (Dtype) (this->kernel_w_ - 1), param_mu1_backprop);
					caffe_gpu_add_scalar(param_size, (Dtype) (this->kernel_h_ - 1), param_mu2_backprop);
				}
			}


			// now we take the blured error data and perform sum over shifted input data with our custom kernel i.e. forward pass
			this->backward_backporp_obj->forward_pass(interm_data,
													  param_mu1_backprop, param_mu2_backprop, filter_weights, DAUConvForward<Dtype>::FGS,
													  this->kernel_w_, this->kernel_h_, actual_max_offset, this->offsets_already_centered_,
													  bottom_error,
													  buffer_fwd_.filtered_images,
													  NULL,
													  NULL,
													  buffer_fwd_.filter_offsets_and_weights, stream_[0]);

		}
	}
	// we need to accumulate gradients to the final buffer and add weights to some derivates
	if (params_propagate_down[0] || params_propagate_down[1] ||
        params_propagate_down[2] || params_propagate_down[3]) {
		// multiply gradients with appropriate weights
		/// add add weight multiplyer as specifed by derivative formula only for mu1,mu2 and sigma
		if (NUM_K > 1 && params_propagate_down[1]) caffe_gpu_mul(param_size, bwd_gradients_data + 1 * param_size, filter_weights, bwd_gradients_data + 1 * param_size); // mu1
		if (NUM_K > 2 && params_propagate_down[2]) caffe_gpu_mul(param_size, bwd_gradients_data + 2 * param_size, filter_weights, bwd_gradients_data + 2 * param_size); // mu2
		if (NUM_K > 3 && params_propagate_down[3]) caffe_gpu_mul(param_size, bwd_gradients_data + 3 * param_size, filter_weights, bwd_gradients_data + 3 * param_size); // sigma

		// for weight gradient we only accumulate to final buffer
		if (NUM_K > 0 && params_propagate_down[0]) caffe_gpu_axpy(param_size, (Dtype)1, bwd_gradients_data + 0 * param_size, param_weights_diff, cublas_handle); // w
		if (NUM_K > 1 && params_propagate_down[1]) caffe_gpu_axpy(param_size, (Dtype)1, bwd_gradients_data + 1 * param_size, param_mu1_diff, cublas_handle); // mu1
		if (NUM_K > 2 && params_propagate_down[2]) caffe_gpu_axpy(param_size, (Dtype)1, bwd_gradients_data + 2 * param_size, param_mu2_diff, cublas_handle); // mu2
		if (NUM_K > 3 && params_propagate_down[3]) caffe_gpu_axpy(param_size, (Dtype)1, bwd_gradients_data + 3 * param_size, param_sigma_diff, cublas_handle); // sigma

        // if we need to ignore last few gauss then make sure we do not update their parameters
        if (this->num_units_ignore > 0) {
            this->set_last_n_gauss_to_zero(param_weights_diff, this->num_units_ignore);
            this->set_last_n_gauss_to_zero(param_mu1_diff, this->num_units_ignore);
            this->set_last_n_gauss_to_zero(param_mu2_diff, this->num_units_ignore);
            this->set_last_n_gauss_to_zero(param_sigma_diff, this->num_units_ignore);
        }

        // convert NaN to 0
        if (NUM_K > 1 && params_propagate_down[1]) caffe_gpu_clip_nan(param_size, param_mu1_diff, param_mu1_diff); // mu1
        if (NUM_K > 2 && params_propagate_down[2]) caffe_gpu_clip_nan(param_size, param_mu2_diff, param_mu2_diff); // mu2
	}

    CUDA_CHECK(cudaEventDestroy(memset_top));
    CUDA_CHECK(cudaEventDestroy(memset_filter));
    CUDA_CHECK(cudaEventDestroy(memset_error));

}

template <typename Dtype>
__global__ void set_last_n_gauss_to_zero_kernel(const int S, const int G, const int F, Dtype* x, int num_gauss_zero) {
    CUDA_KERNEL_LOOP(index, S*G*F) {
        int f = (index % F) ;
        int sg = index / F;
        int g = (sg % G);
        int s = sg / G;

        if (g  >= G - num_gauss_zero)
            x[index] = 0;
    }
}

template <typename Dtype>
void BaseDAUConvLayer<Dtype>::set_last_n_gauss_to_zero(Dtype* array, int num_gauss_zero){
    set_last_n_gauss_to_zero_kernel<Dtype><<<CUDA_GET_BLOCKS(this->conv_in_channels_ * this->units_per_channel * this->conv_out_channels_), CUDA_NUM_THREADS>>>(this->conv_in_channels_, this->units_per_channel, this->conv_out_channels_, array, num_gauss_zero);
}


// TODO: we could speed-up with vectorized read/write

// pre-compute sigma inverse values needed in Gaussian distribution (1/sigma^2, 1/sigma^3 and 1/2*1/sigma^2)
template <typename Dtype>
__global__ void conv_gauss_precompute_sigma_kernel(const int n, Dtype* buf_sigma, Dtype* buf_sigma_square_inv, Dtype* buf_sigma_cube_inv, Dtype* buf_sigma_square_inv_half, const int sigma_lower_bound) {
	CUDA_KERNEL_LOOP(index, n) {
		Dtype sigma_value = buf_sigma[index];

		Dtype sigma2 = sigma_value * sigma_value;
		Dtype sigma2_inv = 1/sigma2;

		buf_sigma[index] = sigma_value;
		buf_sigma_square_inv[index] = sigma2_inv;
		buf_sigma_cube_inv[index] = 1/(sigma2 * sigma_value);
		buf_sigma_square_inv_half[index] = (0.5 * sigma2_inv) ;
	}
}

template <typename Dtype>
__global__ void conv_gauss_distributions_kernel(const int N, const int k_w, int k_h, bool offsets_already_centered,
												const Dtype* W, const Dtype* MU1, const Dtype* MU2, const Dtype* SIGMA_2_INV, const Dtype* SIGMA_3_INV, const Dtype* SIGMA_2_INV_HALF,
												Dtype* guass_dist, Dtype* guass_deriv_mu1, Dtype* guass_deriv_mu2, Dtype* guass_deriv_sigma) {

	const int filter_size = k_w * k_h;

	for (int n = blockIdx.z * blockDim.z + threadIdx.z; n < N; n += blockDim.z * gridDim.z){
		// read w, mu1, mu2, sigma and other data needed to compute gaussian Distributions
		//const Dtype w = W[n];
		const Dtype mu1 = MU1[n] + (offsets_already_centered  ? (int)(k_w/2) : 0);
		const Dtype mu2 = MU2[n] + (offsets_already_centered  ? (int)(k_h/2) : 0);
		const Dtype sigma_square_inv = SIGMA_2_INV[n];
		const Dtype sigma_square_inv_half = SIGMA_2_INV_HALF[n];
		const Dtype sigma_cube_inv = SIGMA_3_INV[n];


		// blockDim by x and y should always be 1 since whole filter will always fit into one block, so just retrive filter x,y indeces and calculate gaussians
		const int x = threadIdx.x;
		const int y = threadIdx.y;

		const Dtype dist_x = x - mu1;
		const Dtype dist_x_2 = dist_x*dist_x;

		const Dtype dist_y = y - mu2;
		const Dtype dist_y_2 = dist_y*dist_y;

		const Dtype dist = dist_x_2 + dist_y_2;
		const Dtype gauss_value = exp( -dist * sigma_square_inv_half);

		const int ptr_offset =  n * filter_size + y * k_w + x;

		guass_dist[ptr_offset] =  gauss_value;
		guass_deriv_mu1[ptr_offset] = (dist_x * sigma_square_inv) * gauss_value;
		guass_deriv_mu2[ptr_offset] = (dist_y * sigma_square_inv) * gauss_value;
		guass_deriv_sigma[ptr_offset] = (dist * sigma_cube_inv) * gauss_value;

	}
}


template <typename Dtype>
__global__ void scal_kernel_batched(const int n, const Dtype* a, const Dtype* x, Dtype* y, const int m) {

	for (int j = blockIdx.y * blockDim.y + threadIdx.y; j < m; j += blockDim.y * gridDim.y) {
		Dtype a_value = a[j];
		for (int i = j * n + blockIdx.x * blockDim.x + threadIdx.x; i < n* (1 + j) ; i += blockDim.x * gridDim.x) {
			y[i] = a_value * x[i];
		}
	}
}


template <typename Dtype>
__global__ void axpby_kernel_batched(const int n, const Dtype a_factor, const Dtype* a, const Dtype* x, const Dtype* b, Dtype* y, const int m) {

	for (int j = blockIdx.y * blockDim.y + threadIdx.y; j < m; j += blockDim.y * gridDim.y) {
		Dtype a_value = a[j] * a_factor;
		Dtype b_value = b[j];
		for (int i = j * n + blockIdx.x * blockDim.x + threadIdx.x; i < n * (1 + j); i += blockDim.x * gridDim.x) {
			y[i] = a_value * x[i] + b_value * y[i];
		}
	}
}

template <typename Dtype>
__global__ void add_sorted_kernel(const int S, const int G, const int F, const int n, const Dtype* unsorted_input, Dtype* sorted_output) {
	for (int f = blockIdx.z * blockDim.z + threadIdx.z; f < F; f += blockDim.z * gridDim.z) {
		for (int s = blockIdx.y * blockDim.y + threadIdx.y; s < S; s += blockDim.y * gridDim.y) {
			for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {

				Dtype sum_g = 0;
				for (int g = 0; g < G; ++g) {
					sum_g += unsorted_input[ ((s*G + g)*F  + f )*n + i];
				}

				sorted_output[(f*S + s)*n + i] = sum_g;
			}
		}
	}
}
template <typename Dtype>
__global__ void inv_kernel(const int n, const Dtype* x, Dtype* y) {
	CUDA_KERNEL_LOOP(index, n) {
		y[index] = 1 / x[index];
	}
}

template <typename Dtype>
__global__ void mirror_kernel(const int S, const int F, const int n, const Dtype* x, Dtype* y) {

	for (int f = blockIdx.z * blockDim.z + threadIdx.z; f < F; f += blockDim.z * gridDim.z) {
		for (int s = blockIdx.y * blockDim.y + threadIdx.y; s < S; s += blockDim.y * gridDim.y) {
			for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
				// perform kernel mirroring by setting y[i] = x[n-i -1]
				// at the same time switch S and F indexes
				y[(s*F + f) * n + i] = x[(f*S + s) * n + n - i -1];
			}
		}
	}
}


template <typename Dtype>
void BaseDAUKernelCompute<Dtype>::get_kernels(BaseDAUKernelParams<Dtype>& input, BaseDAUKernelOutput<Dtype>& output, cublasHandle_t cublas_handle) {

	// we get mutable ptr but we do not modify it, this is just poor code in part of the CUB code
	int* tmp_precomp_index_gpu = this->precomp_index();

	clock_t start_t = clock();

	Dtype* weight = output.weight();

	const Dtype* gauss_params_w = input.weight();
	Dtype* gauss_params_mu1 = input.mu1();
	Dtype* gauss_params_mu2 = input.mu2();
	Dtype* gauss_params_sigma = input.sigma();

	Dtype* gauss_params_sigma_square_inv = this->param_temp(SIGMA_SQUARE_INV);
	Dtype* gauss_params_sigma_cube_inv = this->param_temp(SIGMA_CUBE_INV);
	Dtype* gauss_params_sigma_square_inv_half = this->param_temp(SIGMA_SQUARE_INV_HALF);

	const int S = this->num_in_channels;
	const int F = this->num_out_channels;
	const int G = this->num_gauss;

	const int K_w = this->kernel_w;
	const int K_h = this->kernel_h;

	// clip sigma, mu1 and mu2 to within bounds
	caffe_gpu_clip_lower(S*F*G, this->sigma_lower_bound, gauss_params_sigma, gauss_params_sigma);

	Dtype mu1_lower_limit = this->offsets_already_centered  == false ? (Dtype)component_border_bound : (-1* (int)(kernel_w/2) + component_border_bound);
	Dtype mu2_lower_limit = this->offsets_already_centered  == false ? (Dtype)component_border_bound : (-1* (int)(kernel_h/2) + component_border_bound);

	Dtype mu1_upper_limit = this->offsets_already_centered  == false ? kernel_w-1 - (Dtype)component_border_bound : ((int)(kernel_w/2) - component_border_bound);
	Dtype mu2_upper_limit = this->offsets_already_centered  == false ? kernel_h-1 - (Dtype)component_border_bound : ((int)(kernel_h/2) - component_border_bound);


	caffe_gpu_clip_lower(S*F*G, mu1_lower_limit, gauss_params_mu1, gauss_params_mu1);
	caffe_gpu_clip_lower(S*F*G, mu2_lower_limit, gauss_params_mu2, gauss_params_mu2);

	caffe_gpu_clip_upper(S*F*G, mu1_upper_limit, gauss_params_mu1, gauss_params_mu1);
	caffe_gpu_clip_upper(S*F*G, mu2_upper_limit, gauss_params_mu2, gauss_params_mu2);


	// 0. precompute  sigma^2, sigma^3 and (sigma^2)/2
	conv_gauss_precompute_sigma_kernel<Dtype><<<CUDA_GET_BLOCKS(S*G*F), CUDA_NUM_THREADS>>>(S*G*F, gauss_params_sigma, gauss_params_sigma_square_inv, gauss_params_sigma_cube_inv, gauss_params_sigma_square_inv_half, this->sigma_lower_bound);


	// 1. for each pixel in [SxGxF] x [K_w x K_h] compute G (Gauss distribution), dG/dx, dG/dy, dG/dsigma

	// cuda dimension X runs over K_w, Y over K_h and dimension Z over all filters
	// we translate cuda thread X,Y dimensions directly to filter indexces of size K_w, K_h and assign cuda thread Z dimension with
	// several filters to fill as many CAFFE_CUDA_NUM_THREADS threads available (i.e. multiple filter can be processed in one cuda block)
	dim3 threadsPerBlock(K_w, K_h, CUDA_NUM_THREADS/(K_w * K_h));
	dim3 numBlocks(1, 1, (S*G*F + threadsPerBlock.z - 1) / threadsPerBlock.z);

	Dtype* gauss_dist = this->kernels_temp(GAUSS_DIST);

	size_t d_param_size = S * G* F* K_h * K_w;

	Dtype* deriv_weight = output.d_params() + 0 * d_param_size;
	Dtype* deriv_mu1 = output.d_params() + 1 * d_param_size;
	Dtype* deriv_mu2 = output.d_params() + 2 * d_param_size;
	Dtype* deriv_sigma = output.d_params() + 3 * d_param_size;

	conv_gauss_distributions_kernel<Dtype><<<numBlocks,threadsPerBlock>>>(S*G*F, K_w, K_h, this->offsets_already_centered, gauss_params_w, gauss_params_mu1, gauss_params_mu2, gauss_params_sigma_square_inv, gauss_params_sigma_cube_inv, gauss_params_sigma_square_inv_half, gauss_dist, deriv_mu1, deriv_mu2, deriv_sigma);

	// 2. for each filter (G, dG/dx, dG/dy, dG/dsigma) calculate sums (use different sums if using normalization by square sum)
	Dtype* guass_norm = this->param_temp(GAUSS_NORM);
	Dtype* deriv_mu1_sums = this->param_temp(DERIV_MU1_SUMS);
	Dtype* deriv_mu2_sums = this->param_temp(DERIV_MU2_SUMS);
	Dtype* deriv_sigma_sums = this->param_temp(DERIV_SIGMA_SUMS);

	// TODO: all three sums can be done in parallel, do we need seperate streams to make this run in parallel ?
	if (this->use_unit_normalization == false) {
		// if there is no normalization then there should be no derivative of normalization
		caffe_gpu_set((S*F*G), (Dtype)0, deriv_mu1_sums);
		caffe_gpu_set((S*F*G), (Dtype)0, deriv_mu2_sums);
		caffe_gpu_set((S*F*G), (Dtype)0, deriv_sigma_sums);

	} else if (this->use_square_unit_normalization) {
		// when using square gauss normalization derivatives dG/dx, dG/dy, dG/dsigma need to be multiplied by un-weighted, un-normalized gaussian dstirubution i.e. gauss_dist
		Dtype* deriv_mu1_times_gauss_dist = this->kernels_temp(DERIV_MU1_TIMES_GAUSS_DIST);
		Dtype* deriv_mu2_times_gauss_dist = this->kernels_temp(DERIV_MU2_TIMES_GAUSS_DIST);
		Dtype* deriv_sigma_times_gauss_dist = this->kernels_temp(DERIV_SIGMA_TIMES_GAUSS_DIST);

		caffe_gpu_mul((S*F*G) * (K_w*K_h), gauss_dist, deriv_mu1, deriv_mu1_times_gauss_dist); // deriv_mu1_times_gauss_dist = gauss_dist * deriv_mu1;
		caffe_gpu_mul((S*F*G) * (K_w*K_h), gauss_dist, deriv_mu2, deriv_mu2_times_gauss_dist); // deriv_mu2_times_gauss_dist = gauss_dist * deriv_mu2;
		caffe_gpu_mul((S*F*G) * (K_w*K_h), gauss_dist, deriv_sigma, deriv_sigma_times_gauss_dist); // deriv_sigma_times_gauss_dist = gauss_dist * deriv_sigma;

		caffe_gpu_sum((S*F*G) * (K_w*K_h), deriv_mu1_times_gauss_dist, deriv_mu1_sums, S*F*G, tmp_precomp_index_gpu);
		caffe_gpu_sum((S*F*G) * (K_w*K_h), deriv_mu2_times_gauss_dist, deriv_mu2_sums, S*F*G, tmp_precomp_index_gpu);
		caffe_gpu_sum((S*F*G) * (K_w*K_h), deriv_sigma_times_gauss_dist, deriv_sigma_sums, S*F*G, tmp_precomp_index_gpu);

		caffe_gpu_scal((S*F*G), (Dtype)2, deriv_mu1_sums, cublas_handle);
		caffe_gpu_scal((S*F*G), (Dtype)2, deriv_mu2_sums, cublas_handle);
		caffe_gpu_scal((S*F*G), (Dtype)2, deriv_sigma_sums, cublas_handle);
	} else {

		caffe_gpu_sum((S*F*G) * (K_w*K_h), deriv_mu1, deriv_mu1_sums, S*F*G, tmp_precomp_index_gpu);
		caffe_gpu_sum((S*F*G) * (K_w*K_h), deriv_mu2, deriv_mu2_sums, S*F*G, tmp_precomp_index_gpu);
		caffe_gpu_sum((S*F*G) * (K_w*K_h), deriv_sigma, deriv_sigma_sums, S*F*G, tmp_precomp_index_gpu);
	}

	if (this->use_unit_normalization == false) {
		// set guass_norm to 1 if we sould NOT normalize to sum of 1
		caffe_gpu_set((S*F*G), (Dtype)1, guass_norm);

	} else if (this->use_square_unit_normalization) {
		// we need to normalize to sum of squares to 1
		Dtype* gauss_dist_square = this->kernels_temp(GAUSS_DIST_SQUARE);

		caffe_gpu_mul((S*F*G) * (K_w*K_h), gauss_dist, gauss_dist, gauss_dist_square); // gauss_dist_square = gauss_dist * gauss_dist;
		caffe_gpu_sum((S*F*G) * (K_w*K_h), gauss_dist_square, guass_norm, S*F*G, tmp_precomp_index_gpu);
	} else {
		// we need to normalize to sum of 1
		caffe_gpu_sum((S*F*G) * (K_w*K_h), gauss_dist, guass_norm, S*F*G, tmp_precomp_index_gpu);
	}

	// invert guass_norm i.e. guass_norm = 1/guass_norm
	inv_kernel<Dtype><<<CUDA_GET_BLOCKS(S*G*F), CUDA_NUM_THREADS>>>(S*G*F, guass_norm, guass_norm);

	// gauss_mu1_sum = abs(gauss_mu1_sum) > 1e-10 ? gauss_mu1_sum : 0;
	caffe_gpu_clip_eps(S*F*G, (Dtype)1e-10, deriv_mu1_sums, deriv_mu1_sums);
	caffe_gpu_clip_eps(S*F*G, (Dtype)1e-10, deriv_mu2_sums, deriv_mu2_sums);

	// 3. for each filter G and derivative filters dG/dx, dG/dy, dG/dsigma apply its normalization terms
	threadsPerBlock = dim3(K_w* K_h, CUDA_NUM_THREADS/(K_w * K_h));
	numBlocks = dim3(1, (S*F*G + threadsPerBlock.y - 1) / threadsPerBlock.y);

	// deriv_weight = gauss_dist * guass_norm
	scal_kernel_batched<Dtype><<<numBlocks,threadsPerBlock>>>(K_w * K_h, guass_norm, gauss_dist, deriv_weight, S*F*G);

	// !after! weight and deriv_weight are computed we can add weight to guass_norm which will be used in remaining derivateives and main kernel
	caffe_gpu_mul(S*F*G, gauss_params_w, guass_norm, guass_norm); // guass_norm = gauss_params_w / guass_norm;


	// apply gauss normalization factors directly to sums to avoid additional call to scal_kernel_batched
	caffe_gpu_mul(S*F*G, guass_norm, deriv_mu1_sums, deriv_mu1_sums); // deriv_mu1_sums = deriv_mu1_sums * guass_norm;
	caffe_gpu_mul(S*F*G, guass_norm, deriv_mu2_sums, deriv_mu2_sums); // deriv_mu2_sums = deriv_mu2_sums * guass_norm;
	caffe_gpu_mul(S*F*G, guass_norm, deriv_sigma_sums, deriv_sigma_sums); // deriv_sigma_sums = deriv_sigma_sums * guass_norm;

	// create normalized derivative filters
	axpby_kernel_batched<Dtype><<<numBlocks,threadsPerBlock>>>(K_w * K_h, (Dtype)-1, deriv_mu1_sums, deriv_weight,  guass_norm, deriv_mu1, S*F*G);
	axpby_kernel_batched<Dtype><<<numBlocks,threadsPerBlock>>>(K_w * K_h, (Dtype)-1, deriv_mu2_sums, deriv_weight,  guass_norm, deriv_mu2, S*F*G);
	axpby_kernel_batched<Dtype><<<numBlocks,threadsPerBlock>>>(K_w * K_h, (Dtype)-1, deriv_sigma_sums, deriv_weight,  guass_norm, deriv_sigma, S*F*G);

	// 4. calculate main kernel weights by applying gauss norm and weights, and suming over SxGxF kernels into FxS kernels (in correct order i.e. rearagning them at the same time)

	// gauss_dist = w/norm * gauss_dist (note, guass_norm should be w/norm)
	scal_kernel_batched<Dtype><<<numBlocks,threadsPerBlock>>>(K_w * K_h, guass_norm, gauss_dist, gauss_dist, S*F*G);

	threadsPerBlock = dim3(K_w*K_h, sqrt(CUDA_NUM_THREADS/(K_w * K_h) ), sqrt(CUDA_NUM_THREADS/(K_w * K_h) ) );
	numBlocks = dim3(1, (S + threadsPerBlock.y - 1) / threadsPerBlock.y, (F + threadsPerBlock.z - 1) / threadsPerBlock.z);

	add_sorted_kernel<Dtype><<<numBlocks,threadsPerBlock>>>(S, G, F, K_w*K_h, gauss_dist, weight);

	// 4. calculate seperable filters (WILL NOT IMPLEMENET)

	// 5. create error kernel for back-propagation by reversing the kernel

	Dtype* deriv_error = output.d_error();

	threadsPerBlock = dim3(K_w*K_h, sqrt(CUDA_NUM_THREADS/(K_w * K_h) ), sqrt(CUDA_NUM_THREADS/(K_w * K_h) ) );
	numBlocks = dim3(1, (S + threadsPerBlock.y - 1) / threadsPerBlock.y, (F + threadsPerBlock.z - 1) / threadsPerBlock.z);

	mirror_kernel<Dtype><<<numBlocks,threadsPerBlock>>>(S, F, K_w*K_h, weight, deriv_error);

	//cudaDeviceSynchronize();

	clock_t end_t = clock();
}

template void BaseDAUKernelCompute<float>::get_kernels(BaseDAUKernelParams<float>& input, BaseDAUKernelOutput<float>& output, cublasHandle_t cublas_handle);
template void BaseDAUKernelCompute<double>::get_kernels(BaseDAUKernelParams<double>& input, BaseDAUKernelOutput<double>& output, cublasHandle_t cublas_handle);

template void BaseDAUConvLayer<double>::set_last_n_gauss_to_zero(double* array, int num_gauss_zero);
template void BaseDAUConvLayer<float>::set_last_n_gauss_to_zero(float* array, int num_gauss_zero);


template void BaseDAUConvLayer<double>::Forward_gpu(const double* bottom_data, const vector<int>& bottom_shape,
                                                    double* top_data, const vector<int>& top_shape);
template void BaseDAUConvLayer<float>::Forward_gpu(const float* bottom_data, const vector<int>& bottom_shape,
                                                   float* top_data, const vector<int>& top_shape);
template void BaseDAUConvLayer<double>::Backward_gpu(const double* top_data, const double* top_error, const vector<int>& top_shape, bool propagate_down,
                                                    const double* bottom_data, double* bottom_error, const vector<int>& bottom_shape, const vector<bool>& params_propagate_down );
template void BaseDAUConvLayer<float>::Backward_gpu(const float* top_data, const float* top_error, const vector<int>& top_shape, bool propagate_down,
                                                    const float* bottom_data, float* bottom_error, const vector<int>& bottom_shape, const vector<bool>& params_propagate_down );
}  // namespace dau_conv_impl
