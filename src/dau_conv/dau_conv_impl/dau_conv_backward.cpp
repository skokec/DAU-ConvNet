#include <cmath>
#include <algorithm>

#include "dau_conv/dau_conv_impl/dau_conv_backward.hpp"

#include "dau_conv/util/common.hpp"

namespace DAUConvNet {

template <typename Dtype>
DAUConvBackward<Dtype>::DAUConvBackward(const int img_width_in, const int img_height_in, const int img_width, const int img_height, const int I, const int S, const int F, const int G, const int K, const bool last_k_optional, const bool use_interpolation) :
		img_width_in(img_width_in), img_height_in(img_height_in), img_width(img_width), img_height(img_height), I(I), S(S), F(F), G(G), IN_K(K), use_interpolation(use_interpolation), last_k_optional(last_k_optional) {

	// decide which size of patch to use to minimize wasted memory/processing
    if (img_width == 1 && img_height == 1) {
        patch_size_w = 1;
        patch_size_h = 1;
    } else {
        // do not prefer 8 or 16 sizes since this does not lead to full utilization most of the time (use them only for small sizes so not to waste too much resources)
	    patch_size_w = img_width <= 8 ? 8 :
                        (img_width <= 16 ? 16 : select_optimal_block_size_bw(img_width, 5, 6)); // allowed patch sizes = 2^[5,6] i.e, [32,64]
	    patch_size_h = img_height <= 8 ? 8 :
				    	(img_height <= 16 ? 16 : select_optimal_block_size_bw(img_height, 5, 6)); // allowed patch sizes = 2^[5,6] i.e, [32,64]
	}

	// decide wheather to use:
	//  - 32 pixels per warp
	//  - 16 pixels per warp
    //  - 8 pixels per warp
    //	- 1 pixel per warp

	int boundary_img_width = img_width - floor(img_width/patch_size_w) * patch_size_w;

    // use warp size 1x1 if patch size only 1x1 otherwise use [16,32]x8 (if patch_size_w==8 then use 8x8 but do not prefer it for bigger patches)
	int warp_pixel_size_x = patch_size_w == 1 ? 1 :
                                (patch_size_w <= 8 ? 8 : std::min(patch_size_w, select_optimal_block_size_bw(boundary_img_width, 4,5))); // allowed warp pixels sizes = 2^[4,5] ie, [16,32]

	// we will split image into patches of size [IMG_HEIGHT x IMG_WIDTH] so use that as image size, however,
	// we need to increase the number of images that will be process as each patch is now considered as one image
	// there is no need to recombine the output since we just sum over all patches to get gradients

	int new_img_parts_width = (int)ceil((float)img_width / patch_size_w);
	int new_img_parts_height = (int)ceil((float)img_height / patch_size_h);

	num_images = I* new_img_parts_width * new_img_parts_height;

	single_subfeature = (S % 2 == 0 ? false : true);


	// last_k_optional==false and NUM_K==3 or
	// last_k_optional==true and NUM_K==4 and img_size_w >= 32
	//  - NUM_K = 3, BATCH_K_SIZE = 1, _WARP_PIXELS_X = 32
	//
	// last_k_optional==true and NUM_K==4 and img_size_w == 16
	//  - NUM_K = 4, BATCH_K_SIZE = 2, _WARP_PIXELS_X = 16
    //
    // last_k_optional==true and NUM_K==4 and img_size_w == 8
    //  - NUM_K = 4, BATCH_K_SIZE = 4, _WARP_PIXELS_X = 8

	use_smaller_warp_and_group_k = false;

	OUT_K = K;

	if (K == 4) {
		if (last_k_optional == false) {
			// we can automatically use 16 pixel warps and group K by 2 (or 8 pixel warps and group K by 4)
			use_smaller_warp_and_group_k = true;
		} else {
			// if last K is optional (i.e. we do not care for sigma) then decide to use 16 or 8 pixel warps only if our patch size is smaller then WARP size (i.e. WARP size=32)
			use_smaller_warp_and_group_k = (warp_pixel_size_x < 32 ? true : false);
			// in case that we will be not be grouping then then we can skip last K since it appears to be optional
			// (NOTE: input K must remain the same to correctly load the data !!
			//        for output data we do not need to change anything since output has K as last dimension and we just ignore
			//        last K anyway)

			OUT_K = use_smaller_warp_and_group_k ? K : K - 1;
		}
	} else if (K == 3) {
		// if we have only K==3 then we cannot group K and instead use bigger warp size irrespectively of the patch size
		use_smaller_warp_and_group_k = false;
	} else {
		// not allowed
		throw DAUException(string_format("Unsupported K: %d. Supported only K=3 or K=4 at the moment\n", K));
	}

	// if we do not use smaller warp then ensure patch_size_w is at least 32px
	if (use_smaller_warp_and_group_k == false)
		patch_size_w = std::max(32, patch_size_w);
}

template <typename Dtype>
int DAUConvBackward<Dtype>::select_optimal_block_size_bw(int img_size, int min_power, int max_power) {
	float best_unutilized_percent = 1.0f;
	int best_block_size = 0;
	for (int i = min_power; i <= max_power; ++i) {
		int block_size = pow(2,i);

		float utilization_factor = (img_size / (float)block_size);
		float unutilized_percent = (ceil(utilization_factor) - utilization_factor);
		if (unutilized_percent <= best_unutilized_percent) {
			best_unutilized_percent = unutilized_percent;
			best_block_size = block_size;
		}
	}
	return best_block_size;
}


template <typename Dtype>
void DAUConvBackward<Dtype>::CUDAParams::set_params_for_allocation_call(size_t* alloc_img, size_t* alloc_err, size_t* alloc_w, size_t* alloc_off) {
	this->alloc_img = alloc_img;
	this->alloc_w = alloc_w;
	this->alloc_err = alloc_err;
	this->alloc_off = alloc_off;
}

template <typename Dtype>
void DAUConvBackward<Dtype>::CUDAParams::set_params_for_kernel_call(const Dtype* filtered_images, const Dtype* error_images,
								const Dtype* filter_offsets_float_x, const Dtype* filter_offsets_float_y,
								const Dtype* filter_weights, const int kernel_w, const int kernel_h, const Dtype actual_max_offset,
								Dtype* output,
								Dtype* prepared_filtered_images,
								Dtype* prepared_error_images,
								Dtype* prepared_filter_weights,
								int* prepared_filter_offsets,
								const bool ignore_edge_gradients,
								cudaStream_t streamId) {
	this->filtered_images = filtered_images;
	this->error_images = error_images;
	this->filter_offsets_float_x = filter_offsets_float_x;
	this->filter_offsets_float_y = filter_offsets_float_y;
	this->filter_weights = filter_weights;
	this->kernel_w = kernel_w;
	this->kernel_h = kernel_h;
    this->actual_max_offset = actual_max_offset;
	this->output = output;
	this->prepared_filtered_images = prepared_filtered_images;
	this->prepared_error_images = prepared_error_images;
	this->prepared_filter_weights = prepared_filter_weights;
	this->prepared_filter_offsets = prepared_filter_offsets;
	this->ignore_edge_gradients = ignore_edge_gradients;
	this->streamId = streamId;
}

template <typename Dtype>
void DAUConvBackward<Dtype>::get_allocation_sizes(const int kernel_width, const int kernel_height, const bool offsets_already_centered,
                                                                    size_t* prepared_filtered_images_size,
                                                                    size_t* prepared_error_images_size,
                                                                    size_t* prepared_filter_weights_size,
                                                                    size_t* prepared_filter_offsets_size) {
    float actual_max_offset = (MAX(kernel_width, kernel_height)-1)/2;
    int actual_OUT_K = OUT_K;
    if (actual_max_offset <= 8) {
        // do nothing (i.e. actual_OUT_K = OUT_K)
    }else  if (actual_max_offset <= 16) {
        actual_OUT_K = 3;
    } else if (actual_max_offset <= 32) {
        actual_OUT_K = 1;
    } else{
		throw DAUException(string_format("ERROR: actual offsets larger then what CUDA memory allows (setup max_kernel_size and unit_border_bound correctly to avoid this)!!"));
    }

	CUDAParams params(img_width_in, img_height_in, img_width, img_height, I, S, F, G, actual_OUT_K, IN_K, offsets_already_centered);

	params.set_params_for_allocation_call(prepared_filtered_images_size, prepared_error_images_size, prepared_filter_weights_size, prepared_filter_offsets_size);
	params.set_params_for_kernel_call(NULL, NULL, NULL, NULL, NULL, kernel_width, kernel_height, actual_max_offset, NULL,
									  NULL, NULL, NULL, NULL, false, 0);

	call_cuda_kernel(params);
}

template <typename Dtype>
void DAUConvBackward<Dtype>::backward_pass(const Dtype* filtered_images, const Dtype* error_images,
													  const Dtype* filter_offsets_float_x, const Dtype* filter_offsets_float_y,
													  const Dtype* filter_weights,
													  const int kernel_width, const int kernel_height, const Dtype actual_max_offset,
										   			  const bool offsets_already_centered,
													  Dtype* output,
													  Dtype* prepared_filtered_images,
													  Dtype* prepared_error_images,
													  Dtype* prepared_filter_weights,
													  int* prepared_filter_offsets,
													  const bool ignore_edge_gradients,
													  cudaStream_t streamId) {

	// Optimize the max possible offset that is needed since larger offsets require loading more memory and is less efficent

	// For offsets larger then 8 px then we need to :
	//  * for offsets <= 16px: use OUT_K = 3
	//  * for offsets <= 32px: use OUT_K = 1 and run several times for each K
	//
	// WARNING: this must be synced with RUN_KERNEL_R2 in dau_conv_backward_core.hpp

	int actual_OUT_K = OUT_K;
    int max_offset = 32;

    if (actual_max_offset <= 4)
		max_offset = 4;
	else if (actual_max_offset <= 8)
		max_offset = 8;
	else if (actual_max_offset <= 16) {
		max_offset = 16;
		actual_OUT_K = 3;
	} else if (actual_max_offset <= 32) {
		max_offset = 32;
		actual_OUT_K = 1;
	} else{
        throw DAUConvNet::DAUException("ERROR: actual offsets larger then what CUDA memory allows (setup max_kernel_size and unit_border_bound correctly to avoid this)!!");
    }

	// To ensure we have enough memory we require max_offset not to exceed kernel_width or kernel_height
	// since kernel_width and kernel_height are used in get_allocation_sizes()
	DAU_CHECK(kernel_width >= max_offset*2+1, "Maximum offset values exceeds boundries as defined by kernel_width.");
    DAU_CHECK(kernel_height >= max_offset*2+1, "Maximum offset values exceeds boundries as defined by kernel_height.");

    for (int k = 0; k < OUT_K; k+=actual_OUT_K) {

		// WARNING: this assumes we got K=4 in constructor
		if (k == 3 && last_k_optional)
			continue;

		CUDAParams params(img_width_in, img_height_in, img_width, img_height, I, S, F, G, actual_OUT_K, IN_K, offsets_already_centered);

		params.set_params_for_allocation_call(NULL, NULL, NULL, NULL);
		params.set_params_for_kernel_call(filtered_images + k*img_width*img_height, error_images, filter_offsets_float_x, filter_offsets_float_y, filter_weights, kernel_width, kernel_height, max_offset,
										  output + k * (this->S *this->F * this->G), prepared_filtered_images, prepared_error_images, prepared_filter_weights, prepared_filter_offsets, ignore_edge_gradients, streamId);


		// NOTE: we assume that kernel_width/2 >
		call_cuda_kernel(params);
	}
}
template <>
void DAUConvBackward<float>::call_cuda_kernel(CUDAParams& params) {

	int max_kernel_size = 2*ceil(params.actual_max_offset) + 1;
	//int max_kernel_size = MAX(params.kernel_w, params.kernel_h);

	// calls either DAUConvBackwardCUDA->run_kernel() or DAUConvBackwardCUDA->get_allocation_sizes()
	// if prepared_filtered_images_size, prepared_error_images_size, prepared_filter_weights_size OR prepared_filter_offsets_size are not NULL

	if (patch_size_h == 1 && patch_size_w == 1) {
		DAUConv_backward_multi_subfeatures_patch_1x1(patch_size_w, patch_size_h, max_kernel_size,
													 use_smaller_warp_and_group_k, num_images, use_interpolation,
													 single_subfeature, params);
	} else if (patch_size_h >= 64) {
		if (patch_size_w >= 64) {
			DAUConv_backward_multi_subfeatures_patch_64x64(patch_size_w, patch_size_h, max_kernel_size,
														   use_smaller_warp_and_group_k, num_images, use_interpolation,
														   single_subfeature, params);
		} else if (patch_size_w >= 32) {
			DAUConv_backward_multi_subfeatures_patch_32x64(patch_size_w, patch_size_h, max_kernel_size,
														   use_smaller_warp_and_group_k, num_images, use_interpolation,
														   single_subfeature, params);
		} else if (patch_size_w >= 16) {
			DAUConv_backward_multi_subfeatures_patch_16x64(patch_size_w, patch_size_h, max_kernel_size,
														   use_smaller_warp_and_group_k, num_images, use_interpolation,
														   single_subfeature, params);
		} else {
			DAUConv_backward_multi_subfeatures_patch_8x64(patch_size_w, patch_size_h, max_kernel_size,
														  use_smaller_warp_and_group_k, num_images, use_interpolation,
														  single_subfeature, params);
		}
	} else if (patch_size_h >= 32) {
		if (patch_size_w >= 64) {
			DAUConv_backward_multi_subfeatures_patch_64x32(patch_size_w, patch_size_h, max_kernel_size,
														   use_smaller_warp_and_group_k, num_images, use_interpolation,
														   single_subfeature, params);
		} else if (patch_size_w >= 32) {
			DAUConv_backward_multi_subfeatures_patch_32x32(patch_size_w, patch_size_h, max_kernel_size,
														   use_smaller_warp_and_group_k, num_images, use_interpolation,
														   single_subfeature, params);
		} else if (patch_size_w >= 16) {
			DAUConv_backward_multi_subfeatures_patch_16x32(patch_size_w, patch_size_h, max_kernel_size,
														   use_smaller_warp_and_group_k, num_images, use_interpolation,
														   single_subfeature, params);
		} else {
			DAUConv_backward_multi_subfeatures_patch_8x32(patch_size_w, patch_size_h, max_kernel_size,
														  use_smaller_warp_and_group_k, num_images, use_interpolation,
														  single_subfeature, params);
		}
	} else if (patch_size_h >= 16) {
		if (patch_size_w >= 64) {
			DAUConv_backward_multi_subfeatures_patch_64x16(patch_size_w, patch_size_h, max_kernel_size,
														   use_smaller_warp_and_group_k, num_images, use_interpolation,
														   single_subfeature, params);
		} else if (patch_size_w >= 32) {
			DAUConv_backward_multi_subfeatures_patch_32x16(patch_size_w, patch_size_h, max_kernel_size,
														   use_smaller_warp_and_group_k, num_images, use_interpolation,
														   single_subfeature, params);
		} else if (patch_size_w >= 16) {
			DAUConv_backward_multi_subfeatures_patch_16x16(patch_size_w, patch_size_h, max_kernel_size,
														   use_smaller_warp_and_group_k, num_images, use_interpolation,
														   single_subfeature, params);
		} else {
			DAUConv_backward_multi_subfeatures_patch_8x16(patch_size_w, patch_size_h, max_kernel_size,
														  use_smaller_warp_and_group_k, num_images, use_interpolation,
														  single_subfeature, params);
		}
	} else {
		if (patch_size_w >= 64) {
			DAUConv_backward_multi_subfeatures_patch_64x8(patch_size_w, patch_size_h, max_kernel_size,
														  use_smaller_warp_and_group_k, num_images, use_interpolation,
														  single_subfeature, params);
		} else if (patch_size_w >= 32) {
			DAUConv_backward_multi_subfeatures_patch_32x8(patch_size_w, patch_size_h, max_kernel_size,
														  use_smaller_warp_and_group_k, num_images, use_interpolation,
														  single_subfeature, params);
		} else if (patch_size_w >= 16) {
			DAUConv_backward_multi_subfeatures_patch_16x8(patch_size_w, patch_size_h, max_kernel_size,
														  use_smaller_warp_and_group_k, num_images, use_interpolation,
														  single_subfeature, params);
		} else {
			DAUConv_backward_multi_subfeatures_patch_8x8(patch_size_w, patch_size_h, max_kernel_size,
														 use_smaller_warp_and_group_k, num_images, use_interpolation,
														 single_subfeature, params);
		}
	}

}

template <>
void DAUConvBackward<double>::call_cuda_kernel(CUDAParams& params) {

    throw DAUConvNet::DAUException("Not implemented for double");
}

template DAUConvBackward<float>::DAUConvBackward(const int img_width_in, const int img_height_in, const int img_width, const int img_height, const int I, const int S, const int F, const int G, const int K, const bool last_k_optional, const bool use_interpolation);
template DAUConvBackward<double>::DAUConvBackward(const int img_width_in, const int img_height_in, const int img_width, const int img_height, const int I, const int S, const int F, const int G, const int K, const bool last_k_optional, const bool use_interpolation);

template void DAUConvBackward<float>::get_allocation_sizes(const int kernel_width, const int kernel_height, const bool offsets_already_centered, size_t* prepared_filtered_images_size, size_t* prepared_error_images_size, size_t* prepared_filter_weights_size, size_t* prepared_filter_offsets_size);
template void DAUConvBackward<float>::backward_pass(const float* filtered_images, const float* error_images, const float* filter_offsets_float_x, const float* filter_offsets_float_y, const float* filter_weights, const int kernel_width, const int kernel_height, const float actual_max_offset, const bool offsets_already_centered, float* output, float* prepared_filtered_images, float* prepared_error_images, float* prepared_filter_weights, int* prepared_filter_offsets, const bool ignore_edge_gradients, cudaStream_t streamId);

template void DAUConvBackward<double>::get_allocation_sizes(const int kernel_width, const int kernel_height, const bool offsets_already_centered, size_t* prepared_filtered_images_size, size_t* prepared_error_images_size, size_t* prepared_filter_weights_size, size_t* prepared_filter_offsets_size);
template void DAUConvBackward<double>::backward_pass(const double* filtered_images, const double* error_images, const double* filter_offsets_float_x, const double* filter_offsets_float_y, const double* filter_weights, const int kernel_width, const int kernel_height, const double actual_max_offset, const bool offsets_already_centered, double* output, double* prepared_filtered_images, double* prepared_error_images, double* prepared_filter_weights, int* prepared_filter_offsets, const bool ignore_edge_gradients, cudaStream_t streamId);
}  // namespace caffe
