#include <cmath>
#include <algorithm>

#include "dau_conv/dau_conv_impl/dau_conv_forward.hpp"

namespace DAUConvNet {

#define MAX(x,y) (x > y ? x : y)

int select_optimal_block_size(int img_size, int min_power, int max_power) {
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
DAUConvForward<Dtype>::DAUConvForward(const int img_width_in, const int img_height_in, const int img_width, const int img_height, const int I, const int S, const int F, const int G, const bool use_interpolation)  :
		img_width_in(img_width_in), img_height_in(img_height_in), img_width(img_width), img_height(img_height), I(I), S(S), F(F), G(G), use_interpolation(use_interpolation) {

    // calls either DAUConvForwardCUDA->run_kernel() or DAUConvForwardCUDA->get_allocation_sizes()
    // if prepared_filtered_images_size, prepared_filter_weights_size OR prepared_filter_offsets_size are not NULL

    // decide which size of patch to use to minimize wasted memory/processing
	if (img_width == 1 && img_height == 1) {
		patch_size_w = 1;
		patch_size_h = 1;
	} else {
		patch_size_w = img_width <= 8 ? 8 :
					   	(img_width <= 16 ? 16 : select_optimal_block_size(img_width, 5, 6)); // allowed patch sizes = 2^[5,6] i.e, [32,64]
		patch_size_h = img_height <= 8 ? 8 :
						   (img_height <= 16 ? 16 : select_optimal_block_size(img_height, 5, 6)); // allowed patch sizes = 2^[5,6] i.e, [32,64]
	}

    // decide wheather to use:
    //  - 32 pixels per warp
    // 		- if 32x8 pixels and 1 images per block (full utilization)
    //  - 16 pixels per warp
    // 		- if 16x8 pixels and 2 images per block (full utilization)
    // 		- if 16x8 pixels and 1 images per block (half utilization)
	//  - 8 pixels per warp
	// 		- if 8x8 pixels and 4 images per block (full utilization)
	// 		- if 8x8 pixels and 2 images per block (half utilization)
	// 		- if 8x8 pixels and 1 images per block (1/4 utilization)
	//	- 1 pixel per warp
	//		- if 1x1 pixels and 16 images per block (half utilization) (32 images uses too much shared memory so we cannot have full utilization)

    int boundry_img_width = img_width - floor(img_width/patch_size_w) * patch_size_w;


	// use warp size 1x1 if patch size only 1x1 otherwise use [16,32]x8 (if patch_size_w==8 then use 8x8 but do not prefer it)
	warp_pixel_size_x = patch_size_w == 1 ? 1 :
							(patch_size_w <= 8 ? 8 : std::min(patch_size_w, select_optimal_block_size(boundry_img_width, 4,5))); // allowed warp pixels sizes = 2^[3,4,5] i.e. [8,16,32]
	warp_pixel_size_y = patch_size_h == 1 ? 1 : 8;

    int new_img_parts_width = (int)ceil((float)img_width / patch_size_w);
    int new_img_parts_height = (int)ceil((float)img_height / patch_size_h);

    num_images = I * new_img_parts_width * new_img_parts_height;

    // we compute multiple features by one thread but that depends on interpolation
    int batch_features = 8 * (use_interpolation ? 2 : 4);

    single_feature = F % batch_features == 0 ? false : true;
    single_subfeature = S % 2 == 0 ? false : true;
}

template <typename Dtype>
void DAUConvForward<Dtype>::CUDAParams::set_params_for_allocation_call(size_t *alloc_img, size_t *alloc_w, size_t *alloc_off) {
    this->alloc_img = alloc_img;
    this->alloc_w = alloc_w;
    this->alloc_off = alloc_off;
}

template <typename Dtype>
void DAUConvForward<Dtype>::CUDAParams::set_params_for_kernel_call(const Dtype *filtered_images,
                                                                     const Dtype *filter_offsets_float_x, const Dtype *filter_offsets_float_y,
                                                                     const Dtype *filter_weights, const PARAM_FORMAT param_format, const int kernel_w, const int kernel_h,
                                                                     const Dtype actual_max_offset, Dtype *output,
                                                                     Dtype *prepared_filtered_images,
                                                                     Dtype *prepared_filter_weights,
                                                                     int *prepared_filter_offsets,
                                                                     Dtype *prepared_filter_offsets_and_weights,
                                                                     cudaStream_t streamId) {
    this->filtered_images = filtered_images;
    this->filter_offsets_float_x = filter_offsets_float_x;
    this->filter_offsets_float_y = filter_offsets_float_y;
    this->filter_weights = filter_weights;
    this->kernel_w = kernel_w;
    this->kernel_h = kernel_h;
    this->actual_max_offset = actual_max_offset;
    this->param_format = param_format;
    this->output = output;
    this->prepared_filtered_images = prepared_filtered_images;
    this->prepared_filter_weights = prepared_filter_weights;
    this->prepared_filter_offsets = prepared_filter_offsets;
    this->prepared_filter_offsets_and_weights = prepared_filter_offsets_and_weights;
    this->streamId = streamId;
}
template <typename Dtype>
void DAUConvForward<Dtype>::get_allocation_sizes(const int kernel_width, const int kernel_height, const bool offsets_already_centered,
                                                   size_t* prepared_filtered_images_size,
                                                   size_t* prepared_filter_weights_size,
                                                   size_t* prepared_filter_offsets_size) {

    CUDAParams params(img_width_in, img_height_in, img_width, img_height, I, S, F, G, offsets_already_centered);

    params.set_params_for_allocation_call(prepared_filtered_images_size, prepared_filter_weights_size, prepared_filter_offsets_size);

    params.set_params_for_kernel_call(NULL, NULL, NULL, NULL, PARAM_FORMAT::SGF, kernel_width, kernel_height, (MAX(kernel_width, kernel_height)-1)/2, NULL,
                                      NULL, NULL, NULL, NULL, 0);

    call_cuda_kernel(params);
}


template <typename Dtype>
void DAUConvForward<Dtype>::forward_pass(const Dtype* filtered_images,
                                           const Dtype* filter_offsets_float_x, const Dtype* filter_offsets_float_y,
                                           const Dtype* filter_weights, const PARAM_FORMAT param_format,
										   const int kernel_width, const int kernel_height, const Dtype actual_max_offset,
										   const bool offsets_already_centered,
                                           Dtype* output,
                                           Dtype* prepared_filtered_images,
                                           Dtype* prepared_filter_weights,
                                           int* prepared_filter_offsets,
                                           Dtype* prepared_filter_offsets_and_weights, cudaStream_t streamId) {
	// Optimize the max possible offset that is needed since larger offsets require loading more memory and is less efficent

	// For offsets larger then 8 px then we need to :
	//  * for offsets <= 16px: use OUT_K = 3
	//  * for offsets <= 32px: use OUT_K = 1 and run several times for each K
	//
	// WARNING: this must be synced with RUN_KERNEL_R2 in dau_conv_backward_core.hpp

	float max_offset = 32;

	if (actual_max_offset <= 4)
		max_offset = 4;
	else if (actual_max_offset <= 8)
		max_offset = 8;
	else if (actual_max_offset <= 16) {
		max_offset = 16;
	} else if (actual_max_offset <= 32) {
		max_offset = 32;
	} else {
        std::cout << "ERROR: actual offsets larger then what CUDA memory allows (setup max_kernel_size and unit_border_bound correctly to avoid this)!!" << std::endl;
        throw std::exception();
    }

	// To ensure we have enough memory we require max_offset not to exceed kernel_width or kernel_height
	// since kernel_width and kernel_height are used in get_allocation_sizes()
	CHECK(kernel_width >= max_offset*2+1, "Maximum offset values exceeds boundries as defined by kernel_width.");
	CHECK(kernel_height >= max_offset*2+1, "Maximum offset values exceeds boundries as defined by kernel_height.");

	CUDAParams params(img_width_in, img_height_in, img_width, img_height, I, S, F, G, offsets_already_centered);

	params.set_params_for_allocation_call(NULL, NULL, NULL);
	params.set_params_for_kernel_call(filtered_images, filter_offsets_float_x, filter_offsets_float_y, filter_weights, param_format, kernel_width, kernel_height, max_offset, output,
									  prepared_filtered_images, prepared_filter_weights, prepared_filter_offsets, prepared_filter_offsets_and_weights,
									  streamId);

    call_cuda_kernel(params);
}
template <>
void DAUConvForward<float>::call_cuda_kernel(CUDAParams& params) {

	int max_offset = ceil(params.actual_max_offset);
    //int max_offset = MAX(params.kernel_w, params.kernel_h)/2;

	if (max_offset <= 4) {
		if (single_feature == false && single_subfeature == false) {
			// version where single_feature is false and single_subfeature false
            DAUConv_forward_float_off_4_single_feat_0_single_subfeat_0(patch_size_w, patch_size_h, max_offset,
                                                                       warp_pixel_size_x, warp_pixel_size_y, num_images,
                                                                       use_interpolation, params);

		} else if (single_feature == false && single_subfeature == true) {
			// version where single_feature is false and single_subfeature true
			DAUConv_forward_float_off_4_single_feat_0_single_subfeat_1(patch_size_w, patch_size_h, max_offset,
																	   warp_pixel_size_x, warp_pixel_size_y, num_images,
																	   use_interpolation, params);

		} else if (single_feature == true && single_subfeature == false) {
			// version where single_feature is true and single_subfeature false
			DAUConv_forward_float_off_4_single_feat_1_single_subfeat_0(patch_size_w, patch_size_h, max_offset,
																	   warp_pixel_size_x, warp_pixel_size_y, num_images,
																	   use_interpolation, params);

		} else {
			// version where single_feature is true and single_subfeature true
			DAUConv_forward_float_off_4_single_feat_1_single_subfeat_1(patch_size_w, patch_size_h, max_offset,
																	   warp_pixel_size_x, warp_pixel_size_y, num_images,
																	   use_interpolation, params);
		}
	} else if (max_offset <= 8) {
		if (single_feature == false && single_subfeature == false) {
			// version where single_feature is false and single_subfeature false
			DAUConv_forward_float_off_8_single_feat_0_single_subfeat_0(patch_size_w, patch_size_h, max_offset,
																	   warp_pixel_size_x, warp_pixel_size_y, num_images,
																	   use_interpolation, params);

		} else if (single_feature == false && single_subfeature == true) {
			// version where single_feature is false and single_subfeature true
			DAUConv_forward_float_off_8_single_feat_0_single_subfeat_1(patch_size_w, patch_size_h, max_offset,
																	   warp_pixel_size_x, warp_pixel_size_y, num_images,
																	   use_interpolation, params);

		} else if (single_feature == true && single_subfeature == false) {
			// version where single_feature is true and single_subfeature false
			DAUConv_forward_float_off_8_single_feat_1_single_subfeat_0(patch_size_w, patch_size_h, max_offset,
																	   warp_pixel_size_x, warp_pixel_size_y, num_images,
																	   use_interpolation, params);

		} else {
			// version where single_feature is true and single_subfeature true
			DAUConv_forward_float_off_8_single_feat_1_single_subfeat_1(patch_size_w, patch_size_h, max_offset,
																	   warp_pixel_size_x, warp_pixel_size_y, num_images,
																	   use_interpolation, params);
		}
	} else if (max_offset <= 16) {

        if (single_feature == false)
            DAUConv_forward_float_off_16_single_feat_0_single_subfeat_1(patch_size_w, patch_size_h, max_offset,
                                                                        warp_pixel_size_x, warp_pixel_size_y, num_images,
                                                                        use_interpolation, params);
        else
            DAUConv_forward_float_off_16_single_feat_1_single_subfeat_1(patch_size_w, patch_size_h, max_offset,
                                                                        warp_pixel_size_x, warp_pixel_size_y, num_images,
                                                                        use_interpolation, params);

	} else if (max_offset <= 32) {
        DAUConv_forward_float_off_32_single_feat_1_single_subfeat_1(patch_size_w, patch_size_h, max_offset,
                                                                    warp_pixel_size_x, warp_pixel_size_y, num_images,
                                                                    use_interpolation, params);


    } else {
		printf("Unsupported filter size: %d. Supported only max up to 9x9 and 17x17 at the moment\n", max_offset);
        throw std::exception();
	}



	// CALL RUN_KERNEL_R4 macro that will call run_kernel() function on supplied class where first 4 parameters are replaced with compile-time known variables
	// replacing variables with compile-time known values allows CUDA compiler to generate kernels in advanced with pre-defined sizes
/*
	RUN_KERNEL_R7(DAUConvForwardCUDA, patch_size_w, patch_size_h, max_offset, warp_pixel_size_x, num_images, use_interpolation, single_feature, single_subfeature,
				  img_width, img_height, I, S, F, G,
				  filtered_images, filter_offsets_float_x, filter_offsets_float_y, filter_weights, kernel_width, kernel_height, PARAM_FORMAT, output,
				  prepared_filtered_images, prepared_filter_weights, prepared_filter_offsets, prepared_filter_offsets_and_weights,
				  streamId);
*/
}

template <>
void DAUConvForward<double>::call_cuda_kernel(CUDAParams& params) {
    throw std::exception();
}

template DAUConvForward<float>::DAUConvForward(const int img_width_in, const int img_height_in, const int img_width, const int img_height, const int I, const int S, const int F, const int G, const bool use_interpolation);
template DAUConvForward<double>::DAUConvForward(const int img_width_in, const int img_height_in, const int img_width, const int img_height, const int I, const int S, const int F, const int G, const bool use_interpolation);

template void DAUConvForward<float>::get_allocation_sizes(const int kernel_width, const int kernel_height, const bool offsets_already_centered, size_t* prepared_filtered_images_size, size_t* prepared_filter_weights_size, size_t* prepared_filter_offsets_size);
template void DAUConvForward<double>::get_allocation_sizes(const int kernel_width, const int kernel_height, const bool offsets_already_centered, size_t* prepared_filtered_images_size, size_t* prepared_filter_weights_size, size_t* prepared_filter_offsets_size);

template void DAUConvForward<float>::forward_pass(const float* filtered_images, const float* filter_offsets_float_x, const float* filter_offsets_float_y, const float* filter_weights, const PARAM_FORMAT param_format, const int kernel_width, const int kernel_height, const float actual_max_offset, const bool offsets_already_centered, float* output, float* prepared_filtered_images, float* prepared_filter_weights, int* prepared_filter_offsets, float* prepared_filter_offsets_and_weights, cudaStream_t streamId);
template void DAUConvForward<double>::forward_pass(const double* filtered_images, const double* filter_offsets_float_x, const double* filter_offsets_float_y, const double* filter_weights, const PARAM_FORMAT param_format, const int kernel_width, const int kernel_height, const double actual_max_offset, const bool offsets_already_centered, double* output, double* prepared_filtered_images, double* prepared_filter_weights, int* prepared_filter_offsets, double* prepared_filter_offsets_and_weights, cudaStream_t streamId);

}  // namespace caffe


