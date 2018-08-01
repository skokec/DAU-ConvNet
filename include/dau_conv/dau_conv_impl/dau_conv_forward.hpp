#ifndef DAU_CONV_UTIL_DAU_FORWARD_H_
#define DAU_CONV_UTIL_DAU_FORWARD_H_

#include <stdio.h>

#include "dau_conv/util/common.hpp"

namespace DAUConvNet {

#ifndef CPU_ONLY  // GPU

template <typename Dtype>
class DAUConvForward {
	// fixed params during construction
	const int img_width_in, img_height_in;
	const int img_width, img_height;
	const int I, S, F, G;

	// this parameters are used as template params for DAUConvBackwardCUDA
	int patch_size_w, patch_size_h, max_offset, num_images, warp_pixel_size_x, warp_pixel_size_y;
	bool single_feature, single_subfeature, use_interpolation;

public:
	enum PARAM_FORMAT { SGF, FGS}; // default should be SGF

	DAUConvForward(const int img_width_in, const int img_height_in, const int img_width, const int img_height, const int I, const int S, const int F, const int G, const bool use_interpolation);

	void get_allocation_sizes(const int kernel_width, const int kernel_height, const bool offsets_already_centered,
							  size_t* prepared_filtered_images_size,
							  size_t* prepared_filter_weights_size,
							  size_t* prepared_filter_offsets_size);

	void forward_pass(const Dtype* filtered_images,
					  const Dtype* filter_offsets_float_x, const Dtype* filter_offsets_float_y,
					  const Dtype* filter_weights, const PARAM_FORMAT param_format,
					  const int kernel_width, const int kernel_height, const Dtype actual_max_offset,
                      const bool offsets_already_centered, Dtype* output,
					  Dtype* prepared_filtered_images,
					  Dtype* prepared_filter_weights,
					  int* prepared_filter_offsets,
					  Dtype* prepared_filter_offsets_and_weights, cudaStream_t streamId = NULL);

	class CUDAParams {
	public:
		// fixed params during construction
		const int img_width_in, img_height_in;
		const int img_width, img_height;
		const int I, S, F, G;

		// parameters for setup before call

		// params for get_allocation_sizes call
		size_t *alloc_img, *alloc_w, *alloc_off;

		// params for run_kernel call
		Dtype const *filtered_images, *filter_offsets_float_x, *filter_offsets_float_y, *filter_weights;
		Dtype *output, *prepared_filtered_images, *prepared_filter_weights, *prepared_filter_offsets_and_weights;
		int *prepared_filter_offsets;
		int kernel_w, kernel_h;
		PARAM_FORMAT param_format;
		bool offsets_already_centered;
		cudaStream_t streamId;

		float actual_max_offset;

	public:
		CUDAParams(const int img_width_in, const int img_height_in, const int img_width, const int img_height, const int I, const int S, const int F, const int G, const bool offsets_already_centered) :
				img_width_in(img_width_in), img_height_in(img_height_in), img_width(img_width), img_height(img_height), I(I), S(S), F(F), G(G), offsets_already_centered(offsets_already_centered) {
		}

		void set_params_for_allocation_call(size_t *alloc_img, size_t *alloc_w, size_t *alloc_off);

		void set_params_for_kernel_call(const Dtype *filtered_images,
										const Dtype *filter_offsets_float_x, const Dtype *filter_offsets_float_y,
										const Dtype *filter_weights,
										const PARAM_FORMAT param_format, const int kernel_w, const int kernel_h,
										const Dtype actual_max_offset, Dtype *output,
										Dtype *prepared_filtered_images,
										Dtype *prepared_filter_weights,
										int *prepared_filter_offsets,
										Dtype *prepared_filter_offsets_and_weights,
										cudaStream_t streamId);

	};

private:
	void call_cuda_kernel(CUDAParams& params);

};
// we make explicit functions for different combinations of [OFFSET, USE_SINGLE_FEATURE, USE_SINGLE_SUBFEATURE]
// each function is implemented in separate .cu file to allow for parallel compile
// (there are 288 combination all-together so this way we can reduce compute time by a factor of 8 if enough CPU cores)
void DAUConv_forward_float_off_4_single_feat_0_single_subfeat_0(int IMG_PATCH_SIZE_W, int IMG_PATCH_SIZE_H, int MAX_OFFSET, int WARP_PIXELS_X, int WARP_PIXELS_Y, int BLOCK_IMAGES, int USE_INTERPOLATION, DAUConvForward<float>::CUDAParams &PARAMS);
void DAUConv_forward_float_off_4_single_feat_0_single_subfeat_1(int IMG_PATCH_SIZE_W, int IMG_PATCH_SIZE_H, int MAX_OFFSET, int WARP_PIXELS_X, int WARP_PIXELS_Y, int BLOCK_IMAGES, int USE_INTERPOLATION, DAUConvForward<float>::CUDAParams &PARAMS);
void DAUConv_forward_float_off_4_single_feat_1_single_subfeat_0(int IMG_PATCH_SIZE_W, int IMG_PATCH_SIZE_H, int MAX_OFFSET, int WARP_PIXELS_X, int WARP_PIXELS_Y, int BLOCK_IMAGES, int USE_INTERPOLATION, DAUConvForward<float>::CUDAParams &PARAMS);
void DAUConv_forward_float_off_4_single_feat_1_single_subfeat_1(int IMG_PATCH_SIZE_W, int IMG_PATCH_SIZE_H, int MAX_OFFSET, int WARP_PIXELS_X, int WARP_PIXELS_Y, int BLOCK_IMAGES, int USE_INTERPOLATION, DAUConvForward<float>::CUDAParams &PARAMS);

void DAUConv_forward_float_off_8_single_feat_0_single_subfeat_0(int IMG_PATCH_SIZE_W, int IMG_PATCH_SIZE_H, int MAX_OFFSET, int WARP_PIXELS_X, int WARP_PIXELS_Y, int BLOCK_IMAGES, int USE_INTERPOLATION, DAUConvForward<float>::CUDAParams &PARAMS);
void DAUConv_forward_float_off_8_single_feat_0_single_subfeat_1(int IMG_PATCH_SIZE_W, int IMG_PATCH_SIZE_H, int MAX_OFFSET, int WARP_PIXELS_X, int WARP_PIXELS_Y, int BLOCK_IMAGES, int USE_INTERPOLATION, DAUConvForward<float>::CUDAParams &PARAMS);
void DAUConv_forward_float_off_8_single_feat_1_single_subfeat_0(int IMG_PATCH_SIZE_W, int IMG_PATCH_SIZE_H, int MAX_OFFSET, int WARP_PIXELS_X, int WARP_PIXELS_Y, int BLOCK_IMAGES, int USE_INTERPOLATION, DAUConvForward<float>::CUDAParams &PARAMS);
void DAUConv_forward_float_off_8_single_feat_1_single_subfeat_1(int IMG_PATCH_SIZE_W, int IMG_PATCH_SIZE_H, int MAX_OFFSET, int WARP_PIXELS_X, int WARP_PIXELS_Y, int BLOCK_IMAGES, int USE_INTERPOLATION, DAUConvForward<float>::CUDAParams &PARAMS);

void DAUConv_forward_float_off_16_single_feat_0_single_subfeat_1(int IMG_PATCH_SIZE_W, int IMG_PATCH_SIZE_H, int MAX_OFFSET, int WARP_PIXELS_X, int WARP_PIXELS_Y, int BLOCK_IMAGES, int USE_INTERPOLATION, DAUConvForward<float>::CUDAParams &PARAMS);
void DAUConv_forward_float_off_16_single_feat_1_single_subfeat_1(int IMG_PATCH_SIZE_W, int IMG_PATCH_SIZE_H, int MAX_OFFSET, int WARP_PIXELS_X, int WARP_PIXELS_Y, int BLOCK_IMAGES, int USE_INTERPOLATION, DAUConvForward<float>::CUDAParams &PARAMS);
void DAUConv_forward_float_off_32_single_feat_1_single_subfeat_1(int IMG_PATCH_SIZE_W, int IMG_PATCH_SIZE_H, int MAX_OFFSET, int WARP_PIXELS_X, int WARP_PIXELS_Y, int BLOCK_IMAGES, int USE_INTERPOLATION, DAUConvForward<float>::CUDAParams &PARAMS);



#endif  // !CPU_ONLY

}  // namespace DAUConvNet

#endif  // DAU_CONV_UTIL_DAU_FORWARD_H_
