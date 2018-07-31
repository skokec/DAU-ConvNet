#ifndef DAU_CONV_UTIL_DAU_BACKWARD_H_
#define DAU_CONV_UTIL_DAU_BACKWARD_H_


#include <stdio.h>

#include "dau_conv/util/common.hpp"

namespace DAUConvNet {
#ifndef CPU_ONLY  // GPU

#define MAX(x,y) (x > y ? x : y)

template <typename Dtype>
class DAUConvBackward {
	// TODO:
	//	- make interpolation weights in 16 bit float (they are computed with 32 bit error so cannot use 16 bit float arithmetics)
	//  - make input data in 16 bit float but retain error in 32 bit float and perform computation in 16 bit (this will reduce memory bandwidth required)
	// --> tried but not worked:
	//      float 16 bit does half transfer time, but adds additionl conversions from fp16 to fp32 which brings total time back to the same !!
	//		--> would be possible with new Nvidia VOLTA arch which should have fp16 dot product with aggregation to fp32 !!!
	//
	//  - make data and computation with 16 bit float (only viable version but effect on performance is yet unknown)
public:
	// fixed params during construction
	const int img_width_in, img_height_in;
	const int img_width, img_height;
	const int I, S, F, G, IN_K;
	int OUT_K; // this is const but is calculated in constructor

private:
	// this parameters are used as template params for DAUConvBackwardCUDA
	int patch_size_w, patch_size_h, num_images;
	bool use_smaller_warp_and_group_k, use_interpolation, single_subfeature;
	bool last_k_optional;


public:
	DAUConvBackward(const int img_width_in, const int img_height_in, const int img_width, const int img_height, const int I, const int S, const int F, const int G, const int K, const bool last_k_optional, const bool use_interpolation);

	void get_allocation_sizes(const int kernel_width, const int kernel_height, const bool offsets_already_centered,
                              size_t* prepared_filtered_images_size, size_t* prepared_error_images_size, size_t* prepared_filter_weights_size, size_t* prepared_filter_offsets_size);

	void backward_pass(const Dtype* filtered_images, const Dtype* error_images,
					   const Dtype* filter_offsets_float_x, const Dtype* filter_offsets_float_y,
					   const Dtype* filter_weights,
					   const int kernel_w, const int kernel_h, const Dtype actual_max_offset,
                       const bool offsets_already_centered,
					   Dtype* output,
					   Dtype* prepared_filtered_images,
					   Dtype* prepared_error_images,
					   Dtype* prepared_filter_weights,
					   int* prepared_filter_offsets,
					   const bool ignore_edge_gradients,
					   cudaStream_t streamId = 0);

	class CUDAParams {
	public:
		// fixed params during construction
		const int img_width_in, img_height_in;
		const int img_width, img_height;
		const int I, S, F, G, K, IN_K;

		// parameters to setup before call

		// params for get_allocation_sizes call
		size_t* alloc_img, *alloc_err, *alloc_w, *alloc_off;

		// params for run_kernel call
		Dtype const* filtered_images, *error_images, *filter_offsets_float_x, *filter_offsets_float_y, *filter_weights;
		Dtype* output, *prepared_filtered_images, *prepared_error_images, *prepared_filter_weights;
		int* prepared_filter_offsets;
		int kernel_w, kernel_h;
		bool ignore_edge_gradients;
		bool offsets_already_centered;
		cudaStream_t streamId;

		float actual_max_offset;

		CUDAParams(const int img_width_in, const int img_height_in, const int img_width, const int img_height, const int I, const int S, const int F, const int G, const int K, const int IN_K, const bool offsets_already_centered) :
				img_width_in(img_width_in), img_height_in(img_height_in), img_width(img_width), img_height(img_height), I(I), S(S), F(F), G(G), K(K), IN_K(IN_K), offsets_already_centered(offsets_already_centered){

		}
		void set_params_for_allocation_call(size_t* alloc_img, size_t* alloc_err, size_t* alloc_w, size_t* alloc_off);

		void set_params_for_kernel_call(const Dtype* filtered_images, const Dtype* error_images,
										const Dtype* filter_offsets_float_x, const Dtype* filter_offsets_float_y,
										const Dtype* filter_weights, const int kernel_w, const int kernel_h, const Dtype actual_max_offset,
										Dtype* output,
										Dtype* prepared_filtered_images,
										Dtype* prepared_error_images,
										Dtype* prepared_filter_weights,
										int* prepared_filter_offsets,
										const bool ignore_edge_gradients,
										cudaStream_t streamId);
	};
private:

	void call_cuda_kernel(CUDAParams& params);

	static int select_optimal_block_size_bw(int img_size, int min_power, int max_power);

};



// we make explicit functions for different combinations of
// each function is implemented in separate .cu file to allow for parallel compile
// (there are 288 combination all-together so this way we can reduce compute time by a factor of 8 if enough CPU cores)
void DAUConv_backward_multi_subfeatures_patch_1x1(int IMG_PATCH_SIZE_W, int IMG_PATCH_SIZE_H, int MAX_OFFSET, bool SMALLER_WARP_AND_GROUP_K, int BATCH_IMAGES, bool USE_INTERPOLATION, bool SINGLE_SUBFEATURE, DAUConvBackward<float>::CUDAParams &PARAMS);
void DAUConv_backward_multi_subfeatures_patch_8x8(int IMG_PATCH_SIZE_W, int IMG_PATCH_SIZE_H, int MAX_OFFSET, bool SMALLER_WARP_AND_GROUP_K, int BATCH_IMAGES, bool USE_INTERPOLATION, bool SINGLE_SUBFEATURE, DAUConvBackward<float>::CUDAParams &PARAMS);
void DAUConv_backward_multi_subfeatures_patch_8x16(int IMG_PATCH_SIZE_W, int IMG_PATCH_SIZE_H, int MAX_OFFSET, bool SMALLER_WARP_AND_GROUP_K, int BATCH_IMAGES, bool USE_INTERPOLATION, bool SINGLE_SUBFEATURE, DAUConvBackward<float>::CUDAParams &PARAMS);
void DAUConv_backward_multi_subfeatures_patch_8x32(int IMG_PATCH_SIZE_W, int IMG_PATCH_SIZE_H, int MAX_OFFSET, bool SMALLER_WARP_AND_GROUP_K, int BATCH_IMAGES, bool USE_INTERPOLATION, bool SINGLE_SUBFEATURE, DAUConvBackward<float>::CUDAParams &PARAMS);
void DAUConv_backward_multi_subfeatures_patch_8x64(int IMG_PATCH_SIZE_W, int IMG_PATCH_SIZE_H, int MAX_OFFSET, bool SMALLER_WARP_AND_GROUP_K, int BATCH_IMAGES, bool USE_INTERPOLATION, bool SINGLE_SUBFEATURE, DAUConvBackward<float>::CUDAParams &PARAMS);
void DAUConv_backward_multi_subfeatures_patch_16x8(int IMG_PATCH_SIZE_W, int IMG_PATCH_SIZE_H, int MAX_OFFSET, bool SMALLER_WARP_AND_GROUP_K, int BATCH_IMAGES, bool USE_INTERPOLATION, bool SINGLE_SUBFEATURE, DAUConvBackward<float>::CUDAParams &PARAMS);
void DAUConv_backward_multi_subfeatures_patch_16x16(int IMG_PATCH_SIZE_W, int IMG_PATCH_SIZE_H, int MAX_OFFSET, bool SMALLER_WARP_AND_GROUP_K, int BATCH_IMAGES, bool USE_INTERPOLATION, bool SINGLE_SUBFEATURE, DAUConvBackward<float>::CUDAParams &PARAMS);
void DAUConv_backward_multi_subfeatures_patch_16x32(int IMG_PATCH_SIZE_W, int IMG_PATCH_SIZE_H, int MAX_OFFSET, bool SMALLER_WARP_AND_GROUP_K, int BATCH_IMAGES, bool USE_INTERPOLATION, bool SINGLE_SUBFEATURE, DAUConvBackward<float>::CUDAParams &PARAMS);
void DAUConv_backward_multi_subfeatures_patch_16x64(int IMG_PATCH_SIZE_W, int IMG_PATCH_SIZE_H, int MAX_OFFSET, bool SMALLER_WARP_AND_GROUP_K, int BATCH_IMAGES, bool USE_INTERPOLATION, bool SINGLE_SUBFEATURE, DAUConvBackward<float>::CUDAParams &PARAMS);
void DAUConv_backward_multi_subfeatures_patch_32x8(int IMG_PATCH_SIZE_W, int IMG_PATCH_SIZE_H, int MAX_OFFSET, bool SMALLER_WARP_AND_GROUP_K, int BATCH_IMAGES, bool USE_INTERPOLATION, bool SINGLE_SUBFEATURE, DAUConvBackward<float>::CUDAParams &PARAMS);
void DAUConv_backward_multi_subfeatures_patch_32x16(int IMG_PATCH_SIZE_W, int IMG_PATCH_SIZE_H, int MAX_OFFSET, bool SMALLER_WARP_AND_GROUP_K, int BATCH_IMAGES, bool USE_INTERPOLATION, bool SINGLE_SUBFEATURE, DAUConvBackward<float>::CUDAParams &PARAMS);
void DAUConv_backward_multi_subfeatures_patch_32x32(int IMG_PATCH_SIZE_W, int IMG_PATCH_SIZE_H, int MAX_OFFSET, bool SMALLER_WARP_AND_GROUP_K, int BATCH_IMAGES, bool USE_INTERPOLATION, bool SINGLE_SUBFEATURE, DAUConvBackward<float>::CUDAParams &PARAMS);
void DAUConv_backward_multi_subfeatures_patch_32x64(int IMG_PATCH_SIZE_W, int IMG_PATCH_SIZE_H, int MAX_OFFSET, bool SMALLER_WARP_AND_GROUP_K, int BATCH_IMAGES, bool USE_INTERPOLATION, bool SINGLE_SUBFEATURE, DAUConvBackward<float>::CUDAParams &PARAMS);
void DAUConv_backward_multi_subfeatures_patch_64x8(int IMG_PATCH_SIZE_W, int IMG_PATCH_SIZE_H, int MAX_OFFSET, bool SMALLER_WARP_AND_GROUP_K, int BATCH_IMAGES, bool USE_INTERPOLATION, bool SINGLE_SUBFEATURE, DAUConvBackward<float>::CUDAParams &PARAMS);
void DAUConv_backward_multi_subfeatures_patch_64x16(int IMG_PATCH_SIZE_W, int IMG_PATCH_SIZE_H, int MAX_OFFSET, bool SMALLER_WARP_AND_GROUP_K, int BATCH_IMAGES, bool USE_INTERPOLATION, bool SINGLE_SUBFEATURE, DAUConvBackward<float>::CUDAParams &PARAMS);
void DAUConv_backward_multi_subfeatures_patch_64x32(int IMG_PATCH_SIZE_W, int IMG_PATCH_SIZE_H, int MAX_OFFSET, bool SMALLER_WARP_AND_GROUP_K, int BATCH_IMAGES, bool USE_INTERPOLATION, bool SINGLE_SUBFEATURE, DAUConvBackward<float>::CUDAParams &PARAMS);
void DAUConv_backward_multi_subfeatures_patch_64x64(int IMG_PATCH_SIZE_W, int IMG_PATCH_SIZE_H, int MAX_OFFSET, bool SMALLER_WARP_AND_GROUP_K, int BATCH_IMAGES, bool USE_INTERPOLATION, bool SINGLE_SUBFEATURE,DAUConvBackward<float>::CUDAParams &PARAMS);

#endif  // !CPU_ONLY

}  // namespace DAUConvNet

#endif  // DAU_CONV_UTIL_DAU_BACKWARD_H_
