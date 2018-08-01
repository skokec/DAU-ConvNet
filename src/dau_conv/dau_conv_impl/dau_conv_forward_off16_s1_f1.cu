#include "dau_conv/dau_conv_impl/dau_conv_forward_core.hpp"

namespace  DAUConvNet {

void DAUConv_forward_float_off_16_single_feat_1_single_subfeat_1(int IMG_PATCH_SIZE_W, int IMG_PATCH_SIZE_H,
                                                                int MAX_OFFSET_, int WARP_PIXELS_X, int WARP_PIXELS_Y,
                                                                int BLOCK_IMAGES, int USE_INTERPOLATION,
                                                                DAUConvForward<float>::CUDAParams &PARAMS){
#define SINGLE_FEATURE true
#define SINGLE_SUBFEATURE true
#define MAX_OFFSET 16

    if (IMG_PATCH_SIZE_W == 1 && IMG_PATCH_SIZE_H == 1 && WARP_PIXELS_X == 1 && WARP_PIXELS_Y == 1) {
        if (BLOCK_IMAGES % 2 == 0) {
		    RUN_KERNEL_R1(DAUConvForwardCUDA, 2, 1, MAX_OFFSET, 2, 1, 2, USE_INTERPOLATION, SINGLE_FEATURE, SINGLE_SUBFEATURE, PARAMS);
        } else {
            RUN_KERNEL_R1(DAUConvForwardCUDA, 4, 1, MAX_OFFSET, 4, 1, 1, USE_INTERPOLATION, SINGLE_FEATURE, SINGLE_SUBFEATURE, PARAMS)
		    /*printf("Unsupported BATCH SIZE for 1x1 pixels: Supported only a multiple of 16 (at MAX_OFFSET<=4), 8 (at MAX_OFFSET<=8) or 4 images at the moment\n"); */
            /*throw std::exception();*/
        }
    } else if (IMG_PATCH_SIZE_W == 8 && WARP_PIXELS_X == 8) {
        /* We have 8px WARP_PIXELS_X sizes only for smaller patch sizes - but check just in case (fixing IMG_PATCH_SIZE_W avoids unneeded computation as well) */
        if (BLOCK_IMAGES % 2 == 0) {
		    RUN_KERNEL_R2(DAUConvForwardCUDA, 8, IMG_PATCH_SIZE_H, MAX_OFFSET, 8, 8, 2, USE_INTERPOLATION, SINGLE_FEATURE, SINGLE_SUBFEATURE, PARAMS)
	    } else {
		    RUN_KERNEL_R2(DAUConvForwardCUDA, 8, IMG_PATCH_SIZE_H, MAX_OFFSET, 8, 8, 1, USE_INTERPOLATION, SINGLE_FEATURE, SINGLE_SUBFEATURE, PARAMS)
        }
    } else if (WARP_PIXELS_X == 16) {
		if (BLOCK_IMAGES % 2 == 0) {
		    RUN_KERNEL_R3(DAUConvForwardCUDA, IMG_PATCH_SIZE_W, IMG_PATCH_SIZE_H, MAX_OFFSET, 16, 8, 2, USE_INTERPOLATION, SINGLE_FEATURE, SINGLE_SUBFEATURE, PARAMS)
	    } else {
		    RUN_KERNEL_R3(DAUConvForwardCUDA, IMG_PATCH_SIZE_W, IMG_PATCH_SIZE_H, MAX_OFFSET, 16, 8, 1, USE_INTERPOLATION, SINGLE_FEATURE, SINGLE_SUBFEATURE, PARAMS)
        }
    } else if (WARP_PIXELS_X == 32)  {
        RUN_KERNEL_R3(DAUConvForwardCUDA, IMG_PATCH_SIZE_W, IMG_PATCH_SIZE_H, MAX_OFFSET, 32, 8, 1, USE_INTERPOLATION, SINGLE_FEATURE, SINGLE_SUBFEATURE, PARAMS)
	} else {
		printf("Unsupported WARP_PIXELS_X: %d. Supported only 16 or 32 at the moment (or 1 when WARP_PIXELS_Y==1 as well) \n", WARP_PIXELS_X);
        throw std::exception();
	}
}

}