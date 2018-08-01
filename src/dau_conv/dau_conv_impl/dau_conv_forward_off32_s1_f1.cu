#include "dau_conv/dau_conv_impl/dau_conv_forward_core.hpp"

namespace  DAUConvNet {

void DAUConv_forward_float_off_32_single_feat_1_single_subfeat_1(int IMG_PATCH_SIZE_W, int IMG_PATCH_SIZE_H,
                                                                int MAX_OFFSET, int WARP_PIXELS_X, int WARP_PIXELS_Y,
                                                                int BLOCK_IMAGES, int USE_INTERPOLATION,
                                                                DAUConvForward<float>::CUDAParams &PARAMS){
#define SINGLE_FEATURE true
#define SINGLE_SUBFEATURE true
#define MAX_OFFSET 32

    if (IMG_PATCH_SIZE_W == 1 && IMG_PATCH_SIZE_H == 1 && WARP_PIXELS_X == 1 && WARP_PIXELS_Y == 1) {
        RUN_KERNEL_R1(DAUConvForwardCUDA, 4, 1, MAX_OFFSET, 4, 1, 1, USE_INTERPOLATION, SINGLE_FEATURE, SINGLE_SUBFEATURE, PARAMS)
    } else if (IMG_PATCH_SIZE_W == 8 && WARP_PIXELS_X == 8) {
        RUN_KERNEL_R2(DAUConvForwardCUDA, 8, IMG_PATCH_SIZE_H, MAX_OFFSET, 8, 8, 1, USE_INTERPOLATION, SINGLE_FEATURE, SINGLE_SUBFEATURE, PARAMS)
    } else if (WARP_PIXELS_X == 16) {
        RUN_KERNEL_R3(DAUConvForwardCUDA, IMG_PATCH_SIZE_W, IMG_PATCH_SIZE_H, MAX_OFFSET, 16, 8, 1, USE_INTERPOLATION, SINGLE_FEATURE, SINGLE_SUBFEATURE, PARAMS)
    } else if (WARP_PIXELS_X == 32)  {
        RUN_KERNEL_R3(DAUConvForwardCUDA, IMG_PATCH_SIZE_W, IMG_PATCH_SIZE_H, MAX_OFFSET, 16, 8, 1, USE_INTERPOLATION, SINGLE_FEATURE, SINGLE_SUBFEATURE, PARAMS)
    } else {
        printf("Unsupported WARP_PIXELS_X: %d. Supported only 16 or 32 at the moment (or 1 when WARP_PIXELS_Y==1 as well) \n", WARP_PIXELS_X);
        throw std::exception();
    }
    //RUN_KERNEL_R4(DAUConvForwardCUDA, IMG_PATCH_SIZE_W, IMG_PATCH_SIZE_H, 32, WARP_PIXELS_X, WARP_PIXELS_Y, BLOCK_IMAGES, USE_INTERPOLATION, false, false, PARAMS);
}
}