//
// Created by domen on 3/21/18.
//

#include <memory>
#include "dau_conv/dau_conv_impl/dau_conv_backward.hpp"
#include "dau_conv/base_dau_conv_layer.hpp"


int main(int argc, char** argv) {

/*
    DAUConvSettings layer_param;

    DAUConvLayerCaffeGPU<float> layer;
    const int N = 128;
    const int S = 32;
    const int F = 64;
    const int H = 64;
    const int W = 64;

    Blob<float> input(N,S,H,W);
    Blob<float> output(N,F,H,W);

    vector<Blob<float>*> top;
    vector<bool> param_propagate_down;

    DAUKernelComputeGPU<float>* dau_kernel_compute = new DAUKernelComputeGPU<float>();
    DAUKernelParamsGPU<float>* dau_kernel_params = new DAUKernelParamsGPU<float>();
    DAUKernelOutputGPU<float>* dau_kernel_output = new DAUKernelOutputGPU<float>();


    layer.LayerSetUp(layer_param,
                     dau_kernel_compute, dau_kernel_params, dau_kernel_output,
                     param_propagate_down, input.shape());

    layer.Reshape(input.shape(), output.shape());

    layer.Forward_gpu(input.gpu_data(), input.shape(), output.mutable_gpu_data(), output.shape());

    layer.Backward_gpu(output.gpu_data(), output.gpu_diff(), output.shape(), true,
                       input.gpu_data(), input.mutable_gpu_diff(), input.shape());
*/
}