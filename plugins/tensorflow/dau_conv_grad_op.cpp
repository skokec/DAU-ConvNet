#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "dau_conv/base_dau_conv_layer.hpp"
#include "dau_conv_layer_tensorflow.hpp"
#include "dau_conv/util/math_functions.hpp"


using namespace tensorflow;

using CPUDevice = Eigen::ThreadPoolDevice;
using GPUDevice = Eigen::GpuDevice;

REGISTER_OP("DAUConvGrad")
        .Input("grad: float32") //error
        .Input("input: float32") // input
        .Input("weights: float32") // input
        .Input("mu1: float32") // 4 inputi, w,mu12,sigma
        .Input("mu2: float32") // 4 inputi, w,mu12,sigma
        .Input("sigma: float32") // 4 inputi, w,mu12,sigma
        .Output("grad_input: float32") //error naprej
        .Output("grad_weights: float32") //
        .Output("grad_mu1: float32") //
        .Output("grad_mu2: float32") //
        .Output("grad_sigma: float32")
        .Attr("number_units_x : int  = 2")
        .Attr("number_units_y : int = 2")
        .Attr("num_output : int = 64")
        .Attr("kernel_size: int = 9")
        .Attr("pad: int = 4")
        .Attr("stride: int = 1")
        .Attr("unit_normalization: bool = true")
        .Attr("square_unit_normalization: bool = false")
        .Attr("mean_iteration_step: int = 1")
        .Attr("sigma_iteration_step: int = 1")
        .Attr("component_border_bound: int = 4")
        .Attr("sigma_lower_bound: float = 0.3")
        .Attr("merge_iteration_step: int = 0")
        .Attr("merge_threshold: int = 1")
        .Attr("unit_testing: bool = false")
        .Attr("mu_learning_rate_factor: float = 1.0");
//TODO ADD SETTING INITIALIZATION FROM ATTRIBUTES
template<typename Device, typename Dtype>
class DAUConvGradOp : public OpKernel {
public:
    explicit DAUConvGradOp(OpKernelConstruction *context) : OpKernel(context) {
        int number_units_x;
        int number_units_y;
        int num_output;
        int kernel_size;
        int pad;
        int stride;
        bool unit_normalization;
        bool square_unit_normalization;
        int mean_iteration_step;
        int sigma_iteration_step;
        int component_border_bound;
        float sigma_lower_bound;
        int merge_iteration_step;
        int merge_threshold;
        OP_REQUIRES_OK(context, context->GetAttr("number_units_x", &number_units_x));
        OP_REQUIRES_OK(context, context->GetAttr("number_units_y", &number_units_y));
        OP_REQUIRES_OK(context, context->GetAttr("num_output", &num_output));
        OP_REQUIRES_OK(context, context->GetAttr("kernel_size", &kernel_size));
        OP_REQUIRES_OK(context, context->GetAttr("pad", &pad));
        OP_REQUIRES_OK(context, context->GetAttr("stride", &stride));
        OP_REQUIRES_OK(context, context->GetAttr("unit_normalization", &unit_normalization));
        OP_REQUIRES_OK(context, context->GetAttr("square_unit_normalization", &square_unit_normalization));
        OP_REQUIRES_OK(context, context->GetAttr("mean_iteration_step", &mean_iteration_step));
        OP_REQUIRES_OK(context, context->GetAttr("sigma_iteration_step", &sigma_iteration_step));
        OP_REQUIRES_OK(context, context->GetAttr("component_border_bound", &component_border_bound));
        OP_REQUIRES_OK(context, context->GetAttr("sigma_lower_bound", &sigma_lower_bound));
        OP_REQUIRES_OK(context, context->GetAttr("merge_iteration_step", &merge_iteration_step));
        OP_REQUIRES_OK(context, context->GetAttr("merge_threshold", &merge_threshold));
        OP_REQUIRES_OK(context, context->GetAttr("unit_testing", &this->unit_testing));
        OP_REQUIRES_OK(context, context->GetAttr("mu_learning_rate_factor", &this->mu_learning_rate_factor));
        dau_conv_settings.offsets_already_centered = true;
        dau_conv_settings.num_output = num_output;
        //num units per X and per Y
        dau_conv_settings.number_units.push_back(number_units_x);
        dau_conv_settings.number_units.push_back(number_units_y);
        dau_conv_settings.bias_term = false;
        dau_conv_settings.kernel_size = kernel_size;
        dau_conv_settings.pad = pad;
        dau_conv_settings.stride = stride;
        dau_conv_settings.unit_normalization = unit_normalization;
        dau_conv_settings.square_unit_normalization = square_unit_normalization;
        dau_conv_settings.mean_iteration_step = mean_iteration_step;
        dau_conv_settings.sigma_iteration_step = sigma_iteration_step;
        dau_conv_settings.component_border_bound = component_border_bound;
        dau_conv_settings.sigma_lower_bound = sigma_lower_bound;
        dau_conv_settings.merge_iteration_step = merge_iteration_step;
        dau_conv_settings.merge_threshold = merge_threshold;


    }

    void Compute(OpKernelContext *context) override {


        DCHECK_EQ(6, context->num_inputs());

        // in_train is used only for merge_iteration_step, which is not setup.
        bool in_train = false;

        const Tensor *grad;
        context->input("grad", &grad);
        const Tensor *input;
        context->input("input", &input);
        const Tensor *weights;
        context->input("weights", &weights);
        const Tensor *mu1;
        context->input("mu1", &mu1);
        const Tensor *mu2;
        context->input("mu2", &mu2);
        const Tensor *sigma;
        context->input("sigma", &sigma);

        // create input shape (inferred from the additional attribute `n`)
        TensorShape input_shape = input->shape();
        TensorShape weights_shape = weights->shape();

        //Check if output size of parameters equals to specified number of outputs
        DCHECK_EQ(dau_conv_settings.num_output, weights_shape.dim_size(weights_shape.dims()-1));
        DCHECK_EQ(dau_conv_settings.num_output, mu1->shape().dim_size(mu1->shape().dims()-1));
        DCHECK_EQ(dau_conv_settings.num_output, mu2->shape().dim_size(mu2->shape().dims()-1));
        //DCHECK_EQ(dau_conv_settings.num_output, sigma->shape().dim_size(sigma->shape().dims()-1));


        // create output tensors
        Tensor *grad_input = NULL;
        Tensor *grad_weights = NULL;
        Tensor *grad_mu1 = NULL;
        Tensor *grad_mu2 = NULL;
        Tensor *grad_sigma = NULL;
        OP_REQUIRES_OK(context, context->allocate_output(1, weights_shape, &grad_weights));
        OP_REQUIRES_OK(context, context->allocate_output(2, mu1->shape(), &grad_mu1));
        OP_REQUIRES_OK(context, context->allocate_output(3, mu2->shape(), &grad_mu2));
        OP_REQUIRES_OK(context, context->allocate_output(4, sigma->shape(), &grad_sigma));


        CUDA_CHECK(cudaMemset(TENSOR_DATA_PTR(grad_weights,Dtype),0, grad_weights->NumElements() * sizeof(Dtype)));
        CUDA_CHECK(cudaMemset(TENSOR_DATA_PTR(grad_mu1, Dtype),0, grad_mu1->NumElements() * sizeof(Dtype)));
        CUDA_CHECK(cudaMemset(TENSOR_DATA_PTR(grad_mu2, Dtype),0, grad_mu2->NumElements() * sizeof(Dtype)));
        //CUDA_CHECK(cudaMemset(reinterpret_cast<Dtype*>(grad_sigma->flat<Dtype>().data()),0, grad_sigma->NumElements() * sizeof(Dtype)));

        //Initializer does nothing, input values were of type Filler in caffe
        // tensorflow variables are initialized in python.
        NullDAUComponentInitializerTensorflow<Dtype> param_initializer;

        DAUKernelComputeTFGPU<Dtype> dau_kernel_compute(context);
        DAUKernelParamsTFGPU<Dtype> dau_kernel_params(context);
        DAUKernelOutputTFGPU<Dtype> dau_kernel_output(context);
        //dau_kernel_params.initialize_params(param_w, param_mu1, param_mu2, param_sigma);


        std::vector<int> bottom_shape;
        for (int i = 0; i < input_shape.dims(); i++) {
            bottom_shape.push_back(input_shape.dim_size(i));
        }


        cublasHandle_t handle;
        cublasCreate(&handle);
        //const cudaStream_t* stream = CHECK_NOTNULL(reinterpret_cast<const cudaStream_t*>(context->op_device_context()
        //                                                                                -> stream()->implementation()
        //                                                                                ->CudaStreamMemberHack()) );
        //cublasSetStream(handle, stream);
        //TODO Get stream from context and add it to handle..

        DAUConvLayerTensorflowGPU<Dtype> tf_layer(handle, context, this->unit_testing);

        //set parameters from input tensors
        //tf_layer.InitializeFromInput(dau_conv_settings, &weights_non_const,&mu1_non_const,&mu2_non_const,&sigma_non_const);

        tf_layer.enable_forward(false);
        tf_layer.enable_backward(true);

        // prevent display of allocation size on each call (except when doing unit testing)
        tf_layer.enable_memalloc_info(this->unit_testing == true ? true : false);

        tf_layer.InitializeFromInput(dau_conv_settings, (Tensor *) weights, (Tensor *) mu1, (Tensor *) mu2,
                                     (Tensor *) sigma);

        tf_layer.InitializeGrad(dau_conv_settings, grad_weights, grad_mu1, grad_mu2, grad_sigma);

        tf_layer.LayerSetUp(dau_conv_settings, param_initializer, &dau_kernel_compute, &dau_kernel_params,
                            &dau_kernel_output, bottom_shape, in_train);

        std::vector<int> top_shape;
        top_shape.push_back(input_shape.dim_size(0));
        top_shape.push_back(weights->dim_size(1));
        top_shape.push_back(input_shape.dim_size(2));
        top_shape.push_back(input_shape.dim_size(3));

        tf_layer.Reshape(bottom_shape, top_shape);

        //tf_layer forward_gpu implement..

        TensorShape output_shape;
        for (int i = 0; i < top_shape.size(); i++) output_shape.AddDim(top_shape[i]);

        // grad is top error since it is the error of the output of the layer
        const Dtype *top_error = TENSOR_DATA_PTR_CONST(grad, Dtype);

        const Dtype *bottom_data = TENSOR_DATA_PTR_CONST(input, Dtype);

        OP_REQUIRES_OK(context, context->allocate_output(0, input_shape, &grad_input));
        Dtype *bottom_error = TENSOR_DATA_PTR(grad_input,Dtype);


        tf_layer.Backward_gpu(NULL, top_error, top_shape, true, bottom_data, bottom_error, bottom_shape,
                              {true, true, true, false, false});

        // multiply mu with learning rate if needed
        if (mu_learning_rate_factor != 1.0) {
            Dtype* mu1_data = TENSOR_DATA_PTR(grad_mu1, Dtype);
            Dtype* mu2_data = TENSOR_DATA_PTR(grad_mu2, Dtype);

            DAUConvNet::caffe_gpu_scal<Dtype>(grad_mu1->NumElements(), mu_learning_rate_factor, mu1_data, handle);
            DAUConvNet::caffe_gpu_scal<Dtype>(grad_mu2->NumElements(), mu_learning_rate_factor, mu2_data, handle);
        }

    }
private:
    DAUConvNet::DAUConvSettings dau_conv_settings;
    bool unit_testing;
    float mu_learning_rate_factor;
};

REGISTER_KERNEL_BUILDER(Name("DAUConvGrad").Device(DEVICE_GPU), DAUConvGradOp<GPUDevice, float>);