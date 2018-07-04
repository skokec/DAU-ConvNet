#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "dau_conv/base_dau_conv_layer.hpp"
#include "dau_conv_layer_tensorflow.hpp"


using namespace tensorflow;

using CPUDevice = Eigen::ThreadPoolDevice;
using GPUDevice = Eigen::GpuDevice;

REGISTER_OP("BaseOpGrad")
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
        .Attr("bias_term: bool = true")
        .Attr("kernel_size: int = 9")
        .Attr("pad: int = 4")
        .Attr("stride: int = 1")
        .Attr("unit_normalization: bool = true")
        .Attr("square_unit_normalization: bool = true")
        .Attr("mean_iteration_step: int = 1")
        .Attr("sigma_iteration_step: int = 1")
        .Attr("component_border_bound: int = 4")
        .Attr("sigma_lower_bound: float = 0.3")
        .Attr("merge_iteration_step: int = 0")
        .Attr("merge_threshold: int = 1");
//TODO ADD SETTING INITIALIZATION FROM ATTRIBUTES
template<typename Device, typename Dtype>
class BaseOpGradOp : public OpKernel {
public:
    explicit BaseOpGradOp(OpKernelConstruction *context) : OpKernel(context) {
        int number_units_x;
        int number_units_y;
        bool bias_term;
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
        OP_REQUIRES_OK(context, context->GetAttr("bias_term", &bias_term));
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

        dau_conv_settings.offsets_already_centered = true;
        //TODO calculate from inputs
        dau_conv_settings.num_output = 64;
        //num units per X and per Y
        dau_conv_settings.number_units.push_back(number_units_x);
        dau_conv_settings.number_units.push_back(number_units_y);
        dau_conv_settings.bias_term = bias_term;
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

        const Tensor *grad;
        context->input("grad", &grad);
        const Tensor *input;
        context->input("input", &input);
        const Tensor *weights;
        context->input("weights", &weights);
        for(int i = 0; i < weights->shape().dims(); i++){
            printf("%d\n", weights->shape().dim_size(i));
        }
        const Tensor *mu1;
        context->input("mu1", &mu1);
        const Tensor *mu2;
        context->input("mu2", &mu2);
        const Tensor *sigma;
        context->input("sigma", &sigma);

        // create input shape (inferred from the additional attribute `n`)
        TensorShape input_shape = input->shape();
        TensorShape weights_shape = weights->shape();

        //DCHECK_EQ(input_shape.dim_size(0), weights_shape.dim_size(1));
        //DCHECK_EQ(weights_shape.dim_size(0), grad.shape().dim_size(0));

        // create output tensors
        Tensor *grad_input = NULL;
        Tensor *grad_weights = NULL;
        Tensor *grad_mu1 = NULL;
        Tensor *grad_mu2 = NULL;
        Tensor *grad_sigma = NULL;
        OP_REQUIRES_OK(context, context->allocate_output(1, weights_shape, &grad_weights));
        OP_REQUIRES_OK(context, context->allocate_output(2, weights_shape, &grad_mu1));
        OP_REQUIRES_OK(context, context->allocate_output(3, weights_shape, &grad_mu2));
        OP_REQUIRES_OK(context, context->allocate_output(4, weights_shape, &grad_sigma));
        Dtype *grad_w_buf = static_cast<Dtype *>(grad_weights->flat<Dtype>().data());
        cudaError_t cuda_error_w = cudaMemset(grad_w_buf, 0, sizeof(Dtype) * grad_weights->NumElements());
        if (cuda_error_w != cudaSuccess) printf("Cuda error weights %d \n", cuda_error_w);
        Dtype *grad_mu1_buf = static_cast<Dtype *>(grad_mu1->flat<Dtype>().data());
        cuda_error_w = cudaMemset(grad_mu1_buf, 0, sizeof(Dtype) * grad_mu1->NumElements());
        if (cuda_error_w != cudaSuccess) printf("Cuda error weights %d \n", cuda_error_w);
        Dtype *grad_mu2_buf = static_cast<Dtype *>(grad_mu2->flat<Dtype>().data());
        cuda_error_w = cudaMemset(grad_mu2_buf, 0, sizeof(Dtype) * grad_mu2->NumElements());
        if (cuda_error_w != cudaSuccess) printf("Cuda error weights %d \n", cuda_error_w);
        Dtype *grad_sigma_buf = static_cast<Dtype *>(grad_sigma->flat<Dtype>().data());
        cuda_error_w = cudaMemset(grad_sigma_buf, 0, sizeof(Dtype) * grad_sigma->NumElements());
        if (cuda_error_w != cudaSuccess) printf("Cuda error weights %d \n", cuda_error_w);



        // allocate tensors for DAUKernelParams
        /*
        Tensor param_w;
        Tensor param_mu1;
        Tensor param_mu2;
        Tensor param_sigma;
        TensorShape param_shape(
                {1, input->shape().dim_size(1), weights->shape().dim_size(1), weights->shape().dim_size(3)});
        //TensorShape param_shape({1,1,1,1});
        OP_REQUIRES_OK(context, context->allocate_temp(weights->dtype(), param_shape, &param_w));
        OP_REQUIRES_OK(context, context->allocate_temp(mu1->dtype(), param_shape, &param_mu1));
        OP_REQUIRES_OK(context, context->allocate_temp(mu2->dtype(), param_shape, &param_mu2));
        OP_REQUIRES_OK(context, context->allocate_temp(sigma->dtype(), param_shape, &param_sigma));

        Dtype *param_w_buf = static_cast<Dtype *>(param_w.flat<Dtype>().data());
        cuda_error_w = cudaMemset(param_w_buf, 0, sizeof(Dtype) * param_w.NumElements());
        if (cuda_error_w != cudaSuccess) printf("Cuda error weights %d \n", cuda_error_w);
        Dtype *param_mu1_buf = static_cast<Dtype *>(param_mu1.flat<Dtype>().data());
        CUDA_CHECK(cudaMemset(param_mu1_buf, 0, sizeof(Dtype) * param_mu1.NumElements()));
        Dtype *param_mu2_buf = static_cast<Dtype *>(param_mu2.flat<Dtype>().data());
        CUDA_CHECK(cudaMemset(param_mu2_buf, 0, sizeof(Dtype) * param_mu2.NumElements()));
        Dtype *param_sigma_buf = static_cast<Dtype *>(param_sigma.flat<Dtype>().data());
        CUDA_CHECK(cudaMemset(param_sigma_buf, 0, sizeof(Dtype) * param_sigma.NumElements()));
        */


        //Initializer does nothing, input values were of type Filler in caffe
        // tensorflow variables are initialized in python.
        DAUComponentInitializerTensorflow<Dtype> param_initializer(1, 1, 1);

        DAUKernelComputeTFGPU<Dtype> dau_kernel_compute(context);
        DAUKernelParamsTFGPU<Dtype> dau_kernel_params(context);
        DAUKernelOutputTFGPU<Dtype> dau_kernel_output(context);
        //dau_kernel_params.initialize_params(param_w, param_mu1, param_mu2, param_sigma);

        // TODO check how you can tell if it is in training? maybe pass it as argument in
        // Op call?
        bool in_train = true;

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


        DAUConvLayerTensorflowGPU<Dtype> tf_layer(handle, context);

        //set parameters from input tensors
        //tf_layer.InitializeFromInput(dau_conv_settings, &weights_non_const,&mu1_non_const,&mu2_non_const,&sigma_non_const);

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


        OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &grad_input));
        auto out_data = grad_input->flat<Dtype>();
        //backward pass does not depend on top data?
        Dtype *top_data = static_cast<Dtype *>(out_data.data());
        cuda_error_w = cudaMemset(top_data, 0, sizeof(Dtype) * grad_input->NumElements());

        // grad is top error since it is the error of the output of the layer
        const Dtype *top_error = static_cast<const Dtype *>(grad->flat<Dtype>().data());

        const Dtype *bottom_data = static_cast<const Dtype *>(input->flat<Dtype>().data());

        OP_REQUIRES_OK(context, context->allocate_output(0, input_shape, &grad_input));
        Dtype *bottom_error = static_cast<Dtype *>(grad_input->flat<Dtype>().data());
        cuda_error_w = cudaMemset(bottom_error, 0, sizeof(Dtype) * grad_input->NumElements());
        if (cuda_error_w != cudaSuccess) printf("Cuda error weights %d \n", cuda_error_w);


        tf_layer.Backward_gpu(top_data, top_error, top_shape, true, bottom_data, bottom_error, bottom_shape,
                              {true, true, true, true, true});

        //Backward_gpu(const Dtype* top_data, const Dtype* top_error, const vector<int>& top_shape, bool propagate_down,
        //                          const Dtype* bottom_data, Dtype* bottom_error, const vector<int>& bottom_shape, const vector<bool>& params_propagate_down );


    }
private:
    DAUConvNet::DAUConvSettings dau_conv_settings;

};

REGISTER_KERNEL_BUILDER(Name("BaseOpGrad").Device(DEVICE_GPU), BaseOpGradOp<GPUDevice, float>);