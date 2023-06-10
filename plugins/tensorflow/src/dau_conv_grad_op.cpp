#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"

#include "tensorflow/core/platform/stream_executor.h"

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
        .Attr("number_units_ignore : int = 0")
        .Attr("num_output : int = 64")
        .Attr("kernel_size: int = 9")
        .Attr("pad: int = 4")
        .Attr("stride: int = 1")
        .Attr("unit_normalization: bool = true")
        .Attr("square_unit_normalization: bool = false")
        .Attr("mean_iteration_step: int = 1")
        .Attr("sigma_iteration_step: int = 1")
        .Attr("component_border_bound: float = 0")
        .Attr("sigma_lower_bound: float = 0.3")
        .Attr("merge_iteration_step: int = 0")
        .Attr("merge_threshold: int = 1")
        .Attr("unit_testing: bool = false")
        .Attr("mu_learning_rate_factor: float = 1.0")
        .Attr("single_dim_kernel: bool = false")
        .Attr("forbid_positive_dim1: bool = false")
        .Attr("use_interpolation: bool = true");
//TODO ADD SETTING INITIALIZATION FROM ATTRIBUTES
template<typename Device, typename Dtype>
class DAUConvGradOp : public OpKernel {
public:
    explicit DAUConvGradOp(OpKernelConstruction *context) : OpKernel(context) {
        //Retrieve all Op attributes and set Op private variables.
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
        float component_border_bound;
        float sigma_lower_bound;
        int merge_iteration_step;
        int merge_threshold;
        bool use_interpolation;
        OP_REQUIRES_OK(context, context->GetAttr("number_units_x", &number_units_x));
        OP_REQUIRES_OK(context, context->GetAttr("number_units_y", &number_units_y));
        OP_REQUIRES_OK(context, context->GetAttr("number_units_ignore", &this->number_units_ignore));
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
        OP_REQUIRES_OK(context, context->GetAttr("single_dim_kernel", &this->single_dim_kernel));
        OP_REQUIRES_OK(context, context->GetAttr("forbid_positive_dim1", &this->forbid_positive_dim1));
        OP_REQUIRES_OK(context, context->GetAttr("use_interpolation", &use_interpolation));
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
        dau_conv_settings.use_interpolation = use_interpolation;

        //cublasCreate(&cublas_handle);
    }
    virtual ~DAUConvGradOp() {
        //if (cublas_handle != NULL) cublasDestroy(cublas_handle);
    }
    void Compute(OpKernelContext *context) override {

        DCHECK_EQ(6, context->num_inputs());

        // in_train is used only for merge_iteration_step, which is not setup.
        bool in_train = false;

        // enable learning of sigma by default (tensorflow can then ignore it if trainable is set to false)
        bool learn_sigma = true;

        const Tensor *grad;
        const Tensor *input;
        const Tensor *weights;
        const Tensor *mu1;
        const Tensor *mu2;
        const Tensor *sigma;

        context->input("grad", &grad);
        context->input("input", &input);
        context->input("weights", &weights);
        context->input("mu1", &mu1);
        context->input("mu2", &mu2);
        context->input("sigma", &sigma);

        TensorShape input_shape = input->shape();
        TensorShape weights_shape = weights->shape();
        TensorShape mu1_shape = mu1->shape();
        TensorShape mu2_shape = mu2->shape();
        TensorShape sigma_shape = sigma->shape();
        std::vector<int> bottom_shape;
        for (int i = 0; i < input_shape.dims(); i++)
            bottom_shape.push_back(input_shape.dim_size(i));

        // create a copy of dau_conv_settings since we can change kernel_size and padding to match actual offset values
        DAUConvNet::DAUConvSettings dau_conv_settings_ = this->dau_conv_settings;

        // Check if output size of parameters equals to specified number of outputs
        DCHECK_EQ(dau_conv_settings_.num_output, weights_shape.dim_size(weights_shape.dims()-1));
        DCHECK_EQ(dau_conv_settings_.num_output, mu1_shape.dim_size(mu1_shape.dims()-1));
        DCHECK_EQ(dau_conv_settings_.num_output, mu2_shape.dim_size(mu2_shape.dims()-1));
        DCHECK_EQ(dau_conv_settings_.num_output, sigma_shape.dim_size(sigma_shape.dims()-1));

        // create output tensors
        Tensor *grad_input = NULL;
        Tensor *grad_weights = NULL;
        Tensor *grad_mu1 = NULL;
        Tensor *grad_mu2 = NULL;
        Tensor *grad_sigma = NULL;
        OP_REQUIRES_OK(context, context->allocate_output(1, weights_shape, &grad_weights));
        OP_REQUIRES_OK(context, context->allocate_output(2, mu1_shape, &grad_mu1));
        OP_REQUIRES_OK(context, context->allocate_output(3, mu2_shape, &grad_mu2));
        OP_REQUIRES_OK(context, context->allocate_output(4, sigma_shape, &grad_sigma));

        std::vector<int> top_shape;
        top_shape.push_back(input_shape.dim_size(0));
        top_shape.push_back(weights->dim_size(1));
        top_shape.push_back(input_shape.dim_size(2));
        top_shape.push_back(input_shape.dim_size(3));

        TensorShape output_shape;
        for (int i = 0; i < top_shape.size(); i++) output_shape.AddDim(top_shape[i]);
        OP_REQUIRES_OK(context, context->allocate_output(0, input_shape, &grad_input));

        // grad is top error since it is the error of the output of the

        //Initializer does nothing, Tensorflow variables are initialized in python.
        NullDAUComponentInitializerTensorflow<Dtype> param_initializer;

        DAUKernelComputeTFGPU<Dtype> dau_kernel_compute(context);
        DAUKernelParamsTFGPU<Dtype> dau_kernel_params(context);
        DAUKernelOutputTFGPU<Dtype> dau_kernel_output(context);

        cublasHandle_t cublas_handle = 0;

        try {

            // get Tensorflow stream
            auto* stream = context->op_device_context()->stream();
            // obtain original CUDA's stream id from tensorflow stream
            CUstream default_tf_cuda_stream = reinterpret_cast<CUstream>(stream->implementation()->GpuStreamHack());

            // since we cannot get cublas handle from tensorflow we need to create one
            CUBLAS_CHECK(cublasCreate(&cublas_handle));
            // set cublas to use the same stream as for tensorflow
            CUBLAS_CHECK(cublasSetStream(cublas_handle, default_tf_cuda_stream));

            // initialize output with zeros before we start
            CUDA_CHECK(cudaMemsetAsync(TENSOR_DATA_PTR(grad_weights,Dtype),0, grad_weights->NumElements() * sizeof(Dtype), default_tf_cuda_stream));
            CUDA_CHECK(cudaMemsetAsync(TENSOR_DATA_PTR(grad_mu1, Dtype),0, grad_mu1->NumElements() * sizeof(Dtype), default_tf_cuda_stream));
            CUDA_CHECK(cudaMemsetAsync(TENSOR_DATA_PTR(grad_mu2, Dtype),0, grad_mu2->NumElements() * sizeof(Dtype), default_tf_cuda_stream));
            CUDA_CHECK(cudaMemsetAsync(TENSOR_DATA_PTR(grad_sigma, Dtype),0, grad_sigma->NumElements() * sizeof(Dtype), default_tf_cuda_stream));


            // next find out max offset value and optimize the size of kernels to accommodate all offsets
            {
                // Optimize the size of kernel (dau_conv_settings_.kernel_size) by matching it with the actual max offset
                // from the mu1/mu2 variables
                Dtype max_mu1 = 0, max_mu2 = 0;

                auto mu1__ = mu1->flat<Dtype>();
                int param_size = mu1__.size();

                if (TENSOR_DATA_PTR_CONST(mu1,Dtype) + param_size == TENSOR_DATA_PTR_CONST(mu2,Dtype))
                    // if mu1 and mu2 are in contiguous memory then we can call caffe_gpu_amax only once (to reduces overhead)
                    DAUConvNet::caffe_gpu_amax(param_size*2, TENSOR_DATA_PTR_CONST(mu1,Dtype), &max_mu1, cublas_handle);
                else {
                    DAUConvNet::caffe_gpu_amax(param_size, TENSOR_DATA_PTR_CONST(mu1, Dtype), &max_mu1, cublas_handle);
                    DAUConvNet::caffe_gpu_amax(param_size, TENSOR_DATA_PTR_CONST(mu2, Dtype), &max_mu2, cublas_handle);
                }
                Dtype actual_max_offset = std::max<Dtype>(std::abs(max_mu1), std::abs(max_mu2));

                // apply artificial limit to offset (based on user provided kernel_size and component_border_bound values)
                // (any offset larger then this limit will be artificially cliped by forward/backward call)
                actual_max_offset = std::min<Dtype>(actual_max_offset, dau_conv_settings.kernel_size - dau_conv_settings.component_border_bound);

                if (actual_max_offset <= 4) {
                    dau_conv_settings_.kernel_size = 2 * 4 + 1;
                } else if (actual_max_offset <= 8) {
                    dau_conv_settings_.kernel_size = 2 * 8 + 1;
                } else if (actual_max_offset <= 16) {
                    dau_conv_settings_.kernel_size = 2 * 16 + 1;
                } else if (actual_max_offset <= 32) {
                    dau_conv_settings_.kernel_size = 2 * 32 + 1;
                } else  {
                    OP_REQUIRES_OK_THROW_EX(context, Status(tensorflow::error::Code::INVALID_ARGUMENT,
                                                            "DAUConvGradOp ERROR: actual offsets larger then what CUDA memory allows (setup max_kernel_size and dau_unit_border_bound correctly to avoid this)!!"));
                }

                dau_conv_settings_.pad = (dau_conv_settings_.kernel_size-1)/2;

                if (actual_max_offset!=actual_max_offset) {
                    OP_REQUIRES_OK_THROW_EX(context, Status(tensorflow::error::Code::FAILED_PRECONDITION,
                                                            "DAUConvGradOp ERROR: got NaN value in offset (mu1,mu2) variable"));
                }

            }

            // then create tensorflow implementation of DAUConvLayer and set it to use tensorflow's stream and our cublas handle
            DAUConvLayerTensorflowGPU<Dtype> tf_layer(cublas_handle, context, this->unit_testing);

            tf_layer.set_default_cuda_stream(default_tf_cuda_stream);

            tf_layer.enable_forward(false);
            tf_layer.enable_backward(true);

            // prevent display of allocation size on each call (except when doing unit testing)
            tf_layer.enable_memalloc_info(this->unit_testing == true ? true : false);

            // we do not need to guard unit bounds since this is done in python
            // NOTE: clipping should not be done here since inputs (mu1,mu2,sigma) are not mutable !!
            tf_layer.enable_unit_bounds_guard(false);

            // if single dimensional kernel is reqested then we need to disable blur in second dimension
            tf_layer.set_single_dimensional_kernel(this->single_dim_kernel);
            tf_layer.set_forbid_positive_dim1(this->forbid_positive_dim1);

            // enable learning of sigma
            tf_layer.enable_sigma_learning(learn_sigma);

            tf_layer.InitializeFromInput(dau_conv_settings_, (Tensor *) weights, (Tensor *) mu1, (Tensor *) mu2,
                                         (Tensor *) sigma);

            tf_layer.InitializeGrad(dau_conv_settings_, grad_weights, grad_mu1, grad_mu2, grad_sigma);

            tf_layer.LayerSetUp(dau_conv_settings_, param_initializer, &dau_kernel_compute, &dau_kernel_params,
                                &dau_kernel_output, bottom_shape, number_units_ignore, in_train);

            tf_layer.Reshape(bottom_shape, top_shape);


            const Dtype *top_error = TENSOR_DATA_PTR_CONST(grad, Dtype);

            const Dtype *bottom_data = TENSOR_DATA_PTR_CONST(input, Dtype);

            Dtype *bottom_error = TENSOR_DATA_PTR(grad_input,Dtype);



            tf_layer.Backward_gpu(NULL, top_error, top_shape, true, bottom_data, bottom_error, bottom_shape,
                                  {true, true, this->single_dim_kernel == false ? true : false, learn_sigma, false});

            // multiply mu with learning rate if needed
            if (mu_learning_rate_factor != 1.0) {
                Dtype* mu1_data = TENSOR_DATA_PTR(grad_mu1, Dtype);
                Dtype* mu2_data = TENSOR_DATA_PTR(grad_mu2, Dtype);

                DAUConvNet::caffe_gpu_scal<Dtype>(grad_mu1->NumElements(), mu_learning_rate_factor, mu1_data, cublas_handle);
                DAUConvNet::caffe_gpu_scal<Dtype>(grad_mu2->NumElements(), mu_learning_rate_factor, mu2_data, cublas_handle);
            }

            CUDA_CHECK(cudaDeviceSynchronize());

        } catch (const DAUExceptionTF& ex) {
            std::cout << "ERROR: got TENSORFLOW status error in DAUConvOp" << std::endl;

        } catch (const DAUException& ex) {
            std::cout << "ERROR: got DAUException error in DAUConvOp" << std::endl;

            // report message to tensorflow
            context->CtxFailureWithWarning(Status(tensorflow::error::Code::INTERNAL, ex.what()));
        }
        if (cublas_handle != NULL)
            CUBLAS_CHECK(cublasDestroy(cublas_handle));
    }
private:
    cublasHandle_t cublas_handle_;
    DAUConvNet::DAUConvSettings dau_conv_settings;
    bool unit_testing;
    float mu_learning_rate_factor;
    int number_units_ignore;
    bool single_dim_kernel;
    bool forbid_positive_dim1;
};

REGISTER_KERNEL_BUILDER(Name("DAUConvGrad").Device(DEVICE_GPU), DAUConvGradOp<GPUDevice, float>);