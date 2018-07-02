#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/platform/default/logging.h"
#include "tensorflow/core/framework/shape_inference.h"
//#include "tensorflow/stream_executor/stream.h"
//#include "tensorflow/stream_executor/cuda/cuda_stream.h"
//#include "tensorflow/core/util/cuda_launch_config.h"
#include "dau_conv/base_dau_conv_layer.hpp"
#include "dau_conv_layer_tensorflow.hpp"
//#include "base_op.hpp"
//using DAUConvNet::DAUConvSettings;
using namespace tensorflow;

using CPUDevice = Eigen::ThreadPoolDevice;
using GPUDevice = Eigen::GpuDevice;


/*
     settings.num_output = 64;
    //num units per X and per Y
    settings.number_units.push_back(2);
    settings.number_units.push_back(2);
    settings.bias_term = true;
    settings.kernel_size = 9;
    settings.pad = 4;
    settings.stride = 1;
    settings.unit_normalization = true;
    settings.square_unit_normalization = true;
    settings.mean_iteration_step = 1;
    settings.sigma_iteration_step = 1;
    settings.component_border_bound = 4;
    settings.sigma_lower_bound = 0.3;
    settings.merge_iteration_step = 0;
    settings.merge_threshold = 1;
 */

//initialize with w, mu1, mu2, sigma
REGISTER_OP("BaseOp")
        .Input("input: float")
        .Input("weights: float")
        .Input("mu1: float")
        .Input("mu2: float")
        .Input("sigma: float")
        .Output("output: float")
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
/*.SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
  shape_inference::ShapeHandle input_shape;
  TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 4, &input_shape));

  shape_inference::ShapeHandle weight_shape;
  TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 4, &weight_shape));

  shape_inference::DimensionHandle output_rows = c->Dim(weight_shape, 0);

  shape_inference::DimensionHandle input_rows = c->Dim(input_shape, 0);
  shape_inference::DimensionHandle weight_cols = c->Dim(weight_shape, 1);
  shape_inference::DimensionHandle merged;
  TF_RETURN_IF_ERROR(c->Merge(input_rows, weight_cols, &merged));

  c->set_output(0, c->Matrix(output_rows, 1));
  return Status::OK();
});*/

template <typename Device, typename Dtype>
class BaseOpOp : public OpKernel {
public:
    explicit BaseOpOp(OpKernelConstruction* context) : OpKernel(context) {

    }

    void Compute(OpKernelContext* context) override {

        /*
        memory_status = cudaMemGetInfo(&free_bytes, &total_bytes);
        if(cudaSuccess != memory_status) printf("Error cuda %d\n", memory_status);
        free_db = (double) free_bytes;
        total_db = (double) total_bytes;
        printf("KernelParam buffers allocation Total %f, Free %f\n", total_db, free_db);
        */

        DCHECK_EQ(5, context->num_inputs());

        const Tensor* input;
        context->input("input", &input);
        const Tensor* weights;
        context->input("weights",&weights);
        const Tensor* mu1;
        context->input("mu1",&mu1);
        const Tensor* mu2;
        context->input("mu2",&mu2);
        const Tensor* sigma;
        context->input("sigma",&sigma);



        // allocate tensors for DAUKernelParams
        Tensor param_w;
        Tensor param_mu1;
        Tensor param_mu2;
        Tensor param_sigma;
        TensorShape param_shape({1, input->shape().dim_size(1), weights->shape().dim_size(1), weights->shape().dim_size(3)});
        //TensorShape param_shape({1,1,1,1});
        OP_REQUIRES_OK(context, context->allocate_temp(weights->dtype(),param_shape,&param_w));
        OP_REQUIRES_OK(context, context->allocate_temp(mu1->dtype(),param_shape,&param_mu1));
        OP_REQUIRES_OK(context, context->allocate_temp(mu2->dtype(),param_shape,&param_mu2));
        OP_REQUIRES_OK(context, context->allocate_temp(sigma->dtype(),param_shape,&param_sigma));

        Dtype* param_w_buf = static_cast<Dtype*>(param_w.flat<Dtype>().data());
        cudaError_t cuda_error_w = cudaMemset(param_w_buf,0, sizeof(Dtype)*param_w.NumElements());
        if(cuda_error_w != cudaSuccess) printf("Cuda error weights %d \n", cuda_error_w);
        Dtype* param_mu1_buf = static_cast<Dtype*>(param_mu1.flat<Dtype>().data());
        CUDA_CHECK(cudaMemset(param_mu1_buf,0, sizeof(Dtype)*param_mu1.NumElements()));
        Dtype* param_mu2_buf = static_cast<Dtype*>(param_mu2.flat<Dtype>().data());
        CUDA_CHECK(cudaMemset(param_mu2_buf,0, sizeof(Dtype)*param_mu2.NumElements()));
        Dtype* param_sigma_buf = static_cast<Dtype*>(param_sigma.flat<Dtype>().data());
        CUDA_CHECK(cudaMemset(param_sigma_buf,0, sizeof(Dtype)*param_sigma.NumElements()));


        const TensorShape& input_shape = input->shape();
        const TensorShape& weights_shape = weights->shape();


        //Initializer does nothing, input values were of type Filler in caffe
        // tensorflow variables are initialized in python.
        DAUComponentInitializerTensorflow<Dtype> param_initializer(1,1,1);

        DAUConvNet::DAUConvSettings dau_conv_settings;
        DAUKernelComputeGPU<Dtype> dau_kernel_compute(context);
        DAUKernelParamsGPU<Dtype>* dau_kernel_params = new DAUKernelParamsGPU<Dtype>();
        dau_kernel_params->context_ = context;
        DAUKernelOutputGPU<Dtype>* dau_kernel_output = new DAUKernelOutputGPU<Dtype>();
        dau_kernel_output->context_ = context;
        dau_kernel_params->initialize_params(param_w, param_mu1, param_mu2, param_sigma);

        // TODO check how you can tell if it is in training? maybe pass it as argument in
        // Op call?
        bool in_train = true;

        std::vector<int> bottom_shape;
        for(int i = 0; i < input_shape.dims(); i++){
            bottom_shape.push_back(input_shape.dim_size(i));
        }


        cublasHandle_t handle;
        cublasCreate(&handle);
        //const cudaStream_t* stream = CHECK_NOTNULL(reinterpret_cast<const cudaStream_t*>(context->op_device_context()
        //                                                                                -> stream()->implementation()
        //                                                                                ->CudaStreamMemberHack()) );
        //cublasSetStream(handle, stream);
        //TODO Get stream from context and add it to handle..


        DAUConvLayerTensorflowGPU<Dtype> tf_layer(handle,context);

        //set parameters from input tensors
        //tf_layer.InitializeFromInput(dau_conv_settings, &weights_non_const,&mu1_non_const,&mu2_non_const,&sigma_non_const);
        tf_layer.InitializeFromInput(dau_conv_settings, (Tensor*) weights,(Tensor*) mu1,(Tensor*) mu2,(Tensor*) sigma);

        tf_layer.LayerSetUp(dau_conv_settings, param_initializer, &dau_kernel_compute, dau_kernel_params,dau_kernel_output, bottom_shape, in_train);

        //TensorShape top_tensor_shape({input_shape.dim_size(0), weight_shape.dim_size(1), input_shape.dim_size(2), input_shape.dim_size(3)});
        std::vector<int> top_shape;
        top_shape.push_back(input_shape.dim_size(0));
        top_shape.push_back(weights->dim_size(1));
        top_shape.push_back(input_shape.dim_size(2));
        top_shape.push_back(input_shape.dim_size(3));

        tf_layer.Reshape(bottom_shape, top_shape);

        //tf_layer forward_gpu implement..

        TensorShape output_shape;
        for(int i = 0; i< top_shape.size(); i++) output_shape.AddDim(top_shape[i]);

        Tensor* output;
        OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &output));
        auto out_data = output->flat<Dtype>();
        Dtype* top_data = static_cast<Dtype*>(out_data.data());

        auto input_data = input->flat<Dtype>();
        const Dtype* bottom_data = static_cast<const Dtype*>(input_data.data());

        tf_layer.Forward_gpu(bottom_data, bottom_shape, top_data, top_shape);

    }
};

#define REGISTER_CPU(T) REGISTER_KERNEL_BUILDER(Name("BaseOp").Device(DEVICE_CPU), BaseOpOp<CPUDevice, T>);

//REGISTER_CPU(float);
//REGISTER_CPU(int32);

#ifdef GOOGLE_CUDA
#define REGISTER_GPU(T) \
REGISTER_KERNEL_BUILDER(Name("BaseOp").Device(DEVICE_GPU), BaseOpOp<GPUDevice, T>);

//type constraint is for constraining named attributes like "T"
//REGISTER_KERNEL_BUILDER(Name("BaseOp").Device(DEVICE_CPU).TypeConstraint<T>("T"), BaseOpOp<GPUDevice, T>);


REGISTER_GPU(float);
//REGISTER_GPU(int32);
#endif //google_cuda