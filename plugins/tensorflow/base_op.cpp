#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/platform/default/logging.h"
#include "tensorflow/core/framework/shape_inference.h"
//#include "tensorflow/core/util/cuda_kernel_helper.h"
#include "dau_conv/base_dau_conv_layer.hpp"
#include "dau_conv_layer_tensorflow.hpp"
//#include "base_op.hpp"
//using DAUConvNet::DAUConvSettings;
using namespace tensorflow;

using CPUDevice = Eigen::ThreadPoolDevice;
using GPUDevice = Eigen::GpuDevice;


//initialize with w, mu1, mu2, sigma
REGISTER_OP("BaseOp")
  .Input("input: float")
  .Input("weights: float")
  .Input("mu1: float")
  .Input("mu2: float")
  .Input("sigma: float")
  .Output("inner_product: float");
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

// D is device
template <typename Device, typename Dtype>
class BaseOpOp : public OpKernel {
public:
  explicit BaseOpOp(OpKernelConstruction* context) : OpKernel(context) {
    
  }
  
  void Compute(OpKernelContext* context) override {
    
    DCHECK_EQ(2, context->num_inputs());
    
    const Tensor& input = context->input(0);
    const Tensor& weights = context->input(1);
    const Tensor& mu1 = context->input(2);
    const Tensor& mu2 = context->input(3);
    const Tensor& sigma = context->input(4);


    // cant use copy from, uses shallow copy, underlying data is still const
    // maybe change to const Tensor* in functions
    Tensor weights_non_const;
    Tensor mu1_non_const;
    Tensor mu2_non_const;
    Tensor sigma_non_const;
 
    const TensorShape& w_shape = weights.shape();
    printf("Weights type: %d\n", weights.dtype());
    OP_REQUIRES_OK(context, context->allocate_temp(weights.dtype(),w_shape,&weights_non_const));
    auto weights_data = weights.flat<Dtype>();
    auto nc_weights_data = weights_non_const.flat<Dtype>();
    int n = weights_data.size();
    for(int i = 0; i < n; i++){
      nc_weights_data(i) = weights_data(i);
    }


    OP_REQUIRES_OK(context, context->allocate_temp(mu1.dtype(),mu1.shape(),&mu1_non_const));
    auto mu1_data = mu1.flat<Dtype>();
    auto nc_mu1_data = mu1_non_const.flat<Dtype>();
    for(int i = 0; i < mu1_data.size(); i++){
      nc_mu1_data(i) = mu1_data(i);
    }
 
    OP_REQUIRES_OK(context, context->allocate_temp(mu2.dtype(),mu2.shape(),&mu2_non_const));
    auto mu2_data = mu2.flat<Dtype>();
    auto nc_mu2_data = mu2_non_const.flat<Dtype>();
    for(int i = 0; i < mu2_data.size(); i++){
      nc_mu2_data(i) = mu2_data(i);
    }
 
    OP_REQUIRES_OK(context, context->allocate_temp(sigma.dtype(),sigma.shape(),&sigma_non_const));
    auto sigma_data = sigma.flat<Dtype>();
    auto nc_sigma_data = sigma_non_const.flat<Dtype>();
    for(int i = 0; i < sigma_data.size(); i++){
      nc_sigma_data(i) = sigma_data(i);
    }

    Dtype* w_buf = static_cast<Dtype*>(nc_weights_data.data());
    Dtype* mu1_buf = static_cast<Dtype*>(nc_mu1_data.data());
    Dtype* mu2_buf = static_cast<Dtype*>(nc_mu2_data.data());
    Dtype* sigma_buf = static_cast<Dtype*>(nc_sigma_data.data());


    // allocate tensors for DAUKernelParams
    Tensor param_w;
    Tensor param_mu1;
    Tensor param_mu2;
    Tensor param_sigma;
    TensorShape param_shape({1, 1, 1, 1});
    OP_REQUIRES_OK(context, context->allocate_temp(weights.dtype(),param_shape,&param_w));
    OP_REQUIRES_OK(context, context->allocate_temp(mu1.dtype(),param_shape,&param_mu1));
    OP_REQUIRES_OK(context, context->allocate_temp(mu2.dtype(),param_shape,&param_mu2));
    OP_REQUIRES_OK(context, context->allocate_temp(sigma.dtype(),param_shape,&param_sigma));



    const TensorShape& input_shape = input.shape();
    const TensorShape& weights_shape = weights.shape();
  

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
    //checkError(cublasCreate(&handle),"cublasCreate() error\n");
    //get gpu/cpu device
    //const GPUDevice& tmp_dev = context->eigen_device<GPUDevice>();
    //cudaStream_t tmp_stream = tmp_dev.stream();
    //const cudaStream_t* stream = CHECK_NOTNULL(reinterpret_cast<const cudaStream_t*>(context->op_device_context()
    //                                                                                -> stream()->implementation()
    //                                                                                ->CudaStreamMemberHack()) );
    // NEED TO INCLUDE tensorflow/core/util/cuda_kernel_helper.h but it is not well defined..
    //const cudaStream_t& stream = GetCudaStream(context);
    //cublasSetStream(handle, tmp_dev.stream());
    //TODO Get stream from context and add it to handle..


    DAUConvLayerTensorflowGPU<Dtype> tf_layer(handle,context);
    printf("Initializing from input\n");
    //set parameters from input tensors
    tf_layer.InitializeFromInput(dau_conv_settings, &weights_non_const,&mu1_non_const,&mu2_non_const,&sigma_non_const);
    
    //random input parameters for function that currently does nothing
    
    tf_layer.LayerSetUp(dau_conv_settings, param_initializer, &dau_kernel_compute, dau_kernel_params,dau_kernel_output, bottom_shape, in_train);
    //*/
    printf("Set up layer\n");

    //TensorShape top_tensor_shape({input_shape.dim_size(0), weight_shape.dim_size(1), input_shape.dim_size(2), input_shape.dim_size(3)});
    std::vector<int> top_shape;
    top_shape.push_back(input_shape.dim_size(0));
    top_shape.push_back(weights.dim_size(1));
    top_shape.push_back(input_shape.dim_size(2));
    top_shape.push_back(input_shape.dim_size(3));

    tf_layer.Reshape(bottom_shape, top_shape);
    printf("Reshape done \n");
    
    //tf_layer forward_gpu implement..

    printf("Set shape\n");
    TensorShape output_shape;
    for(int i = 0; i< top_shape.size(); i++) output_shape.AddDim(top_shape[i]);

    printf("Allocate output\n");
    Tensor* output;
    OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &output));
    printf("Allocated output\n");
    auto out_data = output->flat<Dtype>();
    Dtype* top_data = static_cast<Dtype*>(out_data.data());
    //for(int i = 0; i < out_data.size(); i++) out_data(i) = 1.0;
    printf("Allocate output2\n");

    auto input_data = input.flat<Dtype>();
    const Dtype* bottom_data = static_cast<const Dtype*>(input_data.data());

    //crashes at cudaEventRecord(memset_top, paralel_streams[0]) line 76
    tf_layer.Forward_gpu(bottom_data, bottom_shape, top_data, top_shape);


  }
};

#define REGISTER_CPU(T) REGISTER_KERNEL_BUILDER(Name("BaseOp").Device(DEVICE_CPU), BaseOpOp<CPUDevice, T>);

REGISTER_CPU(float);
//REGISTER_CPU(int32);

#ifdef GOOGLE_CUDA
#define REGISTER_GPU(T) \
REGISTER_KERNEL_BUILDER(Name("BaseOp").Device(DEVICE_GPU), BaseOpOp<GPUDevice, T>);

//type constraint is for constraining named attributes like "T"
//REGISTER_KERNEL_BUILDER(Name("BaseOp").Device(DEVICE_CPU).TypeConstraint<T>("T"), BaseOpOp<GPUDevice, T>);


REGISTER_GPU(float);
//REGISTER_GPU(int32);
#endif //google_cuda