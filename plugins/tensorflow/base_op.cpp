#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/platform/default/logging.h"
#include "tensorflow/core/framework/shape_inference.h"
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
  .Output("inner_product: float")
  .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
    shape_inference::ShapeHandle input_shape;
    TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 2, &input_shape));

    shape_inference::ShapeHandle weight_shape;
    TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 2, &weight_shape));
    
    shape_inference::DimensionHandle output_rows = c->Dim(weight_shape, 0);
  
    shape_inference::DimensionHandle input_rows = c->Dim(input_shape, 0);
    shape_inference::DimensionHandle weight_cols = c->Dim(weight_shape, 1);
    shape_inference::DimensionHandle merged;
    TF_RETURN_IF_ERROR(c->Merge(input_rows, weight_cols, &merged));

    c->set_output(0, c->Matrix(output_rows, 1));
    return Status::OK();
  });

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
    OP_REQUIRES_OK(context, context->allocate_temp(weights.dtype(),w_shape,&weights_non_const));
    auto weights_data = weights.flat<float>();
    auto nc_weights_data = weights_non_const.flat<float>();
    int n = weights_data.size();
    for(int i = 0; i < n; i++){
      nc_weights_data(i) = weights_data(i);
    }


    OP_REQUIRES_OK(context, context->allocate_temp(mu1.dtype(),mu1.shape(),&mu1_non_const));
    auto mu1_data = mu1.flat<float>();
    auto nc_mu1_data = mu1_non_const.flat<float>();
    for(int i = 0; i < mu1_data.size(); i++){
      nc_mu1_data(i) = mu1_data(i);
    }
 
    OP_REQUIRES_OK(context, context->allocate_temp(mu2.dtype(),mu2.shape(),&mu2_non_const));
    auto mu2_data = mu2.flat<float>();
    auto nc_mu2_data = mu2_non_const.flat<float>();
    for(int i = 0; i < mu2_data.size(); i++){
      nc_mu2_data(i) = mu2_data(i);
    }
 
    OP_REQUIRES_OK(context, context->allocate_temp(sigma.dtype(),sigma.shape(),&sigma_non_const));
    auto sigma_data = sigma.flat<float>();
    auto nc_sigma_data = sigma_non_const.flat<float>();
    for(int i = 0; i < sigma_data.size(); i++){
      nc_sigma_data(i) = sigma_data(i);
    }

    float* w_buf = static_cast<float*>(nc_weights_data.data());
    float* mu1_buf = static_cast<float*>(nc_mu1_data.data());
    float* mu2_buf = static_cast<float*>(nc_mu2_data.data());
    float* sigma_buf = static_cast<float*>(nc_sigma_data.data());


    const TensorShape& input_shape = input.shape();
    const TensorShape& weights_shape = weights.shape();
  

    //Initializer does nothing, input values were of type Filler in caffe
    // tensorflow variables are initialized in python.
    
    DAUComponentInitializerTensorflow<float> param_initializer(1,1,1);
    
    DAUConvNet::DAUConvSettings dau_conv_settings;
    DAUKernelComputeGPU<float>* dau_kernel_compute = new DAUKernelComputeGPU<float>();
    dau_kernel_compute->context_ = context;  
    DAUKernelParamsGPU<float>* dau_kernel_params = new DAUKernelParamsGPU<float>();
    dau_kernel_params->context_ = context;
    DAUKernelOutputGPU<float>* dau_kernel_output = new DAUKernelOutputGPU<float>();
    dau_kernel_output->context_ = context;

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
    const Device& tmp_dev = context->eigen_device<Device>();

    DAUConvLayerTensorflowGPU<float> tf_layer(handle,context);
    printf("Initializing from input\n");
    //set parameters from input tensors
    tf_layer.InitializeFromInput(dau_conv_settings, &weights_non_const,&mu1_non_const,&mu2_non_const,&sigma_non_const);
    
    //random input parameters for function that currently does nothing
    //THIS CRASHES
    param_initializer.InitializeParameters(dau_conv_settings,w_buf,mu1_buf,mu2_buf,sigma_buf,true,3,3,3,3,3,3,3);
    
    //tf_layer.LayerSetUp(dau_conv_settings, param_initializer, dau_kernel_compute, dau_kernel_params,dau_kernel_output, bottom_shape, in_train);
    //*/


    //BELLOW IS JUST A RANDOM SAMPLE OP

    DCHECK_EQ(input_shape.dims(), 2);
    DCHECK_EQ(input_shape.dim_size(1), 1);
    
    DCHECK_EQ(weights_shape.dims(), 2);
    DCHECK_EQ(input_shape.dim_size(0), weights_shape.dim_size(1));
    
    // create output shape
    TensorShape output_shape;
    output_shape.AddDim(weights_shape.dim_size(0));
    output_shape.AddDim(1);
    
    // create output tensor
    Tensor* output = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &output));

    
    // get the corresponding Eigen tensors for data access
    auto input_tensor = input.matrix<float>();
    auto weights_tensor = weights.matrix<float>();
    auto output_tensor = output->matrix<float>();

    for (int i = 0; i < output->shape().dim_size(0); i++) {
      output_tensor(i, 0) = 0;
      for (int j = 0; j < weights.shape().dim_size(1); j++) {
        output_tensor(i, 0) += weights_tensor(i, j)*input_tensor(j, 0);
      }
    }

    //output->CopyFrom(*output,TensorShape({1,output->shape().dim_size(0)}));
  }
};

#define REGISTER_CPU(T) REGISTER_KERNEL_BUILDER(Name("BaseOp").Device(DEVICE_CPU), BaseOpOp<CPUDevice, T>);

REGISTER_CPU(float);
//REGISTER_CPU(int32);

#ifdef GOOGLE_CUDA
#define REGISTER_GPU(T) \
REGISTER_KERNEL_BUILDER(Name("BaseOp").Device(DEVICE_CPU), BaseOpOp<GPUDevice, T>);

//type constraint is for constraining named attributes like "T"
//REGISTER_KERNEL_BUILDER(Name("BaseOp").Device(DEVICE_CPU).TypeConstraint<T>("T"), BaseOpOp<GPUDevice, T>);


REGISTER_GPU(float);
//REGISTER_GPU(int32);
#endif //google_cuda