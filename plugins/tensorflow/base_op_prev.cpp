#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/platform/default/logging.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "dau_conv/base_dau_conv_layer.hpp"
#include "dau_conv_layer_tensorflow.hpp"

using namespace tensorflow;

REGISTER_OP("BaseOp")
  .Input("input: float")
  .Input("weights: float")
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

class BaseOpOp : public OpKernel {
public:
  explicit BaseOpOp(OpKernelConstruction* context) : OpKernel(context) {
    
  }
  
  void Compute(OpKernelContext* context) override {
    
    DCHECK_EQ(2, context->num_inputs());
    
    const Tensor& input = context->input(0);
    const Tensor& weights = context->input(1);
    
    const TensorShape& input_shape = input.shape();
    const TensorShape& weights_shape = weights.shape();
  
    //basic tensor allocation
    Tensor* test = new Tensor(DT_FLOAT, input_shape);
    const std::vector<int>& t_shape = {1,2,3,4};
    TensorShape tt_shape;
    for(int dm: t_shape){
      tt_shape.AddDim(dm);
    }
    Tensor* test2 = new Tensor(DT_FLOAT, tt_shape);
    DAUConvSettings dau_conv_setings;
    DAUKernelComputeGPU<float>* dau_kernel_compute = new DAUKernelComputeGPU<float>();
    DAUKernelParamsGPU<float>* dau_kernel_params = new DAUKernelParamsGPU<float>();
    DAUKernelOutputGPU<float>* dau_kernel_output = new DAUKernelOutputGPU<float>();
    

    // TODO IMPLEMENT DUMMY GRAD FUNCTIONS SO YOU DONT GET PURE VIRTUAL FUNCTIONS ERROR - done
    cublasHandle_t handle;
    checkError(cublasCreate(&handle),"cublasCreate() error\n");
    DAUConvLayerTensorflowGPU<float> tf_layer(handle);
    
    // TODO GET ALL THE NECESSARY INPUTS FOR LAYER SETUP
    //tf_layer.LayerSetUp(context, dau_conv_settings, );



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
  }
};

REGISTER_KERNEL_BUILDER(Name("BaseOp").Device(DEVICE_CPU), BaseOpOp);
