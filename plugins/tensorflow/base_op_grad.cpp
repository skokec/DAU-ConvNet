#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"

using namespace tensorflow;

REGISTER_OP("BaseOpGrad")
  .Input("grad: float32")
  .Input("input: float32")
  .Input("weights: float32")
  .Output("grad_input: float32")
  .Output("grad_weights: float32");

class BaseOpGradOp : public OpKernel {
public:
  explicit BaseOpGradOp(OpKernelConstruction* context) : OpKernel(context) {
    
  }
  
  void Compute(OpKernelContext* context) override {
    
    DCHECK_EQ(3, context->num_inputs());

    const Tensor& grad = context->input(0);
    const Tensor& input = context->input(1);
    const Tensor& weights = context->input(2);
    
    // create input shape (inferred from the additional attribute `n`)
    TensorShape input_shape = input.shape();
    TensorShape weights_shape = weights.shape();
    
    DCHECK_EQ(input_shape.dim_size(0), weights_shape.dim_size(1));
    DCHECK_EQ(weights_shape.dim_size(0), grad.shape().dim_size(0));
    
    // create output tensors
    Tensor* grad_input = NULL;
    Tensor* grad_weights = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, input_shape, &grad_input));
    OP_REQUIRES_OK(context, context->allocate_output(1, weights_shape, &grad_weights));
    
    // get the Eigen tensors for data access
    auto grad_tensor = grad.matrix<float>();
    auto weights_tensor = weights.matrix<float>();
    auto input_tensor = input.matrix<float>();
    auto grad_input_tensor = grad_input->matrix<float>();
    auto grad_weights_tensor = grad_weights->matrix<float>();
    
    // matmul as placeholder
    for (int i = 0; i < weights_shape.dim_size(0); i++) {
      grad_input_tensor(i, 0) = 0;
      for (int j = 0; j < grad.shape().dim_size(0); j++) {
        grad_input_tensor(i, 0) += grad_tensor(j, 0)*weights_tensor(j, i);
      }
    }
    
    for (int i = 0; i < weights_shape.dim_size(0); i++) {
      for (int j = 0; j < weights_shape.dim_size(1); j++) {
        grad_weights_tensor(i, j) = grad_tensor(i, 0)*input_tensor(j, 0);;
      }
    }
  }
};

REGISTER_KERNEL_BUILDER(Name("BaseOpGrad").Device(DEVICE_CPU), BaseOpGradOp);