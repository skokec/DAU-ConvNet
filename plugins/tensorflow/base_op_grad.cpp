#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"

using namespace tensorflow;

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
  .Output("grad_sigma: float32"); //

class BaseOpGradOp : public OpKernel {
public:
  explicit BaseOpGradOp(OpKernelConstruction* context) : OpKernel(context) {
    
  }
  
  void Compute(OpKernelContext* context) override {
    
    DCHECK_EQ(6, context->num_inputs());

    const Tensor& grad = context->input(0);
    const Tensor& input = context->input(1);
    const Tensor& weights = context->input(2);
    const Tensor& mu1 = context->input(3);
    const Tensor& mu2 = context->input(4);
    const Tensor& sigma = context->input(5);

    // create input shape (inferred from the additional attribute `n`)
    TensorShape input_shape = input.shape();
    TensorShape weights_shape = weights.shape();
    
    DCHECK_EQ(input_shape.dim_size(0), weights_shape.dim_size(1));
    DCHECK_EQ(weights_shape.dim_size(0), grad.shape().dim_size(0));
    
    // create output tensors
    Tensor* grad_input = NULL;
    Tensor* grad_weights = NULL;
    Tensor* grad_mu1 = NULL;
    Tensor* grad_mu2 = NULL;
    Tensor* grad_sigma = NULL;

    OP_REQUIRES_OK(context, context->allocate_output(0, input_shape, &grad_input));
    OP_REQUIRES_OK(context, context->allocate_output(1, weights_shape, &grad_weights));
    OP_REQUIRES_OK(context, context->allocate_output(2, weights_shape, &grad_mu1));
    OP_REQUIRES_OK(context, context->allocate_output(3, weights_shape, &grad_mu2));
    OP_REQUIRES_OK(context, context->allocate_output(4, weights_shape, &grad_sigma));

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

REGISTER_KERNEL_BUILDER(Name("BaseOpGrad").Device(DEVICE_GPU), BaseOpGradOp);