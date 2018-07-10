#include <algorithm>
#include <vector>


#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"


using namespace std;

#include <opencv2/opencv.hpp>
#include <opencv2/gpu/gpu.hpp>


#include <dau_conv_layer_tensorflow.hpp>
#include <dau_conv/util/math_functions.hpp>

using namespace tensorflow;

template <typename Dtype>
void DAUConvLayerTensorflowGPU<Dtype>::InitializeGrad(DAUConvSettings& settings, Tensor* w_grad, Tensor* mu1_grad, Tensor* mu2_grad, Tensor* sigma_grad){

    this->param_buffer_w_grad = w_grad;
    this->param_buffer_mu1_grad = mu1_grad;
    this->param_buffer_mu2_grad = mu2_grad;
    this->param_buffer_sigma_grad = sigma_grad;

    if(settings.bias_term){
        Tensor* bias_temp = new Tensor();
        Tensor& tmp_ten = *bias_temp;
        OP_REQUIRES_OK(this->context_, this->context_->allocate_temp(DT_FLOAT, w_grad->shape(), &tmp_ten));
        this->param_buffer_bias_grad = bias_temp;
    }


}


template <typename Dtype>
void DAUConvLayerTensorflowGPU<Dtype>::InitializeFromInput(DAUConvSettings& settings, Tensor* w, Tensor* mu1, Tensor* mu2, Tensor* sigma){
    //Set the layer parameters from input tensors

    this->param_buffer_w_ = w;
    this->param_buffer_mu1_ = mu1;
    this->param_buffer_mu2_ = mu2;
    this->param_buffer_sigma_ = sigma;

    if(settings.bias_term){
        Tensor* bias_temp = new Tensor();
        Tensor& tmp_ten = *bias_temp;
        OP_REQUIRES_OK(this->context_, this->context_->allocate_temp(DT_FLOAT, w->shape(), &tmp_ten));
        if(this->param_buffer_bias_) delete this->param_buffer_bias_;
        this->param_buffer_bias_ = bias_temp;
    }
    
}

template <typename Dtype>
DAUConvLayerTensorflowGPU<Dtype>::~DAUConvLayerTensorflowGPU(){
    this->deallocate_workspace_mem();
    if(this->bwd_gradients_) delete this->bwd_gradients_;
    if(this->interm_buffer_) delete this->interm_buffer_;
    if(this->tmp_param_buffer_) delete this->tmp_param_buffer_;
    if(this->col_buffer_) delete this->col_buffer_;
    if(this->bias_multiplier_)delete this->bias_multiplier_;

    cublasDestroy(this->cublasHandle);
}

template <typename Dtype>
void* DAUConvLayerTensorflowGPU<Dtype>::allocate_workspace_mem(size_t bytes) {
    // deallocate existing workspace memory
    DataType tensorflow_dtype = DataTypeToEnum<Dtype>::v();

    size_t total_bytes = (bytes/ sizeof(Dtype) + 1)*sizeof(Dtype);
    deallocate_workspace_mem();
    Tensor* tmp_ten = new Tensor();

    Status can_allocate = this->context_->allocate_temp(tensorflow_dtype, TensorShape({total_bytes/sizeof(Dtype)}), tmp_ten);
    if(!TF_PREDICT_TRUE(can_allocate.ok())){
        this->own_workspace_data = NULL;
        return this->own_workspace_data;
    }
    this->own_workspace_tensor = tmp_ten;
    this->own_workspace_data = TENSOR_DATA_PTR(tmp_ten,Dtype);
    return this->own_workspace_data;
}

template <typename Dtype>
void DAUConvLayerTensorflowGPU<Dtype>::deallocate_workspace_mem() {
    if (this->own_workspace_data != NULL){
        delete this->own_workspace_tensor;
        this->own_workspace_data = NULL;
    }
}

template <typename Dtype>
void DAUConvLayerTensorflowGPU<Dtype>::reshape_params(const vector<int>& shape) {
    // reshape DAU parameters
    //ONLY CHECK IF THE SHAPE IS CORRECT

    TensorShape tmp_shape;
    for(int dm: shape){
      tmp_shape.AddDim(dm);
    }

    Tensor* orig_ten_w = (Tensor*) this->param_buffer_w_;
    CHECK(orig_ten_w->shape().IsSameSize(tmp_shape));

    Tensor* orig_ten_mu1 = (Tensor*) this->param_buffer_mu1_;
    CHECK(orig_ten_mu1->shape().IsSameSize(tmp_shape));

    Tensor* orig_ten_mu2 = (Tensor*) this->param_buffer_mu2_;
    CHECK(orig_ten_mu2->shape().IsSameSize(tmp_shape));

    Tensor* orig_ten_sigma = (Tensor*) this->param_buffer_sigma_;
    CHECK(orig_ten_sigma->shape().IsSameSize(tmp_shape));

    // Check gradients for correct sizes (if they are not NULL)
    if(this->param_buffer_w_grad){
        Tensor* orig_ten = (Tensor*) this->param_buffer_w_grad;
        CHECK(orig_ten->shape().IsSameSize(tmp_shape));
    }
    if(this->param_buffer_mu1_grad){
        Tensor* orig_ten = (Tensor*) this->param_buffer_mu1_grad;
        CHECK(orig_ten->shape().IsSameSize(tmp_shape));
    }
    if(this->param_buffer_mu2_grad){
        Tensor* orig_ten = (Tensor*) this->param_buffer_mu2_grad;
        CHECK(orig_ten->shape().IsSameSize(tmp_shape));
    }
    if(this->param_buffer_sigma_grad){
        Tensor* orig_ten = (Tensor*) this->param_buffer_sigma_grad;
        CHECK(orig_ten->shape().IsSameSize(tmp_shape));
    }


    if (this->bias_term_) {
        Tensor* orig_ten_bias = (Tensor*) this->param_buffer_bias_;
        CHECK(orig_ten_bias->shape().IsSameSize(tmp_shape));

    }
}

template <typename Dtype>
void DAUConvLayerTensorflowGPU<Dtype>::LayerSetUp(const DAUConvSettings& settings,
                                             const BaseDAUComponentInitializer<Dtype>& param_initializer,
                                             BaseDAUKernelCompute<Dtype>* kernel_compute,
                                             BaseDAUKernelParams<Dtype>* kernel_param,
                                             BaseDAUKernelOutput<Dtype>* kernel_output,
                                             const vector<int>& bottom_shape, int num_dau_units_ignore, bool in_train) {

    // ensure prod(settings.number_units) must be dividable by ALLOWED_UNITS_GROUP
    int num_units_all = 1;
    for (int i = 0; i < settings.number_units.size(); i++)
        num_units_all *= settings.number_units[i];

    DCHECK_EQ(num_units_all % this->ALLOWED_UNITS_GROUP, 0);

    // call parent to compute all the shape variables and call initialize of parameter shape

    BaseDAUConvLayer<Dtype>::LayerSetUp(settings, param_initializer,
                                        kernel_compute, kernel_param, kernel_output,
                                        bottom_shape, in_train);

    // for tensorflow we expect to get memory for all units (i.e. prod(settings.number_units) must be dividable by ALLOWED_UNITS_GROUP)
    // but if we need only some then we need to setup this here
    // NOTE: this must be done after call to parent LayerSetUp where 'this->num_units_ignore = 0' is called
    this->num_units_ignore = num_dau_units_ignore;

    // we use actual (learnable) sigma parameter when computing kernels so connect that param with the sigma for aggregation
    DAUKernelParamsTFGPU<Dtype>* kernel_param_tf_gpu = reinterpret_cast<DAUKernelParamsTFGPU<Dtype>* >(kernel_param);
    if(kernel_param_tf_gpu->sigma_) delete kernel_param_tf_gpu->sigma_;
    kernel_param_tf_gpu->sigma_ = (Tensor*) this->param_buffer_sigma_;
    
}

template <typename Dtype>
vector<int> DAUConvLayerTensorflowGPU<Dtype>::Reshape(const vector<int>& bottom_shape, const vector<int>& top_shape) {
    // call parent to compute all the shape variables

    //Get tensorflow type from Dtype
    DataType tensorflow_dtype = DataTypeToEnum<Dtype>::v();

    const vector<int> new_top_shape = BaseDAUConvLayer<Dtype>::Reshape(bottom_shape, top_shape);

    const int max_width = std::max(this->width_out_,this->width_);
    const int max_height = std::max(this->height_out_,this->height_);

    // Set up the all ones "bias multiplier" for adding biases
    if (this->bias_term_) {
        if(!this->bias_multiplier_){
            Tensor* tmp_ten = new Tensor();
            OP_REQUIRES_OK_BREAK(this->context_,this->context_->allocate_temp(tensorflow_dtype, TensorShape({1,this->height_out_*this->width_out_}), tmp_ten));
            this->bias_multiplier_ = tmp_ten;
        }else{
            CHECK(this->bias_multiplier_->shape().IsSameSize(TensorShape({1, this->height_out_ * this->width_out_})));
        }
    }

    if (this->enabled_fwd_op || this->enabled_bwd_op) {
        // make sure col_buffer is big enough
        if(!this->col_buffer_){
            Tensor* tmp_ten = new Tensor();
            OP_REQUIRES_OK_BREAK(this->context_, this->context_->allocate_temp(tensorflow_dtype, TensorShape({this->aggregation.kernel_h_, this->aggregation.kernel_w_, this->height_, this->width_}), tmp_ten));
            this->col_buffer_= tmp_ten;
        }else{
            CHECK(this->col_buffer_->shape().IsSameSize(TensorShape({this->aggregation.kernel_h_, this->aggregation.kernel_w_, this->height_, this->width_})));
        }

        int interm_buf_size = 0;
        if (this->enabled_fwd_op) interm_buf_size = std::max(interm_buf_size, this->conv_in_channels_);
        if (this->enabled_bwd_op) interm_buf_size = std::max(interm_buf_size, this->conv_out_channels_ * this->NUM_K);

        // use inter buffer for both fwd and bwd passes so allocate buffer with suitable size for both
        if(!this->interm_buffer_){
            Tensor* tmp_ten = new Tensor();
            OP_REQUIRES_OK_BREAK(this->context_, this->context_->allocate_temp(tensorflow_dtype, TensorShape({this->batch_num_, interm_buf_size, max_height, max_width}), tmp_ten));
            this->interm_buffer_= tmp_ten;
        }else{
            CHECK(this->interm_buffer_->shape().IsSameSize(TensorShape({this->batch_num_, interm_buf_size, max_height, max_width})));
        }
    }
    if (this->enabled_bwd_op) {
        if (!this->bwd_gradients_) {
            //bwd gradients buffer not defined
            Tensor *tmp_ten = new Tensor();
            OP_REQUIRES_OK_BREAK(this->context_, this->context_->allocate_temp(tensorflow_dtype, TensorShape(
                    {this->NUM_K, this->conv_in_channels_, this->units_per_channel, this->conv_out_channels_}),
                                                                               tmp_ten));
            this->bwd_gradients_ = tmp_ten;
        } else {
            CHECK(this->bwd_gradients_->shape().IsSameSize(TensorShape(
                    {this->NUM_K, this->conv_in_channels_, this->units_per_channel, this->conv_out_channels_})));
        }

    // temporary buffer used during the back-propagation of the error where we rotate mu1 and mu2
        if(!this->tmp_param_buffer_){
            Tensor* tmp_ten = new Tensor();
            OP_REQUIRES_OK_BREAK(this->context_, this->context_->allocate_temp(tensorflow_dtype, TensorShape({2, this->conv_in_channels_, this->units_per_channel, this->conv_out_channels_}), tmp_ten));
            this->tmp_param_buffer_= tmp_ten;
        }else{
            CHECK(this->tmp_param_buffer_->shape().IsSameSize(TensorShape({2, this->conv_in_channels_, this->units_per_channel, this->conv_out_channels_})));
        }
    }

    
    return new_top_shape;

}

template <typename Dtype>
bool DAUConvLayerTensorflowGPU<Dtype>::update_prefiltering_kernels(cudaStream_t stream) {
    return BaseDAUConvLayer<Dtype>::update_prefiltering_kernels(stream);
}


template DAUConvLayerTensorflowGPU<double>::~DAUConvLayerTensorflowGPU();
template DAUConvLayerTensorflowGPU<float>::~DAUConvLayerTensorflowGPU();

template void* DAUConvLayerTensorflowGPU<double>::allocate_workspace_mem(size_t bytes);
template void* DAUConvLayerTensorflowGPU<float>::allocate_workspace_mem(size_t bytes);


template void DAUConvLayerTensorflowGPU<double>::deallocate_workspace_mem();
template void DAUConvLayerTensorflowGPU<float>::deallocate_workspace_mem();

template vector<int> DAUConvLayerTensorflowGPU<double>::Reshape(const vector<int>& bottom_shape, const vector<int>& top);
template vector<int> DAUConvLayerTensorflowGPU<float>::Reshape(const vector<int>& bottom_shape, const vector<int>& top);

template void DAUConvLayerTensorflowGPU<float>::LayerSetUp(const DAUConvSettings& settings, const BaseDAUComponentInitializer<float>& param_initializer,
                                                      BaseDAUKernelCompute<float>* kernel_compute, BaseDAUKernelParams<float>* kernel_param, BaseDAUKernelOutput<float>* kernel_output,
                                                      const vector<int>& bottom_shape, int num_dau_units_ignore, bool in_train);
template void DAUConvLayerTensorflowGPU<double>::LayerSetUp(const DAUConvSettings& settings, const BaseDAUComponentInitializer<double>& param_initializer,
                                                       BaseDAUKernelCompute<double>* kernel_compute, BaseDAUKernelParams<double>* kernel_param, BaseDAUKernelOutput<double>* kernel_output,
                                                       const vector<int>& bottom_shape, int num_dau_units_ignore, bool in_train);

template void NullDAUComponentInitializerTensorflow<float>::InitializeParameters(const DAUConvSettings& settings, float* w, float* mu1, float* mu2, float* sigma, bool is_gpu_ptr,
                                                               int num_units_per_x, int num_units_per_y, int num_units_ignore,
                                                               int conv_in_channels, int conv_out_channels, int kernel_h, int kernel_w) const;
template void NullDAUComponentInitializerTensorflow<double>::InitializeParameters(const DAUConvSettings& settings, double* w, double* mu1, double* mu2, double* sigma, bool is_gpu_ptr,
                                                               int num_units_per_x, int num_units_per_y, int num_units_ignore,
                                                               int conv_in_channels, int conv_out_channels, int kernel_h, int kernel_w) const;

template <typename Dtype>
DAUKernelComputeTF<Dtype>::~DAUKernelComputeTF(){
    for (int i = 0; i < this->kernels_buffers_.size(); i++)
        delete this->kernels_buffers_[i];

    for (int i = 0; i < this->param_buffers_.size(); i++)
        delete this->param_buffers_[i];
}

template <typename Dtype>
DAUKernelParamsTF<Dtype>::~DAUKernelParamsTF(){
    if(this->weight_){delete this->weight_; this->weight_ = NULL;}
    if(this->mu1_){delete this->mu1_; this->mu1_ = NULL;}
    if(this->mu2_){delete this->mu2_; this->mu2_ = NULL;}
}

template <typename Dtype>
DAUKernelOutputTF<Dtype>::~DAUKernelOutputTF(){
    if(this->weight_){ delete this->weight_; this->weight_ = NULL;}
    if(this->d_error_){ delete this->d_error_; this->d_error_ = NULL;}
    if(this->d_params_){delete this->d_params_; this->d_params_ = NULL;}
}




template <typename Dtype>
void DAUKernelComputeTF<Dtype>::reshape(int num_in_channels, int num_out_channels, int num_gauss, int kernel_h, int kernel_w){

    DataType tensorflow_dtype = DataTypeToEnum<Dtype>::v();

    this->num_in_channels = num_in_channels;
    this->num_out_channels = num_out_channels;
    this->num_gauss = num_gauss;
    this->kernel_h = kernel_h;
    this->kernel_w = kernel_w;

    // allocate and prepare temporary buffers for kernels
    for (int i = 0; i < this->kernels_buffers_.size(); i++)
        delete this->kernels_buffers_[i];

    this->kernels_buffers_.resize(5);
    for (int i = 0; i < 5; i++){
        Tensor* tmp_ten = new Tensor();
        OP_REQUIRES_OK(this->context_, this->context_->allocate_temp(tensorflow_dtype, TensorShape({num_in_channels, num_gauss, num_out_channels, kernel_h*kernel_w}), tmp_ten));
        this->kernels_buffers_[i] = tmp_ten;

    }


    // allocate and prepare temporary buffers for parameters
    for (int i = 0; i < this->param_buffers_.size(); i++)
        delete this->param_buffers_[i];

    this->param_buffers_.resize(7);
    for (int i = 0; i < 7; i++){
        Tensor* tmp_ten = new Tensor();
        OP_REQUIRES_OK(this->context_, this->context_->allocate_temp(tensorflow_dtype, TensorShape({1, num_in_channels, num_gauss, num_out_channels}), tmp_ten));
        this->param_buffers_[i] = tmp_ten;
    }
    // pre-computed offset indexes for batched sums (when using caffe_gpu_sum)
    this->create_precompute_index(num_in_channels * num_gauss * num_out_channels, kernel_h * kernel_w);

}

template <typename Dtype>
void DAUKernelParamsTF<Dtype>::initialize_params(Tensor w, Tensor mu1, Tensor mu2, Tensor sigma){
    this->weight_ = &w;
    this->mu1_ = &mu1;
    this->mu2_ = &mu2;
    this->sigma_ = &sigma;

}

template <typename Dtype>
void DAUKernelParamsTF<Dtype>::reshape(int num_in_channels, int num_out_channels, int num_gauss) {

    DataType tensorflow_dtype = DataTypeToEnum<Dtype>::v();

    int reshape_total_elements = num_in_channels*num_out_channels*num_gauss;
    
    if (!this->weight_ || (this->weight_)->NumElements() != reshape_total_elements){
        Tensor* tmp_ten = new Tensor();
        OP_REQUIRES_OK(this->context_, this->context_->allocate_temp(tensorflow_dtype, TensorShape({1,num_in_channels, num_gauss, num_out_channels}), tmp_ten));
        if(this->weight_) delete this->weight_;
        this->weight_ = tmp_ten;
    }else{
        Tensor* tmp_ten = (this->weight_);
        CHECK(tmp_ten->shape().IsSameSize(TensorShape({1, num_in_channels, num_gauss, num_out_channels})));
    }
    if (!this->mu1_ || (this->mu1_)->NumElements() != reshape_total_elements){
        Tensor* tmp_ten = new Tensor();
        OP_REQUIRES_OK(this->context_, this->context_->allocate_temp(tensorflow_dtype, TensorShape({1,num_in_channels, num_gauss, num_out_channels}), tmp_ten));
        if(this->mu1_) delete this->mu1_;
        this->mu1_ = tmp_ten;
    }else{
        Tensor* tmp_ten = this->mu1_;
        CHECK(tmp_ten->shape().IsSameSize(TensorShape({1, num_in_channels, num_gauss, num_out_channels})));
    }
    if (!this->mu2_ || (this->mu2_)->NumElements() != reshape_total_elements){
        Tensor* tmp_ten = new Tensor();
        OP_REQUIRES_OK(this->context_, this->context_->allocate_temp(tensorflow_dtype, TensorShape({1,num_in_channels, num_gauss, num_out_channels}), tmp_ten));
        if(this->mu2_) delete this->mu2_;
        this->mu2_ = tmp_ten;
    }else{
        Tensor* tmp_ten = this->mu2_;
        CHECK(tmp_ten->shape().IsSameSize(TensorShape({1, num_in_channels, num_gauss, num_out_channels})));
    }
    if (!this->sigma_ || (this->sigma_)->NumElements() != reshape_total_elements){
        Tensor* tmp_ten = new Tensor();
        OP_REQUIRES_OK(this->context_, this->context_->allocate_temp(tensorflow_dtype, TensorShape({1,num_in_channels, num_gauss, num_out_channels}), tmp_ten));
        if(this->sigma_) delete this->sigma_;
        this->sigma_ = tmp_ten;

    }else{
        Tensor* tmp_ten = this->sigma_;
        CHECK(tmp_ten->shape().IsSameSize(TensorShape({1, num_in_channels, num_gauss, num_out_channels})));
    }

}

template <typename Dtype>
void DAUKernelOutputTF<Dtype>::reshape(int num_in_channels, int num_out_channels, int num_gauss, int kernel_h, int kernel_w) {

    DataType tensorflow_dtype = DataTypeToEnum<Dtype>::v();

    int reshape_total_elements = num_in_channels * num_out_channels * num_gauss * kernel_h * kernel_w;
    
    if(this->weight_ == NULL || this->weight_->NumElements() != reshape_total_elements){
        Tensor* tmp_ten = new Tensor();
        OP_REQUIRES_OK(this->context_, this->context_->allocate_temp(tensorflow_dtype, TensorShape({num_in_channels, num_gauss * num_out_channels, kernel_h, kernel_w}), tmp_ten));
        if(this->weight_) delete this->weight_;
        this->weight_=tmp_ten;
    }else{
        CHECK(this->weight_->shape().IsSameSize(TensorShape({1, num_in_channels, num_gauss, num_out_channels})));
    }

    if(this->d_error_ == NULL || this->d_error_->NumElements() != reshape_total_elements){
        Tensor* tmp_ten = new Tensor();
        OP_REQUIRES_OK(this->context_, this->context_->allocate_temp(tensorflow_dtype, TensorShape({num_in_channels, num_gauss * num_out_channels, kernel_h, kernel_w}), tmp_ten));
        if(this->d_error_) delete this->d_error_;
        this->d_error_=tmp_ten;
    }else{
        CHECK(this->d_error_->shape().IsSameSize(TensorShape({1, num_in_channels, num_gauss, num_out_channels})));
    }

    if(this->d_params_ == NULL || this->d_params_->NumElements() != reshape_total_elements){
        Tensor* tmp_ten = new Tensor();
        OP_REQUIRES_OK(this->context_, this->context_->allocate_temp(tensorflow_dtype, TensorShape({4, num_in_channels, num_gauss * num_out_channels, kernel_h * kernel_w}), tmp_ten));
        if(this->d_params_) delete this->d_params_;
        this->d_params_=tmp_ten;
    }else{
        CHECK(this->d_params_->shape().IsSameSize(TensorShape({4, num_in_channels, num_gauss * num_out_channels, kernel_h * kernel_w})));
    }
    
}

template DAUKernelComputeTF<float>::~DAUKernelComputeTF();
template DAUKernelComputeTF<double>::~DAUKernelComputeTF();

template DAUKernelParamsTF<float>::~DAUKernelParamsTF();
template DAUKernelParamsTF<double>::~DAUKernelParamsTF();

template DAUKernelOutputTF<float>::~DAUKernelOutputTF();
template DAUKernelOutputTF<double>::~DAUKernelOutputTF();

template void DAUKernelComputeTF<float>::reshape(int num_in_channels, int num_out_channels, int num_gauss, int kernel_h, int kernel_w);
template void DAUKernelComputeTF<double>::reshape(int num_in_channels, int num_out_channels, int num_gauss, int kernel_h, int kernel_w);

template void DAUKernelParamsTF<float>::reshape(int num_in_channels, int num_out_channels, int num_gauss);
template void DAUKernelParamsTF<double>::reshape(int num_in_channels, int num_out_channels, int num_gauss);

template void DAUKernelParamsTF<float>::initialize_params(Tensor w, Tensor mu1, Tensor mu2, Tensor sigma);
template void DAUKernelParamsTF<double>::initialize_params(Tensor w, Tensor mu1, Tensor mu2, Tensor sigma);


template void DAUKernelOutputTF<float>::reshape(int num_in_channels, int num_out_channels, int num_gauss, int kernel_h, int kernel_w);
template void DAUKernelOutputTF<double>::reshape(int num_in_channels, int num_out_channels, int num_gauss, int kernel_h, int kernel_w);

template <typename Dtype>
void DAUKernelComputeTF<Dtype>::create_precompute_index(const int index_size, const int kernel_size) {
    DataType tensorflow_dtype = DataTypeToEnum<int32_t>::v();
    Tensor* tmp_ten;
    if(!this->tmp_precomp_index_ || this->tmp_precomp_index_->NumElements() != index_size+1){
        tmp_ten  = new Tensor();
        OP_REQUIRES_OK(this->context_, this->context_->allocate_temp(tensorflow_dtype, TensorShape({1, 1, 1,index_size + 1}), tmp_ten));
        if(this->tmp_precomp_index_) delete this->tmp_precomp_index_;
        this->tmp_precomp_index_=tmp_ten;

    }else{
        CHECK(this->tmp_precomp_index_->shape().IsSameSize(TensorShape({1, 1, 1, index_size + 1})));
        tmp_ten = (this->tmp_precomp_index_);
    }

    int32_t* flt_data = TENSOR_DATA_PTR(tmp_ten,int32_t);
    int32_t* tmp_buffer = new int32_t[tmp_precomp_index_->NumElements()];
    tmp_buffer[0] = (int32_t) 0;
    for (int i = 0; i < this->tmp_precomp_index_->NumElements()-1;i++)
        tmp_buffer[i+1] = (int32_t) kernel_size * (i+1);
    CUDA_CHECK(cudaMemcpy(flt_data,tmp_buffer,this->tmp_precomp_index_->NumElements()*sizeof(int32_t), cudaMemcpyHostToDevice ));
    delete[] tmp_buffer;

}

