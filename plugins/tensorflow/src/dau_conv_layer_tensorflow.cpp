#include <algorithm>

#include "tensorflow/core/framework/op_kernel.h"

using namespace std;

//#include <opencv2/opencv.hpp>

#include <dau_conv_layer_tensorflow.hpp>

using namespace tensorflow;

template <typename Dtype>
void DAUConvLayerTensorflowGPU<Dtype>::InitializeGrad(DAUConvSettings& settings, Tensor* w_grad, Tensor* mu1_grad, Tensor* mu2_grad, Tensor* sigma_grad){

    this->param_buffer_w_grad = w_grad;
    this->param_buffer_mu1_grad = mu1_grad;
    this->param_buffer_mu2_grad = mu2_grad;
    this->param_buffer_sigma_grad = sigma_grad;

    if(settings.bias_term){

        this->param_buffer_bias_grad = new Tensor();
        OP_REQUIRES_OK_THROW_EX(this->context_, this->context_->allocate_temp(DT_FLOAT, w_grad->shape(), this->param_buffer_bias_grad));

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

        if(this->param_buffer_bias_ != NULL) delete this->param_buffer_bias_;
        this->param_buffer_bias_ = new Tensor();
        OP_REQUIRES_OK_THROW_EX(this->context_, this->context_->allocate_temp(DT_FLOAT, w->shape(), (Tensor*) this->param_buffer_bias_));

    }

}

template <typename Dtype>
DAUConvLayerTensorflowGPU<Dtype>::~DAUConvLayerTensorflowGPU(){
    this->deallocate_workspace_mem();
    if(this->bwd_gradients_ != NULL) delete this->bwd_gradients_;
    if(this->interm_buffer_ != NULL) delete this->interm_buffer_;
    if(this->tmp_param_buffer_ != NULL) delete this->tmp_param_buffer_;
    if(this->col_buffer_ != NULL) delete this->col_buffer_;
    if(this->bias_multiplier_ != NULL)delete this->bias_multiplier_;

}

template <typename Dtype>
void* DAUConvLayerTensorflowGPU<Dtype>::allocate_workspace_mem(size_t bytes) {

    DataType tensorflow_dtype = DataTypeToEnum<Dtype>::v();

    //delete previously allocated memory
    deallocate_workspace_mem();

    //get the buffer size for the specific Dtype
    size_t total_bytes = (bytes/ sizeof(Dtype) + 1)*sizeof(Dtype);

    //allocate new memory
    Tensor* tmp_ten = new Tensor();
    OP_REQUIRES_OK_THROW_EX(this->context_, this->context_->allocate_temp(tensorflow_dtype, TensorShape({total_bytes/sizeof(Dtype)}), tmp_ten));

    //set memory pointer and allocated tensor.
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

    //check if the shape of the parameters is correct.

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
    if(this->param_buffer_w_grad != NULL){

        Tensor* orig_ten = (Tensor*) this->param_buffer_w_grad;
        CHECK(orig_ten->shape().IsSameSize(tmp_shape));

    }

    if(this->param_buffer_mu1_grad != NULL){

        Tensor* orig_ten = (Tensor*) this->param_buffer_mu1_grad;
        CHECK(orig_ten->shape().IsSameSize(tmp_shape));

    }

    if(this->param_buffer_mu2_grad != NULL){

        Tensor* orig_ten = (Tensor*) this->param_buffer_mu2_grad;
        CHECK(orig_ten->shape().IsSameSize(tmp_shape));

    }

    if(this->param_buffer_sigma_grad != NULL){

        Tensor* orig_ten = (Tensor*) this->param_buffer_sigma_grad;
        CHECK(orig_ten->shape().IsSameSize(tmp_shape));

    }


    if (this->bias_term_ != NULL) {

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

    CHECK_EQ(num_units_all % this->ALLOWED_UNITS_GROUP, 0);

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

    //Get Tensorflow type from Dtype
    DataType tensorflow_dtype = DataTypeToEnum<Dtype>::v();

    // call parent to compute all the shape variables
    const vector<int> new_top_shape = BaseDAUConvLayer<Dtype>::Reshape(bottom_shape, top_shape);

    const int max_width = std::max(this->width_out_,this->width_);
    const int max_height = std::max(this->height_out_,this->height_);

    // Set up the all ones "bias multiplier" for adding biases
    if (this->bias_term_ != NULL) {

        if(this->bias_multiplier_ == NULL){

            this->bias_multiplier_ = new Tensor();
            OP_REQUIRES_OK_THROW_EX(this->context_,this->context_->allocate_temp(tensorflow_dtype, TensorShape({1,this->height_out_*this->width_out_}), this->bias_multiplier_));

        }else{

            CHECK(this->bias_multiplier_->shape().IsSameSize(TensorShape({1, this->height_out_ * this->width_out_})));

        }
    }

    if (this->enabled_fwd_op || this->enabled_bwd_op) {
        // make sure col_buffer is big enough
        if(this->col_buffer_ == NULL){

            this->col_buffer_ = new Tensor();
            TensorShape col_shape = TensorShape({this->aggregation.kernel_h_, this->aggregation.kernel_w_, this->height_, this->width_});
            OP_REQUIRES_OK_THROW_EX(this->context_, this->context_->allocate_temp(tensorflow_dtype, col_shape, this->col_buffer_));

        }else{

            CHECK(this->col_buffer_->shape().IsSameSize(TensorShape({this->aggregation.kernel_h_, this->aggregation.kernel_w_, this->height_, this->width_})));

        }

        int interm_buf_size = 0;
        if (this->enabled_fwd_op) interm_buf_size = std::max(interm_buf_size, this->conv_in_channels_);
        if (this->enabled_bwd_op) interm_buf_size = std::max(interm_buf_size, this->conv_out_channels_);
        if (this->enabled_bwd_op) interm_buf_size = std::max(interm_buf_size, this->conv_in_channels_ * this->NUM_K);

        // use inter buffer for both fwd and bwd passes so allocate buffer with suitable size for both
        if(this->interm_buffer_ == NULL){

            this->interm_buffer_ = new Tensor();
            TensorShape interm_shape = TensorShape({this->batch_num_, interm_buf_size, max_height, max_width});
            OP_REQUIRES_OK_THROW_EX(this->context_, this->context_->allocate_temp(tensorflow_dtype, interm_shape, this->interm_buffer_));

        }else{

            CHECK(this->interm_buffer_->shape().IsSameSize(TensorShape({this->batch_num_, interm_buf_size, max_height, max_width})));

        }
    }
    if (this->enabled_bwd_op) {

        if (this->bwd_gradients_ == NULL) {

            this->bwd_gradients_ = new Tensor();
            TensorShape bwd_shape = TensorShape({this->NUM_K, this->conv_in_channels_, this->units_per_channel, this->conv_out_channels_});
            OP_REQUIRES_OK_THROW_EX(this->context_, this->context_->allocate_temp(tensorflow_dtype, bwd_shape, this->bwd_gradients_));

        } else {

            CHECK(this->bwd_gradients_->shape().IsSameSize(TensorShape(
                    {this->NUM_K, this->conv_in_channels_, this->units_per_channel, this->conv_out_channels_})));

        }

    // temporary buffer used during the back-propagation of the error where we rotate mu1 and mu2
        if(this->tmp_param_buffer_ == NULL){

            this->tmp_param_buffer_ = new Tensor();
            TensorShape tmp_param_shape = TensorShape({2, this->conv_in_channels_, this->units_per_channel, this->conv_out_channels_});
            OP_REQUIRES_OK_THROW_EX(this->context_, this->context_->allocate_temp(tensorflow_dtype, tmp_param_shape, this->tmp_param_buffer_));

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

    if(this->weight_ != NULL){delete this->weight_;}
    if(this->mu1_ != NULL){delete this->mu1_;}
    if(this->mu2_ != NULL){delete this->mu2_;}

}

template <typename Dtype>
DAUKernelOutputTF<Dtype>::~DAUKernelOutputTF(){

    if(this->weight_ != NULL){ delete this->weight_;}
    if(this->d_error_ != NULL){ delete this->d_error_;}
    if(this->d_params_ != NULL){delete this->d_params_;}

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

        this->kernels_buffers_[i] = new Tensor();
        TensorShape kernel_buf_shape = TensorShape({num_in_channels, num_gauss, num_out_channels, kernel_h*kernel_w});
        OP_REQUIRES_OK_THROW_EX(this->context_, this->context_->allocate_temp(tensorflow_dtype, kernel_buf_shape, this->kernels_buffers_[i]));

    }


    // allocate and prepare temporary buffers for parameters
    for (int i = 0; i < this->param_buffers_.size(); i++)
        delete this->param_buffers_[i];

    this->param_buffers_.resize(7);

    for (int i = 0; i < 7; i++){

        this->param_buffers_[i] = new Tensor();
        TensorShape param_buf_shape = TensorShape({1, num_in_channels, num_gauss, num_out_channels});
        OP_REQUIRES_OK_THROW_EX(this->context_, this->context_->allocate_temp(tensorflow_dtype, param_buf_shape, this->param_buffers_[i]));

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

    //Allocate params if they have not yet been initialized, otherwise only check if they are of appropriate shape.

    DataType tensorflow_dtype = DataTypeToEnum<Dtype>::v();

    int reshape_total_elements = num_in_channels*num_out_channels*num_gauss;

    TensorShape param_shape = TensorShape({1,num_in_channels, num_gauss, num_out_channels});

    if (this->weight_ == NULL || (this->weight_)->NumElements() != reshape_total_elements){

        if(this->weight_ != NULL) delete this->weight_;
        this->weight_ = new Tensor();
        OP_REQUIRES_OK_THROW_EX(this->context_, this->context_->allocate_temp(tensorflow_dtype, param_shape, this->weight_));

    }else{
        Tensor* tmp_ten = (this->weight_);
        CHECK(tmp_ten->shape().IsSameSize(param_shape));
    }

    if (this->mu1_ == NULL || (this->mu1_)->NumElements() != reshape_total_elements){

        if(this->mu1_ != NULL) delete this->mu1_;
        this->mu1_ = new Tensor();
        OP_REQUIRES_OK_THROW_EX(this->context_, this->context_->allocate_temp(tensorflow_dtype, param_shape, this->mu1_));

    }else{

        Tensor* tmp_ten = this->mu1_;
        CHECK(tmp_ten->shape().IsSameSize(param_shape));

    }

    if (this->mu2_ == NULL || (this->mu2_)->NumElements() != reshape_total_elements){

        if(this->mu2_ != NULL) delete this->mu2_;
        this->mu2_ = new Tensor();
        OP_REQUIRES_OK_THROW_EX(this->context_, this->context_->allocate_temp(tensorflow_dtype, param_shape, this->mu2_));

    }else{

        Tensor* tmp_ten = this->mu2_;
        CHECK(tmp_ten->shape().IsSameSize(param_shape));

    }
    if (this->sigma_ == NULL || (this->sigma_)->NumElements() != reshape_total_elements){

        if(this->sigma_ != NULL) delete this->sigma_;
        this->sigma_ = new Tensor();
        OP_REQUIRES_OK_THROW_EX(this->context_, this->context_->allocate_temp(tensorflow_dtype, param_shape, this->sigma_));

    }else{
        Tensor* tmp_ten = this->sigma_;
        CHECK(tmp_ten->shape().IsSameSize(param_shape));
    }

}

template <typename Dtype>
void DAUKernelOutputTF<Dtype>::reshape(int num_in_channels, int num_out_channels, int num_gauss, int kernel_h, int kernel_w) {

    //Allocate params if they have not yet been initialized, otherwise only check if they are of appropriate shape.

    DataType tensorflow_dtype = DataTypeToEnum<Dtype>::v();

    int reshape_total_elements = num_in_channels * num_out_channels * num_gauss * kernel_h * kernel_w;


    if(this->weight_ == NULL || this->weight_->NumElements() != reshape_total_elements){

        if(this->weight_ != NULL) delete this->weight_;
        this->weight_ = new Tensor();
        TensorShape weight_shape = TensorShape({num_in_channels, num_gauss * num_out_channels, kernel_h, kernel_w});
        OP_REQUIRES_OK_THROW_EX(this->context_, this->context_->allocate_temp(tensorflow_dtype, weight_shape, this->weight_));

    }else{

        CHECK(this->weight_->shape().IsSameSize(TensorShape({1, num_in_channels, num_gauss, num_out_channels})));

    }

    if(this->d_error_ == NULL || this->d_error_->NumElements() != reshape_total_elements){

        if(this->d_error_ != NULL) delete this->d_error_;
        this->d_error_ = new Tensor();
        TensorShape error_shape = TensorShape({num_in_channels, num_gauss * num_out_channels, kernel_h, kernel_w});
        OP_REQUIRES_OK_THROW_EX(this->context_, this->context_->allocate_temp(tensorflow_dtype, error_shape, this->d_error_));

    }else{
        CHECK(this->d_error_->shape().IsSameSize(TensorShape({1, num_in_channels, num_gauss, num_out_channels})));
    }

    if(this->d_params_ == NULL || this->d_params_->NumElements() != reshape_total_elements){

        if(this->d_params_ != NULL) delete this->d_params_;
        this->d_params_= new Tensor();
        TensorShape param_shape = TensorShape({4, num_in_channels, num_gauss * num_out_channels, kernel_h * kernel_w});
        OP_REQUIRES_OK_THROW_EX(this->context_, this->context_->allocate_temp(tensorflow_dtype, param_shape, this->d_params_));

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

    if(this->tmp_precomp_index_ == NULL || this->tmp_precomp_index_->NumElements() != index_size+1){

        if(this->tmp_precomp_index_ != NULL) delete this->tmp_precomp_index_;
        this->tmp_precomp_index_= new Tensor();
        TensorShape precomp_shape = TensorShape({1, 1, 1,index_size + 1});
        OP_REQUIRES_OK_THROW_EX(this->context_, this->context_->allocate_temp(tensorflow_dtype, precomp_shape, this->tmp_precomp_index_));
        tmp_ten = this->tmp_precomp_index_;

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

