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

    this->param_buffer_w_grad = make_shared<const Tensor*>(w_grad);
    this->param_buffer_mu1_grad = make_shared<const Tensor* >(mu1_grad);
    this->param_buffer_mu2_grad = make_shared<const Tensor* >(mu2_grad);
    this->param_buffer_sigma_grad = make_shared<const Tensor* >(sigma_grad);

    if(settings.bias_term){
        Tensor* bias_temp = new Tensor();
        Tensor& tmp_ten = *bias_temp;
        OP_REQUIRES_OK(this->context_, this->context_->allocate_temp(DT_FLOAT, w_grad->shape(), &tmp_ten));
        this->param_buffer_bias_grad = make_shared<const Tensor*>(bias_temp);
    }


}


template <typename Dtype>
void DAUConvLayerTensorflowGPU<Dtype>::InitializeFromInput(DAUConvSettings& settings, Tensor* w, Tensor* mu1, Tensor* mu2, Tensor* sigma){
    //Set the layer parameters from input tensors
    if(this->param_buffer_w_)this->param_buffer_w_.reset();
    if(this->param_buffer_mu1_)this->param_buffer_mu1_.reset();
    if(this->param_buffer_mu2_)this->param_buffer_mu2_.reset();
    if(this->param_buffer_sigma_)this->param_buffer_sigma_.reset();

    this->param_buffer_w_ = make_shared<const Tensor*>(w);
    this->param_buffer_mu1_ = make_shared<const Tensor* >(mu1);
    this->param_buffer_mu2_ = make_shared<const Tensor* >(mu2);
    this->param_buffer_sigma_ = make_shared<const Tensor* >(sigma);

    if(settings.bias_term){
        Tensor* bias_temp = new Tensor();
        Tensor& tmp_ten = *bias_temp;
        OP_REQUIRES_OK(this->context_, this->context_->allocate_temp(DT_FLOAT, w->shape(), &tmp_ten));
        if(this->param_buffer_bias_) this->param_buffer_bias_.reset();
        this->param_buffer_bias_ = make_shared<const Tensor*>(bias_temp);
    }
    
}

template <typename Dtype>
DAUConvLayerTensorflowGPU<Dtype>::~DAUConvLayerTensorflowGPU(){
    this->deallocate_workspace_mem();
    cublasDestroy(this->cublasHandle);
}

template <typename Dtype>
void* DAUConvLayerTensorflowGPU<Dtype>::allocate_workspace_mem(size_t bytes) {
    // deallocate existing workspace memory
    DataType tensorflow_dtype = DataTypeToEnum<Dtype>::v();

    deallocate_workspace_mem();
    Tensor* tmp_ten = new Tensor();

    Status can_allocate = this->context_->allocate_temp(tensorflow_dtype, TensorShape({bytes/sizeof(Dtype)}), tmp_ten);
    if(!TF_PREDICT_TRUE(can_allocate.ok())){
        this->own_workspace_data = NULL;
        return this->own_workspace_data;
    }
    this->own_workspace_tensor = tmp_ten;
    this->own_workspace_data = reinterpret_cast<Dtype*>(tmp_ten->flat<Dtype>().data());
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

    Tensor* orig_ten_w = (Tensor*) *(this->param_buffer_w_);
    CHECK(orig_ten_w->shape().IsSameSize(tmp_shape));

    Tensor* orig_ten_mu1 = (Tensor*) *(this->param_buffer_mu1_);
    CHECK(orig_ten_mu1->shape().IsSameSize(tmp_shape));


    Tensor* orig_ten_mu2 = (Tensor*) *(this->param_buffer_mu2_);
    CHECK(orig_ten_mu2->shape().IsSameSize(tmp_shape));


    Tensor* orig_ten_sigma = (Tensor*) *(this->param_buffer_sigma_);
    CHECK(orig_ten_sigma->shape().IsSameSize(tmp_shape));


    if (this->bias_term_) {
        Tensor* orig_ten_bias = (Tensor*) *(this->param_buffer_bias_);
        CHECK(orig_ten_bias->shape().IsSameSize(tmp_shape));

    }
}

template <typename Dtype>
void DAUConvLayerTensorflowGPU<Dtype>::LayerSetUp(const DAUConvSettings& settings,
                                             const BaseDAUComponentInitializer<Dtype>& param_initializer,
                                             BaseDAUKernelCompute<Dtype>* kernel_compute,
                                             BaseDAUKernelParams<Dtype>* kernel_param,
                                             BaseDAUKernelOutput<Dtype>* kernel_output,
                                             const vector<int>& bottom_shape, bool in_train) {

    // call parent to compute all the shape variables and call initialize of parameter shape

    BaseDAUConvLayer<Dtype>::LayerSetUp(settings, param_initializer,
                                        kernel_compute, kernel_param, kernel_output,
                                        bottom_shape, in_train);


    // we use actual (learnable) sigma parameter when computing kernels so connect that param with the sigma for aggregation
    reinterpret_cast<DAUKernelParamsTFGPU<Dtype>* >(kernel_param)->sigma_ = make_shared<Tensor*>((Tensor*)(*this->param_buffer_sigma_));
    
}

template <typename Dtype>
vector<int> DAUConvLayerTensorflowGPU<Dtype>::Reshape(const vector<int>& bottom_shape, const vector<int>& top_shape) {
    //TODO IMPLEMENT CUSTOM OP_REQUIRES_OK MACRO THAT DOES NOT RETURN BUT SETS CtxFailureWithWarning.. replace CHECK
    // call parent to compute all the shape variables
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

    // make sure col_buffer is big enough
    if(!this->col_buffer_){
        Tensor* tmp_ten = new Tensor();
        OP_REQUIRES_OK_BREAK(this->context_, this->context_->allocate_temp(tensorflow_dtype, TensorShape({this->aggregation.kernel_h_, this->aggregation.kernel_w_, this->height_, this->width_}), tmp_ten));
        this->col_buffer_= tmp_ten;
    }else{
        CHECK(this->col_buffer_->shape().IsSameSize(TensorShape({this->aggregation.kernel_h_, this->aggregation.kernel_w_, this->height_, this->width_})));
    }

    // use inter buffer for both fwd and bwd passes so allocate buffer with suitable size for both
    if(!this->interm_buffer_){
        Tensor* tmp_ten = new Tensor();
        OP_REQUIRES_OK_BREAK(this->context_, this->context_->allocate_temp(tensorflow_dtype, TensorShape({this->batch_num_, std::max(this->conv_in_channels_ * this->NUM_K, this->conv_out_channels_), max_height, max_width}), tmp_ten));
        this->interm_buffer_= tmp_ten;
    }else{
        CHECK(this->interm_buffer_->shape().IsSameSize(TensorShape({this->batch_num_, std::max(this->conv_in_channels_ * this->NUM_K, this->conv_out_channels_), max_height, max_width})));
    }

    if(!this->is_forward_op) {
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
    bool updated = BaseDAUConvLayer<Dtype>::update_prefiltering_kernels(stream);


    if (updated) {
        //for debug write kernel with 1 only at center i.e. identity convolution kernel
        if (0) {
            DAUKernelOutputTF<Dtype>* kernels_output = reinterpret_cast<DAUKernelOutputTF<Dtype>*>(this->aggregation.kernels);

            //Dtype*  gauss_kernel = kernels_output->weight_.mutable_cpu_data();
            auto flt_w = kernels_output->weight_->template flat<Dtype>();
            auto dat_w = flt_w.data();
            Dtype* gauss_kernel = reinterpret_cast<Dtype*>(dat_w);

            int deriv_count = this->conv_in_channels_ * this->units_per_channel * this->conv_out_channels_ *
                              this->aggregation.kernel_h_ * this->aggregation.kernel_w_;

            //Dtype*  deriv_weight_kernel = kernels_output->d_params_.mutable_cpu_data() + 0 * deriv_count;
            auto flt = kernels_output->d_params_->template flat<Dtype>();
            auto dat = flt.data();
            Dtype* deriv_weight_kernel = reinterpret_cast<Dtype*>(dat) + 0 * deriv_count;

            //Dtype*  deriv_mu1_kernel = kernels_output->d_params_.mutable_cpu_data() + 1 * deriv_count;
            //auto flt = kernels_output->d_params_->template flat<Dtype>();
            //auto dat = flt.data();
            Dtype* deriv_mu1_kernel = reinterpret_cast<Dtype*>(dat) + 1 * deriv_count;

            //Dtype*  deriv_mu2_kernel = kernels_output->d_params_.mutable_cpu_data() + 2 * deriv_count;
            //auto flt = kernels_output->d_params_->template flat<Dtype>();
            //auto dat = flt.data();
            Dtype* deriv_mu2_kernel = reinterpret_cast<Dtype*>(dat) + 2 * deriv_count;

            //Dtype*  deriv_sigma_kernel = kernels_output->d_params_.mutable_cpu_data() + 3 * deriv_count;
            //auto flt = kernels_output->d_params_->template flat<Dtype>();
            //auto dat = flt.data();
            Dtype* deriv_sigma_kernel = reinterpret_cast<Dtype*>(dat) + 3 * deriv_count;

            //Dtype*  deriv_error_kernel = kernels_output->d_error_.mutable_cpu_data();
            auto flt_err = kernels_output->d_error_->template flat<Dtype>();
            auto dat_err = flt_err.data();
            Dtype* deriv_error_kernel = reinterpret_cast<Dtype*>(dat_err);



            int h_half = this->aggregation.kernel_h_/2;
            int w_half = this->aggregation.kernel_w_/2;
            int index = 0;
            for (int j = -h_half; j <= h_half; ++j) {
                for (int i = -w_half; i <= w_half; ++i) {

                    Dtype val = (i == 0 && j == 0 ? 1 : 0);

                    gauss_kernel[index] = val;
                    deriv_weight_kernel[index] = val;
                    deriv_mu1_kernel[index] = val;
                    deriv_mu2_kernel[index] = val;
                    deriv_sigma_kernel[index] = val;
                    deriv_error_kernel[index] = val;

                    index++;
                }
            }
        }
    }
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
                                                      const vector<int>& bottom_shape, bool in_train);
template void DAUConvLayerTensorflowGPU<double>::LayerSetUp(const DAUConvSettings& settings, const BaseDAUComponentInitializer<double>& param_initializer,
                                                       BaseDAUKernelCompute<double>* kernel_compute, BaseDAUKernelParams<double>* kernel_param, BaseDAUKernelOutput<double>* kernel_output,
                                                       const vector<int>& bottom_shape, bool in_train);

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
void DAUKernelComputeTF<Dtype>::reshape(int num_in_channels, int num_out_channels, int num_gauss, int kernel_h, int kernel_w){

    DataType tensorflow_dtype = DataTypeToEnum<Dtype>::v();
    this->num_in_channels = num_in_channels;
    this->num_out_channels = num_out_channels;
    this->num_gauss = num_gauss;
    this->kernel_h = kernel_h;
    this->kernel_w = kernel_w;
    // allocate and prepare temporary buffers for kernels
    // removed copyFrom, always allocates new memory
    for (int i = 0; i < this->kernels_buffers_.size(); i++)
        delete this->kernels_buffers_[i];

    this->kernels_buffers_.resize(5);
    for (int i = 0; i < 5; i++){
        Tensor* tmp_ten = new Tensor();
        if(this->context_ == NULL)printf("Context is null\n");
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
    this->weight_ = make_shared<Tensor*>(&w);
    this->mu1_ = make_shared<Tensor*>(&mu1);
    this->mu2_ = make_shared<Tensor*>(&mu2);
    this->sigma_ = make_shared<Tensor*>(&sigma);

}

template <typename Dtype>
void DAUKernelParamsTF<Dtype>::reshape(int num_in_channels, int num_out_channels, int num_gauss) {

    DataType tensorflow_dtype = DataTypeToEnum<Dtype>::v();

    int reshape_total_elements = num_in_channels*num_out_channels*num_gauss;
    
    if (!this->weight_ || (*this->weight_)->NumElements() != reshape_total_elements){
        Tensor* tmp_ten = new Tensor();
        OP_REQUIRES_OK(this->context_, this->context_->allocate_temp(tensorflow_dtype, TensorShape({1,num_in_channels, num_gauss, num_out_channels}), tmp_ten));
        if(this->weight_) this->weight_.reset();
        this->weight_ = make_shared<Tensor*>(tmp_ten);
    }else{
        Tensor* tmp_ten = *(this->weight_);
        CHECK(tmp_ten->shape().IsSameSize(TensorShape({1, num_in_channels, num_gauss, num_out_channels})));
        //bool correct_copy = tmp_ten->CopyFrom(*tmp_ten,TensorShape({1, num_in_channels, num_gauss, num_out_channels}));
        //if(!correct_copy) printf("Failed reshape kernelParams\n");
    }
    if (!this->mu1_ || (*this->mu1_)->NumElements() != reshape_total_elements){
        Tensor* tmp_ten = new Tensor();
        OP_REQUIRES_OK(this->context_, this->context_->allocate_temp(tensorflow_dtype, TensorShape({1,num_in_channels, num_gauss, num_out_channels}), tmp_ten));
        if(this->mu1_) this->mu1_.reset();
        this->mu1_ = make_shared<Tensor*>(tmp_ten);
    }else{
        Tensor* tmp_ten = *this->mu1_;
        CHECK(tmp_ten->shape().IsSameSize(TensorShape({1, num_in_channels, num_gauss, num_out_channels})));
        //bool correct_copy = tmp_ten->CopyFrom(*tmp_ten,TensorShape({1, num_in_channels, num_gauss, num_out_channels}));
        //if(!correct_copy) printf("Failed reshape kernelParams\n");
    }
    if (!this->mu2_ || (*this->mu2_)->NumElements() != reshape_total_elements){
        Tensor* tmp_ten = new Tensor();
        OP_REQUIRES_OK(this->context_, this->context_->allocate_temp(tensorflow_dtype, TensorShape({1,num_in_channels, num_gauss, num_out_channels}), tmp_ten));
        if(this->mu2_) this->mu2_.reset();
        this->mu2_ = make_shared<Tensor*>(tmp_ten);
    }else{
        Tensor* tmp_ten = *this->mu2_;
        CHECK(tmp_ten->shape().IsSameSize(TensorShape({1, num_in_channels, num_gauss, num_out_channels})));
        //bool correct_copy = tmp_ten->CopyFrom(*tmp_ten,TensorShape({1, num_in_channels, num_gauss, num_out_channels}));
        //if(!correct_copy) printf("Failed reshape kernelParams\n");
    }
    if (!this->sigma_ || (*this->sigma_)->NumElements() != reshape_total_elements){
        Tensor* tmp_ten = new Tensor();
        OP_REQUIRES_OK(this->context_, this->context_->allocate_temp(tensorflow_dtype, TensorShape({1,num_in_channels, num_gauss, num_out_channels}), tmp_ten));
        if(this->sigma_) this->sigma_.reset();
        this->sigma_ = make_shared<Tensor*>(tmp_ten);

    }else{
        Tensor* tmp_ten = *this->sigma_;
        CHECK(tmp_ten->shape().IsSameSize(TensorShape({1, num_in_channels, num_gauss, num_out_channels})));
        //bool correct_copy = tmp_ten->CopyFrom(*tmp_ten,TensorShape({1, num_in_channels, num_gauss, num_out_channels}));
        //if(!correct_copy) printf("Failed reshape kernelParams\n");
    }

}

template <typename Dtype>
void DAUKernelOutputTF<Dtype>::reshape(int num_in_channels, int num_out_channels, int num_gauss, int kernel_h, int kernel_w) {

    DataType tensorflow_dtype = DataTypeToEnum<Dtype>::v();

    int reshape_total_elements = num_in_channels * num_out_channels * num_gauss * kernel_h * kernel_w;
    
    if(this->weight_ == NULL || this->weight_->NumElements() != reshape_total_elements){
        Tensor* tmp_ten = new Tensor();
        delete this->weight_;
        OP_REQUIRES_OK(this->context_, this->context_->allocate_temp(tensorflow_dtype, TensorShape({num_in_channels, num_gauss * num_out_channels, kernel_h, kernel_w}), tmp_ten));
        this->weight_=tmp_ten;


    }else{
        CHECK(this->weight_->shape().IsSameSize(TensorShape({1, num_in_channels, num_gauss, num_out_channels})));
        //bool correct_copy = this->weight_->CopyFrom(*this->weight_, TensorShape({num_in_channels, num_gauss * num_out_channels, kernel_h, kernel_w}));
        //if(!correct_copy) printf("Failed reshape kernelOutput\n");
    }

    if(this->d_error_ == NULL || this->d_error_->NumElements() != reshape_total_elements){
        Tensor* tmp_ten = new Tensor();
        delete this->d_error_;
        OP_REQUIRES_OK(this->context_, this->context_->allocate_temp(tensorflow_dtype, TensorShape({num_in_channels, num_gauss * num_out_channels, kernel_h, kernel_w}), tmp_ten));
        this->d_error_=tmp_ten;

    }else{
        CHECK(this->d_error_->shape().IsSameSize(TensorShape({1, num_in_channels, num_gauss, num_out_channels})));
        //bool correct_copy = this->d_error_->CopyFrom(*this->d_error_, TensorShape({num_in_channels, num_gauss * num_out_channels, kernel_h, kernel_w}));
        //if(!correct_copy) printf("Failed reshape kernelOutput\n");
    }

    if(this->d_params_ == NULL || this->d_params_->NumElements() != reshape_total_elements){
        Tensor* tmp_ten = new Tensor();
        delete this->d_params_;
        OP_REQUIRES_OK(this->context_, this->context_->allocate_temp(tensorflow_dtype, TensorShape({num_in_channels, num_gauss * num_out_channels, kernel_h, kernel_w}), tmp_ten));
        this->d_params_=tmp_ten;

    }else{
        CHECK(this->d_params_->shape().IsSameSize(TensorShape({1, num_in_channels, num_gauss, num_out_channels})));
        //bool correct_copy = this->d_params_->CopyFrom(*this->d_params_, TensorShape({num_in_channels, num_gauss * num_out_channels, kernel_h, kernel_w}));
        //if(!correct_copy) printf("Failed reshape kernelOutput\n");
    }
    
}

template DAUKernelComputeTF<float>::~DAUKernelComputeTF();
template DAUKernelComputeTF<double>::~DAUKernelComputeTF();

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

    Tensor* tmp_ten = new Tensor();
    if(!this->tmp_precomp_index_ || this->tmp_precomp_index_->NumElements() != index_size+1){

        OP_REQUIRES_OK(this->context_, this->context_->allocate_temp(DT_INT32, TensorShape({1, 1, 1,index_size + 1}), tmp_ten));
        this->tmp_precomp_index_=tmp_ten;

    }else{
        bool correct_copy = tmp_precomp_index_->CopyFrom(*tmp_precomp_index_, TensorShape({1, 1, 1, index_size + 1}));
        if(!correct_copy) printf("Failed reshape create_precompute_index\n");
        tmp_ten = (this->tmp_precomp_index_);
    }


    auto flt = tmp_ten->flat<int32_t>();
    int* flt_data = reinterpret_cast<int*>(flt.data());
    int* tmp_buffer;
    tmp_buffer = new int[tmp_precomp_index_->NumElements()];
    tmp_buffer[0] = 0;
    for (int i = 0; i < this->tmp_precomp_index_->NumElements()-1;i++)
        tmp_buffer[i+1] = kernel_size * (i+1);
    CUDA_CHECK(cudaMemcpy(flt_data,tmp_buffer,this->tmp_precomp_index_->NumElements()*sizeof(Dtype), cudaMemcpyHostToDevice ));
    delete[] tmp_buffer;

}

