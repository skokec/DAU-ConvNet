#include <algorithm>
#include <vector>

//#include "caffe/filler.hpp"

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"


using namespace std;

#include <opencv2/opencv.hpp>
#include <opencv2/gpu/gpu.hpp>


//#include <caffe/layers/dau_conv_layer.hpp>
#include <dau_conv_layer_tensorflow.hpp>
#include <dau_conv/util/math_functions.hpp>

using namespace tensorflow;

//namespace caffe {

template <typename Dtype>
void DAUConvLayerTensorflowGPU<Dtype>::InitializeFromInput(const DAUConvSettings& settings, Tensor* w, Tensor* mu1, Tensor* mu2, Tensor* sigma){
    //Set the layer parameters from input tensors
    //if(this->param_buffer_w_ == NULL) printf("shared_ptr is NULL\n");
    this->param_buffer_w_ = make_shared<Tensor*>(w);
    //if(this->param_buffer_w_ != NULL) printf("shared_ptr is not NULL\n");

    
    this->param_buffer_mu1_ = make_shared<Tensor* >(mu1);

    this->param_buffer_mu2_ = make_shared<Tensor* >(mu2);

    this->param_buffer_sigma_ = make_shared<Tensor* >(sigma);
    
}

template <typename Dtype>
void DAUComponentInitializerTensorflow<Dtype>::InitializeParameters(const DAUConvSettings& settings, Dtype* w, Dtype* mu1, Dtype* mu2, Dtype* sigma, bool is_gpu_ptr,
                                                               int num_units_per_x, int num_units_per_y, int num_units_ignore,
                                                               int conv_in_channels, int conv_out_channels, int kernel_h, int kernel_w) const {

    // THIS IS AN EMPTY FUNCTION BECAUSE THE VARIABLES ARE INITIALIZED AT INPUT AND
    // AND SET AS LAYER PARAMETERS IN DAUConvLayerTensorflowGPU<Dtype>::InitializeFromInput
    printf("Empty initialization\n");
}

template <typename Dtype>
DAUConvLayerTensorflowGPU<Dtype>::~DAUConvLayerTensorflowGPU(){
    this->deallocate_workspace_mem();
}

template <typename Dtype>
void* DAUConvLayerTensorflowGPU<Dtype>::allocate_workspace_mem(size_t bytes) {
    // deallocate existing workspace memory
    deallocate_workspace_mem();

    // then allocate new one
    cudaError_t err = cudaMalloc(&(this->own_workspace_data), bytes);
    if (err != cudaSuccess) {
        // NULL out underlying data
        this->own_workspace_data = NULL;
    }
    return this->own_workspace_data;
}

template <typename Dtype>
void DAUConvLayerTensorflowGPU<Dtype>::deallocate_workspace_mem() {
    if (this->own_workspace_data == NULL)
        CUDA_CHECK(cudaFree(this->own_workspace_data));

}

template <typename Dtype>
void DAUConvLayerTensorflowGPU<Dtype>::reshape_params(const vector<int>& shape) {
    // initialize DAU parameters (learnable)
    //this->param_buffer_w_.reset(new Blob<Dtype>(shape));
    //this->param_buffer_mu1_.reset(new Blob<Dtype>(shape));
    //this->param_buffer_mu2_.reset(new Blob<Dtype>(shape));
    //this->param_buffer_sigma_.reset(new Blob<Dtype>(shape));


    TensorShape tmp_shape;
    for(int dm: shape){
      tmp_shape.AddDim(dm);
    }


    // Calling Tensor constructor directly is deprecated .. use _allocate functions
    Tensor* tmp_ten_w;
    OP_REQUIRES_OK(this->context_, this->context_->allocate_temp(DT_FLOAT, tmp_shape, tmp_ten_w));
    Tensor* tmp_ten_mu1;
    OP_REQUIRES_OK(this->context_, this->context_->allocate_temp(DT_FLOAT, tmp_shape, tmp_ten_mu1));
    Tensor* tmp_ten_mu2;
    OP_REQUIRES_OK(this->context_, this->context_->allocate_temp(DT_FLOAT, tmp_shape, tmp_ten_mu2));
    Tensor* tmp_ten_sigma;
    OP_REQUIRES_OK(this->context_, this->context_->allocate_temp(DT_FLOAT, tmp_shape, tmp_ten_sigma));

    //this->param_buffer_w_.reset(new Tensor(Dtype, tmp_shape));
    this->param_buffer_w_.reset(&tmp_ten_w);
    this->param_buffer_mu1_.reset(&tmp_ten_mu1);
    this->param_buffer_mu2_.reset(&tmp_ten_mu2);
    this->param_buffer_sigma_.reset(&tmp_ten_sigma);

    // If necessary, initialize the biases.
    if (this->bias_term_) {
        //vector<int> bias_shape(1, this->conv_out_channels_);
        //this->param_buffer_bias_.reset(new Blob<Dtype>(bias_shape));
        Tensor* tmp_ten_bias;
        OP_REQUIRES_OK(this->context_, this->context_->allocate_temp(DT_FLOAT, TensorShape({1,this->conv_out_channels_}), tmp_ten_bias));

        this->param_buffer_bias_.reset(&tmp_ten_bias);
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
    /*
    BaseDAUConvLayer<Dtype>::LayerSetUp(settings, param_initializer,
                                        kernel_compute, kernel_param, kernel_output,
                                        bottom_shape, in_train);


    // we use actual (learnable) sigma parameter when computing kernels so connect that param with the sigma for aggregation
    static_cast<DAUKernelParamsGPU<Dtype>* >(kernel_param)->sigma_ = this->param_buffer_sigma_;
    */
    printf("LayerSetUp\n");
}

template <typename Dtype>
vector<int> DAUConvLayerTensorflowGPU<Dtype>::Reshape(const vector<int>& bottom_shape, const vector<int>& top_shape) {

    // call parent to compute all the shape variables
    const vector<int> new_top_shape = BaseDAUConvLayer<Dtype>::Reshape(bottom_shape, top_shape);

    const int max_width = std::max(this->width_out_,this->width_);
    const int max_height = std::max(this->height_out_,this->height_);

    // Set up the all ones "bias multiplier" for adding biases
    if (this->bias_term_) {
        vector<int> bias_multiplier_shape(1, this->height_out_ * this->width_out_);
        //this->bias_multiplier_.Reshape(bias_multiplier_shape);
        //try to reshape tensor -- dont know if this works..
        //*this->bias_multiplier_ = this->bias_multiplier_->shaped<Dtype, 2>(TensorShape({1,this->height_out_*this->width_out_}))
        //maybe this is better -- can we even copy the same tensor with no segfault?..
        bool correct_copy = this->bias_multiplier_->CopyFrom(*this->bias_multiplier_,TensorShape({1,this->height_out_*this->width_out_}));
        if(!correct_copy) printf("Failed reshape\n");
        /*Tensor t = *this->bias_multiplier_;
        auto t_map = t.tensor<Dtype,2>();
        t_map = t.shaped<Dtype, 2>({1,this->height_out_*this->width_out_});
        */

        auto flt = this->bias_multiplier_->template flat<Dtype>();
        auto dat = flt.data();
        Dtype* bias_data = static_cast<Dtype*>(dat);
        const int num_el = this->bias_multiplier_->NumElements();
        //caffe_set(num_el, Dtype(1), bias_data);
        memset(bias_data,0,sizeof(Dtype) * num_el);
        for(int i = 0; i < num_el; i++){
            bias_data[i] = Dtype(1);
        }
    }

    // make sure col_buffer is big enough
    //this->col_buffer_.Reshape(this->aggregation.kernel_h_, this->aggregation.kernel_w_, this->height_, this->width_);
    bool correct_copy = this->col_buffer_->CopyFrom(*this->col_buffer_,TensorShape({this->aggregation.kernel_h_, this->aggregation.kernel_w_, this->height_, this->width_}));
    if(!correct_copy) printf("Failed reshape\n");

    // use inter buffer for both fwd and bwd passes so allocate buffer with suitable size for both
    //this->interm_buffer_.Reshape(this->batch_num_, std::max(this->conv_in_channels_ * this->NUM_K, this->conv_out_channels_), max_height, max_width);
    correct_copy = this->interm_buffer_->CopyFrom(*this->interm_buffer_,TensorShape({this->batch_num_, std::max(this->conv_in_channels_ * this->NUM_K, this->conv_out_channels_), max_height, max_width}));
    if(!correct_copy) printf("Failed reshape\n");

    //this->bwd_gradients_.Reshape(this->NUM_K, this->conv_in_channels_, this->units_per_channel, this->conv_out_channels_);
    correct_copy = this->bwd_gradients_->CopyFrom(*this->bwd_gradients_,TensorShape({this->NUM_K, this->conv_in_channels_, this->units_per_channel, this->conv_out_channels_}));
    if(!correct_copy) printf("Failed reshape\n");

    // temporary buffer used during the back-propagation of the error where we rotate mu1 and mu2
    //this->tmp_param_buffer_.Reshape(2, this->conv_in_channels_, this->units_per_channel, this->conv_out_channels_);
    correct_copy = this->tmp_param_buffer_->CopyFrom(*this->tmp_param_buffer_,TensorShape({2, this->conv_in_channels_, this->units_per_channel, this->conv_out_channels_}));
    if(!correct_copy) printf("Failed reshape\n");
    
    return new_top_shape;

}

template <typename Dtype>
bool DAUConvLayerTensorflowGPU<Dtype>::update_prefiltering_kernels(cudaStream_t stream) {
    bool updated = BaseDAUConvLayer<Dtype>::update_prefiltering_kernels(stream);


    if (updated) {
        //for debug write kernel with 1 only at center i.e. identity convolution kernel
        if (0) {
            DAUKernelOutput<Dtype>* kernels_output = static_cast<DAUKernelOutput<Dtype>*>(this->aggregation.kernels);

            //Dtype*  gauss_kernel = kernels_output->weight_.mutable_cpu_data();
            auto flt_w = kernels_output->weight_->template flat<Dtype>();
            auto dat_w = flt_w.data();
            Dtype* gauss_kernel = static_cast<Dtype*>(dat_w);

            int deriv_count = this->conv_in_channels_ * this->units_per_channel * this->conv_out_channels_ *
                              this->aggregation.kernel_h_ * this->aggregation.kernel_w_;

            //Dtype*  deriv_weight_kernel = kernels_output->d_params_.mutable_cpu_data() + 0 * deriv_count;
            auto flt = kernels_output->d_params_->template flat<Dtype>();
            auto dat = flt.data();
            Dtype* deriv_weight_kernel = static_cast<Dtype*>(dat) + 0 * deriv_count;

            //Dtype*  deriv_mu1_kernel = kernels_output->d_params_.mutable_cpu_data() + 1 * deriv_count;
            //auto flt = kernels_output->d_params_->template flat<Dtype>();
            //auto dat = flt.data();
            Dtype* deriv_mu1_kernel = static_cast<Dtype*>(dat) + 1 * deriv_count;

            //Dtype*  deriv_mu2_kernel = kernels_output->d_params_.mutable_cpu_data() + 2 * deriv_count;
            //auto flt = kernels_output->d_params_->template flat<Dtype>();
            //auto dat = flt.data();
            Dtype* deriv_mu2_kernel = static_cast<Dtype*>(dat) + 2 * deriv_count;

            //Dtype*  deriv_sigma_kernel = kernels_output->d_params_.mutable_cpu_data() + 3 * deriv_count;
            //auto flt = kernels_output->d_params_->template flat<Dtype>();
            //auto dat = flt.data();
            Dtype* deriv_sigma_kernel = static_cast<Dtype*>(dat) + 3 * deriv_count;

            //Dtype*  deriv_error_kernel = kernels_output->d_error_.mutable_cpu_data();
            auto flt_err = kernels_output->d_error_->template flat<Dtype>();
            auto dat_err = flt_err.data();
            Dtype* deriv_error_kernel = static_cast<Dtype*>(dat_err);



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

template <typename Dtype>
DAUKernelCompute<Dtype>::~DAUKernelCompute(){
    for (int i = 0; i < this->kernels_buffers_.size(); i++)
        delete this->kernels_buffers_[i];

    for (int i = 0; i < this->param_buffers_.size(); i++)
        delete this->param_buffers_[i];
}

template <typename Dtype>
void DAUKernelCompute<Dtype>::reshape(int num_in_channels, int num_out_channels, int num_gauss, int kernel_h, int kernel_w){

    this->num_in_channels = num_in_channels;
    this->num_out_channels = num_out_channels;
    this->num_gauss = num_gauss;
    this->kernel_h = kernel_h;
    this->kernel_w = kernel_w;

    // allocate and prepare temporary buffers for kernels
    if (this->kernels_buffers_.size() != 5) {

        for (int i = 0; i < this->kernels_buffers_.size(); i++)
            delete this->kernels_buffers_[i];

        this->kernels_buffers_.resize(5);
        for (int i = 0; i < 5; i++){
            Tensor* tmp_ten;
            OP_REQUIRES_OK(this->context_, this->context_->allocate_temp(DT_FLOAT, TensorShape({num_in_channels, num_gauss, num_out_channels, kernel_h*kernel_w}), tmp_ten));
            this->kernels_buffers_[i] = tmp_ten;
        }

    }else{
        for (int i = 0; i < 5; i++){
            //this->kernels_buffers_[i] = this->kernels_buffers_[i]->template shaped <Dtype,4>({num_in_channels, num_gauss, num_out_channels, kernel_h * kernel_w});
            bool correct_copy = this->kernels_buffers_[i]->CopyFrom(*this->kernels_buffers_[i],TensorShape({num_in_channels, num_gauss, num_out_channels, kernel_h * kernel_w}));
            
            if(!correct_copy) printf("Failed reshape kernelCompute\n");
        }
    }

    //for (int i = 0; i < 5; ++i)
    //    this->kernels_buffers_[i]->Reshape(num_in_channels, num_gauss, num_out_channels, kernel_h * kernel_w);

    // allocate and prepare temporary buffers for parameters
    if (this->param_buffers_.size() != 7){
        for (int i = 0; i < this->param_buffers_.size(); i++)
            delete this->param_buffers_[i];

        this->param_buffers_.resize(7);
        for (int i = 0; i < 7; i++){
            //this->param_buffers_[i] = new Blob<Dtype>();
            Tensor* tmp_ten;
            OP_REQUIRES_OK(this->context_, this->context_->allocate_temp(DT_FLOAT, TensorShape({1, num_in_channels, num_gauss, num_out_channels}), tmp_ten));
            this->param_buffers_[i] = tmp_ten;
        }
    }else{
        for (int i = 0; i < 7; i++){
            //this->param_buffers_[i] = this->param_buffers_[i]->template shaped<Dtype,4>({1, num_in_channels, num_gauss, num_out_channels});
            bool correct_copy = this->param_buffers_[i]->CopyFrom(*this->param_buffers_[i],TensorShape({1, num_in_channels, num_gauss, num_out_channels}));
            if(!correct_copy) printf("Failed reshape kernelCompute\n");
        }

    }

    //already reshaped in upper 4 loop?
    //for (int i = 0; i < 7; ++i)
    //    this->param_buffers_[i]->Reshape(1, num_in_channels, num_gauss, num_out_channels);

    // pre-computed offset indexes for batched sums (when using caffe_gpu_sum)
    this->create_precompute_index(num_in_channels * num_gauss * num_out_channels, kernel_h * kernel_w);

}

template <typename Dtype>
void DAUKernelParams<Dtype>::reshape(int num_in_channels, int num_out_channels, int num_gauss) {


    //SHARED_PTR CAUSES ERRORS

    if (this->weight_ == false){
        //this->weight_.reset(new Blob<Dtype>());
        Tensor* tmp_ten;
        OP_REQUIRES_OK(this->context_, this->context_->allocate_temp(DT_FLOAT, TensorShape({1,num_in_channels, num_gauss, num_out_channels}), tmp_ten));
        this->weight_.reset(&tmp_ten);
    }else{
        //this->weight_ = this->weight_->template shaped<Dtype,4>({1, num_in_channels, num_gauss, num_out_channels});
        //this->weight_->CopyFrom(*this->weight_,TensorShape({1, num_in_channels, num_gauss, num_out_channels}));
        Tensor* tmp_ten = *this->weight_;
        bool correct_copy = tmp_ten->CopyFrom(*tmp_ten,TensorShape({1, num_in_channels, num_gauss, num_out_channels}));
        if(!correct_copy) printf("Failed reshape kernelParams\n");


    }
    if (this->mu1_ == false){ 
        //this->mu1_.reset(new Blob<Dtype>());
        Tensor* tmp_ten;
        OP_REQUIRES_OK(this->context_, this->context_->allocate_temp(DT_FLOAT, TensorShape({1,num_in_channels, num_gauss, num_out_channels}), tmp_ten));
        this->mu1_.reset(&tmp_ten);
    }else{
        //this->mu1_ = this->mu1_->template shaped<Dtype,4>({1, num_in_channels, num_gauss, num_out_channels});
        Tensor* tmp_ten = *this->mu1_;
        bool correct_copy = tmp_ten->CopyFrom(*tmp_ten,TensorShape({1, num_in_channels, num_gauss, num_out_channels}));
        if(!correct_copy) printf("Failed reshape kernelParams\n");

    }
    if (this->mu2_ == false){
        //this->mu2_.reset(new Blob<Dtype>());
        Tensor* tmp_ten;
        OP_REQUIRES_OK(this->context_, this->context_->allocate_temp(DT_FLOAT, TensorShape({1,num_in_channels, num_gauss, num_out_channels}), tmp_ten));
        this->mu2_.reset(&tmp_ten);
    }else{
        //this->mu2_ = this->mu2_->template shaped<Dtype,4>({1, num_in_channels, num_gauss, num_out_channels});       
        Tensor* tmp_ten = *this->mu2_;
        bool correct_copy = tmp_ten->CopyFrom(*tmp_ten,TensorShape({1, num_in_channels, num_gauss, num_out_channels}));
        if(!correct_copy) printf("Failed reshape kernelParams\n");

    }
    if (this->sigma_ == false){
        //this->sigma_.reset(new Blob<Dtype>());
        Tensor* tmp_ten;
        OP_REQUIRES_OK(this->context_, this->context_->allocate_temp(DT_FLOAT, TensorShape({1,num_in_channels, num_gauss, num_out_channels}), tmp_ten));
        this->sigma_.reset(&tmp_ten);
    }else{
        //this->sigma_ = this->sigma_->template shaped<Dtype,4>({1, num_in_channels, num_gauss, num_out_channels});
        Tensor* tmp_ten = *this->sigma_;
        bool correct_copy = tmp_ten->CopyFrom(*tmp_ten,TensorShape({1, num_in_channels, num_gauss, num_out_channels}));
        if(!correct_copy) printf("Failed reshape kernelParams\n");

    }

    /*
    this->weight_->Reshape(1, num_in_channels, num_gauss, num_out_channels);
    this->mu1_->Reshape(1,num_in_channels, num_gauss, num_out_channels);
    this->mu2_->Reshape(1,num_in_channels, num_gauss, num_out_channels);
    this->sigma_->Reshape(1,num_in_channels, num_gauss, num_out_channels);
    */
}

template <typename Dtype>
void DAUKernelOutput<Dtype>::reshape(int num_in_channels, int num_out_channels, int num_gauss, int kernel_h, int kernel_w) {
    //this->weight_.Reshape(num_in_channels, num_gauss * num_out_channels, kernel_h, kernel_w);
    bool correct_copy = this->weight_->CopyFrom(*this->weight_, TensorShape({num_in_channels, num_gauss * num_out_channels, kernel_h, kernel_w}));
    if(!correct_copy) printf("Failed reshape kernelOutput\n");

    //this->d_error_.Reshape(num_in_channels, num_out_channels, kernel_h, kernel_w);
    correct_copy = this->d_error_->CopyFrom(*this->d_error_, TensorShape({num_in_channels, num_out_channels, kernel_h, kernel_w}));
    if(!correct_copy) printf("Failed reshape kernelOutput\n");

    // four params == [w,mu1,mu2,sigma]
    //this->d_params_.Reshape(4, num_in_channels * num_gauss, num_out_channels, kernel_h * kernel_w);
    correct_copy = this->d_params_->CopyFrom(*this->d_params_, TensorShape({4, num_in_channels * num_gauss, num_out_channels, kernel_h * kernel_w}));
    if(!correct_copy) printf("Failed reshape kernelOutput\n");

}

template DAUKernelCompute<float>::~DAUKernelCompute();
template DAUKernelCompute<double>::~DAUKernelCompute();

template void DAUKernelCompute<float>::reshape(int num_in_channels, int num_out_channels, int num_gauss, int kernel_h, int kernel_w);
template void DAUKernelCompute<double>::reshape(int num_in_channels, int num_out_channels, int num_gauss, int kernel_h, int kernel_w);

template void DAUKernelParams<float>::reshape(int num_in_channels, int num_out_channels, int num_gauss);
template void DAUKernelParams<double>::reshape(int num_in_channels, int num_out_channels, int num_gauss);

template void DAUKernelOutput<float>::reshape(int num_in_channels, int num_out_channels, int num_gauss, int kernel_h, int kernel_w);
template void DAUKernelOutput<double>::reshape(int num_in_channels, int num_out_channels, int num_gauss, int kernel_h, int kernel_w);

template <typename Dtype>
void DAUKernelCompute<Dtype>::create_precompute_index(const int index_size, const int kernel_size) {

    //tmp_precomp_index_.Reshape(1, 1, 1, index_size + 1);
    //tmp_precomp_index_ = tmp_precomp_index_->template shaped<Dtype,4>({1, 1, 1, index_size + 1});
    bool correct_copy = tmp_precomp_index_->CopyFrom(*tmp_precomp_index_, TensorShape({1, 1, 1, index_size + 1}));
    if(!correct_copy) printf("Failed reshape create_precompute_index\n");

    auto flt = tmp_precomp_index_->flat<int>();
    auto dat = flt.data();
    int* tmp_precomp_index_cpu = static_cast<int*>(dat);
    //int* tmp_precomp_index_cpu = tmp_precomp_index_.mutable_cpu_data();

    tmp_precomp_index_cpu[0] = 0;
    for (int i = 0; i < tmp_precomp_index_->NumElements()-1;i++)
        tmp_precomp_index_cpu[i+1] = kernel_size * (i+1);

    //for (int i = 0; i < tmp_precomp_index_.count()-1; i++)
    //    tmp_precomp_index_cpu[i+1] = kernel_size * (i+1);

}


// DAUConvolutionLayer for caffe only? not needed? ---------------------------


/*

template <typename Dtype>
void DAUConvolutionLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {

    const DAUConvolutionParameter& param = this->layer_param().dau_conv_param();

    // verify validity of parameters
    CHECK_EQ(param.kernel_size_size(),1) << "Expecting only single kernel_size value in DAUConvolutionParameter";
    CHECK_EQ(param.pad_size(),1) << "Expecting only single pad value in DAUConvolutionParameter";
    CHECK_EQ(param.stride_size(),1) << "Expecting only single stride value in DAUConvolutionParameter";

    // copy them to DAUConvSettings
    DAUConvSettings dau_settings;

    dau_settings.bias_term = param.bias_term();

    dau_settings.num_output = param.num_output();
    dau_settings.number_units.assign(param.number_units().begin(), param.number_units().end());

    dau_settings.kernel_size = param.kernel_size(0);
    dau_settings.pad = param.pad(0);
    dau_settings.stride = param.stride(0);

    dau_settings.unit_normalization = param.unit_normalization();
    dau_settings.square_unit_normalization = param.square_unit_normalization();

    dau_settings.mean_iteration_step = param.mean_iteration_step();
    dau_settings.sigma_iteration_step = param.sigma_iteration_step();

    dau_settings.merge_iteration_step = param.merge_iteration_step();
    dau_settings.merge_threshold = param.merge_threshold();

    dau_settings.sigma_lower_bound = param.sigma_lower_bound();
    dau_settings.component_border_bound = param.component_border_bound();

    // define which param initializer will be used
    DAUComponentInitializerTensorflow<Dtype> param_initializer(param.weight_filler(),
                                                          param.mu_filler(),
                                                          param.sigma_filler());

    // setup layer for DAU-ConvNet object
    dau_compute.LayerSetUp(dau_settings, param_initializer,
                           &this->dau_kernel_compute, &this->dau_kernel_params, &this->dau_kernel_output,
                           bottom[0]->shape(), this->phase_ == TRAIN);


    // we need to manually initialize bias
    if (dau_settings.bias_term) {
        shared_ptr<Filler<Dtype> > bias_filler(GetFiller<Dtype>(param.bias_filler()));
        bias_filler->Fill(this->dau_compute.param_buffer_bias_.get());
    }

    // finally connect param blobs with actual learnable blobs
    this->blobs_.resize(4 + (dau_settings.bias_term ? 1 : 0));

    // we use shared_ptr for params so just asign them to this->blobs_ array
    this->blobs_[0] = this->dau_compute.param_buffer_w_;
    this->blobs_[1] = this->dau_compute.param_buffer_mu1_;
    this->blobs_[2] = this->dau_compute.param_buffer_mu2_;
    this->blobs_[3] = this->dau_compute.param_buffer_sigma_;

    if (dau_settings.bias_term)
        this->blobs_[4] =  this->dau_compute.param_buffer_bias_;

    // Propagate gradients to the parameters (as directed by backward pass).
    this->param_propagate_down_.resize(this->blobs_.size(), true);
}

template <typename Dtype>
void DAUConvolutionLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {

    const vector<int> new_top_shape = this->dau_compute.Reshape(bottom[0]->shape(), top[0]->shape());

    for (int i = 0; i < top.size(); ++i)
        top[i]->Reshape(new_top_shape);

}

template <typename Dtype>
DAUConvolutionLayer<Dtype>::~DAUConvolutionLayer() {


}
template <typename Dtype>
void plot_blob_data(Blob<Dtype>& b) {
    const Dtype* d = b.cpu_data();
    for (int n = 0;  n< b.shape(0); ++n) {
        for (int c = 0;  c< b.shape(1); ++c) {
            for (int j = 0;  j< b.shape(2); ++j) {
                for (int i = 0;  i< b.shape(3); ++i) {
                    printf("%.2f ", d[b.offset(n,c,j,i)]);
                }
                std::cout << std::endl;
            }
        }
    }
    std::cout << std::endl;
}

    template <typename Dtype>
void DAUConvolutionLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {

    for (int i = 0; i < bottom.size(); ++i) {
        this->dau_compute.Forward_gpu(bottom[i]->gpu_data(), bottom[i]->shape(),
                                      top[i]->mutable_gpu_data(), top[i]->shape());
    }

}
template <typename Dtype>
void DAUConvolutionLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom){

    for (int i = 0; i < top.size(); ++i) {
        this->dau_compute.Backward_gpu(top[i]->gpu_data(), top[i]->gpu_diff(), top[i]->shape(), propagate_down[i],
                                       bottom[i]->gpu_data(), bottom[i]->mutable_gpu_diff(), bottom[i]->shape(),
                                       this->param_propagate_down_);

    }
}

    template <typename Dtype>
void DAUConvolutionLayer<Dtype>::Forward_cpu(
        const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {

}

template <typename Dtype>
void DAUConvolutionLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
                                                     const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
}


INSTANTIATE_CLASS(DAUConvolutionLayer);

*/ // DAUConvolutionLayer -- 


//}   // namespace caffe
