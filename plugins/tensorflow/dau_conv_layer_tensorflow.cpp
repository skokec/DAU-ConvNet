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
void DAUConvLayerTensorflowGPU<Dtype>::InitializeFromInput(DAUConvSettings& settings, Tensor* w, Tensor* mu1, Tensor* mu2, Tensor* sigma){
    //Set the layer parameters from input tensors

    settings.offsets_already_centered = true;

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




    //if(this->param_buffer_w_ == NULL) printf("shared_ptr is NULL\n");
    this->param_buffer_w_ = make_shared<Tensor*>(w);
    //if(this->param_buffer_w_ != NULL) printf("shared_ptr is not NULL\n");

    
    this->param_buffer_mu1_ = make_shared<Tensor* >(mu1);

    this->param_buffer_mu2_ = make_shared<Tensor* >(mu2);

    this->param_buffer_sigma_ = make_shared<Tensor* >(sigma);

    if(settings.bias_term){
        Tensor* bias_temp = new Tensor();
        Tensor& tmp_ten = *bias_temp;
        OP_REQUIRES_OK(this->context_, this->context_->allocate_temp(DT_FLOAT, w->shape(), &tmp_ten));
        this->param_buffer_bias_ = make_shared<Tensor*>(bias_temp);    
    }
    
}
template <typename Dtype>
void DAUConvLayerTensorflowGPU<Dtype>::test_allocation(){
    Tensor tmp_ten;
    OP_REQUIRES_OK(this->context_, this->context_->allocate_temp(DT_FLOAT, TensorShape({1,1,1,9}), &tmp_ten));
    this->allocation_test = &tmp_ten;
}


template <typename Dtype>
void DAUComponentInitializerTensorflow<Dtype>::InitializeParameters(const DAUConvSettings& settings, Dtype* w, Dtype* mu1, Dtype* mu2, Dtype* sigma, bool is_gpu_ptr,
                                                               int num_units_per_x, int num_units_per_y, int num_units_ignore,
                                                               int conv_in_channels, int conv_out_channels, int kernel_h, int kernel_w) const {

    // THIS IS AN EMPTY FUNCTION BECAUSE THE VARIABLES ARE INITIALIZED AT INPUT AND
    // AND SET AS LAYER PARAMETERS IN DAUConvLayerTensorflowGPU<Dtype>::InitializeFromInput
}

template <typename Dtype>
DAUConvLayerTensorflowGPU<Dtype>::~DAUConvLayerTensorflowGPU(){
    this->deallocate_workspace_mem();
    cublasDestroy(this->cublasHandle);
}

template <typename Dtype>
void* DAUConvLayerTensorflowGPU<Dtype>::allocate_workspace_mem(size_t bytes) {
    // deallocate existing workspace memory
    deallocate_workspace_mem();
    cudaError_t err = cudaMalloc(&(this->own_workspace_data), bytes);
    if (err != cudaSuccess) {
        // NULL out underlying data
        printf("Cuda allocation failed %d\n", this->own_workspace_data);        
        this->own_workspace_data = NULL;
    }

    return this->own_workspace_data;
}

template <typename Dtype>
void DAUConvLayerTensorflowGPU<Dtype>::deallocate_workspace_mem() {
    if (this->own_workspace_data != NULL){
        CUDA_CHECK(cudaFree(this->own_workspace_data));
    }    

}

template <typename Dtype>
void DAUConvLayerTensorflowGPU<Dtype>::reshape_params(const vector<int>& shape) {
    // initialize DAU parameters (learnable)

    TensorShape tmp_shape;
    for(int dm: shape){
      tmp_shape.AddDim(dm);
    }
    Dtype* w_data = this->param_w();
    Dtype* mu1_data = this->param_mu1();
    Dtype* mu2_data = this->param_mu2();
    Dtype* sigma_data = this->param_sigma();


    Tensor* orig_ten_w = *(this->param_buffer_w_);

    bool correct_copy = orig_ten_w->CopyFrom(*orig_ten_w,tmp_shape);

    if(!correct_copy) printf("Failed reshape params\n");
    Tensor* orig_ten_mu1 = *(this->param_buffer_mu1_);
    correct_copy = orig_ten_mu1->CopyFrom(*orig_ten_mu1,tmp_shape);
    if(!correct_copy) printf("Failed reshape params\n");

    Tensor* orig_ten_mu2 = *(this->param_buffer_mu2_);
    correct_copy = orig_ten_mu2->CopyFrom(*orig_ten_mu2,tmp_shape);
    if(!correct_copy) printf("Failed reshape params\n");

    Tensor* orig_ten_sigma = *(this->param_buffer_sigma_);
    correct_copy = orig_ten_sigma->CopyFrom(*orig_ten_sigma,tmp_shape);
    if(!correct_copy) printf("Failed reshape params\n");



    // If necessary, initialize the biases.
    if (this->bias_term_) {
        Tensor* orig_ten_bias = *(this->param_buffer_bias_);
        correct_copy = orig_ten_bias->CopyFrom(*orig_ten_bias,tmp_shape);
        if(!correct_copy) printf("Failed reshape params bias\n");

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
    static_cast<DAUKernelParamsGPU<Dtype>* >(kernel_param)->sigma_ = this->param_buffer_sigma_;
    
}

template <typename Dtype>
vector<int> DAUConvLayerTensorflowGPU<Dtype>::Reshape(const vector<int>& bottom_shape, const vector<int>& top_shape) {
    //TODO REPLACE CopyFrom with tensor allocation and memcpy

    // call parent to compute all the shape variables

    const vector<int> new_top_shape = BaseDAUConvLayer<Dtype>::Reshape(bottom_shape, top_shape);


    const int max_width = std::max(this->width_out_,this->width_);
    const int max_height = std::max(this->height_out_,this->height_);

    // Set up the all ones "bias multiplier" for adding biases
    if (this->bias_term_) {
        Tensor* tmp_bias_multiplier_ten = new Tensor();
        if(!this->bias_multiplier_){
            Tensor& tmp_ten = *tmp_bias_multiplier_ten;
            Status can_allocate = this->context_->allocate_temp(DT_FLOAT, TensorShape({1,this->height_out_*this->width_out_}), &tmp_ten);
            if(!TF_PREDICT_TRUE(can_allocate.ok())){
                return new_top_shape;
            }
            this->bias_multiplier_ = &tmp_ten;
        }else{
        vector<int> bias_multiplier_shape(1, this->height_out_ * this->width_out_);
        //maybe this is better -- can we even copy the same tensor with no segfault?..
        bool correct_copy = this->bias_multiplier_->CopyFrom(*this->bias_multiplier_,TensorShape({1,this->height_out_*this->width_out_}));
        if(!correct_copy) printf("Failed reshape layerTensorflowGPU\n");            
        }

        auto flt = this->bias_multiplier_->template flat<Dtype>();
        auto dat = flt.data();
        Dtype* bias_data = static_cast<Dtype*>(dat);
        const int num_el = this->bias_multiplier_->NumElements();
        //caffe_set(num_el, Dtype(1), bias_data);
        //memset(bias_data,Dtype(1),sizeof(Dtype) * num_el);
        CUDA_CHECK(cudaMemset(bias_data,Dtype(1), sizeof(Dtype)*num_el));

    }

    // make sure col_buffer is big enough
    //this->col_buffer_.Reshape(this->aggregation.kernel_h_, this->aggregation.kernel_w_, this->height_, this->width_);
    Tensor* col_buffer_ten = new Tensor();
    Tensor* interm_buffer_ten = new Tensor();
    Tensor* bwd_gradients_ten = new Tensor();
    Tensor* tmp_param_buffer_ten = new Tensor();
    if(!this->col_buffer_){
        //Col buffer not defined
        Tensor& tmp_ten = *col_buffer_ten;
        Status can_allocate = this->context_->allocate_temp(DT_FLOAT, TensorShape({this->aggregation.kernel_h_, this->aggregation.kernel_w_, this->height_, this->width_}), &tmp_ten);
        if(!TF_PREDICT_TRUE(can_allocate.ok())){
            return new_top_shape;
        }
        auto tmp_flt = tmp_ten.flat<Dtype>();
        Dtype* tmp_data = static_cast<Dtype*>(tmp_flt.data());
        //for(int i = 0; i < tmp_data.size(); i++) tmp_data(i) = 1.0;
        CUDA_CHECK(cudaMemset(tmp_data,Dtype(1.0), sizeof(Dtype)*tmp_ten.NumElements()));

        this->col_buffer_= &tmp_ten;
    }else{
        bool correct_copy = this->col_buffer_->CopyFrom(*this->col_buffer_,TensorShape({this->aggregation.kernel_h_, this->aggregation.kernel_w_, this->height_, this->width_}));
        if(!correct_copy) printf("Failed reshape layerTensorflowGPU\n");
    }

    // use inter buffer for both fwd and bwd passes so allocate buffer with suitable size for both
    if(!this->interm_buffer_){
        //Iterm buffer not defined
        Tensor& tmp_ten = *interm_buffer_ten;
        Status can_allocate = this->context_->allocate_temp(DT_FLOAT, TensorShape({this->batch_num_, std::max(this->conv_in_channels_ * this->NUM_K, this->conv_out_channels_), max_height, max_width}), &tmp_ten);

        if(!TF_PREDICT_TRUE(can_allocate.ok())){
            return new_top_shape;
        }

        auto tmp_flt = tmp_ten.flat<Dtype>();
        Dtype* tmp_data = static_cast<Dtype*>(tmp_flt.data());
        //for(int i = 0; i < tmp_data.size(); i++) tmp_data(i) = 1.0;
        CUDA_CHECK(cudaMemset(tmp_data,Dtype(1.0), sizeof(Dtype)*tmp_ten.NumElements()));
        this->interm_buffer_= &tmp_ten;


    }else{
        bool correct_copy = this->interm_buffer_->CopyFrom(*this->interm_buffer_,TensorShape({this->batch_num_, std::max(this->conv_in_channels_ * this->NUM_K, this->conv_out_channels_), max_height, max_width}));
        if(!correct_copy) printf("Failed reshape layerTensorflowGPU\n");
    }

    if(!this->bwd_gradients_){
        //bwd gradients buffer not defined
        Tensor& tmp_ten = *bwd_gradients_ten;
        Status can_allocate = this->context_->allocate_temp(DT_FLOAT, TensorShape({this->NUM_K, this->conv_in_channels_, this->units_per_channel, this->conv_out_channels_}), &tmp_ten);

        if(!TF_PREDICT_TRUE(can_allocate.ok())){
            return new_top_shape;
        }
        auto tmp_flt = tmp_ten.flat<Dtype>();
        Dtype* tmp_data = static_cast<Dtype*>(tmp_flt.data());
        //for(int i = 0; i < tmp_data.size(); i++) tmp_data(i) = 1.0;
        CUDA_CHECK(cudaMemset(tmp_data,Dtype(1.0), sizeof(Dtype)*tmp_ten.NumElements()));


        this->bwd_gradients_= &tmp_ten;
    }else{
        bool correct_copy = this->bwd_gradients_->CopyFrom(*this->bwd_gradients_,TensorShape({this->NUM_K, this->conv_in_channels_, this->units_per_channel, this->conv_out_channels_}));
        if(!correct_copy) printf("Failed reshape layerTensorflowGPU\n");
    }

    // temporary buffer used during the back-propagation of the error where we rotate mu1 and mu2
    if(!this->tmp_param_buffer_){
        Tensor& tmp_ten = *tmp_param_buffer_ten;
        Status can_allocate = this->context_->allocate_temp(DT_FLOAT, TensorShape({2, this->conv_in_channels_, this->units_per_channel, this->conv_out_channels_}), &tmp_ten);
        if(!TF_PREDICT_TRUE(can_allocate.ok())){
            return new_top_shape;
        }
        auto tmp_flt = tmp_ten.flat<Dtype>();
        Dtype* tmp_data = static_cast<Dtype*>(tmp_flt.data());
        //for(int i = 0; i < tmp_data.size(); i++) tmp_data(i) = 1.0;
        CUDA_CHECK(cudaMemset(tmp_data,Dtype(1.0), sizeof(Dtype)*tmp_ten.NumElements()));

        this->tmp_param_buffer_= &tmp_ten;
    }else{
        bool correct_copy = this->tmp_param_buffer_->CopyFrom(*this->tmp_param_buffer_,TensorShape({2, this->conv_in_channels_, this->units_per_channel, this->conv_out_channels_}));
        if(!correct_copy) printf("Failed reshape layerTensorflowGPU\n");
    }

    //this->interm_buffer_.Reshape(this->batch_num_, std::max(this->conv_in_channels_ * this->NUM_K, this->conv_out_channels_), max_height, max_width);

    //this->bwd_gradients_.Reshape(this->NUM_K, this->conv_in_channels_, this->units_per_channel, this->conv_out_channels_);

    
    
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

template void DAUComponentInitializerTensorflow<float>::InitializeParameters(const DAUConvSettings& settings, float* w, float* mu1, float* mu2, float* sigma, bool is_gpu_ptr,
                                                               int num_units_per_x, int num_units_per_y, int num_units_ignore,
                                                               int conv_in_channels, int conv_out_channels, int kernel_h, int kernel_w) const;
template void DAUComponentInitializerTensorflow<double>::InitializeParameters(const DAUConvSettings& settings, double* w, double* mu1, double* mu2, double* sigma, bool is_gpu_ptr,
                                                               int num_units_per_x, int num_units_per_y, int num_units_ignore,
                                                               int conv_in_channels, int conv_out_channels, int kernel_h, int kernel_w) const;

template <typename Dtype>
DAUKernelCompute<Dtype>::~DAUKernelCompute(){
    for (int i = 0; i < this->kernels_buffers_.size(); i++)
        delete this->kernels_buffers_[i];

    for (int i = 0; i < this->param_buffers_.size(); i++)
        delete this->param_buffers_[i];
}


template <typename Dtype>
void DAUKernelCompute<Dtype>::reshape(int num_in_channels, int num_out_channels, int num_gauss, int kernel_h, int kernel_w){
    //TODO REPLACE CopyFrom with tensor allocation and memcpy

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
            Tensor* ker_buff_ten = new Tensor();
            Tensor& tmp_ten = *ker_buff_ten;
            if(this->context_ == NULL)printf("Context is null\n");
            OP_REQUIRES_OK(this->context_, this->context_->allocate_temp(DT_FLOAT, TensorShape({num_in_channels, num_gauss, num_out_channels, kernel_h*kernel_w}), &tmp_ten));

            this->kernels_buffers_[i] = &tmp_ten;
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
            Tensor* par_buf_ten = new Tensor();
            Tensor& tmp_ten = *par_buf_ten;
            OP_REQUIRES_OK(this->context_, this->context_->allocate_temp(DT_FLOAT, TensorShape({1, num_in_channels, num_gauss, num_out_channels}), &tmp_ten));
            this->param_buffers_[i] = &tmp_ten;
        }
    }else{
        for (int i = 0; i < 7; i++){
            //this->param_buffers_[i] = this->param_buffers_[i]->template shaped<Dtype,4>({1, num_in_channels, num_gauss, num_out_channels});
            bool correct_copy = this->param_buffers_[i]->CopyFrom(*this->param_buffers_[i],TensorShape({1, num_in_channels, num_gauss, num_out_channels}));
        }

    }

    // pre-computed offset indexes for batched sums (when using caffe_gpu_sum)
    this->create_precompute_index(num_in_channels * num_gauss * num_out_channels, kernel_h * kernel_w);

}

template <typename Dtype>
void DAUKernelParams<Dtype>::initialize_params(Tensor w, Tensor mu1, Tensor mu2, Tensor sigma){
    this->weight_ = make_shared<Tensor*>(&w);
    this->mu1_ = make_shared<Tensor*>(&mu1);
    this->mu2_ = make_shared<Tensor*>(&mu2);
    this->sigma_ = make_shared<Tensor*>(&sigma);

}

template <typename Dtype>
void DAUKernelParams<Dtype>::reshape(int num_in_channels, int num_out_channels, int num_gauss) {
    //TODO REPLACE CopyFrom with tensor allocation and memcpy

    Tensor* weight_ten = new Tensor();
    Tensor* sigma_ten = new Tensor();
    Tensor* mu1_ten = new Tensor();
    Tensor* mu2_ten = new Tensor();
    
    //SHARED_PTR CAUSES ERRORS
    if (this->weight_ == false){
        //this->weight_.reset(new Blob<Dtype>());
        Tensor& tmp_ten = *weight_ten;
        OP_REQUIRES_OK(this->context_, this->context_->allocate_temp(DT_FLOAT, TensorShape({1,num_in_channels, num_gauss, num_out_channels}), &tmp_ten));
        this->weight_.reset(&weight_ten);
    }else{
        //this->weight_ = this->weight_->template shaped<Dtype,4>({1, num_in_channels, num_gauss, num_out_channels});
        //this->weight_->CopyFrom(*this->weight_,TensorShape({1, num_in_channels, num_gauss, num_out_channels}));
        Tensor* tmp_ten = *(this->weight_);
        bool correct_copy = tmp_ten->CopyFrom(*tmp_ten,TensorShape({1, num_in_channels, num_gauss, num_out_channels}));
        if(!correct_copy) printf("Failed reshape kernelParams\n");


    }
    if (this->mu1_ == false){
        //this->mu1_.reset(new Blob<Dtype>());
        Tensor& tmp_ten = *mu1_ten;
        OP_REQUIRES_OK(this->context_, this->context_->allocate_temp(DT_FLOAT, TensorShape({1,num_in_channels, num_gauss, num_out_channels}), &tmp_ten));
        this->mu1_.reset(&mu1_ten);
    }else{
        //this->mu1_ = this->mu1_->template shaped<Dtype,4>({1, num_in_channels, num_gauss, num_out_channels});
        Tensor* tmp_ten = *this->mu1_;
        bool correct_copy = tmp_ten->CopyFrom(*tmp_ten,TensorShape({1, num_in_channels, num_gauss, num_out_channels}));
        if(!correct_copy) printf("Failed reshape kernelParams\n");

    }
    if (this->mu2_ == false){
        //this->mu2_.reset(new Blob<Dtype>());
        Tensor& tmp_ten = *mu2_ten;
        OP_REQUIRES_OK(this->context_, this->context_->allocate_temp(DT_FLOAT, TensorShape({1,num_in_channels, num_gauss, num_out_channels}), &tmp_ten));

        this->mu2_.reset(&mu2_ten);
    }else{
        //this->mu2_ = this->mu2_->template shaped<Dtype,4>({1, num_in_channels, num_gauss, num_out_channels});       
        Tensor* tmp_ten = *this->mu2_;
        bool correct_copy = tmp_ten->CopyFrom(*tmp_ten,TensorShape({1, num_in_channels, num_gauss, num_out_channels}));
        if(!correct_copy) printf("Failed reshape kernelParams\n");

    }
    if (this->sigma_ == false){
        //this->sigma_.reset(new Blob<Dtype>());
        Tensor& tmp_ten = *sigma_ten;
        OP_REQUIRES_OK(this->context_, this->context_->allocate_temp(DT_FLOAT, TensorShape({1,num_in_channels, num_gauss, num_out_channels}), &tmp_ten));

        this->sigma_.reset(&sigma_ten);
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
    //TODO REPLACE CopyFrom with tensor allocation and memcpy

    //this->weight_.Reshape(num_in_channels, num_gauss * num_out_channels, kernel_h, kernel_w);
    Tensor* tmp_weight = new Tensor();
    Tensor* tmp_error = new Tensor();
    Tensor* tmp_params = new Tensor();
    
    
    if(this->weight_ == NULL){
        Tensor& tmp_ten = *tmp_weight;
        OP_REQUIRES_OK(this->context_, this->context_->allocate_temp(DT_FLOAT, TensorShape({num_in_channels, num_gauss * num_out_channels, kernel_h, kernel_w}), &tmp_ten));
        this->weight_=&tmp_ten;

    }else{
        bool correct_copy = this->weight_->CopyFrom(*this->weight_, TensorShape({num_in_channels, num_gauss * num_out_channels, kernel_h, kernel_w}));
        if(!correct_copy) printf("Failed reshape kernelOutput\n");
    }

    if(this->d_error_ == NULL){
        Tensor& tmp_ten = *tmp_error;
        OP_REQUIRES_OK(this->context_, this->context_->allocate_temp(DT_FLOAT, TensorShape({num_in_channels, num_gauss * num_out_channels, kernel_h, kernel_w}), &tmp_ten));
        this->d_error_=&tmp_ten;

    }else{
        bool correct_copy = this->weight_->CopyFrom(*this->weight_, TensorShape({num_in_channels, num_gauss * num_out_channels, kernel_h, kernel_w}));
        if(!correct_copy) printf("Failed reshape kernelOutput\n");
    }

    if(this->d_params_ == NULL){
        Tensor& tmp_ten = *tmp_params;
        OP_REQUIRES_OK(this->context_, this->context_->allocate_temp(DT_FLOAT, TensorShape({num_in_channels, num_gauss * num_out_channels, kernel_h, kernel_w}), &tmp_ten));
        this->d_params_=&tmp_ten;

    }else{
        bool correct_copy = this->weight_->CopyFrom(*this->weight_, TensorShape({num_in_channels, num_gauss * num_out_channels, kernel_h, kernel_w}));
        if(!correct_copy) printf("Failed reshape kernelOutput\n");
    }
    
}

template DAUKernelCompute<float>::~DAUKernelCompute();
template DAUKernelCompute<double>::~DAUKernelCompute();

template void DAUKernelCompute<float>::reshape(int num_in_channels, int num_out_channels, int num_gauss, int kernel_h, int kernel_w);
template void DAUKernelCompute<double>::reshape(int num_in_channels, int num_out_channels, int num_gauss, int kernel_h, int kernel_w);

template void DAUKernelParams<float>::reshape(int num_in_channels, int num_out_channels, int num_gauss);
template void DAUKernelParams<double>::reshape(int num_in_channels, int num_out_channels, int num_gauss);

template void DAUKernelParams<float>::initialize_params(Tensor w, Tensor mu1, Tensor mu2, Tensor sigma);
template void DAUKernelParams<double>::initialize_params(Tensor w, Tensor mu1, Tensor mu2, Tensor sigma);


template void DAUKernelOutput<float>::reshape(int num_in_channels, int num_out_channels, int num_gauss, int kernel_h, int kernel_w);
template void DAUKernelOutput<double>::reshape(int num_in_channels, int num_out_channels, int num_gauss, int kernel_h, int kernel_w);

template <typename Dtype>
void DAUKernelCompute<Dtype>::create_precompute_index(const int index_size, const int kernel_size) {

    //TODO REPLACE CopyFrom with tensor allocation and memcpy
    //tmp_precomp_index_.Reshape(1, 1, 1, index_size + 1);
    //tmp_precomp_index_ = tmp_precomp_index_->template shaped<Dtype,4>({1, 1, 1, index_size + 1});
    Tensor* precompute_tensor = new Tensor();
    Tensor& tmp_ten = *precompute_tensor;
    if(!this->tmp_precomp_index_){

        OP_REQUIRES_OK(this->context_, this->context_->allocate_temp(DT_INT32, TensorShape({1, 1, 1,index_size + 1}), &tmp_ten));
        this->tmp_precomp_index_=&tmp_ten;

    }else{
        bool correct_copy = tmp_precomp_index_->CopyFrom(*tmp_precomp_index_, TensorShape({1, 1, 1, index_size + 1}));
        if(!correct_copy) printf("Failed reshape create_precompute_index\n");
        tmp_ten = *(this->tmp_precomp_index_);
    }


    auto flt = tmp_ten.flat<int32_t>();
    int* flt_data = static_cast<int*>(flt.data());
    int* tmp_buffer;
    tmp_buffer = new int[tmp_precomp_index_->NumElements()];
    tmp_buffer[0] = 0;
    for (int i = 0; i < this->tmp_precomp_index_->NumElements()-1;i++)
        tmp_buffer[i+1] = kernel_size * (i+1);
    CUDA_CHECK(cudaMemcpy(flt_data,tmp_buffer,this->tmp_precomp_index_->NumElements()*sizeof(Dtype), cudaMemcpyHostToDevice ));
    delete[] tmp_buffer;

}

