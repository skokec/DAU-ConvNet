#ifndef CAFFE_DAU_CONV_LAYER_HPP_
#define CAFFE_DAU_CONV_LAYER_HPP_

#include <string>
#include <utility>
#include <vector>

//#include "caffe/layer.hpp"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/platform/default/logging.h"
#include "tensorflow/core/framework/shape_inference.h"

#include "dau_conv/base_dau_conv_layer.hpp"

//#include "caffe/util/device_alternate.hpp"

// we will be using base classes from DAUConvNet
using DAUConvNet::DAUConvSettings;

using DAUConvNet::BaseDAUConvLayer;
using DAUConvNet::BaseDAUComponentInitializer;

using DAUConvNet::BaseDAUKernelCompute;
using DAUConvNet::BaseDAUKernelOutput;
using DAUConvNet::BaseDAUKernelParams;
using namespace std;
using namespace tensorflow;



////////////////////////////////////////////////////////////////////////////////
// Tensorflow implementation of buffers used in DAUKernel*

template <typename Dtype>
class DAUKernelParams : public  BaseDAUKernelParams<Dtype> {
public:
	//explicit DAUKernelParams(OpKernelContext* context)
	//	: context_(context){}

	void reshape(int num_in_channels, int num_out_channels, int num_gauss);

	void initialize_params(Tensor w, Tensor mu1, Tensor mu2, Tensor sigma);

	shared_ptr<Tensor*> weight_, mu1_, mu2_, sigma_; // CPU for setting (once) GPU for computing, except for sigma

	OpKernelContext* context_ = NULL;

};


template <typename Dtype>
class DAUKernelOutput : public BaseDAUKernelOutput<Dtype> {
public:
	//explicit DAUKernelOutput(OpKernelContext* context)
	//: context_(context){}

	virtual void reshape(int num_in_channels, int num_out_channels, int num_gauss, int kernel_h, int kernel_w);

	// main filter weights
	Tensor* weight_ = NULL;

	// derivative weights for back-propagation and all four parameters
	Tensor* d_error_ = NULL;
	Tensor* d_params_ = NULL; // four params == [w,mu1,mu2,sigma]
	OpKernelContext* context_ = NULL;
};

template <typename Dtype>
class DAUKernelCompute : public BaseDAUKernelCompute<Dtype> {
public:
	explicit DAUKernelCompute(OpKernelContext* context)
		: context_(context){}

	virtual ~DAUKernelCompute();

	virtual void reshape(int num_in_channels, int num_out_channels, int num_gauss,
						 int kernel_h, int kernel_w);

	OpKernelContext* context_ = NULL;

protected:
	void create_precompute_index(const int index_size, const int kernel_size);

	// intermediate buffers when computing derivative kernels in precompute_guassian_weights_gpu
	// temporary buffers for pre-computed sigma^2, sigma^3 and 1/2*sigma^2
	//vector<Blob<Dtype>* > param_buffers_;
	//vector<Blob<Dtype>* > kernels_buffers_;
	
	//dereference Tensor** ? ..
	//vector<Tensor**> param_buffers_;
	//vector<Tensor**> kernels_buffers_;

	vector<Tensor*> param_buffers_;
	vector<Tensor*> kernels_buffers_;


	//Blob<int> tmp_precomp_index_;// pre-computed indexes for caffe_gpu_sum in get_kernels
	Tensor* tmp_precomp_index_ = NULL;
	
};


////////////////////////////////////////////////////////////////////////////////
// GPU version of Caffe buffers used in DAUKernel*

template <typename Dtype>
class DAUKernelParamsGPU : public  DAUKernelParams<Dtype> {
public:

	//virtual Dtype* weight() { return this->weight_->mutable_gpu_flat<Dtype>().data(); }
	//virtual Dtype* mu1() { return this->mu1_->mutable_gpu_flat<Dtype>().data(); }
	//virtual Dtype* mu2() { return this->mu2_->mutable_gpu_flat<Dtype>().data(); }
	//virtual Dtype* sigma() { return this->sigma_->mutable_gpu_flat<Dtype>().data(); }

	//either keep this or change to auto?
	// also instead of matrix maybe use tensor->flat<Dtype>().data()
	virtual Dtype* weight() { Tensor* tmp_ten = *(this->weight_); auto flt = tmp_ten->flat<Dtype>(); auto dat = flt.data(); return static_cast<Dtype*>(dat); }
	virtual Dtype* mu1() { Tensor* tmp_ten = *(this->mu1_); auto flt = tmp_ten->flat<Dtype>(); auto dat = flt.data(); return static_cast<Dtype*>(dat);}
	virtual Dtype* mu2() { Tensor* tmp_ten = *(this->mu2_); auto flt = tmp_ten->flat<Dtype>(); auto dat = flt.data(); return static_cast<Dtype*>(dat);}
	virtual Dtype* sigma() { Tensor* tmp_ten = *(this->sigma_); auto flt = tmp_ten->flat<Dtype>(); auto dat = flt.data(); return static_cast<Dtype*>(dat); }

};

template <typename Dtype>
class DAUKernelOutputGPU : public DAUKernelOutput<Dtype> {
public:
	virtual Dtype* weight() { Tensor* tmp_ten = this->weight_;printf("Getting weight data\n"); auto dat = tmp_ten->flat<Dtype>().data();printf("Got flat\n"); return static_cast<Dtype*>(dat); }
	virtual Dtype* d_error() { Tensor* tmp_ten = this->d_error_; auto dat = tmp_ten->flat<Dtype>().data(); return static_cast<Dtype*>(dat); }
	virtual Dtype* d_params() { Tensor* tmp_ten = this->d_params_; auto dat = tmp_ten->flat<Dtype>().data(); return static_cast<Dtype*>(dat); }
};

template <typename Dtype>
class DAUKernelComputeGPU : public DAUKernelCompute<Dtype> {
public:

	explicit DAUKernelComputeGPU(OpKernelContext* context)
		: DAUKernelCompute<Dtype>(context), context_(context){}

	OpKernelContext* context_;

	virtual Dtype* param_temp(typename BaseDAUKernelCompute<Dtype>::Param_IDX index) { Tensor* tmp_ten = this->param_buffers_[index]; auto dat = tmp_ten->flat<Dtype>().data();
	return static_cast<Dtype*>(dat);}
	virtual Dtype* kernels_temp(typename BaseDAUKernelCompute<Dtype>::Kernel_IDX index) { Tensor* tmp_ten = this->param_buffers_[index]; auto dat = tmp_ten->flat<Dtype>().data();
	return static_cast<Dtype*>(dat);}
	virtual int* precomp_index() { Tensor* tmp_ten = this->tmp_precomp_index_; auto dat = tmp_ten->flat<int>().data();
	return static_cast<int*>(dat);}

};

//
template <typename Dtype>
class DAUKernelParamsCPU : public  DAUKernelParams<Dtype> {
public:

	virtual Dtype* weight() { Tensor* tmp_ten = this->weight_; auto dat = tmp_ten->flat<Dtype>().data(); return static_cast<Dtype*>(dat); }
	virtual Dtype* mu1() { Tensor* tmp_ten = this->mu1_; auto dat = tmp_ten->flat<Dtype>().data(); return static_cast<Dtype*>(dat);}
	virtual Dtype* mu2() { Tensor* tmp_ten = this->mu2_; auto dat = tmp_ten->flat<Dtype>().data(); return static_cast<Dtype*>(dat); }
	virtual Dtype* sigma() { Tensor* tmp_ten = this->sigma_; auto dat = tmp_ten->flat<Dtype>().data(); return static_cast<Dtype*>(dat); }
};

template <typename Dtype>
class DAUKernelOutputCPU : public DAUKernelOutput<Dtype> {
public:
	virtual Dtype* weight() { Tensor* tmp_ten = this->weight_; auto dat = tmp_ten->flat<Dtype>().data(); return static_cast<Dtype*>(dat); }
	virtual Dtype* d_error() { Tensor* tmp_ten  =this->d_error_; auto dat = tmp_ten->flat<Dtype>().data(); return static_cast<Dtype*>(dat); }
	virtual Dtype* d_params() { Tensor* tmp_ten = this->d_params_; auto dat = tmp_ten->flat<Dtype>().data(); return static_cast<Dtype*>(dat); }
};

template <typename Dtype>
class DAUKernelComputeCPU : public DAUKernelCompute<Dtype> {
public:

	virtual Dtype* param_temp(typename BaseDAUKernelCompute<Dtype>::Param_IDX index) { Tensor* tmp_ten = this->param_buffers_[index]; auto dat = tmp_ten->flat<Dtype>().data();
	return static_cast<Dtype*>(dat);}
	virtual Dtype* kernels_temp(typename BaseDAUKernelCompute<Dtype>::Kernel_IDX index) { Tensor* tmp_ten = this->param_buffers_[index]; auto dat = tmp_ten->flat<Dtype>().data();
	return static_cast<Dtype*>(dat);}
	virtual int* precomp_index() { Tensor* tmp_ten = this->tmp_precomp_index_; auto dat = tmp_ten->flat<Dtype>().data();
	return static_cast<Dtype*>(dat);}

};

////////////////////////////////////////////////////////////////////////////////
// Caffe GPU version of DAUConvolution layer (BaseDAUConvLayer)

template <typename Dtype>
class DAUComponentInitializerTensorflow : public BaseDAUComponentInitializer<Dtype> {
public:

	//replace FillerParameter with int for now?
	explicit DAUComponentInitializerTensorflow(int weight_filler,
								 int mu_filler,
								 int sigma_filler) :
			weight_filler_(weight_filler), mu_filler_(mu_filler), sigma_filler_(sigma_filler) {
	}

	void InitializeParameters(const DAUConvSettings& settings, Dtype* w, Dtype* mu1, Dtype* mu2, Dtype* sigma, bool is_gpu_ptr,
                                      int num_units_per_x, int num_units_per_y, int num_units_ignore,
									  int conv_in_channels, int conv_out_channels, int kernel_h, int kernel_w) const;
private:
	// param fillers
	// replace with ints for now?
	//FillerParameter weight_filler_;
	//FillerParameter mu_filler_;
	//FillerParameter sigma_filler_;
	int weight_filler_;
	int mu_filler_;
	int sigma_filler_;
};



template <typename Dtype>
class DAUConvLayerTensorflowGPU : public  BaseDAUConvLayer<Dtype> {
public:

	explicit DAUConvLayerTensorflowGPU(cublasHandle_t cublas_handle,OpKernelContext* context, bool ignore_edge_gradients = false)
			: BaseDAUConvLayer<Dtype>(cublas_handle, ignore_edge_gradients), context_(context),own_workspace_data(0), do_on_gpu_(true), cublasHandle(cublas_handle) {

	}

	virtual ~DAUConvLayerTensorflowGPU();

	virtual void LayerSetUp(const DAUConvSettings& settings,
							const BaseDAUComponentInitializer<Dtype>& param_initializer,
							BaseDAUKernelCompute<Dtype>* kernel_compute,
							BaseDAUKernelParams<Dtype>* kernel_param,
							BaseDAUKernelOutput<Dtype>* kernel_output,
							const vector<int>& bottom_shape, bool in_train = true);

	//Dtype* w, Dtype* mu1, Dtype* mu2, Dtype* sigma, bool is_gpu_ptr,
    //                                                           int num_units_per_x, int num_units_per_y, int num_units_ignore,
    //                                                           int conv_in_channels, int conv_out_channels, int kernel_h, int kernel_w
	virtual void InitializeFromInput(DAUConvSettings& settings, Tensor* w, Tensor* mu1, Tensor* mu2, Tensor* sigma);
	virtual void test_allocation();
	virtual vector<int> Reshape(const vector<int>& bottom_shape, const vector<int>& top);

	// make compute_output_shape() public
	virtual void compute_output_shape() { return BaseDAUConvLayer<Dtype>::compute_output_shape(); }

	void set_processing_on_gpu(bool do_on_gpu) { do_on_gpu_ = do_on_gpu; }

	// parameters to learn
	shared_ptr<Tensor* > param_buffer_w_;
	shared_ptr<Tensor* > param_buffer_mu1_;
	shared_ptr<Tensor* > param_buffer_mu2_;
	shared_ptr<Tensor* > param_buffer_sigma_;
	shared_ptr<Tensor* > param_buffer_bias_;
	OpKernelContext* context_ = NULL;
	Tensor* allocation_test = NULL;
	cublasHandle_t cublasHandle;
protected:
	virtual bool is_data_on_gpu() { return do_on_gpu_; }

    virtual void reshape_params(const vector<int>& shape) ;

	virtual bool update_prefiltering_kernels(cudaStream_t stream);

	// learnable parameters of size
	/*virtual Dtype* param_w() { return is_data_on_gpu() ? param_buffer_w_->mutable_gpu_flat<Dtype>().data() : param_buffer_w_->mutable_cpu_flat<Dtype>().data(); }
	virtual Dtype* param_mu1() { return is_data_on_gpu() ? param_buffer_mu1_->mutable_gpu_flat<Dtype>().data() : param_buffer_mu1_->mutable_cpu_flat<Dtype>().data(); }
	virtual Dtype* param_mu2() { return is_data_on_gpu() ? param_buffer_mu2_->mutable_gpu_flat<Dtype>().data() : param_buffer_mu2_->mutable_cpu_flat<Dtype>().data(); }
	virtual Dtype* param_sigma() { return is_data_on_gpu() ? param_buffer_sigma_->mutable_gpu_flat<Dtype>().data() : param_buffer_sigma_->mutable_cpu_flat<Dtype>().data(); }
	virtual Dtype* param_bias() { return is_data_on_gpu() ? param_buffer_bias_->mutable_gpu_flat<Dtype>().data() : param_buffer_bias_->mutable_cpu_flat<Dtype>().data(); }
	*/

	virtual Dtype* param_w() { 
		Tensor* tmp_ten = *(param_buffer_w_.get());		
		auto tdat = tmp_ten -> flat<Dtype>();
		auto dat = tdat.data();
		Dtype* t_dat = static_cast<Dtype*>(dat);
		return t_dat;
	}
	virtual Dtype* param_mu1() { Tensor* tmp_ten = *param_buffer_mu1_; auto dat = tmp_ten->flat<Dtype>().data(); return static_cast<Dtype*>(dat); }
	virtual Dtype* param_mu2() { Tensor* tmp_ten = *param_buffer_mu2_; auto dat = tmp_ten->flat<Dtype>().data(); return static_cast<Dtype*>(dat); }
	virtual Dtype* param_sigma() { Tensor* tmp_ten = *param_buffer_sigma_; auto dat = tmp_ten->flat<Dtype>().data(); return static_cast<Dtype*>(dat); }
	virtual Dtype* param_bias() { Tensor* tmp_ten = *param_buffer_bias_; auto dat = tmp_ten->flat<Dtype>().data(); return static_cast<Dtype*>(dat); }


	// gradient buffers for learnable parameters
	// implement after Op implementation
	// IMPLEMENTATION JUST FOR COMPILATION
	virtual Dtype* param_w_grad() { Tensor* tmp_ten = *param_buffer_w_; auto dat = tmp_ten -> flat<Dtype>().data(); return static_cast<Dtype*>(dat); }
	virtual Dtype* param_mu1_grad() { Tensor* tmp_ten = *param_buffer_mu1_; auto dat = tmp_ten->flat<Dtype>().data(); return static_cast<Dtype*>(dat); }
	virtual Dtype* param_mu2_grad() { Tensor* tmp_ten = *param_buffer_mu2_; auto dat = tmp_ten->flat<Dtype>().data(); return static_cast<Dtype*>(dat); }
	virtual Dtype* param_sigma_grad(){ Tensor* tmp_ten = *param_buffer_sigma_; auto dat = tmp_ten->flat<Dtype>().data(); return static_cast<Dtype*>(dat); }
	virtual Dtype* param_bias_grad() { Tensor* tmp_ten = *param_buffer_bias_; auto dat = tmp_ten->flat<Dtype>().data(); return static_cast<Dtype*>(dat); }
	//*/


	// remaining intermediate/temporary buffers

	/*	
	virtual Dtype* temp_bwd_gradients() { return is_data_on_gpu() ? bwd_gradients_.mutable_gpu_flat<Dtype>().data() : bwd_gradients_.mutable_cpu_flat<Dtype>().data() ; }
	virtual Dtype* temp_interm_buffer() { return is_data_on_gpu() ? interm_buffer_.mutable_gpu_flat<Dtype>().data() : interm_buffer_.mutable_cpu_flat<Dtype>().data() ; }
	virtual Dtype* temp_param_buffer() { return is_data_on_gpu() ? tmp_param_buffer_.mutable_gpu_flat<Dtype>().data() : tmp_param_buffer_.mutable_cpu_flat<Dtype>().data() ; }
	virtual Dtype* temp_col_buffer() { return is_data_on_gpu() ? col_buffer_.mutable_gpu_flat<Dtype>().data() : col_buffer_.mutable_cpu_flat<Dtype>().data() ; }
	virtual Dtype* temp_bias_multiplier() { return is_data_on_gpu() ? bias_multiplier_.mutable_gpu_flat<Dtype>().data() : bias_multiplier_.mutable_cpu_flat<Dtype>().data() ; }
	*/

	virtual Dtype* temp_bwd_gradients() { auto dat = bwd_gradients_->flat<Dtype>().data(); return static_cast<Dtype*>(dat); }
	virtual Dtype* temp_interm_buffer() { auto dat = interm_buffer_->flat<Dtype>().data(); return static_cast<Dtype*>(dat); }
	virtual Dtype* temp_param_buffer() { auto dat = tmp_param_buffer_->flat<Dtype>().data(); return static_cast<Dtype*>(dat); }
	virtual Dtype* temp_col_buffer() { auto dat = col_buffer_->flat<Dtype>().data(); return static_cast<Dtype*>(dat); }
	virtual Dtype* temp_bias_multiplier() { auto dat = bias_multiplier_->flat<Dtype>().data(); return static_cast<Dtype*>(dat); }


	

	virtual void* allocate_workspace_mem(size_t bytes);
	virtual void deallocate_workspace_mem();

	// accumulated gradients
	//Blob<Dtype> bwd_gradients_;
	Tensor* bwd_gradients_ = NULL;


	// additional buffers
	//Blob<Dtype> interm_buffer_; // GPU only
	//Blob<Dtype> tmp_param_buffer_; // GPU and CPU
	Tensor* interm_buffer_ = NULL; // GPU only
	Tensor* tmp_param_buffer_ = NULL; // GPU and CPU


	//Blob<Dtype> col_buffer_; // CPU only
	//Blob<Dtype> bias_multiplier_; // GPU and CPU
	Tensor* col_buffer_=NULL; // CPU only
	Tensor* bias_multiplier_=NULL; // GPU and CPU


	// workspace memory that we have allocated
	void* own_workspace_data = NULL;

	bool do_on_gpu_;
};


/**
 * DAUConvolutionLayer
 *
 * Implementation of Deep Compositional Layer that introduces two constraints which results in Displaced Aggregation
 * Units (DAU) as presented in CVPR18. This implementation is efficient and allows for 3-5 times faster computation
 * of inference and learning compared to Deep Compositional Layer from ICPR16 paper. This does introduces a slight
 * loss of information and is only an aproximation of the original GaussianConvLayer, but perofrmance is not impacted.
 *
 * DAUConvolutionLayer implements two constraints on composition/units :
 *  - single sigma/variance for the whole layer (shared across all features)
 *  - internal discretization of mu1/mu2 position values
 * Due to discretization of mu1/mu2 values this implementation handles sub-pixel offsets using bilinear interpolation
 * of input channels.
 *
 * Due to CUDA implementation this method does not compute accuretely on bottom/right border (1px). Those values
 * are used in gradient accumulation unless ignore_edge_gradients_ is set to true. Border values are back-propagated
 * nevertheless.
 *
 *
 * TODO:
 *  - add sharing of GPU memory accross layers that are computed in sequence
 *  - add stride>1 (currently allows only stride=1)
 *  - improve cudaStream for forward and backward pass
 *  - combine convolve and input preparation forward and backward pass (might shave 5-10% off the whole computation time)
 *
 *
 * @tparam Dtype
 */

/*
template <typename Dtype>
class DAUConvolutionLayer : public Layer<Dtype>{
public:

	explicit DAUConvolutionLayer(const LayerParameter& param, bool ignore_edge_gradients = false)
	  : Layer<Dtype>(param), dau_compute(Caffe::cublas_handle(), ignore_edge_gradients) {}

	virtual ~DAUConvolutionLayer();

	virtual inline const char* type() const { return "DAUConvolutionLayer"; }

	virtual inline int MinBottomBlobs() const { return 1; }
	virtual inline int MinTopBlobs() const { return 1; }
	virtual inline bool EqualNumBottomTopBlobs() const { return true; }

    virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
    virtual void Reshape(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);

	virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
	virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
	virtual void Backward_cpu(const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
	virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

protected:

	virtual void compute_output_shape() { return dau_compute.compute_output_shape(); }
	virtual inline bool reverse_dimensions() { return false; }

private:
	// compute obj and buffers (param and output) for our Gaussian kernel
	// (we actually have only one kernel in buffer but data structure is general)
	DAUKernelComputeGPU<Dtype> dau_kernel_compute;
	DAUKernelParamsGPU<Dtype> dau_kernel_params;
	DAUKernelOutputGPU<Dtype> dau_kernel_output;

public:
    // must be public only for testing reasons
    DAUConvLayerTensorflowGPU<Dtype> dau_compute;
};

//*/ //DAUConvolutionLayer

  // namespace caffe

#endif  // CAFFE_DAU_CONV_LAYER_HPP_
