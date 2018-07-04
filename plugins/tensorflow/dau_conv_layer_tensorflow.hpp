#ifndef CAFFE_DAU_CONV_LAYER_HPP_
#define CAFFE_DAU_CONV_LAYER_HPP_

#include <string>
#include <utility>
#include <vector>

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/platform/default/logging.h"
#include "tensorflow/core/framework/shape_inference.h"

#include "dau_conv/base_dau_conv_layer.hpp"


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
class DAUKernelParamsTF : public  BaseDAUKernelParams<Dtype> {
public:
	explicit DAUKernelParamsTF(OpKernelContext* context)
		: context_(context){}

	void reshape(int num_in_channels, int num_out_channels, int num_gauss);

	void initialize_params(Tensor w, Tensor mu1, Tensor mu2, Tensor sigma);

	shared_ptr<Tensor*> weight_, mu1_, mu2_, sigma_; // CPU for setting (once) GPU for computing, except for sigma

private:
	OpKernelContext* context_ = NULL;

};


template <typename Dtype>
class DAUKernelOutputTF : public BaseDAUKernelOutput<Dtype> {
public:
	explicit DAUKernelOutputTF(OpKernelContext* context)
	: context_(context){}

	virtual void reshape(int num_in_channels, int num_out_channels, int num_gauss, int kernel_h, int kernel_w);

	// main filter weights
	Tensor* weight_ = NULL;

	// derivative weights for back-propagation and all four parameters
	Tensor* d_error_ = NULL;
	Tensor* d_params_ = NULL; // four params == [w,mu1,mu2,sigma]

private:
	OpKernelContext* context_ = NULL;

};

template <typename Dtype>
class DAUKernelComputeTF : public BaseDAUKernelCompute<Dtype> {
public:
	explicit DAUKernelComputeTF(OpKernelContext* context)
		: context_(context){}

	virtual ~DAUKernelComputeTF();

	virtual void reshape(int num_in_channels, int num_out_channels, int num_gauss,
						 int kernel_h, int kernel_w);


protected:
	void create_precompute_index(const int index_size, const int kernel_size);

	// intermediate buffers when computing derivative kernels in precompute_guassian_weights_gpu
	// temporary buffers for pre-computed sigma^2, sigma^3 and 1/2*sigma^2
	vector<Tensor*> param_buffers_;
	vector<Tensor*> kernels_buffers_;

	// pre-computed indexes for caffe_gpu_sum in get_kernels
	Tensor* tmp_precomp_index_ = NULL;

private:
	OpKernelContext* context_ = NULL;

};


////////////////////////////////////////////////////////////////////////////////
// GPU version of Caffe buffers used in DAUKernel*

template <typename Dtype>
class DAUKernelParamsTFGPU : public  DAUKernelParamsTF<Dtype> {
public:
    explicit DAUKernelParamsTFGPU(OpKernelContext* context)
    : DAUKernelParamsTF<Dtype>(context), context_(context){}

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

private:
    OpKernelContext* context_;

};

template <typename Dtype>
class DAUKernelOutputTFGPU : public DAUKernelOutputTF<Dtype> {
public:
    explicit DAUKernelOutputTFGPU(OpKernelContext* context)
    : DAUKernelOutputTF<Dtype>(context), context_(context){}

    virtual Dtype* weight() { Tensor* tmp_ten = this->weight_; auto dat = tmp_ten->flat<Dtype>().data(); return static_cast<Dtype*>(dat); }
	virtual Dtype* d_error() { Tensor* tmp_ten = this->d_error_; auto dat = tmp_ten->flat<Dtype>().data(); return static_cast<Dtype*>(dat); }
	virtual Dtype* d_params() { Tensor* tmp_ten = this->d_params_; auto dat = tmp_ten->flat<Dtype>().data(); return static_cast<Dtype*>(dat); }

private:
    OpKernelContext* context_;

};

template <typename Dtype>
class DAUKernelComputeTFGPU : public DAUKernelComputeTF<Dtype> {
public:

	explicit DAUKernelComputeTFGPU(OpKernelContext* context)
		: DAUKernelComputeTF<Dtype>(context), context_(context){}


	virtual Dtype* param_temp(typename BaseDAUKernelCompute<Dtype>::Param_IDX index) { Tensor* tmp_ten = this->param_buffers_[index]; auto dat = tmp_ten->flat<Dtype>().data();
	return static_cast<Dtype*>(dat);}
	virtual Dtype* kernels_temp(typename BaseDAUKernelCompute<Dtype>::Kernel_IDX index) { Tensor* tmp_ten = this->param_buffers_[index]; auto dat = tmp_ten->flat<Dtype>().data();
	return static_cast<Dtype*>(dat);}
	virtual int* precomp_index() { Tensor* tmp_ten = this->tmp_precomp_index_; auto dat = tmp_ten->flat<int>().data();
	return static_cast<int*>(dat);}

private:
    OpKernelContext* context_;

};

//
template <typename Dtype>
class DAUKernelParamsTFCPU : public DAUKernelParamsTF<Dtype> {
public:

	virtual Dtype* weight() { Tensor* tmp_ten = (Tensor*) this->weight_; auto dat = tmp_ten->flat<Dtype>().data(); return static_cast<Dtype*>(dat); }
	virtual Dtype* mu1() { Tensor* tmp_ten = (Tensor*) this->mu1_; auto dat = tmp_ten->flat<Dtype>().data(); return static_cast<Dtype*>(dat);}
	virtual Dtype* mu2() { Tensor* tmp_ten = (Tensor*) this->mu2_; auto dat = tmp_ten->flat<Dtype>().data(); return static_cast<Dtype*>(dat); }
	virtual Dtype* sigma() { Tensor* tmp_ten = (Tensor*) this->sigma_; auto dat = tmp_ten->flat<Dtype>().data(); return static_cast<Dtype*>(dat); }
};

template <typename Dtype>
class DAUKernelOutputTFCPU : public DAUKernelOutputTF<Dtype> {
public:
	virtual Dtype* weight() { Tensor* tmp_ten = this->weight_; auto dat = tmp_ten->flat<Dtype>().data(); return static_cast<Dtype*>(dat); }
	virtual Dtype* d_error() { Tensor* tmp_ten  =this->d_error_; auto dat = tmp_ten->flat<Dtype>().data(); return static_cast<Dtype*>(dat); }
	virtual Dtype* d_params() { Tensor* tmp_ten = this->d_params_; auto dat = tmp_ten->flat<Dtype>().data(); return static_cast<Dtype*>(dat); }
};

template <typename Dtype>
class DAUKernelComputeTFCPU : public DAUKernelComputeTF<Dtype> {
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
	virtual void InitializeGrad(DAUConvSettings& settings, Tensor* w_grad, Tensor* mu1_grad, Tensor* mu2_grad, Tensor* sigma_grad);
	virtual void test_allocation();
	virtual vector<int> Reshape(const vector<int>& bottom_shape, const vector<int>& top);

	// make compute_output_shape() public
	virtual void compute_output_shape() { return BaseDAUConvLayer<Dtype>::compute_output_shape(); }

	void set_processing_on_gpu(bool do_on_gpu) { do_on_gpu_ = do_on_gpu; }

	// parameters to learn
	shared_ptr<const Tensor* > param_buffer_w_;
	shared_ptr<const Tensor* > param_buffer_mu1_;
	shared_ptr<const Tensor* > param_buffer_mu2_;
	shared_ptr<const Tensor* > param_buffer_sigma_;
	shared_ptr<const Tensor* > param_buffer_bias_;
    shared_ptr<const Tensor* > param_buffer_w_grad;
    shared_ptr<const Tensor* > param_buffer_mu1_grad;
    shared_ptr<const Tensor* > param_buffer_mu2_grad;
    shared_ptr<const Tensor* > param_buffer_sigma_grad;
    shared_ptr<const Tensor* > param_buffer_bias_grad;

    OpKernelContext* context_ = NULL;
	Tensor* allocation_test = NULL;
	cublasHandle_t cublasHandle;
protected:
	virtual bool is_data_on_gpu() { return do_on_gpu_; }

    virtual void reshape_params(const vector<int>& shape) ;

	virtual bool update_prefiltering_kernels(cudaStream_t stream);

	// learnable parameters of size

	virtual Dtype* param_w() { Tensor* tmp_ten = (Tensor*) *(param_buffer_w_.get()); auto t_flat = tmp_ten -> flat<Dtype>(); return static_cast<Dtype*>(t_flat.data()); }
	virtual Dtype* param_mu1() { Tensor* tmp_ten = (Tensor*) *(param_buffer_mu1_.get()); auto dat = tmp_ten->flat<Dtype>().data(); return static_cast<Dtype*>(dat); }
	virtual Dtype* param_mu2() { Tensor* tmp_ten = (Tensor*) *(param_buffer_mu2_.get()); auto dat = tmp_ten->flat<Dtype>().data(); return static_cast<Dtype*>(dat); }
	virtual Dtype* param_sigma() { Tensor* tmp_ten = (Tensor*) *(param_buffer_sigma_.get()); auto dat = tmp_ten->flat<Dtype>().data(); return static_cast<Dtype*>(dat); }
	virtual Dtype* param_bias() { Tensor* tmp_ten = (Tensor*) *(param_buffer_bias_.get()); auto dat = tmp_ten->flat<Dtype>().data(); return static_cast<Dtype*>(dat); }


	// gradient buffers for learnable parameters
	// implement after Op implementation
	// IMPLEMENTATION JUST FOR COMPILATION
	virtual Dtype* param_w_grad() { Tensor* tmp_ten = (Tensor*) *(param_buffer_w_grad.get()); auto dat = tmp_ten -> flat<Dtype>().data(); return static_cast<Dtype*>(dat); }
	virtual Dtype* param_mu1_grad() { Tensor* tmp_ten = (Tensor*) *(param_buffer_mu1_grad.get()); auto dat = tmp_ten->flat<Dtype>().data(); return static_cast<Dtype*>(dat); }
	virtual Dtype* param_mu2_grad() { Tensor* tmp_ten = (Tensor*) *(param_buffer_mu2_grad.get()); auto dat = tmp_ten->flat<Dtype>().data(); return static_cast<Dtype*>(dat); }
	virtual Dtype* param_sigma_grad(){ Tensor* tmp_ten = (Tensor*) *(param_buffer_sigma_grad.get()); auto dat = tmp_ten->flat<Dtype>().data(); return static_cast<Dtype*>(dat); }
	virtual Dtype* param_bias_grad() { Tensor* tmp_ten = (Tensor*) *(param_buffer_bias_grad.get()); auto dat = tmp_ten->flat<Dtype>().data(); return static_cast<Dtype*>(dat); }
	//*/


	// remaining intermediate/temporary buffers

	virtual Dtype* temp_bwd_gradients() { auto dat = bwd_gradients_->flat<Dtype>().data(); return static_cast<Dtype*>(dat); }
	virtual Dtype* temp_interm_buffer() { auto dat = interm_buffer_->flat<Dtype>().data(); return static_cast<Dtype*>(dat); }
	virtual Dtype* temp_param_buffer() { auto dat = tmp_param_buffer_->flat<Dtype>().data(); return static_cast<Dtype*>(dat); }
	virtual Dtype* temp_col_buffer() { auto dat = col_buffer_->flat<Dtype>().data(); return static_cast<Dtype*>(dat); }
	virtual Dtype* temp_bias_multiplier() { auto dat = bias_multiplier_->flat<Dtype>().data(); return static_cast<Dtype*>(dat); }


	

	virtual void* allocate_workspace_mem(size_t bytes);
	virtual void deallocate_workspace_mem();

	// accumulated gradients
	Tensor* bwd_gradients_ = NULL;


	// additional buffers
	Tensor* interm_buffer_ = NULL; // GPU only
	Tensor* tmp_param_buffer_ = NULL; // GPU and CPU


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


#endif  // CAFFE_DAU_CONV_LAYER_HPP_
