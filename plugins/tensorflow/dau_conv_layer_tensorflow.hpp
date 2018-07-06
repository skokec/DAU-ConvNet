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

    virtual ~DAUKernelParamsTF();

    void reshape(int num_in_channels, int num_out_channels, int num_gauss);

	void initialize_params(Tensor w, Tensor mu1, Tensor mu2, Tensor sigma);

	Tensor* weight_=  nullptr;
    Tensor* mu1_= nullptr;
    Tensor* mu2_ = nullptr;
    Tensor* sigma_= nullptr;
    // CPU for setting (once) GPU for computing, except for sigma

private:
	OpKernelContext* context_ = NULL;

};


template <typename Dtype>
class DAUKernelOutputTF : public BaseDAUKernelOutput<Dtype> {
public:
	explicit DAUKernelOutputTF(OpKernelContext* context)
	: context_(context){}

    virtual ~DAUKernelOutputTF();

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

	virtual Dtype* weight() {return reinterpret_cast<Dtype*>(this->weight_->template flat<Dtype>().data()); }
	virtual Dtype* mu1() { return reinterpret_cast<Dtype*>(this->mu1_->template flat<Dtype>().data());}
	virtual Dtype* mu2() { return reinterpret_cast<Dtype*>(this->mu2_->template flat<Dtype>().data());}
	virtual Dtype* sigma() {return reinterpret_cast<Dtype*>(this->sigma_->template flat<Dtype>().data()); }

private:
    OpKernelContext* context_;

};

template <typename Dtype>
class DAUKernelOutputTFGPU : public DAUKernelOutputTF<Dtype> {
public:
    explicit DAUKernelOutputTFGPU(OpKernelContext* context)
    : DAUKernelOutputTF<Dtype>(context), context_(context){}

    virtual Dtype* weight() { Tensor* tmp_ten = this->weight_; auto dat = tmp_ten->flat<Dtype>().data(); return reinterpret_cast<Dtype*>(dat); }
	virtual Dtype* d_error() { Tensor* tmp_ten = this->d_error_; auto dat = tmp_ten->flat<Dtype>().data(); return reinterpret_cast<Dtype*>(dat); }
	virtual Dtype* d_params() { Tensor* tmp_ten = this->d_params_; auto dat = tmp_ten->flat<Dtype>().data(); return reinterpret_cast<Dtype*>(dat); }

private:
    OpKernelContext* context_;

};

template <typename Dtype>
class DAUKernelComputeTFGPU : public DAUKernelComputeTF<Dtype> {
public:

	explicit DAUKernelComputeTFGPU(OpKernelContext* context)
		: DAUKernelComputeTF<Dtype>(context), context_(context){}


	virtual Dtype* param_temp(typename BaseDAUKernelCompute<Dtype>::Param_IDX index) { Tensor* tmp_ten = this->param_buffers_[index]; auto dat = tmp_ten->flat<Dtype>().data();
	return reinterpret_cast<Dtype*>(dat);}
	virtual Dtype* kernels_temp(typename BaseDAUKernelCompute<Dtype>::Kernel_IDX index) { Tensor* tmp_ten = this->param_buffers_[index]; auto dat = tmp_ten->flat<Dtype>().data();
	return reinterpret_cast<Dtype*>(dat);}
	virtual int* precomp_index() { Tensor* tmp_ten = this->tmp_precomp_index_; auto dat = tmp_ten->flat<int>().data();
	return reinterpret_cast<int*>(dat);}

private:
    OpKernelContext* context_;

};

//
template <typename Dtype>
class DAUKernelParamsTFCPU : public DAUKernelParamsTF<Dtype> {
public:

	virtual Dtype* weight() { return reinterpret_cast<Dtype*>(this->weight_->template flat<Dtype>().data());  }
	virtual Dtype* mu1() { return reinterpret_cast<Dtype*>(this->mu1_->template flat<Dtype>().data()); }
	virtual Dtype* mu2() { return reinterpret_cast<Dtype*>(this->mu2_->template flat<Dtype>().data()); }
	virtual Dtype* sigma() { return reinterpret_cast<Dtype*>(this->sigma_->template flat<Dtype>().data()); }
};

template <typename Dtype>
class DAUKernelOutputTFCPU : public DAUKernelOutputTF<Dtype> {
public:
	virtual Dtype* weight() {return reinterpret_cast<Dtype*>(this->weight_->template flat<Dtype>().data());}
	virtual Dtype* d_error() {return reinterpret_cast<Dtype*>(this->d_error_->template flat<Dtype>().data()); }
	virtual Dtype* d_params() {return reinterpret_cast<Dtype*>(this->d_params_->template flat<Dtype>().data()); }
};

template <typename Dtype>
class DAUKernelComputeTFCPU : public DAUKernelComputeTF<Dtype> {
public:

	virtual Dtype* param_temp(typename BaseDAUKernelCompute<Dtype>::Param_IDX index) { Tensor* tmp_ten = this->param_buffers_[index]; auto dat = tmp_ten->flat<Dtype>().data();
	return reinterpret_cast<Dtype*>(dat);}
	virtual Dtype* kernels_temp(typename BaseDAUKernelCompute<Dtype>::Kernel_IDX index) { Tensor* tmp_ten = this->param_buffers_[index]; auto dat = tmp_ten->flat<Dtype>().data();
	return reinterpret_cast<Dtype*>(dat);}
	virtual int* precomp_index() { Tensor* tmp_ten = this->tmp_precomp_index_; auto dat = tmp_ten->flat<Dtype>().data();
	return reinterpret_cast<Dtype*>(dat);}

};

////////////////////////////////////////////////////////////////////////////////
// Caffe GPU version of DAUConvolution layer (BaseDAUConvLayer)

template <typename Dtype>
class NullDAUComponentInitializerTensorflow : public BaseDAUComponentInitializer<Dtype> {
public:

	explicit NullDAUComponentInitializerTensorflow(){
	}

	void InitializeParameters(const DAUConvSettings& settings, Dtype* w, Dtype* mu1, Dtype* mu2, Dtype* sigma, bool is_gpu_ptr,
                                      int num_units_per_x, int num_units_per_y, int num_units_ignore,
									  int conv_in_channels, int conv_out_channels, int kernel_h, int kernel_w) const {};
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
	virtual vector<int> Reshape(const vector<int>& bottom_shape, const vector<int>& top);

	// make compute_output_shape() public
	virtual void compute_output_shape() { return BaseDAUConvLayer<Dtype>::compute_output_shape(); }

	void set_processing_on_gpu(bool do_on_gpu) { do_on_gpu_ = do_on_gpu; }

	// parameters to learn
	const Tensor* param_buffer_w_;
	const Tensor* param_buffer_mu1_;
	const Tensor* param_buffer_mu2_;
	const Tensor*param_buffer_sigma_;
	const Tensor* param_buffer_bias_;
    const Tensor* param_buffer_w_grad;
    const Tensor* param_buffer_mu1_grad;
    const Tensor* param_buffer_mu2_grad;
    const Tensor* param_buffer_sigma_grad;
    const Tensor* param_buffer_bias_grad;

    OpKernelContext* context_ = NULL;
	cublasHandle_t cublasHandle;

    bool is_forward_op;
protected:
	virtual bool is_data_on_gpu() { return do_on_gpu_; }

    virtual void reshape_params(const vector<int>& shape) ;

	virtual bool update_prefiltering_kernels(cudaStream_t stream);

    virtual Dtype* param_w() {return reinterpret_cast<Dtype*>( ((Tensor*) param_buffer_w_)->flat<Dtype>().data()); }
    virtual Dtype* param_mu1() {return reinterpret_cast<Dtype*>( ((Tensor*) param_buffer_mu1_)->flat<Dtype>().data());}
    virtual Dtype* param_mu2() {return reinterpret_cast<Dtype*>( ((Tensor*) param_buffer_mu2_)->flat<Dtype>().data());}
    virtual Dtype* param_sigma() {return reinterpret_cast<Dtype*>( ((Tensor*) param_buffer_sigma_)->flat<Dtype>().data());}
    //virtual Dtype* param_bias() {return reinterpret_cast<Dtype*>(((Tensor*) *param_buffer_bias_.get())->flat<Dtype>().data());}
    virtual Dtype* param_bias() {return NULL;}


    virtual Dtype* param_w_grad() {return reinterpret_cast<Dtype*>( ((Tensor*) param_buffer_w_grad)->flat<Dtype>().data());}
    virtual Dtype* param_mu1_grad() {return reinterpret_cast<Dtype*>( ((Tensor*) param_buffer_mu1_grad)->flat<Dtype>().data());}
    virtual Dtype* param_mu2_grad() {return reinterpret_cast<Dtype*>( ((Tensor*) param_buffer_mu2_grad)->flat<Dtype>().data());}
    virtual Dtype* param_sigma_grad(){return reinterpret_cast<Dtype*>( ((Tensor*) param_buffer_sigma_grad)->flat<Dtype>().data());}
    //virtual Dtype* param_bias_grad() {return reinterpret_cast<Dtype*>(((Tensor*) *param_buffer_bias_grad.get())->flat<Dtype>().data());}
    virtual Dtype* param_bias_grad() {return NULL;}

    // remaining intermediate/temporary buffers
    virtual Dtype* temp_bwd_gradients() {return reinterpret_cast<Dtype*>(bwd_gradients_->flat<Dtype>().data()); }
    virtual Dtype* temp_interm_buffer() {return reinterpret_cast<Dtype*>(interm_buffer_->flat<Dtype>().data()); }
    virtual Dtype* temp_param_buffer() {return reinterpret_cast<Dtype*>(tmp_param_buffer_->flat<Dtype>().data()); }
    virtual Dtype* temp_col_buffer() {return reinterpret_cast<Dtype*>(col_buffer_->flat<Dtype>().data()); }
    virtual Dtype* temp_bias_multiplier() {return reinterpret_cast<Dtype*>(bias_multiplier_->flat<Dtype>().data()); }

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
    //tensor that holds the workspace memory
    Tensor* own_workspace_tensor = NULL;

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

//OP_REQUIRES_OK uses return, problematic for compilation in non void functions
#define OP_REQUIRES_OK_BREAK(CTX, ...)                      \
  do {                                                      \
    ::tensorflow::Status _s(__VA_ARGS__);                   \
    if (!TF_PREDICT_TRUE(_s.ok())) {                        \
      (CTX)->CtxFailureWithWarning(__FILE__, __LINE__, _s); \
      break;                                                \
    }                                                       \
  } while (0)


#endif  // CAFFE_DAU_CONV_LAYER_HPP_
