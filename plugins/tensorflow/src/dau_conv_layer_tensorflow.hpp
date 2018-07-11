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


#define TENSOR_DATA_PTR(t, TYPE) (t == NULL ? NULL : reinterpret_cast<TYPE*>((t)->template flat<TYPE>().data()))
#define TENSOR_DATA_PTR_CONST(t, TYPE) (t == NULL ? NULL : reinterpret_cast<const TYPE*>((t)->template flat<TYPE>().data()))

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

	Tensor* weight_=  NULL;
    Tensor* mu1_= NULL;
    Tensor* mu2_ = NULL;
    Tensor* sigma_= NULL;

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
// GPU version of Tensorflow buffers used in DAUKernel*

template <typename Dtype>
class DAUKernelParamsTFGPU : public  DAUKernelParamsTF<Dtype> {
public:
    explicit DAUKernelParamsTFGPU(OpKernelContext* context)
    : DAUKernelParamsTF<Dtype>(context), context_(context){}

	virtual Dtype* weight() { return TENSOR_DATA_PTR(this->weight_, Dtype); }
	virtual Dtype* mu1() { return TENSOR_DATA_PTR(this->mu1_, Dtype); }
	virtual Dtype* mu2() { return TENSOR_DATA_PTR(this->mu2_, Dtype); }
	virtual Dtype* sigma() { return TENSOR_DATA_PTR(this->sigma_, Dtype); }

private:
    OpKernelContext* context_;

};

template <typename Dtype>
class DAUKernelOutputTFGPU : public DAUKernelOutputTF<Dtype> {
public:
    explicit DAUKernelOutputTFGPU(OpKernelContext* context)
    : DAUKernelOutputTF<Dtype>(context), context_(context){}

    virtual Dtype* weight() { return TENSOR_DATA_PTR(this->weight_, Dtype); }
	virtual Dtype* d_error() { return TENSOR_DATA_PTR(this->d_error_, Dtype); }
	virtual Dtype* d_params() { return TENSOR_DATA_PTR(this->d_params_, Dtype); }

private:
    OpKernelContext* context_;

};

template <typename Dtype>
class DAUKernelComputeTFGPU : public DAUKernelComputeTF<Dtype> {
public:

	explicit DAUKernelComputeTFGPU(OpKernelContext* context)
		: DAUKernelComputeTF<Dtype>(context), context_(context){}


	virtual Dtype* param_temp(typename BaseDAUKernelCompute<Dtype>::Param_IDX index) { return TENSOR_DATA_PTR(this->param_buffers_[index], Dtype); }
	virtual Dtype* kernels_temp(typename BaseDAUKernelCompute<Dtype>::Kernel_IDX index) { return TENSOR_DATA_PTR(this->param_buffers_[index], Dtype); }
	virtual int* precomp_index() { return TENSOR_DATA_PTR(this->tmp_precomp_index_, int); }

private:
    OpKernelContext* context_;

};

//
template <typename Dtype>
class DAUKernelParamsTFCPU : public DAUKernelParamsTF<Dtype> {
public:

	virtual Dtype* weight() { return TENSOR_DATA_PTR(this->weight_, Dtype); }
	virtual Dtype* mu1() { return TENSOR_DATA_PTR(this->mu1_, Dtype); }
	virtual Dtype* mu2() { return TENSOR_DATA_PTR(this->mu2_, Dtype); }
	virtual Dtype* sigma() { return TENSOR_DATA_PTR(this->sigma_, Dtype); }
};

template <typename Dtype>
class DAUKernelOutputTFCPU : public DAUKernelOutputTF<Dtype> {
public:
	virtual Dtype* weight() {return TENSOR_DATA_PTR(this->weight_, Dtype); }
	virtual Dtype* d_error() {return TENSOR_DATA_PTR(this->d_error_, Dtype); }
	virtual Dtype* d_params() {return TENSOR_DATA_PTR(this->d_params_, Dtype); }
};

template <typename Dtype>
class DAUKernelComputeTFCPU : public DAUKernelComputeTF<Dtype> {
public:

	virtual Dtype* param_temp(typename BaseDAUKernelCompute<Dtype>::Param_IDX index) { return TENSOR_DATA_PTR(this->param_buffers_[index], Dtype); }
	virtual Dtype* kernels_temp(typename BaseDAUKernelCompute<Dtype>::Kernel_IDX index) { return TENSOR_DATA_PTR(this->param_buffers_[index], Dtype); }
	virtual int* precomp_index() { return TENSOR_DATA_PTR(this->tmp_precomp_index_, Dtype); }

};

////////////////////////////////////////////////////////////////////////////////
// Tensorflow version of DAUComponentInitializer:
//  - variable initialization happens directly in Python so we do not need it here
//  - this implements empty initialization (no operation)

template <typename Dtype>
class NullDAUComponentInitializerTensorflow : public BaseDAUComponentInitializer<Dtype> {
public:

	explicit NullDAUComponentInitializerTensorflow(){
	}

	void InitializeParameters(const DAUConvSettings& settings, Dtype* w, Dtype* mu1, Dtype* mu2, Dtype* sigma, bool is_gpu_ptr,
                                      int num_units_per_x, int num_units_per_y, int num_units_ignore,
									  int conv_in_channels, int conv_out_channels, int kernel_h, int kernel_w) const {};
};

////////////////////////////////////////////////////////////////////////////////
// Tensorflow GPU version of DAUConvolution layer (BaseDAUConvLayer)

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
							const vector<int>& bottom_shape, int num_dau_units_ignore, bool in_train = true);

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
	const Tensor* param_buffer_w_ = NULL;
	const Tensor* param_buffer_mu1_ = NULL;
	const Tensor* param_buffer_mu2_ = NULL;
	const Tensor* param_buffer_sigma_ = NULL;
	const Tensor* param_buffer_bias_ = NULL;
    Tensor* param_buffer_w_grad = NULL;
    Tensor* param_buffer_mu1_grad = NULL;
    Tensor* param_buffer_mu2_grad = NULL;
    Tensor* param_buffer_sigma_grad = NULL;
    Tensor* param_buffer_bias_grad = NULL;

    OpKernelContext* context_ = NULL;
	cublasHandle_t cublasHandle;
	
protected:
	virtual bool is_data_on_gpu() { return do_on_gpu_; }

    virtual void reshape_params(const vector<int>& shape) ;

	virtual bool update_prefiltering_kernels(cudaStream_t stream);

    virtual Dtype* param_w() { return (Dtype*)TENSOR_DATA_PTR_CONST(param_buffer_w_, Dtype); }
    virtual Dtype* param_mu1() { return (Dtype*)TENSOR_DATA_PTR_CONST(param_buffer_mu1_, Dtype); }
    virtual Dtype* param_mu2() { return (Dtype*)TENSOR_DATA_PTR_CONST(param_buffer_mu2_, Dtype); }
    virtual Dtype* param_sigma() { return (Dtype*)TENSOR_DATA_PTR_CONST(param_buffer_sigma_, Dtype); }
    virtual Dtype* param_bias() { return (Dtype*)TENSOR_DATA_PTR_CONST(param_buffer_bias_, Dtype); }


    virtual Dtype* param_w_grad() { return TENSOR_DATA_PTR(param_buffer_w_grad, Dtype); }
    virtual Dtype* param_mu1_grad() { return TENSOR_DATA_PTR(param_buffer_mu1_grad, Dtype); }
    virtual Dtype* param_mu2_grad() { return TENSOR_DATA_PTR(param_buffer_mu2_grad, Dtype); }
    virtual Dtype* param_sigma_grad(){ return TENSOR_DATA_PTR(param_buffer_sigma_grad, Dtype); }
    virtual Dtype* param_bias_grad() { return TENSOR_DATA_PTR(param_buffer_bias_grad, Dtype); }

    // remaining intermediate/temporary buffers
    virtual Dtype* temp_bwd_gradients() { return TENSOR_DATA_PTR(bwd_gradients_, Dtype); }
    virtual Dtype* temp_interm_buffer() { return TENSOR_DATA_PTR(interm_buffer_, Dtype); }
    virtual Dtype* temp_param_buffer() { return TENSOR_DATA_PTR(tmp_param_buffer_, Dtype); }
    virtual Dtype* temp_col_buffer() { return TENSOR_DATA_PTR(col_buffer_, Dtype); }
    virtual Dtype* temp_bias_multiplier() { return TENSOR_DATA_PTR(bias_multiplier_, Dtype); }

	virtual void* allocate_workspace_mem(size_t bytes);
	virtual void deallocate_workspace_mem();

	// accumulated gradients
	Tensor* bwd_gradients_ = NULL;
	// additional buffers
	Tensor* interm_buffer_ = NULL; // GPU only
	Tensor* tmp_param_buffer_ = NULL; // GPU and CPU

	Tensor* col_buffer_= NULL; // CPU only
	Tensor* bias_multiplier_= NULL; // GPU and CPU

	// workspace memory that we have allocated
	void* own_workspace_data = NULL;
    //tensor that holds the workspace memory
    Tensor* own_workspace_tensor = NULL;

	bool do_on_gpu_;
};

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
