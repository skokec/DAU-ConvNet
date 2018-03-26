//
// Created by domen on 3/22/18.
//

#ifndef DAU_CONV_BASE_DAU_CONV_LAYER_HPP_H
#define DAU_CONV_BASE_DAU_CONV_LAYER_HPP_H

#include <string>
#include <utility>
#include <vector>

#include "dau_conv/dau_conv_impl/dau_conv_backward.hpp"
#include "dau_conv/dau_conv_impl/dau_conv_forward.hpp"

#include "dau_conv/util/math_functions.hpp"
#include "dau_conv/util/common.hpp"

namespace DAUConvNet {
////////////////////////////////////////////////////////////////////////////////
// Base classes for buffers used in DAUKernel*

template<typename Dtype>
class BaseDAUKernelParams {
public:
    virtual void reshape(int num_in_channels, int num_out_channels, int num_gauss) = 0;

    virtual Dtype *weight() = 0;
    virtual Dtype *mu1() = 0;
    virtual Dtype *mu2() = 0;
    virtual Dtype *sigma() = 0;
};


template<typename Dtype>
class BaseDAUKernelOutput {
public:
    virtual void reshape(int num_in_channels, int num_out_channels, int num_gauss, int kernel_h, int kernel_w) = 0;

    virtual Dtype *weight() = 0;
    virtual Dtype *d_error() = 0;
    virtual Dtype *d_params() = 0;
};

template<typename Dtype>
class BaseDAUKernelCompute {
public:
    // for tensors/blobs of the size [1, num_in_channels, num_gauss, num_out_channels]
    enum Param_IDX {
        SIGMA_SQUARE_INV = 0,
        SIGMA_CUBE_INV = 1,
        SIGMA_SQUARE_INV_HALF = 2,
        GAUSS_NORM = 3,
        DERIV_MU1_SUMS = 4,
        DERIV_MU2_SUMS = 5,
        DERIV_SIGMA_SUMS = 6
    };
    // for tensors/blobs of the size [num_in_channels, num_gauss, num_out_channels, kernel_h * kernel_w]
    enum Kernel_IDX {
        GAUSS_DIST = 0,
        GAUSS_DIST_SQUARE = 1,
        DERIV_MU1_TIMES_GAUSS_DIST = 2,
        DERIV_MU2_TIMES_GAUSS_DIST = 3,
        DERIV_SIGMA_TIMES_GAUSS_DIST = 4
    };

    void setup(bool use_gmm_gauss_normalization,
               bool use_gmm_square_gauss_normalization,
               Dtype gmm_sigma_lower_bound,
               Dtype gmm_component_border_bound) {
        this->use_unit_normalization = use_gmm_gauss_normalization;
        this->use_square_unit_normalization = use_gmm_square_gauss_normalization;
        this->sigma_lower_bound = gmm_sigma_lower_bound;
        this->component_border_bound = gmm_component_border_bound;
    }

    void get_kernels(BaseDAUKernelParams<Dtype> &input, BaseDAUKernelOutput<Dtype> &output, cublasHandle_t cublas_handle);

    virtual void reshape(int num_in_channels, int num_out_channels, int num_gauss,
                         int kernel_h, int kernel_w) = 0;

    virtual Dtype *param_temp(Param_IDX index) = 0;
    virtual Dtype *kernels_temp(Kernel_IDX index) = 0;
    virtual int *precomp_index() = 0;

protected:
    int num_in_channels, num_out_channels, num_gauss;
    int kernel_h, kernel_w;
    bool use_unit_normalization;
    bool use_square_unit_normalization;
    Dtype sigma_lower_bound;
    Dtype component_border_bound;

};

struct DAUConvSettings {
    int num_output;
    vector<int> number_units;

    bool bias_term;

    // Pad, kernel size, and stride are all given as a single value for equal
    // dimensions in all spatial dimensions, or once per spatial dimension.
    int pad, kernel_size, stride;

    bool unit_normalization, square_unit_normalization;

    int mean_iteration_step, sigma_iteration_step;

    float component_border_bound, sigma_lower_bound;

    int merge_iteration_step;
    float merge_threshold;

};

template <typename Dtype>
class BaseDAUComponentInitializer {
public:

    virtual void InitializeParameters(const DAUConvSettings& settings, Dtype* w, Dtype* mu1, Dtype* mu2, Dtype* sigma, bool is_gpu_ptr,
                                      int num_units_per_x, int num_units_per_y, int num_units_ignore,
                                      int conv_in_channels, int conv_out_channels, int kernel_h, int kernel_w) const = 0;
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
template <typename Dtype>
class BaseDAUConvLayer {
public:
    // we setup with allowing num of DAUs only as a factor of two (due to CUDA implementation)
    // (when computing gradinets we need to make sure we ignore the last one if )

    static const int ALLOWED_UNITS_GROUP = 2;

    explicit BaseDAUConvLayer(cublasHandle_t cublas_handle, bool ignore_edge_gradients = false)
            : cublas_handle(cublas_handle), handles_setup_(false), ignore_edge_gradients_(ignore_edge_gradients) {
        this->aggregation.param = NULL;
        this->aggregation.kernels = NULL;
    }

    virtual ~BaseDAUConvLayer();


    virtual void LayerSetUp(const DAUConvSettings& settings,
                            const BaseDAUComponentInitializer<Dtype>& param_initializer,
                            BaseDAUKernelCompute<Dtype>* kernel_compute,
                            BaseDAUKernelParams<Dtype>* kernel_param,
                            BaseDAUKernelOutput<Dtype>* kernel_output,
                            const vector<int>& bottom_shape, bool in_train = true);

    virtual vector<int> Reshape(const vector<int>& bottom_shape, const vector<int>& top_shape);

    virtual void Forward_cpu(const Dtype* bottom_data, const vector<int> bottom_shape,
                             Dtype* top_data, const vector<int> top_shape);
    virtual void Backward_cpu(const Dtype* top_data, const Dtype* top_error, const vector<int>& top_shape, bool propagate_down,
                              const Dtype* bottom_data, Dtype* bottom_error, const vector<int>& bottom_shape, const vector<bool>& params_propagate_down );

    virtual void Forward_gpu(const Dtype* bottom_data, const vector<int> bottom_shape,
                             Dtype* top_data, const vector<int> top_shape);
    virtual void Backward_gpu(const Dtype* top_data, const Dtype* top_error, const vector<int>& top_shape, bool propagate_down,
                              const Dtype* bottom_data, Dtype* bottom_error, const vector<int>& bottom_shape, const vector<bool>& params_propagate_down );


protected:
    virtual void compute_output_shape();

    virtual bool is_data_on_gpu() = 0;

    virtual void reshape_params(const vector<int>& shape) = 0;

    // learnable parameters of size [1, conv_in_channels_, units_per_channel, conv_out_channels_]
    virtual Dtype* param_w() = 0;
    virtual Dtype* param_mu1() = 0;
    virtual Dtype* param_mu2() = 0;
    virtual Dtype* param_sigma() = 0;
    virtual Dtype* param_bias() = 0;

    // gradient buffers for learnable parameters
    virtual Dtype* param_w_grad() = 0;
    virtual Dtype* param_mu1_grad() = 0;
    virtual Dtype* param_mu2_grad() = 0;
    virtual Dtype* param_sigma_grad() = 0;
    virtual Dtype* param_bias_grad() = 0;

    // remaining intermediate/temporary buffers
    // requrested size = [NUM_K, conv_in_channels_, units_per_channel, conv_out_channels_]
    virtual Dtype* temp_bwd_gradients() = 0;

    // requrested size = [batch_num_, std::max(conv_in_channels_ * NUM_K, conv_out_channels_), max(height_,height_out_), max(width_,width_out_)]
    virtual Dtype* temp_interm_buffer() = 0;

    // requrested size = [2, conv_in_channels_, units_per_channel, conv_out_channels_]
    virtual Dtype* temp_param_buffer() = 0;

    // requrested size = [aggregation.kernel_h_, aggregation.kernel_w_, height_, width_]
    virtual Dtype* temp_col_buffer() = 0;

    // requrested size = [1, height_out_ * width_out_]
    virtual Dtype* temp_bias_multiplier() = 0;

    virtual void* allocate_workspace_mem(size_t bytes) = 0;
    virtual void deallocate_workspace_mem() = 0;

    Dtype get_sigma_val() {
        Dtype sigma;
        // if sigma ptr is on GPU we need to copy it
        if (this->is_data_on_gpu()) {
            CUDA_CHECK(cudaMemcpy(&sigma, this->param_sigma(), sizeof(Dtype), cudaMemcpyDefault));
        } else {
            sigma = this->param_sigma()[0];
        }
        return sigma;
    }


    void forward_cpu_bias(Dtype* output, const Dtype* bias);
    void backward_cpu_bias(Dtype* bias, const Dtype* input);

    void forward_gpu_bias(Dtype* output, const Dtype* bias);
    void backward_gpu_bias(Dtype* bias, const Dtype* input);

    virtual bool update_prefiltering_kernels(cudaStream_t stream = 0);

    Dtype* get_gaussian_kernel(cudaStream_t stream = 0);
    Dtype* get_deriv_kernel_params(cudaStream_t stream = 0);
    Dtype* get_deriv_kernel_error(cudaStream_t stream = 0);

    void set_last_n_gauss_to_zero(Dtype* array, int num_gauss_zero);

    // TODO: add support for K=4 as well (K== number of parameter types i.e., K=4 for [w,mu1,mu2,sigma])
    // NOTE: allthough we set NUM_K=4 we also set last_k_optional=true which allows underlaying system to
    //       ignore last K (i.e. sigma) since at the moment we are not training them
    const int NUM_K = 4;

    bool bias_term_;

    int num_spatial_axes_, channel_axis_;
    int bottom_dim_, top_dim_, out_spatial_dim_;

    int batch_num_;

    int conv_out_channels_, conv_in_channels_;

    int kernel_h_, kernel_w_;

    int stride_h_, stride_w_;
    int pad_h_, pad_w_;
    int height_, width_;
    int height_out_, width_out_;

    int units_per_channel;
    int num_units_ignore;

    bool use_unit_normalization;
    bool use_square_unit_normalization;

    Dtype unit_border_bound;
    Dtype unit_sigma_lower_bound;

    int mean_iteration_step;
    int sigma_iteration_step;
    int unit_merge_iteration_step;
    float unit_merge_threshold;

    int current_iteration_index;

    bool use_interpolation_;

    // since right/bottom edge values will not be computed properly we can ignore gradients at right/bottom image edge
    // NOTE: since gradients are not avegred but summed this should not be an issue, so this is used only for unit-testing (to make it compatible with cpu version)
    bool ignore_edge_gradients_ = false;
    bool last_k_optional = true;

    bool handles_setup_;

    cublasHandle_t cublas_handle;
    cudaStream_t*  stream_;
    cudaStream_t* paralel_streams; // parallel streams for custom back-propagation kernels

    // main classes for computing forward and backward pass
    std::shared_ptr<DAUConvNet::DAUConvBackward<Dtype> > backward_grad_obj;
    std::shared_ptr<DAUConvNet::DAUConvForward<Dtype> > backward_backporp_obj;
    std::shared_ptr<DAUConvNet::DAUConvForward<Dtype> > forward_obj;

    BaseDAUKernelCompute<Dtype>* kernel_compute;

    // aggregation related variables (i.e. (A)ggregation in DAU or bluring)
    struct {
        BaseDAUKernelParams<Dtype>* param;
        BaseDAUKernelOutput<Dtype>* kernels;

        int kernel_h_, kernel_w_;
        int pad_h_, pad_w_;
        int stride_h_, stride_w_;

        Dtype current_sigma;
    } aggregation;


    struct {
        Dtype* filtered_images;
        Dtype* filter_weights;
        int* filter_offsets;
        Dtype* filter_offsets_and_weights;

        size_t filtered_images_sizes_;
        size_t filter_weights_sizes_;
        size_t filter_offsets_sizes_;

        // this is used during backward pass for error back-propagation, but since we use forward pass for that we share
        // the same buffer, however, size of buffer must accommodate both
        size_t filtered_error_sizes_;
        size_t filter_error_weights_sizes_;
        size_t filter_error_offsets_sizes_;
    } buffer_fwd_;

    struct {
        Dtype* error_images;
        Dtype* filtered_images;
        Dtype* filter_weights;
        int* filter_offsets;
        Dtype* resized_top_for_bwd;

        size_t error_image_sizes_;
        size_t filtered_images_sizes_;
        size_t filter_weights_sizes_;
        size_t filter_offsets_sizes_;
        size_t resized_top_for_bwd_sizes_;
    } buffer_bwd_;

    // size of workspace storage that is used in buffer_fwd_ and buffer_bwd_
    // (this storage must be allocated by child who owns the data and needs to clean it up !!)
    size_t workspaceSizeInBytes;
};
}
#endif //DAU_CONV_BASE_DAU_CONV_LAYER_HPP_H
