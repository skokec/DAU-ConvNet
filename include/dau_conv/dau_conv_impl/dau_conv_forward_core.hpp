#include <cuda_runtime_api.h>
#include <cmath>

#include "dau_conv_forward.hpp"
#include "dau_conv/util/math_functions.hpp"

#include <cub/cub.cuh>

namespace DAUConvNet {

// TODO: using hardcoded warp size may not be portable (should use warpSize) but this way allows compiler optimization and avoids using dynamic memory allocation
#define WARP_SIZE 32

#define MIN(x,y) (x < y ? x : y)
#define MAX(x,y) (x > y ? x : y)


#define OFFSET(l,k,j,i, num_l, num_k, num_j, num_i) ((( (l)*(num_k) + (k)) * (num_j) + (j))*(num_i) + (i) )

#define OFFSET6(n, m, l,k,j,i, num_n, num_m, num_l, num_k, num_j, num_i) ((( ((((n) * (num_m)) + (m))*(num_l) + (l))*(num_k) + (k)) * (num_j) + (j))*(num_i) + (i) )

#define OFFSET5(m, l,k,j,i, num_m, num_l, num_k, num_j, num_i) ((( ((m)*(num_l) + (l))*(num_k) + (k)) * (num_j) + (j))*(num_i) + (i) )

#define OFFSET8(i8, i7, i6, i5, i4, i3, i2, i1, num_i8, num_i7, num_i6, num_i5, num_i4, num_i3, num_i2, num_i1) \
            ((( (( ( ((i8) * (num_i7) + i7)* (num_i6)  + (i6)  )*(num_i5)  + (i5)  )   * (num_i4) + (i4))*(num_i3) + (i3)) * (num_i2) + (i2))*(num_i1) + (i1) )

#define OFFSET9(i9, i8, i7, i6, i5, i4, i3, i2, i1, num_i9, num_i8, num_i7, num_i6, num_i5, num_i4, num_i3, num_i2, num_i1) \
            ((( (( ( (((i9)*(num_i8) + i8) * (num_i7) + i7)* (num_i6)  + (i6)  )*(num_i5)  + (i5)  )   * (num_i4) + (i4))*(num_i3) + (i3)) * (num_i2) + (i2))*(num_i1) + (i1) )

#define CEILING(x,y) (((x) + (y) - 1) / (y))

#define IS_VALID_PIXEL(X,Y,MAX_X,MAX_Y) (X >= 0 && X < MAX_X && Y >= 0 && Y < MAX_Y)

struct  __builtin_align__(16) ptr4
{
    float* quad[4];
};


template <int _NUM_SM,
        int _Bx, int _By,
        int _BLOCK_FEATURES,
        int _BLOCK_IMAGES,
        int _BATCH_PIXELS_SIZE_X,
        int _BATCH_PIXELS_SIZE_Y,
        int _PIXELS_INTERPOLATION_Dx,
        int _PIXELS_INTERPOLATION_Dy,
        int _BATCH_FEATURES_SIZE,
        int _BATCH_COMPUTE_FEATURES_SIZE,
        int _BATCH_COMPUTE_SUBFEATURES_SIZE,
        int _BATCH_MEM_SUBFEATURES_SIZE,
        int _BATCH_GAUSS_SIZE,
        int _BATCH_IMAGES,
        int _IMG_WIDTH, int _IMG_HEIGHT,
        int _MAX_OFFSET,
        bool _USE_SEPARATE_WEIGHTS_AND_OFFSETS,
        int _LOAD_DATA_INDEX,
        int _LOAD_OFFSET_INDEX>
class BlockIndexing {
public:

    enum {
        NUM_SM = _NUM_SM,
        Bx = _Bx,
        By = _By,
        BLOCK_FEATURES = _BLOCK_FEATURES,
        BLOCK_IMAGES = _BLOCK_IMAGES,
        BATCH_PIXELS_SIZE_X = _BATCH_PIXELS_SIZE_X,
        BATCH_PIXELS_SIZE_Y = _BATCH_PIXELS_SIZE_Y,
        PIXELS_INTERPOLATION_Dx = _PIXELS_INTERPOLATION_Dx,
        PIXELS_INTERPOLATION_Dy = _PIXELS_INTERPOLATION_Dy,
        BATCH_FEATURES_SIZE = _BATCH_FEATURES_SIZE,
        BATCH_COMPUTE_FEATURES_SIZE = _BATCH_COMPUTE_FEATURES_SIZE,
        BATCH_COMPUTE_SUBFEATURES_SIZE = _BATCH_COMPUTE_SUBFEATURES_SIZE,
        BATCH_MEM_SUBFEATURES_SIZE = _BATCH_MEM_SUBFEATURES_SIZE,
        BATCH_GAUSS_SIZE = _BATCH_GAUSS_SIZE,
        BATCH_IMAGES = _BATCH_IMAGES,
        IMG_WIDTH = _IMG_WIDTH,
        IMG_HEIGHT = _IMG_HEIGHT,
        MAX_OFFSET = _MAX_OFFSET,
        NUM_THREADS = Bx* By * BLOCK_FEATURES,
        USE_SEPARATE_WEIGHTS_AND_OFFSETS = _USE_SEPARATE_WEIGHTS_AND_OFFSETS,
        LOAD_DATA_INDEX = _LOAD_DATA_INDEX,
        LOAD_OFFSET_INDEX = _LOAD_OFFSET_INDEX,
    };

    // CPU only functions
    class Launch {
    public:
        dim3 getThreadsPerBlock(int num_images, int num_features, int num_subfeatures, int img_width, int img_height) {
            // number of threads per blocks
            return dim3(Bx * By * BLOCK_FEATURES * BLOCK_IMAGES, 1, 1);
        }

        dim3 getBlocksPerGrid(int num_images, int num_features, int num_subfeatures, int num_gaussian, int img_width, int img_height) {
            checkInputSize(num_features, BLOCK_FEATURES * BATCH_FEATURES_SIZE, "num_features");
            checkInputSize(num_subfeatures, BATCH_COMPUTE_SUBFEATURES_SIZE * BATCH_MEM_SUBFEATURES_SIZE, "num_subfeatures");
            checkInputSize(img_width, Bx * BATCH_PIXELS_SIZE_X, "img_width", false);
            checkInputSize(img_height, By * BATCH_PIXELS_SIZE_Y, "img_height", false);
            checkInputSize(num_gaussian, BATCH_GAUSS_SIZE, "num_gaussian");
            checkInputSize(num_images, BATCH_IMAGES*BLOCK_IMAGES, "num_images");


            int num_feature_blocks = (int)ceil(num_features/(BLOCK_FEATURES * BATCH_FEATURES_SIZE));
            int num_gaussian_blocks = (int)ceil(num_gaussian/(float)(BATCH_GAUSS_SIZE));


            int num_pixel_blocks_x = (int)ceil(img_width /  (float)(Bx * BATCH_PIXELS_SIZE_X) );
            int num_pixel_blocks_y = (int)ceil(img_height / (float)(By * BATCH_PIXELS_SIZE_Y) );

            int num_image_blocs = (int)ceil((num_images / (BATCH_IMAGES*BLOCK_IMAGES)) / (float)(NUM_SM) );

            // number of blocks per kernel launch
            return dim3 ( num_feature_blocks * num_gaussian_blocks,
                          num_pixel_blocks_x * num_pixel_blocks_y,
                          num_image_blocs
            );
        }
        void checkInputSize(int input_size, int min_allowed, const std::string& param_name, bool allow_only_multiple_of_min = true) {
            if (input_size < min_allowed) {
                throw DAUConvNet::DAUException(string_format("Invalid %s value of %d in DAUConvForwardCUDA. Min allowed %d.\n", param_name.c_str(), input_size, min_allowed));
            }
            if (allow_only_multiple_of_min && input_size % min_allowed != 0) {
                throw DAUConvNet::DAUException(string_format("Invalid %s value of %d in DAUConvForwardCUDA. Only a multiple of %d allowed.\n", param_name.c_str(), input_size, min_allowed));
            }
        }
    };

    // GPU only functions
    class Kernel {
    public:
        int2 img_size;

        int img_thread_idx;
        int f_thread_idx;
        int px_thread_idx;

        int img_block_idx;
        int f_block_idx;
        int g_block_idx;

        __device__ Kernel(int img_width, int img_height, int G) {
            img_size.x = img_width;
            img_size.y = img_height;

            img_thread_idx = threadIdx.x % BLOCK_IMAGES;
            f_thread_idx = (threadIdx.x / BLOCK_IMAGES) / (Bx * By);
            px_thread_idx = (threadIdx.x / BLOCK_IMAGES) % (Bx * By);

            f_block_idx = blockIdx.x / (G / BATCH_GAUSS_SIZE);
            g_block_idx = blockIdx.x % (G / BATCH_GAUSS_SIZE);
            img_block_idx = blockIdx.z;
        }

        // return global image index that thread block handles
        __device__ int getImageBlockIdx() {
            //return img_block_idx * (BLOCK_IMAGES * BATCH_IMAGES) + img_thread_idx * BATCH_IMAGES;
            return img_block_idx;
        }

        __device__ int getImageIdx() {
            return img_thread_idx;
        }
        // return global feature index that specific thread handles
        // since each thread handles multiple features (BATCH_FEATURES_SIZE) and each block handles
        // multiple features as well (BLOCK_FEATURES) this returns offset to F that specific thread will use
        __device__ int getFeatureIdx() {
            return f_block_idx * (BLOCK_FEATURES * BATCH_FEATURES_SIZE)  + f_thread_idx * BATCH_FEATURES_SIZE;
        }

        // return local index that specific thread handles
        // since one block handles multiple feature (BLOCK_FEATURES) this returns index of feature for within one block
        __device__ int getFeatureBlockIdx() {
            return f_thread_idx * BATCH_FEATURES_SIZE;
        }

        __device__ int getSubfeatureIdx() {
            return 0;
        }

        __device__ int getGaussianBlockIdx() {
            return g_block_idx * BATCH_GAUSS_SIZE;
        }

        __device__ int2 getPosBlockSize() {
            return make_int2(Bx * BATCH_PIXELS_SIZE_X,
                             By * BATCH_PIXELS_SIZE_Y);
        }

        __device__ int2 getPosBlockIdx() {

            int blockIdx_x = blockIdx.y % (img_size.x / (Bx * BATCH_PIXELS_SIZE_X));
            int blockIdx_y = blockIdx.y / (img_size.x / (Bx * BATCH_PIXELS_SIZE_X));

            return make_int2(BATCH_PIXELS_SIZE_X * (blockIdx_x * Bx),
                             BATCH_PIXELS_SIZE_Y * (blockIdx_y * By));
        }

        __device__ int2 getPosThreadIdx() {

            int threadIdx_x = px_thread_idx % (Bx);
            int threadIdx_y = px_thread_idx / (Bx);

            return make_int2(BATCH_PIXELS_SIZE_X * threadIdx_x,
                             BATCH_PIXELS_SIZE_Y * threadIdx_y);
        }
        __device__ int2 getInterleavedPosThreadIdx() {

            int threadIdx_x = px_thread_idx % (Bx);
            int threadIdx_y = px_thread_idx / (Bx);

            return make_int2(threadIdx_x * BATCH_PIXELS_SIZE_X * BATCH_IMAGES * BLOCK_IMAGES + img_thread_idx,
                             (threadIdx_y * BATCH_PIXELS_SIZE_Y )  );

        }
        static __forceinline__ __device__ unsigned warp_lane_id()
        {
            unsigned ret;
            asm volatile ("mov.u32 %0, %laneid;" : "=r"(ret));
            return ret;
        }
    };
};



template <int _NUM_THREADS, int _WIDTH, int _HEIGHT, int _APRON_SIZE_X, int _APRON_SIZE_Y, int _NUM_BUFFER_REPEAT, typename _ELEMENT_TYPE, int _BATCH_ELEMENTS>
class BlockSharedMemory {

public:
    typedef _ELEMENT_TYPE ELEMENT_TYPE;

    enum {
        NUM_THREADS = _NUM_THREADS,
        WIDTH = _WIDTH,
        HEIGHT = _HEIGHT,
        APRON_SIZE_X = _APRON_SIZE_X,
        APRON_SIZE_Y = _APRON_SIZE_Y,
        NUM_BUFFER_REPEAT = _NUM_BUFFER_REPEAT,
        BATCH_ELEMENTS = _BATCH_ELEMENTS,

        PIXELS_X = WIDTH + 2*APRON_SIZE_X,
        PIXELS_Y = HEIGHT + 2*APRON_SIZE_Y,

        ALLOC_HEIGHT = PIXELS_Y,

        // we can load N pixels with one LOAD operation so buffer needs to be a multiple of N
        // also make sure to add LOAD for last few pixels
        ALLOC_WIDTH = (PIXELS_X + BATCH_ELEMENTS-1)/BATCH_ELEMENTS,

        // PITCHED_WIDTH == actualy width of allocated data in floats
        PITCHED_WIDTH = ALLOC_WIDTH * BATCH_ELEMENTS,

        // actual size of buffer for each [WIDTH x HEIGHT] patch in number of basic elements (i.e. float)
        // inclues padding to make sure it alights with BATCH_ELEMENTS
        PATCH_SIZE = PITCHED_WIDTH *  ALLOC_HEIGHT,

        // distribute number  of threads per width and height
        // assign consecutive tid to adjecent memory elemnets where each id handled N-elements (N==BATCH_ELEMENTS)
        WIDTH_TRANSFER = PIXELS_X >= 64 ? PIXELS_X : ( PIXELS_X >= 32 ? 32 : (PIXELS_X >= 16 ? 16 : (PIXELS_X >= 8 ? 8 : 4))),
        NUM_THREADS_WIDTH = MIN(NUM_THREADS, WIDTH_TRANSFER / BATCH_ELEMENTS),
        NUM_THREADS_HEIGHT = MAX(1, NUM_THREADS / NUM_THREADS_WIDTH),

        // number of iterations that will be performed during load/store by one thread
        NUM_ITERATION_Y = MAX(1,CEILING((HEIGHT + 2*APRON_SIZE_Y), NUM_THREADS_HEIGHT)),
        NUM_ITERATION_X = MAX(1, CEILING(WIDTH + 2*APRON_SIZE_X,NUM_THREADS_WIDTH * BATCH_ELEMENTS))
    };

private:

    typedef BlockSharedMemory<NUM_THREADS, WIDTH, HEIGHT, _APRON_SIZE_X, _APRON_SIZE_Y, NUM_BUFFER_REPEAT, ELEMENT_TYPE, BATCH_ELEMENTS> BlockSharedMemoryT;

    struct _Data {
        ELEMENT_TYPE data[NUM_BUFFER_REPEAT][ALLOC_HEIGHT][ALLOC_WIDTH];
    };
    struct _LoadingData {
        ELEMENT_TYPE data[NUM_ITERATION_Y][NUM_ITERATION_X];
    };
public:
    float* storage_data_for_writing;
    float* storage_data_for_reading;


    _Data& storage;

    // thread indexing for storing/writing data from global mem
    int2 thread_indexing_writing;

    // thread indexing for reading data by each thread (MUST be user defined in constructor)
    int2 thread_indexing_reading;


    typedef _Data Data;
    typedef _LoadingData LoadingData;

    __device__
    BlockSharedMemory(Data &_storage, int2 read_thread_idx) : storage(_storage), thread_indexing_reading(read_thread_idx) {
        thread_indexing_writing = calcThreadIdx();
        storage_data_for_writing = getDataAt(0, thread_indexing_writing.x/ BATCH_ELEMENTS, thread_indexing_writing.y);
        storage_data_for_reading = getDataAt(0, (thread_indexing_reading.x + APRON_SIZE_X) / BATCH_ELEMENTS, thread_indexing_reading.y + APRON_SIZE_Y) + (thread_indexing_reading.x + APRON_SIZE_X) % BATCH_ELEMENTS;
    }

    __device__
    float* getData(int buffer_index = 0) {
        return reinterpret_cast<float*>(storage.data[buffer_index]);
    }
    __device__
    float* getDataThreadIndexingWrite(int buffer_index = 0){
        return storage_data_for_writing + buffer_index * ALLOC_HEIGHT * ALLOC_WIDTH * sizeof(ELEMENT_TYPE) / sizeof(float);
    }

    __device__
    float* getDataThreadIndexingRead(int buffer_index = 0){
        // TODO: if using BATCH_ELEMENTS > 1 then storage_data_for_reading points to the first group of 4 elements; it works OK if using replicate (NUM_BUFFER_REPEAT>1) but will not produce correct indexing for NUM_BUFFER_REPEAT=1
        // TODO: indexing for reading should be fixed !!!
        // TODO: could probably be fixed by adding:     (thread_indexing_reading.x + APRON_SIZE) % BATCH_ELEMENTS

        return storage_data_for_reading + buffer_index * ALLOC_HEIGHT * ALLOC_WIDTH * sizeof(ELEMENT_TYPE) / sizeof(float);
    }
    __device__
    float* getDataAt(int index, int x, int y){
        return reinterpret_cast<float*>(&storage.data[index][y][x]);
    }

    template <typename T>
    __device__
    size_t getOffsetAt(int i, int j) {
        return j * ALLOC_WIDTH  * sizeof(ELEMENT_TYPE) / sizeof(T) + i;
    }

    __device__
    int2& getThreadIdx() {
        return this->thread_indexing_writing;
    }

    template <int _GLOBAL_DATA_WIDTH, int REPLICATE_OFFSETED, bool USE_FILL, int FILL_VALUE> // GLOBAL_WIDTH .. size of image row in ELEMENT_TYPE elements i.e. if ELEMENT_TYPE == float4 then GLOBAL_WIDTH counts 4 floats as one
    __device__
    void load_global(const ELEMENT_TYPE* global_data, ELEMENT_TYPE* shared_data, int GLOBAL_DATA_WIDTH = -1, LoadingData *loaded_data = NULL) {

        if (GLOBAL_DATA_WIDTH < 0)
            GLOBAL_DATA_WIDTH = _GLOBAL_DATA_WIDTH;

        // global_data MUST be positioned at [0,0] in global data without APRON, i.e., at [APRON,APRON] / BATCH_ELEMENTS  in shared storage.data
#pragma unroll
        for (int j = -APRON_SIZE_Y; j < HEIGHT + APRON_SIZE_Y; j+=NUM_THREADS_HEIGHT) {
#pragma unroll
            for (int i = -APRON_SIZE_X; i < WIDTH + APRON_SIZE_X; i+=NUM_THREADS_WIDTH * BATCH_ELEMENTS) {

                // load data without if guard (NOTE: must ensure data will not be read outside of global mem!!)
                // lodaing without if guard eliminates extra MOV operation and allows STS (store) to happen after processing occures
                ELEMENT_TYPE tmp;

                // USING GLOBAL - working
                if (USE_FILL) {
                    if (BATCH_ELEMENTS > 0) tmp.x = FILL_VALUE;
                    if (BATCH_ELEMENTS > 1) tmp.y = FILL_VALUE;
                    if (BATCH_ELEMENTS > 2) tmp.z = FILL_VALUE;
                    if (BATCH_ELEMENTS > 3) tmp.w = FILL_VALUE;
                } else {
                    tmp = global_data[j * GLOBAL_DATA_WIDTH / BATCH_ELEMENTS + i / BATCH_ELEMENTS];
                }

                if (thread_indexing_writing.x < (WIDTH + APRON_SIZE_X - i)  && thread_indexing_writing.y < HEIGHT + APRON_SIZE_Y - j)  {


                    int write_offset = (j + APRON_SIZE_Y) * ALLOC_WIDTH  + (i + APRON_SIZE_X) / BATCH_ELEMENTS;


                    if (loaded_data != NULL) {
                        loaded_data->data[(j + APRON_SIZE_Y)/NUM_THREADS_HEIGHT][(i + APRON_SIZE_X) / (NUM_THREADS_WIDTH * BATCH_ELEMENTS)] = tmp;
                    } else {
                        // load to sharred data
                        shared_data[write_offset] = tmp;

                        // replicate the value several times in an offested manner to enable alinged access using float2 or float4 even for unalinged offsets
                        for (int replication_index = 0; replication_index < REPLICATE_OFFSETED; ++replication_index) {
                            ELEMENT_TYPE* replication_shared_data = shared_data + (replication_index+1) * ALLOC_HEIGHT*ALLOC_WIDTH;

                            if (BATCH_ELEMENTS > 0) reinterpret_cast<float*>(replication_shared_data + write_offset)[-1 * (replication_index+1) + 0] = tmp.x;
                            if (BATCH_ELEMENTS > 1) reinterpret_cast<float*>(replication_shared_data + write_offset)[-1 * (replication_index+1) + 1] = tmp.y;
                            if (BATCH_ELEMENTS > 2) reinterpret_cast<float*>(replication_shared_data + write_offset)[-1 * (replication_index+1) + 2] = tmp.z;
                            if (BATCH_ELEMENTS > 3) reinterpret_cast<float*>(replication_shared_data + write_offset)[-1 * (replication_index+1) + 3] = tmp.w;
                        }
                    }
                }
            }
        }
        //print();
    }


    template <int REPLICATE_OFFSETED>
    __device__
    void store_shared(const LoadingData& loaded_data, ELEMENT_TYPE* shared_data) {

#pragma unroll
        for (int j = -APRON_SIZE_Y; j < HEIGHT + APRON_SIZE_Y; j+=NUM_THREADS_HEIGHT) {
#pragma unroll
            for (int i = -APRON_SIZE_X; i < WIDTH + APRON_SIZE_X; i+=NUM_THREADS_WIDTH * BATCH_ELEMENTS) {
                // current_image already at position for this block

                if (thread_indexing_writing.x < (WIDTH + APRON_SIZE_X - i)  && thread_indexing_writing.y < HEIGHT + APRON_SIZE_Y - j)  {

                    int write_offset = (j + APRON_SIZE_Y) * ALLOC_WIDTH  + (i + APRON_SIZE_X) / BATCH_ELEMENTS;

                    int jj = (j + APRON_SIZE_Y)/NUM_THREADS_HEIGHT;
                    int ii = (i + APRON_SIZE_X) / (NUM_THREADS_WIDTH * BATCH_ELEMENTS);

                    ELEMENT_TYPE tmp = loaded_data.data[jj][ii];

                    // load to shared data
                    shared_data[write_offset] = tmp;

                    // replicate the value several times in an offested manner to enable alinged access using float2 or float4 even for unalinged offsets
                    for (int replication_index = 0; replication_index < REPLICATE_OFFSETED; ++replication_index) {
                        ELEMENT_TYPE* replication_shared_data = shared_data + (replication_index+1) * ALLOC_HEIGHT*ALLOC_WIDTH;

                        if (BATCH_ELEMENTS > 0) reinterpret_cast<float*>(replication_shared_data + write_offset)[-1 * (replication_index+1) + 0] = tmp.x;
                        if (BATCH_ELEMENTS > 1) reinterpret_cast<float*>(replication_shared_data + write_offset)[-1 * (replication_index+1) + 1] = tmp.y;
                        if (BATCH_ELEMENTS > 2) reinterpret_cast<float*>(replication_shared_data + write_offset)[-1 * (replication_index+1) + 2] = tmp.z;
                        if (BATCH_ELEMENTS > 3) reinterpret_cast<float*>(replication_shared_data + write_offset)[-1 * (replication_index+1) + 3] = tmp.w;
                    }
                }
            }
        }
        //print();
    }

    __device__
    void print() {
        __syncthreads();
        if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0 && blockIdx.y == 0) {


            printf("printing shared memory:\n");

            for (int s = 0; s < NUM_BUFFER_REPEAT; ++s) {
                for (int j = 0; j < ALLOC_HEIGHT; ++j){
                    for (int i = 0; i < ALLOC_WIDTH; ++i){
                        float4 tmp = storage.data[s][j][i];
                        printf("%d %d %d %d ", (int)tmp.x, (int)tmp.y, (int)tmp.z, (int)tmp.w);
                    }
                    printf("\n");
                }
                printf("\nend of NUM_BUFFER_REPEAT %d\n",s);
            }
            printf("\nend of double buffer\n");
        }

        __syncthreads();
    }

private:
    __device__
    static int2 calcThreadIdx() {
        // thread indexes for using to load shared memory
        // we will load N-pixels at onece using LDG.128 and STS.128 so
        // indexing will account for N-pixels in a row handled by a single thread,
        // however since threadIdx may be partitioned differenty, we now need to re-index it

        int thread_index = (blockDim.y * threadIdx.z + threadIdx.y) * blockDim.x + threadIdx.x;

        int2 new_index;
        new_index.x = (thread_index % (NUM_THREADS_WIDTH)) * BATCH_ELEMENTS; // mod by New_x
        new_index.y = thread_index / (NUM_THREADS_WIDTH ); // mod by New_y

        return new_index;
    }

};

template <int _SIZE>
class NDIndexingZero {
public:
    enum {
        SIZE = _SIZE
    };

    template< int DIM>
    static __device__
    int getIndex(int index) {
        if (DIM == 0)
            return index % SIZE;
        else
            return -1;
    }
    static __device__
    int getElementSize() {
        return SIZE;
    }
};

template <int _SIZE, class _PARENT >
class NDIndexing {
public:
    enum {
        SIZE = _SIZE
    };
    typedef _PARENT PARENT;

    template< int DIM>
    static __device__
    int getIndex(int index) {
        if (DIM > 0)
            return PARENT::getIndex<DIM-1>(index);
        else
            return (index / PARENT::getElementSize()) % SIZE;
    }

    static __device__
    int getElementSize() {
        return SIZE * PARENT::getElementSize();
    }
};



template <int BATCH_PIXELS_SIZE_X,
        int BATCH_PIXELS_SIZE_Y,
        bool BATCH_PIXELS_BY_WIDTH,
        int PIXELS_INTERPOLATION_Dx,
        int PIXELS_INTERPOLATION_Dy,
        int BATCH_FEATURES_SIZE,
        int BATCH_COMPUTE_FEATURES_SIZE,
        int BATCH_COMPUTE_SUBFEATURES_SIZE,
        int BATCH_MEM_SUBFEATURES_SIZE,
        int BLOCK_FEATURES,
        int BATCH_IMAGES,
        int BLOCK_IMAGES,
        int BATCH_PIXELS_FLOAT4,
        typename  _BlockSharedMemoryT>
class PipelineEngine {

    enum {
        PIXELS_INTERPOLATION_SIZE = PIXELS_INTERPOLATION_Dx * PIXELS_INTERPOLATION_Dy,
    };

    _BlockSharedMemoryT& shared_mem;
public:
    typedef _BlockSharedMemoryT BlockSharedMemoryT;
    typedef typename _BlockSharedMemoryT::ELEMENT_TYPE ELEMENT_TYPE;

    __device__
    PipelineEngine(BlockSharedMemoryT& shared_mem_)
            : shared_mem(shared_mem_) {
    }

    struct {
//		typename BlockSharedMemoryT::LoadingData ld;

        bool enabled;
        typename BlockSharedMemoryT::ELEMENT_TYPE const* reading_ptr;
        typename BlockSharedMemoryT::ELEMENT_TYPE * writing_ptr;
        int img_read_width;
    } load_global;

    // load offset
    struct {
        bool enabled;
        int* offset_address;
        float* base_address;
        ptr4* output;
    } load_offset;

    // load w
    struct {
        bool enabled;
        float const* address;
        float4* output;
    } load_weights;

    // load data
    struct {
        bool enabled;
        ptr4* address;
        float4* output;	// [BATCH_F][BATCH_PIXELS/4]
    } load_data;

    // compute
    struct {
        bool enabled;
        float4* weights;
        float4* data;
        float4* output; // [BATCH_F][BATCH_PIXELS/4]
    } compute;

    // block
    int block_x;
    int block_y;

    int thread_x;
    int thread_y;

    int interleaved_thread_x;
    int interleaved_thread_y;

    int s_index;
    int f_index;
    int g_index;

    int image_index;
    int warp_id;

#define COPY_VECTOR4(Y,X) \
{ \
(Y).x = (X).x; \
(Y).y = (X).y; \
(Y).z = (X).z; \
(Y).w = (X).w; \
}

    __device__
    bool should_run(int current_index, int unit_start_delay, int max_iter) {
        return (current_index - unit_start_delay >= 0 && current_index - unit_start_delay < max_iter  ? true : false );
    }

    __device__
    void execute_step() {
        static const int NUM_READ_FEATURES =  BATCH_COMPUTE_FEATURES_SIZE >= 4 ? 4 :
                                              (BATCH_COMPUTE_FEATURES_SIZE >= 2 ? 2 : 1);
        // load quad of w for next one
        if (load_weights.enabled) {
            for (int i = 0; i < PIXELS_INTERPOLATION_SIZE; ++i) {
                for (int f_quad_index = 0; f_quad_index < BATCH_COMPUTE_FEATURES_SIZE/NUM_READ_FEATURES; ++f_quad_index ) {
                    // weights for F[0], F[1], F[2], F[3]
                    if (BATCH_COMPUTE_FEATURES_SIZE > 0) load_weights.output[i* BATCH_COMPUTE_FEATURES_SIZE/NUM_READ_FEATURES + f_quad_index].x = load_weights.address[(i * BATCH_FEATURES_SIZE/NUM_READ_FEATURES  + f_quad_index) * BLOCK_FEATURES * NUM_READ_FEATURES + 0];
                    if (BATCH_COMPUTE_FEATURES_SIZE > 1) load_weights.output[i* BATCH_COMPUTE_FEATURES_SIZE/NUM_READ_FEATURES + f_quad_index].y = load_weights.address[(i * BATCH_FEATURES_SIZE/NUM_READ_FEATURES  + f_quad_index) * BLOCK_FEATURES * NUM_READ_FEATURES + 1];
                    if (BATCH_COMPUTE_FEATURES_SIZE > 2) load_weights.output[i* BATCH_COMPUTE_FEATURES_SIZE/NUM_READ_FEATURES + f_quad_index].z = load_weights.address[(i * BATCH_FEATURES_SIZE/NUM_READ_FEATURES  + f_quad_index) * BLOCK_FEATURES * NUM_READ_FEATURES + 2];
                    if (BATCH_COMPUTE_FEATURES_SIZE > 3) load_weights.output[i* BATCH_COMPUTE_FEATURES_SIZE/NUM_READ_FEATURES + f_quad_index].w = load_weights.address[(i * BATCH_FEATURES_SIZE/NUM_READ_FEATURES  + f_quad_index) * BLOCK_FEATURES * NUM_READ_FEATURES + 3];

                }
            }
        }

        // load quad of offsets for next one and make it directly into pointer to data
        if (load_offset.enabled) {
            //*(p.next_offset) = *((float4*)p.next_offset_address);

            for (int f_quad_index = 0; f_quad_index < BATCH_COMPUTE_FEATURES_SIZE/NUM_READ_FEATURES; ++f_quad_index ) {
                if (BATCH_COMPUTE_FEATURES_SIZE > 0) load_offset.output[f_quad_index].quad[0] = (float*)((char*)load_offset.base_address + load_offset.offset_address[f_quad_index * NUM_READ_FEATURES + 0]); // F[0]
                if (BATCH_COMPUTE_FEATURES_SIZE > 1) load_offset.output[f_quad_index].quad[1] = (float*)((char*)load_offset.base_address + load_offset.offset_address[f_quad_index * NUM_READ_FEATURES + 1]); // F[1]
                if (BATCH_COMPUTE_FEATURES_SIZE > 2) load_offset.output[f_quad_index].quad[2] = (float*)((char*)load_offset.base_address + load_offset.offset_address[f_quad_index * NUM_READ_FEATURES + 2]); // F[2]
                if (BATCH_COMPUTE_FEATURES_SIZE > 3) load_offset.output[f_quad_index].quad[3] = (float*)((char*)load_offset.base_address + load_offset.offset_address[f_quad_index * NUM_READ_FEATURES + 3]); // F[3]

            }
        }


        NDIndexing<BATCH_IMAGES,
                NDIndexing<PIXELS_INTERPOLATION_Dx,
                NDIndexing<(BATCH_PIXELS_SIZE_X * BATCH_PIXELS_SIZE_Y)/BATCH_PIXELS_FLOAT4,
                NDIndexing<PIXELS_INTERPOLATION_Dy,
                NDIndexingZero<BATCH_COMPUTE_FEATURES_SIZE> > > > > indexing;

#pragma unroll
        for (int i = 0; i < BATCH_IMAGES * BATCH_COMPUTE_FEATURES_SIZE * PIXELS_INTERPOLATION_SIZE * (BATCH_PIXELS_SIZE_X * BATCH_PIXELS_SIZE_Y) /BATCH_PIXELS_FLOAT4; ++i) {

            // i goes over [BATCH_N][BATCH_FEATURES_SIZE][PIXELS_INTERPOLATION_SIZE][BATCH_PIXELS_SIZE_/4] array so get indexes for both manually
            int n = indexing.getIndex<0>(i);
            int f = indexing.getIndex<4>(i);
            int interpolation_j = indexing.getIndex<3>(i);
            int interpolation_i = indexing.getIndex<1>(i);
            int px = indexing.getIndex<2>(i);

            // since we store weight and offset into float4/int4 we need a proper index to access array of quad vectors
            int f_quad_index = f/NUM_READ_FEATURES;

            int px_x = px % (BATCH_PIXELS_BY_WIDTH ? BATCH_PIXELS_SIZE_X/BATCH_PIXELS_FLOAT4 : BATCH_PIXELS_SIZE_X);
            int px_y = px / (BATCH_PIXELS_BY_WIDTH ? BATCH_PIXELS_SIZE_X/BATCH_PIXELS_FLOAT4 : BATCH_PIXELS_SIZE_X);

            // since array batches 4 pixels in float4 then get actual px address by multiplying with 4
            px_x = px_x * (BATCH_PIXELS_BY_WIDTH ?  BATCH_PIXELS_FLOAT4 : 1);
            px_y = px_y * (BATCH_PIXELS_BY_WIDTH ?  1 : BATCH_PIXELS_FLOAT4);

            // add interpolation offset to px_x and px_y
            px_x = px_x + interpolation_i;
            px_y = px_y + interpolation_j;

            int data_index = OFFSET5(f, n, interpolation_j, interpolation_i, px,
                                     BATCH_COMPUTE_FEATURES_SIZE, BATCH_IMAGES, PIXELS_INTERPOLATION_Dy, PIXELS_INTERPOLATION_Dx, (BATCH_PIXELS_SIZE_X * BATCH_PIXELS_SIZE_Y) /BATCH_PIXELS_FLOAT4 );

            // load data for next loop
            if (load_data.enabled) {

                int data_address_index = f_quad_index;
                int data_quad_index = f % NUM_READ_FEATURES;

                if (BATCH_PIXELS_BY_WIDTH) {
                    //printf("loading data from address: %llu for f:%d, px_x: %d, px_y: %d and px: %d\n", load_data.address[f_quad_index].quad[f % 4] + px_x + (px_y) * BlockSharedMemoryT::PITCHED_WIDTH, f, px_x, px_y, px);
                    if (BATCH_PIXELS_BY_WIDTH >= 4)
                        load_data.output[data_index] = reinterpret_cast<float4*>(load_data.address[data_address_index].quad[data_quad_index] + px_x * BATCH_IMAGES * BLOCK_IMAGES + (px_y) * BlockSharedMemoryT::PITCHED_WIDTH)[0];
                    else if (BATCH_PIXELS_BY_WIDTH >= 2){
                        load_data.output[data_index].x = reinterpret_cast<float2*>(load_data.address[data_address_index].quad[data_quad_index] + px_x * BATCH_IMAGES * BLOCK_IMAGES + (px_y) * BlockSharedMemoryT::PITCHED_WIDTH)[0].x;
                        load_data.output[data_index].y = reinterpret_cast<float2*>(load_data.address[data_address_index].quad[data_quad_index] + px_x * BATCH_IMAGES * BLOCK_IMAGES + (px_y) * BlockSharedMemoryT::PITCHED_WIDTH)[0].y;
                    }

                } else {
                    /*if (BATCH_PIXELS_FLOAT4 > 0) load_data.output[data_index].x = reinterpret_cast<float2*>(load_data.address[data_address_index].quad[data_quad_index] + px_x* BATCH_IMAGES * BLOCK_IMAGES + (px_y + 0) * BlockSharedMemoryT::PITCHED_WIDTH)[0];
                    if (BATCH_PIXELS_FLOAT4 > 1) load_data.output[data_index].y = reinterpret_cast<float2*>(load_data.address[data_address_index].quad[data_quad_index] + px_x* BATCH_IMAGES * BLOCK_IMAGES + (px_y + 1) * BlockSharedMemoryT::PITCHED_WIDTH)[0];
                    if (BATCH_PIXELS_FLOAT4 > 2) load_data.output[data_index].z = reinterpret_cast<float2*>(load_data.address[data_address_index].quad[data_quad_index] + px_x* BATCH_IMAGES * BLOCK_IMAGES + (px_y + 2) * BlockSharedMemoryT::PITCHED_WIDTH)[0];
                    if (BATCH_PIXELS_FLOAT4 > 3) load_data.output[data_index].w = reinterpret_cast<float2*>(load_data.address[data_address_index].quad[data_quad_index] + px_x* BATCH_IMAGES * BLOCK_IMAGES + (px_y + 3) * BlockSharedMemoryT::PITCHED_WIDTH)[0];
                    */
                    float* addr_x = load_data.address[data_address_index].quad[data_quad_index] + px_x * BATCH_IMAGES * BLOCK_IMAGES + n + (px_y + 0) * BlockSharedMemoryT::PITCHED_WIDTH;

                    if (n == 0) {
                        if (BATCH_IMAGES <= 1) {

                            if (interpolation_i == 0) {

                                float val = reinterpret_cast<float *>(addr_x)[0];
                                load_data.output[data_index].x = val;
                            } else {
                                int prev_data_index = OFFSET5(f, n, interpolation_j, interpolation_i-1, px,
                                                              BATCH_COMPUTE_FEATURES_SIZE, BATCH_IMAGES, PIXELS_INTERPOLATION_Dy, PIXELS_INTERPOLATION_Dx, (BATCH_PIXELS_SIZE_X * BATCH_PIXELS_SIZE_Y) /BATCH_PIXELS_FLOAT4 );
                                // get the value from neigbooring pixels
                                //load_data.output[data_index].x = __shfl_up(load_data.output[prev_data_index].x, warp_id,1);
                                load_data.output[data_index].x = load_data.output[prev_data_index].x;

                            }
                        }  else if (BATCH_IMAGES <= 2) {
                            float2 val = reinterpret_cast<float2*>(addr_x)[0];

                            int data_index_next_n = OFFSET5(f, n+1, interpolation_j, interpolation_i, px,
                                                            BATCH_COMPUTE_FEATURES_SIZE, BATCH_IMAGES, PIXELS_INTERPOLATION_Dy, PIXELS_INTERPOLATION_Dx, (BATCH_PIXELS_SIZE_X * BATCH_PIXELS_SIZE_Y) /BATCH_PIXELS_FLOAT4 );
                            load_data.output[data_index].x = val.x;
                            load_data.output[data_index_next_n].x = val.y;
                        } else {
                            float4 val = reinterpret_cast<float4*>(addr_x)[0];

                            int data_index_next_n1 = OFFSET5(f, n+1, interpolation_j, interpolation_i, px,
                                                             BATCH_COMPUTE_FEATURES_SIZE, BATCH_IMAGES, PIXELS_INTERPOLATION_Dy, PIXELS_INTERPOLATION_Dx, (BATCH_PIXELS_SIZE_X * BATCH_PIXELS_SIZE_Y) /BATCH_PIXELS_FLOAT4 );
                            int data_index_next_n2 = OFFSET5(f, n+2, interpolation_j, interpolation_i, px,
                                                             BATCH_COMPUTE_FEATURES_SIZE, BATCH_IMAGES, PIXELS_INTERPOLATION_Dy, PIXELS_INTERPOLATION_Dx, (BATCH_PIXELS_SIZE_X * BATCH_PIXELS_SIZE_Y) /BATCH_PIXELS_FLOAT4 );
                            int data_index_next_n3 = OFFSET5(f, n+3, interpolation_j, interpolation_i, px,
                                                             BATCH_COMPUTE_FEATURES_SIZE, BATCH_IMAGES, PIXELS_INTERPOLATION_Dy, PIXELS_INTERPOLATION_Dx, (BATCH_PIXELS_SIZE_X * BATCH_PIXELS_SIZE_Y) /BATCH_PIXELS_FLOAT4 );
                            load_data.output[data_index].x = val.x;
                            load_data.output[data_index_next_n1].x = val.y;
                            load_data.output[data_index_next_n2].x = val.z;
                            load_data.output[data_index_next_n3].x = val.w;

                        }
                    }
                    /*if ((thread_x == 1 || thread_x == 0) && thread_y == 0 && f_index == 0 && f == 0 && px_x == 0 && px_y == 0 && block_x == 0 && block_y == 0 && interpolation_j == 0 && interpolation_i == 0)
                    {
                        printf("thread j,i: %d,%d, with data load address: %p and data value %f\n ", thread_y, thread_x, addr_x, addr_x);
                        //printf("computed sum %f from current value from w %f * data %f = computed_value %f at interpolation index j,i=%d,%d, s=%d, f=%d, and block y,x=%d,%d\n",
                        //	   sum_val, w, data_org, computeed_val, interpolation_j,interpolation_i, s_index, f_index + f, block_y, block_x);
                    }*/
                }
            }

            // compute for current loop
            if (compute.enabled) {

                // weights index must include interpolation index to get interpolation weight
                int weights_index = (interpolation_j * PIXELS_INTERPOLATION_Dx + interpolation_i) * BATCH_COMPUTE_FEATURES_SIZE/NUM_READ_FEATURES +  f_quad_index;

                // compute index must NOT include interpolation index since we sum all interpolation values into the same output
                int compute_index = OFFSET5(0, f, n, interpolation_i == 0 ? 0 : 1, px,
                                            1,BATCH_FEATURES_SIZE, BATCH_IMAGES,2,(BATCH_PIXELS_SIZE_X * BATCH_PIXELS_SIZE_Y)/BATCH_PIXELS_FLOAT4);

                float w = 0;

                if ((f % NUM_READ_FEATURES) % 4 == 0)
                    w = compute.weights[weights_index].x;
                else if ((f % NUM_READ_FEATURES) % 4 == 1)
                    w = compute.weights[weights_index].y;
                else if ((f % NUM_READ_FEATURES) % 4 == 2)
                    w = compute.weights[weights_index].z;
                else
                    w = compute.weights[weights_index].w;

                if (BATCH_PIXELS_FLOAT4 > 0) compute.output[compute_index].x += w * compute.data[data_index].x;
                if (BATCH_PIXELS_FLOAT4 > 1) compute.output[compute_index].y += w * compute.data[data_index].y;
                if (BATCH_PIXELS_FLOAT4 > 2) compute.output[compute_index].z += w * compute.data[data_index].z;
                if (BATCH_PIXELS_FLOAT4 > 3) compute.output[compute_index].w += w * compute.data[data_index].w;

                {
                    /*float data_org = compute.data[data_index].x;
                    float computeed_val = w * compute.data[data_index].x;
                    float sum_val = compute.output[compute_index].x;*/
                    /*if ((thread_x == 0 ) && thread_y == 0 && f_index == 0 && f == 0 && px_x == 0 && px_y == 0 && block_x == 0 && block_y == 0 && interpolation_j == 0 && interpolation_i == 0 && image_index == 2 && g_index == 0)
                    {

                        //printf("thread j,i: %d,%d, with data read offset: %d and data value %f\n ", thread_y, thread_x, data_index, data_org);
                        printf("computed sum %f from current value from w %f * data %f = computed_value %f at interpolation index j,i=%d,%d, s=%d, f=%d, and block y,x=%d,%d, batched n=%d\n",
                               sum_val, w, data_org, computeed_val, interpolation_j,interpolation_i, s_index, f_index + f, block_y, block_x, n);

                        //printf("s_index=%d, with weight=%f\n ", s_index, w);
                    }*/
                    /*if ((thread_x == 31 ) && thread_y == 0 && f_index == 0 && f == 0 && block_x == 0 && block_y == 0 )
                    {
                        printf("s_index=%d, with weight=%f and px=%d,%d and interpol=%d,%d with data val=%f\n ", s_index, w, px_y, px_x, interpolation_j, interpolation_i, data_org);
                    }*/
                }
            }
        }
    }

};



template <typename BlockIndexingT, int _LOAD_DATA_INDEX, int _LOAD_OFFSET_INDEX>
__global__ void //__launch_bounds__(128, 3)
DAUConv_forward_pipeline_kernel(const float *filtered_images,
                                const int *filter_offsets, const float *filter_weights,
                                const float *filter_offsets_and_weights,
                                float *output,
                                const int N, const int S, const int F, const int G,
                                const int img_width_, const int img_height_,
                                const int new_img_parts_width, const int new_img_parts_height) {

// INPUT: filtered images  	[I x S x H x W]
//		  filter offsets   	[F x S x G]
//		  filter weights   	[F x S x G]
// OUTPUT output  		 	[I x F x H x W]

#ifndef CUBIN_EMBEDDING

    typedef class BlockIndexingT::Kernel BlockIndexingKernel;

    static const int NUM_SM = BlockIndexingT::NUM_SM;
    static const int Bx = BlockIndexingT::Bx;
    static const int By = BlockIndexingT::By;
    static const int BLOCK_FEATURES = BlockIndexingT::BLOCK_FEATURES;
    static const int BLOCK_IMAGES = BlockIndexingT::BLOCK_IMAGES;
    static const int BATCH_PIXELS_SIZE_X = BlockIndexingT::BATCH_PIXELS_SIZE_X;
    static const int BATCH_PIXELS_SIZE_Y = BlockIndexingT::BATCH_PIXELS_SIZE_Y;
    static const int PIXELS_INTERPOLATION_Dx = BlockIndexingT::PIXELS_INTERPOLATION_Dx;
    static const int PIXELS_INTERPOLATION_Dy = BlockIndexingT::PIXELS_INTERPOLATION_Dy;
    static const int BATCH_FEATURES_SIZE = BlockIndexingT::BATCH_FEATURES_SIZE;
    static const int BATCH_COMPUTE_FEATURES_SIZE = BlockIndexingT::BATCH_COMPUTE_FEATURES_SIZE;
    static const int BATCH_COMPUTE_SUBFEATURES_SIZE = BlockIndexingT::BATCH_COMPUTE_SUBFEATURES_SIZE;
    static const int BATCH_MEM_SUBFEATURES_SIZE = BlockIndexingT::BATCH_MEM_SUBFEATURES_SIZE;
    static const int BATCH_GAUSS_SIZE = BlockIndexingT::BATCH_GAUSS_SIZE;
    static const int BATCH_IMAGES = BlockIndexingT::BATCH_IMAGES;
    static const int IMG_WIDTH = BlockIndexingT::IMG_WIDTH;
    static const int IMG_HEIGHT = BlockIndexingT::IMG_HEIGHT;
    static const int MAX_OFFSET = BlockIndexingT::MAX_OFFSET;

    static const int NUM_THREADS = BlockIndexingT::NUM_THREADS;

    static const bool USE_SEPARATE_WEIGHTS_AND_OFFSETS = BlockIndexingT::USE_SEPARATE_WEIGHTS_AND_OFFSETS;
    static const int LOAD_DATA_INDEX = _LOAD_DATA_INDEX >= 0 ? _LOAD_DATA_INDEX : BlockIndexingT::LOAD_DATA_INDEX;
    static const int LOAD_OFFSET_INDEX = _LOAD_OFFSET_INDEX >= 0 ? _LOAD_OFFSET_INDEX : BlockIndexingT::LOAD_OFFSET_INDEX;


    static const int BATCH_PIXELS_FLOAT4 = 1;

    static const bool BATCH_PIXELS_BY_WIDTH = false; //BATCH_PIXELS_SIZE_X % BATCH_PIXELS_FLOAT4 == 0;


    // since we can load 4 weights and offsets from single LDS.128 we can batch 4 computes of features
    //static const int BATCH_COMPUTE_FEATURES_SIZE = 2;

    static const int NUM_READ_FEATURES =  BATCH_COMPUTE_FEATURES_SIZE >= 4 ? 4 :
                                          (BATCH_COMPUTE_FEATURES_SIZE >= 2 ? 2 : 1);

    // using float4 to load so use
    static const int BATCH_SH_PIXELS_SIZE = 4;

    static const int DOUBLE_BUFFERING = 2;

    static const int NUM_REPLICATE_OFFSETED = 0;

    static const int PIXELS_INTERPOLATION_SIZE = PIXELS_INTERPOLATION_Dx * PIXELS_INTERPOLATION_Dy;

    float* output_batch = reinterpret_cast<float*>(output);

    int img_width = IMG_WIDTH; // img_width_
    int img_height = IMG_HEIGHT; // img_height_

    BlockIndexingKernel block_indexing(img_width, img_height, G);

    int I = N / (BATCH_IMAGES * BLOCK_IMAGES) * new_img_parts_width * new_img_parts_height;

    int n = block_indexing.getImageBlockIdx();

    int g_offset = block_indexing.getGaussianBlockIdx();

    int f_offset = block_indexing.getFeatureIdx();

    int f_block_idx = block_indexing.getFeatureBlockIdx();

    int block_width = block_indexing.getPosBlockSize().x;
    int block_height = block_indexing.getPosBlockSize().y;

    int block_x = block_indexing.getPosBlockIdx().x;
    int block_y = block_indexing.getPosBlockIdx().y;

    int thread_x = block_indexing.getPosThreadIdx().x;
    int thread_y = block_indexing.getPosThreadIdx().y;

    // if this block handles all pixels that are out of the actual image then there is no need to continue processing

    {
        // each image is split into patches of fixed sizes so we need to recombine them to original size
        int patch_idx = n / (N / (BATCH_IMAGES * BLOCK_IMAGES));

        int patch_j = patch_idx / new_img_parts_width;
        int patch_i = patch_idx % new_img_parts_width;

        // get actual output pixel location, but for first block only
        int out_px_y = patch_j * IMG_HEIGHT + (block_y);
        int out_px_x = patch_i * IMG_WIDTH + (block_x);

        bool within_valid_pixels = 0 <= out_px_x && out_px_x < img_width_ && out_px_y + 0 < img_height_;

        if (within_valid_pixels == false)
            return;
    }
    int G_MEM_SIZE = G / BATCH_GAUSS_SIZE;
    int S_MEM_SIZE = S / (BATCH_COMPUTE_SUBFEATURES_SIZE * BATCH_MEM_SUBFEATURES_SIZE);
    int F_MEM_SIZE = F / (BATCH_FEATURES_SIZE * BLOCK_FEATURES);

    static const int OFFSET_BLOCK_MEM_SIZE = MAX(4,BATCH_COMPUTE_SUBFEATURES_SIZE * BATCH_MEM_SUBFEATURES_SIZE * BATCH_GAUSS_SIZE * BATCH_FEATURES_SIZE *  BLOCK_FEATURES);
    static const int WEIGHT_BLOCK_MEM_SIZE = BATCH_COMPUTE_SUBFEATURES_SIZE * BATCH_MEM_SUBFEATURES_SIZE * BATCH_GAUSS_SIZE * PIXELS_INTERPOLATION_SIZE * BATCH_FEATURES_SIZE * BLOCK_FEATURES;

    typedef BlockSharedMemory<NUM_THREADS,
            Bx * BATCH_PIXELS_SIZE_X * BATCH_IMAGES * BLOCK_IMAGES,
            By * BATCH_PIXELS_SIZE_Y,
            MAX_OFFSET * BATCH_IMAGES * BLOCK_IMAGES,
            MAX_OFFSET,
            (NUM_REPLICATE_OFFSETED+1) * DOUBLE_BUFFERING * BATCH_MEM_SUBFEATURES_SIZE,
            float4,
            BATCH_SH_PIXELS_SIZE> SharedMem;

    __shared__ typename SharedMem::Data data;

    int interleaved_thread_x = block_indexing.getInterleavedPosThreadIdx().x;
    int interleaved_thread_y = block_indexing.getInterleavedPosThreadIdx().y;

    SharedMem image_sh_class(data, make_int2(interleaved_thread_x, interleaved_thread_y));

    int thread_sh_x = image_sh_class.getThreadIdx().x;
    int thread_sh_y = image_sh_class.getThreadIdx().y;

    typedef BlockSharedMemory<NUM_THREADS, WEIGHT_BLOCK_MEM_SIZE, 1, 0,0, DOUBLE_BUFFERING, float4, BATCH_SH_PIXELS_SIZE> SharedMemWeights;
    typedef BlockSharedMemory<NUM_THREADS, OFFSET_BLOCK_MEM_SIZE, 1, 0,0, DOUBLE_BUFFERING, int4, BATCH_SH_PIXELS_SIZE> SharedMemOffsets;

    typedef BlockSharedMemory<NUM_THREADS, OFFSET_BLOCK_MEM_SIZE+WEIGHT_BLOCK_MEM_SIZE, 1, 0,0, DOUBLE_BUFFERING, float4, BATCH_SH_PIXELS_SIZE>
            SharedMemOffsetsAndWeights;


    // for when using individual
    __shared__ typename SharedMemWeights::Data data_weights;
    __shared__ typename SharedMemOffsets::Data data_offsets;

    SharedMemWeights weights_sh_class(data_weights, make_int2(thread_x, thread_y));
    SharedMemOffsets offsets_sh_class(data_offsets, make_int2(thread_x, thread_y));

    // for when using combined
    __shared__ typename SharedMemOffsetsAndWeights::Data data_offsets_and_weights;

    SharedMemOffsetsAndWeights offsets_and_weights_sh_class(data_offsets_and_weights, make_int2(thread_x, thread_y));

    float* weights_batch_sh;
    int* offset_batch_sh;

    if (USE_SEPARATE_WEIGHTS_AND_OFFSETS) {
        weights_batch_sh = (float*)weights_sh_class.getData(0);
        offset_batch_sh = (int*)offsets_sh_class.getData(0);
    } else {

        offset_batch_sh = (int *) offsets_and_weights_sh_class.getData(0);
        weights_batch_sh = ((float *) offsets_and_weights_sh_class.getData(0)) + OFFSET_BLOCK_MEM_SIZE;
    }

    // [2] ==> this is to calculate output for neigbooring pixels that use the same pixel but with different weight when using interpolation
    float4 out_val[BATCH_FEATURES_SIZE][BATCH_IMAGES][2][(BATCH_PIXELS_SIZE_X*BATCH_PIXELS_SIZE_Y)/BATCH_PIXELS_FLOAT4];

    for (int f = 0; f < BATCH_FEATURES_SIZE; ++f) {
        for (int i = 0; i < BATCH_IMAGES; ++i) {
            for (int j = 0; j < 2; ++j) {
                for (int px = 0; px < (BATCH_PIXELS_SIZE_X * BATCH_PIXELS_SIZE_Y) / BATCH_PIXELS_FLOAT4; ++px) {

                    out_val[f][i][j][px].x = 0;
                    out_val[f][i][j][px].y = 0;
                    out_val[f][i][j][px].z = 0;
                    out_val[f][i][j][px].w = 0;
                }
            }
        }
    }

    PipelineEngine<BATCH_PIXELS_SIZE_X,
    BATCH_PIXELS_SIZE_Y,
    BATCH_PIXELS_BY_WIDTH,
    PIXELS_INTERPOLATION_Dx,
    PIXELS_INTERPOLATION_Dy,
    BATCH_FEATURES_SIZE,
    BATCH_COMPUTE_FEATURES_SIZE,
    BATCH_COMPUTE_SUBFEATURES_SIZE,
    BATCH_MEM_SUBFEATURES_SIZE,
    BLOCK_FEATURES,
    BATCH_IMAGES,
    BLOCK_IMAGES,
    //IMG_WIDTH, IMG_HEIGHT,
    BATCH_PIXELS_FLOAT4,
    SharedMem> pipeline(image_sh_class);

    pipeline.warp_id = BlockIndexingT::Kernel::warp_lane_id();

    // those are for debugging purpuse only
    pipeline.block_x = block_x;
    pipeline.block_y = block_y;
    pipeline.thread_x = thread_x;
    pipeline.thread_y = thread_y;
    pipeline.image_index = n;

    const int f_start_block = f_offset - f_block_idx;

    const float* _image_global_current = filtered_images + OFFSET(n,
                                                                  0,
                                                                  (MAX_OFFSET + block_y) + image_sh_class.getThreadIdx().y,
                                                                  (MAX_OFFSET + block_x)* BATCH_IMAGES * BLOCK_IMAGES + image_sh_class.getThreadIdx().x,
                                                                  I, S, (img_height + 2*MAX_OFFSET), (img_width + 2*MAX_OFFSET)* BATCH_IMAGES * BLOCK_IMAGES);


    const int* _filter_offset_current = filter_offsets +  OFFSET(f_start_block / (BLOCK_FEATURES*BATCH_FEATURES_SIZE),
                                                                 0,
                                                                 g_offset / (BATCH_GAUSS_SIZE),
                                                                 0,
                                                                 F_MEM_SIZE, S_MEM_SIZE, G_MEM_SIZE, OFFSET_BLOCK_MEM_SIZE);

    const float* _filter_weights_current = filter_weights + OFFSET(f_start_block/ (BLOCK_FEATURES*BATCH_FEATURES_SIZE) ,
                                                                   0,
                                                                   g_offset / (BATCH_GAUSS_SIZE),
                                                                   0,
                                                                   F_MEM_SIZE, S_MEM_SIZE, G_MEM_SIZE, WEIGHT_BLOCK_MEM_SIZE);

    const int* _filter_offset_next = _filter_offset_current + offsets_sh_class.getThreadIdx().x;
    const float* _filter_weights_next = _filter_weights_current + weights_sh_class.getThreadIdx().x;

    if (USE_SEPARATE_WEIGHTS_AND_OFFSETS)
        if (1){


            // load offsets and weights for the first one
            offsets_sh_class.template load_global<OFFSET_BLOCK_MEM_SIZE,0,false,0>(reinterpret_cast<const int4*>(_filter_offset_current + offsets_sh_class.getThreadIdx().x),
                    //offsets_sh_class.template load_global<OFFSET_BLOCK_MEM_SIZE,0,true,0>(reinterpret_cast<const int4*>(_filter_offset_current + offsets_sh_class.getThreadIdx().x),
                                                                                   reinterpret_cast<int4*>(offsets_sh_class.getDataThreadIndexingWrite(0)));

            weights_sh_class.template load_global<WEIGHT_BLOCK_MEM_SIZE,0,false,1>(reinterpret_cast<const float4*>(_filter_weights_current + weights_sh_class.getThreadIdx().x),
                                                                                   reinterpret_cast<float4*>(weights_sh_class.getDataThreadIndexingWrite(0)));

        }

    const float* _filter_offset_and_weights_current = filter_offsets_and_weights +  OFFSET(f_start_block / (BLOCK_FEATURES*BATCH_FEATURES_SIZE),
                                                                                           0,
                                                                                           g_offset / (BATCH_GAUSS_SIZE),
                                                                                           0,
                                                                                           F_MEM_SIZE, S_MEM_SIZE, G_MEM_SIZE, OFFSET_BLOCK_MEM_SIZE+WEIGHT_BLOCK_MEM_SIZE);



    const float* _filter_offset_and_weights_next = _filter_offset_and_weights_current + offsets_and_weights_sh_class.getThreadIdx().x;


    if (USE_SEPARATE_WEIGHTS_AND_OFFSETS == false)
        if (1){

            offsets_and_weights_sh_class.template load_global<OFFSET_BLOCK_MEM_SIZE+WEIGHT_BLOCK_MEM_SIZE,0,false,0>(
            //!!#!! offsets_and_weights_sh_class.template load_global<OFFSET_BLOCK_MEM_SIZE+WEIGHT_BLOCK_MEM_SIZE,0,true,0>(
                    reinterpret_cast<const float4*>(_filter_offset_and_weights_current + offsets_and_weights_sh_class.getThreadIdx().x),
                    reinterpret_cast<float4*>(offsets_and_weights_sh_class.getDataThreadIndexingWrite(0)));

        }

    if (1){
        // load first batch of subfeatures/input data into shared memory

        for (int s = 0 ; s < BATCH_MEM_SUBFEATURES_SIZE; ++s) {
            // TODO: getDataThreadIndexingWrite(s % BATCH_MEM_SUBFEATURES_SIZE) is not correct !!!
            // since NUM_REPEAT buffers for image_sh_class are DOUBLE_BUFFERING x BATCH_MEM_SUBFEATURES_SIZE x NUM_REPLICATE_OFFSETED+1
            // we should use instead the following buffer_index (NOTE: This was fixed when doing fast_gauss_backward, but not tested yet !!!!)
            int buffer_index = OFFSET(0, 0, s, 0, 1, DOUBLE_BUFFERING, BATCH_MEM_SUBFEATURES_SIZE, NUM_REPLICATE_OFFSETED+1);

            //image_sh_class.template load_global<(IMG_WIDTH + 2 * MAX_OFFSET) * BATCH_IMAGES * BLOCK_IMAGES,NUM_REPLICATE_OFFSETED,true,1>(reinterpret_cast<const float4*>(_image_global_current + (s) * ((img_width + 2*MAX_OFFSET) * (img_height + 2*MAX_OFFSET) * BATCH_IMAGES * BLOCK_IMAGES)),
            image_sh_class.template load_global<(IMG_WIDTH + 2 * MAX_OFFSET) * BATCH_IMAGES * BLOCK_IMAGES,NUM_REPLICATE_OFFSETED,false,1>(reinterpret_cast<const float4*>(_image_global_current + (s) * ((img_width + 2*MAX_OFFSET) * (img_height + 2*MAX_OFFSET) * BATCH_IMAGES * BLOCK_IMAGES)),
                                                                                                                                           reinterpret_cast<float4*>(image_sh_class.getDataThreadIndexingWrite(buffer_index)),
                                                                                                                                           (img_width + 2 * MAX_OFFSET) * BATCH_IMAGES * BLOCK_IMAGES);
        }
        //image_sh_class.print();
    }

    __syncthreads();

    const int MAX_S_OUTER_INDEX = S /  BATCH_COMPUTE_SUBFEATURES_SIZE * BATCH_MEM_SUBFEATURES_SIZE;

    for (int s_offset_outer = 0; s_offset_outer < S; s_offset_outer+=BATCH_COMPUTE_SUBFEATURES_SIZE * BATCH_MEM_SUBFEATURES_SIZE) {

        const int s_outer_index = s_offset_outer / (BATCH_COMPUTE_SUBFEATURES_SIZE * BATCH_MEM_SUBFEATURES_SIZE);

        const int s_buffer_index = s_outer_index % DOUBLE_BUFFERING;
        const int s_next_buffer_index = (s_outer_index +1) % DOUBLE_BUFFERING;

        // when using seperabe buffers for weights and offsets
        const int* filter_offset_current  = _filter_offset_current + OFFSET(0, s_outer_index, 0, 0,
                                                                            F_MEM_SIZE, S_MEM_SIZE, G_MEM_SIZE, OFFSET_BLOCK_MEM_SIZE);
        const float* filter_weights_current  = _filter_weights_current + OFFSET(0, s_outer_index, 0, 0,
                                                                                F_MEM_SIZE, S_MEM_SIZE, G_MEM_SIZE, WEIGHT_BLOCK_MEM_SIZE);

        const int* filter_offset_next  = _filter_offset_next + OFFSET(0, s_outer_index + 1 , 0, 0,
                                                                      F_MEM_SIZE, S_MEM_SIZE, G_MEM_SIZE, OFFSET_BLOCK_MEM_SIZE);
        const float* filter_weights_next = _filter_weights_next + OFFSET(0, s_outer_index + 1, 0, 0,
                                                                         F_MEM_SIZE, S_MEM_SIZE, G_MEM_SIZE, WEIGHT_BLOCK_MEM_SIZE);

        // when using combined buffers for weights and offsets
        const float* filter_offset_and_weights_current  = _filter_offset_and_weights_current + OFFSET5(0, s_outer_index, 0, 0, 0,
                                                                                                       F_MEM_SIZE, S_MEM_SIZE, G_MEM_SIZE, 1, WEIGHT_BLOCK_MEM_SIZE + WEIGHT_BLOCK_MEM_SIZE);
        const float* filter_offset_and_weights_next  = _filter_offset_and_weights_next + OFFSET5(0, s_outer_index + 1, 0, 0, 0,
                                                                                                 F_MEM_SIZE, S_MEM_SIZE, G_MEM_SIZE, 1, OFFSET_BLOCK_MEM_SIZE + WEIGHT_BLOCK_MEM_SIZE);


        const float* image_global_current = _image_global_current + OFFSET(0, s_offset_outer, 0, 0,
                                                                           I, S, (img_height + 2*MAX_OFFSET), (img_width + 2*MAX_OFFSET)* BATCH_IMAGES * BLOCK_IMAGES);
        const float* image_global_next = _image_global_current + OFFSET(0, s_offset_outer + BATCH_COMPUTE_SUBFEATURES_SIZE * BATCH_MEM_SUBFEATURES_SIZE, 0, 0,
                                                                        I, S, (img_height + 2*MAX_OFFSET), (img_width + 2*MAX_OFFSET)* BATCH_IMAGES * BLOCK_IMAGES);

        ptr4 off_A[BATCH_GAUSS_SIZE][BATCH_COMPUTE_FEATURES_SIZE/NUM_READ_FEATURES],
                off_B[BATCH_GAUSS_SIZE][BATCH_COMPUTE_FEATURES_SIZE/NUM_READ_FEATURES];

        float4 w_A[BATCH_GAUSS_SIZE][PIXELS_INTERPOLATION_SIZE][BATCH_COMPUTE_FEATURES_SIZE/NUM_READ_FEATURES],
                w_B[BATCH_GAUSS_SIZE][PIXELS_INTERPOLATION_SIZE][BATCH_COMPUTE_FEATURES_SIZE/NUM_READ_FEATURES];

        float4 d_A[BATCH_GAUSS_SIZE][BATCH_FEATURES_SIZE][BATCH_IMAGES][PIXELS_INTERPOLATION_SIZE][(BATCH_PIXELS_SIZE_X * BATCH_PIXELS_SIZE_Y)/BATCH_PIXELS_FLOAT4],
                d_B[BATCH_GAUSS_SIZE][BATCH_FEATURES_SIZE][BATCH_IMAGES][PIXELS_INTERPOLATION_SIZE][(BATCH_PIXELS_SIZE_X * BATCH_PIXELS_SIZE_Y)/BATCH_PIXELS_FLOAT4];


        typename SharedMem::LoadingData ld_data[BATCH_MEM_SUBFEATURES_SIZE];

        typename SharedMemWeights::LoadingData ld_weights;
        typename SharedMemOffsets::LoadingData ld_offsets;
        typename SharedMemOffsetsAndWeights::LoadingData ld_offsets_and_weights;

        struct IterIndex {
            int s; // sub-feature index
            int f; // feature index
            int g; // gauss component index
        };


        // global loading is done imediately (no delay)
        // to simplyfiy the code for global loading we can force global loading to be done BATCH_MEM_SUBFEATURES_SIZE * BATCH_FEATURES_SIZE/BATCH_COMPUTE_FEATURES_SIZE loops before
        // other units start
        static const int start_delay_global_load = LOAD_DATA_INDEX; //5;

        static const int start_delay_offset_load = 0;// + BATCH_MEM_SUBFEATURES_SIZE * BATCH_FEATURES_SIZE/BATCH_COMPUTE_FEATURES_SIZE;
        static const int start_delay_w_load = 1;// + BATCH_MEM_SUBFEATURES_SIZE * BATCH_FEATURES_SIZE/BATCH_COMPUTE_FEATURES_SIZE;
        static const int start_delay_data_load = 1;// + BATCH_MEM_SUBFEATURES_SIZE * BATCH_FEATURES_SIZE/BATCH_COMPUTE_FEATURES_SIZE;
        static const int start_delay_compute = 2;// + BATCH_MEM_SUBFEATURES_SIZE * BATCH_FEATURES_SIZE/BATCH_COMPUTE_FEATURES_SIZE;

        // NOTE: EXTRA_LOOPS is max value out of start_delay_global_load, start_delay_offset_load, start_delay_w_load, start_delay_data_load and start_delay_compute
        static const int EXTRA_LOOPS = MAX(start_delay_global_load,
                                           MAX(start_delay_offset_load,
                                               MAX(start_delay_w_load,
                                                   MAX(start_delay_data_load, start_delay_compute))));

        int NUM_ITER = BATCH_COMPUTE_SUBFEATURES_SIZE * BATCH_MEM_SUBFEATURES_SIZE * BATCH_GAUSS_SIZE * BATCH_FEATURES_SIZE/BATCH_COMPUTE_FEATURES_SIZE;


        // iterations go over subsets of [S x F ] i.e. [BATCH_MEM_SUBFEATURES_SIZE] * [BATCH_FEATURES_SIZE/BATCH_COMPUTE_FEATURES_SIZE]

        NDIndexing<BATCH_COMPUTE_SUBFEATURES_SIZE * BATCH_MEM_SUBFEATURES_SIZE,
                NDIndexing<BATCH_GAUSS_SIZE,
                NDIndexingZero<BATCH_FEATURES_SIZE/BATCH_COMPUTE_FEATURES_SIZE> > > indexing;
        // do all in one loop
#pragma unroll
        for (int index = 0 ; index < NUM_ITER + EXTRA_LOOPS; ++index)  {

            IterIndex load_global;
            IterIndex load_offset_index;
            IterIndex load_w_index;
            IterIndex load_data_index;
            IterIndex compute_index;

            // get flags to run each unit based on index number and its delay
            pipeline.load_offset.enabled = pipeline.should_run(index, start_delay_offset_load, NUM_ITER);
            pipeline.load_weights.enabled = pipeline.should_run(index, start_delay_w_load, NUM_ITER);
            pipeline.load_data.enabled = pipeline.should_run(index, start_delay_data_load, NUM_ITER);
            pipeline.compute.enabled = pipeline.should_run(index, start_delay_compute, NUM_ITER);

            bool load_global_enabled = pipeline.should_run(index, start_delay_global_load, NUM_ITER);

            typename SharedMem::ELEMENT_TYPE const* global_load_reading_ptr;
            typename SharedMem::ELEMENT_TYPE* global_load_writing_ptr;

            float const* image_global_load;

            int global_d = -1;
            int shared_d_off = -1;
            int shared_d_current = -1;
            int shared_d_next = -1;
            int global_start_s = -1;
            {
                // global loading is done immedately
                load_global.s = indexing.getIndex<0>(index - start_delay_global_load);
                load_global.g = indexing.getIndex<1>(index - start_delay_global_load);
                load_global.f = indexing.getIndex<2>(index - start_delay_global_load) * BATCH_COMPUTE_FEATURES_SIZE;

                global_start_s = load_global.s;
                // we actually load next batch of subfeatures so add BATCH_MEM_SUBFEATURES_SIZE
                load_global.s = load_global.s + BATCH_MEM_SUBFEATURES_SIZE;

                if (load_global_enabled)
                    load_global_enabled = load_global.f == 0 && load_global.g == 0;

                // TODO: do not load if this is last s_offset_outer index

                int double_buffer_index = (s_buffer_index + load_global.s/BATCH_MEM_SUBFEATURES_SIZE) % 2;
                int subfeat_buffer_index = load_global.s % BATCH_MEM_SUBFEATURES_SIZE;

                // if this is last iteration before moving to next s_offset_outer index then load for the next one
                bool load_next_s_outer = load_global.s < BATCH_MEM_SUBFEATURES_SIZE * BATCH_COMPUTE_SUBFEATURES_SIZE;

                image_global_load =  load_next_s_outer ? image_global_current : image_global_next;

                load_global.s = load_global.s % (BATCH_MEM_SUBFEATURES_SIZE * BATCH_COMPUTE_SUBFEATURES_SIZE);

                global_d = double_buffer_index;

                int buffer_index = OFFSET(0, double_buffer_index, subfeat_buffer_index, 0, 1, DOUBLE_BUFFERING, BATCH_MEM_SUBFEATURES_SIZE, NUM_REPLICATE_OFFSETED+1);

                global_load_reading_ptr = reinterpret_cast<const float4*>(image_global_load + (load_global.s) * (img_width + 2*MAX_OFFSET) * (img_height + 2*MAX_OFFSET) * BATCH_IMAGES * BLOCK_IMAGES);
                global_load_writing_ptr = reinterpret_cast<float4*>(image_sh_class.getDataThreadIndexingWrite(buffer_index));

            }
            bool require_sync = false;
            bool load_offset_reg_A;
            {
                // offset loading is done with no delay

                load_offset_index.s = indexing.getIndex<0>(index - start_delay_offset_load);
                load_offset_index.g = indexing.getIndex<1>(index - start_delay_offset_load);
                load_offset_index.f = indexing.getIndex<2>(index - start_delay_offset_load) * BATCH_COMPUTE_FEATURES_SIZE;

                int double_buffer_index =   (s_buffer_index + load_offset_index.s/BATCH_MEM_SUBFEATURES_SIZE) % DOUBLE_BUFFERING;
                int subfeat_buffer_index = load_offset_index.s % BATCH_MEM_SUBFEATURES_SIZE;

                int next_s = indexing.getIndex<0>((index - start_delay_offset_load) - 1);

                // enforce sync to ensure data is fully written to shared memory before we will be reading it
                // we do this each time before we switch to another buffer
                // note: we do this when load_offset starts which is the first operation so that sync will be nicly segemtnated between batches
                int s_mem_index = load_offset_index.s >= 0 ? (load_offset_index.s / BATCH_MEM_SUBFEATURES_SIZE) : ((load_offset_index.s + 1) /  BATCH_MEM_SUBFEATURES_SIZE + 1);
                int s_mem_index_next = next_s >= 0 ? (next_s / BATCH_MEM_SUBFEATURES_SIZE) : ((next_s + 1) /  BATCH_MEM_SUBFEATURES_SIZE + 1);

                int current_double_buffer_index = (s_buffer_index + s_mem_index) % DOUBLE_BUFFERING;
                int next_double_buffer_index = (s_buffer_index + s_mem_index_next) % DOUBLE_BUFFERING;

                if (pipeline.load_offset.enabled) {

                    if (load_offset_index.s >= 0 && load_offset_index.f >= 0 && load_offset_index.g >=0 ) {
                        require_sync = current_double_buffer_index == next_double_buffer_index ? false : true;

                        // handle first loading where prev index goes to negative and will not be handled by the previous line
                        if (load_offset_index.s == 0 && load_offset_index.f == 0 && load_offset_index.g == 0)
                            require_sync = true;
                    }
                }

                shared_d_next = next_double_buffer_index;

                // switch between registers every iteration
                bool use_reg_A = (index - start_delay_offset_load) % 2 == 0 ? true : false;

                load_offset_reg_A = use_reg_A;

                shared_d_off = double_buffer_index;

                int address_off = OFFSET5(load_offset_index.s, load_offset_index.g, load_offset_index.f/NUM_READ_FEATURES, f_block_idx/BATCH_FEATURES_SIZE, 0,
                                          BATCH_COMPUTE_SUBFEATURES_SIZE * BATCH_MEM_SUBFEATURES_SIZE, BATCH_GAUSS_SIZE, BATCH_FEATURES_SIZE/NUM_READ_FEATURES, BLOCK_FEATURES, NUM_READ_FEATURES);

                static const int OFFSET_BUFFER_SIZE =  USE_SEPARATE_WEIGHTS_AND_OFFSETS ? OFFSET_BLOCK_MEM_SIZE : (WEIGHT_BLOCK_MEM_SIZE + OFFSET_BLOCK_MEM_SIZE);

                // load offset
                pipeline.load_offset.offset_address = offset_batch_sh + address_off + (s_buffer_index) * (OFFSET_BUFFER_SIZE);
                //pipeline.load_offset.offset_address = &reinterpret_cast<int4*>((int*)filter_offset_current)[address_off];

                //pipeline.load_offset.offset_address = use_reg_A ?
                //										offset_batch_sh + address_off + (s_offset_outer % DOUBLE_BUFFERING) * OFFSET_BLOCK_MEM_SIZE/4
                //										: &reinterpret_cast<int4*>((int*)filter_offset_current)[address_off];
                //pipeline.load_offset.offset_address = &offset_batch_sh[load_offset_index.s][load_offset_index.g][0][load_offset_index.f/4][f_block_idx/BATCH_FEATURES_SIZE];

                int buffer_index = OFFSET(0, double_buffer_index, subfeat_buffer_index, 0, 1, DOUBLE_BUFFERING, BATCH_MEM_SUBFEATURES_SIZE, NUM_REPLICATE_OFFSETED+1);

                // TODO: getDataThreadIndexingRead() does not return correct address unless offseted replicate of data is used !!!
                pipeline.load_offset.base_address = image_sh_class.getDataThreadIndexingRead(buffer_index);
                pipeline.load_offset.output = (ptr4*)(use_reg_A ? &off_A[load_offset_index.g][0] : &off_B[load_offset_index.g][0]);

                /*if ((thread_x == 1 || thread_x == 0) && thread_y == 0 && load_offset_index.s == 0 && load_offset_index.f == 0 && load_offset_index.g == 0 && block_x == 0 && block_y == 0)
                {
                    printf("thread j,i: %d,%d, with load_offset.base_address=%p and thread indexing for reading j,i=%d,%d \n", thread_y, thread_x, pipeline.load_offset.base_address, image_sh_class.thread_indexing_reading.y, image_sh_class.thread_indexing_reading.x );
                }*/
            }
            bool load_w_reg_A;
            {
                // w and data loading is done with single delay

                load_w_index.s = indexing.getIndex<0>(index - start_delay_w_load);
                load_w_index.g = indexing.getIndex<1>(index - start_delay_w_load);
                load_w_index.f = indexing.getIndex<2>(index - start_delay_w_load) * BATCH_COMPUTE_FEATURES_SIZE;
                // switch between registers every iteration
                bool use_reg_A = (index - start_delay_w_load) % 2 == 0 ? true : false;

                load_w_reg_A = use_reg_A;

                int address_off = OFFSET6(load_w_index.s, load_w_index.g, 0, load_w_index.f/NUM_READ_FEATURES, f_block_idx/BATCH_FEATURES_SIZE, 0,
                                          BATCH_COMPUTE_SUBFEATURES_SIZE * BATCH_MEM_SUBFEATURES_SIZE, BATCH_GAUSS_SIZE, PIXELS_INTERPOLATION_SIZE, BATCH_FEATURES_SIZE/NUM_READ_FEATURES, BLOCK_FEATURES, NUM_READ_FEATURES);

                // load w

                //pipeline.load_weights.address = weights_batch_sh + address_off + (s_offset_outer % 2) * WEIGHT_BLOCK_MEM_SIZE/4;
                //pipeline.load_weights.address = &reinterpret_cast<float4*>((float*)filter_weights_current)[address_off];

                static const int WEIGHT_BUFFER_SIZE = USE_SEPARATE_WEIGHTS_AND_OFFSETS ? WEIGHT_BLOCK_MEM_SIZE : (WEIGHT_BLOCK_MEM_SIZE + OFFSET_BLOCK_MEM_SIZE);

                // we can utilize texture/L1 cache and load every second one from global data
                pipeline.load_weights.address = //(index - start_delay_w_load) % 4 == 1 ?
                        weights_batch_sh +  address_off + (s_buffer_index) * (WEIGHT_BUFFER_SIZE);
                //// TODO: offset_batch_sh and offset_address have been changed to int*	: &reinterpret_cast<float4*>((float*)filter_weights_current)[address_off];
                //filter_weights_current + address_off;

                //pipeline.load_weights.address = &weights_batch_sh[load_w_index.s][load_w_index.g][0][load_w_index.f/4][f_block_idx/BATCH_FEATURES_SIZE];

                pipeline.load_weights.output = (float4*)(use_reg_A ? w_A[load_w_index.g][0] : w_B[load_w_index.g][0]);

            }
            bool load_data_reg_A;
            {

                load_data_index.s = indexing.getIndex<0>(index - start_delay_data_load);
                load_data_index.g = indexing.getIndex<1>(index - start_delay_data_load);
                load_data_index.f = indexing.getIndex<2>(index - start_delay_data_load) * BATCH_COMPUTE_FEATURES_SIZE;

                // switch between registers every iteration
                bool use_reg_A = (index - start_delay_data_load) % 2 == 0 ? true : false;

                //shared_d_current = current_double_buffer_index;

                load_data_reg_A = use_reg_A;
                // load data

                pipeline.load_data.address = (ptr4*)(use_reg_A ? &off_A[load_data_index.g][0] : &off_B[load_data_index.g][0]);
                pipeline.load_data.output = (float4*)(use_reg_A ? d_A[load_data_index.g][load_data_index.f] : d_B[load_data_index.g][load_data_index.f]);

            }

            bool compute_reg_A;
            {
                // computation is done with double  delay

                compute_index.s = indexing.getIndex<0>(index - start_delay_compute);
                compute_index.g = indexing.getIndex<1>(index - start_delay_compute);
                compute_index.f = indexing.getIndex<2>(index - start_delay_compute) * BATCH_COMPUTE_FEATURES_SIZE;

                // switch between registers every iteration
                bool use_reg_A = (index - start_delay_compute) % 2 == 0 ? true : false;

                compute_reg_A = use_reg_A;
                // compute
                pipeline.compute.weights = (float4*)(use_reg_A ? w_A[compute_index.g][0] : w_B[compute_index.g][0]);
                pipeline.compute.data = (float4*)(use_reg_A ? d_A[compute_index.g][compute_index.f] : d_B[compute_index.g][compute_index.f]);
                pipeline.compute.output = out_val[compute_index.f][0][0];

            }


            // sync only before data buffer is switched
            if (require_sync) {
                // NOTE: sync is not needed if we have more then enough operations to cover the latency of sore operations
                // we can rughly say that if there is more then 128 operations then STS latency should be hidden (STS latency should not be more then 100 operations on different platforms)
                // however since store may be issued half way through operations then use 512 operations as limit
                //if (BATCH_PIXELS_SIZE_X * BATCH_PIXELS_SIZE_Y * BATCH_FEATURES_SIZE * BATCH_GAUSS_SIZE * PIXELS_INTERPOLATION_SIZE * BATCH_MEM_SUBFEATURES_SIZE * BATCH_COMPUTE_SUBFEATURES_SIZE < 512)
                {
                    __syncthreads();
                }
            }

            pipeline.load_global.enabled = load_global_enabled;
            pipeline.load_global.reading_ptr = global_load_reading_ptr;
            pipeline.load_global.writing_ptr = global_load_writing_ptr;
            pipeline.load_global.img_read_width = (img_width + 2 * MAX_OFFSET)* BATCH_IMAGES * BLOCK_IMAGES;


            /*if (thread_x == 0 && thread_y == 0 &&  block_x == 0 && block_y == 0 && f_offset == 0 && pipeline.image_index == 2) {
                if (require_sync)
                    printf("iter: %d, sycned\n", index);

                printf("pipeline s_offset: %d:, iter %d "
                               "gl %d (s:%d, org s:%d, g:%d, f:%d, buff:%d, read addr: %p (base: %p), write addr: %p), "
                               "off %d (s:%d, g:%d, f:%d, reg:%d, buff:%d, addr: %p, base data addr: %p), "
                               "w %d (s:%d, g:%d, f:%d, reg:%d), "
                               "data %d (s:%d, g:%d, f:%d, buff:%d, reg:%d), "
                               "compute %d (s:%d, g:%d, f:%d, reg:%d)\n",
                         s_offset_outer, index,
                        (int)load_global_enabled, load_global.s, global_start_s, load_global.g, load_global.f, global_d, pipeline.load_global.reading_ptr, image_global_load, pipeline.load_global.writing_ptr,
                        (int)pipeline.load_offset.enabled, load_offset_index.s, load_offset_index.g, load_offset_index.f, (int)load_offset_reg_A, shared_d_next, pipeline.load_offset.offset_address, pipeline.load_offset.base_address,
                        (int)pipeline.load_weights.enabled, load_w_index.s, load_w_index.g, load_w_index.f, (int)load_w_reg_A,
                        (int)pipeline.load_data.enabled, load_data_index.s, load_data_index.g, load_data_index.f, shared_d_current, (int)load_data_reg_A,
                        (int)pipeline.compute.enabled, compute_index.s, compute_index.g, compute_index.f, (int)compute_reg_A);


            }*/


            pipeline.s_index = s_offset_outer + compute_index.s;
            pipeline.f_index = f_offset + compute_index.f;
            pipeline.g_index = g_offset + compute_index.g;

            if (load_global_enabled) {
                if (0)
                    //image_sh_class.template load_global<(IMG_WIDTH + 2 * MAX_OFFSET)*BATCH_IMAGES * BLOCK_IMAGES,NUM_REPLICATE_OFFSETED,true,1>(pipeline.load_global.reading_ptr,
                    image_sh_class.template load_global<(IMG_WIDTH + 2 * MAX_OFFSET)*BATCH_IMAGES * BLOCK_IMAGES,NUM_REPLICATE_OFFSETED,false,1>(pipeline.load_global.reading_ptr,
                                                                                                                                                 pipeline.load_global.writing_ptr,
                                                                                                                                                 pipeline.load_global.img_read_width);
                else
                    image_sh_class.template load_global<(IMG_WIDTH + 2 * MAX_OFFSET)*BATCH_IMAGES * BLOCK_IMAGES,NUM_REPLICATE_OFFSETED,false,1>(pipeline.load_global.reading_ptr,
                    //image_sh_class.template load_global<(IMG_WIDTH + 2 * MAX_OFFSET)*BATCH_IMAGES * BLOCK_IMAGES,NUM_REPLICATE_OFFSETED,true,1>(pipeline.load_global.reading_ptr,
                                                                                                                                                 pipeline.load_global.writing_ptr,
                                                                                                                                                 pipeline.load_global.img_read_width,
                                                                                                                                                 &ld_data[load_global.s]);
            }


            pipeline.execute_step();
#if __CUDA_ARCH__ >= 200
            //if (pipeline.compute.enabled && compute_index.g != indexing.getIndex<1>(index-1 - start_delay_compute))
        //	asm("bar.arrive 15, 1536;"); // # of threads must be greater than 0
#endif

            if (1)
                if (load_global_enabled) {
                    image_sh_class.template store_shared<NUM_REPLICATE_OFFSETED>(ld_data[load_global.s], pipeline.load_global.writing_ptr);
                }

            if (USE_SEPARATE_WEIGHTS_AND_OFFSETS) {

                // if next iteration is not the last one then load offsets and weights for the next one - using double buffering so we do not intereput computation of the current one
                //if (s_outer_index + 1 < MAX_S_OUTER_INDEX && index == NUM_ITER + EXTRA_LOOPS - 4 )
                //if (s_outer_index + 1 < MAX_S_OUTER_INDEX && index == 0)
                if (s_outer_index + 1 < MAX_S_OUTER_INDEX && index == LOAD_OFFSET_INDEX)
                {
                    float* off_write_addr = offsets_sh_class.getDataThreadIndexingWrite(s_next_buffer_index);
                    float* w_write_addr = weights_sh_class.getDataThreadIndexingWrite(s_next_buffer_index);

                    if (1){

                        //offsets_sh_class.template load_global<OFFSET_BLOCK_MEM_SIZE,0,true,0>(reinterpret_cast<const int4*>(filter_offset_next),
                        offsets_sh_class.template load_global<OFFSET_BLOCK_MEM_SIZE,0,false,1>(reinterpret_cast<const int4*>(filter_offset_next),
                                                                                               reinterpret_cast<int4*>(off_write_addr));

                        //weights_sh_class.template load_global<WEIGHT_BLOCK_MEM_SIZE,0,true,1>(reinterpret_cast<const float4*>(filter_weights_next),
                        weights_sh_class.template load_global<WEIGHT_BLOCK_MEM_SIZE,0,false,1>(reinterpret_cast<const float4*>(filter_weights_next),
                                                                                               reinterpret_cast<float4*>(w_write_addr));
                    } else {
                        offsets_sh_class.template load_global<OFFSET_BLOCK_MEM_SIZE,0,false,1>(reinterpret_cast<const int4*>(filter_offset_next),
                                                                                               reinterpret_cast<int4*>(off_write_addr),
                                                                                               0, &ld_offsets);

                        weights_sh_class.template load_global<WEIGHT_BLOCK_MEM_SIZE,0,false,1>(reinterpret_cast<const float4*>(filter_weights_next),
                                                                                               reinterpret_cast<float4*>(w_write_addr),
                                                                                               0, &ld_weights);

                        //if (thread_x == 0 && thread_y == 0 &&  n == 0 && block_x == 0 && block_y == 0 && f_offset == 0) {
                        //	printf("loaded offset  (addr %p) and weights (addr %p) shared memory\n", off_write_addr, w_write_addr);
                        //}
                    }
                }

                if (0)
                    if (s_outer_index + 1 < MAX_S_OUTER_INDEX && index == NUM_ITER + EXTRA_LOOPS - 2) {
                        //if (s_outer_index + 1 < MAX_S_OUTER_INDEX && index == 3) {
#if __CUDA_ARCH__ >= 200
                        //asm("bar.arrive 15, 1536;"); // # of threads must be greater than 0
#endif
                        float* off_write_addr = offsets_sh_class.getDataThreadIndexingWrite(s_next_buffer_index);
                        float* w_write_addr = weights_sh_class.getDataThreadIndexingWrite(s_next_buffer_index);

                        offsets_sh_class.template store_shared<0>(ld_offsets,reinterpret_cast<int4*>(off_write_addr));
                        weights_sh_class.template store_shared<0>(ld_weights,reinterpret_cast<float4*>(w_write_addr));
                    }
            } else {

                // if next iteration is not the last one then load offsets and weights for the next one - using double buffering so we do not intereput computation of the current one
                //if (s_outer_index + 1 < MAX_S_OUTER_INDEX && index == NUM_ITER + EXTRA_LOOPS - 4 )
                //if (s_outer_index + 1 < MAX_S_OUTER_INDEX && index == 0)
                if (s_outer_index + 1 < MAX_S_OUTER_INDEX && index == LOAD_OFFSET_INDEX)
                {
                    float* off_and_w_write_addr = offsets_and_weights_sh_class.getDataThreadIndexingWrite(s_next_buffer_index);


                    if (1){

                        //!!#!! offsets_and_weights_sh_class.template load_global<OFFSET_BLOCK_MEM_SIZE+WEIGHT_BLOCK_MEM_SIZE,0,true,0>(
                        offsets_and_weights_sh_class.template load_global<OFFSET_BLOCK_MEM_SIZE+WEIGHT_BLOCK_MEM_SIZE,0,false,1>(
                                reinterpret_cast<const float4*>(filter_offset_and_weights_next),
                                reinterpret_cast<float4*>(off_and_w_write_addr));

                        /*if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0 && blockIdx.y == 0)
                            printf("loaded new offsets and weights from %p to %p where buffer index=%d:\n", filter_offset_and_weights_next, off_and_w_write_addr, s_next_buffer_index);
                        offsets_and_weights_sh_class.print();*/

                    } else {
                        offsets_and_weights_sh_class.template load_global<OFFSET_BLOCK_MEM_SIZE+WEIGHT_BLOCK_MEM_SIZE,0,false,1>(
                                reinterpret_cast<const float4*>(filter_offset_and_weights_next),
                                reinterpret_cast<float4*>(off_and_w_write_addr),
                                0, &ld_offsets_and_weights);

                        //if (thread_x == 0 && thread_y == 0 &&  n == 0 && block_x == 0 && block_y == 0 && f_offset == 0) {
                        //	printf("loaded offset  (addr %p) and weights (addr %p) shared memory\n", off_write_addr, w_write_addr);
                        //}
                    }
                }

                if (0)
                    //if (s_outer_index + 1 < MAX_S_OUTER_INDEX && index == NUM_ITER + EXTRA_LOOPS - 2) {
                    if (s_outer_index + 1 < MAX_S_OUTER_INDEX && index == 1) {
                        //if (s_outer_index + 1 < MAX_S_OUTER_INDEX && index == 3) {
#if __CUDA_ARCH__ >= 200
                        //asm("bar.arrive 15, 1536;"); // # of threads must be greater than 0
#endif
                        float* off_and_w_write_addr = offsets_and_weights_sh_class.getDataThreadIndexingWrite(s_next_buffer_index);

                        offsets_and_weights_sh_class.template store_shared<0>(ld_offsets_and_weights,reinterpret_cast<float4*>(off_and_w_write_addr));
                    }
                if (compute_index.g != indexing.getIndex<1>(index-1 - start_delay_compute))
                {
#if __CUDA_ARCH__ >= 200
                    //asm("bar.arrive 15, 1536;"); // # of threads must be greater than 0
#endif
                }

            }

        }

#if __CUDA_ARCH__ >= 200
        asm("bar.arrive 15, 1536;"); // # of threads must be greater than 0
#endif
    }

    // each image is split into patches of fixed sizes so we need to recombine them to original size
    int nn = (n % (N/(BATCH_IMAGES * BLOCK_IMAGES))) * BATCH_IMAGES * BLOCK_IMAGES + block_indexing.getImageIdx() * BATCH_IMAGES;
    int patch_idx = n / (N/(BATCH_IMAGES* BLOCK_IMAGES));

    int patch_j = patch_idx / new_img_parts_width;
    int patch_i = patch_idx % new_img_parts_width;


    // TODO: we can perform shuffle between output registers and ensure only coalesed output using only STG.128
#pragma unroll
    for (int f = 0; f < BATCH_FEATURES_SIZE; ++f) {
        for (int i = 0; i < BATCH_IMAGES; ++i ){
            for (int j = 0; j < 2; ++j ){
                if (BATCH_PIXELS_BY_WIDTH) {
                    // version for loading per 4 pixels by width and 1 pixel per height
                    /*#pragma unroll
                    for (int px_y = 0; px_y < BATCH_PIXELS_SIZE_Y; ++px_y) {
                        #pragma unroll
                        for (int px_x = 0; px_x < BATCH_PIXELS_SIZE_X; px_x+=BATCH_PIXELS_FLOAT4) {
                            //float4 tmp;
                            //tmp.x = 8; tmp.y = 8; tmp.z = 8; tmp.w = 8;
                            //reinterpret_cast<float4*>(output_batch)[OFFSET(n, f + f_offset, (block_y + thread_y + px_y), (block_x + thread_x + px_x), I, F, img_height, img_width + 2*MAX_OFFSET)/4] = tmp;
                            if (BATCH_PIXELS_FLOAT4 >= 4)
                                reinterpret_cast<float4*>(output_batch)[OFFSET(n, f + f_offset, (block_y + thread_y + px_y), (block_x + thread_x + px_x), I, F, img_height + 2*MAX_OFFSET, img_width + 2*MAX_OFFSET)/4] = out_val[f][(px_y * BATCH_PIXELS_SIZE_X + px_x)/4];
                            else if (BATCH_PIXELS_FLOAT4 >= 2) {
                                reinterpret_cast<float2*>(output_batch)[OFFSET(n, f + f_offset, (block_y + thread_y + px_y), (block_x + thread_x + px_x), I, F, img_height + 2*MAX_OFFSET, img_width + 2*MAX_OFFSET)/2].x = out_val[f][(px_y * BATCH_PIXELS_SIZE_X + px_x)/2].x;
                                reinterpret_cast<float2*>(output_batch)[OFFSET(n, f + f_offset, (block_y + thread_y + px_y), (block_x + thread_x + px_x), I, F, img_height + 2*MAX_OFFSET, img_width + 2*MAX_OFFSET)/2].y = out_val[f][(px_y * BATCH_PIXELS_SIZE_X + px_x)/2].y;
                            }
                            //else
                            //	reinterpret_cast<float*>(output_batch)[OFFSET(n, f + f_offset, (block_y + thread_y + px_y), (block_x + thread_x + px_x), I, F, img_height + 2*MAX_OFFSET, img_width + 2*MAX_OFFSET)/1] = out_val[f][(px_y * BATCH_PIXELS_SIZE_X + px_x)/1];
                        }
                    }*/
                } else {
                    // version for loading per 1 pixels by width and 4 pixel per height
#pragma unroll
                    for (int px_y = 0; px_y < BATCH_PIXELS_SIZE_Y; px_y+=BATCH_PIXELS_FLOAT4) {
#pragma unroll
                        for (int px_x = 0; px_x < BATCH_PIXELS_SIZE_X; ++px_x) {

                            int out_px_y = patch_j * IMG_HEIGHT + (block_y + thread_y + 0 + px_y);
                            int out_px_x = patch_i * IMG_WIDTH + (block_x + thread_x + px_x - j);

                            //int out_idx = OFFSET(nn+i, f + f_offset, out_px_y + 0, out_px_x, N, F, img_height_, img_width_);
                            //if (patch_i * IMG_WIDTH + (block_x + thread_x + px_x - j) == 31 && nn+i == 0 && f + f_offset == 0)
                            //if (out_idx == 31)
                            //	printf("adding val %f to  loc %d,%d with writing index =%d and existing val %f\n", out_val[f][i][j][px_y/BATCH_PIXELS_FLOAT4 * BATCH_PIXELS_SIZE_X + px_x].x, out_px_y, out_px_x, out_idx, output_batch[out_idx]);


                            if (BATCH_PIXELS_FLOAT4 > 0 && 0 <= out_px_x && out_px_x < img_width_ && out_px_y + 0 < img_height_) atomicAdd(&output_batch[OFFSET(nn+i, f + f_offset, out_px_y + 0, out_px_x, N, F, img_height_, img_width_)], out_val[f][i][j][px_y/BATCH_PIXELS_FLOAT4 * BATCH_PIXELS_SIZE_X + px_x].x);
                            if (BATCH_PIXELS_FLOAT4 > 1 && 0 <= out_px_x && out_px_x < img_width_ && out_px_y + 1 < img_height_) atomicAdd(&output_batch[OFFSET(nn+i, f + f_offset, out_px_y + 1, out_px_x, N, F, img_height_, img_width_)], out_val[f][i][j][px_y/BATCH_PIXELS_FLOAT4 * BATCH_PIXELS_SIZE_X + px_x].y);
                            if (BATCH_PIXELS_FLOAT4 > 2 && 0 <= out_px_x && out_px_x < img_width_ && out_px_y + 2 < img_height_) atomicAdd(&output_batch[OFFSET(nn+i, f + f_offset, out_px_y + 2, out_px_x, N, F, img_height_, img_width_)], out_val[f][i][j][px_y/BATCH_PIXELS_FLOAT4 * BATCH_PIXELS_SIZE_X + px_x].z);
                            if (BATCH_PIXELS_FLOAT4 > 3 && 0 <= out_px_x && out_px_x < img_width_ && out_px_y + 3 < img_height_) atomicAdd(&output_batch[OFFSET(nn+i, f + f_offset, out_px_y + 3, out_px_x, N, F, img_height_, img_width_)], out_val[f][i][j][px_y/BATCH_PIXELS_FLOAT4 * BATCH_PIXELS_SIZE_X + px_x].w);
                        }
                    }
                }
            }
        }
    }

#endif
}

template <int TILE_DIM_YX, int TILE_DIM_S, int TILE_DIM_IMAGE, int BATCH_PIXELS_X, int BATCH_IMAGES, int BLOCK_IMAGES, int NEW_WIDTH_, int NEW_HEIGHT_, int BORDER_SIZE_>
__global__ void
interleave_input_data_kernel(const float* input_data, float* output_data, const int N, const int S, const int img_width_in, const int img_height_in, const int img_width, const int img_height, const int num_img_patches_width, const int num_img_patches_height) {

// INPUT: input_data  	[ N x S x H x W]
// OUTPUT output  		[ N/BATCH_N x S x H x BATCH_PIXELS_X x (W / BATCH_PIXELS_X) x BATCH_N]

#ifndef CUBIN_EMBEDDING
    int in_border_y = img_height_in/2 - img_height/2;
    int in_border_x = img_width_in/2 - img_width/2;

    int NEW_WIDTH = NEW_WIDTH_;
    int NEW_HEIGHT = NEW_HEIGHT_;
    int BORDER_SIZE_X = BORDER_SIZE_;
    int BORDER_SIZE_Y = BORDER_SIZE_;

    // adjust new witdh/height and border size based on difference between input and output sizes
    // this will allow us to copy border pixels as well
    if (in_border_y > 0) {
        int max_height = NEW_HEIGHT + 2*BORDER_SIZE_Y;
        NEW_HEIGHT = min(max_height,NEW_HEIGHT + in_border_y*2);
        BORDER_SIZE_Y = max(0,BORDER_SIZE_Y - in_border_y);
    }
    if (in_border_x > 0) {
        int max_width = NEW_WIDTH + 2*BORDER_SIZE_X;
        NEW_WIDTH = min(max_width,NEW_WIDTH + in_border_x*2);
        BORDER_SIZE_X = max(0,BORDER_SIZE_X - in_border_x);
    }
    // TILE_DIM_X <= 256 (NUM threads)

    __shared__ float tile[TILE_DIM_IMAGE][TILE_DIM_YX+1];

    // threadIdx.x => over [1 .. WxH]
    // threadIdx.z => over [1 .. NxS]

    int thread_yx = threadIdx.x;

    int yx = blockIdx.x * TILE_DIM_YX + thread_yx;
    int s = blockIdx.y * TILE_DIM_S + threadIdx.y;
    int n = blockIdx.z * TILE_DIM_IMAGE + threadIdx.z;

    int x = yx % img_width_in;
    int y = yx / img_width_in;

    if (yx < img_height_in * img_width_in)
#pragma unroll
        for (int i = 0; i < TILE_DIM_IMAGE; ++i)
            tile[i][thread_yx] = input_data[OFFSET(n+i,s,y,x, N,S,img_height_in,img_width_in)];

    // transpose block offset
    int tid = threadIdx.x +
              TILE_DIM_YX * threadIdx.y +
              TILE_DIM_YX * TILE_DIM_S * threadIdx.z;

    int transposed_thread_yx = tid;

    int transposed_yx = blockIdx.x * TILE_DIM_YX + transposed_thread_yx;

    // convert to x,y location in transposed output image
    int transposed_y = transposed_yx / img_width_in;
    int transposed_x = transposed_yx % img_width_in;

    // then split img into patches of uniform size [NEW_HEIGHT, NEW_WIDTH]
    // get index of patch that this pixel will belong to
    int img_patch_x = transposed_x / NEW_WIDTH;
    int img_patch_y = transposed_y / NEW_HEIGHT;


    __syncthreads();

    if (transposed_yx < img_height_in * img_width_in)
        //int dy = 0;
#pragma unroll
        for (int dy = -1; dy <= 1; ++dy)
        {
            //int dx = 0;
#pragma unroll
            for (int dx = -1; dx <= 1; ++dx)
            {

                // transposed px location then needs to be converted into x,y within a patch
                int transposed_patch_x = transposed_x % NEW_WIDTH;
                int transposed_patch_y = transposed_y % NEW_HEIGHT;

                // add offset for neigbooring patch
                transposed_patch_x -= dx * NEW_WIDTH;
                transposed_patch_y -= dy * NEW_HEIGHT;

                // add border to output offset, i.e., border is added to width and height on both sides
                transposed_patch_x += BORDER_SIZE_X;
                transposed_patch_y += BORDER_SIZE_Y;

                int transposed_patch_x_outer = transposed_patch_x % BATCH_PIXELS_X;
                int transposed_patch_x_inner = transposed_patch_x / BATCH_PIXELS_X;

                int current_patch_y = img_patch_y + dy;
                int current_patch_x = img_patch_x + dx;

                // we write only if x,y values inside valid patch index
                // this notation works both for main patch as well as for neigbooring patches that need pixels as its border values
                if (0 <= transposed_patch_x && transposed_patch_x < NEW_WIDTH+2*BORDER_SIZE_X &&
                    0 <= transposed_patch_y && transposed_patch_y < NEW_HEIGHT+2*BORDER_SIZE_Y &&
                    0 <= current_patch_x && current_patch_x < num_img_patches_width	&&
                    0 <= current_patch_y && current_patch_y < num_img_patches_height)
                {

#pragma unroll
                    for (int i = 0; i < TILE_DIM_IMAGE; i+=(BATCH_IMAGES*BLOCK_IMAGES)) {
#pragma unroll
                        for (int i2 = 0; i2 < (BATCH_IMAGES*BLOCK_IMAGES); ++i2) {

                            int output_index = OFFSET8(current_patch_y, current_patch_x, (n+i)/(BATCH_IMAGES*BLOCK_IMAGES), s, transposed_patch_y,transposed_patch_x_outer, transposed_patch_x_inner, i2,
                                                       num_img_patches_height, num_img_patches_width, N/(BATCH_IMAGES*BLOCK_IMAGES), S, NEW_HEIGHT + 2*BORDER_SIZE_Y, BATCH_PIXELS_X, (NEW_WIDTH + 2*BORDER_SIZE_X)/ BATCH_PIXELS_X,  BATCH_IMAGES*BLOCK_IMAGES);

                            /*if (transposed_yx == 0)
                                printf("writing value %f to position %d at patch j,i=%d,%d,  transposed_patch y,x=%d,%d  (transposed y,x=(%d,%d)) img=%d,patch_y=%d,patch_x_outer=%d,k_outer=%d,patch_x_inner=%d,k_inner=%d\n",
                                       tile[i][transposed_thread_yx], output_index, current_patch_y, current_patch_x, transposed_patch_y, transposed_patch_x, transposed_y,transposed_x, n+i,transposed_patch_y,transposed_patch_x_outer, -1, transposed_patch_x_inner, -1);
*/
                            output_data[output_index] = tile[i+i2][transposed_thread_yx];
                        }
                    }
                }
            }
        }
#endif
}


template <typename BlockIndexingT>
class DAUConvFwdInputImage {
private:
    enum {
        // values from main block indexing sizes
        BATCH_PIXELS_X = BlockIndexingT::BATCH_PIXELS_SIZE_X,
        BATCH_IMAGES = BlockIndexingT::BATCH_IMAGES,
        BLOCK_IMAGES = BlockIndexingT::BLOCK_IMAGES,
        NEW_WIDTH = BlockIndexingT::IMG_WIDTH,
        NEW_HEIGHT = BlockIndexingT::IMG_HEIGHT,
        BORDER_SIZE = BlockIndexingT::MAX_OFFSET,

        // values specific for this kernel
        CUDA_THREADS = 256,

        // TILE_DIM_X * TILE_DIM_Y * TILE_DIM_IMAGE gets us to 512 of shared data per block (i.e. 2 kB)
        TILE_DIM_XY = CUDA_THREADS,
        TILE_DIM_S = 1,
        TILE_DIM_IMAGE_HUGE = 32,
        TILE_DIM_IMAGE_BIG = 16,
        TILE_DIM_IMAGE_SMALL = 4,
        TILE_DIM_IMAGE_XSMALL = 2
    };
    const int img_width_in, img_height_in;
    const int img_width;
    const int img_height;
    const int N;
    const int S;

    int TILE_DIM_IMAGE;

    int new_img_parts_width;
    int new_img_parts_height;

    dim3 threadsPerBlock;
    dim3 numBlocks;

public:
    DAUConvFwdInputImage(const int img_width_in, const int img_height_in, const int img_width, const int img_height, const int N, const int S, int new_img_parts_width, int new_img_parts_height) :
            img_width_in(img_width_in), img_height_in(img_height_in), img_width(img_width), img_height(img_height), N(N), S(S), new_img_parts_width(new_img_parts_width), new_img_parts_height(new_img_parts_height) {

        threadsPerBlock = dim3 (TILE_DIM_XY, 1, 1);

        if (N < BLOCK_IMAGES * BATCH_IMAGES) {
            throw DAUConvNet::DAUException(string_format("Invalid number of images: %d. Min allowed %d due to BLOCK_IMAGES=%d and BATCH_IMAGES=%d.\n", N, BLOCK_IMAGES * BATCH_IMAGES, BLOCK_IMAGES, BATCH_IMAGES));
        }
        if (N >= TILE_DIM_IMAGE_HUGE && BLOCK_IMAGES * BATCH_IMAGES <= TILE_DIM_IMAGE_HUGE)
            TILE_DIM_IMAGE = TILE_DIM_IMAGE_HUGE;
        else if (N >= TILE_DIM_IMAGE_BIG && BLOCK_IMAGES * BATCH_IMAGES <= TILE_DIM_IMAGE_BIG)
            TILE_DIM_IMAGE = TILE_DIM_IMAGE_BIG;
        else if (N >= TILE_DIM_IMAGE_SMALL && BLOCK_IMAGES * BATCH_IMAGES <= TILE_DIM_IMAGE_SMALL)
            TILE_DIM_IMAGE = TILE_DIM_IMAGE_SMALL;
        else if (N >= TILE_DIM_IMAGE_XSMALL && BLOCK_IMAGES * BATCH_IMAGES <= TILE_DIM_IMAGE_XSMALL)
            TILE_DIM_IMAGE = TILE_DIM_IMAGE_XSMALL;
        else if (BLOCK_IMAGES * BATCH_IMAGES <= 1)
            TILE_DIM_IMAGE = 1;
        else {
            throw DAUConvNet::DAUException(string_format("Invalid number of images %d due to incompatability with TILE_DIM_IMAGE=[32,16,4,2 or 1], and (BLOCK_IMAGES=%d,BATCH_IMAGES=%d).\n", N, BLOCK_IMAGES, BATCH_IMAGES));
        }
        numBlocks = dim3( ((int)ceil(img_width_in*img_height_in) + threadsPerBlock.x - 1) / threadsPerBlock.x,	// over image width and height
                          ((int)ceil(S/(float)TILE_DIM_S) + threadsPerBlock.z - 1) / threadsPerBlock.z, // over S
                          ((int)ceil(N/(float)TILE_DIM_IMAGE) + threadsPerBlock.z - 1) / threadsPerBlock.z);					// over N
    }
    size_t get_allocation_size() {
        return sizeof(float) * (NEW_WIDTH + 2*BORDER_SIZE) * (NEW_HEIGHT + 2*BORDER_SIZE) * S *  (N+1) * new_img_parts_width * new_img_parts_height;
    }
    float* create_input(float* interleaved_images_output, const float* filtered_images, cudaStream_t streamId = NULL) {

        if (TILE_DIM_IMAGE >= TILE_DIM_IMAGE_HUGE)
            interleave_input_data_kernel<TILE_DIM_XY,TILE_DIM_S,TILE_DIM_IMAGE_HUGE, BATCH_PIXELS_X, BATCH_IMAGES, BLOCK_IMAGES, NEW_WIDTH, NEW_HEIGHT, BORDER_SIZE><<<numBlocks,threadsPerBlock, 0, streamId>>>(filtered_images, interleaved_images_output, N,S, img_width_in, img_height_in, img_width, img_height, new_img_parts_width, new_img_parts_height);
        else if (TILE_DIM_IMAGE >= TILE_DIM_IMAGE_BIG)
            interleave_input_data_kernel<TILE_DIM_XY,TILE_DIM_S,TILE_DIM_IMAGE_BIG, BATCH_PIXELS_X, BATCH_IMAGES, BLOCK_IMAGES, NEW_WIDTH, NEW_HEIGHT, BORDER_SIZE><<<numBlocks,threadsPerBlock, 0, streamId>>>(filtered_images, interleaved_images_output, N,S, img_width_in, img_height_in, img_width, img_height, new_img_parts_width, new_img_parts_height);
        else if (TILE_DIM_IMAGE >= TILE_DIM_IMAGE_SMALL)
            interleave_input_data_kernel<TILE_DIM_XY,TILE_DIM_S,TILE_DIM_IMAGE_SMALL, BATCH_PIXELS_X, BATCH_IMAGES, BLOCK_IMAGES, NEW_WIDTH, NEW_HEIGHT, BORDER_SIZE><<<numBlocks,threadsPerBlock, 0, streamId>>>(filtered_images, interleaved_images_output, N,S, img_width_in, img_height_in, img_width, img_height, new_img_parts_width, new_img_parts_height);
        else if (TILE_DIM_IMAGE >= TILE_DIM_IMAGE_XSMALL)
            interleave_input_data_kernel<TILE_DIM_XY,TILE_DIM_S,TILE_DIM_IMAGE_XSMALL, BATCH_PIXELS_X, BATCH_IMAGES, BLOCK_IMAGES, NEW_WIDTH, NEW_HEIGHT, BORDER_SIZE><<<numBlocks,threadsPerBlock, 0, streamId>>>(filtered_images, interleaved_images_output, N,S, img_width_in, img_height_in, img_width, img_height, new_img_parts_width, new_img_parts_height);
        else
            interleave_input_data_kernel<TILE_DIM_XY,TILE_DIM_S,1, BATCH_PIXELS_X, BATCH_IMAGES, BLOCK_IMAGES, NEW_WIDTH, NEW_HEIGHT, BORDER_SIZE><<<numBlocks,threadsPerBlock, 0, streamId>>>(filtered_images, interleaved_images_output, N,S, img_width_in, img_height_in, img_width, img_height, new_img_parts_width, new_img_parts_height);

        if (0) {
            float* filtered_images_cpu = new float[(NEW_WIDTH + 2*BORDER_SIZE)*( NEW_HEIGHT + 2*BORDER_SIZE)* N*S * new_img_parts_width*new_img_parts_height];

            for (int i = 0; i < (NEW_WIDTH + 2*BORDER_SIZE)*( NEW_HEIGHT + 2*BORDER_SIZE)* N*S * new_img_parts_width*new_img_parts_height; ++i)
                filtered_images_cpu[i] = -1;

            cudaMemcpy(filtered_images_cpu, interleaved_images_output, sizeof(float)* (NEW_WIDTH + 2*BORDER_SIZE)*( NEW_HEIGHT + 2*BORDER_SIZE)* N*S*new_img_parts_width*new_img_parts_height, cudaMemcpyDeviceToHost);
            cudaDeviceSynchronize();

            int BATCH_PIXELS_X_ =BATCH_PIXELS_X;
            int BATCH_IMAGES_ = BATCH_IMAGES;
            int BLOCK_IMAGES_ = BLOCK_IMAGES;
            int NEW_WIDTH_ = NEW_WIDTH;
            int NEW_HEIGHT_ = NEW_HEIGHT;
            int BORDER_SIZE_ = BORDER_SIZE;

            //for (int i = 0; i < (img_width + 2*MAX_OFFSET)*( img_height + 2*MAX_OFFSET)* I*S; ++i) {
            int index =0;
            for (int i = 0; i < N/(BATCH_IMAGES *BLOCK_IMAGES) * new_img_parts_width*new_img_parts_height; ++i) {
                for (int s = 0; s < S; ++s) {
                    std::cout << "img: " << i <<" s: " << s << std::endl;
                    for (int l =0; l < NEW_HEIGHT + 2*BORDER_SIZE; ++l){
                        for (int n = 0; n < NEW_WIDTH + 2*BORDER_SIZE; ++n) {
                            for (int i2 = 0; i2 < BATCH_IMAGES * BLOCK_IMAGES; ++i2)
                                std::cout << filtered_images_cpu[index++] << " ";
                        }
                        std::cout << std::endl;
                    }
                    std::cout << std:: endl << "end of s" << std::endl;
                }
                std::cout << std::endl;

            }
            std::cout << std::endl;
            std::cout << std::endl;
            std::cout << std::endl;
            std::cout << std::endl;
            //return;
        }
        return interleaved_images_output;
    }
};


template <typename BlockIndexingT,
        typename ELEMENT_FLOAT_TYPE,
        typename ELEMENT_INT_TYPE>
__global__  void
perpare_weights_and_offsets(const float* filter_weights, const float* filter_offsets_x, const float* filter_offsets_y,
                            float *prepared_filter_weights, int *prepared_filter_offsets, float* prepared_filter_offsets_and_weights,
                            int S, int G, int F, int kernel_w, int kernel_h,
                            const DAUConvForward<float>::PARAM_FORMAT param_format, const bool offset_already_centered) {

    static const int NUM_SM = BlockIndexingT::NUM_SM;
    static const int Bx = BlockIndexingT::Bx;
    static const int By = BlockIndexingT::By;
    static const int BLOCK_FEATURES = BlockIndexingT::BLOCK_FEATURES;
    static const int BLOCK_IMAGES = BlockIndexingT::BLOCK_IMAGES;
    static const int BATCH_PIXELS_SIZE_X = BlockIndexingT::BATCH_PIXELS_SIZE_X;
    static const int BATCH_PIXELS_SIZE_Y = BlockIndexingT::BATCH_PIXELS_SIZE_Y;
    static const int PIXELS_INTERPOLATION_Dx = BlockIndexingT::PIXELS_INTERPOLATION_Dx;
    static const int PIXELS_INTERPOLATION_Dy = BlockIndexingT::PIXELS_INTERPOLATION_Dy;
    static const int BATCH_FEATURES_SIZE = BlockIndexingT::BATCH_FEATURES_SIZE;
    static const int BATCH_COMPUTE_FEATURES_SIZE = BlockIndexingT::BATCH_COMPUTE_FEATURES_SIZE;
    static const int BATCH_COMPUTE_SUBFEATURES_SIZE = BlockIndexingT::BATCH_COMPUTE_SUBFEATURES_SIZE;
    static const int BATCH_MEM_SUBFEATURES_SIZE = BlockIndexingT::BATCH_MEM_SUBFEATURES_SIZE;
    static const int BATCH_GAUSS_SIZE = BlockIndexingT::BATCH_GAUSS_SIZE;
    static const int BATCH_IMAGES = BlockIndexingT::BATCH_IMAGES;
    static const int IMG_WIDTH = BlockIndexingT::IMG_WIDTH;
    static const int IMG_HEIGHT = BlockIndexingT::IMG_HEIGHT;
    static const int MAX_OFFSET = BlockIndexingT::MAX_OFFSET;
    static const int NUM_THREADS = BlockIndexingT::NUM_THREADS;

    static const int PIXELS_INTERPOLATION_SIZE = PIXELS_INTERPOLATION_Dx * PIXELS_INTERPOLATION_Dy;

    static const int NUM_READ_FEATURES =  BATCH_COMPUTE_FEATURES_SIZE >= 4 ? 4 :
                                          (BATCH_COMPUTE_FEATURES_SIZE >= 2 ? 2 : 1);

    // inputs in quad vectors
    const ELEMENT_FLOAT_TYPE* filter_weights4 = reinterpret_cast<const ELEMENT_FLOAT_TYPE*>(filter_weights);
    const ELEMENT_FLOAT_TYPE* filter_offsets_x4 = reinterpret_cast<const ELEMENT_FLOAT_TYPE*>(filter_offsets_x);
    const ELEMENT_FLOAT_TYPE* filter_offsets_y4 = reinterpret_cast<const ELEMENT_FLOAT_TYPE*>(filter_offsets_y);

    // outputs in quad vectors
    ELEMENT_FLOAT_TYPE* prepared_filter_weights4 = reinterpret_cast<ELEMENT_FLOAT_TYPE*>(prepared_filter_weights);
    ELEMENT_INT_TYPE* prepared_filter_offsets4 = reinterpret_cast<ELEMENT_INT_TYPE*>(prepared_filter_offsets);


    int f_input_index = (blockIdx.x * blockDim.x  + threadIdx.x) * NUM_READ_FEATURES;
    int g_input_index = blockIdx.y * blockDim.y  + threadIdx.y;
    int s_input_index = blockIdx.z * blockDim.z  + threadIdx.z;

    // input data is of the form
    // float4 of size [S x G x F]

    int input_f_offset = -1;

    int input_index = -1;
    if (param_format == DAUConvForward<float>::SGF ) {
        input_index = OFFSET(0, s_input_index, g_input_index, f_input_index, 1, S, G, F);//(( s_input_index )*G + g_input_index ) * F + f_input_index ;
        input_f_offset = 1;
    } else if (param_format == DAUConvForward<float>::FGS) {
        input_index = OFFSET(0, f_input_index, g_input_index, s_input_index, 1, F, G, S);
        input_f_offset = G * S;
    }

    // output data is of the form:
    // float4 of size [F / (BATCH_FEATURES_SIZE * BLOCK_FEATURES)] x [S / (BATCH_COMPUTE_SUBFEATURES_SIZE*BATCH_MEM_SUBFEATURES_SIZE)] x [G / BATCH_GAUSS_SIZE]
    //				 	x [BATCH_COMPUTE_SUBFEATURES_SIZE * BATCH_MEM_SUBFEATURES_SIZE] x [BATCH_GAUSS_SIZE] x [PIXELS_INTERPOLATION_SIZE] x [BATCH_FEATURES_SIZE/4] x [BLOCK_FEATURES];

    static const int dim1_size = BLOCK_FEATURES;
    static const int dim2_size = BATCH_FEATURES_SIZE/NUM_READ_FEATURES;
    static const int dim3_size = PIXELS_INTERPOLATION_SIZE;
    static const int dim4_size = BATCH_GAUSS_SIZE;
    static const int dim5_size = BATCH_COMPUTE_SUBFEATURES_SIZE * BATCH_MEM_SUBFEATURES_SIZE;

    int dim6_size = G / dim4_size;
    int dim7_size = S / dim5_size;
    int dim8_size = F / (dim1_size * dim2_size * NUM_READ_FEATURES);


    int main_f_index = f_input_index / (dim1_size * dim2_size * NUM_READ_FEATURES);
    int f_block_tid = f_input_index % (dim1_size * dim2_size * NUM_READ_FEATURES);

    int main_s_index = s_input_index / dim5_size;
    int s_block_tid = s_input_index % dim5_size;

    int main_g_index = g_input_index / dim4_size;
    int g_block_tid = g_input_index % dim4_size;

    int s_mem_index = s_block_tid;
    int g_index = g_block_tid;

    // switch between block and batch indexes so that consecutive features and stored in [BATCH_FEATURES_SIZE/4]
    int f_batch_index = (f_block_tid / NUM_READ_FEATURES) % (dim2_size);
    int f_block_index = (f_block_tid / NUM_READ_FEATURES) / (dim2_size);

    //printf("f,s,g=(%d,%d,%d) with translate to f_block_tid=%d, f_batch_index=%d, f_block_index=%d\n", f_input_index, s_input_index, g_input_index, f_block_tid, f_batch_index, f_block_index);


    /*int output_index = OFFSET8(main_f_index,
                               main_s_index,
                               main_g_index,
                               s_mem_index,
                               g_index,
                               0,
                               f_batch_index,
                               f_block_index,
                               dim8_size, dim7_size, dim6_size, dim5_size, dim4_size, 1, dim2_size, dim1_size) * NUM_READ_FEATURES;*/

    int output_index = (OFFSET(main_f_index,
                               main_s_index,
                               main_g_index,
                               0,
                               dim8_size, dim7_size, dim6_size, max(4,dim5_size * dim4_size * dim2_size * dim1_size)) +
                        OFFSET(s_mem_index,
                                g_index,
                                f_batch_index,
                                f_block_index,
                                dim5_size, dim4_size , dim2_size,  dim1_size))* NUM_READ_FEATURES;

    /*printf("input index %d goes to output index %d: input s: %d, g: %d, f: %d: (out of S,G,F=%d,%d,%d) output dims: main_f_index=%d, main_s_index=%d, main_g_index=%d, s_mem_index=%d, g_index=%d, f_batch_index=%d, f_block_index=%d\n",
           input_index, output_index, s_input_index, g_input_index, f_input_index, S,G,F,
           main_f_index,
            main_s_index,
            main_g_index,
            s_mem_index,
             g_index,
            f_batch_index,
            f_block_index);*/


    // for offsets we need to combine X and Y coordinates and transform them directly to int values approproate for using specific BLOCK_ and BATCH_ sizes

    float4 offset_x;
    float4 offset_y;

    // read offset and convert them from [0..k_w] into [-k_w/2 ... k_w/2 ] i.e. convert to offsets with coord at center of kernel
    if (NUM_READ_FEATURES > 0) offset_x.x = reinterpret_cast<const float*>(filter_offsets_x4)[input_index + 0 * input_f_offset] - (offset_already_centered == false ? kernel_w/2 : 0);
    if (NUM_READ_FEATURES > 1) offset_x.y = reinterpret_cast<const float*>(filter_offsets_x4)[input_index + 1 * input_f_offset] - (offset_already_centered == false ? kernel_w/2 : 0);
    if (NUM_READ_FEATURES > 2) offset_x.z = reinterpret_cast<const float*>(filter_offsets_x4)[input_index + 2 * input_f_offset] - (offset_already_centered == false ? kernel_w/2 : 0);
    if (NUM_READ_FEATURES > 3) offset_x.w = reinterpret_cast<const float*>(filter_offsets_x4)[input_index + 3 * input_f_offset] - (offset_already_centered == false ? kernel_w/2 : 0);

    if (NUM_READ_FEATURES > 0) offset_y.x = reinterpret_cast<const float*>(filter_offsets_y4)[input_index + 0 * input_f_offset] - (offset_already_centered == false ? kernel_h/2 : 0);
    if (NUM_READ_FEATURES > 1) offset_y.y = reinterpret_cast<const float*>(filter_offsets_y4)[input_index + 1 * input_f_offset] - (offset_already_centered == false ? kernel_h/2 : 0);
    if (NUM_READ_FEATURES > 2) offset_y.z = reinterpret_cast<const float*>(filter_offsets_y4)[input_index + 2 * input_f_offset] - (offset_already_centered == false ? kernel_h/2 : 0);
    if (NUM_READ_FEATURES > 3) offset_y.w = reinterpret_cast<const float*>(filter_offsets_y4)[input_index + 3 * input_f_offset] - (offset_already_centered == false ? kernel_h/2 : 0);

    // offset is relative to shared memory organization which is defined by  BlockSharedMemory parameters:
    //		- SharedMem::ALLOC_WIDTH
    //		- SharedMem::ALLOC_HEIGHT

    // using float4 to load so use
    static const int BATCH_SH_PIXELS_SIZE = 4;

    static const int DOUBLE_BUFFERING = 2;

    typedef BlockSharedMemory<NUM_THREADS,
            Bx * BATCH_PIXELS_SIZE_X * BATCH_IMAGES * BLOCK_IMAGES,
            By * BATCH_PIXELS_SIZE_Y,
            MAX_OFFSET * BATCH_IMAGES * BLOCK_IMAGES,
            MAX_OFFSET,
            DOUBLE_BUFFERING * BATCH_MEM_SUBFEATURES_SIZE,
            float4,
            BATCH_SH_PIXELS_SIZE> SharedMem;

    int4 output_offset;

    //printf("offset at index %d, s: %d, g: %d, f: %d, has been transform from %d,%d to %d\n", input_index, output_index, s_input_index, g_input_index, f_input_index, offset_y.x, offset_y.y, output_offset.x);

    // offset should be in bytes !!! (not in 4 bytes as for float or 16 bytes as for float4)
    if (NUM_READ_FEATURES > 0) output_offset.x = ((int)floorf(offset_y.x) * (SharedMem::PITCHED_WIDTH) + (int)floorf(offset_x.x) * BATCH_IMAGES * BLOCK_IMAGES) * sizeof(float);
    if (NUM_READ_FEATURES > 1) output_offset.y = ((int)floorf(offset_y.y) * (SharedMem::PITCHED_WIDTH) + (int)floorf(offset_x.y) * BATCH_IMAGES * BLOCK_IMAGES) * sizeof(float);
    if (NUM_READ_FEATURES > 2) output_offset.z = ((int)floorf(offset_y.z) * (SharedMem::PITCHED_WIDTH) + (int)floorf(offset_x.z) * BATCH_IMAGES * BLOCK_IMAGES) * sizeof(float);
    if (NUM_READ_FEATURES > 3) output_offset.w = ((int)floorf(offset_y.w) * (SharedMem::PITCHED_WIDTH) + (int)floorf(offset_x.w) * BATCH_IMAGES * BLOCK_IMAGES) * sizeof(float);

    // If offset is to odd number then access to shared memory will not be alligned (e.g. cannot use float2)
    // so we replicate data in shared memory for accesses to odd offsets with addintional buffer.
    // We now need to ensure that offset here will address that buffer instead - buffers are stored one after
    // another so we just need to add the size of one buffer minus the alignment value in x dimension

    // NOTE: we disabled this as it is not needed with the current version of the algorithm
    if (0) {
        if (NUM_READ_FEATURES > 0) if (output_offset.x % 2 == 1) output_offset.x += (SharedMem::PITCHED_WIDTH * SharedMem::ALLOC_HEIGHT) - 1;
        if (NUM_READ_FEATURES > 1) if (output_offset.y % 2 == 1) output_offset.y += (SharedMem::PITCHED_WIDTH * SharedMem::ALLOC_HEIGHT) - 1;
        if (NUM_READ_FEATURES > 2) if (output_offset.z % 2 == 1) output_offset.z += (SharedMem::PITCHED_WIDTH * SharedMem::ALLOC_HEIGHT) - 1;
        if (NUM_READ_FEATURES > 3) if (output_offset.w % 2 == 1) output_offset.w += (SharedMem::PITCHED_WIDTH * SharedMem::ALLOC_HEIGHT) - 1;
    }
    // outer index position for output with offset and weights combined
    // in this case we have the following structure
    //    [F / (BATCH_FEATURES_SIZE * BLOCK_FEATURES)] x [S / (BATCH_COMPUTE_SUBFEATURES_SIZE*BATCH_MEM_SUBFEATURES_SIZE)] x [G / BATCH_GAUSS_SIZE]
    //			x [OFFSET_MEMORY + WEIGHTS_MEMORY]
    //  where
    //      OFFSET_MEMORY = [BATCH_COMPUTE_SUBFEATURES_SIZE * BATCH_MEM_SUBFEATURES_SIZE] x [BATCH_GAUSS_SIZE] x [BATCH_FEATURES_SIZE/4] x [BLOCK_FEATURES];
    //      WEIGHTS_MEMORY = [BATCH_COMPUTE_SUBFEATURES_SIZE * BATCH_MEM_SUBFEATURES_SIZE] x [BATCH_GAUSS_SIZE] x [PIXELS_INTERPOLATION_SIZE] x [BATCH_FEATURES_SIZE/4] x [BLOCK_FEATURES];


    int output_index_offsets = (OFFSET(main_f_index,
                                       main_s_index,
                                       main_g_index,
                                       0,
                                       dim8_size, dim7_size, dim6_size, (max(4,dim5_size * dim4_size * 1 * dim2_size * dim1_size) +
                                                                         dim5_size * dim4_size * dim3_size * dim2_size * dim1_size)) +
                                OFFSET5(s_mem_index,
                                        g_index,
                                        0,
                                        f_batch_index,
                                        f_block_index,
                                        dim5_size, dim4_size, 1, dim2_size, dim1_size)) * NUM_READ_FEATURES;
    /*int output_index_offsets = (OFFSET8(main_f_index,
                                        main_s_index,
                                        main_g_index,
                                        s_mem_index,
                                        g_index,
                                        0,
                                        f_batch_index,
                                        f_block_index,

                                       dim8_size, dim7_size, dim6_size, dim5_size , dim4_size , (dim3_size+1), dim2_size , dim1_size)) * NUM_READ_FEATURES;*/
    int* out_off[2];
    // default version
    out_off[0] = prepared_filter_offsets4 != NULL ? reinterpret_cast<int*>(prepared_filter_offsets4) + output_index : NULL;
    // for version where offsets and weights are in the same buffer
    out_off[1] = prepared_filter_offsets_and_weights != NULL ? reinterpret_cast<int*>(prepared_filter_offsets_and_weights) + output_index_offsets : NULL;

    for (int i = 0; i < 2; ++i) {
        if (out_off[i] == NULL) continue;

        if (NUM_READ_FEATURES > 0) out_off[i][0] = output_offset.x;
        if (NUM_READ_FEATURES > 1) out_off[i][1] = output_offset.y;
        if (NUM_READ_FEATURES > 2) out_off[i][2] = output_offset.z;
        if (NUM_READ_FEATURES > 3) out_off[i][3] = output_offset.w;

    }

    // for weights we integrate interpolation values into four sets of weights
    int output_index_0[2];
    output_index_0[0] =  OFFSET8(main_f_index,
                                 main_s_index,
                                 main_g_index,
                                 s_mem_index,
                                 g_index,
                                 0,
                                 f_batch_index,
                                 f_block_index,
                                 dim8_size, dim7_size, dim6_size, dim5_size, dim4_size, dim3_size, dim2_size, dim1_size) * NUM_READ_FEATURES;

    output_index_0[1] =  (OFFSET(main_f_index,
                                 main_s_index,
                                 main_g_index,
                                 0,
                                 dim8_size, dim7_size, dim6_size, (max(4, dim5_size * dim4_size * 1 * dim2_size * dim1_size) +
                                                                   dim5_size * dim4_size * dim3_size * dim2_size * dim1_size)) +
                          + max(4,dim5_size * dim4_size * 1 * dim2_size * dim1_size) + // offset for OFFSET_BUFFER
                          OFFSET5(s_mem_index,
                                  g_index,
                                  0,
                                  f_batch_index,
                                  f_block_index,
                                  dim5_size, dim4_size, dim3_size, dim2_size, dim1_size) )* NUM_READ_FEATURES;
    /*output_index_0[1] = (OFFSET8(main_f_index,
                                        main_s_index,
                                        main_g_index,
                                        s_mem_index,
                                        g_index,
                                        1,
                                        f_batch_index,
                                        f_block_index,

                                        dim8_size, dim7_size, dim6_size, dim5_size , dim4_size , (dim3_size+1), dim2_size , dim1_size)) * NUM_READ_FEATURES;*/

    ELEMENT_FLOAT_TYPE* out_prepared_filter_weights4[2];

    out_prepared_filter_weights4[0] = prepared_filter_weights4;
    out_prepared_filter_weights4[1] = reinterpret_cast<ELEMENT_FLOAT_TYPE*>(prepared_filter_offsets_and_weights);

    // prepare factors for interpolation
    // but set value to zero if not using interpolation for appropriate corrners
    float4 interp_offset_y,interp_offset_x;

    // get x-floor(x)
    if (NUM_READ_FEATURES > 0) interp_offset_x.x = PIXELS_INTERPOLATION_Dx == 2 ? offset_x.x - floorf(offset_x.x) : 0;
    if (NUM_READ_FEATURES > 1) interp_offset_x.y = PIXELS_INTERPOLATION_Dx == 2 ? offset_x.y - floorf(offset_x.y) : 0;
    if (NUM_READ_FEATURES > 2) interp_offset_x.z = PIXELS_INTERPOLATION_Dx == 2 ? offset_x.z - floorf(offset_x.z) : 0;
    if (NUM_READ_FEATURES > 3) interp_offset_x.w = PIXELS_INTERPOLATION_Dx == 2 ? offset_x.w - floorf(offset_x.w) : 0;

    // get y-floor(y)
    if (NUM_READ_FEATURES > 0) interp_offset_y.x = PIXELS_INTERPOLATION_Dy == 2 ? offset_y.x - floorf(offset_y.x) : 0;
    if (NUM_READ_FEATURES > 1) interp_offset_y.y = PIXELS_INTERPOLATION_Dy == 2 ? offset_y.y - floorf(offset_y.y) : 0;
    if (NUM_READ_FEATURES > 2) interp_offset_y.z = PIXELS_INTERPOLATION_Dy == 2 ? offset_y.z - floorf(offset_y.z) : 0;
    if (NUM_READ_FEATURES > 3) interp_offset_y.w = PIXELS_INTERPOLATION_Dy == 2 ? offset_y.w - floorf(offset_y.w) : 0;




    float4 factor_00, factor_01, factor_10, factor_11;

    // Instead of interpolation of data we perform interpolation on error to share data over several sub-features (w,mu1,mu2,sigma)
    // and reduce data loading.
    // To achieve interpolation of error we need to reverse the interpolation factors

    if (NUM_READ_FEATURES > 0) factor_11.x = interp_offset_x.x * interp_offset_y.x;
    if (NUM_READ_FEATURES > 1) factor_11.y = interp_offset_x.y * interp_offset_y.y;
    if (NUM_READ_FEATURES > 2) factor_11.z = interp_offset_x.z * interp_offset_y.z;
    if (NUM_READ_FEATURES > 3) factor_11.w = interp_offset_x.w * interp_offset_y.w;

    if (NUM_READ_FEATURES > 0) factor_01.x = (interp_offset_x.x) * (1-interp_offset_y.x);
    if (NUM_READ_FEATURES > 1) factor_01.y = (interp_offset_x.y) * (1-interp_offset_y.y);
    if (NUM_READ_FEATURES > 2) factor_01.z = (interp_offset_x.z) * (1-interp_offset_y.z);
    if (NUM_READ_FEATURES > 3) factor_01.w = (interp_offset_x.w) * (1-interp_offset_y.w);

    if (NUM_READ_FEATURES > 0) factor_10.x = (1-interp_offset_x.x) * (interp_offset_y.x);
    if (NUM_READ_FEATURES > 1) factor_10.y = (1-interp_offset_x.y) * (interp_offset_y.y);
    if (NUM_READ_FEATURES > 2) factor_10.z = (1-interp_offset_x.z) * (interp_offset_y.z);
    if (NUM_READ_FEATURES > 3) factor_10.w = (1-interp_offset_x.w) * (interp_offset_y.w);

    if (NUM_READ_FEATURES > 0) factor_00.x = (1-interp_offset_x.x) * (1-interp_offset_y.x);
    if (NUM_READ_FEATURES > 1) factor_00.y = (1-interp_offset_x.y) * (1-interp_offset_y.y);
    if (NUM_READ_FEATURES > 2) factor_00.z = (1-interp_offset_x.z) * (1-interp_offset_y.z);
    if (NUM_READ_FEATURES > 3) factor_00.w = (1-interp_offset_x.w) * (1-interp_offset_y.w);

    const float* w = reinterpret_cast<const float*>(filter_weights4) + input_index;


    for (int i = 0 ; i< 2; ++i) {

        if (out_prepared_filter_weights4[i] == NULL)
            continue;

        // create weights with interpolation factors
        // dx=0,dy=0
        if (NUM_READ_FEATURES > 0) reinterpret_cast<float*>(out_prepared_filter_weights4[i])[output_index_0[i] + 0] = w[0 * input_f_offset] * factor_00.x;
        if (NUM_READ_FEATURES > 1) reinterpret_cast<float*>(out_prepared_filter_weights4[i])[output_index_0[i] + 1] = w[1 * input_f_offset] * factor_00.y;
        if (NUM_READ_FEATURES > 2) reinterpret_cast<float*>(out_prepared_filter_weights4[i])[output_index_0[i] + 2] = w[2 * input_f_offset] * factor_00.z;
        if (NUM_READ_FEATURES > 3) reinterpret_cast<float*>(out_prepared_filter_weights4[i])[output_index_0[i] + 3] = w[3 * input_f_offset] * factor_00.w;

        if (PIXELS_INTERPOLATION_Dx == 2)  {
            // dx=1,dy=0
            int output_index_1 = output_index_0[i] + 1 *  (dim1_size * dim2_size) * NUM_READ_FEATURES;
            if (NUM_READ_FEATURES > 0) reinterpret_cast<float*>(out_prepared_filter_weights4[i])[output_index_1 + 0] = w[0 * input_f_offset] * factor_01.x;
            if (NUM_READ_FEATURES > 1) reinterpret_cast<float*>(out_prepared_filter_weights4[i])[output_index_1 + 1] = w[1 * input_f_offset] * factor_01.y;
            if (NUM_READ_FEATURES > 2) reinterpret_cast<float*>(out_prepared_filter_weights4[i])[output_index_1 + 2] = w[2 * input_f_offset] * factor_01.z;
            if (NUM_READ_FEATURES > 3) reinterpret_cast<float*>(out_prepared_filter_weights4[i])[output_index_1 + 3] = w[3 * input_f_offset] * factor_01.w;
        }
        if (PIXELS_INTERPOLATION_Dy == 2)  {
            // dx=0,dy=1
            int output_index_2 = output_index_0[i] + 2 *  (dim1_size * dim2_size) * NUM_READ_FEATURES;
            if (NUM_READ_FEATURES > 0) reinterpret_cast<float*>(out_prepared_filter_weights4[i])[output_index_2 + 0] = w[0 * input_f_offset] * factor_10.x;
            if (NUM_READ_FEATURES > 1) reinterpret_cast<float*>(out_prepared_filter_weights4[i])[output_index_2 + 1] = w[1 * input_f_offset] * factor_10.y;
            if (NUM_READ_FEATURES > 2) reinterpret_cast<float*>(out_prepared_filter_weights4[i])[output_index_2 + 2] = w[2 * input_f_offset] * factor_10.z;
            if (NUM_READ_FEATURES > 3) reinterpret_cast<float*>(out_prepared_filter_weights4[i])[output_index_2 + 3] = w[3 * input_f_offset] * factor_10.w;
        }
        if (PIXELS_INTERPOLATION_Dx == 2 && PIXELS_INTERPOLATION_Dy == 2)  {
            // dx=1,dy=1
            int output_index_3 = output_index_0[i] + 3 *  (dim1_size * dim2_size) * NUM_READ_FEATURES;
            if (NUM_READ_FEATURES > 0) reinterpret_cast<float*>(out_prepared_filter_weights4[i])[output_index_3 + 0] = w[0 * input_f_offset] * factor_11.x;
            if (NUM_READ_FEATURES > 1) reinterpret_cast<float*>(out_prepared_filter_weights4[i])[output_index_3 + 1] = w[1 * input_f_offset] * factor_11.y;
            if (NUM_READ_FEATURES > 2) reinterpret_cast<float*>(out_prepared_filter_weights4[i])[output_index_3 + 2] = w[2 * input_f_offset] * factor_11.z;
            if (NUM_READ_FEATURES > 3) reinterpret_cast<float*>(out_prepared_filter_weights4[i])[output_index_3 + 3] = w[3 * input_f_offset] * factor_11.w;
        }
    }
}


template <typename BlockIndexingT>
class DAUConvFwdInputWeightAndOffsets {
    enum {
        // values from main block indexing sizes
        NUM_BATCH_FEATURES =  BlockIndexingT::BATCH_COMPUTE_FEATURES_SIZE >= 4 ? 4 :
                              (BlockIndexingT::BATCH_COMPUTE_FEATURES_SIZE >= 2 ? 2 : 1),

        OFFSET_BLOCK_MEM_SIZE = MAX(4,BlockIndexingT::BATCH_COMPUTE_SUBFEATURES_SIZE * BlockIndexingT::BATCH_MEM_SUBFEATURES_SIZE * BlockIndexingT::BATCH_GAUSS_SIZE * BlockIndexingT::BATCH_FEATURES_SIZE * BlockIndexingT::BLOCK_FEATURES),
        WEIGHT_BLOCK_MEM_SIZE = BlockIndexingT::BATCH_COMPUTE_SUBFEATURES_SIZE * BlockIndexingT::BATCH_MEM_SUBFEATURES_SIZE * BlockIndexingT::BATCH_GAUSS_SIZE * BlockIndexingT::PIXELS_INTERPOLATION_Dx*BlockIndexingT::PIXELS_INTERPOLATION_Dy * BlockIndexingT::BATCH_FEATURES_SIZE * BlockIndexingT::BLOCK_FEATURES,

    };
    const int img_width;
    const int img_height;
    const int N;
    const int S;
    const int G;
    const int F;

    dim3 threadsPerBlock;
    dim3 numBlocks;

public:

    DAUConvFwdInputWeightAndOffsets(const int img_width, const int img_height, const int N, const int F, const int S, const int G) :
            img_width(img_width), img_height(img_height), N(N), F(F), S(S), G(G)
    {

        int NUM_THREADS_PER_BLOCK = F/NUM_BATCH_FEATURES >= 16 && S > 16? 16 :
                                    (F/NUM_BATCH_FEATURES >= 8 && S > 8? 8 :
                                     (F/NUM_BATCH_FEATURES >= 4 && S > 4 ? 4 : 1));

        threadsPerBlock = dim3(NUM_THREADS_PER_BLOCK, 1, NUM_THREADS_PER_BLOCK);

        numBlocks = dim3((int)ceil((F/NUM_BATCH_FEATURES)/threadsPerBlock.x),
                         (int)ceil(G/threadsPerBlock.y),
                         (int)ceil(S/threadsPerBlock.z));
    }

    size_t get_offsets_allocation_size() {
        return sizeof(float) * ( S*G*F + OFFSET_BLOCK_MEM_SIZE);
    }
    size_t get_weights_allocation_size() {
        return sizeof(float) * ( BlockIndexingT::PIXELS_INTERPOLATION_Dx*BlockIndexingT::PIXELS_INTERPOLATION_Dy*S*G*F + WEIGHT_BLOCK_MEM_SIZE);
    }

    size_t get_allocation_size() {
        return sizeof(float) * ( (1 + BlockIndexingT::PIXELS_INTERPOLATION_Dx*BlockIndexingT::PIXELS_INTERPOLATION_Dy) * S*G*F + OFFSET_BLOCK_MEM_SIZE);
    }

    void create_input(float* prepared_filter_weights, int* prepared_filter_offsets, float* prepared_filter_offsets_and_weights, // OUTPUT
                      const float* filter_weights, const float* filter_offsets_float_x, const float* filter_offsets_float_y, // INPUT
                      const int kernel_w, const int kernel_h, const DAUConvForward<float>::PARAM_FORMAT param_format, const bool offsets_already_centered, cudaStream_t streamId = NULL) {

        if (NUM_BATCH_FEATURES == 4)
            perpare_weights_and_offsets<BlockIndexingT, float4, int4><<<numBlocks,threadsPerBlock, 0, streamId>>>(filter_weights, filter_offsets_float_x, filter_offsets_float_y, prepared_filter_weights, prepared_filter_offsets, prepared_filter_offsets_and_weights, S, G, F, kernel_w, kernel_h, param_format, offsets_already_centered);
        else if (NUM_BATCH_FEATURES == 2)
            perpare_weights_and_offsets<BlockIndexingT, float2, int2><<<numBlocks,threadsPerBlock, 0, streamId>>>(filter_weights, filter_offsets_float_x, filter_offsets_float_y, prepared_filter_weights, prepared_filter_offsets, prepared_filter_offsets_and_weights, S, G, F, kernel_w, kernel_h, param_format, offsets_already_centered);
        else
            perpare_weights_and_offsets<BlockIndexingT, float, int><<<numBlocks,threadsPerBlock, 0, streamId>>>(filter_weights, filter_offsets_float_x, filter_offsets_float_y, prepared_filter_weights, prepared_filter_offsets, prepared_filter_offsets_and_weights, S, G, F, kernel_w, kernel_h, param_format, offsets_already_centered);

        if (0) {

            static const int NUM_READ_FEATURES =  BlockIndexingT::BATCH_COMPUTE_FEATURES_SIZE >= 4 ? 4 :
                                                  (BlockIndexingT::BATCH_COMPUTE_FEATURES_SIZE >= 2 ? 2 : 1);


            static const int dim1_size = BlockIndexingT::BLOCK_FEATURES;
            static const int dim2_size = BlockIndexingT::BATCH_FEATURES_SIZE/NUM_READ_FEATURES;
            static const int dim3_size = BlockIndexingT::PIXELS_INTERPOLATION_Dx * BlockIndexingT::PIXELS_INTERPOLATION_Dy;
            static const int dim4_size = BlockIndexingT::BATCH_GAUSS_SIZE;
            static const int dim5_size = BlockIndexingT::BATCH_COMPUTE_SUBFEATURES_SIZE * BlockIndexingT::BATCH_MEM_SUBFEATURES_SIZE;

            int dim6_size = G / dim4_size;
            int dim7_size = S / dim5_size;
            int dim8_size = F / (dim1_size * dim2_size * NUM_READ_FEATURES);

            size_t num_el = this->get_allocation_size()/sizeof(float);
            float* prepared_filter_offsets_and_weights_cpu = new float[num_el];
            int* prepared_filter_offsets_and_weights_cpu_as_int = (int*)prepared_filter_offsets_and_weights_cpu;

            for (int i = 0; i < num_el; ++i)
                prepared_filter_offsets_and_weights_cpu[i] = -1;

            cudaMemcpy(prepared_filter_offsets_and_weights_cpu, prepared_filter_offsets_and_weights, this->get_allocation_size(), cudaMemcpyDeviceToHost);
            cudaDeviceSynchronize();

            //for (int i = 0; i < (img_width + 2*MAX_OFFSET)*( img_height + 2*MAX_OFFSET)* I*S; ++i) {
            int index =0;
            for (int f_main_idx = 0; f_main_idx < dim8_size; ++f_main_idx) {
                for (int s_main_idx = 0; s_main_idx < dim7_size; ++s_main_idx) {
                    for (int g_main_idx = 0; g_main_idx < dim6_size; ++g_main_idx) {
                        printf("main_idx: f,s,g= %d,%d,%d \n", f_main_idx,s_main_idx,g_main_idx);
                        printf("offsets:\n");
                        for (int sh_mem_idx = 0; sh_mem_idx < dim5_size; ++sh_mem_idx) {
                            for (int g = 0; g < dim4_size; ++g) {
                                printf("sh=%d, g=%d: ", sh_mem_idx, g);
                                for (int f_batch = 0; f_batch < dim2_size; ++f_batch) {
                                    for (int f_block = 0; f_block < dim1_size; ++f_block) {
                                        for (int ff = 0; ff < NUM_READ_FEATURES; ++ff) {
                                            std::cout << prepared_filter_offsets_and_weights_cpu_as_int[index] << " ";
                                            index++;
                                        }
                                    }
                                }
                                printf("\n");
                            }
                        }
                        printf("weights:\n");
                        for (int sh_mem_idx = 0; sh_mem_idx < dim5_size; ++sh_mem_idx) {
                            for (int g = 0; g < dim4_size; ++g) {
                                printf("sh=%d, g=%d: \n", sh_mem_idx, g);
                                for (int px = 0; px < dim3_size; ++px) {
                                    printf("px=%d:", px);
                                    for (int f_batch = 0; f_batch < dim2_size; ++f_batch) {
                                        for (int f_block = 0; f_block < dim1_size; ++f_block) {
                                            for (int ff = 0; ff < NUM_READ_FEATURES; ++ff) {
                                                std::cout << prepared_filter_offsets_and_weights_cpu[index] << " ";
                                                index++;
                                            }
                                        }
                                    }
                                    printf("\n");
                                }
                            }
                        }
                        printf("\n");
                    }
                }
            }
            std::cout << std::endl;
            std::cout << std::endl;
            std::cout << std::endl;
            std::cout << std::endl;
        }
    }
};

template<int _IMG_SIZE_W, int _IMG_SIZE_H, int _MAX_OFFSET, int _WARP_PIXELS_X, int _WARP_PIXELS_Y, int _BLOCK_IMAGES, bool _USE_INTERPOLATION, bool _SINGLE_FEATURE, bool _SINGLE_SUBFEATURE>
class DAUConvForwardCUDA {
    enum {
        // Variable parameters

        // IMG_WIDTH and IMG_HEIGHT: 	32x32 .. N and M > 32
        // 								16x16 .. otherwise

        // current implementation allows min 32px in width, but this can be reduced by setting BATCH_IMAGES=2 or higher
        // however, this has not been tested yet !!


        // BATCH_IMAGES :	2	.. N >= 2
        // 					1	.. N == 1

        // MAX_OFFSET:	4 if kernels <= 9
        //				8 if kernels <= 17
        //				16 if kernels <= 33

        // BATCH_FEATURES_SIZE * BLOCK_FEATURES:  	16 min allowed
        // BLOCK_SUBFEATURES:  	2 min allowed
        // BATCH_GAUSS_SIZE:	2 min allowed

        // split threads in a WARP over multiple images to handle smaller images
        BLOCK_IMAGES = _BLOCK_IMAGES,

        // make use of 32 pixels per WARP but we can split them per multiple images to handle 16px wide images or smaller - decision to use 16px x 2 BATCH_IMGs is done by caller
        // in case we have 1x1 image then use 1x1 for _WARP_PIXELS_X x _WARP_PIXELS_X and adjust batching of images and features acordingly
        WARP_PIXELS_X = _WARP_PIXELS_X, // this should be only 1, 16 or 32 !!
        WARP_PIXELS_Y = _WARP_PIXELS_Y, // this should be only 1 or 8

        IMG_WIDTH = MAX(WARP_PIXELS_X,_IMG_SIZE_W), // NOTE: 32 <= BLOCK_X * BATCH_PIXELS_SIZE_X
        IMG_HEIGHT = MAX(WARP_PIXELS_Y,_IMG_SIZE_H), // NOTE:  8 <= BLOCK_Y * BATCH_PIXELS_SIZE_Y
        MAX_OFFSET = _MAX_OFFSET,
        //BATCH_IMAGES = _USE_INTERPOLATION ? _BATCH_IMAGES : 2,
        BATCH_IMAGES=1,

        // to use 16 pixels wide input we can use:
        // BLOCK_IMAGES = 2,

        // special cases for:
        //	- BATCH_GAUSS_SIZE == 1
        //	- INTERPOLATION == false

        PIXELS_INTERPOLATION_Dx = _USE_INTERPOLATION ? 2 : 1,
        PIXELS_INTERPOLATION_Dy = _USE_INTERPOLATION ? 2 : 1,

        // each block of multiple threads handles:
        //  - pixel:        BLOCK_X * BLOCK_Y
        //  - features:     BLOCK_FEATURES * BATCH_FEATURES_SIZE
        //  - subfeatures:	all subfeatures
        //  - gauss krn:    BATCH_GAUSS_SIZE

        // within block each thread handles:
        //  - pixels:       BATCH_PIXELS_SIZE_X * BATCH_PIXELS_SIZE_Y
        //  - features:     BATCH_FEATURES_SIZE
        //  - subfeatures:  all subfeatures
        //  - gauss krn:    BATCH_GAUSS_SIZE

        // each thread handles features and subfeatures as:
        //  - features:     one warp always handles only BATCH_FEATURES_SIZE features, but N warps are used for different features where N=BLOCK_FEATURES
        //  - subfeatures:  iterates over all subfeatures, but at once only BATCH_MEM_SUBFEATURES_SIZE*BATCH_COMPUTE_SUBFEATURES_SIZE subfeatures are loaded

        // a multiple of BATCH_GAUSS_SIZE gaussian kernels is allowed and each batch of BATCH_GAUSS_SIZE is handled by individual block

        // Fixed parameters
        NUM_SM = 1, // number of streaming multiprocessors (not used)

        BATCH_PIXELS_SIZE_X = 1,
        BATCH_PIXELS_SIZE_Y = WARP_PIXELS_Y,

        BLOCK_X = MAX(1,WARP_PIXELS_X/BATCH_PIXELS_SIZE_X),
        BLOCK_Y = MAX(1,WARP_PIXELS_Y/BATCH_PIXELS_SIZE_Y),
        BLOCK_FEATURES = _SINGLE_FEATURE ? 1 : 8,

        BATCH_FEATURES_SIZE = _SINGLE_FEATURE ? 1 : ( (_USE_INTERPOLATION ? 2 : 4) * (WARP_PIXELS_Y == 1 ? 1 : 1) ),
        BATCH_COMPUTE_FEATURES_SIZE = _SINGLE_FEATURE ? 1 : 2,

        BATCH_COMPUTE_SUBFEATURES_SIZE = 1,
        BATCH_MEM_SUBFEATURES_SIZE = _SINGLE_SUBFEATURE ? 1 : 2,
        BATCH_GAUSS_SIZE = 2,
    };
    const int img_width,img_height;
    const int I,S,F,G;

    int new_img_parts_width;
    int new_img_parts_height;

    dim3 threadsPerBlock;
    dim3 numBlocks;
public:

    typedef class BlockIndexing<NUM_SM,
            BLOCK_X, BLOCK_Y, BLOCK_FEATURES, BLOCK_IMAGES,
            BATCH_PIXELS_SIZE_X, BATCH_PIXELS_SIZE_Y,
            PIXELS_INTERPOLATION_Dx, PIXELS_INTERPOLATION_Dy,
            BATCH_FEATURES_SIZE,
            BATCH_COMPUTE_FEATURES_SIZE,
            BATCH_COMPUTE_SUBFEATURES_SIZE,
            BATCH_MEM_SUBFEATURES_SIZE,
            BATCH_GAUSS_SIZE,
            BATCH_IMAGES,
            IMG_WIDTH, IMG_HEIGHT,
            MAX_OFFSET,
            //false, 5, 2> BlockIndexingPipelineT;
            false, 4, 2> BlockIndexingPipelineT;
    // false, 4, 2 == USE_SEPERATE_OFFSET_AND_WEIGHTS_BUFFER, LOAD_DATA_DELAY, LOAD_W_AND_OFF_DELAY

    DAUConvFwdInputImage<BlockIndexingPipelineT> image_cuda_prepare;
    DAUConvFwdInputWeightAndOffsets<BlockIndexingPipelineT> weight_and_offsets_cuda_prepare;

    DAUConvForwardCUDA(const DAUConvForward<float>::CUDAParams& p) :
            img_width(p.img_width), img_height(p.img_height), I(p.I), S(p.S), F(p.F), G(p.G),

            // we will split image into patches of size [IMG_HEIGHT x IMG_WIDTH] so use that as image size, however,
            // we need to increase the number of images that will be process as each patch is now considered as one image
            // there is no need to recombine the output since we just sum over all patches to get gradients

            new_img_parts_width((int)ceil((float)img_width / IMG_WIDTH)),
            new_img_parts_height((int)ceil((float)img_height / IMG_HEIGHT)),

            // initialize classes that will generate inputs
            image_cuda_prepare(p.img_width_in, p.img_height_in, img_width, img_height, I, S, new_img_parts_width,new_img_parts_height),
            weight_and_offsets_cuda_prepare(img_width, img_height, I, F, S, G) {

        class BlockIndexingPipelineT::Launch block_indexing;

        threadsPerBlock = block_indexing.getThreadsPerBlock(I * new_img_parts_width * new_img_parts_height, F, S, IMG_WIDTH, IMG_HEIGHT);
        numBlocks = block_indexing.getBlocksPerGrid(I * new_img_parts_width * new_img_parts_height, F, S, G, IMG_WIDTH, IMG_HEIGHT);

    }


    void get_allocation_sizes(DAUConvForward<float>::CUDAParams& p) {

        if (p.alloc_img != NULL) *p.alloc_img = image_cuda_prepare.get_allocation_size();
        if (p.alloc_w != NULL) *p.alloc_w = weight_and_offsets_cuda_prepare.get_weights_allocation_size();
        if (p.alloc_off != NULL) *p.alloc_off = weight_and_offsets_cuda_prepare.get_offsets_allocation_size();
    }

    void run_kernel(DAUConvForward<float>::CUDAParams& p) {

        //caffe_gpu_set<float>(I * F * img_width * img_height, (float)0, p.output);
        //caffe_gpu_set<float>(image_cuda_prepare.get_allocation_size()/sizeof(float), (float)0, p.prepared_filtered_images);


        //CUDA_CHECK(cudaMemsetAsync(p.output, 0, sizeof(float) * I * F * img_width * img_height, p.streamId));
        //CUDA_CHECK(cudaMemsetAsync(p.prepared_filtered_images, 0, image_cuda_prepare.get_allocation_size(), p.streamId));

        //CUDA_CHECK(cudaDeviceSynchronize());
        {
//#define PROFILE_CUDA
#ifdef PROFILE_CUDA
            std::cout << "started create_input_with_border" << std::endl;

    clock_t start_t = clock();
#endif
            p.prepared_filtered_images = image_cuda_prepare.create_input(p.prepared_filtered_images, p.filtered_images, p.streamId);
#ifdef PROFILE_CUDA
            cudaDeviceSynchronize();

    clock_t end_t = clock();
    CUDA_POST_KERNEL_CHECK;
    std::cout << "create_input_with_border in " << (((float)(end_t-start_t))/CLOCKS_PER_SEC) << std::endl;
#endif
            CUDA_POST_KERNEL_CHECK;
            //CUDA_CHECK(cudaDeviceSynchronize());
        }

        {
#ifdef PROFILE_CUDA
            std::cout << "started copy_permute_weights" << std::endl;

    clock_t start_t = clock();
#endif
            weight_and_offsets_cuda_prepare.create_input(p.prepared_filter_weights, p.prepared_filter_offsets, p.prepared_filter_offsets_and_weights, p.filter_weights, p.filter_offsets_float_x, p.filter_offsets_float_y, p.kernel_w, p.kernel_h, p.param_format, p.offsets_already_centered, p.streamId);
#ifdef PROFILE_CUDA
            cudaDeviceSynchronize();

    clock_t end_t = clock();
    CUDA_POST_KERNEL_CHECK;

    std::cout << "copy_permute_weights in " << (((float)(end_t-start_t))/CLOCKS_PER_SEC) << std::endl;
#endif
            CUDA_POST_KERNEL_CHECK;
            //CUDA_CHECK(cudaDeviceSynchronize());
        }
#ifdef PROFILE_CUDA
        std::cout << "started DAUConv_forward_pipeline_kernel" << std::endl;



std::cout << "threadsPerBlock " << threadsPerBlock.x << "," << threadsPerBlock.y << "," << threadsPerBlock.z << std::endl;
std::cout << "numBlocks " << numBlocks.x << "," << numBlocks.y << "," << numBlocks.z << std::endl;

for (int jj = 0; jj < 1; ++jj) {

    clock_t start_t = clock();
#endif
        DAUConv_forward_pipeline_kernel < BlockIndexingPipelineT,-1,-1><<<numBlocks,threadsPerBlock,0, p.streamId>>>(p.prepared_filtered_images, p.prepared_filter_offsets, p.prepared_filter_weights, p.prepared_filter_offsets_and_weights, p.output, I, S, F, G, img_width, img_height, new_img_parts_width, new_img_parts_height);
#ifdef PROFILE_CUDA
        cudaDeviceSynchronize();

    clock_t end_t = clock();
    CUDA_POST_KERNEL_CHECK;

    std::cout << "DAUConv_forward_pipeline_kernel in " << (((float)(end_t-start_t))/CLOCKS_PER_SEC) << std::endl;
}
#endif
        CUDA_POST_KERNEL_CHECK;
        //CUDA_CHECK(cudaDeviceSynchronize());

    }
};

#ifdef DAU_USE_DUMMY_CUDA_IMPL
#define RUN_KERNEL_R0(CLASS_NAME, IMG_PATCH_SIZE_W, IMG_PATCH_SIZE_H, MAX_OFFSET, WARP_PIXELS_X, WARP_PIXELS_Y, BLOCK_IMAGES, USE_INTERPOLATION, SINGLE_FEATURE, SINGLE_SUBFEATURE, PARAMS, ...) ;
#else
#define RUN_KERNEL_R0(CLASS_NAME, IMG_PATCH_SIZE_W, IMG_PATCH_SIZE_H, MAX_OFFSET, WARP_PIXELS_X, WARP_PIXELS_Y, BLOCK_IMAGES, USE_INTERPOLATION, SINGLE_FEATURE, SINGLE_SUBFEATURE, PARAMS, ...) \
    { \
        CLASS_NAME<IMG_PATCH_SIZE_W, IMG_PATCH_SIZE_H, MAX_OFFSET, WARP_PIXELS_X, WARP_PIXELS_Y, BLOCK_IMAGES, USE_INTERPOLATION, SINGLE_FEATURE, SINGLE_SUBFEATURE> _kernel_class(PARAMS); \
		if (PARAMS.alloc_img != NULL || 	\
			PARAMS.alloc_w != NULL ||	 	\
			PARAMS.alloc_off != NULL) { 	\
			_kernel_class.get_allocation_sizes(PARAMS); \
		} else { \
			_kernel_class.run_kernel(PARAMS); \
		} \
	}
#endif

#ifdef DAU_ALLOW_INTERPOLATION_OFF
#define RUN_KERNEL_R1(CLASS_NAME, IMG_PATCH_SIZE_W, IMG_PATCH_SIZE_H, MAX_OFFSET, WARP_PIXELS_X, WARP_PIXELS_Y, BLOCK_IMAGES, USE_INTERPOLATION, SINGLE_FEATURE, SINGLE_SUBFEATURE, PARAMS,  ...) \
	if (USE_INTERPOLATION) { \
		RUN_KERNEL_R0(CLASS_NAME, IMG_PATCH_SIZE_W, IMG_PATCH_SIZE_H, MAX_OFFSET, WARP_PIXELS_X, WARP_PIXELS_Y, BLOCK_IMAGES, true, SINGLE_FEATURE, SINGLE_SUBFEATURE, PARAMS, __VA_ARGS__) \
	} else { \
		RUN_KERNEL_R0(CLASS_NAME, IMG_PATCH_SIZE_W, IMG_PATCH_SIZE_H, MAX_OFFSET, WARP_PIXELS_X, WARP_PIXELS_Y, BLOCK_IMAGES, false, SINGLE_FEATURE, SINGLE_SUBFEATURE, PARAMS, __VA_ARGS__) \
	}
#else
#define RUN_KERNEL_R1(CLASS_NAME, IMG_PATCH_SIZE_W, IMG_PATCH_SIZE_H, MAX_OFFSET, WARP_PIXELS_X, WARP_PIXELS_Y, BLOCK_IMAGES, USE_INTERPOLATION, SINGLE_FEATURE, SINGLE_SUBFEATURE, PARAMS,  ...) \
	if (USE_INTERPOLATION) { \
		RUN_KERNEL_R0(CLASS_NAME, IMG_PATCH_SIZE_W, IMG_PATCH_SIZE_H, MAX_OFFSET, WARP_PIXELS_X, WARP_PIXELS_Y, BLOCK_IMAGES, true, SINGLE_FEATURE, SINGLE_SUBFEATURE, PARAMS, __VA_ARGS__) \
	} else { \
		throw DAUConvNet::DAUException(string_format("Not compiled with DAU_ALLOW_INTERPOLATION_OFF flag: support for non-interpolation disabled.\n")); \
	}
#endif

#define RUN_KERNEL_R2(CLASS_NAME, IMG_PATCH_SIZE_W, IMG_PATCH_SIZE_H, MAX_OFFSET, WARP_PIXELS_X, WARP_PIXELS_Y, BLOCK_IMAGES, USE_INTERPOLATION, SINGLE_FEATURE, SINGLE_SUBFEATURE, PARAMS, ...) \
	if (IMG_PATCH_SIZE_H >= 64) { \
		RUN_KERNEL_R1(CLASS_NAME, IMG_PATCH_SIZE_W, 64, MAX_OFFSET, WARP_PIXELS_X, WARP_PIXELS_Y, BLOCK_IMAGES, USE_INTERPOLATION, SINGLE_FEATURE, SINGLE_SUBFEATURE, PARAMS, __VA_ARGS__) \
	} else if (IMG_PATCH_SIZE_H >= 32) { \
		RUN_KERNEL_R1(CLASS_NAME, IMG_PATCH_SIZE_W, 32, MAX_OFFSET, WARP_PIXELS_X, WARP_PIXELS_Y, BLOCK_IMAGES, USE_INTERPOLATION, SINGLE_FEATURE, SINGLE_SUBFEATURE, PARAMS, __VA_ARGS__) \
	} else if (IMG_PATCH_SIZE_H >= 16) { \
        RUN_KERNEL_R1(CLASS_NAME, IMG_PATCH_SIZE_W, 16, MAX_OFFSET, WARP_PIXELS_X, WARP_PIXELS_Y, BLOCK_IMAGES, USE_INTERPOLATION, SINGLE_FEATURE, SINGLE_SUBFEATURE, PARAMS, __VA_ARGS__) \
	} else { \
        RUN_KERNEL_R1(CLASS_NAME, IMG_PATCH_SIZE_W, 8, MAX_OFFSET, WARP_PIXELS_X, WARP_PIXELS_Y, BLOCK_IMAGES, USE_INTERPOLATION, SINGLE_FEATURE, SINGLE_SUBFEATURE, PARAMS, __VA_ARGS__) \
	}

#define RUN_KERNEL_R3(CLASS_NAME, IMG_PATCH_SIZE_W, IMG_PATCH_SIZE_H, MAX_OFFSET, WARP_PIXELS_X, WARP_PIXELS_Y, BLOCK_IMAGES, USE_INTERPOLATION, SINGLE_FEATURE, SINGLE_SUBFEATURE, PARAMS, ...) \
    if (IMG_PATCH_SIZE_W >= 64) { \
		RUN_KERNEL_R2(CLASS_NAME, 64, IMG_PATCH_SIZE_H, MAX_OFFSET, WARP_PIXELS_X, WARP_PIXELS_Y, BLOCK_IMAGES, USE_INTERPOLATION, SINGLE_FEATURE, SINGLE_SUBFEATURE, PARAMS, __VA_ARGS__) \
	} else if (IMG_PATCH_SIZE_W >= 32) { \
		RUN_KERNEL_R2(CLASS_NAME, 32, IMG_PATCH_SIZE_H, MAX_OFFSET, WARP_PIXELS_X, WARP_PIXELS_Y, BLOCK_IMAGES, USE_INTERPOLATION, SINGLE_FEATURE, SINGLE_SUBFEATURE, PARAMS, __VA_ARGS__) \
	} else if (IMG_PATCH_SIZE_W >= 16) { \
		RUN_KERNEL_R2(CLASS_NAME, 16, IMG_PATCH_SIZE_H, MAX_OFFSET, WARP_PIXELS_X, WARP_PIXELS_Y, BLOCK_IMAGES, USE_INTERPOLATION, SINGLE_FEATURE, SINGLE_SUBFEATURE, PARAMS, __VA_ARGS__) \
	} else { \
        RUN_KERNEL_R2(CLASS_NAME, 8, IMG_PATCH_SIZE_H, MAX_OFFSET, WARP_PIXELS_X, WARP_PIXELS_Y, BLOCK_IMAGES, USE_INTERPOLATION, SINGLE_FEATURE, SINGLE_SUBFEATURE, PARAMS, __VA_ARGS__)  \
	}


#define RUN_KERNEL_R4_2(CLASS_NAME, IMG_PATCH_SIZE_W, IMG_PATCH_SIZE_H, MAX_OFFSET, WARP_PIXELS_X, WARP_PIXELS_Y, BLOCK_IMAGES, USE_INTERPOLATION, SINGLE_FEATURE, SINGLE_SUBFEATURE, PARAMS, ...) \
    if (WARP_PIXELS_X == 1 && WARP_PIXELS_Y == 1) { \
        /* just pass through if 1x1 since it was defined by RUN_KERNEL_R4 call*/ \
        RUN_KERNEL_R3(CLASS_NAME, IMG_PATCH_SIZE_W, IMG_PATCH_SIZE_H, MAX_OFFSET, WARP_PIXELS_X, WARP_PIXELS_Y, BLOCK_IMAGES, USE_INTERPOLATION, SINGLE_FEATURE, SINGLE_SUBFEATURE, PARAMS, __VA_ARGS__) \
    } else if (WARP_PIXELS_Y == 8) { \
        RUN_KERNEL_R3(CLASS_NAME, IMG_PATCH_SIZE_W, IMG_PATCH_SIZE_H, MAX_OFFSET, WARP_PIXELS_X, 8, BLOCK_IMAGES, USE_INTERPOLATION, SINGLE_FEATURE, SINGLE_SUBFEATURE, PARAMS, __VA_ARGS__) \
    } else { \
		throw DAUConvNet::DAUException(string_format("Unsupported WARP_PIXELS_Y: %d. Supported only 8 at the moment (or 1 when WARP_PIXELS_X==1 as well) \n", WARP_PIXELS_Y)); \
	}

#define RUN_KERNEL_R4(CLASS_NAME, IMG_PATCH_SIZE_W, IMG_PATCH_SIZE_H, MAX_OFFSET, WARP_PIXELS_X, WARP_PIXELS_Y, BLOCK_IMAGES, USE_INTERPOLATION, SINGLE_FEATURE, SINGLE_SUBFEATURE, PARAMS, ...) \
    if (IMG_PATCH_SIZE_W == 1 && IMG_PATCH_SIZE_H == 1 && WARP_PIXELS_X == 1 && WARP_PIXELS_Y == 1) { \
        /* We need to use 2x1 instead of 1x1 to ensure proper calculation for interpolated values (due to current implmenetation that may not work properly on right edges) */ \
        if (BLOCK_IMAGES % 16 == 0 && MAX_OFFSET <= 4) { \
            /* We cannot handle offsets bigger then 4 px when using 16 block images */ \
            RUN_KERNEL_R1(CLASS_NAME, 2, 1, 4, 2, 1, 16, USE_INTERPOLATION, SINGLE_FEATURE, SINGLE_SUBFEATURE, PARAMS, __VA_ARGS__) \
        } else if (BLOCK_IMAGES % 8 == 0 && MAX_OFFSET <= 8) { \
		    RUN_KERNEL_R1(CLASS_NAME, 2, 1, MAX_OFFSET, 2, 1, 8, USE_INTERPOLATION, SINGLE_FEATURE, SINGLE_SUBFEATURE, PARAMS, __VA_ARGS__) \
        } else if (BLOCK_IMAGES % 2 == 0) { \
		    RUN_KERNEL_R1(CLASS_NAME, 2, 1, MAX_OFFSET, 2, 1, 2, USE_INTERPOLATION, SINGLE_FEATURE, SINGLE_SUBFEATURE, PARAMS, __VA_ARGS__) \
        } else { \
            RUN_KERNEL_R1(CLASS_NAME, 4, 1, MAX_OFFSET, 4, 1, 1, USE_INTERPOLATION, SINGLE_FEATURE, SINGLE_SUBFEATURE, PARAMS, __VA_ARGS__) \
        } \
    } else if (IMG_PATCH_SIZE_W == 8 && WARP_PIXELS_X == 8) { \
        /* We have 8px WARP_PIXELS_X sizes only for smaller patch sizes - but check just in case (fixing IMG_PATCH_SIZE_W avoids unneeded computation as well) */ \
        if (BLOCK_IMAGES % 4 == 0) { \
            RUN_KERNEL_R2(CLASS_NAME, 8, IMG_PATCH_SIZE_H, MAX_OFFSET, 8, 8, 4, USE_INTERPOLATION, SINGLE_FEATURE, SINGLE_SUBFEATURE, PARAMS, __VA_ARGS__) \
        } else if (BLOCK_IMAGES % 2 == 0) { \
		    RUN_KERNEL_R2(CLASS_NAME, 8, IMG_PATCH_SIZE_H, MAX_OFFSET, 8, 8, 2, USE_INTERPOLATION, SINGLE_FEATURE, SINGLE_SUBFEATURE, PARAMS, __VA_ARGS__) \
	    } else { \
		    RUN_KERNEL_R2(CLASS_NAME, 8, IMG_PATCH_SIZE_H, MAX_OFFSET, 8, 8, 1, USE_INTERPOLATION, SINGLE_FEATURE, SINGLE_SUBFEATURE, PARAMS, __VA_ARGS__) \
        } \
    } else if (WARP_PIXELS_X == 16) { \
		if (BLOCK_IMAGES % 2 == 0) { \
		    RUN_KERNEL_R3(CLASS_NAME, IMG_PATCH_SIZE_W, IMG_PATCH_SIZE_H, MAX_OFFSET, 16, 8, 2, USE_INTERPOLATION, SINGLE_FEATURE, SINGLE_SUBFEATURE, PARAMS, __VA_ARGS__) \
	    } else { \
		    RUN_KERNEL_R3(CLASS_NAME, IMG_PATCH_SIZE_W, IMG_PATCH_SIZE_H, MAX_OFFSET, 16, 8, 1, USE_INTERPOLATION, SINGLE_FEATURE, SINGLE_SUBFEATURE, PARAMS, __VA_ARGS__) \
        } \
    } else if (WARP_PIXELS_X == 32)  { \
        RUN_KERNEL_R3(CLASS_NAME, IMG_PATCH_SIZE_W, IMG_PATCH_SIZE_H, MAX_OFFSET, 32, 8, 1, USE_INTERPOLATION, SINGLE_FEATURE, SINGLE_SUBFEATURE, PARAMS, __VA_ARGS__) \
	} else { \
		throw DAUConvNet::DAUException(string_format("Unsupported WARP_PIXELS_X: %d. Supported only 16 or 32 at the moment (or 1 when WARP_PIXELS_Y==1 as well) \n", WARP_PIXELS_X)); \
	}
// NOTE: RUN_KERNEL_R5, RUN_KERNEL_R6 and RUN_KERNEL_R7 below are not called directly - instead they are implemented in seperate files to allow for parallel computation
#define RUN_KERNEL_R5(CLASS_NAME, IMG_PATCH_SIZE_W, IMG_PATCH_SIZE_H, MAX_OFFSET, WARP_PIXELS_X, WARP_PIXELS_Y, BLOCK_IMAGES, USE_INTERPOLATION, SINGLE_FEATURE, SINGLE_SUBFEATURE, PARAMS, ...) \
	if (MAX_OFFSET <= 4) { \
		RUN_KERNEL_R4(CLASS_NAME, IMG_PATCH_SIZE_W, IMG_PATCH_SIZE_H, 4, WARP_PIXELS_X, WARP_PIXELS_Y, BLOCK_IMAGES, USE_INTERPOLATION, SINGLE_FEATURE, SINGLE_SUBFEATURE, PARAMS, __VA_ARGS__) \
	} else if (MAX_OFFSET <= 8) { \
        RUN_KERNEL_R4(CLASS_NAME, IMG_PATCH_SIZE_W, IMG_PATCH_SIZE_H, 8, WARP_PIXELS_X, BLOCK_IMAGES, USE_INTERPOLATION, SINGLE_FEATURE, SINGLE_SUBFEATURE, PARAMS, __VA_ARGS__) \
	} else if (MAX_OFFSET <= 32) { \
        RUN_KERNEL_R4(CLASS_NAME, IMG_PATCH_SIZE_W, IMG_PATCH_SIZE_H, 32, WARP_PIXELS_X, BLOCK_IMAGES, USE_INTERPOLATION, SINGLE_FEATURE, SINGLE_SUBFEATURE, PARAMS, __VA_ARGS__) \
	} else { \
        throw DAUConvNet::DAUException(string_format("Unsupported filter size: %d. Supported only max up to 9x9 and 17x17 at the moment\n", MAX_OFFSET)); \
    }
    /*else if (MAX_OFFSET <= 33) { \
        RUN_KERNEL_R2(CLASS_NAME, IMG_PATCH_SIZE, 16, BATCH_IMAGES, USE_INTERPOLATION, __VA_ARGS__) \
    */

#define RUN_KERNEL_R6(CLASS_NAME, IMG_PATCH_SIZE_W, IMG_PATCH_SIZE_H, MAX_OFFSET, WARP_PIXELS_X, WARP_PIXELS_Y, BLOCK_IMAGES, USE_INTERPOLATION, SINGLE_FEATURE, SINGLE_SUBFEATURE, PARAMS,  ...) \
	if (SINGLE_SUBFEATURE) { \
		RUN_KERNEL_R5(CLASS_NAME, IMG_PATCH_SIZE_W, IMG_PATCH_SIZE_H, MAX_OFFSET, WARP_PIXELS_X, WARP_PIXELS_Y, BLOCK_IMAGES, USE_INTERPOLATION, SINGLE_FEATURE, true, PARAMS, __VA_ARGS__) \
	} else { \
		RUN_KERNEL_R5(CLASS_NAME, IMG_PATCH_SIZE_W, IMG_PATCH_SIZE_H, MAX_OFFSET, WARP_PIXELS_X, WARP_PIXELS_Y, BLOCK_IMAGES, USE_INTERPOLATION, SINGLE_FEATURE, false, PARAMS, __VA_ARGS__) \
	}

#define RUN_KERNEL_R7(CLASS_NAME, IMG_PATCH_SIZE_W, IMG_PATCH_SIZE_H, MAX_OFFSET, WARP_PIXELS_X, WARP_PIXELS_Y, BLOCK_IMAGES, USE_INTERPOLATION, SINGLE_FEATURE, SINGLE_SUBFEATURE, PARAMS,  ...) \
	if (SINGLE_FEATURE) { \
		RUN_KERNEL_R6(CLASS_NAME, IMG_PATCH_SIZE_W, IMG_PATCH_SIZE_H, MAX_OFFSET, WARP_PIXELS_X, WARP_PIXELS_Y, BLOCK_IMAGES, USE_INTERPOLATION, true, SINGLE_SUBFEATURE, PARAMS, __VA_ARGS__) \
	} else { \
		RUN_KERNEL_R6(CLASS_NAME, IMG_PATCH_SIZE_W, IMG_PATCH_SIZE_H, MAX_OFFSET, WARP_PIXELS_X, WARP_PIXELS_Y, BLOCK_IMAGES, USE_INTERPOLATION, false, SINGLE_SUBFEATURE, PARAMS, __VA_ARGS__) \
	}

}
