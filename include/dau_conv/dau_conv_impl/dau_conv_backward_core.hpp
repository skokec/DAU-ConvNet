#include <math_functions.h>  // CUDA's, not dau_conv_impl's, for fabs, signbit

#include <cmath>
#include <device_launch_parameters.h>

#include "dau_conv/dau_conv_impl/dau_conv_backward.hpp"
#include "dau_conv/util/math_functions.hpp"

#include <cub/cub/cub.cuh>

namespace DAUConvNet {

// TODO: using hardcoded warp size may not be portable (should use warpSize) but this way allows compiler optimization and avoids using dynamic memory allocation
#define WARP_SIZE 32

#define MAX(x,y) (x > y ? x : y)
#define MIN(x,y) (x < y ? x : y)

#define OFFSET(l,k,j,i, num_l, num_k, num_j, num_i) ((( (l)*(num_k) + (k)) * (num_j) + (j))*(num_i) + (i) )

#define OFFSET5(m, l,k,j,i, num_m, num_l, num_k, num_j, num_i) ((( ((m)*(num_l) + (l))*(num_k) + (k)) * (num_j) + (j))*(num_i) + (i) )

#define OFFSET6(n, m, l,k,j,i, num_n, num_m, num_l, num_k, num_j, num_i) ((( ((((n) * (num_m)) + (m))*(num_l) + (l))*(num_k) + (k)) * (num_j) + (j))*(num_i) + (i) )

#define OFFSET8(i8, i7, i6, i5, i4, i3, i2, i1, num_i8, num_i7, num_i6, num_i5, num_i4, num_i3, num_i2, num_i1) \
				((( (( ( ((i8) * (num_i7) + i7)* (num_i6)  + (i6)  )*(num_i5)  + (i5)  )   * (num_i4) + (i4))*(num_i3) + (i3)) * (num_i2) + (i2))*(num_i1) + (i1) )

#define CEILING(x,y) (((x) + (y) - 1) / (y))

#define IS_VALID_PIXEL(X,Y,MAX_X,MAX_Y) (X >= 0 && X < MAX_X && Y >= 0 && Y < MAX_Y)

    struct  __builtin_align__(16) ptr4
    {
        float* quad[4];
    };


    template <int _NUM_SM,
            int _Bx, int _By,
            int _BLOCK_FEATURES,
            int _BLOCK_SUBFEATURES,
            int _BATCH_PIXELS_SIZE_X,
            int _BATCH_PIXELS_SIZE_Y,
            int _BATCH_IMAGES,
            int _NUM_K,
            int _BATCH_K_SIZE,
            int _PIXELS_INTERPOLATION_Dx,
            int _PIXELS_INTERPOLATION_Dy,
            int _BATCH_FEATURES_SIZE,
            int _BATCH_COMPUTE_FEATURES_SIZE,
            int _BATCH_MEM_SUBFEATURES_SIZE,
            int _BATCH_GAUSS_SIZE,
            int _IMG_WIDTH, int _IMG_HEIGHT,
            int _MAX_OFFSET>
    class BlockIndexing {
    public:

        enum {
            NUM_SM = _NUM_SM,
            Bx = _Bx,
            By = _By,
            BLOCK_FEATURES = _BLOCK_FEATURES,
            BLOCK_SUBFEATURES = _BLOCK_SUBFEATURES,
            BATCH_PIXELS_SIZE_X = _BATCH_PIXELS_SIZE_X,
            BATCH_PIXELS_SIZE_Y = _BATCH_PIXELS_SIZE_Y,
            BATCH_IMAGES = _BATCH_IMAGES,
            NUM_K = _NUM_K,
            BATCH_K_SIZE = _BATCH_K_SIZE,
            PIXELS_INTERPOLATION_Dx = _PIXELS_INTERPOLATION_Dx,
            PIXELS_INTERPOLATION_Dy = _PIXELS_INTERPOLATION_Dy,
            BATCH_FEATURES_SIZE = _BATCH_FEATURES_SIZE,
            BATCH_COMPUTE_FEATURES_SIZE = _BATCH_COMPUTE_FEATURES_SIZE,
            BATCH_MEM_SUBFEATURES_SIZE = _BATCH_MEM_SUBFEATURES_SIZE,
            BATCH_GAUSS_SIZE = _BATCH_GAUSS_SIZE,
            IMG_WIDTH = _IMG_WIDTH,
            IMG_HEIGHT = _IMG_HEIGHT,
            MAX_OFFSET = _MAX_OFFSET,
            NUM_THREADS = Bx* By * BLOCK_FEATURES,
            NUM_WARPS = Bx*By*BLOCK_FEATURES >= WARP_SIZE ? ((Bx*By*BLOCK_FEATURES) / WARP_SIZE) : 1
        };

        // CPU only functions
        class Launch {
        public:
            dim3 getThreadsPerBlock(int num_images, int num_features, int num_subfeatures, int img_width, int img_height) {
                // number of threads per blocks
                return dim3(Bx * By * BLOCK_FEATURES, 1, 1);
                // only BLOCK_FEATURES are distributed over threads while BLOCK_SUBFEATURES are iterated over inside of each thread
            }
            dim3 getBlocksPerGrid(int num_images, int num_features, int num_subfeatures, int num_gaussian, int img_width, int img_height) {

                checkInputSize(num_features, BLOCK_FEATURES * BATCH_FEATURES_SIZE, "num_features");
                checkInputSize(num_subfeatures, BLOCK_SUBFEATURES * BATCH_MEM_SUBFEATURES_SIZE, "num_subfeatures");
                checkInputSize(img_width, Bx * BATCH_PIXELS_SIZE_X, "img_width", false);
                checkInputSize(img_height, By * BATCH_PIXELS_SIZE_Y, "img_height", false);
                checkInputSize(num_images, BATCH_IMAGES, "num_images", false);
                checkInputSize(num_gaussian, BATCH_GAUSS_SIZE, "num_gaussian");


                int num_feature_blocks = (int)ceil(num_features/(float)(BLOCK_FEATURES * BATCH_FEATURES_SIZE));
                int num_subfeature_blocks = (int)ceil(num_subfeatures/(float)(BLOCK_SUBFEATURES * BATCH_MEM_SUBFEATURES_SIZE));
                int num_gaussian_blocks = (int)ceil(num_gaussian/(float)(BATCH_GAUSS_SIZE));


                int num_pixel_blocks_x = (int)ceil(img_width /  (float)(Bx * BATCH_PIXELS_SIZE_X) );
                int num_pixel_blocks_y = (int)ceil(img_height / (float)(By * BATCH_PIXELS_SIZE_Y) );

                int num_image_blocs = (int)ceil(num_images / (float)(BATCH_IMAGES) );

                // number of blocks per kernel launch
                return dim3 ( num_feature_blocks,
                              num_subfeature_blocks * num_gaussian_blocks,
                              num_pixel_blocks_x * num_pixel_blocks_y * num_image_blocs);
            }
        private:
            void checkInputSize(int input_size, int min_allowed, const std::string& param_name, bool allow_only_multiple_of_min = true) {
                if (input_size < min_allowed) {
                    printf("Invalid %s value of %d. Min allowed %d.\n", param_name.c_str(), input_size, min_allowed);
                    throw std::exception();
                }
                if (allow_only_multiple_of_min && input_size % min_allowed != 0) {
                    printf("Invalid %s value of %d. Only a multiple of %d allowed.\n", param_name.c_str(), input_size, min_allowed);
                    throw std::exception();
                }
            }

        };

        // GPU only functions
        class Kernel {
        public:
            int2 img_size;

            int f_thread_idx;
            int px_thread_idx;

            int img_block_idx;
            int f_block_idx;
            int s_block_idx;
            int g_block_idx;
            int px_block_idx;

            __device__ Kernel(int img_width, int img_height, int num_gaussian) {
                img_size.x = img_width;
                img_size.y = img_height;

                f_thread_idx = threadIdx.x / (Bx * By);
                px_thread_idx = threadIdx.x % (Bx * By);

                f_block_idx = blockIdx.x;

                int num_gaussian_blocks = (int)ceil(num_gaussian/(float)(BATCH_GAUSS_SIZE));

                g_block_idx = blockIdx.y % num_gaussian_blocks;
                s_block_idx = blockIdx.y / num_gaussian_blocks;

                int num_pixel_blocks_x = (int)ceil(img_width /  (float)(Bx * BATCH_PIXELS_SIZE_X) );
                int num_pixel_blocks_y = (int)ceil(img_height / (float)(By * BATCH_PIXELS_SIZE_Y) );

                px_block_idx = blockIdx.z % (num_pixel_blocks_x*num_pixel_blocks_y);
                img_block_idx = blockIdx.z / (num_pixel_blocks_x*num_pixel_blocks_y);
            }
            // return global image index that specific thread handles
            __device__ int getImageIdx() {
                return img_block_idx * BATCH_IMAGES;
            }

            // return global feature index that specific thread handles
            // since each thread handles multiple features (BATCH_FEATURES_SIZE) and each block handles
            // multiple features as well (BLOCK_FEATURES) this returns offset to F that specific thread will use
            __device__ int getFeatureIdx() {
                return f_block_idx * (BLOCK_FEATURES * BATCH_FEATURES_SIZE)  + f_thread_idx * BATCH_FEATURES_SIZE;
            }

            // return local index that specific thread handles
            // since one block handles multiple features (BLOCK_FEATURES) this returns index of feature for within one block
            __device__ int getFeatureBlockIdx() {
                return f_thread_idx * BATCH_FEATURES_SIZE;
            }

            __device__ int getSubfeatureIdx() {
                return s_block_idx * (BLOCK_SUBFEATURES * BATCH_MEM_SUBFEATURES_SIZE)  + 0;
            }

            __device__ int getGaussianIdx() {
                return g_block_idx * BATCH_GAUSS_SIZE + 0;
            }

            __device__ int2 getPosBlockSize() {
                return make_int2(Bx * BATCH_PIXELS_SIZE_X,
                                 By * BATCH_PIXELS_SIZE_Y);
            }

            __device__ int2 getPosBlockIdx() {

                int blockIdx_x = px_block_idx % (img_size.x / (Bx * BATCH_PIXELS_SIZE_X));
                int blockIdx_y = px_block_idx / (img_size.x / (Bx * BATCH_PIXELS_SIZE_X));

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

                int threadIdx_x_inner = threadIdx_x / BATCH_K_SIZE;
                int threadIdx_x_outer = threadIdx_x % BATCH_K_SIZE;


                return make_int2(threadIdx_x * BATCH_K_SIZE,
                                 (threadIdx_y * BATCH_PIXELS_SIZE_X * BATCH_PIXELS_SIZE_Y ) * NUM_K / BATCH_K_SIZE );

            }

            __device__ int getThreadId() {
                return threadIdx.x +
                       threadIdx.y * blockDim.x +
                       threadIdx.z * blockDim.x * blockDim.y;
            }

            static __device__ int getNumWarps() {
                return NUM_WARPS;
            }
            __device__ int getWarpId() {
                return getThreadId() / warpSize;
            }

            static __forceinline__ __device__ unsigned warp_lane_id()
            {
                unsigned ret;
                asm volatile ("mov.u32 %0, %laneid;" : "=r"(ret));
                return ret;
            }

            static __forceinline__ __device__ unsigned warp_id()
            {
                unsigned ret;
                asm volatile ("mov.u32 %0, %warpid;" : "=r"(ret));
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
            NUM_THREADS_WIDTH = MIN(NUM_THREADS,WIDTH_TRANSFER / BATCH_ELEMENTS),
            NUM_THREADS_HEIGHT = MAX(1, NUM_THREADS / NUM_THREADS_WIDTH),

            // number of iterations that will be performed during load/store by one thread
            NUM_ITERATION_Y = MAX(1,CEILING((HEIGHT + 2*APRON_SIZE_Y), NUM_THREADS_HEIGHT)),
            NUM_ITERATION_X = MAX(1, CEILING(WIDTH + 2*APRON_SIZE_X,NUM_THREADS_WIDTH * BATCH_ELEMENTS))

        };

    private:

        typedef BlockSharedMemory<NUM_THREADS, WIDTH, HEIGHT, APRON_SIZE_X, APRON_SIZE_Y, NUM_BUFFER_REPEAT, ELEMENT_TYPE, BATCH_ELEMENTS> BlockSharedMemoryT;

        struct _Data {
            ELEMENT_TYPE data[NUM_BUFFER_REPEAT][ALLOC_HEIGHT][ALLOC_WIDTH];
        };

        struct _LoadingData {
            ELEMENT_TYPE data[NUM_ITERATION_Y][NUM_ITERATION_X];
        };

        float* storage_data_for_writing;
        float* storage_data_for_reading;


        _Data& storage;

        // thread indexing for storing/writing data from global mem
        int2 thread_indexing_writing;

        // thread indexing for reading data by each thread (MUST be user defined in constructor)
        int2 thread_indexing_reading;

    public:
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

#define ASM_CASES(ASM_CMD, INDEX, ...) \
			if (INDEX == 0) ASM_CMD(1,__VA_ARGS__); \
			 if (INDEX == 1) ASM_CMD(2,__VA_ARGS__); \
			 if (INDEX == 2) ASM_CMD(3,__VA_ARGS__); \
			 if (INDEX == 3) ASM_CMD(4,__VA_ARGS__); \
			 if (INDEX == 4) ASM_CMD(5,__VA_ARGS__); \
			 if (INDEX == 5) ASM_CMD(6,__VA_ARGS__); \
			 if (INDEX == 6) ASM_CMD(7,__VA_ARGS__); \
			 if (INDEX == 7) ASM_CMD(8,__VA_ARGS__); \
			 if (INDEX == 9) ASM_CMD(9,__VA_ARGS__); \
			 if (INDEX == 10) ASM_CMD(10,__VA_ARGS__); \
			 if (INDEX == 11) ASM_CMD(11,__VA_ARGS__); \
			 if (INDEX == 12) ASM_CMD(12,__VA_ARGS__); \
			 if (INDEX == 13) ASM_CMD(13,__VA_ARGS__); \
			 if (INDEX == 14) ASM_CMD(14,__VA_ARGS__); \
			 if (INDEX == 15) ASM_CMD(15,__VA_ARGS__); \
			 if (INDEX == 16) ASM_CMD(16,__VA_ARGS__); \
			 if (INDEX == 17) ASM_CMD(17,__VA_ARGS__); \
			 if (INDEX == 19) ASM_CMD(18,__VA_ARGS__); \
			 if (INDEX == 20) ASM_CMD(19,__VA_ARGS__);

        __device__
        void define_custom_reg_and_pred() {

            // defines registers and predicates for each iteration that loading/storing is done
            // using ASM_CASES this works only up to NUM_ITERATION_Y * NUM_ITERATION_X < 10 at the moment !!
#pragma unroll
            for (int index = 0; index < NUM_ITERATION_Y * NUM_ITERATION_X; ++index) {

#define ASM_REG_AND_PRED_V2(IND,NONE) \
            	asm(".reg .v2 .f32 t" #IND ";\n\t" ".reg .pred p" #IND ";\n\t")

#define ASM_REG_AND_PRED_V4(IND,NONE) \
            	asm(".reg .v4 .f32 t" #IND ";\n\t" ".reg .pred p" #IND ";\n\t")

                if (BATCH_ELEMENTS == 2) { ASM_CASES(ASM_REG_AND_PRED_V2,index) }
                if (BATCH_ELEMENTS == 4) { ASM_CASES(ASM_REG_AND_PRED_V4,index) }
            }

        }
        template <unsigned int _GLOBAL_DATA_WIDTH, int REPLICATE_OFFSETED, bool USE_FILL, int FILL_VALUE> // GLOBAL_WIDTH .. size of image row in ELEMENT_TYPE elements i.e. if ELEMENT_TYPE == float4 then GLOBAL_WIDTH counts 4 floats as one
        __device__
        void load_global_asm(const ELEMENT_TYPE* global_data_, ELEMENT_TYPE* shared_data, int GLOBAL_DATA_WIDTH = -1) {

            // version of load_global with inlined asm PTX commands
            // this prevents compiler from generating additional MOV after LDG which can cause significant memory dependency delays

            const char*  global_data = (const char*)(global_data_  + (-APRON_SIZE_Y) * GLOBAL_DATA_WIDTH / BATCH_ELEMENTS + (-APRON_SIZE_X) / BATCH_ELEMENTS);

            int index = 0;
#pragma unroll
            for (unsigned int j = 0; j < HEIGHT + 2*APRON_SIZE_Y; j+=NUM_THREADS_HEIGHT) {
#pragma unroll
                for (unsigned int i = 0; i < WIDTH + 2*APRON_SIZE_X; i+=NUM_THREADS_WIDTH * BATCH_ELEMENTS) {
                    // current_image already at position for this block

                    // avoid using IF guards for reading to prevent from using MOV operations
                    // we can read global data outside of "bounds" but require global data to have additional extra buffer (as if having one additional sample)

                    //if (thread_indexing_writing.x < (WIDTH + 2*APRON_SIZE_X - i)  && thread_indexing_writing.y < HEIGHT + 2*APRON_SIZE_Y - j)
                    {
                        unsigned int read_offset = ((j) * _GLOBAL_DATA_WIDTH / BATCH_ELEMENTS + (i) / BATCH_ELEMENTS) * sizeof(ELEMENT_TYPE);

#define ASM_LD_GLOBAL_IND_V2(IND, DATA_ADDR) asm(" ld.global.nc.v2.f32 t" #IND ", [%0];\n\t" : : "l"( DATA_ADDR) : "memory");
#define ASM_LD_GLOBAL_IND_V4(IND, DATA_ADDR) asm(" ld.global.nc.v4.f32 t" #IND ", [%0];\n\t" : : "l"( DATA_ADDR) : "memory");

                        if (BATCH_ELEMENTS == 2) { ASM_CASES(ASM_LD_GLOBAL_IND_V2, index, global_data + read_offset) }
                        if (BATCH_ELEMENTS == 4) { ASM_CASES(ASM_LD_GLOBAL_IND_V4, index, global_data + read_offset) }
                    }
                    index++;
                }
            }
        }
        template <int REPLICATE_OFFSETED>
        __device__
        void store_shared_asm(ELEMENT_TYPE* shared_data) {
            // make sure address will be to shared data !!
            char* shared_base_addr;
            asm("cvta.to.shared.u64 %0, %1;" : "=l"(shared_base_addr) : "l"(shared_data): "memory" );

            int index = 0;
#pragma unroll
            for (unsigned int j = 0; j < HEIGHT + 2*APRON_SIZE_Y; j+=NUM_THREADS_HEIGHT) {
#pragma unroll
                for (unsigned int i = 0; i < WIDTH + 2*APRON_SIZE_X; i+=NUM_THREADS_WIDTH * BATCH_ELEMENTS) {
                    // current_image already at position for this block

                    if (thread_indexing_writing.x < (WIDTH + 2*APRON_SIZE_X - i)  && thread_indexing_writing.y < HEIGHT + 2*APRON_SIZE_Y - j)
                    {
                        unsigned int write_offset = ((j ) * ALLOC_WIDTH  + (i) / BATCH_ELEMENTS) * sizeof(ELEMENT_TYPE);

                        // use C/C++ if guards instead of PTX predicates since this appears to be faster
                        //#define ASM_ST_SHARED_IND_v2(IND, DATA_ADDR) asm("@p" #IND " st.shared.v2.f32 [%0], t" #IND ";\n\t" : : "l"( DATA_ADDR) : "memory");
                        //#define ASM_ST_SHARED_IND_v4(IND, DATA_ADDR) asm("@p" #IND " st.shared.v4.f32 [%0], t" #IND ";\n\t" : : "l"( DATA_ADDR) : "memory");
#define ASM_ST_SHARED_IND_V2(IND, DATA_ADDR) asm(" st.shared.v2.f32 [%0], t" #IND ";\n\t" : : "l"( DATA_ADDR) : "memory");
#define ASM_ST_SHARED_IND_V4(IND, DATA_ADDR) asm(" st.shared.v4.f32 [%0], t" #IND ";\n\t" : : "l"( DATA_ADDR) : "memory");

                        if (BATCH_ELEMENTS == 2) { ASM_CASES(ASM_ST_SHARED_IND_V2, index, shared_base_addr + write_offset) }
                        if (BATCH_ELEMENTS == 4) { ASM_CASES(ASM_ST_SHARED_IND_V4, index, shared_base_addr + write_offset) }

                    }
                    index++;
                }
            }
            //print();
        }

        __device__
        void print() {
            __syncthreads();
            if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0 && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0)
            {

                printf("printing shared memory:\n");
                printf("using following settings:\n");
                printf("WIDTH_TRANSFER: %d\n",WIDTH_TRANSFER);
                printf("NUM_THREADS_WIDTH: %d\n",NUM_THREADS_WIDTH);
                printf("NUM_THREADS_HEIGHT: %d\n",NUM_THREADS_HEIGHT);
                printf("NUM_ITERATION_Y: %d\n",NUM_ITERATION_Y);
                printf("NUM_ITERATION_X: %d\n",NUM_ITERATION_X);

                for (int s = 0; s < NUM_BUFFER_REPEAT; ++s) {
                    for (int j = 0; j < ALLOC_HEIGHT; ++j){
                        for (int i = 0; i < ALLOC_WIDTH; ++i){
                            ELEMENT_TYPE tmp = storage.data[s][j][i];
                            if (BATCH_ELEMENTS == 4) printf("%f %f %f %f ", (float)tmp.x, (float)tmp.y, (float)tmp.z, (float)tmp.w);
                            if (BATCH_ELEMENTS == 2) printf("%f %f ", (float)tmp.x, (float)tmp.y);
                        }
                        printf("\n");
                    }
                    printf("\nend of NUM_BUFFER_REPEAT %d\n",s);
                }
                printf("\nend of double buffer\n");
            }
            __syncthreads();

        }

        __device__
        void print_int() {
            __syncthreads();
            if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0 /*&& blockIdx.x == 0 && blockIdx.y && blockIdx.z == 0*/)
            {

                printf("printing shared memory:\n");

                for (int s = 0; s < NUM_BUFFER_REPEAT; ++s) {
                    for (int j = 0; j < ALLOC_HEIGHT; ++j){
                        for (int i = 0; i < ALLOC_WIDTH; ++i){
                            ELEMENT_TYPE tmp = storage.data[s][j][i];
                            if (BATCH_ELEMENTS == 4) printf("%d %d %d %d ", (int)tmp.x, (int)tmp.y, (int)tmp.z, (int)tmp.w);
                            if (BATCH_ELEMENTS == 2) printf("%d %d ", (int)tmp.x, (int)tmp.y);
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
            int NUM_K,
            int BATCH_K_SIZE,
            int BATCH_FEATURES_SIZE,
            int BATCH_COMPUTE_FEATURES_SIZE,
            int BATCH_MEM_SUBFEATURES_SIZE,
            int BLOCK_FEATURES,
            //int IMG_WIDTH, int IMG_HEIGHT,
            int BATCH_PIXELS_FLOAT4,
            typename  _BlockSharedMemoryT,
            int GLOBAL_IMG_READ_WIDTH,
            int NUM_REPLICATE_OFFSETED,
            bool LOAD_SHARED_DATA_WITH_ASM>
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
            typename BlockSharedMemoryT::LoadingData ld;

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
            float* address;
            float4* output;
        } load_weights;

        // load data
        struct {
            bool enabled;
            ptr4* address;
            float4* output;
        } load_data;

        // compute
        struct {
            bool enabled;
            float4* weights;
            float4* errors;
            float4* data;
            float4* output;
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

#define COPY_VECTOR4(Y,X) \
{ \
	(Y).x = (X).x; \
	(Y).y = (X).y; \
	(Y).z = (X).z; \
	(Y).w = (X).w; \
}

#define COPY_VECTOR2(Y,X) \
{ \
	(Y).x = (X).x; \
	(Y).y = (X).y; \
}

        __device__
        bool should_run(int current_index, int unit_start_delay, int max_iter) {
            return (current_index - unit_start_delay >= 0 && current_index - unit_start_delay < max_iter  ? true : false );
        }


        __device__
        void execute_step() {

            if (load_global.enabled) {
                // default version with load/store
                if (0)
                    shared_mem.template load_global<GLOBAL_IMG_READ_WIDTH,NUM_REPLICATE_OFFSETED,false,1>(load_global.reading_ptr,
                                                                                                          load_global.writing_ptr,
                                                                                                          load_global.img_read_width);
                if (1)
                    // version with load call only and a separate store call later
                    if (LOAD_SHARED_DATA_WITH_ASM == false)
                        shared_mem.template load_global<GLOBAL_IMG_READ_WIDTH,NUM_REPLICATE_OFFSETED,false,1>(load_global.reading_ptr,
                                                                                                              load_global.writing_ptr,
                                                                                                              load_global.img_read_width, &load_global.ld);
                    else
                        // PTX asm version with load call only and a separate store call later
                        shared_mem.template load_global_asm<GLOBAL_IMG_READ_WIDTH,NUM_REPLICATE_OFFSETED,false,1>(load_global.reading_ptr,
                                                                                                                  load_global.writing_ptr,
                                                                                                                  load_global.img_read_width);
            }
            static const int NUM_READ_FEATURES =  BATCH_COMPUTE_FEATURES_SIZE >= 4 ? 4 :
                                                  (BATCH_COMPUTE_FEATURES_SIZE >= 2 ? 2 : 1);

            // load quad of w for next one
            if (load_weights.enabled) {
                // p.next_w = *(float4)p.next_w_address
                for (int i = 0; i < PIXELS_INTERPOLATION_SIZE; ++i) {
                    for (int f_quad_index = 0; f_quad_index < BATCH_COMPUTE_FEATURES_SIZE/NUM_READ_FEATURES; f_quad_index++) {
                        {
                            if (BATCH_COMPUTE_FEATURES_SIZE > 0) load_weights.output[(i* BATCH_COMPUTE_FEATURES_SIZE/NUM_READ_FEATURES + f_quad_index)].x = load_weights.address[(i * BATCH_FEATURES_SIZE/NUM_READ_FEATURES + f_quad_index) * BLOCK_FEATURES * NUM_READ_FEATURES + 0];
                            if (BATCH_COMPUTE_FEATURES_SIZE > 1) load_weights.output[(i* BATCH_COMPUTE_FEATURES_SIZE/NUM_READ_FEATURES + f_quad_index)].y = load_weights.address[(i * BATCH_FEATURES_SIZE/NUM_READ_FEATURES + f_quad_index) * BLOCK_FEATURES * NUM_READ_FEATURES + 1];
                            if (BATCH_COMPUTE_FEATURES_SIZE > 2) load_weights.output[(i* BATCH_COMPUTE_FEATURES_SIZE/NUM_READ_FEATURES + f_quad_index)].z = load_weights.address[(i * BATCH_FEATURES_SIZE/NUM_READ_FEATURES + f_quad_index) * BLOCK_FEATURES * NUM_READ_FEATURES + 2];
                            if (BATCH_COMPUTE_FEATURES_SIZE > 3) load_weights.output[(i* BATCH_COMPUTE_FEATURES_SIZE/NUM_READ_FEATURES + f_quad_index)].w = load_weights.address[(i * BATCH_FEATURES_SIZE/NUM_READ_FEATURES + f_quad_index) * BLOCK_FEATURES * NUM_READ_FEATURES + 3];
                        } //COPY_VECTOR4(); // weights for F[0], F[1], F[2], F[3]
                    }
                }
            }

            // load quad of offsets for next one and make it directly into pointer to data
            if (load_offset.enabled) {
                for (int f_quad_index = 0; f_quad_index < MAX(1,BATCH_COMPUTE_FEATURES_SIZE/NUM_READ_FEATURES); ++f_quad_index ) {
                    if (BATCH_COMPUTE_FEATURES_SIZE > 0) load_offset.output[f_quad_index].quad[0] = (float*)((char*)load_offset.base_address + load_offset.offset_address[f_quad_index * BLOCK_FEATURES * NUM_READ_FEATURES + 0]); // F[0]
                    if (BATCH_COMPUTE_FEATURES_SIZE > 1) load_offset.output[f_quad_index].quad[1] = (float*)((char*)load_offset.base_address + load_offset.offset_address[f_quad_index * BLOCK_FEATURES * NUM_READ_FEATURES + 1]); // F[1]
                    if (BATCH_COMPUTE_FEATURES_SIZE > 2) load_offset.output[f_quad_index].quad[2] = (float*)((char*)load_offset.base_address + load_offset.offset_address[f_quad_index * BLOCK_FEATURES * NUM_READ_FEATURES + 2]); // F[2]
                    if (BATCH_COMPUTE_FEATURES_SIZE > 3) load_offset.output[f_quad_index].quad[3] = (float*)((char*)load_offset.base_address + load_offset.offset_address[f_quad_index * BLOCK_FEATURES * NUM_READ_FEATURES + 3]); // F[3]

                    /*if (f_quad_index == 0 && f_index == 0 && thread_x == 0 && thread_y == 0) {
                        printf("reading offset from addr %p with value %d for s,g,f=(%d,%d,%d)\n", &load_offset.offset_address[f_quad_index * BLOCK_FEATURES * NUM_READ_FEATURES + 0], load_offset.offset_address[f_quad_index * BLOCK_FEATURES * NUM_READ_FEATURES + 0], s_index, g_index, f_index+0);
                        printf("reading offset from addr %p with value %d for s,g,f=(%d,%d,%d)\n", &load_offset.offset_address[f_quad_index * BLOCK_FEATURES * NUM_READ_FEATURES + 1], load_offset.offset_address[f_quad_index * BLOCK_FEATURES * NUM_READ_FEATURES + 1], s_index, g_index, f_index+1);
                    }*/

                }
            }

            float weighted_errors[BATCH_COMPUTE_FEATURES_SIZE][(BATCH_PIXELS_SIZE_X * BATCH_PIXELS_SIZE_Y)/BATCH_PIXELS_FLOAT4];

            for (int f = 0; f < BATCH_COMPUTE_FEATURES_SIZE; ++f) {
                for (int px = 0; px < (BATCH_PIXELS_SIZE_X * BATCH_PIXELS_SIZE_Y)/BATCH_PIXELS_FLOAT4; ++px) {
                    weighted_errors[f][px] = 0;
                }
            }

            {


                NDIndexing<BATCH_COMPUTE_FEATURES_SIZE/NUM_READ_FEATURES,
                        NDIndexing<(BATCH_PIXELS_SIZE_X * BATCH_PIXELS_SIZE_Y)/BATCH_PIXELS_FLOAT4,
                        NDIndexing<PIXELS_INTERPOLATION_Dy,
                        NDIndexingZero<PIXELS_INTERPOLATION_Dx> > > > indexing;

                static const int NUM_ITER = BATCH_COMPUTE_FEATURES_SIZE/NUM_READ_FEATURES * PIXELS_INTERPOLATION_Dy * PIXELS_INTERPOLATION_Dx * (BATCH_PIXELS_SIZE_X * BATCH_PIXELS_SIZE_Y) /BATCH_PIXELS_FLOAT4;

#pragma unroll
                for (int i = 0; i < NUM_ITER; ++i) {

                    int f_quad_index = indexing.getIndex<0>(i);
                    int px = indexing.getIndex<1>(i);
                    int interpolation_j = indexing.getIndex<2>(i);
                    int interpolation_i = indexing.getIndex<3>(i);

                    // get raw index based on how data is stored (not based on iteration order over it !!)
                    int4 error_in_index;

                    error_in_index.x = OFFSET(NUM_READ_FEATURES*f_quad_index + 0, PIXELS_INTERPOLATION_Dy - 1 - interpolation_j, PIXELS_INTERPOLATION_Dx - 1 - interpolation_i, px, BATCH_FEATURES_SIZE, PIXELS_INTERPOLATION_Dy, PIXELS_INTERPOLATION_Dx,(BATCH_PIXELS_SIZE_X * BATCH_PIXELS_SIZE_Y)/BATCH_PIXELS_FLOAT4);
                    if (NUM_READ_FEATURES > 1) error_in_index.y = error_in_index.x + 1 * PIXELS_INTERPOLATION_Dy * PIXELS_INTERPOLATION_Dx * (BATCH_PIXELS_SIZE_X * BATCH_PIXELS_SIZE_Y)/BATCH_PIXELS_FLOAT4;
                    if (NUM_READ_FEATURES > 2) error_in_index.z = error_in_index.x + 2 * PIXELS_INTERPOLATION_Dy * PIXELS_INTERPOLATION_Dx * (BATCH_PIXELS_SIZE_X * BATCH_PIXELS_SIZE_Y)/BATCH_PIXELS_FLOAT4;
                    if (NUM_READ_FEATURES > 3) error_in_index.w = error_in_index.x + 3 * PIXELS_INTERPOLATION_Dy * PIXELS_INTERPOLATION_Dx * (BATCH_PIXELS_SIZE_X * BATCH_PIXELS_SIZE_Y)/BATCH_PIXELS_FLOAT4;

                    int weight_in_index = OFFSET(0, interpolation_j, interpolation_i, f_quad_index,
                                                 1, PIXELS_INTERPOLATION_Dx, PIXELS_INTERPOLATION_Dy, BATCH_COMPUTE_FEATURES_SIZE/NUM_READ_FEATURES);

                    //if (interpolation_j == 0 && interpolation_i == 0)
                    {
                        if (NUM_READ_FEATURES > 0) weighted_errors[NUM_READ_FEATURES*f_quad_index + 0][px] += (PIXELS_INTERPOLATION_SIZE > 1 ? compute.weights[weight_in_index].x : 1) * compute.errors[error_in_index.x].x;
                        if (NUM_READ_FEATURES > 1) weighted_errors[NUM_READ_FEATURES*f_quad_index + 1][px] += (PIXELS_INTERPOLATION_SIZE > 1 ? compute.weights[weight_in_index].y : 1) * compute.errors[error_in_index.y].x;
                        if (NUM_READ_FEATURES > 2) weighted_errors[NUM_READ_FEATURES*f_quad_index + 2][px] += (PIXELS_INTERPOLATION_SIZE > 1 ? compute.weights[weight_in_index].z : 1) * compute.errors[error_in_index.z].x;
                        if (NUM_READ_FEATURES > 3) weighted_errors[NUM_READ_FEATURES*f_quad_index + 3][px] += (PIXELS_INTERPOLATION_SIZE > 1 ? compute.weights[weight_in_index].w : 1) * compute.errors[error_in_index.w].x;
                    }
                    /*if (f_quad_index == 0 && s_index == 0 && f_index == 0 && thread_x == 0 && thread_y == 0) {
                        printf("weighted_errors value from error %f * weight %f = weighted_errors %f at position j,i=%d,%d and interpolation dy,dx=%d,%d\n", compute.errors[error_in_index.x].x, compute.weights[weight_in_index].x,weighted_errors[f_quad_index + 0][px] , px / BATCH_PIXELS_SIZE_X, px % BATCH_PIXELS_SIZE_X, PIXELS_INTERPOLATION_Dy - 1 - interpolation_j, PIXELS_INTERPOLATION_Dx - 1 - interpolation_i);
                    }*/
                }
            }

            NDIndexing<BATCH_COMPUTE_FEATURES_SIZE,
                    NDIndexing<(BATCH_PIXELS_SIZE_X * BATCH_PIXELS_SIZE_Y)/BATCH_PIXELS_FLOAT4,
                    NDIndexingZero<NUM_K> >  > indexing;

            static const int NUM_ITER = BATCH_COMPUTE_FEATURES_SIZE * (BATCH_PIXELS_SIZE_X * BATCH_PIXELS_SIZE_Y) /BATCH_PIXELS_FLOAT4 * (NUM_K);

#pragma unroll
            for (int i = 0; i < NUM_ITER; ++i) {

                // i goes over [BATCH_PIXELS_SIZE_/4][PIXELS_INTERPOLATION_SIZE][BATCH_COMPUTE_FEATURES_SIZE][NUM_K] array so get indexes for them manually

                int f = indexing.getIndex<0>(i);
                int px = indexing.getIndex<1>(i);
                int k = indexing.getIndex<2>(i);

                // since we store weight and offset into float4/int4 we need a proper index to access array of quad vectors
                int f_quad_index = f/NUM_READ_FEATURES;

                int k_inner = k % BATCH_K_SIZE;
                int k_outer = k / BATCH_K_SIZE;

                // get raw index based on how data is stored (not based on iteration order over it !!)
                //int error_in_index = OFFSET(f, interpolation_j, interpolation_i, px,
                //                             BATCH_FEATURES_SIZE, PIXELS_INTERPOLATION_Dx, PIXELS_INTERPOLATION_Dy,(BATCH_PIXELS_SIZE_X * BATCH_PIXELS_SIZE_Y)/BATCH_PIXELS_FLOAT4);

                int data_in_index = OFFSET(0, f, k, px,
                                           0, BATCH_FEATURES_SIZE, NUM_K,(BATCH_PIXELS_SIZE_X * BATCH_PIXELS_SIZE_Y)/BATCH_PIXELS_FLOAT4);

                int data_out_index = f_quad_index * NUM_K + k;

                int load_offset = k_inner +
                                  k_outer * BlockSharedMemoryT::PITCHED_WIDTH +
                                  px * (NUM_K / BATCH_K_SIZE) * BlockSharedMemoryT::PITCHED_WIDTH;

                // load data for next loop
                if (load_data.enabled) {

                    int data_address_index = f_quad_index;
                    int data_quad_index = f % NUM_READ_FEATURES;

                    if (BATCH_K_SIZE % 4 == 0) {

                        float4 tmp = reinterpret_cast<float4*>(load_data.address[data_address_index].quad[data_quad_index] + load_offset)[0];

                        if (load_offset % 4 == 0)
                            load_data.output[data_in_index].x = tmp.x;
                        else if (load_offset % 4 == 1)
                            load_data.output[data_in_index].x = tmp.x;
                        else if (load_offset % 4 == 2)
                            load_data.output[data_in_index].x = tmp.x;
                        else if (load_offset % 4 == 3)
                            load_data.output[data_in_index].x = tmp.x;

                    } else if (BATCH_K_SIZE % 2 == 0) {
                        float2 tmp = reinterpret_cast<float2*>(load_data.address[data_address_index].quad[data_quad_index] + load_offset)[0];

                        load_data.output[data_in_index].x = (load_offset % 2 == 0) ? tmp.x : tmp.x;

                    } else {
                        load_data.output[data_in_index].x = reinterpret_cast<float*>(load_data.address[data_address_index].quad[data_quad_index] + load_offset)[0];
                    }
                }

                // compute for current loop
                if (compute.enabled) {

                    float computed_value = weighted_errors[f][px] * compute.data[data_in_index].x;
                    //float computed_value = compute.errors[error_in_index].x * compute.data[data_in_index].x;
                    /*if (BATCH_PIXELS_FLOAT4 > 1) computed_value += compute.errors[error_in_index].y * compute.data[data_in_index].y;
                    if (BATCH_PIXELS_FLOAT4 > 2) computed_value += compute.errors[error_in_index].z * compute.data[data_in_index].z;
                    if (BATCH_PIXELS_FLOAT4 > 3) computed_value += compute.errors[error_in_index].w * compute.data[data_in_index].w;*/

                    if (f % NUM_READ_FEATURES == 0)
                        compute.output[data_out_index].x += computed_value;
                    else if (f % NUM_READ_FEATURES == 1)
                        compute.output[data_out_index].y += computed_value;
                    else if (f % NUM_READ_FEATURES == 2)
                        compute.output[data_out_index].z += computed_value;
                    else if (f % NUM_READ_FEATURES == 3)
                        compute.output[data_out_index].w += computed_value;

                    /*if (f + f_index == 15 && thread_x == 0 && thread_y == 0 &&  k == 0 && block_x == 0 && block_y == 0 && g_index == 0 && px == 0) {
                        float vv = -1;
                        if (f % NUM_READ_FEATURES == 0)
                            vv = compute.output[data_out_index].x;
                        else if (f % NUM_READ_FEATURES == 1)
                            vv = compute.output[data_out_index].y;
                        else if (f % NUM_READ_FEATURES == 2)
                            vv = compute.output[data_out_index].z;
                        else if (f % NUM_READ_FEATURES == 3)
                            vv = compute.output[data_out_index].w;
                        printf("computed sum %f from current value from error %f * data %f = computed_value %f at position j,i=%d,%d and k=%d, block y,x=%d,%d and image index:%d and s=%d\n",vv , weighted_errors[f][px], compute.data[data_in_index].x, computed_value, px / BATCH_PIXELS_SIZE_X, px % BATCH_PIXELS_SIZE_X, k, block_y, block_x,image_index, s_index);
                    }*/

                    /*if (g_index == 0 && (f_index +f== 0 || f_index +f == 0 || f_index +f== 0) && s_index == 0 && k == 0)
                    {

                        int px_x = block_x + thread_x + px % BATCH_PIXELS_SIZE_X;
                        int px_y = block_y + thread_y + px / BATCH_PIXELS_SIZE_X;

                        printf("a(%d,%d,%d,%d,%d,%d) = %f;\n",image_index+1, s_index+1, g_index+1, f_index+f+1,  px_y+1, px_x+1, weighted_errors[f][px]);
                        //printf("a(%d,%d,%d,%d,%d,%d) = %f;\n",image_index+1, s_index+1, g_index+1, f_index+f+1,  px_y+1, px_x+1, compute.data[data_in_index].x);
                        //printf("a(%d,%d,%d,%d,%d,%d) = %f;\n",image_index+1, s_index+1, g_index+1, f_index+f+1,  px_y+1, px_x+1, computed_value);
                    }*/
                }
                if (1)
                    // we load somewhere halfway through processing to buffer with additional time needed to load from global memory (usually takes 300 cycles if not in L2 cache)
                    if (load_global.enabled && i == NUM_ITER-1) {
                        // separate store call
                        if (LOAD_SHARED_DATA_WITH_ASM == false)
                            shared_mem.template store_shared<NUM_REPLICATE_OFFSETED>(load_global.ld,load_global.writing_ptr);
                        else
                            // PTX asm version
                            shared_mem.template store_shared_asm<NUM_REPLICATE_OFFSETED>(load_global.writing_ptr);
                    }
            }
        }
    };



    template <typename BlockIndexingT>
    __global__ void //__launch_bounds__(128, 3)
    DAUConv_bwd_multi_pipeline_kernel(const float *filtered_images, const float *error_images,
                                      const int *filter_offsets, const float *filter_weights, float *output,
                                      const int I, const int S, const int F, const int G, const int K,
                                      const int img_width_, const int img_height_) {

// INPUT: filtered images  	[I x S x H x W]
//        error images  	[I x S x H x W]
//		  filter offsets   	[F / (BATCH_FEATURES_SIZE * BLOCK_FEATURES)] x [S / (BATCH_MEM_SUBFEATURES_SIZE * BLOCK_SUBFEATURES)] x [G / BATCH_GAUSS_SIZE]
// 				 	            x [ BATCH_MEM_SUBFEATURES_SIZE * BLOCK_SUBFEATURES] x [BATCH_GAUSS_SIZE] x [BATCH_FEATURES_SIZE/4] x [BLOCK_FEATURES];
// OUTPUT output  		 	[S x G x F]

#ifndef CUBIN_EMBEDDING


        typedef class BlockIndexingT::Kernel BlockIndexingKernel;

        static const int NUM_SM = BlockIndexingT::NUM_SM;
        static const int Bx = BlockIndexingT::Bx;
        static const int By = BlockIndexingT::By;
        static const int BLOCK_FEATURES = BlockIndexingT::BLOCK_FEATURES;
        static const int BLOCK_SUBFEATURES = BlockIndexingT::BLOCK_SUBFEATURES;
        static const int BATCH_PIXELS_SIZE_X = BlockIndexingT::BATCH_PIXELS_SIZE_X;
        static const int BATCH_PIXELS_SIZE_Y = BlockIndexingT::BATCH_PIXELS_SIZE_Y;
        static const int PIXELS_INTERPOLATION_Dx = BlockIndexingT::PIXELS_INTERPOLATION_Dx;
        static const int PIXELS_INTERPOLATION_Dy = BlockIndexingT::PIXELS_INTERPOLATION_Dy;
        static const int BATCH_FEATURES_SIZE = BlockIndexingT::BATCH_FEATURES_SIZE;
        static const int BATCH_COMPUTE_FEATURES_SIZE = BlockIndexingT::BATCH_COMPUTE_FEATURES_SIZE;
        static const int BATCH_MEM_SUBFEATURES_SIZE = BlockIndexingT::BATCH_MEM_SUBFEATURES_SIZE;
        static const int BATCH_GAUSS_SIZE = BlockIndexingT::BATCH_GAUSS_SIZE;
        static const int BATCH_IMAGES = BlockIndexingT::BATCH_IMAGES;
        static const int IMG_WIDTH = BlockIndexingT::IMG_WIDTH;
        static const int IMG_HEIGHT = BlockIndexingT::IMG_HEIGHT; // may not be needed
        static const int MAX_OFFSET = BlockIndexingT::MAX_OFFSET;

        static const int NUM_THREADS = BlockIndexingT::NUM_THREADS;

        static const int NUM_K = BlockIndexingT::NUM_K;
        static const int BATCH_K_SIZE = BlockIndexingT::BATCH_K_SIZE;


        // using float4 to load so use
        static const int LOAD_SHARED_DATA_BATCH_SIZE = 4;
        static const int LOAD_SHARED_OFFSET_BATCH_SIZE = 4;

        static const bool LOAD_SHARED_DATA_WITH_ASM = true;

        // using float4 for computing pixels
        static const int BATCH_PIXELS_FLOAT4 = 1;

        static const int DOUBLE_BUFFERING = 2;

        static const int NUM_REPLICATE_OFFSETED = 0;

        static const int PIXELS_INTERPOLATION_SIZE = PIXELS_INTERPOLATION_Dx * PIXELS_INTERPOLATION_Dy;

        const int img_width =  IMG_WIDTH; // img_width_; //
        const int img_height = IMG_HEIGHT; // img_height_; //

        const int error_width = IMG_WIDTH + PIXELS_INTERPOLATION_Dx -1;
        const int error_height = IMG_HEIGHT + PIXELS_INTERPOLATION_Dy -1;

        BlockIndexingKernel block_indexing(img_width, img_height, G);

        int n_offset = block_indexing.getImageIdx();

        int f_offset = block_indexing.getFeatureIdx();

        int f_block_idx = block_indexing.getFeatureBlockIdx();

        int s_offset = block_indexing.getSubfeatureIdx();

        int g_offset = block_indexing.getGaussianIdx();

        int block_width = block_indexing.getPosBlockSize().x;
        int block_height = block_indexing.getPosBlockSize().y;

        int block_x = block_indexing.getPosBlockIdx().x;
        int block_y = block_indexing.getPosBlockIdx().y;

        int thread_x = block_indexing.getPosThreadIdx().x;
        int thread_y = block_indexing.getPosThreadIdx().y;

        // number of features that a single read/compute handles
        static const int NUM_READ_FEATURES =  BATCH_COMPUTE_FEATURES_SIZE >= 4 ? 4 :
                                              (BATCH_COMPUTE_FEATURES_SIZE >= 2 ? 2 : 1);

        int G_MEM_SIZE = G / BATCH_GAUSS_SIZE;
        int S_MEM_SIZE = S / ( BATCH_MEM_SUBFEATURES_SIZE * BLOCK_SUBFEATURES);
        int F_MEM_SIZE = F / (BATCH_FEATURES_SIZE * BLOCK_FEATURES);

        static const int OFFSET_BLOCK_MEM_SIZE = BATCH_MEM_SUBFEATURES_SIZE * BLOCK_SUBFEATURES * BATCH_GAUSS_SIZE * BATCH_FEATURES_SIZE *  BLOCK_FEATURES;
        static const int WEIGHTS_BLOCK_MEM_SIZE = BATCH_MEM_SUBFEATURES_SIZE * BLOCK_SUBFEATURES * BATCH_GAUSS_SIZE * PIXELS_INTERPOLATION_SIZE * BATCH_FEATURES_SIZE *  BLOCK_FEATURES;

        typedef BlockSharedMemory<NUM_THREADS,
                Bx * BATCH_K_SIZE,
                By * BATCH_PIXELS_SIZE_Y * BATCH_PIXELS_SIZE_X * (NUM_K / BATCH_K_SIZE),
                MAX_OFFSET * BATCH_K_SIZE,
                MAX_OFFSET * BATCH_PIXELS_SIZE_X * (NUM_K / BATCH_K_SIZE),
                (NUM_REPLICATE_OFFSETED+1) * DOUBLE_BUFFERING * BATCH_MEM_SUBFEATURES_SIZE,
                float4,
                LOAD_SHARED_DATA_BATCH_SIZE> SharedMem;

        __shared__ typename SharedMem::Data data;


        // get interlevaed indexes that are prepared for shared memory layout
        // i.e., we consider memory arranged as [By * BATCH_PIXELS_SIZE_Y * BATCH_PIXELS_SIZE_X * (NUM_K / BATCH_K_SIZE) ] x [ Bx * BATCH_K_SIZE]
        // and  getInterleavedPosThreadIdx() translates thread x,y directly to offsets for this layout
        int interleaved_thread_x = block_indexing.getInterleavedPosThreadIdx().x;
        int interleaved_thread_y = block_indexing.getInterleavedPosThreadIdx().y;

        SharedMem image_sh_class(data, make_int2(interleaved_thread_x, interleaved_thread_y));

        static const int GLOBAL_IMG_READ_WIDTH = ((IMG_WIDTH/BATCH_PIXELS_SIZE_X) + 2 * MAX_OFFSET) * BATCH_K_SIZE;
        const int global_img_read_width = ((img_width/BATCH_PIXELS_SIZE_X) + 2 * MAX_OFFSET) * BATCH_K_SIZE;

        image_sh_class.define_custom_reg_and_pred();

        // disable double buffering for offsets and weights since we can load all of them before first loop (values: 1 == one buffer, 2 == two buffers)
        static const int DOUBLE_BUFFERING_OFFSETS = 1;
        static const int DOUBLE_BUFFERING_WEIGHTS = 1;

        typedef BlockSharedMemory<NUM_THREADS, OFFSET_BLOCK_MEM_SIZE,
                1, 0, 0, DOUBLE_BUFFERING_OFFSETS, int4, LOAD_SHARED_OFFSET_BATCH_SIZE> SharedMemOffsets;
        typedef BlockSharedMemory<NUM_THREADS, WEIGHTS_BLOCK_MEM_SIZE,
                1, 0, 0, DOUBLE_BUFFERING_WEIGHTS, float4, LOAD_SHARED_OFFSET_BATCH_SIZE> SharedMemWeights;

        __shared__ typename SharedMemOffsets::Data data_offsets;
        __shared__ typename SharedMemWeights::Data data_weights;

        SharedMemOffsets offsets_sh_class(data_offsets, make_int2(0, 0)); // [thread_x, thread_y] offsets are not needed since we directly use offsets_sh_class.getData() instead
        SharedMemWeights weights_sh_class(data_weights, make_int2(0, 0)); // [thread_x, thread_y] offsets are not needed since we directly use offsets_sh_class.getData() instead

        int* offset_batch_sh = (int*)offsets_sh_class.getData(0);
        float* weights_batch_sh = weights_sh_class.getData(0);

        float4 out_val[BATCH_GAUSS_SIZE][BATCH_MEM_SUBFEATURES_SIZE * BLOCK_SUBFEATURES][BATCH_FEATURES_SIZE/NUM_READ_FEATURES][NUM_K];

#pragma unroll
        for (int g = 0; g < BATCH_GAUSS_SIZE; ++g) {
#pragma unroll
            for (int s = 0; s < BATCH_MEM_SUBFEATURES_SIZE * BLOCK_SUBFEATURES; ++s) {
#pragma unroll
                for (int f = 0; f < BATCH_FEATURES_SIZE / NUM_READ_FEATURES; ++f) {
#pragma unroll
                    for (int k = 0; k < NUM_K; ++k) {
                        if (NUM_READ_FEATURES > 0) out_val[g][s][f][k].x = 0;
                        if (NUM_READ_FEATURES > 1) out_val[g][s][f][k].y = 0;
                        if (NUM_READ_FEATURES > 2) out_val[g][s][f][k].z = 0;
                        if (NUM_READ_FEATURES > 3) out_val[g][s][f][k].w = 0;
                    }
                }
            }
        }

        PipelineEngine<BATCH_PIXELS_SIZE_X,
                BATCH_PIXELS_SIZE_Y,
                false,
                PIXELS_INTERPOLATION_Dx,
                PIXELS_INTERPOLATION_Dy,
                NUM_K,
                BATCH_K_SIZE,
                BATCH_FEATURES_SIZE,
                BATCH_COMPUTE_FEATURES_SIZE,
                BATCH_MEM_SUBFEATURES_SIZE,
                BLOCK_FEATURES,
                //IMG_WIDTH, IMG_HEIGHT,
                BATCH_PIXELS_FLOAT4,
                SharedMem,
                GLOBAL_IMG_READ_WIDTH,
                NUM_REPLICATE_OFFSETED,
                LOAD_SHARED_DATA_WITH_ASM> pipeline(image_sh_class);

        // those are for debugging purpuse only
        pipeline.block_x = block_x;
        pipeline.block_y = block_y;
        pipeline.thread_x = thread_x;
        pipeline.thread_y = thread_y;
        pipeline.interleaved_thread_x = interleaved_thread_x;
        pipeline.interleaved_thread_y = interleaved_thread_y;


        const int f_start_block = f_offset - f_block_idx;

        const int* _filter_offset_current = filter_offsets +  OFFSET(f_start_block / (BLOCK_FEATURES * BATCH_FEATURES_SIZE),
                                                                     s_offset / (BLOCK_SUBFEATURES * BATCH_MEM_SUBFEATURES_SIZE),
                                                                     g_offset/ BATCH_GAUSS_SIZE,
                                                                     0,
                                                                     F_MEM_SIZE, S_MEM_SIZE, G_MEM_SIZE, OFFSET_BLOCK_MEM_SIZE);

        const float* _filter_weights_current = filter_weights +  OFFSET(f_start_block / (BLOCK_FEATURES * BATCH_FEATURES_SIZE),
                                                                        s_offset / (BLOCK_SUBFEATURES * BATCH_MEM_SUBFEATURES_SIZE),
                                                                        g_offset/ BATCH_GAUSS_SIZE,
                                                                        0,
                                                                        F_MEM_SIZE, S_MEM_SIZE, G_MEM_SIZE, WEIGHTS_BLOCK_MEM_SIZE);

        const int* _filter_offset_next = _filter_offset_current + offsets_sh_class.getThreadIdx().x;
        const float* _filter_weights_next = _filter_weights_current + weights_sh_class.getThreadIdx().x;

        if (1){
            // load offsets and weights
            // we only load them once and are then shared over many images
            offsets_sh_class.template load_global<OFFSET_BLOCK_MEM_SIZE,0,false,0>(reinterpret_cast<const int4*>(_filter_offset_current + offsets_sh_class.getThreadIdx().x),
                                                                                   reinterpret_cast<int4*>(offsets_sh_class.getDataThreadIndexingWrite(0)));

            weights_sh_class.template load_global<WEIGHTS_BLOCK_MEM_SIZE,0,false,0>(reinterpret_cast<const float4*>(_filter_weights_current + weights_sh_class.getThreadIdx().x),
                                                                                    reinterpret_cast<float4*>(weights_sh_class.getDataThreadIndexingWrite(0)));
        }


        int block_x_inner = block_x / BATCH_PIXELS_SIZE_X;

        if (1){
            // load first batch of subfeatures/input data into shared memory
            const float* _image_global_current = filtered_images + OFFSET(n_offset,
                                                                          s_offset,
                                                                          image_sh_class.getThreadIdx().y + (MAX_OFFSET + block_y) * NUM_K / BATCH_K_SIZE * BATCH_PIXELS_SIZE_X,
                                                                          image_sh_class.getThreadIdx().x + (MAX_OFFSET + block_x_inner) * BATCH_K_SIZE,
                                                                          I, S, (img_height + 2*MAX_OFFSET) * BATCH_PIXELS_SIZE_X * NUM_K / BATCH_K_SIZE, (img_width / BATCH_PIXELS_SIZE_X + 2*MAX_OFFSET) * BATCH_K_SIZE);

            for (int s = 0 ; s < BATCH_MEM_SUBFEATURES_SIZE; ++s) {

                int buffer_index = OFFSET(0, 0, s, 0,
                                          1, DOUBLE_BUFFERING, BATCH_MEM_SUBFEATURES_SIZE, NUM_REPLICATE_OFFSETED+1);

                //image_sh_class.template load_global<GLOBAL_IMG_READ_WIDTH,NUM_REPLICATE_OFFSETED,true,1>(reinterpret_cast<const SharedMem::ELEMENT_TYPE*>(_image_global_current + (s) * (img_width + 2*MAX_OFFSET) * (img_height + 2*MAX_OFFSET) * NUM_K),
                image_sh_class.template load_global<GLOBAL_IMG_READ_WIDTH,NUM_REPLICATE_OFFSETED,false,1>(reinterpret_cast<const typename SharedMem::ELEMENT_TYPE*>(_image_global_current + (s) * (img_width + 2*MAX_OFFSET) * (img_height + 2*MAX_OFFSET) * NUM_K),
                                                                                                          reinterpret_cast<typename SharedMem::ELEMENT_TYPE*>(image_sh_class.getDataThreadIndexingWrite(buffer_index)),
                                                                                                          global_img_read_width);
            }
        }

        // load error values for all the features that are needed in this thread i.e. for BATCH_FEATURES_SIZE (or BATCH_COMPUTE_FEATURES_SIZE ?!!)
        float4 err_vals[BATCH_FEATURES_SIZE][PIXELS_INTERPOLATION_SIZE][(BATCH_PIXELS_SIZE_X * BATCH_PIXELS_SIZE_Y)/BATCH_PIXELS_FLOAT4];


        __syncthreads();

        for (int nn = 0; nn < BATCH_IMAGES ; ++nn) {
            int n = n_offset + nn;
            // skip if batch is out of bound
            if (n >= I)
                break;
            {
                pipeline.image_index = n;

                const float* _image_global_current = filtered_images + OFFSET(n,
                                                                              s_offset,
                                                                              image_sh_class.getThreadIdx().y + (MAX_OFFSET + block_y) * NUM_K / BATCH_K_SIZE * BATCH_PIXELS_SIZE_X,
                                                                              image_sh_class.getThreadIdx().x + (MAX_OFFSET + block_x_inner) * BATCH_K_SIZE,
                                                                              I, S, (img_height + 2*MAX_OFFSET) * BATCH_PIXELS_SIZE_X * NUM_K / BATCH_K_SIZE, (img_width / BATCH_PIXELS_SIZE_X + 2*MAX_OFFSET) * BATCH_K_SIZE);

                // this is actually the next image
                const float* _image_global_next = filtered_images + OFFSET(n+1,
                                                                           s_offset,
                                                                           image_sh_class.getThreadIdx().y + (MAX_OFFSET + block_y) * NUM_K / BATCH_K_SIZE * BATCH_PIXELS_SIZE_X,
                                                                           image_sh_class.getThreadIdx().x + (MAX_OFFSET + block_x_inner) * BATCH_K_SIZE,
                                                                           I, S, (img_height + 2*MAX_OFFSET) * BATCH_PIXELS_SIZE_X * NUM_K / BATCH_K_SIZE, (img_width / BATCH_PIXELS_SIZE_X + 2*MAX_OFFSET) * BATCH_K_SIZE);

                const float* _error_global_current = error_images + OFFSET5(n,
                                                                            f_offset/BATCH_FEATURES_SIZE,
                                                                            block_y + thread_y + 1,
                                                                            block_x + thread_x + 1, 0,
                                                                            I, F/BATCH_FEATURES_SIZE, error_height, error_width, BATCH_FEATURES_SIZE);

                const float* _error_global_next = error_images + OFFSET5(n+1,
                                                                         f_offset/BATCH_COMPUTE_FEATURES_SIZE,
                                                                         block_y + thread_y + 1,
                                                                         block_x + thread_x + 1, 0,
                                                                         I, F/BATCH_FEATURES_SIZE, error_height, error_width, BATCH_FEATURES_SIZE);

                // load error values for all the features that are needed in this thread i.e. for BATCH_FEATURES_SIZE (or BATCH_COMPUTE_FEATURES_SIZE ?!!)
                float4 err_vals[BATCH_FEATURES_SIZE][PIXELS_INTERPOLATION_SIZE][(BATCH_PIXELS_SIZE_X * BATCH_PIXELS_SIZE_Y)/BATCH_PIXELS_FLOAT4];
                {

                    float const* error_to_load = _error_global_current;

                    if (1) {
#pragma unroll
                        for (int dx = 0; dx < PIXELS_INTERPOLATION_SIZE; ++dx) {
                            int dx_x = dx % PIXELS_INTERPOLATION_Dx;
                            int dx_y = dx / PIXELS_INTERPOLATION_Dx;


                            int ff_base = 0;
#pragma unroll
                            for (int ff = 0; ff < BATCH_FEATURES_SIZE; ff+=NUM_READ_FEATURES)
                            {
#pragma unroll
                                for (int py = 0; py < BATCH_PIXELS_SIZE_Y; py+=BATCH_PIXELS_FLOAT4) {
#pragma unroll
                                    for (int px = 0; px < BATCH_PIXELS_SIZE_X ; ++px) {
                                        const int err_py = py + dx_y - PIXELS_INTERPOLATION_Dy +1;
                                        const int err_px = px + dx_x - PIXELS_INTERPOLATION_Dx +1;

                                        float4 tmp;

                                        int offset_read = OFFSET5(0, 0, err_py, err_px, ff_base + ff , I, F/(BATCH_FEATURES_SIZE), error_height, error_width, BATCH_FEATURES_SIZE);
                                        //int offset_read = OFFSET5(0, ff/BATCH_COMPUTE_FEATURES_SIZE, err_py, err_px, ff % BATCH_COMPUTE_FEATURES_SIZE, I, F/BATCH_COMPUTE_FEATURES_SIZE, error_height, error_width, BATCH_COMPUTE_FEATURES_SIZE)
                                        if (NUM_READ_FEATURES == 4) {
                                            const __restrict__ float4 tmp_ = __ldg(reinterpret_cast<const float4*>(error_to_load + offset_read));

                                            tmp = tmp_;
                                        } else if (NUM_READ_FEATURES == 2) {
                                            const __restrict__ float2 tmp_ = __ldg(reinterpret_cast<const float2*>(error_to_load + offset_read)) ;

                                            tmp.x = tmp_.x;
                                            tmp.y = tmp_.y;
                                        } else if (NUM_READ_FEATURES == 1) {
                                            const float tmp_ = __ldg(reinterpret_cast<const float*>(error_to_load + offset_read)) ;

                                            tmp.x = tmp_;
                                        }

                                        if (BATCH_COMPUTE_FEATURES_SIZE > 0) err_vals[ff_base+ff + 0][dx][py/BATCH_PIXELS_FLOAT4 * BATCH_PIXELS_SIZE_X + px].x = tmp.x;
                                        if (BATCH_COMPUTE_FEATURES_SIZE > 1) err_vals[ff_base+ff + 1][dx][py/BATCH_PIXELS_FLOAT4 * BATCH_PIXELS_SIZE_X + px].x = tmp.y;
                                        if (BATCH_COMPUTE_FEATURES_SIZE > 2) err_vals[ff_base+ff + 2][dx][py/BATCH_PIXELS_FLOAT4 * BATCH_PIXELS_SIZE_X + px].x = tmp.z;
                                        if (BATCH_COMPUTE_FEATURES_SIZE > 3) err_vals[ff_base+ff + 3][dx][py/BATCH_PIXELS_FLOAT4 * BATCH_PIXELS_SIZE_X + px].x = tmp.w;

                                    }
                                }
                            }
                        }
                    }
                }
                const int MAX_S_OUTER_INDEX = BLOCK_SUBFEATURES; //S /  BATCH_MEM_SUBFEATURES_SIZE;

#pragma unroll
                for (int s_outer_index = 0; s_outer_index <  BLOCK_SUBFEATURES; s_outer_index+=BATCH_MEM_SUBFEATURES_SIZE) {

                    const int s_buffer_index = (nn*BLOCK_SUBFEATURES + s_outer_index/BATCH_MEM_SUBFEATURES_SIZE) % DOUBLE_BUFFERING;

                    const float* image_global_current = _image_global_current + OFFSET8(0, 0, s_outer_index, 0, 0, 0, 0, 0, 1,
                                                                                        I, S, img_height + 2*MAX_OFFSET, BATCH_PIXELS_SIZE_X , NUM_K / BATCH_K_SIZE, img_width / BATCH_PIXELS_SIZE_X + 2*MAX_OFFSET, BATCH_K_SIZE);

                    const float* image_global_next_s_offset = _image_global_current + OFFSET8(0, 0, s_outer_index + BATCH_MEM_SUBFEATURES_SIZE, 0, 0, 0, 0, 0,
                                                                                              1, I, S, img_height + 2*MAX_OFFSET, BATCH_PIXELS_SIZE_X , NUM_K / BATCH_K_SIZE, img_width / BATCH_PIXELS_SIZE_X + 2*MAX_OFFSET, BATCH_K_SIZE);

                    const float* image_global_next_image = _image_global_next + OFFSET8(0, 0, 0, 0,0,0,0,0, 1, I, S, img_height + 2*MAX_OFFSET, BATCH_PIXELS_SIZE_X , NUM_K / BATCH_K_SIZE, img_width / BATCH_PIXELS_SIZE_X + 2*MAX_OFFSET, BATCH_K_SIZE);

                    ptr4 off_A[BATCH_GAUSS_SIZE][MAX(1,BATCH_COMPUTE_FEATURES_SIZE/NUM_READ_FEATURES)],
                            off_B[BATCH_GAUSS_SIZE][MAX(1,BATCH_COMPUTE_FEATURES_SIZE/NUM_READ_FEATURES)];

                    float4 w_A[BATCH_GAUSS_SIZE][PIXELS_INTERPOLATION_SIZE][MAX(1,BATCH_COMPUTE_FEATURES_SIZE/NUM_READ_FEATURES)],
                            w_B[BATCH_GAUSS_SIZE][PIXELS_INTERPOLATION_SIZE][MAX(1,BATCH_COMPUTE_FEATURES_SIZE/NUM_READ_FEATURES)];

                    float4 d_A[BATCH_GAUSS_SIZE][BATCH_FEATURES_SIZE][NUM_K][(BATCH_PIXELS_SIZE_X * BATCH_PIXELS_SIZE_Y)/BATCH_PIXELS_FLOAT4],
                            d_B[BATCH_GAUSS_SIZE][BATCH_FEATURES_SIZE][NUM_K][(BATCH_PIXELS_SIZE_X * BATCH_PIXELS_SIZE_Y)/BATCH_PIXELS_FLOAT4];


                    struct IterIndex {
                        int s; // sub-feature index
                        int f; // feature index
                        int g; // gauss component index
                    };


                    // global loading is done imediately (no delay)
                    // to simplyfiy the code for global loading we can force global loading to be done BATCH_MEM_SUBFEATURES_SIZE * BATCH_FEATURES_SIZE/BATCH_COMPUTE_FEATURES_SIZE loops before
                    // other units start
                    static const int start_delay_global_load = 0; //1;
                    //static const int start_delay_offset_load = 0;// + BATCH_MEM_SUBFEATURES_SIZE * BATCH_FEATURES_SIZE/BATCH_COMPUTE_FEATURES_SIZE;
                    //static const int start_delay_w_load = 1;// + BATCH_MEM_SUBFEATURES_SIZE * BATCH_FEATURES_SIZE/BATCH_COMPUTE_FEATURES_SIZE;
                    //static const int start_delay_data_load = 1;// + BATCH_MEM_SUBFEATURES_SIZE * BATCH_FEATURES_SIZE/BATCH_COMPUTE_FEATURES_SIZE;
                    //static const int start_delay_compute = 2;// + BATCH_MEM_SUBFEATURES_SIZE * BATCH_FEATURES_SIZE/BATCH_COMPUTE_FEATURES_SIZE;

                    static const int start_delay_offset_load = 0;// + BATCH_MEM_SUBFEATURES_SIZE * BATCH_FEATURES_SIZE/BATCH_COMPUTE_FEATURES_SIZE;
                    static const int start_delay_w_load = 0;// + BATCH_MEM_SUBFEATURES_SIZE * BATCH_FEATURES_SIZE/BATCH_COMPUTE_FEATURES_SIZE;
                    static const int start_delay_data_load = 0;// + BATCH_MEM_SUBFEATURES_SIZE * BATCH_FEATURES_SIZE/BATCH_COMPUTE_FEATURES_SIZE;
                    static const int start_delay_compute = 0;// + BATCH_MEM_SUBFEATURES_SIZE * BATCH_FEATURES_SIZE/BATCH_COMPUTE_FEATURES_SIZE;

                    static const int start_delay_shared_store = start_delay_compute; // global loading is split into LDG and STS, we perform STS (shared store after compute is performed!!)
                    // NOTE: EXTRA_LOOPS is max value out of start_delay_global_load, start_delay_offset_load, start_delay_w_load, start_delay_data_load and start_delay_compute
                    static const int EXTRA_LOOPS = MAX(start_delay_global_load,
                                                       MAX(start_delay_offset_load,
                                                           MAX(start_delay_w_load,
                                                               MAX(start_delay_data_load, start_delay_compute))));

                    int NUM_ITER = BATCH_MEM_SUBFEATURES_SIZE * BATCH_GAUSS_SIZE * BATCH_FEATURES_SIZE/BATCH_COMPUTE_FEATURES_SIZE;


                    /*if (thread_x == 0 && thread_y == 0 && block_x == 0 && block_y == 0 && f_offset == 0 && s_offset == 0) {

                        image_sh_class.print_int();

                    }*/

                    // iterations go over subsets of [S x G x F ] i.e. [BATCH_MEM_SUBFEATURES_SIZE] * [BATCH_GAUSS_SIZE] * [BATCH_FEATURES_SIZE/BATCH_COMPUTE_FEATURES_SIZE]
                    typename SharedMem::LoadingData ld[BATCH_MEM_SUBFEATURES_SIZE];

                    NDIndexing<BATCH_MEM_SUBFEATURES_SIZE,
                            NDIndexing<BATCH_FEATURES_SIZE/BATCH_COMPUTE_FEATURES_SIZE,
                            NDIndexingZero<BATCH_GAUSS_SIZE> > > indexing;
                    // do all in one loop
#pragma unroll
                    for (int index = 0 ; index < NUM_ITER + EXTRA_LOOPS; ++index)  {

                        IterIndex load_error;
                        IterIndex load_global;
                        IterIndex store_shared;
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
                        bool store_shared_enabled = pipeline.should_run(index, start_delay_shared_store, NUM_ITER);

                        typename SharedMem::ELEMENT_TYPE const* global_load_reading_ptr;
                        typename SharedMem::ELEMENT_TYPE* global_load_writing_ptr;
                        typename SharedMem::ELEMENT_TYPE* shared_store_writing_ptr; // used only when LDG and STS are split

                        int global_d = -1;
                        int shared_d_off = -1;
                        int shared_d_current = -1;
                        int shared_d_next = -1;
                        int global_is_next_img = -1;
                        int off_addr = -1;
                        int w_addr = -1;
                        {
                            // global loading is done immedately
                            load_global.s = indexing.getIndex<0>(index - start_delay_global_load);
                            load_global.g = indexing.getIndex<2>(index - start_delay_global_load);
                            load_global.f = indexing.getIndex<1>(index - start_delay_global_load) * BATCH_COMPUTE_FEATURES_SIZE;

                            if (load_global_enabled)
                                load_global_enabled = load_global.f == 0 && load_global.g == 0;


                            // if this is last s_outer_index index the load next image
                            bool load_next_image = s_outer_index >= MAX_S_OUTER_INDEX - 1 ? true : false;

                            // use next s buffer
                            int double_buffer_index = (s_buffer_index + 1) % DOUBLE_BUFFERING;
                            int subfeat_buffer_index = load_global.s % BATCH_MEM_SUBFEATURES_SIZE;

                            // we actually load next batch of subfeatures so use image_global_next_s_offset or use next image if last one
                            const float* image_global_load =  load_next_image ? image_global_next_image : image_global_next_s_offset;

                            global_is_next_img = load_next_image ? 1 : 0;

                            load_global.s = load_global.s % (BATCH_MEM_SUBFEATURES_SIZE);

                            if (load_next_image && n >= I-1)
                                load_global_enabled = false;

                            int buffer_index = OFFSET(0, double_buffer_index, subfeat_buffer_index, 0, 1, DOUBLE_BUFFERING, BATCH_MEM_SUBFEATURES_SIZE, NUM_REPLICATE_OFFSETED+1);

                            global_d = buffer_index;

                            global_load_reading_ptr = reinterpret_cast<const typename SharedMem::ELEMENT_TYPE*>(image_global_load + (load_global.s) * (img_width + 2*MAX_OFFSET) * (img_height + 2*MAX_OFFSET)* NUM_K);
                            global_load_writing_ptr = reinterpret_cast<typename SharedMem::ELEMENT_TYPE*>(image_sh_class.getDataThreadIndexingWrite(buffer_index));
                        }
                        bool require_sync = false;
                        bool load_offset_reg_A;
                        {
                            // offset loading is done with no delay

                            load_offset_index.s = indexing.getIndex<0>(index - start_delay_offset_load);
                            load_offset_index.g = indexing.getIndex<2>(index - start_delay_offset_load);
                            load_offset_index.f = indexing.getIndex<1>(index - start_delay_offset_load) * BATCH_COMPUTE_FEATURES_SIZE;

                            int double_buffer_index =   (s_buffer_index + load_offset_index.s/BATCH_MEM_SUBFEATURES_SIZE ) % DOUBLE_BUFFERING;
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

                                    require_sync = require_sync && (n == 0 && s_outer_index == 0) == false;
                                }
                            }

                            shared_d_next = next_double_buffer_index;

                            // switch between registers every iteration
                            bool use_reg_A = (index - start_delay_offset_load) % 2 == 0 ? true : false;

                            load_offset_reg_A = use_reg_A;

                            int address_off = OFFSET5(s_outer_index + load_offset_index.s, load_offset_index.g, load_offset_index.f/NUM_READ_FEATURES, f_block_idx/BATCH_FEATURES_SIZE, 0,
                                                      BATCH_COMPUTE_SUBFEATURES_SIZE * BATCH_MEM_SUBFEATURES_SIZE, BATCH_GAUSS_SIZE, BATCH_FEATURES_SIZE/NUM_READ_FEATURES, BLOCK_FEATURES, NUM_READ_FEATURES);

                            off_addr = address_off;

                            // load offset
                            pipeline.load_offset.offset_address = offset_batch_sh + address_off + (s_outer_index % DOUBLE_BUFFERING_OFFSETS) * OFFSET_BLOCK_MEM_SIZE;

                            int buffer_index = OFFSET(0, double_buffer_index, subfeat_buffer_index, 0, 1, DOUBLE_BUFFERING, BATCH_MEM_SUBFEATURES_SIZE, NUM_REPLICATE_OFFSETED+1);

                            shared_d_off = buffer_index;

                            pipeline.load_offset.base_address = image_sh_class.getDataThreadIndexingRead(buffer_index);
                            pipeline.load_offset.output = (ptr4*)(use_reg_A ? &off_A[load_offset_index.g][0] : &off_B[load_offset_index.g][0]);

                        }
                        bool load_w_reg_A;
                        {
                            // w and data loading is done with single delay

                            load_w_index.s = indexing.getIndex<0>(index - start_delay_w_load);
                            load_w_index.g = indexing.getIndex<2>(index - start_delay_w_load);
                            load_w_index.f = indexing.getIndex<1>(index - start_delay_w_load) * BATCH_COMPUTE_FEATURES_SIZE;
                            // switch between registers every iteration
                            bool use_reg_A = (index - start_delay_w_load) % 2 == 0 ? true : false;

                            load_w_reg_A = use_reg_A;

                            int address_off = OFFSET6(s_outer_index + load_w_index.s, load_w_index.g, 0, load_w_index.f/NUM_READ_FEATURES, f_block_idx/BATCH_FEATURES_SIZE, 0,
                                                      BATCH_COMPUTE_SUBFEATURES_SIZE * BATCH_MEM_SUBFEATURES_SIZE, BATCH_GAUSS_SIZE, PIXELS_INTERPOLATION_SIZE, BATCH_FEATURES_SIZE/NUM_READ_FEATURES, BLOCK_FEATURES, NUM_READ_FEATURES);

                            w_addr = address_off;

                            // load w
                            pipeline.load_weights.address = weights_batch_sh + address_off + (s_outer_index % DOUBLE_BUFFERING_WEIGHTS) * WEIGHTS_BLOCK_MEM_SIZE;
                            pipeline.load_weights.output = (float4*)(use_reg_A ? w_A[load_w_index.g][0] : w_B[load_w_index.g][0]);

                        }
                        bool load_data_reg_A;
                        {

                            load_data_index.s = indexing.getIndex<0>(index - start_delay_data_load);
                            load_data_index.g = indexing.getIndex<2>(index - start_delay_data_load);
                            load_data_index.f = indexing.getIndex<1>(index - start_delay_data_load) * BATCH_COMPUTE_FEATURES_SIZE;

                            // switch between registers every iteration
                            bool use_reg_A = (index - start_delay_data_load) % 2 == 0 ? true : false;

                            load_data_reg_A = use_reg_A;
                            // load data

                            pipeline.load_data.address = (ptr4*)(use_reg_A ? &off_A[load_data_index.g][0] : &off_B[load_data_index.g][0]);
                            pipeline.load_data.output = (float4*)(use_reg_A ? d_A[load_data_index.g][load_data_index.f] : d_B[load_data_index.g][load_data_index.f]);

                        }

                        bool compute_reg_A;
                        {
                            // computation is done with double  delay

                            compute_index.s = indexing.getIndex<0>(index - start_delay_compute);
                            compute_index.g = indexing.getIndex<2>(index - start_delay_compute);
                            compute_index.f = indexing.getIndex<1>(index - start_delay_compute) * BATCH_COMPUTE_FEATURES_SIZE;

                            // switch between registers every iteration
                            bool use_reg_A = (index - start_delay_compute) % 2 == 0 ? true : false;

                            compute_reg_A = use_reg_A;
                            // compute
                            pipeline.compute.weights = (float4*)(use_reg_A ? w_A[compute_index.g][0] : w_B[compute_index.g][0]);
                            pipeline.compute.errors = (float4*)err_vals[compute_index.f];
                            pipeline.compute.data = (float4*)(use_reg_A ? d_A[compute_index.g][compute_index.f] : d_B[compute_index.g][compute_index.f]);
                            pipeline.compute.output = &out_val[compute_index.g][s_outer_index * BATCH_MEM_SUBFEATURES_SIZE + compute_index.s][compute_index.f/NUM_READ_FEATURES][0];

                        }

                        // sync only before data buffer is switched
                        if (require_sync) {
                            // NOTE: sync is not needed if we have more then enough operations to cover the latency of sore operations
                            // we can rughly say that if there is more then 256 operations then STS latency should be hidden (STS latency should not be more then 100 operations on different platforms)
                            // however since store may be issued half way through operations then use 512 operations as limit
                            //if (BATCH_PIXELS_SIZE_X * BATCH_PIXELS_SIZE_Y * BATCH_FEATURES_SIZE * BATCH_GAUSS_SIZE * PIXELS_INTERPOLATION_SIZE * BATCH_MEM_SUBFEATURES_SIZE * 4  < 256)
                            {
                                __syncthreads();
                            }
#if __CUDA_ARCH__ >= 200
                            //asm("bar.arrive 15, 1536;"); // # of threads must be greater than 0
#endif
                        }

                        {
                            // load error directly here
                            load_error.s = indexing.getIndex<0>(index - start_delay_compute);
                            load_error.g = indexing.getIndex<2>(index - start_delay_compute);
                            load_error.f = indexing.getIndex<1>(index - start_delay_compute) * BATCH_COMPUTE_FEATURES_SIZE;

                            static const bool LOAD_ERROR_WITH_DOUBLE_BUFFER = false; // NOTE: leave at false as it does not work properly !!

                            float const* error_to_load = _error_global_current;

                            bool do_error_loading;

                            if (LOAD_ERROR_WITH_DOUBLE_BUFFER) {
                                // we actually load next batch of features so add BATCH_COMPUTE_FEATURES_SIZE
                                load_error.f = load_error.f + BATCH_COMPUTE_FEATURES_SIZE;

                                load_error.f = load_error.f % BATCH_FEATURES_SIZE;

                                bool load_next_image = load_error.f == 0 ? true : false;

                                error_to_load = load_next_image ? _error_global_next : _error_global_current;

                                do_error_loading = (load_error.g == BATCH_GAUSS_SIZE-1 ? true : false);

                            } else {
                                do_error_loading = (load_error.g == 0 && load_error.f == 0) ? true : false;
                            }



                            if (do_error_loading && 0) {
#pragma unroll
                                for (int dx = 0; dx < PIXELS_INTERPOLATION_SIZE; ++dx) {
                                    int dx_x = dx % PIXELS_INTERPOLATION_Dx;
                                    int dx_y = dx / PIXELS_INTERPOLATION_Dx;


                                    int ff_base = LOAD_ERROR_WITH_DOUBLE_BUFFER == false ? 0 : load_error.f;
#pragma unroll
                                    for (int ff = 0; ff < (LOAD_ERROR_WITH_DOUBLE_BUFFER == false ? BATCH_FEATURES_SIZE : BATCH_COMPUTE_FEATURES_SIZE); ff+=NUM_READ_FEATURES)
                                        //for (int ff = 0; ff < BATCH_COMPUTE_FEATURES_SIZE; ff+=NUM_READ_FEATURES)
                                    {
#pragma unroll
                                        for (int py = 0; py < BATCH_PIXELS_SIZE_Y; py+=BATCH_PIXELS_FLOAT4) {
#pragma unroll
                                            for (int px = 0; px < BATCH_PIXELS_SIZE_X ; ++px) {
                                                const int err_py = py + dx_y - PIXELS_INTERPOLATION_Dy +1;
                                                const int err_px = px + dx_x - PIXELS_INTERPOLATION_Dx +1;

                                                float4 tmp;

                                                int offset_read = OFFSET5(0, 0, err_py, err_px, ff_base + ff , I, F/(BATCH_FEATURES_SIZE), error_height, error_width, BATCH_FEATURES_SIZE);
                                                //int offset_read = OFFSET5(0, ff/BATCH_COMPUTE_FEATURES_SIZE, err_py, err_px, ff % BATCH_COMPUTE_FEATURES_SIZE, I, F/BATCH_COMPUTE_FEATURES_SIZE, error_height, error_width, BATCH_COMPUTE_FEATURES_SIZE)
                                                if (NUM_READ_FEATURES == 4) {
                                                    const __restrict__ float4 tmp_ = __ldg(reinterpret_cast<const float4*>(error_to_load + offset_read));

                                                    tmp = tmp_;
                                                } else if (NUM_READ_FEATURES == 2) {
                                                    const __restrict__ float2 tmp_ = __ldg(reinterpret_cast<const float2*>(error_to_load + offset_read)) ;

                                                    tmp.x = tmp_.x;
                                                    tmp.y = tmp_.y;
                                                } else if (NUM_READ_FEATURES == 1) {
                                                    const float tmp_ = __ldg(reinterpret_cast<const float*>(error_to_load + offset_read)) ;

                                                    tmp.x = tmp_;
                                                }

                                                if (BATCH_COMPUTE_FEATURES_SIZE > 0) err_vals[ff_base+ff + 0][dx][py/BATCH_PIXELS_FLOAT4 * BATCH_PIXELS_SIZE_X + px].x = tmp.x;
                                                if (BATCH_COMPUTE_FEATURES_SIZE > 1) err_vals[ff_base+ff + 1][dx][py/BATCH_PIXELS_FLOAT4 * BATCH_PIXELS_SIZE_X + px].x = tmp.y;
                                                if (BATCH_COMPUTE_FEATURES_SIZE > 2) err_vals[ff_base+ff + 2][dx][py/BATCH_PIXELS_FLOAT4 * BATCH_PIXELS_SIZE_X + px].x = tmp.z;
                                                if (BATCH_COMPUTE_FEATURES_SIZE > 3) err_vals[ff_base+ff + 3][dx][py/BATCH_PIXELS_FLOAT4 * BATCH_PIXELS_SIZE_X + px].x = tmp.w;

                                            }
                                        }
                                    }
                                }
                            }
                        }
                        // pipeline handles loading global (features) data, loading offsets, loading shared (features) data and computing final output

                        // pipeline load global -> shared data handles:
                        //  - subfeatures:  BATCH_MEM_SUBFEATURES_SIZE
                        //
                        // pipeline load offset handles:
                        //  - subfeatures:  BATCH_MEM_SUBFEATURES_SIZE

                        // pipeline compute handles:
                        //  - pixels:       BATCH_PIXELS_SIZE_X * BATCH_PIXELS_SIZE_Y
                        //  - features:     BATCH_COMPUTE_FEATURES_SIZE
                        //  - subfeatures:  one subfeature only
                        //  - gauss krn.:   one gaussian kernel only

                        // matrix of compute values is of [1,                1,                          BATCH_COMPUTE_FEATURES_SIZE,                       BATCH_PIXELS_SIZE_X * BATCH_PIXELS_SIZE_Y] size
                        // computation is repeated        [BATCH_GAUSS_SIZE, BATCH_MEM_SUBFEATURES_SIZE, BATCH_FEATURES_SIZE/BATCH_COMPUTE_FEATURES_SIZE,   1]-times

                        /*if (thread_x == 0 && thread_y == 0 && block_x == 0 && block_y == 0 && f_offset == 0 && s_offset == 0) {
                            if (require_sync)
                                printf("iter: %d, sycned\n", index);
                            printf("pipeline n=%d, s,f=(%d+%d,%d+%d) index %d "
                                           "gl %d (s:%d, f:%d, buff:%d, read addr=%p, write addr=%p, next_img=%d), "
                                           "off %d (s:%d, f:%d, reg:%d, buff:%d, base_addr=%p, offset_addr=%p (index %d)), "
                                           "data %d (s:%d, g:%d, f:%d, buff:%d, reg:%d, w_addr=%p (index %d)), "
                                           "compute %d (s:%d, g:%d f:%d, reg:%d)\n",
                                    n, s_offset, s_outer_index, f_offset, 0,
                                    index,
                                    (int)load_global_enabled, load_global.s, load_global.f, global_d, global_load_reading_ptr,global_load_writing_ptr, global_is_next_img,
                                    pipeline.load_offset.enabled ? 1 : 0 , load_offset_index.s, load_offset_index.f, (int)load_offset_reg_A, shared_d_off, pipeline.load_offset.base_address, pipeline.load_offset.offset_address, off_addr,
                                    pipeline.load_data.enabled ? 1 : 0 , load_data_index.s, load_data_index.g, load_data_index.f, shared_d_current, (int)load_data_reg_A, pipeline.load_weights.address, w_addr,
                                    pipeline.compute.enabled ? 1 : 0 , compute_index.s, compute_index.g, compute_index.f, (int)compute_reg_A);


                        }*/

                        typename SharedMem::LoadingData ld;

                        pipeline.load_global.enabled = load_global_enabled;
                        pipeline.load_global.reading_ptr = global_load_reading_ptr;
                        pipeline.load_global.writing_ptr = global_load_writing_ptr;
                        pipeline.load_global.img_read_width = global_img_read_width;

                        pipeline.f_index = f_offset + compute_index.f;
                        pipeline.g_index = g_offset + compute_index.g;
                        pipeline.s_index = s_offset + compute_index.s + s_outer_index;

                        pipeline.execute_step();
                    }

                }
            }
            //__syncthreads();

        }
        __syncthreads();

        // we sum results over all threads in this block that worked on the same features into a single value using warp reduce
        typedef cub::WarpReduce<float> WarpReduce;

        // Allocate WarpReduce shared memory for one warp
        __shared__ typename WarpReduce::TempStorage warp_reduce_storage[BlockIndexingT::NUM_WARPS];

        int warp_id = block_indexing.getWarpId();
        WarpReduce warp_reduce(warp_reduce_storage[warp_id]);

        int s_offset_org = s_offset;
        int f_offset_org = f_offset;

        float4 out_val_sum[BATCH_GAUSS_SIZE][BATCH_MEM_SUBFEATURES_SIZE * BLOCK_SUBFEATURES][BATCH_FEATURES_SIZE/NUM_READ_FEATURES][NUM_K];

        // if one warp handles multiple features then we need to perform warp reduce for each feature individually
        for (int i = 0; i < WARP_SIZE; i+=Bx) {
            // for each feature we will pass values from threads responsible for that feature back to first N threads
            // and then compute warp-reduce only on valid items, e.g:
            //   if Bx == 16, then we first perform warp-reduce on [0-16] threads/lane ids (default one) and additional
            //   warp-reduce on [16-32] threads/lane ids; we let only first 16 lanes perform actual computation and just
            //   copy data from [16-32] threads/lane ids back to [0-16] threads/lane ids
            //   (this code can be repeated even smaller Bx)

            s_offset = __shfl_down_sync(0xFFFFFFFF, s_offset_org, i);
            f_offset = __shfl_down_sync(0xFFFFFFFF, f_offset_org, i);

#pragma unroll
            for (int g = 0; g < BATCH_GAUSS_SIZE; ++g) {
#pragma unroll
                for (int s = 0; s < BATCH_MEM_SUBFEATURES_SIZE * BLOCK_SUBFEATURES; ++s) {
#pragma unroll
                    for (int f = 0; f < BATCH_FEATURES_SIZE / NUM_READ_FEATURES; ++f) {
#pragma unroll
                        for (int k = 0; k < NUM_K; ++k) {
                            //if (n_offset == 1)
                            //    printf("added %f from s,g,f: %d,%d,%d, block: %d,%d and img: %d\n",
                            //           out_val[g][s][f][k].x, s_offset + s , g_offset + g , f_offset + f * NUM_READ_FEATURES + 0, block_y, block_x, n_offset);

                            if (NUM_READ_FEATURES > 0) out_val_sum[g][s][f][k].x = warp_reduce.Sum(__shfl_down_sync(0xFFFFFFFF, out_val[g][s][f][k].x,i),Bx);
                            if (NUM_READ_FEATURES > 1) out_val_sum[g][s][f][k].y = warp_reduce.Sum(__shfl_down_sync(0xFFFFFFFF, out_val[g][s][f][k].y,i),Bx);
                            if (NUM_READ_FEATURES > 2) out_val_sum[g][s][f][k].z = warp_reduce.Sum(__shfl_down_sync(0xFFFFFFFF, out_val[g][s][f][k].z,i),Bx);
                            if (NUM_READ_FEATURES > 3) out_val_sum[g][s][f][k].w = warp_reduce.Sum(__shfl_down_sync(0xFFFFFFFF, out_val[g][s][f][k].w,i),Bx);
                        }
                    }
                }
            }

            // now we finally write values to global mem
            // only one thread at [0,0] needs to write it (other threads should have the same value)

            if (block_indexing.warp_lane_id() == 0) {
#pragma unroll
                for (int s = 0; s < BATCH_MEM_SUBFEATURES_SIZE * BLOCK_SUBFEATURES ; ++s) {
#pragma unroll
                    for (int g = 0; g < BATCH_GAUSS_SIZE; ++g) {
#pragma unroll
                        for (int f = 0; f < BATCH_FEATURES_SIZE/NUM_READ_FEATURES; ++f) {
#pragma unroll
                            for (int k = 0; k < NUM_K; ++k) {
                                //if (s_offset + s == 0 && g_offset + g == 0 && f_offset + f * NUM_READ_FEATURES + 0 == 15 && k == 0) {
                                //    printf("added %f from s,g,f: %d,%d,%d, block: %d,%d and img: %d\n", out_val_sum[g][s][f][k].x, s_offset + s , g_offset + g , f_offset + f * NUM_READ_FEATURES + 0, block_y, block_x, n_offset);
                                //}
                                if (NUM_READ_FEATURES > 0) atomicAdd(&(output[OFFSET(k, s_offset + s, g_offset + g, f_offset + f * NUM_READ_FEATURES + 0, NUM_K, S, G, F)]), out_val_sum[g][s][f][k].x);
                                if (NUM_READ_FEATURES > 1) atomicAdd(&(output[OFFSET(k, s_offset + s, g_offset + g, f_offset + f * NUM_READ_FEATURES + 1, NUM_K, S, G, F)]), out_val_sum[g][s][f][k].y);
                                if (NUM_READ_FEATURES > 2) atomicAdd(&(output[OFFSET(k, s_offset + s, g_offset + g, f_offset + f * NUM_READ_FEATURES + 2, NUM_K, S, G, F)]), out_val_sum[g][s][f][k].z);
                                if (NUM_READ_FEATURES > 3) atomicAdd(&(output[OFFSET(k, s_offset + s, g_offset + g, f_offset + f * NUM_READ_FEATURES + 3, NUM_K, S, G, F)]), out_val_sum[g][s][f][k].w);
                            }
                        }
                    }
                }
            }

        }
#endif
    }

#include <iostream>

    template <typename BlockIndexingT,
            typename ELEMENT_FLOAT_TYPE,
            typename ELEMENT_INT_TYPE>
    __global__  void
    perpare_weights_and_offsets_bw_multi(const float* filter_weights, const float* filter_offsets_x, const float* filter_offsets_y,
                                         float *prepared_filter_weights, int *prepared_filter_offsets,
                                         int S, int G, int F, int kernel_w, int kernel_h, bool offsets_already_centered) {

        static const int NUM_SM = BlockIndexingT::NUM_SM;
        static const int Bx = BlockIndexingT::Bx;
        static const int By = BlockIndexingT::By;
        static const int BLOCK_FEATURES = BlockIndexingT::BLOCK_FEATURES;
        static const int BLOCK_SUBFEATURES = BlockIndexingT::BLOCK_SUBFEATURES;
        static const int BATCH_PIXELS_SIZE_X = BlockIndexingT::BATCH_PIXELS_SIZE_X;
        static const int BATCH_PIXELS_SIZE_Y = BlockIndexingT::BATCH_PIXELS_SIZE_Y;
        static const int BATCH_FEATURES_SIZE = BlockIndexingT::BATCH_FEATURES_SIZE;
        static const int BATCH_COMPUTE_FEATURES_SIZE = BlockIndexingT::BATCH_COMPUTE_FEATURES_SIZE;
        static const int BATCH_MEM_SUBFEATURES_SIZE = BlockIndexingT::BATCH_MEM_SUBFEATURES_SIZE;
        static const int BATCH_GAUSS_SIZE = BlockIndexingT::BATCH_GAUSS_SIZE;
        static const int IMG_WIDTH = BlockIndexingT::IMG_WIDTH;
        static const int IMG_HEIGHT = BlockIndexingT::IMG_HEIGHT;
        static const int MAX_OFFSET = BlockIndexingT::MAX_OFFSET;
        static const int NUM_THREADS = BlockIndexingT::NUM_THREADS;

        static const int NUM_K = BlockIndexingT::NUM_K;
        static const int BATCH_K_SIZE = BlockIndexingT::BATCH_K_SIZE;

        static const int NUM_REPLICATE_OFFSETED = 0;

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

        int input_index = (( s_input_index )*G + g_input_index ) * F + f_input_index;

        // output data is of the form:
        // float4 of size [F / (BATCH_FEATURES_SIZE * BLOCK_FEATURES)] x [S / (BATCH_MEM_SUBFEATURES_SIZE*BLOCK_SUBFEATURES)] x [G / BATCH_GAUSS_SIZE]
        //				 	x [BATCH_MEM_SUBFEATURES_SIZE * BLOCK_SUBFEATURES] x [BATCH_GAUSS_SIZE] x [PIXELS_INTERPOLATION_SIZE] x [BATCH_FEATURES_SIZE/NUM_READ_FEATURES] x [BLOCK_FEATURES];
        // NOTE: x,y,z,w in float4 go over 4 elements of BATCH_FEATURES_SIZE that a single thread handles
        // NOTE: float4 is defined by ELEMENT_FLOAT_TYPE and its real size is NUM_READ_FEATURES

        static const int dim1_size = BLOCK_FEATURES;
        static const int dim2_size = BATCH_FEATURES_SIZE/NUM_READ_FEATURES;
        static const int dim3_size = 4;
        static const int dim4_size = BATCH_GAUSS_SIZE;
        static const int dim5_size = BATCH_MEM_SUBFEATURES_SIZE * BLOCK_SUBFEATURES;

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


        int output_index = OFFSET8(main_f_index,
                                   main_s_index,
                                   main_g_index,
                                   s_mem_index,
                                   g_index,
                                   0,
                                   f_batch_index,
                                   f_block_index,
                                   dim8_size, dim7_size, dim6_size, dim5_size, dim4_size, 1, dim2_size, dim1_size) * NUM_READ_FEATURES;

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
        if (NUM_READ_FEATURES > 0) offset_x.x = reinterpret_cast<const float*>(filter_offsets_x4)[input_index + 0] - (offsets_already_centered == false ? kernel_w/2 : 0);
        if (NUM_READ_FEATURES > 1) offset_x.y = reinterpret_cast<const float*>(filter_offsets_x4)[input_index + 1] - (offsets_already_centered == false ? kernel_w/2 : 0);
        if (NUM_READ_FEATURES > 2) offset_x.z = reinterpret_cast<const float*>(filter_offsets_x4)[input_index + 2] - (offsets_already_centered == false ? kernel_w/2 : 0);
        if (NUM_READ_FEATURES > 3) offset_x.w = reinterpret_cast<const float*>(filter_offsets_x4)[input_index + 3] - (offsets_already_centered == false ? kernel_w/2 : 0);

        if (NUM_READ_FEATURES > 0) offset_y.x = reinterpret_cast<const float*>(filter_offsets_y4)[input_index + 0] - (offsets_already_centered == false ? kernel_h/2 : 0);
        if (NUM_READ_FEATURES > 1) offset_y.y = reinterpret_cast<const float*>(filter_offsets_y4)[input_index + 1] - (offsets_already_centered == false ? kernel_h/2 : 0);
        if (NUM_READ_FEATURES > 2) offset_y.z = reinterpret_cast<const float*>(filter_offsets_y4)[input_index + 2] - (offsets_already_centered == false ? kernel_h/2 : 0);
        if (NUM_READ_FEATURES > 3) offset_y.w = reinterpret_cast<const float*>(filter_offsets_y4)[input_index + 3] - (offsets_already_centered == false ? kernel_h/2 : 0);

        /*
        // DEBUG ONLY !!!
        if (f_block_index % 2 == 1) {
            if (NUM_READ_FEATURES > 0) offset_x.x+=0;
            if (NUM_READ_FEATURES > 1) offset_x.y+=0;
            if (NUM_READ_FEATURES > 2) offset_x.z+=0;
            if (NUM_READ_FEATURES > 3) offset_x.w+=0;
        }*/


        // offset is relative to shared memory organization which is defined by  BlockSharedMemory parameters:
        //		- SharedMem::ALLOC_WIDTH
        //		- SharedMem::ALLOC_HEIGHT

        //static const int BATCH_COMPUTE_FEATURES_SIZE = 4;

        // using float4 to load so use
        static const int BATCH_SH_PIXELS_SIZE = 4;

        static const int DOUBLE_BUFFERING = 2;

        typedef BlockSharedMemory<NUM_THREADS,
                Bx * BATCH_K_SIZE,
                By * BATCH_PIXELS_SIZE_Y * BATCH_PIXELS_SIZE_X * (NUM_K / BATCH_K_SIZE),
                MAX_OFFSET * BATCH_K_SIZE,
                MAX_OFFSET * BATCH_PIXELS_SIZE_X * (NUM_K / BATCH_K_SIZE),
                (NUM_REPLICATE_OFFSETED+1) * DOUBLE_BUFFERING * BATCH_MEM_SUBFEATURES_SIZE,
                float4,
                BATCH_SH_PIXELS_SIZE> SharedMem;

        // shared memory has the following data layout:
        //  [By + 2 * MAX_OFFSET] x [BATCH_PIXELS_SIZE_X] x [NUM_K / BATCH_K_SIZE] x [By + 2 * MAX_OFFSET] x [BATCH_K_SIZE]
        // where NUM_K is number of subfeatures that can share the same offsets (for calculation of w, mu1, mu2, sigma)
        // and is organized in BATCH_K_SIZE batches that can be loaded with one command (normally BATCH_K_SIZE = 2)
        // pixels are split so that BATCH_PIXELS_SIZE_X are in the same row



        int4 offset_x_inner;
        int4 offset_x_outer;

        // first we calculate indexes for inner and outer pixels (for x dimension only)
        if (NUM_READ_FEATURES > 0) offset_x_inner.x = (int)floorf(offset_x.x) / BATCH_PIXELS_SIZE_X;
        if (NUM_READ_FEATURES > 1) offset_x_inner.y = (int)floorf(offset_x.y) / BATCH_PIXELS_SIZE_X;
        if (NUM_READ_FEATURES > 2) offset_x_inner.z = (int)floorf(offset_x.z) / BATCH_PIXELS_SIZE_X;
        if (NUM_READ_FEATURES > 3) offset_x_inner.w = (int)floorf(offset_x.w) / BATCH_PIXELS_SIZE_X;

        if (NUM_READ_FEATURES > 0) offset_x_outer.x = (int)floorf(offset_x.x) % BATCH_PIXELS_SIZE_X;
        if (NUM_READ_FEATURES > 1) offset_x_outer.y = (int)floorf(offset_x.y) % BATCH_PIXELS_SIZE_X;
        if (NUM_READ_FEATURES > 2) offset_x_outer.z = (int)floorf(offset_x.z) % BATCH_PIXELS_SIZE_X;
        if (NUM_READ_FEATURES > 3) offset_x_outer.w = (int)floorf(offset_x.w) % BATCH_PIXELS_SIZE_X;

        int4 output_offset;

        // then compute offset based on upper memory layout and inner/outer offsets
        if (NUM_READ_FEATURES > 0) output_offset.x = OFFSET5((int)floorf(offset_y.x), offset_x_outer.x, 0, offset_x_inner.x, 0, By + 2 * MAX_OFFSET, BATCH_PIXELS_SIZE_X, NUM_K / BATCH_K_SIZE, SharedMem::PITCHED_WIDTH/BATCH_K_SIZE, BATCH_K_SIZE);
        if (NUM_READ_FEATURES > 1) output_offset.y = OFFSET5((int)floorf(offset_y.y), offset_x_outer.y, 0, offset_x_inner.y, 0, By + 2 * MAX_OFFSET, BATCH_PIXELS_SIZE_X, NUM_K / BATCH_K_SIZE, SharedMem::PITCHED_WIDTH/BATCH_K_SIZE, BATCH_K_SIZE);
        if (NUM_READ_FEATURES > 2) output_offset.z = OFFSET5((int)floorf(offset_y.z), offset_x_outer.z, 0, offset_x_inner.z, 0, By + 2 * MAX_OFFSET, BATCH_PIXELS_SIZE_X, NUM_K / BATCH_K_SIZE, SharedMem::PITCHED_WIDTH/BATCH_K_SIZE, BATCH_K_SIZE);
        if (NUM_READ_FEATURES > 3) output_offset.w = OFFSET5((int)floorf(offset_y.w), offset_x_outer.w, 0, offset_x_inner.w, 0, By + 2 * MAX_OFFSET, BATCH_PIXELS_SIZE_X, NUM_K / BATCH_K_SIZE, SharedMem::PITCHED_WIDTH/BATCH_K_SIZE, BATCH_K_SIZE);

        // offset should be in bytes !!! (not in 4 bytes as for float or 16 bytes as for float4)
        if (NUM_READ_FEATURES > 0) output_offset.x *= sizeof(float);
        if (NUM_READ_FEATURES > 1) output_offset.y *= sizeof(float);
        if (NUM_READ_FEATURES > 2) output_offset.z *= sizeof(float);
        if (NUM_READ_FEATURES > 3) output_offset.w *= sizeof(float);


        int* out_off = reinterpret_cast<int*>(prepared_filter_offsets4) + output_index;

        if (NUM_READ_FEATURES > 0) out_off[0] = output_offset.x;
        if (NUM_READ_FEATURES > 1) out_off[1] = output_offset.y;
        if (NUM_READ_FEATURES > 2) out_off[2] = output_offset.z;
        if (NUM_READ_FEATURES > 3) out_off[3] = output_offset.w;


        // for weights we integrate interpolation values into four sets of weights

        int output_index_0 =  OFFSET8(main_f_index,
                                      main_s_index,
                                      main_g_index,
                                      s_mem_index,
                                      g_index,
                                      0,
                                      f_batch_index,
                                      f_block_index,
                                      dim8_size, dim7_size, dim6_size, dim5_size, dim4_size, dim3_size, dim2_size, dim1_size) * NUM_READ_FEATURES;

        // prepare factors for interpolation

        float4 interp_offset_y,interp_offset_x;

        // get x-floor(x)
        if (NUM_READ_FEATURES > 0) interp_offset_x.x = offset_x.x - floorf(offset_x.x);
        if (NUM_READ_FEATURES > 1) interp_offset_x.y = offset_x.y - floorf(offset_x.y);
        if (NUM_READ_FEATURES > 2) interp_offset_x.z = offset_x.z - floorf(offset_x.z);
        if (NUM_READ_FEATURES > 3) interp_offset_x.w = offset_x.w - floorf(offset_x.w);

        // get y-floor(y)
        if (NUM_READ_FEATURES > 0) interp_offset_y.x = offset_y.x - floorf(offset_y.x);
        if (NUM_READ_FEATURES > 1) interp_offset_y.y = offset_y.y - floorf(offset_y.y);
        if (NUM_READ_FEATURES > 2) interp_offset_y.z = offset_y.z - floorf(offset_y.z);
        if (NUM_READ_FEATURES > 3) interp_offset_y.w = offset_y.w - floorf(offset_y.w);


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

        // we should not use weights here since not all derivatives have w in their formula
        // for derivative that require weights this has to be done seperately in the end (i.e., we can just multiply accumulated gradients with weights)
        //const float* w = reinterpret_cast<const float*>(filter_weights4) + input_index;
        const float w[4] = {1.0f,1.0f,1.0f,1.0f}; // dummy variable so that we do not consider weights


        // create weights with interpolation factors
        // dx=0,dy=0
        if (NUM_READ_FEATURES > 0) reinterpret_cast<float*>(prepared_filter_weights4)[output_index_0 + 0] = w[0] * factor_00.x;
        if (NUM_READ_FEATURES > 1) reinterpret_cast<float*>(prepared_filter_weights4)[output_index_0 + 1] = w[1] * factor_00.y;
        if (NUM_READ_FEATURES > 2) reinterpret_cast<float*>(prepared_filter_weights4)[output_index_0 + 2] = w[2] * factor_00.z;
        if (NUM_READ_FEATURES > 3) reinterpret_cast<float*>(prepared_filter_weights4)[output_index_0 + 3] = w[3] * factor_00.w;

        // dx=1,dy=0
        int output_index_1 = output_index_0 + 1 *  (dim1_size * dim2_size) * NUM_READ_FEATURES;
        if (NUM_READ_FEATURES > 0) reinterpret_cast<float*>(prepared_filter_weights4)[output_index_1 + 0] = w[0] * factor_01.x;
        if (NUM_READ_FEATURES > 1) reinterpret_cast<float*>(prepared_filter_weights4)[output_index_1 + 1] = w[1] * factor_01.y;
        if (NUM_READ_FEATURES > 2) reinterpret_cast<float*>(prepared_filter_weights4)[output_index_1 + 2] = w[2] * factor_01.z;
        if (NUM_READ_FEATURES > 3) reinterpret_cast<float*>(prepared_filter_weights4)[output_index_1 + 3] = w[3] * factor_01.w;

        // dx=0,dy=1
        int output_index_2 = output_index_0 + 2 *  (dim1_size * dim2_size) * NUM_READ_FEATURES;
        if (NUM_READ_FEATURES > 0) reinterpret_cast<float*>(prepared_filter_weights4)[output_index_2 + 0] = w[0] * factor_10.x;
        if (NUM_READ_FEATURES > 1) reinterpret_cast<float*>(prepared_filter_weights4)[output_index_2 + 1] = w[1] * factor_10.y;
        if (NUM_READ_FEATURES > 2) reinterpret_cast<float*>(prepared_filter_weights4)[output_index_2 + 2] = w[2] * factor_10.z;
        if (NUM_READ_FEATURES > 3) reinterpret_cast<float*>(prepared_filter_weights4)[output_index_2 + 3] = w[3] * factor_10.w;

        // dx=1,dy=1
        int output_index_3 = output_index_0 + 3 *  (dim1_size * dim2_size) * NUM_READ_FEATURES;
        if (NUM_READ_FEATURES > 0) reinterpret_cast<float*>(prepared_filter_weights4)[output_index_3 + 0] = w[0] * factor_11.x;
        if (NUM_READ_FEATURES > 1) reinterpret_cast<float*>(prepared_filter_weights4)[output_index_3 + 1] = w[1] * factor_11.y;
        if (NUM_READ_FEATURES > 2) reinterpret_cast<float*>(prepared_filter_weights4)[output_index_3 + 2] = w[2] * factor_11.z;
        if (NUM_READ_FEATURES > 3) reinterpret_cast<float*>(prepared_filter_weights4)[output_index_3 + 3] = w[3] * factor_11.w;
    }


    template <int TILE_DIM_X, int TILE_DIM_Y, int TILE_DIM_IMAGE, int BATCH_FEATURES_SIZE, int NEW_WIDTH, int NEW_HEIGHT, int PIXELS_INTERPOLATION_Dx, int PIXELS_INTERPOLATION_Dy>
    __global__ void
    interleave_error_data_kernel(const float* error_data, float* output_data, const int N, const int F, const int img_width, const int img_height, const int num_img_patches_width, const int num_img_patches_height, const bool ignore_edge_gradients) {

// INPUT: input_data  	[N x F x H x W]
// OUTPUT output  		[(num_img_patches_width*num_img_patches_height*N) x F / BATCH_FEATURES_SIZE x NEW_HEIGHT x NEW_WIDTH x BATCH_FEATURES_SIZE]

#ifndef CUBIN_EMBEDDING
        //static const int BLOCK_COLUMNS = 1;

        // TILE_DIM_X <= 256 (NUM threads) / BATCH_FEATURES_SIZE
        // TILE_DIM_Y <= BATCH_FEATURES_SIZE

        __shared__ float tile[TILE_DIM_IMAGE][TILE_DIM_Y][TILE_DIM_X+1];

        // threadIdx.x => over [1 .. WxH]
        // threadIdx.y => over [1 .. BATCH_FEATURES_SIZE]
        // threadIdx.z => over [1 .. N x F / BATCH_FEATURES_SIZE]

        int thread_yx = threadIdx.x;
        int thread_f = threadIdx.y;

        int xy = blockIdx.x * TILE_DIM_X + thread_yx;
        int f = blockIdx.y * TILE_DIM_Y + thread_f;
        int n = blockIdx.z * TILE_DIM_IMAGE + threadIdx.z;

        if (xy < img_height * img_width)
#pragma unroll
            for (int i = 0; i < TILE_DIM_IMAGE; ++i)
                tile[i][thread_f][thread_yx] = error_data[OFFSET(0,n+i,f,xy, 1,N * F/BATCH_FEATURES_SIZE,BATCH_FEATURES_SIZE,(img_height * img_width))];

        int x = xy % img_width;
        int y = xy / img_width;

        // transpose block offset
        int tid = threadIdx.x +
                  TILE_DIM_X * threadIdx.y +
                  TILE_DIM_X * TILE_DIM_Y * threadIdx.z;

        int transposed_thread_f = tid % (TILE_DIM_Y);
        int transposed_thread_yx = tid / (TILE_DIM_Y);

        int transposed_f = blockIdx.y * TILE_DIM_Y + transposed_thread_f;
        int transposed_yx = blockIdx.x * TILE_DIM_X + transposed_thread_yx;

        // convert to x,y location in transposed output image
        int transposed_y = transposed_yx / img_width;
        int transposed_x = transposed_yx % img_width;

        // then split img into patches of uniform size [NEW_HEIGHT, NEW_WIDTH]
        // get index of patch that this pixel will belong to
        int img_patch_x = transposed_x / NEW_WIDTH;
        int img_patch_y = transposed_y / NEW_HEIGHT;

        // transposed px location then needs to be converted into x,y within a patch
        //int transposed_patch_x = transposed_x % NEW_WIDTH;
        //int transposed_patch_y = transposed_y % NEW_HEIGHT;

        __syncthreads();

        if (transposed_yx < img_height * img_width)
            /*#pragma unroll
            for (int i = 0; i < TILE_DIM_IMAGE; ++i) {
                int output_index = OFFSET8(0, 0, img_patch_y, img_patch_x, n+i, transposed_patch_y + 1, transposed_patch_x + 1, transposed_f,
                                           1, 1, num_img_patches_height, num_img_patches_width, N * F / BATCH_FEATURES_SIZE, NEW_HEIGHT+PIXELS_INTERPOLATION_Dy-1, NEW_WIDTH+PIXELS_INTERPOLATION_Dx-1,  BATCH_FEATURES_SIZE);

                output_data[output_index] = tile[i][transposed_thread_f][transposed_thread_yx];
            }*/
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
                    transposed_patch_x += PIXELS_INTERPOLATION_Dx-1;
                    transposed_patch_y += PIXELS_INTERPOLATION_Dy-1;

                    int current_patch_y = img_patch_y + dy;
                    int current_patch_x = img_patch_x + dx;

                    // we write only if x,y values inside valid patch index
                    // this notation works both for main patch as well as for neigbooring patches that need pixels as its border values
                    if (0 <= transposed_patch_x && transposed_patch_x < NEW_WIDTH+PIXELS_INTERPOLATION_Dx-1 &&
                        0 <= transposed_patch_y && transposed_patch_y < NEW_HEIGHT+PIXELS_INTERPOLATION_Dy-1 &&
                        0 <= current_patch_x && current_patch_x < num_img_patches_width	&&
                        0 <= current_patch_y && current_patch_y < num_img_patches_height)
                    {

                        // make sure to set errors at the right/bootom edges to zero if requested so
                        if (ignore_edge_gradients &&
                            ((current_patch_x == num_img_patches_width - 1 && transposed_patch_x == NEW_WIDTH+PIXELS_INTERPOLATION_Dx-2) ||
                             (current_patch_y == num_img_patches_height - 1 && transposed_patch_y == NEW_HEIGHT+PIXELS_INTERPOLATION_Dy-2)))
                            continue;

#pragma unroll
                        for (int i = 0; i < TILE_DIM_IMAGE; ++i) {
                            /*
                            int output_index = OFFSET8(current_patch_y, current_patch_x, n+i,transposed_patch_y,transposed_patch_x_outer, transposed_k_outer, transposed_patch_x_inner, transposed_k_inner,
                                                       num_img_patches_height, num_img_patches_width, N, NEW_HEIGHT + 2*BORDER_SIZE, BATCH_PIXELS_X, K / BATCH_K, (NEW_WIDTH + 2*BORDER_SIZE)/ BATCH_PIXELS_X,  BATCH_K);

                            //if (transposed_k == 0 && transposed_yx == 35)
                            //	printf("writing value %f to position %d at patch j,i=%d,%d,  transposed_patch y,x=%d,%d  (transposed y,x=(%d,%d)) img=%d,patch_y=%d,patch_x_outer=%d,k_outer=%d,patch_x_inner=%d,k_inner=%d\n",
                            //		   tile[i][transposed_thread_k][transposed_thread_yx], output_index, current_patch_y, current_patch_x, transposed_patch_y, transposed_patch_x, transposed_y,transposed_x, n+i,transposed_patch_y,transposed_patch_x_outer, transposed_k_outer, transposed_patch_x_inner, transposed_k_inner);

                            output_data[output_index] = tile[i][transposed_thread_k][transposed_thread_yx];*/

                            int output_index = OFFSET8(0, 0, current_patch_y, current_patch_x, n+i, transposed_patch_y, transposed_patch_x, transposed_f,
                                                       1, 1, num_img_patches_height, num_img_patches_width, N * F / BATCH_FEATURES_SIZE, NEW_HEIGHT+PIXELS_INTERPOLATION_Dy-1, NEW_WIDTH+PIXELS_INTERPOLATION_Dx-1,  BATCH_FEATURES_SIZE);

                            output_data[output_index] = tile[i][transposed_thread_f][transposed_thread_yx];
                        }
                    }
                }
            }

#endif
    }

    template <int TILE_DIM_YX, int TILE_DIM_K, int TILE_DIM_IMAGE, int BATCH_PIXELS_X, int BATCH_K, int NEW_WIDTH_, int NEW_HEIGHT_, int BORDER_SIZE_>
    __global__ void
    interleave_input_data_kernel(const float* input_data, float* output_data, const int N, const int K, const int IN_K, const int img_width_in, const int img_height_in, const int img_width, const int img_height, const int num_img_patches_width, const int num_img_patches_height) {

// INPUT: input_data  	[(N*S) x K x H x W]
// OUTPUT output  		[(N*S) x H x BATCH_PIXELS_X x (K / BATCH_K) x (W / BATCH_PIXELS_X) x BATCH_K]
        // NOTE: N <= N*S

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

        // TILE_DIM_X <= 256 (NUM threads) / K = 64
        // TILE_DIM_Y <= K

        __shared__ float tile[TILE_DIM_IMAGE][TILE_DIM_K][TILE_DIM_YX+1];

        // threadIdx.x => over [1 .. WxH]
        // threadIdx.y => over [1 .. K]
        // threadIdx.z => over [1 .. N*S]

        int thread_yx = threadIdx.x;
        int thread_k = threadIdx.y;

        int yx = blockIdx.x * TILE_DIM_YX + thread_yx;
        int k = blockIdx.y * TILE_DIM_K + thread_k;
        int n = blockIdx.z * TILE_DIM_IMAGE + threadIdx.z;

        int x = yx % img_width_in;
        int y = yx / img_width_in;

        if (yx < img_height_in * img_width_in)
#pragma unroll
            for (int i = 0; i < TILE_DIM_IMAGE; ++i)
                tile[i][thread_k][thread_yx] = input_data[OFFSET(n+i,k,y,x, N,IN_K,img_height_in, img_width_in)];

        // transpose block offset
        int tid = threadIdx.x +
                  TILE_DIM_YX * threadIdx.y +
                  TILE_DIM_YX * TILE_DIM_K * threadIdx.z;

        int transposed_thread_k = tid % (TILE_DIM_K);
        int transposed_thread_yx = (tid / (TILE_DIM_K));// * TILE_DIM_K)  % TILE_DIM_YX;

        int transposed_k = blockIdx.y * TILE_DIM_K + transposed_thread_k;
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

                    int transposed_k_inner = transposed_k % BATCH_K;
                    int transposed_k_outer = transposed_k / BATCH_K;

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
                        for (int i = 0; i < TILE_DIM_IMAGE; ++i) {

                            int output_index = OFFSET8(current_patch_y, current_patch_x, n+i,transposed_patch_y,transposed_patch_x_outer, transposed_k_outer, transposed_patch_x_inner, transposed_k_inner,
                                                       num_img_patches_height, num_img_patches_width, N, NEW_HEIGHT + 2*BORDER_SIZE_X, BATCH_PIXELS_X, K / BATCH_K, (NEW_WIDTH + 2*BORDER_SIZE_Y)/ BATCH_PIXELS_X,  BATCH_K);

                            //if (transposed_k == 0 && transposed_yx == 35)
                            //	printf("writing value %f to position %d at patch j,i=%d,%d,  transposed_patch y,x=%d,%d  (transposed y,x=(%d,%d)) img=%d,patch_y=%d,patch_x_outer=%d,k_outer=%d,patch_x_inner=%d,k_inner=%d\n",
                            //		   tile[i][transposed_thread_k][transposed_thread_yx], output_index, current_patch_y, current_patch_x, transposed_patch_y, transposed_patch_x, transposed_y,transposed_x, n+i,transposed_patch_y,transposed_patch_x_outer, transposed_k_outer, transposed_patch_x_inner, transposed_k_inner);

                            output_data[output_index] = tile[i][transposed_thread_k][transposed_thread_yx];
                        }
                    }
                }
            }
#endif
    }

    template <typename BlockIndexingT>
    class DAUConvBwdInputImage {
    private:
        enum {
            // values from main block indexing sizes
            NUM_K = BlockIndexingT::NUM_K,
            BATCH_PIXELS_X = BlockIndexingT::BATCH_PIXELS_SIZE_X,
            BATCH_K = BlockIndexingT::BATCH_K_SIZE,
            NEW_WIDTH = BlockIndexingT::IMG_WIDTH,
            NEW_HEIGHT = BlockIndexingT::IMG_HEIGHT,
            BORDER_SIZE = BlockIndexingT::MAX_OFFSET,

            // values specific for this kernel
            CUDA_THREADS = 256,

            // TILE_DIM_X * TILE_DIM_Y * TILE_DIM_IMAGE gets us to 512 of shared data per block (i.e. 2 kB)
            TILE_DIM_X = CUDA_THREADS/8,
            TILE_DIM_Y = NUM_K,
            TILE_DIM_IMAGE = BlockIndexingT::BATCH_IMAGES >= 4 ? 4 : 1	//
        };
        const int img_width_in, img_height_in;
        const int img_width;
        const int img_height;
        const int N;
        const int S;
        const int K;
        const int IN_K;

        int new_img_parts_width;
        int new_img_parts_height;

        dim3 threadsPerBlock;
        dim3 numBlocks;

    public:
        DAUConvBwdInputImage(const int img_width_in, const int img_height_in, const int img_width, const int img_height, const int N, const int S, const int K, const int IN_K, int new_img_parts_width, int new_img_parts_height) :
                img_width_in(img_width_in), img_height_in(img_height_in), img_width(img_width), img_height(img_height), N(N), S(S), K(K), IN_K(IN_K), new_img_parts_width(new_img_parts_width), new_img_parts_height(new_img_parts_height) {

            threadsPerBlock = dim3 (TILE_DIM_X, TILE_DIM_Y, 1);

            numBlocks = dim3( ((int)ceil(img_width_in*img_height_in) + threadsPerBlock.x - 1) / threadsPerBlock.x,	// over image width and height
                              ((int)ceil(K) + threadsPerBlock.y - 1) / threadsPerBlock.y,												// over K
                              ((int)ceil(N*S/(float)TILE_DIM_IMAGE) + threadsPerBlock.z - 1) / threadsPerBlock.z);					// over N * S

        }
        size_t get_allocation_size() {
            return sizeof(float) * (NEW_WIDTH + 2*BORDER_SIZE) * (NEW_HEIGHT + 2*BORDER_SIZE) * K * S *  (N+1) * new_img_parts_width * new_img_parts_height;
        }
        float* create_input(float* interleaved_images_output, const float* filtered_images, cudaStream_t streamId = NULL) {


            interleave_input_data_kernel<TILE_DIM_X,TILE_DIM_Y,TILE_DIM_IMAGE, BATCH_PIXELS_X, BATCH_K, NEW_WIDTH, NEW_HEIGHT, BORDER_SIZE><<<numBlocks,threadsPerBlock, 0, streamId>>>(filtered_images, interleaved_images_output, N*S, K, IN_K, img_width_in, img_height_in, img_width, img_height, new_img_parts_width, new_img_parts_height);


            if (0){
                // DEBUG ONLY
                // check correctness of interleaved data
                float* filtered_images_cpu = new float[img_width * img_height * K * S *  N ];
                float* interleaved_images_cpu = new float[get_allocation_size()/sizeof(float)];

                cudaMemcpy(interleaved_images_cpu, interleaved_images_output, get_allocation_size(), cudaMemcpyDeviceToHost);
                cudaMemcpy(filtered_images_cpu, filtered_images, sizeof(float)* img_width * img_height * K * S *  N, cudaMemcpyDeviceToHost);
                cudaDeviceSynchronize();

                for (int i = 0; i < N*new_img_parts_width * new_img_parts_height; ++i) {
                    printf("img index: %d\n" ,i);
                    for (int s = 0; s < S; ++s)
                    {
                        //int s = 0;
                        printf("s index: %d\n" ,s);
                        for (int y = 0; y < NEW_HEIGHT + 2*BORDER_SIZE; ++y) {
                            for (int k2 = 0; k2 < NUM_K/BATCH_K; ++k2) {
                                for (int x = 0; x < NEW_WIDTH+ 2*BORDER_SIZE; ++x) {
                                    for (int k1 = 0; k1 < BATCH_K; ++k1) {

                                        //OFFSET5(i,s,y,x,k, N,S,img_height,img_width,K)
                                        //int interleaved_offset = OFFSET8(0, i,s,y,x_outer, k_outer, x_inner, k_inner,
                                        //								1, N,S,img_height, BATCH_PIXELS_X, K / BATCH_K, img_width / BATCH_PIXELS_X,  BATCH_K);

                                        float val = interleaved_images_cpu[OFFSET6(i,s,y,k2, x,k1, N,S,NEW_HEIGHT+ 2*BORDER_SIZE,NUM_K/BATCH_K, NEW_WIDTH + 2*BORDER_SIZE, BATCH_K )];
                                        printf("%d ", (int)val);
                                    }
                                }
                                printf("\n");
                            }
                        }
                        printf("\n");
                    }
                }
            }
            return interleaved_images_output;
        }
    };

    template <typename BlockIndexingT>
    class DAUConvBwdInputError {
        enum {
            // values from main block indexing sizes
            BATCH_FEATURES_SIZE = BlockIndexingT::BATCH_FEATURES_SIZE,
            NEW_WIDTH = BlockIndexingT::IMG_WIDTH,
            NEW_HEIGHT = BlockIndexingT::IMG_HEIGHT,

            PIXELS_INTERPOLATION_Dx = BlockIndexingT::PIXELS_INTERPOLATION_Dx,
            PIXELS_INTERPOLATION_Dy = BlockIndexingT::PIXELS_INTERPOLATION_Dy,

            // values specific for this kerne
            CUDA_THREADS = 256,

            // TILE_DIM_X * TILE_DIM_Y * TILE_DIM_IMAGE gets us to 512 of shared data per block (i.e. 2 kB)
            TILE_DIM_X = CUDA_THREADS/8,
            TILE_DIM_Y = BATCH_FEATURES_SIZE,
            TILE_DIM_IMAGE = BlockIndexingT::BATCH_IMAGES >= 8 ? 8 : 1
        };
        const int img_width;
        const int img_height;
        const int N;
        const int F;

        int new_img_parts_width;
        int new_img_parts_height;

        dim3 threadsPerBlock;
        dim3 numBlocks;

    public:

        DAUConvBwdInputError(const int img_width, const int img_height, const int N, const int F, int new_img_parts_width, int new_img_parts_height) :
                img_width(img_width), img_height(img_height), N(N), F(F), new_img_parts_width(new_img_parts_width), new_img_parts_height(new_img_parts_height)
        {
            threadsPerBlock = dim3 (TILE_DIM_X, TILE_DIM_Y, 1);

            numBlocks = dim3 ( ((int)ceil(img_width*img_height) + threadsPerBlock.x - 1) / threadsPerBlock.x,	// over image width and height
                               ((int)ceil((float)BATCH_FEATURES_SIZE) + threadsPerBlock.y - 1) / threadsPerBlock.y,												// over BATCH_FEATURES_SIZE
                               ((int)ceil((N*F/BATCH_FEATURES_SIZE)/(float)TILE_DIM_IMAGE) + threadsPerBlock.z - 1) / threadsPerBlock.z);					// over N * F/BATCH_FEATURES_SIZE
        }

        size_t get_allocation_size() {
            return sizeof(float) * (NEW_WIDTH+PIXELS_INTERPOLATION_Dx-1) * (NEW_HEIGHT+PIXELS_INTERPOLATION_Dy-1) * F * (N+1) * new_img_parts_width * new_img_parts_height;
        }

        float* create_input(float* interleaved_error_output, const float* error_images, const bool ignore_edge_gradients, cudaStream_t streamId = NULL) {

            interleave_error_data_kernel<TILE_DIM_X,TILE_DIM_Y,TILE_DIM_IMAGE, BATCH_FEATURES_SIZE, NEW_WIDTH, NEW_HEIGHT, PIXELS_INTERPOLATION_Dx, PIXELS_INTERPOLATION_Dy><<<numBlocks,threadsPerBlock, 0, streamId>>>(error_images, interleaved_error_output, N, F, img_width, img_height, new_img_parts_width, new_img_parts_height, ignore_edge_gradients);

            if (0){
                // DEBUG ONLY

                // check correctness of interleaved data
                //float* filtered_images_cpu = new float[img_width * img_height * F *  N];
                float* interleaved_images_cpu = new float[get_allocation_size()/sizeof(float)];

                for (int i = 0; i < get_allocation_size()/sizeof(float); ++i)
                    interleaved_images_cpu[i] = -1;

                cudaMemcpy(interleaved_images_cpu, interleaved_error_output, get_allocation_size(), cudaMemcpyDeviceToHost);
                //cudaMemcpy(filtered_images_cpu, error_images, sizeof(int)* img_width * img_height * F *  N, cudaMemcpyDeviceToHost);
                cudaDeviceSynchronize();

                for (int n = 0; n < N*new_img_parts_width * new_img_parts_height; ++n) {
                    printf("img index: %d\n" ,n);
                    //for (int f = 0; f < F; ++f)
                    int f = 0;
                    {
                        for (int y = 0; y < NEW_HEIGHT+PIXELS_INTERPOLATION_Dy-1; ++y) {
                            for (int x = 0; x < NEW_WIDTH+PIXELS_INTERPOLATION_Dx-1; ++x) {
                                for (int k = 0; k < BATCH_FEATURES_SIZE; ++k) {
                                    float interleaved_val = interleaved_images_cpu[OFFSET5(n, f, y, x, k, N,F/BATCH_FEATURES_SIZE, NEW_HEIGHT+PIXELS_INTERPOLATION_Dy-1, NEW_WIDTH+PIXELS_INTERPOLATION_Dx-1, BATCH_FEATURES_SIZE)];
                                    printf("%d ", (int)interleaved_val);
                                }
                            }
                            printf("\n");
                        }
                    }
                    printf("\n\n");
                }
            }
            return interleaved_error_output;
        }
    };



    template <typename BlockIndexingT>
    class DAUConvBwdInputWeightAndOffsets {
        enum {
            // values from main block indexing sizes
            BATCH_FEATURES_SIZE = BlockIndexingT::BATCH_FEATURES_SIZE,
            NUM_BATCH_FEATURES =  BlockIndexingT::BATCH_COMPUTE_FEATURES_SIZE >= 4 ? 4 :
                                  (BlockIndexingT::BATCH_COMPUTE_FEATURES_SIZE >= 2 ? 2 : 1),

            OFFSET_BLOCK_MEM_SIZE = BlockIndexingT::BLOCK_SUBFEATURES * BlockIndexingT::BATCH_MEM_SUBFEATURES_SIZE * BlockIndexingT::BATCH_GAUSS_SIZE * BATCH_FEATURES_SIZE *  BlockIndexingT::BLOCK_FEATURES,
            WEIGHT_BLOCK_MEM_SIZE = BlockIndexingT::BLOCK_SUBFEATURES * BlockIndexingT::BATCH_MEM_SUBFEATURES_SIZE * BlockIndexingT::BATCH_GAUSS_SIZE * BlockIndexingT::PIXELS_INTERPOLATION_Dx*BlockIndexingT::PIXELS_INTERPOLATION_Dy * BATCH_FEATURES_SIZE * BlockIndexingT::BLOCK_FEATURES,

            // values specific for this kernel
            CUDA_THREADS = 256,

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

        DAUConvBwdInputWeightAndOffsets(const int img_width, const int img_height, const int N, const int F, const int S, const int G) :
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
            return sizeof(float) * ( 4*S*G*F + WEIGHT_BLOCK_MEM_SIZE);
        }

        void create_input(float* prepared_filter_weights, int* prepared_filter_offsets, // OUTPUT
                          const float* filter_weights, const float* filter_offsets_float_x, const float* filter_offsets_float_y, // INPUT
                          const int kernel_w, const int kernel_h, const bool offsets_already_centered, cudaStream_t streamId = NULL) {

            if (NUM_BATCH_FEATURES == 4)
                perpare_weights_and_offsets_bw_multi<BlockIndexingT, float4, int4><<<numBlocks,threadsPerBlock, 0, streamId>>>(filter_weights, filter_offsets_float_x, filter_offsets_float_y, prepared_filter_weights, prepared_filter_offsets, S, G, F, kernel_w, kernel_h, offsets_already_centered);
            else if (NUM_BATCH_FEATURES == 2)
                perpare_weights_and_offsets_bw_multi<BlockIndexingT, float2, int2><<<numBlocks,threadsPerBlock, 0, streamId>>>(filter_weights, filter_offsets_float_x, filter_offsets_float_y, prepared_filter_weights, prepared_filter_offsets, S, G, F, kernel_w, kernel_h, offsets_already_centered);
            else
                perpare_weights_and_offsets_bw_multi<BlockIndexingT, float, int><<<numBlocks,threadsPerBlock, 0, streamId>>>(filter_weights, filter_offsets_float_x, filter_offsets_float_y, prepared_filter_weights, prepared_filter_offsets, S, G, F, kernel_w, kernel_h, offsets_already_centered);

            if (0) {
                // DEBUG ONLY

                static const int NUM_READ_FEATURES =  BlockIndexingT::BATCH_COMPUTE_FEATURES_SIZE >= 4 ? 4 :
                                                      (BlockIndexingT::BATCH_COMPUTE_FEATURES_SIZE >= 2 ? 2 : 1);

                static const int dim1_size = BlockIndexingT::BLOCK_FEATURES;
                static const int dim2_size = BlockIndexingT::BATCH_FEATURES_SIZE/NUM_READ_FEATURES;
                static const int dim3_size = BlockIndexingT::PIXELS_INTERPOLATION_Dx * BlockIndexingT::PIXELS_INTERPOLATION_Dy;
                static const int dim4_size = BlockIndexingT::BATCH_GAUSS_SIZE;
                static const int dim5_size = BlockIndexingT::BATCH_MEM_SUBFEATURES_SIZE * BlockIndexingT::BLOCK_SUBFEATURES;

                int dim6_size = G / dim4_size;
                int dim7_size = S / dim5_size;
                int dim8_size = F / (dim1_size * dim2_size * NUM_READ_FEATURES);

                size_t num_el_off = this->get_offsets_allocation_size()/sizeof(float);
                size_t num_el_w = this->get_weights_allocation_size()/sizeof(float);

                int* offsets_cpu = new int[num_el_off];
                float* weights_cpu = new float[num_el_w];

                for (int i = 0; i < num_el_off; ++i) offsets_cpu[i] = -1;
                for (int i = 0; i < num_el_w; ++i) weights_cpu[i] = -1;

                cudaMemcpy(offsets_cpu, prepared_filter_offsets, this->get_offsets_allocation_size(), cudaMemcpyDeviceToHost);
                cudaMemcpy(weights_cpu, prepared_filter_weights, this->get_weights_allocation_size(), cudaMemcpyDeviceToHost);
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
                                                std::cout << offsets_cpu[index] << " ";
                                                index++;
                                            }
                                        }
                                    }
                                    printf("\n");
                                }
                            }
                        }
                    }
                }
                index =0;
                for (int f_main_idx = 0; f_main_idx < dim8_size; ++f_main_idx) {
                    for (int s_main_idx = 0; s_main_idx < dim7_size; ++s_main_idx) {
                        for (int g_main_idx = 0; g_main_idx < dim6_size; ++g_main_idx) {
                            printf("weights: (s,g,f=%d,%d,%d)\n", s_main_idx, g_main_idx, f_main_idx);
                            for (int sh_mem_idx = 0; sh_mem_idx < dim5_size; ++sh_mem_idx) {
                                for (int g = 0; g < dim4_size; ++g) {
                                    printf("sh=%d, g=%d: \n", sh_mem_idx, g);
                                    for (int px = 0; px < dim3_size; ++px) {
                                        printf("px=%d:", px);
                                        for (int f_batch = 0; f_batch < dim2_size; ++f_batch) {
                                            for (int f_block = 0; f_block < dim1_size; ++f_block) {
                                                for (int ff = 0; ff < NUM_READ_FEATURES; ++ff) {
                                                    std::cout << weights_cpu[index] << " ";
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

    template<int _IMG_SIZE_W, int _IMG_SIZE_H, int _MAX_OFFSET, int _NUM_K, int _BATCH_K_SIZE, int _WARP_PIXELS_X, int _WARP_PIXELS_Y, int _BATCH_IMAGES, bool _USE_INTERPOLATION, bool _SINGLE_SUBFEATURE>
    class DAUConvBackwardCUDA {
        enum {
            // Variable parameters

            // IMG_WIDTH and IMG_HEIGHT: 	32x32 .. N and M > 32
            // 								16x16 .. otherwise

            // current implementation allows min 32px in width, but this can be reduced by setting BATCH_K_SIZE=2 and NUM_K=4
            // however, this has not been extensively tested !!

            // BATCH_IMAGES :	256	.. N > 256
            // 					128	.. N > 128
            // 					32	.. N > 32
            // 					1	.. N > 1

            // MAX_OFFSET:	4 if kernels <= 9
            //				8 if kernels <= 17
            //				16 if kernels <= 33

            // BATCH_FEATURES_SIZE * BLOCK_FEATURES:  	16 min allowed
            // BLOCK_SUBFEATURES:  	2 min allowed
            // BATCH_GAUSS_SIZE:	2 min allowed

            // special cases for:
            //	- BATCH_GAUSS_SIZE == 1
            //	- INTERPOLATION == false

            WARP_PIXELS_X = _WARP_PIXELS_X,
            WARP_PIXELS_Y = _WARP_PIXELS_Y,

            IMG_WIDTH = MAX(WARP_PIXELS_X,_IMG_SIZE_W), // NOTE: 32 <= BLOCK_X * BATCH_PIXELS_SIZE_X
            IMG_HEIGHT = MAX(WARP_PIXELS_Y,_IMG_SIZE_H), // NOTE:  8 <= BLOCK_Y * BATCH_PIXELS_SIZE_Y
            MAX_OFFSET = _MAX_OFFSET,
            BATCH_IMAGES = _BATCH_IMAGES,

            PIXELS_INTERPOLATION_Dx = _USE_INTERPOLATION ? 2 : 1,
            PIXELS_INTERPOLATION_Dy = _USE_INTERPOLATION ? 2 : 1,


            // each block of multiple threads handles:
            //  - pixel:        BLOCK_X * BLOCK_Y
            //  - features:     BLOCK_FEATURES * BATCH_FEATURES_SIZE
            //  - subfeatures:  BLOCK_SUBFEATURES * BATCH_MEM_SUBFEATURES_SIZE
            //  - gauss krn:    BATCH_GAUSS_SIZE

            // within block each thread handles:
            //  - pixels:       BATCH_PIXELS_SIZE_X * BATCH_PIXELS_SIZE_Y
            //  - features:     BATCH_FEATURES_SIZE
            //  - subfeatures:  BLOCK_SUBFEATURES * BATCH_MEM_SUBFEATURES_SIZE
            //  - gauss krn:    BATCH_GAUSS_SIZE

            // each thread handles features and subfeatures as:
            //  - features:     one warp always handles only BATCH_FEATURES_SIZE features, but N warps are used for different features where N=BLOCK_FEATURES
            //  - subfeatures:  at once only BATCH_MEM_SUBFEATURES_SIZE subfeatures are loaded, but N-times iterated over that where N=BLOCK_SUBFEATURES

            // Fixed parameters
            NUM_SM = 1, // number of streaming multiprocessors

            // we are processing with K subfeature types (w,mu1,mu2,sigma) that will all have the same offsets and use the same error data
            NUM_K = _NUM_K,

            // number of K elements to load at once (we use only LDS.64 since it only leads to two-way bank conflict which has minor penalty
            BATCH_K_SIZE = _BATCH_K_SIZE,

            BATCH_PIXELS_SIZE_X = 1,
            BATCH_PIXELS_SIZE_Y = _WARP_PIXELS_Y,

            BLOCK_X = WARP_PIXELS_X / BATCH_PIXELS_SIZE_X,
            BLOCK_Y = WARP_PIXELS_Y / BATCH_PIXELS_SIZE_Y,

            BLOCK_FEATURES = 8 * MAX(1,MIN(2,WARP_SIZE / WARP_PIXELS_X)), // increase number of BLOCK_FEATURES to match WARP_PIXELS to WARP_SIZE
            BLOCK_SUBFEATURES = _SINGLE_SUBFEATURE ? 1 : 2  ,

            BATCH_FEATURES_SIZE = 2,
            BATCH_COMPUTE_FEATURES_SIZE = 2,
            BATCH_MEM_SUBFEATURES_SIZE = 1,
            BATCH_GAUSS_SIZE = 2,

        };
        const int img_width,img_height;
        const int I,S,F,G,K, IN_K;

        int new_img_parts_width;
        int new_img_parts_height;

        dim3 threadsPerBlock;
        dim3 numBlocks;
    public:

        typedef BlockIndexing<NUM_SM,
                BLOCK_X, BLOCK_Y, BLOCK_FEATURES, BLOCK_SUBFEATURES,
                BATCH_PIXELS_SIZE_X, BATCH_PIXELS_SIZE_Y,
                BATCH_IMAGES,
                NUM_K, BATCH_K_SIZE,
                PIXELS_INTERPOLATION_Dx, PIXELS_INTERPOLATION_Dy,
                BATCH_FEATURES_SIZE,
                BATCH_COMPUTE_FEATURES_SIZE,
                BATCH_MEM_SUBFEATURES_SIZE,
                BATCH_GAUSS_SIZE,
                IMG_WIDTH, IMG_HEIGHT,
                MAX_OFFSET> BlockIndexingPipelineT;

        DAUConvBwdInputImage<BlockIndexingPipelineT> image_cuda_prepare;
        DAUConvBwdInputError<BlockIndexingPipelineT> error_cuda_prepare;
        DAUConvBwdInputWeightAndOffsets<BlockIndexingPipelineT> weight_and_offsets_cuda_prepare;

        DAUConvBackwardCUDA(const DAUConvBackward<float>::CUDAParams& p) :
                img_width(p.img_width), img_height(p.img_height), I(p.I), S(p.S), F(p.F), G(p.G), K(p.K), IN_K(p.IN_K),

                // we will split image into patches of size [IMG_HEIGHT x IMG_WIDTH] so use that as image size, however,
                // we need to increase the number of images that will be process as each patch is now considered as one image
                // there is no need to recombine the output since we just sum over all patches to get gradients

                new_img_parts_width((int)ceil((float)img_width / IMG_WIDTH)),
                new_img_parts_height((int)ceil((float)img_height / IMG_HEIGHT)),

                // initialize classes that will generate inputs
                image_cuda_prepare(p.img_width_in, p.img_height_in, img_width, img_height, I, S, K, IN_K, new_img_parts_width,new_img_parts_height),
                error_cuda_prepare(img_width, img_height, I, F, new_img_parts_width,new_img_parts_height),
                weight_and_offsets_cuda_prepare(img_width, img_height, I, F, S, G) {

            if (NUM_K != K) {
                printf("Invalid input K %d in DAUConvBackwardCUDA. Only a value of %d supported.\n", K, NUM_K);
                throw std::exception();
            }

            class BlockIndexingPipelineT::Launch block_indexing;

            threadsPerBlock = block_indexing.getThreadsPerBlock(I * new_img_parts_width * new_img_parts_height, F, S, IMG_WIDTH, IMG_HEIGHT);
            numBlocks = block_indexing.getBlocksPerGrid(I * new_img_parts_width * new_img_parts_height, F, S, G, IMG_WIDTH, IMG_HEIGHT);
        }

        void get_allocation_sizes(DAUConvBackward<float>::CUDAParams& p) {

            if (p.alloc_img != NULL) *p.alloc_img = image_cuda_prepare.get_allocation_size();
            if (p.alloc_err != NULL) *p.alloc_err = error_cuda_prepare.get_allocation_size();
            if (p.alloc_w != NULL) *p.alloc_w = weight_and_offsets_cuda_prepare.get_weights_allocation_size();
            if (p.alloc_off != NULL) *p.alloc_off = weight_and_offsets_cuda_prepare.get_offsets_allocation_size();
        }

        void run_kernel(DAUConvBackward<float>::CUDAParams& p) {

            //CUDA_CHECK(cudaMemsetAsync(p.prepared_error_images, 0, error_cuda_prepare.get_allocation_size(), p.streamId));

            {
//#define PROFILE_CUDA
#ifdef PROFILE_CUDA
                std::cout << "started create_input_with_border_bw" << std::endl;

			clock_t start_t = clock();
#endif
                p.prepared_filtered_images = image_cuda_prepare.create_input(p.prepared_filtered_images, p.filtered_images, p.streamId);
#ifdef PROFILE_CUDA
                cudaDeviceSynchronize();

			clock_t end_t = clock();
			CUDA_POST_KERNEL_CHECK;
			std::cout << "create_input_with_border_bw_multi in " << (((float)(end_t-start_t))/CLOCKS_PER_SEC) << std::endl;
#endif
            }

            {
#ifdef PROFILE_CUDA
                clock_t start_t = clock();
#endif
                p.prepared_error_images = error_cuda_prepare.create_input(p.prepared_error_images, p.error_images, p.ignore_edge_gradients, p.streamId);
#ifdef PROFILE_CUDA
                cudaDeviceSynchronize();

			clock_t end_t = clock();
			CUDA_POST_KERNEL_CHECK;
			std::cout << "create_error_bw_multi in " << (((float)(end_t-start_t))/CLOCKS_PER_SEC) << std::endl;
#endif
            }

            {
#ifdef PROFILE_CUDA
                std::cout << "started copy_permute_weights" << std::endl;

			clock_t start_t = clock();
#endif
                weight_and_offsets_cuda_prepare.create_input(p.prepared_filter_weights, p.prepared_filter_offsets, p.filter_weights, p.filter_offsets_float_x, p.filter_offsets_float_y, p.kernel_w, p.kernel_h, p.offsets_already_centered, p.streamId);
#ifdef PROFILE_CUDA
                cudaDeviceSynchronize();

			clock_t end_t = clock();
			CUDA_POST_KERNEL_CHECK;

			std::cout << "copy_permute_weights in " << (((float)(end_t-start_t))/CLOCKS_PER_SEC) << std::endl;
#endif
            }
#ifdef PROFILE_CUDA
            std::cout << "started DAUConv_backward_multi_pipeline_kernel" << std::endl;

		std::cout << "threadsPerBlock " << threadsPerBlock.x << "," << threadsPerBlock.y << "," << threadsPerBlock.z << std::endl;
		std::cout << "numBlocks " << numBlocks.x << "," << numBlocks.y << "," << numBlocks.z << std::endl;

		for (int jj = 0; jj < 1; ++jj) {

			clock_t start_t = clock();
#endif
            DAUConv_bwd_multi_pipeline_kernel < BlockIndexingPipelineT><<<numBlocks,threadsPerBlock, 0, p.streamId>>>(p.prepared_filtered_images, p.prepared_error_images, p.prepared_filter_offsets, p.prepared_filter_weights, p.output, I * new_img_parts_width * new_img_parts_height, S, F, G, K, img_width, img_height);
#ifdef PROFILE_CUDA
            cudaDeviceSynchronize();

			clock_t end_t = clock();
			CUDA_POST_KERNEL_CHECK;

			std::cout << "DAUConv_backward_multi_pipeline_kernel in " << (((float)(end_t-start_t))/CLOCKS_PER_SEC) << std::endl;
		}
#endif

        }
    };

#ifdef DAU_USE_DUMMY_CUDA_IMPL
#define RUN_KERNEL_R0(CLASS_NAME, IMG_PATCH_SIZE_W, IMG_PATCH_SIZE_H, MAX_OFFSET, NUM_K, BATCH_K_SIZE, WARP_PIXELS_X, WARP_PIXELS_Y, BATCH_IMAGES, USE_INTERPOLATION, SINGLE_SUBFEATURE, PARAMS, ...);
#else
#define RUN_KERNEL_R0(CLASS_NAME, IMG_PATCH_SIZE_W, IMG_PATCH_SIZE_H, MAX_OFFSET, NUM_K, BATCH_K_SIZE, WARP_PIXELS_X, WARP_PIXELS_Y, BATCH_IMAGES, USE_INTERPOLATION, SINGLE_SUBFEATURE, PARAMS, ...) \
    { \
        CLASS_NAME<IMG_PATCH_SIZE_W, IMG_PATCH_SIZE_H, MAX_OFFSET, NUM_K, BATCH_K_SIZE, WARP_PIXELS_X, WARP_PIXELS_Y, BATCH_IMAGES, USE_INTERPOLATION, SINGLE_SUBFEATURE> _kernel_class(PARAMS); \
        if (PARAMS.alloc_img != NULL ||    \
            PARAMS.alloc_err != NULL ||        \
            PARAMS.alloc_w != NULL ||        \
            PARAMS.alloc_off != NULL) {    \
            _kernel_class.get_allocation_sizes(PARAMS); \
        } else { \
            _kernel_class.run_kernel(PARAMS); \
        } \
    }
#endif

#define RUN_KERNEL_R1(CLASS_NAME, IMG_PATCH_SIZE_W, IMG_PATCH_SIZE_H, MAX_OFFSET, NUM_K, BATCH_K_SIZE, WARP_PIXELS_X, WARP_PIXELS_Y, BATCH_IMAGES, USE_INTERPOLATION, SINGLE_SUBFEATURE, PARAMS, ...) \
    if (USE_INTERPOLATION) { \
        RUN_KERNEL_R0(CLASS_NAME, IMG_PATCH_SIZE_W, IMG_PATCH_SIZE_H, MAX_OFFSET, NUM_K, BATCH_K_SIZE, WARP_PIXELS_X, WARP_PIXELS_Y, BATCH_IMAGES, true, SINGLE_SUBFEATURE, PARAMS, __VA_ARGS__) \
    } else { \
        /*RUN_KERNEL_R0(CLASS_NAME, IMG_PATCH_SIZE_W, IMG_PATCH_SIZE_H, MAX_OFFSET, NUM_K, BATCH_K_SIZE, WARP_PIXELS_X, WARP_PIXELS_Y, BATCH_IMAGES, false, SINGLE_SUBFEATURE, PARAMS, __VA_ARGS__)*/ \
		printf("Support for non-interpolation currently disabled. Non-interpolation has not been extensivly tested so disabling support.\n"); \
        throw std::exception(); \
    }


#define RUN_KERNEL_R2(CLASS_NAME, IMG_PATCH_SIZE_W, IMG_PATCH_SIZE_H, MAX_OFFSET, NUM_K, BATCH_K_SIZE, WARP_PIXELS_X, WARP_PIXELS_Y, BATCH_IMAGES, USE_INTERPOLATION, SINGLE_SUBFEATURE, PARAMS, ...) \
	if (MAX_OFFSET <= 9) { \
	  RUN_KERNEL_R1(CLASS_NAME, IMG_PATCH_SIZE_W, IMG_PATCH_SIZE_H, 4, NUM_K, BATCH_K_SIZE, WARP_PIXELS_X, WARP_PIXELS_Y, BATCH_IMAGES, USE_INTERPOLATION, SINGLE_SUBFEATURE, PARAMS, __VA_ARGS__)  \
	} else if (MAX_OFFSET <= 17) { \
        RUN_KERNEL_R1(CLASS_NAME, IMG_PATCH_SIZE_W, IMG_PATCH_SIZE_H, 8, NUM_K, BATCH_K_SIZE, WARP_PIXELS_X, WARP_PIXELS_Y, BATCH_IMAGES, USE_INTERPOLATION, SINGLE_SUBFEATURE, PARAMS, __VA_ARGS__) \
	} else if (MAX_OFFSET <= 33) { \
        RUN_KERNEL_R1(CLASS_NAME, IMG_PATCH_SIZE_W, IMG_PATCH_SIZE_H, 16, 3, 1, 16, 8, BATCH_IMAGES, USE_INTERPOLATION, SINGLE_SUBFEATURE, PARAMS, __VA_ARGS__) \
	} else if (MAX_OFFSET <= 65) { \
        RUN_KERNEL_R1(CLASS_NAME, IMG_PATCH_SIZE_W, IMG_PATCH_SIZE_H, 32, 1, 1, 16, 8, BATCH_IMAGES, USE_INTERPOLATION, SINGLE_SUBFEATURE, PARAMS, __VA_ARGS__) \
	} else { \
        printf("Unsupported filter size: %d. Supported only max up to 9x9, 17x17, 33x33 and 65x65 at the moment\n", MAX_OFFSET); \
        throw std::exception(); \
    }
    /*else if (MAX_OFFSET <= 33) { \
        RUN_KERNEL_R2(CLASS_NAME, IMG_PATCH_SIZE, 16, BATCH_IMAGES, USE_INTERPOLATION, __VA_ARGS__) \
    */

#define RUN_KERNEL_R3(CLASS_NAME, IMG_PATCH_SIZE_W, IMG_PATCH_SIZE_H, MAX_OFFSET, NUM_K, BATCH_K_SIZE, WARP_PIXELS_X, WARP_PIXELS_Y, BATCH_IMAGES, USE_INTERPOLATION, SINGLE_SUBFEATURE, PARAMS, ...) \
	if (BATCH_IMAGES >= 128) { \
        RUN_KERNEL_R2(CLASS_NAME, IMG_PATCH_SIZE_W, IMG_PATCH_SIZE_H, MAX_OFFSET, NUM_K, BATCH_K_SIZE, WARP_PIXELS_X, WARP_PIXELS_Y, 128, USE_INTERPOLATION, SINGLE_SUBFEATURE, PARAMS, __VA_ARGS__) \
	} else if (BATCH_IMAGES >= 16) { \
        RUN_KERNEL_R2(CLASS_NAME, IMG_PATCH_SIZE_W, IMG_PATCH_SIZE_H, MAX_OFFSET, NUM_K, BATCH_K_SIZE, WARP_PIXELS_X, WARP_PIXELS_Y, 16, USE_INTERPOLATION, SINGLE_SUBFEATURE, PARAMS, __VA_ARGS__) \
	} else { \
        RUN_KERNEL_R2(CLASS_NAME, IMG_PATCH_SIZE_W, IMG_PATCH_SIZE_H, MAX_OFFSET, NUM_K, BATCH_K_SIZE, WARP_PIXELS_X, WARP_PIXELS_Y, 1, USE_INTERPOLATION, SINGLE_SUBFEATURE, PARAMS, __VA_ARGS__) \
	}
    /*else if (BATCH_IMAGES >= 32) { \
        RUN_KERNEL_R2(CLASS_NAME, IMG_PATCH_SIZE, MAX_OFFSET, 32, USE_INTERPOLATION, IMG_WIDTH, IMG_HEIGHT, I, S, F, G, K, __VA_ARGS__) \
    }*/

#define RUN_KERNEL_R4(CLASS_NAME, IMG_PATCH_SIZE_W, IMG_PATCH_SIZE_H, MAX_OFFSET, NUM_K, BATCH_K_SIZE, WARP_PIXELS_X, WARP_PIXELS_Y, BATCH_IMAGES, USE_INTERPOLATION, SINGLE_SUBFEATURE, PARAMS, ...) \
    if (SINGLE_SUBFEATURE) { \
        RUN_KERNEL_R3(CLASS_NAME, IMG_PATCH_SIZE_W, IMG_PATCH_SIZE_H, MAX_OFFSET, NUM_K, BATCH_K_SIZE, WARP_PIXELS_X, WARP_PIXELS_Y, BATCH_IMAGES, USE_INTERPOLATION, true, PARAMS, __VA_ARGS__) \
    } else { \
        RUN_KERNEL_R3(CLASS_NAME, IMG_PATCH_SIZE_W, IMG_PATCH_SIZE_H, MAX_OFFSET, NUM_K, BATCH_K_SIZE, WARP_PIXELS_X, WARP_PIXELS_Y, BATCH_IMAGES, USE_INTERPOLATION, false, PARAMS, __VA_ARGS__) \
    }

#define RUN_KERNEL_R5(CLASS_NAME, IMG_PATCH_SIZE_W, IMG_PATCH_SIZE_H, MAX_OFFSET, SMALLER_WARP_AND_GROUP_K, BATCH_IMAGES, USE_INTERPOLATION, SINGLE_SUBFEATURE, PARAMS, ...) \
    if (SMALLER_WARP_AND_GROUP_K) { \
        if (IMG_PATCH_SIZE_W <= 8) { \
            RUN_KERNEL_R4(CLASS_NAME, IMG_PATCH_SIZE_W, IMG_PATCH_SIZE_H, MAX_OFFSET, 4, 4, 8, 8, BATCH_IMAGES, USE_INTERPOLATION, SINGLE_SUBFEATURE, PARAMS, __VA_ARGS__) \
        } else { \
            RUN_KERNEL_R4(CLASS_NAME, IMG_PATCH_SIZE_W, IMG_PATCH_SIZE_H, MAX_OFFSET, 4, 2, 16, 8, BATCH_IMAGES, USE_INTERPOLATION, SINGLE_SUBFEATURE, PARAMS, __VA_ARGS__) \
        } \
    } else { \
        RUN_KERNEL_R4(CLASS_NAME, IMG_PATCH_SIZE_W, IMG_PATCH_SIZE_H, MAX_OFFSET, 3, 1, 32, 8, BATCH_IMAGES, USE_INTERPOLATION, SINGLE_SUBFEATURE, PARAMS, __VA_ARGS__) \
    }
// NOTE: RUN_KERNEL_R6 and RUN_KERNEL_R7 below are not called directly - instead they are implemented in seperate files to allow for parallel computation
#define RUN_KERNEL_R6(CLASS_NAME, IMG_PATCH_SIZE_W, IMG_PATCH_SIZE_H, MAX_OFFSET, SMALLER_WARP_AND_GROUP_K, BATCH_IMAGES, USE_INTERPOLATION, SINGLE_SUBFEATURE, PARAMS, ...) \
	if (IMG_PATCH_SIZE_W >= 64) { \
		RUN_KERNEL_R5(CLASS_NAME, 64, IMG_PATCH_SIZE_H, MAX_OFFSET, SMALLER_WARP_AND_GROUP_K, BATCH_IMAGES, USE_INTERPOLATION, SINGLE_SUBFEATURE, PARAMS, __VA_ARGS__) \
	} else if (IMG_PATCH_SIZE_W >= 32) { \
        RUN_KERNEL_R5(CLASS_NAME, 16, IMG_PATCH_SIZE_H, MAX_OFFSET, SMALLER_WARP_AND_GROUP_K, BATCH_IMAGES, USE_INTERPOLATION, SINGLE_SUBFEATURE, PARAMS, __VA_ARGS__) \
    } else if (IMG_PATCH_SIZE_W >= 16) { \
        RUN_KERNEL_R5(CLASS_NAME, 16, IMG_PATCH_SIZE_H, MAX_OFFSET, SMALLER_WARP_AND_GROUP_K, BATCH_IMAGES, USE_INTERPOLATION, SINGLE_SUBFEATURE, PARAMS, __VA_ARGS__) \
    } else { \
        RUN_KERNEL_R5(CLASS_NAME, 8, IMG_PATCH_SIZE_H, MAX_OFFSET, SMALLER_WARP_AND_GROUP_K, BATCH_IMAGES, USE_INTERPOLATION, SINGLE_SUBFEATURE, PARAMS, __VA_ARGS__) \
    }

#define RUN_KERNEL_R7(CLASS_NAME, IMG_PATCH_SIZE_W, IMG_PATCH_SIZE_H, MAX_OFFSET, SMALLER_WARP_AND_GROUP_K, BATCH_IMAGES, USE_INTERPOLATION, SINGLE_SUBFEATURE, PARAMS, ...) \
    if (IMG_PATCH_SIZE_H >= 64) { \
        RUN_KERNEL_R6(CLASS_NAME, IMG_PATCH_SIZE_W, 64, MAX_OFFSET, SMALLER_WARP_AND_GROUP_K, BATCH_IMAGES, USE_INTERPOLATION, SINGLE_SUBFEATURE, PARAMS, __VA_ARGS__) \
    } else if (IMG_PATCH_SIZE_H >= 32) { \
        RUN_KERNEL_R6(CLASS_NAME, IMG_PATCH_SIZE_W, 32, MAX_OFFSET, SMALLER_WARP_AND_GROUP_K, BATCH_IMAGES, USE_INTERPOLATION, SINGLE_SUBFEATURE, PARAMS, __VA_ARGS__) \
    } else if (IMG_PATCH_SIZE_H >= 16) { \
        RUN_KERNEL_R6(CLASS_NAME, IMG_PATCH_SIZE_W, 16, MAX_OFFSET, SMALLER_WARP_AND_GROUP_K, BATCH_IMAGES, USE_INTERPOLATION, SINGLE_SUBFEATURE, PARAMS, __VA_ARGS__) \
    } else {\
        RUN_KERNEL_R6(CLASS_NAME, IMG_PATCH_SIZE_W, 8, MAX_OFFSET, SMALLER_WARP_AND_GROUP_K, BATCH_IMAGES, USE_INTERPOLATION, SINGLE_SUBFEATURE, PARAMS, __VA_ARGS__) \
	}

}  // namespace caffe
