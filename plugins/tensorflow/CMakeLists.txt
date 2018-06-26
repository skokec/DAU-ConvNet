cmake_minimum_required(VERSION 2.8)
set(BUILD_TENSORFLOW_PLUGIN ON)
if (NOT BUILD_TENSORFLOW_PLUGIN)
    MESSAGE("BUILD_TENSORFLOW_PLUGIN not set.")
    return()
endif()

execute_process(COMMAND python3.5 -c "import tensorflow; print(tensorflow.sysconfig.get_include(), end='')" OUTPUT_VARIABLE Tensorflow_INCLUDE_DIRS)

# C++11 required for tensorflow
execute_process(COMMAND python3.5 -c "import tensorflow as tf; print(' '.join(tf.sysconfig.get_compile_flags()),end='')" OUTPUT_VARIABLE TF_CFLAGS)
execute_process(COMMAND python3.5 -c "import tensorflow as tf; print(' '.join(tf.sysconfig.get_link_flags()),end='')" OUTPUT_VARIABLE TF_LFLAGS)
execute_process(COMMAND python3.5 -c "import tensorflow as tf; print(tf.sysconfig.get_lib()+'/libtensorflow_framework.so',end='')" OUTPUT_VARIABLE TF_LIB)
#TF_LFLAGS=( $(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))') )

set(Tensorflow_INCLUDE_DIRS "/home/vitjan/.local/lib/python3.5/site-packages/tensorflow/include")
set(TF_LIB "/home/vitjan/.local/lib/python3.5/site-packages/tensorflow/libtensorflow_framework.so")
set(TF_CFLAGS "-I/home/vitjan/.local/lib/python3.5/site-packages/tensorflow/include -D_GLIBCXX_USE_CXX11_ABI=0")
set(TF_LFLAGS "-L/home/vitjan/.local/lib/python3.5/site-packages/tensorflow -ltensorflow_framework")

#-- flags:  -std=c++11 -fPIC -Wall -Wno-sign-compare -Wno-uninitialized -std=c++11 -D GOOGLE_CUDA=1 -I/home/vitjan/.local/lib/python3.5/site-packages/tensorflow/include -D_GLIBCXX_USE_CXX11_ABI=0
#-- flags:  -L/home/vitjan/.local/lib/python3.5/site-packages/tensorflow -ltensorflow_framework
#-- LINKER LIBS: PUBLIC;/usr/local/cuda/lib64/libcudart.so;/usr/local/cuda/lib64/libcurand.so;/usr/local/cuda/lib64/libcublas.so;/usr/local/cuda/lib64/libcublas_device.a;PUBLIC;opencv_core;opencv_highgui;opencv_imgproc;opencv_imgcodecs;opencv_core;opencv_highgui;opencv_imgproc;opencv_imgcodecs;PUBLIC;/usr/lib/liblapack.so;/usr/lib/libcblas.so;/usr/lib/libatlas.so
#-- INCLUDE DIRS: PUBLIC;/usr/local/cuda/include;PUBLIC;/usr/local/include;/usr/local/include/opencv;PUBLIC;/usr/include;/usr/include/atlas
#-- INCLUDE DIR: /home/vitjan/Documents/5.letnik/gauss_conv/new_impl/VitjanZ/DAU-ConvNet/include
#-- TENSORFLOW DIRS: /home/vitjan/.local/lib/python3.5/site-packages/tensorflow/include
#-- TENSORFLOW LIB: /home/vitjan/.local/lib/python3.5/site-packages/tensorflow/libtensorflow_framework.so
#-- tf link flags: -L/home/vitjan/.local/lib/python3.5/site-packages/tensorflow -ltensorflow_framework
#-- tf compiler flags: -I/home/vitjan/.local/lib/python3.5/site-packages/tensorflow/include -D_GLIBCXX_USE_CXX11_ABI=0


set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D GOOGLE_CUDA=1")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TF_CFLAGS}")
set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} ${TF_LFLAGS}")

#TEMP INSERT FROM BASE CMAKE

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake/Modules)

include(cmake/Utils.cmake)
include(cmake/Targets.cmake)
include(cmake/ConfigGen.cmake)
#include(cmake/Cuda.cmake)
include(cmake/Dependencies.cmake)



#################################
#if GCC > 5
#if (CMAKE_CXX_COMPILER_VERSION VERSION_GREATER 5.0 OR CMAKE_CXX_COMPILER_VERSION VERSION_EQUAL 5.0)
#  set(CMAKE_CXX_FLAGS "-D_GLIBCXX_USE_CXX11_ABI=0 ${CMAKE_CXX_FLAGS}")
#  message(STATUS "Set GLIBCXX_USE_CXX11_ABI=0")
#endif()


#g++ -std=c++11 -shared zero_out.cc -o zero_out.so -fPIC ${TF_CFLAGS[@]} ${TF_LFLAGS[@]} -O2
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TF_LFLAGS}")
message(STATUS "flags: ${CMAKE_CXX_FLAGS}")
message(STATUS "flags: ${CMAKE_SHARED_LINKER_FLAGS}")
message(STATUS "tf link flags: ${TF_LFLAGS}")
message(STATUS "tf compiler flags: ${TF_CFLAGS}")



#list(APPEND DAUConvNet_INCLUDE_DIRS PUBLIC ${Tensorflow_INCLUDE_DIRS})
#list(APPEND DAUConvNet_LINKER_LIBS PUBLIC tensorflow_framework)

message(STATUS "LINKER LIBS: ${DAUConvNet_LINKER_LIBS}")
message(STATUS "INCLUDE DIRS: ${DAUConvNet_INCLUDE_DIRS}")
message(STATUS "INCLUDE DIR: ${DAUConvNet_INCLUDE_DIR}")
message(STATUS "TENSORFLOW DIRS: ${Tensorflow_INCLUDE_DIRS}")
message(STATUS "TENSORFLOW LIB: ${TF_LIB}")




#add_subdirectory(../../src/dau_conv)

# build the gradient operation which is used in base_op_grad.py
# to register it
#LAYER ADD LIBRARY
#LINK_DIRECTORIES(${TF_LIB})
include_directories(${Tensorflow_INCLUDE_DIRS})
include_directories(/home/vitjan/Documents/5.letnik/gauss_conv/nsync/public)
add_library(dau_conv_tensorflow SHARED dau_conv_layer_tensorflow.cpp dau_conv_layer_tensorflow.hpp)

#target_include_directories(dau_conv_tensorflow PUBLIC "../../include/dau_conv/")
target_include_directories(dau_conv_tensorflow PUBLIC ${DAUConvNet_INCLUDE_DIR})
target_include_directories(dau_conv_tensorflow PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(dau_conv_tensorflow PUBLIC ${Tensorflow_INCLUDE_DIRS})
target_include_directories(dau_conv_tensorflow PUBLIC ${DAUConvNet_INCLUDE_DIRS})

#target_link_libraries(dau_conv_tensorflow ${DAUConvNet_LINKER_LIBS} ${TF_LIB} dau-conv)
target_link_libraries(dau_conv_tensorflow ${TF_LIB} dau-conv)

#target_link_libraries(dau_conv_tensorflow dau_conv)
#list(APPEND DAUConvNet_LINKER_LIBS PUBLIC dau_conv_tensorflow dau-conv)
############333




#LINK_DIRECTORIES(${TF_LIB})
# build the actual operation which can be used directory

add_library(base_op_grad SHARED base_op_grad.cpp)
#target_include_directories(base_op_grad PRIVATE "../../include/")
#target_include_directories(base_op_grad PUBLIC "/home/vitjan/Documents/5.letnik/gauss_conv/nsync/public")
target_link_libraries(base_op_grad dau_conv_tensorflow)
target_link_libraries(base_op_grad ${DAUConvNet_LINKER_LIBS})
target_include_directories(base_op_grad PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(base_op_grad PUBLIC "/usr/local/")
target_include_directories(base_op_grad ${DAUConvNet_INCLUDE_DIRS} PUBLIC ${DAUConvNet_INCLUDE_DIR})

#target_include_directories(base_op_grad PUBLIC ${Tensorflow_INCLUDE_DIRS})
#target_include_directories(base_op_grad PUBLIC ${DAUConvNet_INCLUDE_DIR})



#LINK_DIRECTORIES(${TF_LIB})
add_library(base_op SHARED base_op.cpp)

#target_link_libraries(base_op ${DAUConvNet_LINKER_LIBS} ${TF_LIB} dau_conv_tensorflow)
target_link_libraries(base_op dau_conv_tensorflow)
target_include_directories(base_op PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(base_op PUBLIC "/usr/local/")
target_include_directories(base_op ${DAUConvNet_INCLUDE_DIRS} PUBLIC ${DAUConvNet_INCLUDE_DIR})

#target_include_directories(base_op PUBLIC ${DAUConvNet_INCLUDE_DIR})



