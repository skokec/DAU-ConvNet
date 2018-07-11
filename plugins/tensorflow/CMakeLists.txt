cmake_minimum_required(VERSION 2.8)

if (NOT BUILD_TENSORFLOW_PLUGIN)
    MESSAGE("BUILD_TENSORFLOW_PLUGIN not set.")
    return()
endif()

#find_package (Python COMPONENTS Interpreter Development)
find_package( PythonInterp REQUIRED )
MESSAGE(STATUS "Python exec ${PYTHON_EXECUTABLE}")

#SET VARIABLE AS OTHER PYTHON EXECUTABLE IF PythonInterp FINDS A DIFFERENT EXECUTABLE
#set(PYTHON_EXECUTABLE python3)

#get TF VERSION
execute_process(COMMAND ${PYTHON_EXECUTABLE} -c "import tensorflow; import sys; sys.stdout.write(tensorflow.__version__)" OUTPUT_VARIABLE TF_VERSION)

if( ${TF_VERSION} VERSION_LESS "1.5.0")
    MESSAGE(STATUS "TF VER LOWER THAN 1.5.: ${TF_VERSION}")
    execute_process(COMMAND ${PYTHON_EXECUTABLE} -c "import tensorflow; import sys; sys.stdout.write(tensorflow.sysconfig.get_include())" OUTPUT_VARIABLE Tensorflow_INCLUDE_DIRS)
    execute_process(COMMAND ${PYTHON_EXECUTABLE} -c "import tensorflow as tf; import sys; sys.stdout.write(tf.sysconfig.get_lib()+'/libtensorflow_framework.so')" OUTPUT_VARIABLE TF_LIB)
    execute_process(COMMAND ${PYTHON_EXECUTABLE} -c "import tensorflow as tf; import sys; sys.stdout.write(tf.sysconfig.get_lib())" OUTPUT_VARIABLE TF_LIB_DIR)
    set(TF_CFLAGS "-I ${Tensorflow_INCLUDE_DIRS} -D_GLIBCXX_USE_CXX11_ABI=0")
    set(TF_LFLAGS "-L ${TF_LIB_DIR} -ltensorflow_framework")
else()
    MESSAGE(STATUS "TF VER: ${TF_VERSION}")
    execute_process(COMMAND ${PYTHON_EXECUTABLE} -c "import tensorflow; import sys; sys.stdout.write(tensorflow.sysconfig.get_include())" OUTPUT_VARIABLE Tensorflow_INCLUDE_DIRS)
    execute_process(COMMAND ${PYTHON_EXECUTABLE} -c "import tensorflow as tf; import sys; sys.stdout.write(' '.join(tf.sysconfig.get_compile_flags()))" OUTPUT_VARIABLE TF_CFLAGS)
    execute_process(COMMAND ${PYTHON_EXECUTABLE} -c "import tensorflow as tf; import sys; sys.stdout.write(' '.join(tf.sysconfig.get_link_flags()))" OUTPUT_VARIABLE TF_LFLAGS)
    execute_process(COMMAND ${PYTHON_EXECUTABLE} -c "import tensorflow as tf; import sys; sys.stdout.write(tf.sysconfig.get_lib()+'/libtensorflow_framework.so')" OUTPUT_VARIABLE TF_LIB)
endif()


set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D GOOGLE_CUDA=1")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TF_CFLAGS}")
set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} ${TF_LFLAGS}")

#TEMP INSERT FROM BASE CMAKE

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake/Modules)


message(STATUS "flags: ${CMAKE_CXX_FLAGS}")
message(STATUS "flags: ${CMAKE_SHARED_LINKER_FLAGS}")
message(STATUS "tf link flags: ${TF_LFLAGS}")
message(STATUS "tf compiler flags: ${TF_CFLAGS}")



message(STATUS "LINKER LIBS: ${DAUConvNet_LINKER_LIBS}")
message(STATUS "INCLUDE DIRS: ${DAUConvNet_INCLUDE_DIRS}")
message(STATUS "INCLUDE DIR: ${DAUConvNet_INCLUDE_DIR}")
message(STATUS "TENSORFLOW DIRS: ${Tensorflow_INCLUDE_DIRS}")
message(STATUS "TENSORFLOW LIB: ${TF_LIB}")


# build the gradient operation which is used in base_op_grad.py
# to register it
#LAYER ADD LIBRARY
#LINK_DIRECTORIES(${TF_LIB})
include_directories(${Tensorflow_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src)

add_library(dau_conv_tensorflow SHARED src/dau_conv_layer_tensorflow.cpp src/dau_conv_layer_tensorflow.hpp ${DAUConvNet_OBJS})

target_include_directories(dau_conv_tensorflow PUBLIC ${DAUConvNet_INCLUDE_DIR})
target_include_directories(dau_conv_tensorflow PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(dau_conv_tensorflow PUBLIC ${Tensorflow_INCLUDE_DIRS})
target_include_directories(dau_conv_tensorflow PUBLIC ${DAUConvNet_INCLUDE_DIRS})

message(STATUS ${DAUConvNet_LINKER_LIBS})
target_link_libraries(dau_conv_tensorflow PUBLIC ${TF_LIB} )
target_link_libraries(dau_conv_tensorflow ${DAUConvNet_LINKER_LIBS})



#LINK_DIRECTORIES(${TF_LIB})
# build the actual operation which can be used directory

add_library(dau_conv_grad_op SHARED src/dau_conv_grad_op.cpp)
target_link_libraries(dau_conv_grad_op PUBLIC dau_conv_tensorflow)
target_link_libraries(dau_conv_grad_op PUBLIC ${DAUConvNet_LINKER_LIBS})
target_include_directories(dau_conv_grad_op PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(dau_conv_grad_op PUBLIC "/usr/local/")
target_include_directories(dau_conv_grad_op ${DAUConvNet_INCLUDE_DIRS} PUBLIC ${DAUConvNet_INCLUDE_DIR})




#LINK_DIRECTORIES(${TF_LIB})
add_library(dau_conv_op SHARED src/dau_conv_op.cpp)

target_link_libraries(dau_conv_op dau_conv_tensorflow)
target_include_directories(dau_conv_op PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(dau_conv_op PUBLIC "/usr/local/")
target_include_directories(dau_conv_op ${DAUConvNet_INCLUDE_DIRS} PUBLIC ${DAUConvNet_INCLUDE_DIR})


include(GNUInstallDirs)

set(CMAKE_INSTALL_PREFIX "/usr/local/lib/")

message(STATUS ${CMAKE_INSTALL_FULL_LIBDIR})
message(STATUS ${CMAKE_INSTALL_FULL_BINDIR})
install(TARGETS dau_conv_tensorflow dau_conv_op dau_conv_grad_op
        ARCHIVE  DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/lib/
        LIBRARY  DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/lib/
        RUNTIME  DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/bin/)


set(DAU_CONV_OP_PATH "${CMAKE_INSTALL_PREFIX}${CMAKE_SHARED_LIBRARY_PREFIX}dau_conv_op${CMAKE_SHARED_LIBRARY_SUFFIX}")
set(DAU_CONV_GRAD_OP_PATH "${CMAKE_INSTALL_PREFIX}${CMAKE_SHARED_LIBRARY_PREFIX}dau_conv_grad_op${CMAKE_SHARED_LIBRARY_SUFFIX}")
set(DAU_CONV_TENSORFLOW_PATH "${CMAKE_INSTALL_PREFIX}${CMAKE_SHARED_LIBRARY_PREFIX}dau_conv_tensorflow${CMAKE_SHARED_LIBRARY_SUFFIX}")
message(STATUS ${DAU_CONV_OP_PATH})
message(STATUS ${DAU_CONV_GRAD_OP_PATH})
message(STATUS ${DAU_CONV_TENSORFLOW_PATH})

install(CODE "execute_process(COMMAND ldconfig ${DAU_CONV_OP_PATH})")
install(CODE "execute_process(COMMAND ldconfig ${DAU_CONV_GRAD_OP_PATH})")
install(CODE "execute_process(COMMAND ldconfig ${DAU_CONV_TENSORFLOW_PATH})")

set(DAU_CONV_MODULE_NAME dau_conv)

set(SETUP_PY_IN "${CMAKE_CURRENT_SOURCE_DIR}/setup.py.in")
set(SETUP_PY    "${CMAKE_CURRENT_BINARY_DIR}/setup.py")
set(DEPS        "${CMAKE_CURRENT_SOURCE_DIR}/${DAU_CONV_MODULE_NAME}/__init__.py")
set(OUTPUT      "${CMAKE_CURRENT_BINARY_DIR}/build/timestamp")

configure_file(${SETUP_PY_IN} ${SETUP_PY})

add_custom_command(OUTPUT ${OUTPUT}
        COMMAND ${PYTHON_EXECUTABLE} ${SETUP_PY} build
        COMMAND ${CMAKE_COMMAND} -E touch ${OUTPUT}
        DEPENDS ${DEPS})

add_custom_target(target ALL DEPENDS ${OUTPUT})

install(CODE "execute_process(COMMAND ${PYTHON_EXECUTABLE} ${SETUP_PY} install)")
