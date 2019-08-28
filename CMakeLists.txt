# DAU-ConvNet project exposes the following varaibles:
#  - DAUConvNet_OBJ_TARGET: name of object target (that can be used as dependency)
#  - DAUConvNet_OBJS: pre-compiled .o objects (.cpp and .cu files)
#  - DAUConvNet_CU_OBJS; pre-compiled .cu.o objects (only resutling CUDA objects !!)
#  - DAUConvNet_INCLUDE_DIRS: include dirs of dependencies for DAU-ConvNet
#  - DAUConvNet_LINKER_LIBS: linker libs of dependencies for DAU-ConvNet
#  - DAUConvNet_INCLUDE_DIR: include dir for DAU-ConvNet (i.e. 3thparty/DAU-ConvNet/include)

cmake_minimum_required(VERSION 2.8.8)
if(POLICY CMP0046)
  cmake_policy(SET CMP0046 NEW)
endif()
if(POLICY CMP0054)
  cmake_policy(SET CMP0054 NEW)
endif()

# ---[ DAU-ConvNet project
project(DAUConvNet C CXX)

set(PACKAGE_VERSION "1.0" CACHE STRING "DAU-ConvNet version number")

# ---[ Using cmake scripts and modules
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake/Modules)

include(cmake/Utils.cmake)
include(cmake/Targets.cmake)
include(cmake/ConfigGen.cmake)

# ---[ Using C++11

include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)
if(COMPILER_SUPPORTS_CXX11)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
elseif(COMPILER_SUPPORTS_CXX0X)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
else()
  message(STATUS "The compiler ${CMAKE_CXX_COMPILER} has no C++11 support. Please use a different C++ compiler.")
endif()

# ---[ Configuration types
set(CMAKE_CONFIGURATION_TYPES "Debug;Release" CACHE STRING "Possible configurations" FORCE)
mark_as_advanced(CMAKE_CONFIGURATION_TYPES)

if(DEFINED CMAKE_BUILD_TYPE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS ${CMAKE_CONFIGURATION_TYPES})
endif()

# --[ If user doesn't specify build type then assume release
if("${CMAKE_BUILD_TYPE}" STREQUAL "")
  set(CMAKE_BUILD_TYPE Release)
endif()

if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
  set(CMAKE_COMPILER_IS_CLANGXX TRUE)
endif()

# ---[ Solution folders
dau_conv_option(USE_PROJECT_FOLDERS "IDE Solution folders" (MSVC_IDE OR CMAKE_GENERATOR MATCHES Xcode) )

if(USE_PROJECT_FOLDERS)
  set_property(GLOBAL PROPERTY USE_FOLDERS ON)
  set_property(GLOBAL PROPERTY PREDEFINED_TARGETS_FOLDER "CMakeTargets")
endif()

# ---[ RPATH settings
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE CACHE BOOLEAN "Use link paths for shared library rpath")
set(CMAKE_MACOSX_RPATH TRUE)

list(FIND CMAKE_PLATFORM_IMPLICIT_LINK_DIRECTORIES ${CMAKE_INSTALL_PREFIX}/lib __is_systtem_dir)
if(${__is_systtem_dir} STREQUAL -1)
  set(CMAKE_INSTALL_RPATH ${CMAKE_INSTALL_PREFIX}/lib)
endif()

# ---[ Set debug postfix
set(DAUConvNet_DEBUG_POSTFIX "-d")

set(DAUConvNet_POSTFIX "")
if(CMAKE_BUILD_TYPE MATCHES "Debug")
  set(DAUConvNet_POSTFIX ${DAUConvNet_DEBUG_POSTFIX})
endif()


# ---[ Options
dau_conv_option(BUILD_TENSORFLOW_PLUGIN "Builds TensorFlow plugin" OFF)
dau_conv_option(BUILD_SHARED_LIBS "Build shared libraries" ON)
dau_conv_option(USE_DUMMY_CUDA_IMPL "For debugging purpose; do not compile CUDA kernels (fast compile time)" OFF)
dau_conv_option(ALLOW_INTERPOLATION_OFF "Build support for disabling interpolation in DAUs" OFF)

# ---[ Dependencies
include(cmake/Dependencies.cmake)

# ---[ Flags
if(UNIX OR APPLE)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -Wall")
endif()

if(USE_libstdcpp)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -stdlib=libstdc++")
  message("-- Warning: forcing libstdc++ (controlled by USE_libstdcpp option in cmake)")
endif()

if(${USE_DUMMY_CUDA_IMPL})
  list(APPEND DAUConvNet_DEFINITIONS "-DDAU_USE_DUMMY_CUDA_IMPL")
endif()

if(${ALLOW_INTERPOLATION_OFF})
  list(APPEND DAUConvNet_DEFINITIONS "-DDAU_ALLOW_INTERPOLATION_OFF")
endif()

# ---[ Warnings
dau_conv_warnings_disable(CMAKE_CXX_FLAGS -Wno-sign-compare -Wno-uninitialized)

# ---[ Config generation
configure_file(cmake/Templates/dau_conv_config.h.in "${PROJECT_BINARY_DIR}/dau_conv_config.h")

# ---[ Includes
set(DAUConvNet_INCLUDE_DIR ${PROJECT_SOURCE_DIR}/include)
set(DAUConvNet_SRC_DIR ${PROJECT_SOURCE_DIR}/src)
include_directories(${PROJECT_BINARY_DIR})

# ---[ Includes & defines for CUDA

# cuda_compile() does not have per-call dependencies or include pathes
# (cuda_compile() has per-call flags, but we set them here too for clarity)
#
# list(REMOVE_ITEM ...) invocations remove PRIVATE and PUBLIC keywords from collected definitions and include pathes
if(HAVE_CUDA)
  # pass include pathes to cuda_include_directories()
  set(DAUConvNet_ALL_INCLUDE_DIRS ${DAUConvNet_INCLUDE_DIRS})
  list(REMOVE_ITEM DAUConvNet_ALL_INCLUDE_DIRS PRIVATE PUBLIC)
  cuda_include_directories(${DAUConvNet_INCLUDE_DIR} ${DAUConvNet_SRC_DIR} ${DAUConvNet_ALL_INCLUDE_DIRS})

  # add definitions to nvcc flags directly
  set(DAUConvNet_ALL_DEFINITIONS ${DAUConvNet_DEFINITIONS})
  list(REMOVE_ITEM DAUConvNet_ALL_DEFINITIONS PRIVATE PUBLIC)
  list(APPEND CUDA_NVCC_FLAGS ${DAUConvNet_ALL_DEFINITIONS})
else()
  message( FATAL_ERROR "MISSING CUDA: DAU-ConvNet implementation requires CUDA")
endif()

# ---[ Subdirectories
add_subdirectory(src/dau_conv)
add_subdirectory(plugins/tensorflow)

# ---[ Export configs generation
dau_conv_generate_export_configs()

get_directory_property(has_parent PARENT_DIRECTORY)

if (has_parent)
  set(DAUConvNet_OBJ_TARGET ${DAUConvNet_OBJ_TARGET} PARENT_SCOPE)
  set(DAUConvNet_OBJS ${DAUConvNet_OBJS} PARENT_SCOPE)
  set(DAUConvNet_CU_OBJS ${DAUConvNet_CU_OBJS} PARENT_SCOPE)
  set(DAUConvNet_INCLUDE_DIRS ${DAUConvNet_INCLUDE_DIRS} PARENT_SCOPE)
  set(DAUConvNet_INCLUDE_DIR ${DAUConvNet_INCLUDE_DIR} PARENT_SCOPE)
  set(DAUConvNet_LINKER_LIBS ${DAUConvNet_LIBS} PARENT_SCOPE)
endif()
