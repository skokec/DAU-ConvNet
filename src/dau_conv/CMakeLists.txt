
# --[ DAU-ConvNet library

# creates 'test_srcs', 'srcs', 'test_cuda', 'cuda' lists
dau_conv_pickup_sources(${PROJECT_SOURCE_DIR})

set(cuda_objs "")

# make cuda objects
dau_conv_cuda_compile(cuda_objs ${cuda})
add_custom_target(dauc-conv-cu DEPENDS ${cuda_objs} SOURCES ${cuda})

# make objects from src
add_library(dau-conv-obj OBJECT ${srcs} )
dau_conv_default_properties(dau-conv-obj)
target_include_directories(dau-conv-obj ${DAUConvNet_INCLUDE_DIRS}
                                 PUBLIC
                                 $<BUILD_INTERFACE:${DAUConvNet_INCLUDE_DIR}>
                                 $<INSTALL_INTERFACE:include>)
target_compile_definitions(dau-conv-obj ${DAUConvNet_DEFINITIONS})
add_dependencies(dau-conv-obj dauc-conv-cu)
if(DAUConvNet_COMPILE_OPTIONS)
  target_compile_options(dau-conv-obj ${DAUConvNet_COMPILE_OPTIONS})
endif()

list(APPEND DAUConvNet_OBJS ${cuda_objs} ${cuda} $<TARGET_OBJECTS:dau-conv-obj>)

# save list of .o objects (both src and cuda) so that parrent project can directly embedd into .so
set(DAUConvNet_OBJS ${DAUConvNet_OBJS} PARENT_SCOPE)
set(DAUConvNet_CU_OBJS ${cuda_objs} PARENT_SCOPE)
set(DAUConvNet_CU_SRC ${cuda} PARENT_SCOPE)
set(DAUConvNet_OBJ_TARGET "dau-conv-obj" PARENT_SCOPE)
set(DAUConvNet_LINKER_LIBS ${DAUConvNet_LINKER_LIBS} PARENT_SCOPE)

# crate shared object
add_library(dau-conv ${DAUConvNet_OBJS})
target_link_libraries(dau-conv ${DAUConvNet_LINKER_LIBS})

add_executable(main ../main.cpp)
target_link_libraries(main dau-conv)
dau_conv_default_properties(main)


