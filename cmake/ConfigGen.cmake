

################################################################################################
# Function for generation DAU-ConvNet build- and install- tree export config files
# Usage:
#  dau_conv_generate_export_configs()
function(dau_conv_generate_export_configs)
  set(install_cmake_suffix "share/DAUConvNet")

  if(NOT HAVE_CUDA)
    set(HAVE_CUDA FALSE)
  endif()

  # ---[ Configure build-tree DAUConvNetConfig.cmake file ]---

  configure_file("cmake/Templates/DAUConvNetConfig.cmake.in" "${PROJECT_BINARY_DIR}/DAUConvNetConfig.cmake" @ONLY)

  # Add targets to the build-tree export set
  export(TARGETS dau-conv FILE "${PROJECT_BINARY_DIR}/DAUConvNetTargets.cmake")
  export(PACKAGE DAUConvNet)

  # ---[ Configure install-tree DAUConvNetConfig.cmake file ]---

  configure_file("cmake/Templates/DAUConvNetConfig.cmake.in" "${PROJECT_BINARY_DIR}/cmake/DAUConvNetConfig.cmake" @ONLY)

  # Install the DAUConvNetConfig.cmake and export set to use with install-tree
  install(FILES "${PROJECT_BINARY_DIR}/cmake/DAUConvNetConfig.cmake" DESTINATION ${install_cmake_suffix})

endfunction()


