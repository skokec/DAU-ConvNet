################################################################################################
# Defines global DAUConvNet_LINK flag, This flag is required to prevent linker from excluding
# some objects which are not addressed directly but are registered via static constructors
macro(dau_conv_set_link)
  if(BUILD_SHARED_LIBS)
    set(DAUConvNet_LINK dau-conv)
  else()
    if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
      set(DAUConvNet_LINK -Wl,-force_load dau-conv)
    elseif("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
      set(DAUConvNet_LINK -Wl,--whole-archive dau-conv -Wl,--no-whole-archive)
    endif()
  endif()
endmacro()
################################################################################################
# Convenient command to setup source group for IDEs that support this feature (VS, XCode)
# Usage:
#   dau_conv_source_group(<group> GLOB[_RECURSE] <globbing_expression>)
function(dau_conv_source_group group)
  cmake_parse_arguments(DAU_CONV_SOURCE_GROUP "" "" "GLOB;GLOB_RECURSE" ${ARGN})
  if(DAU_CONV_SOURCE_GROUP_GLOB)
    file(GLOB srcs1 ${DAU_CONV_SOURCE_GROUP_GLOB})
    source_group(${group} FILES ${srcs1})
  endif()

  if(DAU_CONV_SOURCE_GROUP_GLOB_RECURSE)
    file(GLOB_RECURSE srcs2 ${DAU_CONV_SOURCE_GROUP_GLOB_RECURSE})
    source_group(${group} FILES ${srcs2})
  endif()
endfunction()

################################################################################################
# Collecting sources from globbing and appending to output list variable
# Usage:
#   dau_conv_collect_sources(<output_variable> GLOB[_RECURSE] <globbing_expression>)
function(dau_conv_collect_sources variable)
  cmake_parse_arguments(DAU_CONV_COLLECT_SOURCES "" "" "GLOB;GLOB_RECURSE" ${ARGN})
  if(DAU_CONV_COLLECT_SOURCES_GLOB)
    file(GLOB srcs1 ${DAU_CONV_COLLECT_SOURCES_GLOB})
    set(${variable} ${variable} ${srcs1})
  endif()

  if(DAU_CONV_COLLECT_SOURCES_GLOB_RECURSE)
    file(GLOB_RECURSE srcs2 ${DAU_CONV_COLLECT_SOURCES_GLOB_RECURSE})
    set(${variable} ${variable} ${srcs2})
  endif()
endfunction()

################################################################################################
# Short command getting dau_conv_impl sources (assuming standard DAUConvNet code tree)
# Usage:
#   dau_conv_pickup_dau_conv_sources(<root>)
function(dau_conv_pickup_sources root)
  # put all files in source groups (visible as subfolder in many IDEs)
  dau_conv_source_group("Include"        GLOB "${root}/include/dau_conv/*.h*")
  dau_conv_source_group("Include\\Util"  GLOB "${root}/include/dau_conv/util/*.h*")
  dau_conv_source_group("Include"        GLOB "${PROJECT_BINARY_DIR}/dau_conv_config.h*")
  dau_conv_source_group("Source"         GLOB "${root}/src/dau_conv/*.cpp")
  dau_conv_source_group("Source\\Util"   GLOB "${root}/src/dau_conv/util/*.cpp")
  dau_conv_source_group("Source\\Layers" GLOB "${root}/src/dau_conv/layers/*.cpp")
  dau_conv_source_group("Source\\Cuda"   GLOB "${root}/src/dau_conv/layers/*.cu")
  dau_conv_source_group("Source\\Cuda"   GLOB "${root}/src/dau_conv/util/*.cu")

  # collect files
  file(GLOB_RECURSE hdrs ${root}/include/dau_cconv/*.h*)
  file(GLOB_RECURSE srcs ${root}/src/dau_conv/*.cpp)

  # adding headers to make the visible in some IDEs (Qt, VS, Xcode)
  list(APPEND srcs ${hdrs} ${PROJECT_BINARY_DIR}/dau_conv_config.h)

  # collect cuda files
  file(GLOB_RECURSE cuda ${root}/src/dau_conv/*.cu)

  # convert to absolute paths
  dau_conv_convert_absolute_paths(srcs)
  dau_conv_convert_absolute_paths(cuda)

  # propagate to parent scope
  set(srcs ${srcs} PARENT_SCOPE)
  set(cuda ${cuda} PARENT_SCOPE)
endfunction()

################################################################################################
# Short command for setting default target properties
# Usage:
#   dau_conv_default_properties(<target>)
function(dau_conv_default_properties target)
  set_target_properties(${target} PROPERTIES
    DEBUG_POSTFIX ${DAUConvNet_DEBUG_POSTFIX}
    ARCHIVE_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/lib"
    LIBRARY_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/lib"
    RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/bin")
  # make sure we build all external dependencies first
  if (DEFINED external_project_dependencies)
    add_dependencies(${target} ${external_project_dependencies})
  endif()
endfunction()

################################################################################################
# Short command for setting runtime directory for build target
# Usage:
#   dau_conv_set_runtime_directory(<target> <dir>)
function(dau_conv_set_runtime_directory target dir)
  set_target_properties(${target} PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${dir}")
endfunction()

################################################################################################
# Short command for setting solution folder property for target
# Usage:
#   dau_conv_set_solution_folder(<target> <folder>)
function(dau_conv_set_solution_folder target folder)
  if(USE_PROJECT_FOLDERS)
    set_target_properties(${target} PROPERTIES FOLDER "${folder}")
  endif()
endfunction()

################################################################################################
# Reads lines from input file, prepends source directory to each line and writes to output file
# Usage:
#   dau_conv_configure_testdatafile(<testdatafile>)
function(dau_conv_configure_testdatafile file)
  file(STRINGS ${file} __lines)
  set(result "")
  foreach(line ${__lines})
    set(result "${result}${PROJECT_SOURCE_DIR}/${line}\n")
  endforeach()
  file(WRITE ${file}.gen.cmake ${result})
endfunction()

################################################################################################
# Filter out all files that are not included in selected list
# Usage:
#   dau_conv_leave_only_selected_tests(<filelist_variable> <selected_list>)
function(dau_conv_leave_only_selected_tests file_list)
  if(NOT ARGN)
    return() # blank list means leave all
  endif()
  string(REPLACE "," ";" __selected ${ARGN})
  list(APPEND __selected dau_conv_main)

  set(result "")
  foreach(f ${${file_list}})
    get_filename_component(name ${f} NAME_WE)
    string(REGEX REPLACE "^test_" "" name ${name})
    list(FIND __selected ${name} __index)
    if(NOT __index EQUAL -1)
      list(APPEND result ${f})
    endif()
  endforeach()
  set(${file_list} ${result} PARENT_SCOPE)
endfunction()

