################################################################################################
# Command alias for debugging messages
# Usage:
#   dmsg(<message>)
function(dmsg)
  message(STATUS ${ARGN})
endfunction()

################################################################################################
# Removes duplicates from list(s)
# Usage:
#   dau_conv_list_unique(<list_variable> [<list_variable>] [...])
macro(dau_conv_list_unique)
  foreach(__lst ${ARGN})
    if(${__lst})
      list(REMOVE_DUPLICATES ${__lst})
    endif()
  endforeach()
endmacro()

################################################################################################
# Clears variables from list
# Usage:
#   dau_conv_clear_vars(<variables_list>)
macro(dau_conv_clear_vars)
  foreach(_var ${ARGN})
    unset(${_var})
  endforeach()
endmacro()


################################################################################################
# Converts all paths in list to absolute
# Usage:
#   dau_conv_convert_absolute_paths(<list_variable>)
function(dau_conv_convert_absolute_paths variable)
  set(__dlist "")
  foreach(__s ${${variable}})
    get_filename_component(__abspath ${__s} ABSOLUTE)
    list(APPEND __list ${__abspath})
  endforeach()
  set(${variable} ${__list} PARENT_SCOPE)
endfunction()


########################################################################################################
# An option that the user can select. Can accept condition to control when option is available for user.
# Usage:
#   dau_conv_option(<option_variable> "doc string" <initial value or boolean expression> [IF <condition>])
function(dau_conv_option variable description value)
  set(__value ${value})
  set(__condition "")
  set(__varname "__value")
  foreach(arg ${ARGN})
    if(arg STREQUAL "IF" OR arg STREQUAL "if")
      set(__varname "__condition")
    else()
      list(APPEND ${__varname} ${arg})
    endif()
  endforeach()
  unset(__varname)
  if("${__condition}" STREQUAL "")
    set(__condition 2 GREATER 1)
  endif()

  if(${__condition})
    if("${__value}" MATCHES ";")
      if(${__value})
        option(${variable} "${description}" ON)
      else()
        option(${variable} "${description}" OFF)
      endif()
    elseif(DEFINED ${__value})
      if(${__value})
        option(${variable} "${description}" ON)
      else()
        option(${variable} "${description}" OFF)
      endif()
    else()
      option(${variable} "${description}" ${__value})
    endif()
  else()
    unset(${variable} CACHE)
  endif()
endfunction()


################################################################################################
# Command for disabling warnings for different platforms (see below for gcc and VisualStudio)
# Usage:
#   dau_conv_warnings_disable(<CMAKE_[C|CXX]_FLAGS[_CONFIGURATION]> -Wshadow /wd4996 ..,)
macro(dau_conv_warnings_disable)
  set(_flag_vars "")
  set(_msvc_warnings "")
  set(_gxx_warnings "")

  foreach(arg ${ARGN})
    if(arg MATCHES "^CMAKE_")
      list(APPEND _flag_vars ${arg})
    elseif(arg MATCHES "^/wd")
      list(APPEND _msvc_warnings ${arg})
    elseif(arg MATCHES "^-W")
      list(APPEND _gxx_warnings ${arg})
    endif()
  endforeach()

  if(NOT _flag_vars)
    set(_flag_vars CMAKE_C_FLAGS CMAKE_CXX_FLAGS)
  endif()

  if(MSVC AND _msvc_warnings)
    foreach(var ${_flag_vars})
      foreach(warning ${_msvc_warnings})
        set(${var} "${${var}} ${warning}")
      endforeach()
    endforeach()
  elseif((CMAKE_COMPILER_IS_GNUCXX OR CMAKE_COMPILER_IS_CLANGXX) AND _gxx_warnings)
    foreach(var ${_flag_vars})
      foreach(warning ${_gxx_warnings})
        if(NOT warning MATCHES "^-Wno-")
          string(REPLACE "${warning}" "" ${var} "${${var}}")
          string(REPLACE "-W" "-Wno-" warning "${warning}")
        endif()
        set(${var} "${${var}} ${warning}")
      endforeach()
    endforeach()
  endif()
  dau_conv_clear_vars(_flag_vars _msvc_warnings _gxx_warnings)
endmacro()


################################################################################################
# Helper function to detect Darwin version, i.e. 10.8, 10.9, 10.10, ....
# Usage:
#   dau_conv_detect_darwin_version(<version_variable>)
function(dau_conv_detect_darwin_version output_var)
  if(APPLE)
    execute_process(COMMAND /usr/bin/sw_vers -productVersion
            RESULT_VARIABLE __sw_vers OUTPUT_VARIABLE __sw_vers_out
            ERROR_QUIET OUTPUT_STRIP_TRAILING_WHITESPACE)

    set(${output_var} ${__sw_vers_out} PARENT_SCOPE)
  else()
    set(${output_var} "" PARENT_SCOPE)
  endif()
endfunction()