# FindNCCL.cmake
#
# Finds NVIDIA NCCL
#
# Result variables:
#   NCCL_FOUND
#   NCCL_INCLUDE_DIRS
#   NCCL_LIBRARIES
#
# Imported target:
#   NCCL::NCCL

include(FindPackageHandleStandardArgs)

set(_nccl_python_hint_root "")
set(_nccl_python_hint_inc "")
set(_nccl_python_hint_lib "")

execute_process(
    COMMAND "${Python3_EXECUTABLE}" -c
      "import sys, sysconfig, pathlib
try:
    import nvidia.nccl as m
    p = pathlib.Path(m.__file__).resolve().parent
except Exception:
    pure = sysconfig.get_paths().get('purelib') or sysconfig.get_paths().get('platlib')
    p = pathlib.Path(pure).resolve() / 'nvidia' / 'nccl'
print(str(p))"
    OUTPUT_VARIABLE _nccl_python_hint_root
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
)

if(_nccl_python_hint_root)
  set(_nccl_python_hint_inc "${_nccl_python_hint_root}/include")
  set(_nccl_python_hint_lib "${_nccl_python_hint_root}/lib")
endif()

# Allow user hints
set(_NCCL_HINT_DIRS
    ${NCCL_ROOT}
    $ENV{NCCL_ROOT}
    ${_nccl_python_hint_root}
    /usr
    /usr/local
    /opt/nccl
    /opt
)

# Header search
find_path(NCCL_INCLUDE_DIR
    NAMES nccl.h
    HINTS ${_NCCL_HINT_DIRS}
    PATHS
        ${_nccl_python_hint_inc}
        $ENV{HOME}/opt/include
        /usr/include
        /usr/local/include
        /opt/include
        /opt/nccl/include
        /usr/local/cuda/include
    PATH_SUFFIXES
        include
)

# Library search
find_library(NCCL_LIBRARY
    NAMES nccl libnccl.so libnccl.so.2
    HINTS ${_NCCL_HINT_DIRS}
    PATHS
        ${_nccl_python_hint_lib}
        $ENV{HOME}/opt/lib
        /usr/lib
        /usr/lib64
        /usr/local/lib
        /usr/local/lib64
        /opt/lib
        /opt/nccl/lib
        /usr/local/cuda/lib64
    PATH_SUFFIXES
        lib
        lib64
)

# Standard handling
find_package_handle_standard_args(
    NCCL
    REQUIRED_VARS NCCL_LIBRARY NCCL_INCLUDE_DIR
)

if(NCCL_FOUND)
    set(NCCL_LIBRARIES ${NCCL_LIBRARY})
    set(NCCL_INCLUDE_DIRS ${NCCL_INCLUDE_DIR})

    if(NOT TARGET NCCL::NCCL)
        add_library(NCCL::NCCL UNKNOWN IMPORTED)
        set_target_properties(NCCL::NCCL PROPERTIES
            IMPORTED_LOCATION "${NCCL_LIBRARY}"
            INTERFACE_INCLUDE_DIRECTORIES "${NCCL_INCLUDE_DIR}"
        )
    endif()
endif()

mark_as_advanced(
    NCCL_INCLUDE_DIR
    NCCL_LIBRARY
)
