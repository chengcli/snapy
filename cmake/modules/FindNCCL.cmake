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

# Allow user hints
set(_NCCL_HINT_DIRS
    ${NCCL_ROOT}
    $ENV{NCCL_ROOT}
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
    NAMES nccl
    HINTS ${_NCCL_HINT_DIRS}
    PATHS
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
