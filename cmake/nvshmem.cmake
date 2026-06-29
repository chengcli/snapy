set(NVSHMEM ${CUDA} CACHE INTERNAL
    "Enable NVSHMEM-backed kernel-side communication" FORCE)

if(NOT NVSHMEM)
  return()
endif()

get_filename_component(_torch_lib_dir "${TORCH_LIBRARY}" DIRECTORY)
get_filename_component(_torch_package_dir "${_torch_lib_dir}" DIRECTORY)
get_filename_component(_torch_site_packages_dir "${_torch_package_dir}" DIRECTORY)

if(TARGET torch_nvshmem)
  find_path(NVSHMEM_INCLUDE_DIR nvshmem.h
    HINTS "${_torch_site_packages_dir}/nvidia/nvshmem"
    PATH_SUFFIXES include)
  set(NVSHMEM_TORCH_TARGET torch_nvshmem)
  message(STATUS "NVSHMEM: using LibTorch torch_nvshmem target")
else()
  find_path(NVSHMEM_INCLUDE_DIR nvshmem.h
    HINTS
      "${_torch_site_packages_dir}/nvidia/nvshmem"
      ENV NVSHMEM_HOME
      ENV NVSHMEM_ROOT
    PATH_SUFFIXES include)
  find_library(NVSHMEM_LIBRARY
    NAMES torch_nvshmem nvshmem
    HINTS
      "${_torch_lib_dir}"
      "${_torch_site_packages_dir}/nvidia/nvshmem"
      ENV NVSHMEM_HOME
      ENV NVSHMEM_ROOT
    PATH_SUFFIXES lib lib64)
endif()

if(NOT NVSHMEM_INCLUDE_DIR OR (NOT NVSHMEM_LIBRARY AND NOT NVSHMEM_TORCH_TARGET))
  message(FATAL_ERROR
    "snapy requires NVSHMEM headers and either LibTorch's "
    "torch_nvshmem target or a standalone libnvshmem when CUDA=ON. "
    "Set NVSHMEM_HOME or NVSHMEM_ROOT if using a standalone install.")
endif()
