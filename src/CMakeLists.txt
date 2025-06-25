# Install libsnap library
#
#   library: libsnap_debug.o
#   library: libsnap_release.o
#
# Define the following symbols
#
#   SNAP_INCLUDE_DIR
#   libsnap::snap
#   libsnap::snap_cu
#
# Normal usage would be:
#
#   include_directories(${SNAP_INCLUDE_DIR})
#   target_link_libraries(libsnap::snap)

include_directories(
  "${CMAKE_CURRENT_SOURCE_DIR}/.."
  ${CMAKE_BINARY_DIR}
  ${NETCDF_INCLUDES}
  ${PNETCDF_INCLUDE_DIR}
  SYSTEM ${TORCH_API_INCLUDE_DIR}
  SYSTEM ${TORCH_INCLUDE_DIR}
  SYSTEM ${EIGEN3_INCLUDE_DIR}
  )

string(TOLOWER ${CMAKE_BUILD_TYPE} buildl)
string(TOUPPER ${CMAKE_BUILD_TYPE} buildu)

set(namel snap)

file(GLOB src_files
    *.cpp
		coord/*.cpp
    eos/*.cpp
    hydro/*.cpp
    riemann/*.cpp
    recon/*.cpp
    column/*.cpp
    mesh/*.cpp
    output/*.cpp
    scalar/*.cpp
    utils/*.cpp
    bc/*.cpp
    forcing/*.cpp
    implicit/*.cpp
    intg/*.cpp
    input/*.cpp
    )

add_library(${namel}_${buildl}
    STATIC
    ${src_files}
    )

set_target_properties(${namel}_${buildl}
    PROPERTIES
    COMPILE_FLAGS "${CMAKE_CXX_FLAGS_${buildu}}"
    )

target_link_libraries(${namel}_${buildl}
    PUBLIC
    kintera::kintera
    kintera::vapors
    ${TORCH_LIBRARY}
    ${TORCH_CPU_LIBRARY}
    ${C10_LIBRARY}
    ${NETCDF_LIBRARIES}
    )

add_library(libsnap::snap ALIAS ${namel}_${buildl})

# CUDA source codes
if (CUDAToolkit_FOUND)
  file(GLOB cu_src_files
      bc/*.cu
      eos/*.cu
      intg/*.cu
      recon/*.cu
      riemann/*.cu
      implicit/*.cu
      utils/*.cu
      )

  add_library(${namel}_cuda_${buildl}
      STATIC
      ${cu_src_files}
      )

  set_target_properties(${namel}_cuda_${buildl}
      PROPERTIES
      CUDA_SEPARABLE_COMPILATION ON
      )

  target_link_libraries(${namel}_cuda_${buildl}
      PUBLIC
      CUDA::cudart
      CUDA::cudart_static
      ${TORCH_CUDA_LIBRARY}
      ${C10_CUDA_LIBRARY}
      )

  add_library(libsnap::snap_cu ALIAS ${namel}_cuda_${buildl})
endif()

set(SNAP_INCLUDE_DIR
  "${CMAKE_CURRENT_SOURCE_DIR}/.."
  CACHE INTERNAL "snap include directory")
