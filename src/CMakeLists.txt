# Install snapy library
#
#   library: snapy_debug.o
#   library: snapy_release.o
#
# Define the following symbols
#
#   SNAP_INCLUDE_DIR
#   snapy::snap
#   snapy::snap_cu
#   snapy::bc
#
# Normal usage would be:
#
#   include_directories(${SNAP_INCLUDE_DIR})
#   target_link_libraries(snapy::snap snapy::snap_cu snapy::bc)

include_directories(
  "${CMAKE_CURRENT_SOURCE_DIR}/.."
  ${CMAKE_BINARY_DIR}
  ${NETCDF_INCLUDES}
  ${PNETCDF_INCLUDE_DIR}
  SYSTEM ${TORCH_API_INCLUDE_DIR}
  SYSTEM ${TORCH_INCLUDE_DIR}
  SYSTEM ${KINTERA_INCLUDE_DIR}
  SYSTEM ${HARP_INCLUDE_DIR}
  )

string(TOLOWER ${CMAKE_BUILD_TYPE} buildl)
string(TOUPPER ${CMAKE_BUILD_TYPE} buildu)

set(namel snap)
string(TOUPPER ${namel} nameu)

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
    forcing/*.cpp
    implicit/*.cpp
    intg/*.cpp
    input/*.cpp
    sedimentation/*.cpp
    )

add_library(${namel}_${buildl}
    SHARED
    ${src_files}
    )

set_target_properties(${namel}_${buildl}
    PROPERTIES
    COMPILE_FLAGS "${CMAKE_CXX_FLAGS_${buildu}}"
    )

target_link_libraries(${namel}_${buildl}
    PUBLIC
    snapy::bc
    ${NETCDF_LIBRARIES}
    ${PNETCDF_LIBRARIES}
    ${TORCH_LIBRARY}
    ${TORCH_CPU_LIBRARY}
    ${C10_LIBRARY}
    ${DISORT_LIBRARY}
    ${HARP_LIBRARY}
    ${KINTERA_LIBRARY}
    #-Wl,--no-as-needed
    #${VAPORS_LIBRARY}
    #-Wl,--as-needed
    )

add_library(snapy::snap ALIAS ${namel}_${buildl})

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
      SHARED
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

  add_library(snapy::snap_cu ALIAS ${namel}_cuda_${buildl})
endif()

set(SNAP_INCLUDE_DIR
  "${CMAKE_CURRENT_SOURCE_DIR}/.."
  CACHE INTERNAL "snap include directory")

add_subdirectory(bc)
