# define default parameters

set_if_empty(NMASS 0)

# cuda options
if(CUDA)
  set(CUDA_OPTION "USE_CUDA")
else()
  set(CUDA_OPTION "NOT_USE_CUDA")
endif()

# ucx options
if(UCX)
  set(UCX_OPTION "USE_UCX")
else()
  set(UCX_OPTION "NOT_USE_UCX")
endif()

# netcdf options
if(NOT NETCDF OR NOT DEFINED NETCDF)
  set(NETCDF_OPTION "NO_NETCDFOUTPUT")
else()
  set(NETCDF_OPTION "NETCDFOUTPUT")
  find_package(NetCDF REQUIRED)
endif()

# pnetcdf options
if(NOT PNETCDF OR NOT DEFINED PNETCDF)
  set(PNETCDF_OPTION "NO_PNETCDFOUTPUT")
else()
  set(PNETCDF_OPTION "PNETCDFOUTPUT")
  execute_process(
    COMMAND "${Python3_EXECUTABLE}" -c
            "import pinc; print(pinc.include_dir); print(pinc.library_dir / 'libpinc.so')"
    OUTPUT_VARIABLE _pnc_info
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE _pnc_probe)
  if(NOT _pnc_probe EQUAL 0)
    message(FATAL_ERROR
            "PNETCDF=ON now requires Python package 'pinc' in ${Python3_EXECUTABLE}")
  endif()
  string(REPLACE "\n" ";" _pnc_lines "${_pnc_info}")
  list(GET _pnc_lines 0 PNETCDF_INCLUDE_DIR)
  list(GET _pnc_lines 1 PNETCDF_LIBRARY)
  set(PNETCDF_LIBRARIES "${PNETCDF_LIBRARY}")
endif()
