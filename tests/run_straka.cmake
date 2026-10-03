# run_straka_test.cmake

get_filename_component(build_dir "${CMAKE_CURRENT_BINARY_DIR}/.." ABSOLUTE)
set(work_dir "${CMAKE_CURRENT_BINARY_DIR}/reference_straka")
file(REMOVE_RECURSE "${work_dir}")
file(MAKE_DIRECTORY "${work_dir}")

set(download_link "https://zenodo.org/records/18121953/files/straka-ref.nc")

if(EXISTS "straka-ref.nc")
  set(_status 0)
else()
  file(DOWNLOAD
    "${download_link}"
    "straka-ref.nc"
    STATUS _status
    SHOW_PROGRESS
  )
endif()

if(NOT _status EQUAL 0)
  message(FATAL_ERROR "Failed to download reference file with exit code ${_status}")
endif()

file(READ "${build_dir}/bin/straka.yaml" config)
file(READ "${build_dir}/configure.h" configure_h)
if(configure_h MATCHES "NO_PNETCDFOUTPUT")
  string(REPLACE "type: pnetcdf" "type: netcdf" config "${config}")
endif()
file(WRITE "${work_dir}/straka.yaml" "${config}")

execute_process(
  COMMAND torchrun --no-python --nproc-per-node=2 "${build_dir}/bin/straka.${buildl}"
  WORKING_DIRECTORY "${work_dir}"
  RESULT_VARIABLE res
)

if(NOT res EQUAL 0)
  message(FATAL_ERROR "torchrun failed with exit code ${res}")
endif()

execute_process(
  COMMAND pd-combine 1 -o main
  WORKING_DIRECTORY "${work_dir}"
  RESULT_VARIABLE res
)
if(NOT res EQUAL 0)
  message(FATAL_ERROR "pd-combine failed with exit code ${res}")
endif()

execute_process(
  COMMAND python "${CMAKE_CURRENT_LIST_DIR}/test_straka.py"
          straka-main.nc "${CMAKE_CURRENT_BINARY_DIR}/straka-ref.nc"
  WORKING_DIRECTORY "${work_dir}"
  RESULT_VARIABLE res
)
if(NOT res EQUAL 0)
  message(FATAL_ERROR "test_straka failed with exit code ${res}")
endif()
