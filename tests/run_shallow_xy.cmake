# run_shallow_xy_test.cmake

get_filename_component(build_dir "${CMAKE_CURRENT_BINARY_DIR}/.." ABSOLUTE)
set(work_dir "${CMAKE_CURRENT_BINARY_DIR}/reference_shallow_xy")
file(REMOVE_RECURSE "${work_dir}")
file(MAKE_DIRECTORY "${work_dir}")

set(download_link "https://zenodo.org/records/18121953/files/shallow_xy-ref.nc")

if(EXISTS "shallow_xy-ref.nc")
  set(_status 0)
else()
  file(DOWNLOAD
    "${download_link}"
    "shallow_xy-ref.nc"
    STATUS _status
    SHOW_PROGRESS
  )
endif()

if(NOT _status EQUAL 0)
  message(FATAL_ERROR "Failed to download reference file with exit code ${_status}")
endif()

configure_file("${build_dir}/bin/shallow_xy.yaml"
               "${work_dir}/shallow_xy.yaml" COPYONLY)

execute_process(
  COMMAND torchrun --no-python --nproc-per-node=4 "${build_dir}/bin/shallow_xy.${buildl}"
  WORKING_DIRECTORY "${work_dir}"
  RESULT_VARIABLE res
)
if(NOT res EQUAL 0)
  message(FATAL_ERROR "torchrun failed with exit code ${res}")
endif()

execute_process(
  COMMAND pd-combine 0 -o main
  WORKING_DIRECTORY "${work_dir}"
  RESULT_VARIABLE res
)
if(NOT res EQUAL 0)
  message(FATAL_ERROR "pd-combine failed with exit code ${res}")
endif()

execute_process(
  COMMAND python "${CMAKE_CURRENT_LIST_DIR}/test_shallow_xy.py"
          shallow_xy-main.nc "${CMAKE_CURRENT_BINARY_DIR}/shallow_xy-ref.nc"
  WORKING_DIRECTORY "${work_dir}"
  RESULT_VARIABLE res
)
if(NOT res EQUAL 0)
  message(FATAL_ERROR "test_shallow_xy failed with exit code ${res}")
endif()
