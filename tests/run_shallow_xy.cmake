# run_shallow_xy_test.cmake

if(NOT DEFINED nproc)
  set(nproc 4)
endif()

set(input_yaml "../bin/shallow_xy_mesh4.yaml")
if(nproc EQUAL 2)
  set(input_yaml "../bin/shallow_xy_proc2_mesh2.yaml")
endif()

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

execute_process(COMMAND ln -sf ${input_yaml} shallow_xy.yaml)

execute_process(
  COMMAND torchrun --no-python --nproc-per-node=${nproc} ../bin/shallow_xy.${buildl}
  RESULT_VARIABLE res
)
if(NOT res EQUAL 0)
  message(FATAL_ERROR "torchrun failed with exit code ${res}")
endif()

execute_process(
  COMMAND pd-combine 0 -o main
  RESULT_VARIABLE res
)
if(NOT res EQUAL 0)
  message(FATAL_ERROR "pd-combine failed with exit code ${res}")
endif()

execute_process(
  COMMAND python test_shallow_xy.py shallow_xy-main.nc shallow_xy-ref.nc
  RESULT_VARIABLE res
)
if(NOT res EQUAL 0)
  message(FATAL_ERROR "test_shallow_xy failed with exit code ${res}")
endif()
