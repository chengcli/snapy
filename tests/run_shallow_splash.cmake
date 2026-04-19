# run_shallow_splash_test.cmake

if(NOT DEFINED nproc)
  set(nproc 6)
endif()

set(input_yaml "../bin/shallow_splash_mesh6.yaml")
if(nproc EQUAL 3)
  set(input_yaml "../bin/shallow_splash_proc3_mesh2.yaml")
endif()

set(download_link "https://zenodo.org/records/18121953/files/shallow_splash-ref.nc")

if(EXISTS "shallow_splash-ref.nc")
  set(_status 0)
else()
  file(DOWNLOAD
    "${download_link}"
    "shallow_splash-ref.nc"
    STATUS _status
    SHOW_PROGRESS
  )
endif()

if(NOT _status EQUAL 0)
  message(FATAL_ERROR "Failed to download reference file with exit code ${_status}")
endif()

execute_process(COMMAND ln -sf ${input_yaml} shallow_splash.yaml)

execute_process(
  COMMAND torchrun --no-python --nproc-per-node=${nproc} ../bin/shallow_splash.${buildl}
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
  COMMAND python test_shallow_splash.py shallow_splash-main.nc shallow_splash-ref.nc
  RESULT_VARIABLE res
)
if(NOT res EQUAL 0)
  message(FATAL_ERROR "test_shallow_splash failed with exit code ${res}")
endif()
