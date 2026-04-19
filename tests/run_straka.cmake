# run_straka_test.cmake

if(NOT DEFINED nproc)
  set(nproc 2)
endif()

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

execute_process(COMMAND ln -sf ../bin/straka.yaml straka.yaml)

execute_process(
  COMMAND torchrun --no-python --nproc-per-node=${nproc} ../bin/straka.${buildl}
  RESULT_VARIABLE res
)

if(NOT res EQUAL 0)
  message(FATAL_ERROR "torchrun failed with exit code ${res}")
endif()

execute_process(
  COMMAND pd-combine 1 -o main
  RESULT_VARIABLE res
)
if(NOT res EQUAL 0)
  message(FATAL_ERROR "pd-combine failed with exit code ${res}")
endif()

execute_process(
  COMMAND python test_straka.py straka-main.nc straka-ref.nc
  RESULT_VARIABLE res
)
if(NOT res EQUAL 0)
  message(FATAL_ERROR "test_straka failed with exit code ${res}")
endif()
