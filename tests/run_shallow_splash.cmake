# run_shallow_splash_test.cmake

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

if(NOT res EQUAL 0)
  message(FATAL_ERROR "Failed to download reference file with exit code ${res}")
endif()

execute_process(COMMAND ln -sf ../bin/shallow_splash.yaml shallow_splash.yaml)

execute_process(
  COMMAND pd-run 6 ../bin/shallow_splash.${buildl}
  RESULT_VARIABLE res
)
if(NOT res EQUAL 0)
  message(FATAL_ERROR "pd-run failed with exit code ${res}")
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
