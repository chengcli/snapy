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

list(GET _status 0 _status_code)
if(NOT _status_code EQUAL 0)
  list(GET _status 1 _status_message)
  message(FATAL_ERROR
          "Failed to download reference file: ${_status_code} ${_status_message}")
endif()

file(GLOB shallow_splash_outputs
  "shallow_splash-main.nc"
  "shallow_splash.out*.nc"
  "shallow_splash.[0-9][0-9][0-9][0-9][0-9].restart"
  "shallow_splash.final.restart"
)
if(shallow_splash_outputs)
  file(REMOVE ${shallow_splash_outputs})
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
