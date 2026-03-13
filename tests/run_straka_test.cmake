# run_straka_test.cmake

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

list(GET _status 0 _status_code)
if(NOT _status_code EQUAL 0)
  list(GET _status 1 _status_message)
  message(FATAL_ERROR
          "Failed to download reference file: ${_status_code} ${_status_message}")
endif()

file(GLOB straka_outputs
  "straka-main.nc"
  "straka.out*.nc"
  "straka.[0-9][0-9][0-9][0-9][0-9].restart"
  "straka.final.restart"
)
if(straka_outputs)
  file(REMOVE ${straka_outputs})
endif()

execute_process(COMMAND ln -sf ../bin/straka.yaml straka.yaml)

execute_process(
  COMMAND pd-run 2 ../bin/straka.${buildl}
  RESULT_VARIABLE res
)

if(NOT res EQUAL 0)
  message(FATAL_ERROR "pd-run failed with exit code ${res}")
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
