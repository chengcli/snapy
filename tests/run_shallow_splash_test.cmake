# run_shallow_splash_test.cmake

set(download_link "https://zenodo.org/records/18121953/files/shallow_splash-ref.nc")

if(EXISTS "shallow_splash-ref.nc")
  set(res 0)
else()
  execute_process(
    COMMAND curl -L -o shallow_splash-ref.nc ${download_link}
    RESULT_VARIABLE res
  )
endif()

if(NOT res EQUAL 0)
  message(FATAL_ERROR "Failed to download reference file with exit code ${res}")
endif()

execute_process(
  COMMAND pd-run 6 ./shallow_splash.${buildl}
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
