# run_straka_test.cmake

set(download_link "https://zenodo.org/records/18121953/files/straka-ref.nc")

if(EXISTS "straka-ref.nc")
  set(res 0)
else()
  execute_process(
    COMMAND curl -L -o straka-ref.nc ${download_link}
    RESULT_VARIABLE res
  )
endif()

if(NOT res EQUAL 0)
  message(FATAL_ERROR "Failed to download reference file with exit code ${res}")
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
