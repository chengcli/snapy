# run_shallow_xy_test.cmake

set(ref_file "../bin/shallow_xy_single.out0.00001.nc")
if(NOT EXISTS "${ref_file}")
  message(FATAL_ERROR "Missing shallow_xy reference file ${ref_file}")
endif()

file(GLOB shallow_xy_outputs
  "shallow_xy-main.nc"
  "shallow_xy.out*.nc"
  "shallow_xy.[0-9][0-9][0-9][0-9][0-9].restart"
  "shallow_xy.final.restart"
)
if(shallow_xy_outputs)
  file(REMOVE ${shallow_xy_outputs})
endif()

execute_process(COMMAND ln -sf ../bin/shallow_xy_single.yaml shallow_xy.yaml)

execute_process(
  COMMAND pd-run 1 ../bin/shallow_xy.${buildl} shallow_xy.yaml
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
  COMMAND python test_shallow_xy.py shallow_xy-main.nc ${ref_file}
  RESULT_VARIABLE res
)
if(NOT res EQUAL 0)
  message(FATAL_ERROR "test_shallow_xy failed with exit code ${res}")
endif()
