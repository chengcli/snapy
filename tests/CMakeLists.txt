# ========== #
# Unit Tests #
# ========== #

enable_testing()

setup_test(test_eos)
setup_test(test_plm)
setup_test(test_weno)
setup_test(test_reconstruct)
setup_test(test_riemann)
setup_test(test_internal_boundary)
setup_test(test_rectify)
setup_test(test_pull_2d)
setup_test(test_pull_3d)
setup_test(test_slab)
setup_test(test_cubed)
setup_test(test_cubed_sphere)
setup_test(test_coordinate)
#setup_test(test_read_topo)
#setup_cuda_test(test_thomas_solver)
#setup_test(test_aneos)

file(GLOB inputs *.yaml *.py)
# create tests folder if does not exist
file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/tests)
foreach(input ${inputs})
  execute_process(COMMAND ln -sf ${input} ${CMAKE_BINARY_DIR}/tests/${inp})
endforeach()

setup_parallel_test(test_exchange 6)

# ============= #
# Example Tests #
# ============= #
add_test(
  NAME test_straka
  COMMAND ${CMAKE_COMMAND}
          -Dbuildl=release
          -P ${CMAKE_CURRENT_SOURCE_DIR}/run_straka_test.cmake
)

set_tests_properties(test_straka PROPERTIES
  WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/tests
)

add_test(
  NAME test_shallow_xy
  COMMAND ${CMAKE_COMMAND}
          -Dbuildl=release
          -P ${CMAKE_CURRENT_SOURCE_DIR}/run_shallow_xy.cmake
)

set_tests_properties(test_shallow_xy PROPERTIES
  WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/tests
)

add_test(
  NAME test_shallow_splash
  COMMAND ${CMAKE_COMMAND}
          -Dbuildl=release
          -P ${CMAKE_CURRENT_SOURCE_DIR}/run_shallow_splash_test.cmake
)

set_tests_properties(test_shallow_splash PROPERTIES
  WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/tests
)
