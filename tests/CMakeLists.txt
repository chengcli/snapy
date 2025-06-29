# ===============================================
# Examples for testing
# ===============================================

enable_testing()

setup_test(test_eos)
setup_test(test_weno)
setup_test(test_reconstruct)
setup_test(test_riemann)
setup_test(test_internal_boundary)
setup_test(test_rectify)

file(GLOB inputs *.inp *.dat *.yaml)
foreach(input ${inputs})
  execute_process(COMMAND ln -sf ${input} ${CMAKE_BINARY_DIR}/bin/${inp})
endforeach()
