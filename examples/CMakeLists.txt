# =================
# Examples Problems
# =================

setup_problem(example_shock)
setup_problem(straka)

file(GLOB inputs *.yaml)
foreach(input ${inputs})
  execute_process(COMMAND ln -sf ${input} ${CMAKE_BINARY_DIR}/bin/${inp})
endforeach()
