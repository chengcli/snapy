# =================
# Examples Problems
# =================

setup_problem(shock)
setup_problem(straka)
setup_problem(earth_moist)

file(GLOB inputs *.yaml *.py)
foreach(input ${inputs})
  execute_process(COMMAND ln -sf ${input} ${CMAKE_BINARY_DIR}/bin/${inp})
endforeach()
