# =================
# Examples Problems
# =================

setup_problem(shock)
setup_problem(straka)
setup_problem(shock_aneos)
setup_problem(example_crm)
setup_problem(example_topo)
setup_problem(expansion)

file(GLOB inputs *.yaml *.py example.speos)
foreach(input ${inputs})
  execute_process(COMMAND ln -sf ${input} ${CMAKE_BINARY_DIR}/bin/${inp})
endforeach()
