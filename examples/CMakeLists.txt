# =================
# Examples Problems
# =================

setup_problem(shock)
setup_problem(straka)
setup_problem(run_hydro)
setup_problem(shallow_xy)
setup_problem(shallow_splash)

file(GLOB inputs *.yaml *.py)
# create bin folder if does not exist
file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
foreach(input ${inputs})
  execute_process(COMMAND ln -sf ${input} ${CMAKE_BINARY_DIR}/bin/${inp})
endforeach()
