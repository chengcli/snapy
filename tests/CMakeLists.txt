# ========== #
# Unit Tests #
# ========== #

enable_testing()

setup_test(test_eos)
setup_test(test_radiating_boundary)
setup_test(test_plm)
setup_test(test_weno)
setup_test(test_reconstruct)
setup_test(test_riemann)
setup_test(test_backward_substitution)
setup_test(test_scalar)
setup_test(test_internal_boundary)
setup_test(test_rectify)
setup_test(test_pull_2d)
setup_test(test_pull_3d)
setup_test(test_slab)
setup_test(test_cubed)
setup_test(test_cubed_sphere)
setup_test(test_coordinate)
setup_test(test_hydrostatic)
setup_test(test_hydro_options)
setup_test(test_hydro_ref_x1)
setup_test(test_output_dir)
setup_test(test_netcdf_utils)
setup_test(test_user_output)
setup_test(test_process_group)
setup_test(test_refine)
setup_test(test_diffusion)
setup_test(test_diffusion_moist)
setup_test(test_condensate_conservation)
setup_test(test_wall_saturation)
setup_test(test_forcing)
setup_parallel_test(test_exchange 2)

if(UCX_FOUND)
  add_test(
    NAME test_exchange_ucx
    COMMAND torchrun
            --no-python
            --nproc-per-node=2
            $<TARGET_FILE:test_exchange.${buildl}>
  )
  set_tests_properties(test_exchange_ucx PROPERTIES
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/tests
    ENVIRONMENT
      "BACKEND=ucx;BLOCKS_PER_PROCESS=3;EXPECT_LOCAL_NEIGHBOR=1;EXPECT_REMOTE_NEIGHBOR=1"
    LABELS "ucx;exchange"
  )

  if(CUDA)
    add_test(
      NAME test_exchange_ucx_cuda
      COMMAND ${Python3_EXECUTABLE}
              ${CMAKE_CURRENT_SOURCE_DIR}/run_exchange_backend.py
              --build-dir ${CMAKE_BINARY_DIR}
              --build-type ${buildl}
              --backend ucx
              --device cuda
    )
    set_tests_properties(test_exchange_ucx_cuda PROPERTIES
      WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/tests
      SKIP_RETURN_CODE 125
      LABELS "ucx;exchange;gpu"
    )
  endif()
endif()
#setup_test(test_read_topo)
#setup_cuda_test(test_thomas_solver)
#setup_test(test_aneos)

set_tests_properties(test_exchange.release PROPERTIES
  ENVIRONMENT
    "BLOCKS_PER_PROCESS=3;EXPECT_LOCAL_NEIGHBOR=1;EXPECT_REMOTE_NEIGHBOR=1"
)

add_test(
  NAME test_radiating_boundary_python
  COMMAND ${Python3_EXECUTABLE}
          ${CMAKE_CURRENT_SOURCE_DIR}/test_radiating_boundary_python.py
)
set_tests_properties(test_radiating_boundary_python PROPERTIES
  WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/tests
  LABELS "python;boundary"
)
set_property(GLOBAL APPEND PROPERTY SNAPY_PYTHON_TESTS test_radiating_boundary_python)

# Every python ctest that imports snapy goes through this, so it is enrolled for
# SNAPY_TEST_PYTHONPATH below; the raw add_test ones drive a built binary and import none.
function(snapy_add_python_test name)
  cmake_parse_arguments(ARG "" "TIMEOUT" "LABELS" ${ARGN})
  add_test(
    NAME ${name}_python
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/${name}.py
  )
  set_tests_properties(${name}_python PROPERTIES
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/tests
    LABELS "${ARG_LABELS}"
  )
  if(ARG_TIMEOUT)
    set_tests_properties(${name}_python PROPERTIES TIMEOUT ${ARG_TIMEOUT})
  endif()
  set_property(GLOBAL APPEND PROPERTY SNAPY_PYTHON_TESTS ${name}_python)
endfunction()

snapy_add_python_test(test_mesh_exchange LABELS "python;exchange")
snapy_add_python_test(test_cubed_sphere_vertical_velocity_exchange
                      LABELS "python;exchange;cubed-sphere;diagnostic")
snapy_add_python_test(test_jit_user_forcing TIMEOUT 30 LABELS "python;forcing")

file(GLOB inputs *.yaml *.py)
# create tests folder if does not exist
file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/tests)
foreach(input ${inputs})
  get_filename_component(input_name ${input} NAME)
  execute_process(COMMAND ln -sf ${input} ${CMAKE_BINARY_DIR}/tests/${input_name})
endforeach()

if (DEFINED KINTERA_DATA_DIR AND EXISTS "${KINTERA_DATA_DIR}/nasa9.dat")
  execute_process(
    COMMAND ln -sf ${KINTERA_DATA_DIR}/nasa9.dat
            ${CMAKE_BINARY_DIR}/tests/nasa9.dat)
endif()

# ---- Python ctests: say which snapy they test -------------------------------------------
# `COMMAND ${Python3_EXECUTABLE} <test>.py` imports `snapy` through sys.path, so with nothing
# set these exercise whatever snapy is installed, not the tree they were configured from --
# a false red when the branch is fine and the install is stale, and a false green when the
# branch is broken and the install is fine. cmake builds only the C++ libraries here (the
# importable package comes from `pip install`), so there is nothing in the build tree to point
# at by default; give the directory that holds this build's installed snapy:
#     cmake -B build -DSNAPY_TEST_PYTHONPATH=/path/to/site-packages
set(SNAPY_TEST_PYTHONPATH "" CACHE PATH
    "PYTHONPATH for the *_python ctests; empty means they test the installed snapy")
get_property(SNAPY_PYTHON_TESTS GLOBAL PROPERTY SNAPY_PYTHON_TESTS)
if(SNAPY_TEST_PYTHONPATH)
  set_tests_properties(${SNAPY_PYTHON_TESTS} PROPERTIES
                       ENVIRONMENT "PYTHONPATH=${SNAPY_TEST_PYTHONPATH}")
  message(STATUS "python ctests import snapy from ${SNAPY_TEST_PYTHONPATH}")
else()
  message(WARNING
    "SNAPY_TEST_PYTHONPATH is unset: the *_python ctests will import the installed snapy, "
    "not this tree. Their verdicts do not constrain this branch.")
endif()

# ============= #
# Example Tests #
# ============= #
add_test(
  NAME test_straka
  COMMAND ${CMAKE_COMMAND}
          -Dbuildl=release
          -P ${CMAKE_CURRENT_SOURCE_DIR}/run_straka.cmake
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
          -P ${CMAKE_CURRENT_SOURCE_DIR}/run_shallow_splash.cmake
)

set_tests_properties(test_shallow_splash PROPERTIES
  WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/tests
)

if (NOT APPLE)
    add_test(
      NAME test_restart_multiblock
      COMMAND ${Python3_EXECUTABLE}
              ${CMAKE_CURRENT_SOURCE_DIR}/run_restart_multiblock.py
              --build-dir ${CMAKE_BINARY_DIR}
              --build-type release
    )

    set_tests_properties(test_restart_multiblock PROPERTIES
      WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/tests
      TIMEOUT 600
      LABELS "restart;examples"
    )

    add_test(
      NAME test_restart_cycle_limit
      COMMAND ${Python3_EXECUTABLE}
              ${CMAKE_CURRENT_SOURCE_DIR}/run_restart_cycle_limit.py
              --build-dir ${CMAKE_BINARY_DIR}
              --build-type release
    )

    set_tests_properties(test_restart_cycle_limit PROPERTIES
      WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/tests
      SKIP_RETURN_CODE 125
      TIMEOUT 600
      LABELS "restart;examples"
    )
endif()

if (DEFINED FULL_TESTS)
  setup_parallel_test(test_mesh_multi_block 2)

  add_test(
    NAME test_exchange_decomp
    COMMAND ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/run_exchange_decomp.py
            --build-dir ${CMAKE_BINARY_DIR}
            --build-type release
  )

  set_tests_properties(test_exchange_decomp PROPERTIES
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/tests
    PROCESSORS 6
    RUN_SERIAL TRUE
    TIMEOUT 1200
    LABELS "decomp;gloo"
  )

  add_test(
    NAME test_shallow_xy_decomp
    COMMAND ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/run_shallow_xy_decomp.py
            --build-dir ${CMAKE_BINARY_DIR}
            --build-type release
  )

  set_tests_properties(test_shallow_xy_decomp PROPERTIES
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/tests
    PROCESSORS 4
    RUN_SERIAL TRUE
    TIMEOUT 1200
    LABELS "decomp;gloo"
  )

  add_test(
    NAME test_shallow_splash_decomp
    COMMAND ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/run_shallow_splash_decomp.py
            --build-dir ${CMAKE_BINARY_DIR}
            --build-type release
            --backend gloo
  )

  set_tests_properties(test_shallow_splash_decomp PROPERTIES
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/tests
    PROCESSORS 6
    RUN_SERIAL TRUE
    TIMEOUT 2400
    LABELS "decomp;gloo"
  )

  add_test(
    NAME test_shallow_splash_ucx_cuda_decomp
    COMMAND ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/run_shallow_splash_decomp.py
            --build-dir ${CMAKE_BINARY_DIR}
            --build-type release
            --backend ucx
            --device cuda
  )

  set_tests_properties(test_shallow_splash_ucx_cuda_decomp PROPERTIES
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/tests
    SKIP_RETURN_CODE 125
    PROCESSORS 2
    RUN_SERIAL TRUE
    TIMEOUT 2400
    LABELS "gpu;ucx;long"
  )
endif()
