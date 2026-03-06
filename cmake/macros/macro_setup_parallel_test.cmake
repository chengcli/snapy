# A small macro used for setting up the build of a test.
#
# Usage: setup_parallel_test(name)

string(TOLOWER ${CMAKE_BUILD_TYPE} buildl)
string(TOUPPER ${CMAKE_BUILD_TYPE} buildu)

macro(setup_parallel_test namel cores)
  add_executable(${namel}.${buildl} ${namel}.cpp)

  set_target_properties(${namel}.${buildl}
                        PROPERTIES COMPILE_FLAGS ${CMAKE_CXX_FLAGS_${buildu}})

  target_include_directories(
    ${namel}.${buildl}
    PRIVATE ${CMAKE_BINARY_DIR}
            ${KINTERA_INCLUDE_DIR}
            ${SNAP_INCLUDE_DIR}
            ${HARP_INCLUDE_DIR}
            ${NETCDF_INCLUDES}
            ${TORCH_INCLUDE_DIR}
            ${TORCH_API_INCLUDE_DIR}
            ${CMAKE_SOURCE_DIR}/external/gloo)

  target_link_libraries(${namel}.${buildl}
    PRIVATE snapy::snap
            $<IF:$<BOOL:${CUDAToolkit_FOUND}>,snapy::snap_cu,>
            gtest_main)

  add_test(NAME ${namel}.${buildl} COMMAND pd-run ${cores} ${namel}.${buildl})
endmacro()
