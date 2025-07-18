# A small macro used for setting up the build of a test.
#
# Usage: setup_test(name)

string(TOLOWER ${CMAKE_BUILD_TYPE} buildl)
string(TOUPPER ${CMAKE_BUILD_TYPE} buildu)

macro(setup_test namel)
  add_executable(${namel}.${buildl} ${namel}.cpp)

  set_target_properties(${namel}.${buildl}
                        PROPERTIES COMPILE_FLAGS ${CMAKE_CXX_FLAGS_${buildu}})

  target_include_directories(
    ${namel}.${buildl}
    PRIVATE ${CMAKE_BINARY_DIR} ${KINTERA_INCLUDE_DIR} ${SNAP_INCLUDE_DIR}
            ${NETCDF_INCLUDES} ${TORCH_INCLUDE_DIR} ${TORCH_API_INCLUDE_DIR})

  if(APPLE)
    target_link_libraries(
      ${namel}.${buildl} PRIVATE snapy::snap ${VAPORS_LIBRARY} snapy::bc
                                 gtest_main)
  else()
    target_link_libraries(
      ${namel}.${buildl}
      PRIVATE snapy::snap
              -Wl,--no-as-needed
              ${VAPORS_LIBRARY}
              snapy::bc
              $<IF:$<BOOL:${CUDAToolkit_FOUND}>,${KINTERA_CUDA_LIBRARY},>
              $<IF:$<BOOL:${CUDAToolkit_FOUND}>,snapy::snap_cu,>
              -Wl,--as-needed
              gtest_main)
  endif()

  add_test(NAME ${namel}.${buildl} COMMAND ${namel}.${buildl})
endmacro()
