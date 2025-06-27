# A small macro used for setting up the build of a problem.
#
# Usage: setup_problem(name)

string(TOLOWER ${CMAKE_BUILD_TYPE} buildl)
string(TOUPPER ${CMAKE_BUILD_TYPE} buildu)

macro(setup_problem namel)
  add_executable(${namel}.${buildl} ${namel}.cpp)

  set_target_properties(
    ${namel}.${buildl}
    PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
               COMPILE_FLAGS ${CMAKE_CXX_FLAGS_${buildu}})

  target_include_directories(
    ${namel}.${buildl}
    PRIVATE ${CMAKE_BINARY_DIR} ${KINTERA_INCLUDE_DIR} ${SNAP_INCLUDE_DIR}
            ${TORCH_INCLUDE_DIR} ${TORCH_API_INCLUDE_DIR})

  if(APPLE)
    target_link_libraries(
      ${namel}.${buildl}
      PRIVATE ${KINTERA_LIBRARY} ${VAPORS_LIBRARY} snapy::bc snapy::snap
              $<IF:$<BOOL:${CUDAToolkit_FOUND}>,snapy::snap_cu,>)
  else()
    target_link_libraries(
      ${namel}.${buildl}
      PRIVATE ${KINTERA_LIBRARY}
              -Wl,--no-as-needed
              ${VAPORS_LIBRARY}
              snapy::bc
              -Wl,--as-needed
              snapy::snap
              $<IF:$<BOOL:${CUDAToolkit_FOUND}>,snapy::snap_cu,>)
  endif()
endmacro()
