# Install bc library

set(namel bc)
string(TOUPPER ${namel} nameu)

file(GLOB src_files
  *.cpp
  )

add_library(${namel}_${buildl}
    SHARED
    ${src_files}
    )

set_target_properties(${namel}_${buildl}
    PROPERTIES
    COMPILE_FLAGS "${CMAKE_CXX_FLAGS_${buildu}}"
    )

target_link_libraries(${namel}_${buildl}
    PUBLIC
    fmt::fmt
    yaml-cpp::yaml-cpp
    ${TORCH_LIBRARY}
    ${TORCH_CPU_LIBRARY}
    ${C10_LIBRARY}
    )

add_library(snapy::bc ALIAS ${namel}_${buildl})
