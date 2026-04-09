# CMakeLists.txt for GMRES solver

# Find MPI (optional, but required for gmres solver)
find_package(MPI COMPONENTS C)

if(MPI_C_FOUND)
  message(STATUS "MPI found - building GMRES solver")
  
  # Create GMRES solver library
  add_library(gmres_solver STATIC
    gmres.c
  )
  
  target_include_directories(gmres_solver
    PUBLIC
      ${CMAKE_CURRENT_SOURCE_DIR}
      ${MPI_C_INCLUDE_DIRS}
  )
  
  target_link_libraries(gmres_solver
    PUBLIC
      MPI::MPI_C
      m
  )
  
  # Install header
  install(FILES gmres.h
    DESTINATION include/solver
  )
  
  # Install library
  install(TARGETS gmres_solver
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
  )
  
  # Build test program if tests are enabled
  if(BUILD_TESTS)
    add_executable(gmres_test
      gmres_test.c
    )
    
    target_link_libraries(gmres_test
      PRIVATE
        gmres_solver
        MPI::MPI_C
        m
    )
    
    # Add test
    add_test(NAME gmres_solver_test
      COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 1 
              ${MPIEXEC_PREFLAGS} $<TARGET_FILE:gmres_test> 100
              ${MPIEXEC_POSTFLAGS}
    )
    
    add_test(NAME gmres_solver_test_parallel
      COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2
              ${MPIEXEC_PREFLAGS} $<TARGET_FILE:gmres_test> 100
              ${MPIEXEC_POSTFLAGS}
    )
  endif()
  
else()
  message(WARNING "MPI not found - skipping GMRES solver")
endif()
