cmake_minimum_required(VERSION 3.14)

project(cxxmpiTests LANGUAGES CXX C)

include(../cmake/project-is-top-level.cmake)
include(../cmake/folders.cmake)

# ---- Dependencies ----

if(PROJECT_IS_TOP_LEVEL)
  find_package(cxxmpi REQUIRED)
  enable_testing()
endif()

find_package(Catch2 REQUIRED)
include(Catch)

# ---- Options ----
option(cxxmpi_ENABLE_MPI_TESTS "Enable MPI-based tests" ON)

set(cxxmpi_MPI_TEST_PROCESSES "2,4" 
    CACHE STRING 
    "Number of processes for MPI tests (comma-separated)")


# ---- Tests ----
# Regular tests
file(GLOB_RECURSE TEST_SOURCES CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/source/*_test.cpp")

# MPI specific tests
file(GLOB_RECURSE MPI_TEST_SOURCES CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/source/*_mpitest.cpp")


# add_executable(cxxmpi_test
#     ${TEST_SOURCES}
# )
add_executable(cxxmpi_mpitest
    ${MPI_TEST_SOURCES}
    source/cxxmpi_mpitest_main.cpp
)
# target_link_libraries(
#     cxxmpi_test PRIVATE
#     cxxmpi::cxxmpi
#     Catch2::Catch2WithMain
# )
target_link_libraries(
    cxxmpi_mpitest PRIVATE
    cxxmpi::cxxmpi
    Catch2::Catch2
)
# target_compile_features(cxxmpi_test PRIVATE cxx_std_20)
target_compile_features(cxxmpi_mpitest PRIVATE cxx_std_20)

function(add_mpi_test TEST_NAME NUM_PROCS)
    add_test(
        NAME ${TEST_NAME}
        COMMAND ${MPIEXEC_EXECUTABLE} 
                ${MPIEXEC_NUMPROC_FLAG} ${NUM_PROCS}
                ${MPIEXEC_PREFLAGS}
                $<TARGET_FILE:cxxmpi_mpitest>
                [mpi]
                ${MPIEXEC_POSTFLAGS}
    )
    
    set_tests_properties(${TEST_NAME} PROPERTIES
        PROCESSORS ${NUM_PROCS}
    )
endfunction()

# Regular (non-MPI) tests
# catch_discover_tests(cxxmpi_test)

# MPI tests
if(cxxmpi_ENABLE_MPI_TESTS)
    string(REPLACE "," ";" PROCESS_NUMBERS "${cxxmpi_MPI_TEST_PROCESSES}")
    foreach(NUM_PROCS ${PROCESS_NUMBERS})
        add_mpi_test(mpi_test_${NUM_PROCS}procs ${NUM_PROCS})
    endforeach()
endif()


# ---- End-of-file commands ----

add_folders(Test)
