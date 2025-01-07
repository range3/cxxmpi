include(CMakeFindDependencyMacro)

find_dependency(MPI REQUIRED CXX)

include("${CMAKE_CURRENT_LIST_DIR}/cxxmpiTargets.cmake")

set(cxxmpi_FOUND TRUE)
