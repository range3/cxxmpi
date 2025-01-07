cmake_minimum_required(VERSION 3.14)

include(cmake/prelude.cmake)

project(
    cxxmpi
    VERSION 0.1.5
    DESCRIPTION "Modern C++ wrapper for MPI"
    HOMEPAGE_URL "https://github.com/range3/cxxmpi"
    LANGUAGES CXX
)

include(cmake/project-is-top-level.cmake)
include(cmake/variables.cmake)

# ---- Declare library ----

add_library(cxxmpi_cxxmpi INTERFACE)
add_library(cxxmpi::cxxmpi ALIAS cxxmpi_cxxmpi)

set_property(
    TARGET cxxmpi_cxxmpi PROPERTY
    EXPORT_NAME cxxmpi
)

target_include_directories(
    cxxmpi_cxxmpi ${warning_guard}
    INTERFACE
    "\$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>"
)

target_compile_features(cxxmpi_cxxmpi INTERFACE cxx_std_20)

find_package(MPI REQUIRED CXX)
target_link_libraries(cxxmpi_cxxmpi INTERFACE MPI::MPI_CXX)

# ---- Install rules ----

if(NOT CMAKE_SKIP_INSTALL_RULES)
  include(cmake/install-rules.cmake)
endif()

# ---- Developer mode ----

if(NOT cxxmpi_DEVELOPER_MODE)
  return()
elseif(NOT PROJECT_IS_TOP_LEVEL)
  message(
      AUTHOR_WARNING
      "Developer mode is intended for developers of cxxmpi"
  )
endif()

include(cmake/dev-mode.cmake)
