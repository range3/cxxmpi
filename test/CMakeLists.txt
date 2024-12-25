cmake_minimum_required(VERSION 3.14)

project(cxxmpiTests LANGUAGES CXX)

include(../cmake/project-is-top-level.cmake)
include(../cmake/folders.cmake)

# ---- Dependencies ----

if(PROJECT_IS_TOP_LEVEL)
  find_package(cxxmpi REQUIRED)
  enable_testing()
endif()

find_package(Catch2 REQUIRED)
include(Catch)

# ---- Tests ----

add_executable(cxxmpi_test source/cxxmpi_test.cpp)
target_link_libraries(
    cxxmpi_test PRIVATE
    cxxmpi::cxxmpi
    Catch2::Catch2WithMain
)
target_compile_features(cxxmpi_test PRIVATE cxx_std_20)

catch_discover_tests(cxxmpi_test)

# ---- End-of-file commands ----

add_folders(Test)
