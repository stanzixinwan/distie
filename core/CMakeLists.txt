cmake_minimum_required(VERSION 3.18)
project(distie_core LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(MSVC)
  add_compile_options(/W3 /EHsc /utf-8)
endif()

add_library(distie_memory STATIC src/block_pool.cpp)
target_include_directories(distie_memory PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
if(MSVC)
  target_compile_definitions(distie_memory PRIVATE _CRT_SECURE_NO_WARNINGS)
endif()

# --- Python module (pybind11 installed into the active interpreter) ---
find_package(Python COMPONENTS Interpreter Development REQUIRED)
execute_process(
  COMMAND "${Python_EXECUTABLE}" -c "import pybind11; print(pybind11.get_cmake_dir())"
  OUTPUT_VARIABLE pybind11_DIR
  OUTPUT_STRIP_TRAILING_WHITESPACE
  RESULT_VARIABLE pybind11_DIR_RESULT
)
if(NOT pybind11_DIR_RESULT EQUAL 0)
  message(FATAL_ERROR "pybind11 is not installed. Run: python -m pip install pybind11")
endif()
find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(distie_core python/bindings.cpp)
target_link_libraries(distie_core PRIVATE distie_memory)
set_target_properties(distie_core PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
  LIBRARY_OUTPUT_DIRECTORY_RELEASE "${CMAKE_BINARY_DIR}"
  RUNTIME_OUTPUT_DIRECTORY_RELEASE "${CMAKE_BINARY_DIR}"
  LIBRARY_OUTPUT_DIRECTORY_DEBUG "${CMAKE_BINARY_DIR}"
  RUNTIME_OUTPUT_DIRECTORY_DEBUG "${CMAKE_BINARY_DIR}"
)

# --- C++ tests ---
include(FetchContent)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/refs/tags/v1.15.2.zip
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)
FetchContent_MakeAvailable(googletest)

enable_testing()
add_executable(block_pool_test tests/test_block_pool.cpp)
target_link_libraries(block_pool_test PRIVATE distie_memory GTest::gtest_main)
include(GoogleTest)
gtest_discover_tests(block_pool_test DISCOVERY_MODE PRE_TEST)
