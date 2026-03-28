cmake_minimum_required(VERSION 3.18)
project(gpt_cpp)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(/usr/include/eigen3)

find_package(Python REQUIRED COMPONENTS Interpreter Development)

execute_process(
    COMMAND "${Python_EXECUTABLE}" -c "import pybind11; print(pybind11.get_cmake_dir())"
    OUTPUT_VARIABLE pybind11_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

find_package(pybind11 REQUIRED)

pybind11_add_module(gpt_cpp
    cpp/src/gpt.cpp
    cpp/src/bindings.cpp)

target_include_directories(gpt_cpp PRIVATE cpp/include)

target_compile_options(gpt_cpp PRIVATE -03 -march=native)

