# Specify the minimum CMake version required
cmake_minimum_required(VERSION 3.0)

# Set the project name
project(my_program)
  
# Set the path to the compiler
set(MXCC_PATH $ENV{MACA_PATH})
set(CMAKE_CXX_COMPILER ${MXCC_PATH}/mxgpu_llvm/bin/mxcc)

# Set the compiler flags
set(MXCC_COMPILE_FLAGS -x maca)
add_compile_options(${MXCC_COMPILE_FLAGS})

# Add source files
File(GLOB SRCS src/*.cpp main.cpp)
add_executable(my_program ${SRCS})

# Set the include paths
target_include_directories(my_program PRIVATE include)
