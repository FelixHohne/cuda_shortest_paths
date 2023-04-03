cmake_minimum_required(VERSION 3.24)

project(GraphAlgorithmsWithCUDA)
set(CMAKE_CXX_STANDARD 17)
add_executable(GraphAlgorithmsWithCUDA main.cpp graphLoading.cpp graphLoading.h)