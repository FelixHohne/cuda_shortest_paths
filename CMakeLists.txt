cmake_minimum_required(VERSION 3.24)

project(GraphAlgorithmsWithCUDA)
set(CMAKE_CXX_STANDARD 20)
add_executable(GraphAlgorithmsWithCUDA src/main.cpp src/GraphLoading.cpp src/GraphLoading.h src/GraphRepresentations.cpp src/GraphRepresentations.h src/GraphAlgorithms.cpp src/GraphAlgorithms.h src/GraphAlgorithmsGPU.cu src/GraphAlgorithmsGPU.h)
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -arch=sm_60")