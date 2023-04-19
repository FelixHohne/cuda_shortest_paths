cmake_minimum_required(VERSION 3.16)
enable_language(CUDA)

project(GraphAlgorithmsWithCUDA)
set(CMAKE_CXX_STANDARD 20)
add_executable(GraphAlgorithmsWithCUDA src/main.cpp src/GraphLoading.cpp src/GraphLoading.h src/GraphRepresentations.cpp src/GraphRepresentations.h src/GraphAlgorithms.cpp src/GraphAlgorithms.h src/GraphAlgorithmsGPU.cuh src/GraphAlgorithmsGPU.cu)
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -arch=sm_60")

# Copy the job scripts
configure_file(job-cpu job-cpu COPYONLY)
configure_file(job-gpu job-gpu COPYONLY)