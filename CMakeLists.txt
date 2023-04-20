cmake_minimum_required(VERSION 3.16)

project(graph_cuda)
set(CMAKE_CXX_STANDARD 20)
add_executable(graph_cuda src/main.cpp src/loading.cpp src/loading.h src/graph.cpp src/graph.h src/serial.cpp src/serial.h src/gpu.cuh src/gpu.cpp)
# set(CMAKE_CXX_FLAGS_RELEASE "-O3")

# Copy the job scripts
configure_file(job-cpu job-cpu COPYONLY)