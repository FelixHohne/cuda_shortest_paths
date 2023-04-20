#!/bin/bash

if [ "$1" == "build" ]; then
  # remove existing build directory if one exists
  if [ -d "build" ]; then
    rm -r build
  fi
  # CMake prefers out-of-tree builds- make a build directory
  mkdir build
  cd build
  if [ "$USER" == "shiyuanhuang" ]; then
    # configure local build on mac
    cmake -DCMAKE_BUILD_TYPE=Release .. -DCMAKE_C_COMPILER=/opt/homebrew/Cellar/gcc@11/11.3.0/bin/gcc-11 -DCMAKE_CXX_COMPILER=/opt/homebrew/Cellar/gcc@11/11.3.0/bin/g++-11
  else
    cmake -DCMAKE_BUILD_TYPE=Release ..
  fi
elif [ "$1" == "run-cpu" ]; then
  cd build
  make
  if [[ "$HOSTNAME" == *"login"* ]]; then
    sbatch job-cpu
  else
    ./graph_cuda ../datasets/roadNet-CA.txt Dijkstra 0
  fi
elif [ "$1" == "run-gpu-batch" ]; then
  cd build
  make
  sbatch job-gpu
elif [ "$1" == "run-gpu-salloc" ]; then
  cd build
  make
  ./graph_cuda ../datasets/roadNet-CA.txt Bellman-Ford 0
  head -n 100 ../gpu_bellman_ford.txt > ../gpu_bellman_ford_100.txt
elif [ "$1" == "status" ]; then
  squeue -u $USER
elif [ "$1" == "view" ]; then
  if [ -f build/slurm*.out ]; then
    cat build/slurm*.out
  else
    echo "Results not available"
  fi
elif [ "$1" == "salloc" ]; then
  salloc --nodes 1 --qos interactive --time 01:00:00 --constraint gpu --gpus 1 --account=m4341
elif [ "$1" == "check-correct" ]; then
  echo "Correctness check has not yet been implemented."
elif [ "$1" == "correct" ]; then
  # this is intended to generated the correct output
  # haven't figured out how to run this on Perlmutter yet since I'm not sure
  # if we're allowed to install conda, so run locally for now
  HOSTNAME=$(uname -n)
  if [[ "$HOSTNAME" == *"login"* ]]; then
    echo "correct not supported on Perlmutter, please run on local machine."
    # module load python
  else
    python3 src/correct.py /datasets/roadNet-CA.txt
    echo "Writing outputs to python_shortest_path_outs.txt"
  fi
fi