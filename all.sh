#!/bin/bash

if [ "$1" == "build" ]; then
  if [ -d "build" ]; then
    rm -r build
  fi
  mkdir build
  cd build
  cmake -DCMAKE_BUILD_TYPE=Release ..
elif [ "$1" == "run-cpu" ]; then
  cd build
  make
  if [[ "$HOSTNAME" == *"login"* ]]; then
    sbatch job-cpu
  else
    ./GraphAlgorithmsWithCUDA ../datasets/roadNet-CA.txt Dijkstra 0
  fi
elif [ "$1" == "run-gpu-batch" ]; then
  cd build
  make
  sbatch job-gpu
elif [ "$1" == "run-gpu-salloc" ]; then
  cd build
  make
  ./GraphAlgorithmsWithCUDA ../datasets/roadNet-CA.txt Bellman-Ford 0
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
    python3 src/GraphAlgorithmsPython.py /datasets/roadNet-CA.txt
    echo "Writing outputs to python_shortest_path_outs.txt"
  fi
fi