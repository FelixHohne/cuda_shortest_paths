#!/bin/bash
CORRECTNESS="../../hw2/correctness"
CORRECT="../../hw2/correct"

if [ "$1" == "build" ]; then
  if [ -d "build" ]; then
    rm -r build
  fi
  mkdir build
  cd build
  cmake -DCMAKE_BUILD_TYPE=Release ..
elif [ "$1" == "run" ]; then
  cd build
  make
  ./GraphAlgorithmsWithCUDA ../datasets/roadNet-CA.txt Dijkstra 0
elif [ "$1" == "status" ]; then
  squeue -u sh2223
elif [ "$1" == "view" ]; then
  if [ -f build/slurm*.out ]; then
    cat build/slurm*.out
  else
    echo "Results not available"
  fi
elif [ "$1" == "salloc" ]; then
  salloc --nodes 1 --qos interactive --time 01:00:00 --constraint gpu --gpus 1 --account=m4341
elif [ "$1" == "mac" ]; then
  if [ -d "build" ]; then
    rm -r build
  fi
  mkdir build
  cd build
  cmake -DCMAKE_BUILD_TYPE=Release .. -DCMAKE_C_COgpuLER=/opt/homebrew/Cellar/gcc@11/11.3.0/bin/gcc-11 -DCMAKE_CXX_COgpuLER=/opt/homebrew/Cellar/gcc@11/11.3.0/bin/g++-11
elif [ "$1" == "check-correct" ]; then
  CORRECTNESS="../hw2/correctness"
  CORRECT="../hw2/correct"
  scp sh2223@perlmutter-p1.nersc.gov:~/cs5220-sp23/hw4/build/gpu.parts.out .
  # scp sh2223@perlmutter-p1.nersc.gov:~/cs5220-sp23/hw4/build/gpu-s-42-n-100000.parts.out .
  # scp sh2223@perlmutter-p1.nersc.gov:~/cs5220-sp23/hw4/build/gpu-s-91-n-100000.parts.out .
  python3 $CORRECTNESS/correctness-check.py gpu.parts.out $CORRECTNESS/verf.out
  # python3 $CORRECTNESS/correctness-check.py gpu-s-42-n-100000.parts.out $CORRECT/correct-s-42-n-100000.parts.out
  # python3 $CORRECTNESS/correctness-check.py gpu-s-91-n-100000.parts.out $CORRECT/correct-s-91-n-100000.parts.out
elif [ "$1" == "correct" ]; then
  cd build
  make correct
  ./correct -s 42 -n 100000 -o $CORRECT/correct-s-42-n-100000.parts.out
  ./correct -s 91 -n 100000 -o $CORRECT/correct-s-91-n-100000.parts.out
elif [ "$1" == "tar" ]; then
  cd build
  rm CS5220Group60_hw4.tar.gz
  rm -rf temp-tar
  cmake -DGROUP_NO=60 ..
  make package
  tar tfz CS5220Group60_hw4.tar.gz
  mkdir temp-tar
  cp CS5220Group60_hw4.tar.gz temp-tar
  cd temp-tar
  tar -xvf CS5220Group60_hw4.tar.gz
fi