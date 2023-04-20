
#ifndef GRAPHCUDA_GPU_H
#define GRAPHCUDA_GPU_H

#include "graph.h"

std::pair<int*, int*> initializeBellmanFord(CSR graphCSR, int source);

#endif