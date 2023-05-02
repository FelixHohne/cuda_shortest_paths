
#ifndef GRAPHCUDA_GPU_H
#define GRAPHCUDA_GPU_H

#include "graph.h"

void initializeBellmanFord(CSR graphCSR, int source, int num_nodes, int* d, int* p);

void initializeDeltaStepping(CSR graphCSR, int source, int num_nodes, int* d, int* p, int Delta);

#endif