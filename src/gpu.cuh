
#ifndef GRAPHCUDA_GPU_H
#define GRAPHCUDA_GPU_H
#include "graph.h"

#define EDGE_WISE_SHARED_MEMORY false // Requires EDGE_WISE_BELLMAN_FORD = true
static const bool ASYNC_MEMORY = false; 
static const bool ELEMENT_WISE_TIMING = true; 
static const bool EDGE_WISE_BELLMAN_FORD = false; 
static const bool EDGE_WISE_DELTA_STEPPING = true; 

void initializeBellmanFord(CSR graphCSR, int source, int num_nodes, int* d, int* p);

void initializeDeltaStepping(CSR graphCSR, int source, int max_node, int* d, int* p, int Delta);

#endif
