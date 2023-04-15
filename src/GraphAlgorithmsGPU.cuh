
#ifndef GRAPHALGORITHMSWITHCUDA_GRAPHALGORITHMSGPU_H
#define GRAPHALGORITHMSWITHCUDA_GRAPHALGORITHMSGPU_H

#include "GraphRepresentations.h"

std::pair<int*, int*> initializeBellmanFord(CSR graphCSR, int source);

#endif