#include "GraphAlgorithmsGPU.cuh"

// a dummy version of GraphAlgorithmsGPU.cu for CPU-only builds

std::pair<int*, int*> initializeBellmanFord(CSR graphCSR, int source) {
    int* dists = new int[1];
    int* preds = new int[1];
    return std::make_pair(dists, preds);
}