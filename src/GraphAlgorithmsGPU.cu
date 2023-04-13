#include "common.h"
#include <chrono>
#include <cmath>
#include <string>
#include <cuda.h>
#include <fstream>
#include <iostream>
#include <random>
#include <thrust>
#include <vector>
#include "GraphRepresentations.h"
#include <bits/stdc++.h>


#define NUM_THREADS 1024

__global__ void BellmanFord(int num_nodes, int num_edges, int* d_dists, int* d_preds, int* d_row_ptrs, int* d_neighbor_nodes, int* edge_weights) {

    int tid = threadIdx.x + blockDim.x * blockIdx.x;

    if (tid >= num_nodes - 1) {
        return;
    }
    
    // for each edge (u, v) with weight w in edges do
            // if distance[u] + w < distance[v] then
            //     distance[v] := distance[u] + w
            //     predecessor[v] := u

}


std::Pair<int*, int*> initializeBellmanFord(CSR graphCSR, int source) {
    int* d_dists;
    int* d_preds;
    int* d_row_ptrs;
    int* d_neighbor_nodes;
    int* d_edge_weights;
    int blks = (graphCSR.numNodes + NUM_THREADS - 1) / NUM_THREADS;

    cudaMalloc((void**) &d_dists, graphCSR.numNodes * sizeof(int));
    cudaMalloc((void**) &d_preds, graphCSR.numNodes * sizeof(int));
    cudaMalloc((void**) &d_row_ptrs, graphCSR.numNodes * sizeof(int));
    cudaMalloc((void**) &d_neighbor_nodes, graphCSR.numEdges * sizeof(int));
    cudaMalloc((void**) &d_edge_weights, graphCSR.numEdges * sizeof(int));
    
    cudaMemset(d_dists, INT_MAX, graphCSR.numNodes * sizeof(int)); 
    
    // Sets d_dists[source] = 0. TODO check pointer arithmetic. 
    cudaMemset(d_dists + source, 0, sizeof(int));

    for (int i = 0; i < graphCSR.numNodes - 1; i++) {
       BellmanFord<<<blks, NUM_THREADS>>>( graphCSR.numNodes, graphCSR.numEdges, d_dists, d_preds, d_row_ptrs, d_neighbor_nodes, d_edge_weights);
    }

    int* dists = int[graphCSR.numNodes];
    int* preds = int[graphCSR.numNodes];
    cudaMemcpy(dists, d_dists, graphCSR.numNodes * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(preds, d_preds, graphCSR.numNodes * sizeof(int), cudaMemcpyDeviceToHost);

    return std::make_pair(dists, preds); 
}
