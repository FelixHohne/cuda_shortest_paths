#include <chrono>
#include <cmath>
#include <string>
#include <cuda.h>
#include <fstream>
#include <iostream>
#include <random>
#include <vector>
#include "gpu.cuh"
#include <assert.h>
#include <bits/stdc++.h>
#include <stdio.h>
#include <sys/time.h>


#define NUM_THREADS 1024

double get_time(timeval& t1, timeval& t2){
    return (1000000.0*(t2.tv_sec-t1.tv_sec) + t2.tv_usec-t1.tv_usec)/1000.0;
}

__global__ void BellmanFord(int num_nodes, int num_edges, int* d_dists, int* d_preds, int* d_row_ptrs, int* d_neighbor_nodes, int* d_edge_weights) {

    int tid = threadIdx.x + blockDim.x * blockIdx.x;

    if (tid > num_nodes - 1) {
        return;
    }

    // each thread is responsible for a node u
    // for each edge (u, v) with weight w
    for (int i = d_row_ptrs[tid]; i < d_row_ptrs[tid + 1]; i++) {
        int v = d_neighbor_nodes[i];
        // if distance[tid] + w < distance[v]
        if (d_dists[tid] != INT_MAX && d_dists[tid] + d_edge_weights[i] < d_dists[v]) {
            // update d_dist and d_preds
            // distance[v] := distance[tid] + w
            

            // Note: Both of these operations need to be atomic; thus this is incorrect. 
            // Solution: Using long long int to update both simultaneously. 
            // https://stackoverflow.com/questions/64397044/how-do-i-apply-atomic-operation-for-struct-on-cuda
            // https://stackoverflow.com/questions/17411493/how-can-i-implement-a-custom-atomic-function-involving-several-variables
            // A third approach involves using undefined behavior via union types: 
            // https://stackoverflow.com/questions/5792704/convert-int2-to-long
            
            // d_dists[v] = d_dists[tid] + d_edge_weights[i];
            atomicExch(d_dists + v,  d_dists[tid] + d_edge_weights[i]);
            // predecessor[v] := tid
            
            // d_preds[v] = tid;
            atomicExch(d_preds + v, tid);
        }
        
    }
}

__global__ void initialize_dists_array(int* d_dists, int num_nodes, int source) {
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    if (tid >= num_nodes) {
        return;
    }
    
    d_dists[tid] = INT_MAX; 
    
    if (tid == 0) {
        d_dists[source] = 0;
    }
}

/**
 * Requires: no negative-weight cycles
 */
void initializeBellmanFord(CSR graphCSR, int source, int num_nodes, int* d, int* p) {

    struct timeval start, stop;

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

    gettimeofday(&start, 0);

    initialize_dists_array<<<blks, NUM_THREADS>>>(d_dists, graphCSR.numNodes, source);
    
    cudaDeviceSynchronize();
    gettimeofday(&stop, 0);
    double init_time = get_time(start, stop);
    printf("Bellman init time:  %3.1f ms \n", init_time);


    cudaMemcpy(d_row_ptrs, graphCSR.rowPointers, graphCSR.numNodes * sizeof(int), cudaMemcpyHostToDevice);

    cudaMemcpy(d_neighbor_nodes, graphCSR.neighborNodes, graphCSR.numEdges * sizeof(int), cudaMemcpyHostToDevice);

    cudaMemcpy(d_edge_weights, graphCSR.edgeWeights, graphCSR.numEdges * sizeof(int), cudaMemcpyHostToDevice);

    float total_BF_time = 0;
    for (int i = 0; i < graphCSR.numNodes - 1; i++) {
        gettimeofday(&start, 0);

        BellmanFord<<<blks, NUM_THREADS>>>( graphCSR.numNodes, graphCSR.numEdges, d_dists, d_preds, d_row_ptrs, d_neighbor_nodes, d_edge_weights);

        cudaDeviceSynchronize();
        gettimeofday(&stop, 0);
        total_BF_time += get_time(start, stop);
    }

    printf("Bellman kernels total time:  %3.1f ms \n", total_BF_time);
    printf("Bellman kernels avg time per node:  %3.1f ms \n", total_BF_time / (graphCSR.numNodes-1));

    cudaMemcpy(d, d_dists, graphCSR.numNodes * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(p, d_preds, graphCSR.numNodes * sizeof(int), cudaMemcpyDeviceToHost);
}



/**
 * Requires: no negative-weight cycles
 */
void initializeDeltaStepping(CSR graphCSR, int source, int num_nodes, int* d, int* p, int Delta) {
    
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
    
    initialize_dists_array<<<blks, NUM_THREADS>>>(d_dists, graphCSR.numNodes);

    cudaMemcpy(d_row_ptrs, graphCSR.rowPointers, graphCSR.numNodes * sizeof(int), cudaMemcpyHostToDevice);

    cudaMemcpy(d_neighbor_nodes, graphCSR.neighborNodes, graphCSR.numEdges * sizeof(int), cudaMemcpyHostToDevice);

    cudaMemcpy(d_edge_weights, graphCSR.edgeWeights, graphCSR.numEdges * sizeof(int), cudaMemcpyHostToDevice);

    for (int i = 0; i < graphCSR.numNodes - 1; i++) {
       BellmanFord<<<blks, NUM_THREADS>>>( graphCSR.numNodes, graphCSR.numEdges, d_dists, d_preds, d_row_ptrs, d_neighbor_nodes, d_edge_weights);
    }

    cudaMemcpy(d, d_dists, graphCSR.numNodes * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(p, d_preds, graphCSR.numNodes * sizeof(int), cudaMemcpyDeviceToHost);
}


typedef struct BucketElement {
    int bucketIndex;
    int nodeId;      
} BucketElement;


typedef struct ReqElement { 
    int nodeId; 
    int dist;
}


bool compareBucketElements(BucketElement &a, BucketElement &b) {
    return a.bucketIndex < b.bucketIndex;
}

/*
For correctness, need to sort B after this call.
*/
__global__ void relax(int v, int new_dist, BucketElement* B, int* dists, int* binCounter, int* sizeB, int delta) {
    // TODO: Figure out how tid and v are actually related. 
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    if (tid >= num_nodes) {
        return;
    }
    
    // TODO: Update the size of B used for termination. 
    if (new_dist < dists[v]) {
        B[v].bucketId = floor(new_dist / delta); 
        atomicAdd(binCounter +  bucketId, 1);
        atomicAdd(sizeB, 1);
    }

    dists[v] = new_dist; 
}

__global__ void initialize_light_heavy_arrays(int* d_light, int* d_heavy, int* d_neighbor_nodes, int* d_edge_weights, int num_edges, int Delta) {
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    if (tid >= num_edges) {
        return;
    }

    if (d_edge_weights[tid] > Delta) {
        d_heavy[tid] = d_neighbor_nodes[tid];
        d_light[tid] = -1;
    } else {
        d_light[tid] = d_neighbor_nodes[tid];
        d_heavy[tid] = -1;
    }
}

__global__ void computeLightReq(int* d_B, int* d_light, int* d_row_ptrs, int* d_edge_weights, int num_nodes, int i, int* d_Req_size) {
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    if (tid >= num_nodes) {
        return;
    }

    // Req = {(w, dist[v] + dist(v, w)) : v \in B[i] && (v, w) \in light(v)}
    if (d_B[tid].bucketIndex != i) {
        return;
    }

    int v = d_B[tid].nodeId;
    for (int i = d_row_ptrs[v]; i < d_row_ptrs[v + 1]; i++) {
        if (d_light[i] != -1) {
            ReqElement elem = {
                .nodeId = d_light[i],
                .dist = dists[v] + d_edge_weights[i]
            };
            int old_index = atomicAdd(&d_Req_size, 1);
            d_Req[old_index] = elem;
        }
    }

    
}





void initializeDeltaStepping(CSR graphCSR, int source, int num_nodes, int* d, int* p, int Delta) {

    int node_blks = (graphCSR.numNodes + NUM_THREADS - 1) / NUM_THREADS;
    int edge_blks = (graphCSR.numEdges + NUM_THREADS - 1) / NUM_THREADS;

    int* d_row_ptrs;
    int* d_neighbor_nodes;
    int* d_edge_weights;

    cudaMemcpy(d_row_ptrs, graphCSR.rowPointers, graphCSR.numNodes * sizeof(int), cudaMemcpyHostToDevice);

    cudaMemcpy(d_neighbor_nodes, graphCSR.neighborNodes, graphCSR.numEdges * sizeof(int), cudaMemcpyHostToDevice);

    cudaMemcpy(d_edge_weights, graphCSR.edgeWeights, graphCSR.numEdges * sizeof(int), cudaMemcpyHostToDevice);
    

    int* d_S;
    int validS = 0; 
    cudaMalloc((void**) &d_S, graphCSR.numNodes * sizeof(int));

    BucketElement* d_B; 
    cudaMalloc((void**) &d_B, graphCSR.numNodes * sizeof(BucketElement));

    // (device) total size of all buckets
    int* dSizeB;
    cudaMalloc((void**) &dSizeB, sizeof(int));
    cudaMemset((void**) &dSizeB, 0, sizeof(int));

    // (host) total size of all buckets
    int hSizeB = 0;

    initialize_dists_array<<<node_blks, NUM_THREADS>>>(d_dists, graphCSR.numNodes, source);

    /*
    d_light and d_heavy are arrays of the same length as graphCSR.numEdges
    They contain a valid node id if the neighbor node corresponds to
    a light or heavy edge of the vertex, otherwise -1.
    Essentially a masked version of CSR for only light or only 
    heavy edges. 
    */
    int* d_light;
    int* d_heavy;
    cudaMalloc((void**) &d_light, graphCSR.numEdges * sizeof(int));
    cudaMalloc((void**) &d_heavy, graphCSR.numEdges * sizeof(int));
    initialize_light_heavy_arrays<<<edge_blks, NUM_THREADS>>>(d_light, d_heavy, d_neighbor_nodes, d_edge_weights, graphCSR.numEdges, Delta);

    int i = 0; 

    // used to test isEmpty(B[i]) 
    int* bi_size = new int[1];
    bi_size[0] = 1;
    int* d_bi_size; 

    cudaMalloc((void**) &d_bi_size, sizeof(int)); 
    cudaMemset((void**) &d_bi_size, 0, sizeof(int));

    // Device Req array
    // Elements up to d_Req_size are correct, beyond is garbage
    ReqElement* d_Req; 
    cudaMalloc((void**) &d_Req, graphCSR.numEdges * sizeof(ReqElement)); 

    int* d_Req_size; 
    cudaMalloc((void**) &d_Req_size, sizeof(int)); 
    cudaMemset((void**) &d_Req_size, 0, sizeof(int));

    while (hSizeB > 0) {

        validS = 0; 

        while (bi_size[0] > 0) {
            
            computeLightReq<<<node_blks, NUM_THREADS>>>(d_B, d_light, d_row_ptrs, d_edge_weights, num_nodes, i, d_Req_size);
            

        }



        i++;
        cudaMemcpy(hSizeB, dSizeB, sizeof(int), cudaMemcpyDeviceToHost);
    }


    cudaMemcpy(d, d_dists, graphCSR.numNodes * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(p, d_preds, graphCSR.numNodes * sizeof(int), cudaMemcpyDeviceToHost);


}