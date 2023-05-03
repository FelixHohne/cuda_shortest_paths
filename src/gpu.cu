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
#include <thrust/sort.h>
#include <thrust/execution_policy.h>


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

__global__ void bellman_initialize_dists_array(int* d_dists, int num_nodes, int source) {
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

    bellman_initialize_dists_array<<<blks, NUM_THREADS>>>(d_dists, graphCSR.numNodes, source);
    
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


// ___________________________________________________________________


// Initializes d_dists to INT_MAX 
// Sets source distance to 0 
// Clears B[i]
// Adds source to B[i]. 
__global__ void delta_stepping_initialize(int* d_dists, int* d_B, int num_nodes, int source) {
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    if (tid >= num_nodes) {
        return;
    }
    
    d_dists[tid] = INT_MAX; 
    d_B[tid] = -1;
    
    if (tid == source) {
        d_dists[source] = 0;
        d_B[source] = 0; 
    }
}

typedef struct ReqElement { 
    int nodeId; 
    int dist;
    __host__ __device__
    bool operator < (const ReqElement &b) const {
        return nodeId < b.nodeId || dist < b.dist;
    }
} ReqElement;


__global__ void relax(int* d_B, ReqElement* d_Req, int* d_dists, int d_Req_size, int delta) {
    int tid = threadIdx.x + blockDim.x * blockIdx.x;

    if (tid > d_Req_size) {
        return;
    }

    if (tid == 0 || d_Req[tid].nodeId > d_Req[tid - 1].nodeId) {
        int v = d_Req[tid].nodeId;
        int new_dist = d_Req[tid].dist;
        if (new_dist < d_dists[v]) {
            // printf("d_dists[v]: %d, new_dist: %d\n", d_dists[v], new_dist);
            d_B[v] = floor((double) new_dist / delta);
            d_dists[v] = new_dist;
        }
    }
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

// Node-based parallelization.
__global__ void computeLightReq(int* d_B, int* d_light, int* d_row_ptrs, int* d_edge_weights, int num_nodes, int i, int* d_Req_size, int* d_dists, ReqElement* d_Req) {
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    if (tid >= num_nodes) {
        return;
    }
    // Req = {(w, dist[v] + dist(v, w)) : v \in B[i] && (v, w) \in light(v)}
    if (d_B[tid] != i) {
        return;
    }

    int v = tid;
    for (int i = d_row_ptrs[v]; i < d_row_ptrs[v + 1]; i++) {
        if (d_light[i] != -1) {
            ReqElement elem = {
                .nodeId = d_light[i],
                .dist = d_dists[v] + d_edge_weights[i]
            };
            int old_index = atomicAdd(d_Req_size, 1);
            d_Req[old_index] = elem;
        }
    }

    for (int j = 0; j < d_Req_size[0]; j++) {
        printf("curReq: %d\n", d_Req[j].nodeId); 
    }

    printf("Hello World\n");


}

__global__ void printB(int* d_B, int num_nodes) {

    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    if (tid >= num_nodes) {
        return;
    }

    printf("d_B element %d: %d", tid, d_B[tid]);
    printf("\n");
}
__global__ void computeHeavyReq(int* d_S, int* d_heavy, int* d_row_ptrs, int* d_edge_weights, int num_nodes, int i, int* d_Req_size, int* d_dists, ReqElement* d_Req) {
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    if (tid >= num_nodes) {
        return;
    }

    // Req = {(w, dist[v] + dist(v, w)) : v \in B[i] && (v, w) \in light(v)}
    if (d_S[tid] != i) {
        return;
    }

    int v = tid;
    for (int i = d_row_ptrs[v]; i < d_row_ptrs[v + 1]; i++) {
        if (d_heavy[i] != -1) {
            ReqElement elem = {
                .nodeId = d_heavy[i],
                .dist = d_dists[v] + d_edge_weights[i]
            };
            int old_index = atomicAdd(d_Req_size, 1);
            d_Req[old_index] = elem;
        }
    }
    
}

// S = emptyset
__global__ void clear_S(int* d_S, int num_nodes) {
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    if (tid >= num_nodes) {
        return;
    }
    d_S[tid] = -1;
}


// S = S \Cup B[i]; B[i] = emptyset 
__global__ void update_S_clear_B_i(int* d_B, int* d_S, int i, int num_nodes) {
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    if (tid >= num_nodes) {
        return;
    }

    if (d_B[tid] == i) {
        d_S[tid] = tid;
        d_B[tid] = -1; 
    }
}

__global__ void is_empty_B(int* d_B, int num_nodes, bool* is_empty) {
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    if (tid >= num_nodes) {
        return;
    }

    if (d_B[tid] != -1) {
        is_empty[0] = false;
    }
}

__global__ void is_empty_B_i(int* d_B, int num_nodes, bool* is_empty, int i) {
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    
    if (tid >= num_nodes) {
        return;
    }

    if (d_B[tid] == i) {
        printf("is empty false\n");
        is_empty[0] = false;
    }

}

void initializeDeltaStepping(CSR graphCSR, int source, int max_node, int* d, int* p, int Delta) {

    int num_nodes = max_node + 1; 
    std :: cout << "max node: " << max_node << "num nodes:" << num_nodes << std :: endl;
    std::cout << "Begin CUDA Delta Stepping" << std::endl;
    int node_blks = (graphCSR.numNodes + NUM_THREADS - 1) / NUM_THREADS;
    int edge_blks = (graphCSR.numEdges + NUM_THREADS - 1) / NUM_THREADS;

    int* d_row_ptrs;
    int* d_neighbor_nodes;
    int* d_edge_weights;

    cudaMalloc((void**) &d_row_ptrs, (graphCSR.numNodes + 1) * sizeof(int));
    cudaMemcpy(d_row_ptrs, graphCSR.rowPointers, (graphCSR.numNodes + 1) * sizeof(int), cudaMemcpyHostToDevice);

    cudaMalloc((void**) &d_neighbor_nodes, (graphCSR.numEdges) * sizeof(int));
    cudaMemcpy(d_neighbor_nodes, graphCSR.neighborNodes, graphCSR.numEdges * sizeof(int), cudaMemcpyHostToDevice);

    cudaMalloc((void**) &d_edge_weights, (graphCSR.numEdges) * sizeof(int));
    cudaMemcpy(d_edge_weights, graphCSR.edgeWeights, graphCSR.numEdges * sizeof(int), cudaMemcpyHostToDevice);    

    int* d_S;
    cudaMalloc((void**) &d_S, graphCSR.numNodes * sizeof(int));

    // Invariant: non-dense pack. 
    // Invariant: -1 means element is empty. 
    int* d_B; 
    cudaMalloc((void**) &d_B, graphCSR.numNodes * sizeof(int));

    int* d_dists;
    cudaMalloc((void**) &d_dists, graphCSR.numNodes * sizeof(int));
    int* d_preds;
    cudaMalloc((void**) &d_preds, graphCSR.numNodes * sizeof(int));

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

    // Device Req array
    // Elements up to d_Req_size are correct, beyond is garbage
    ReqElement* d_Req; 
    cudaMalloc((void**) &d_Req, graphCSR.numEdges * sizeof(ReqElement)); 

    int* d_Req_size; 
    cudaMallocManaged(&d_Req_size, sizeof(int)); 
    d_Req_size[0] = 0; 

    bool* is_empty; 
    cudaMallocManaged(&is_empty, sizeof(bool)); 
    is_empty[0] = false; 

    delta_stepping_initialize<<<node_blks, NUM_THREADS>>>(d_dists, d_B, graphCSR.numNodes, source);
    std :: cout << "Before while loop" << std :: endl;

    printB<<<node_blks, NUM_THREADS>>>(d_B, num_nodes);

    while (!is_empty[0]) {
        clear_S<<<node_blks, NUM_THREADS>>>(d_S, num_nodes);
    
        // Checks if B_i is not empty. 
        // First iteration, B_i never empty. 
        // In later iterations, is_empty_Bi will update is_empty. 
        while (!is_empty[0]) {
            
            std :: cout << "i: " << i << std :: endl;
            computeLightReq<<<node_blks, NUM_THREADS>>>(d_B, d_light, d_row_ptrs, d_edge_weights, num_nodes, i, d_Req_size, d_dists, d_Req);

            cudaDeviceSynchronize();
            std::cout<<"early d Req size: " << d_Req_size[0] << std :: endl;

            update_S_clear_B_i<<<node_blks, NUM_THREADS>>>(
                d_B, d_S, i, num_nodes
            );

            std :: cout << "Before clear" << std :: endl;
            printB<<<node_blks, NUM_THREADS>>>(d_B, num_nodes);

            cudaDeviceSynchronize();

            std :: cout << "After clear: " << std :: endl;
            printB<<<node_blks, NUM_THREADS>>>(d_B, num_nodes);

            cudaDeviceSynchronize();


            if (d_Req_size[0] > 0) {
                thrust::sort(thrust::device, d_Req, d_Req + d_Req_size[0]);
            }

            cudaDeviceSynchronize();
            std :: cout << "dReq size: " << d_Req_size[0] << std :: endl;

            relax<<<edge_blks, NUM_THREADS>>>(d_B, d_Req, d_dists, d_Req_size[0], Delta);
            std :: cout << "After relax" << std :: endl;
            printB<<<node_blks, NUM_THREADS>>>(d_B, num_nodes);

            cudaDeviceSynchronize();
            is_empty[0] = true;
            is_empty_B_i<<<node_blks, NUM_THREADS>>>(d_B, num_nodes, is_empty, i);
            
            // Note this line is required for correctness, as otherwise the updates to is_empty 
            // will not propagate to host code. 
            cudaDeviceSynchronize();
            std :: cout << "B_i done: " << std :: endl;
            if (is_empty[0]) {
                std :: cout << "True" << std :: endl;
            }
            else {
                std:: cout << "False" << std :: endl;
            }
        }
        
        // Just set d_Req_size = 0 to clear Req?
        cudaMemset((void**) &d_Req_size, 0, sizeof(int));

        computeHeavyReq<<<node_blks, NUM_THREADS>>>(d_S, d_heavy, d_row_ptrs, d_edge_weights, num_nodes, i, d_Req_size, d_dists, d_Req);

        relax<<<edge_blks, NUM_THREADS>>>(d_B, d_Req, d_dists, *d_Req_size, Delta);
        cudaDeviceSynchronize();

        i++;
        is_empty[0] = true; 
        is_empty_B<<<node_blks, NUM_THREADS>>>(d_B, num_nodes, is_empty); 
    }


    cudaMemcpy(d, d_dists, graphCSR.numNodes * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(p, d_preds, graphCSR.numNodes * sizeof(int), cudaMemcpyDeviceToHost);

    std :: cout << "d[0]: " << d[0] << std :: endl;
    std :: cout << "d[1]" << d[1] << std :: endl;
    std :: cout << "d[2]" << d[2] << std :: endl;

}