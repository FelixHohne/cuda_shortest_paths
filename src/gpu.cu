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
    
    if (tid >= num_nodes) {
        return;
    }

    for (int i = d_row_ptrs[tid]; i < d_row_ptrs[tid + 1]; i++) {
        int v = d_neighbor_nodes[i];
        // if distance[tid] + w < distance[v]
        if (d_dists[tid] != INT_MAX && d_dists[tid] + d_edge_weights[i] < d_dists[v]) {
            
            atomicExch(d_dists + v,  d_dists[tid] + d_edge_weights[i]);
            atomicExch(d_preds + v, tid);
        }
        
    }
}

/*
Node wise parallel function. 
Computes d_node_of_edge mapping, such that if int v = d_neighbor_nodes[tid], 
then d_node_of_edge[v] = u where (u, v) \in E. 
*/

__global__ void computeNodeOfEdge(int num_edges, int num_nodes, int* d_row_ptrs, int* d_neighbor_nodes, int* d_node_of_edge) 
{
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    if (tid >= num_nodes) {
        return;
    }

    /*
    node u has edges in d_neighbor_nodes in indices [node_edge_start)
    */
    int node_edge_start = d_row_ptrs[tid]; 
    int node_edge_end = d_row_ptrs[tid + 1]; 
    
    for (int k = node_edge_start; k < node_edge_end; k++) {
        d_node_of_edge[k] = tid; 
    }
}

/*
Edge Wise Parallelized. 
*/
__global__ void EdgeWiseBellmanFord(int num_nodes, int num_edges, int* d_dists, int* d_preds, int* d_node_of_edge, int* d_neighbor_nodes, int* d_edge_weights) {

    /*
    Invariants: We consider relaxation of edge (u, v). 
    v = d_neighbor_nodes[tid]. 
    u = node_of_edge[v]. 
    As u & v are nodes, 0 <= u <= max_node and 0 <= v <= max_node. 
    */
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    
    if (tid >= num_edges) {
        return;
    }

    if (EDGE_WISE_SHARED_MEMORY) {

    __shared__ int tmp_neighbor_nodes[NUM_THREADS]; 
    __shared__ int tmp_node_of_edge[NUM_THREADS]; 

    tmp_neighbor_nodes[threadIdx.x] = d_neighbor_nodes[tid]; 
    tmp_node_of_edge[threadIdx.x] = d_node_of_edge[tid]; 
    __syncthreads();

    int v = tmp_neighbor_nodes[threadIdx.x];
    int u = tmp_node_of_edge[threadIdx.x];

    // TODO: Check if d_edge_weights[u] or d_edge_weights[v]. 
    if (d_dists[u] != INT_MAX && d_dists[u] + d_edge_weights[u] < d_dists[v]) {
        atomicExch(d_dists + v,  d_dists[u] + d_edge_weights[u]);
        // atomicExch(d_preds + v, u);
    }

    } 
    else {
        int v = d_neighbor_nodes[tid]; 
        int u = d_node_of_edge[tid]; 

        // TODO: Check if d_edge_weights[u] or d_edge_weights[v]. 
        if (d_dists[u] != INT_MAX && d_dists[u] + d_edge_weights[u] < d_dists[v]) {
            atomicExch(d_dists + v,  d_dists[u] + d_edge_weights[u]);
            // atomicExch(d_preds + v, u);
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
void initializeBellmanFord(CSR graphCSR, int source, int max_node, int* d, int* p) {

    struct timeval start, stop;
    int num_nodes = max_node + 1; 

    int* d_dists;
    int* d_preds;
    int* d_row_ptrs;
    int* d_neighbor_nodes;
    int* d_edge_weights;
    int blks = (graphCSR.numNodes + NUM_THREADS - 1) / NUM_THREADS;
    int edge_blks = (graphCSR.numEdges + NUM_THREADS - 1) / NUM_THREADS;
    int* d_node_of_edge; 

    cudaMalloc((void**) &d_dists, graphCSR.numNodes * sizeof(int));
    cudaMalloc((void**) &d_preds, graphCSR.numNodes * sizeof(int));
    cudaMalloc((void**) &d_row_ptrs, (graphCSR.numNodes + 1) * sizeof(int));
    cudaMalloc((void**) &d_neighbor_nodes, graphCSR.numEdges * sizeof(int));
    cudaMalloc((void**) &d_edge_weights, graphCSR.numEdges * sizeof(int));
    cudaMalloc((void**) &d_node_of_edge, graphCSR.numEdges * sizeof(int));

    gettimeofday(&start, 0);
    bellman_initialize_dists_array<<<blks, NUM_THREADS>>>(d_dists, num_nodes, source);

    cudaMemcpy(d_row_ptrs, graphCSR.rowPointers, (graphCSR.numNodes + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_neighbor_nodes, graphCSR.neighborNodes, graphCSR.numEdges * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_edge_weights, graphCSR.edgeWeights, graphCSR.numEdges * sizeof(int), cudaMemcpyHostToDevice);

    if (EDGE_WISE_BELLMAN_FORD) {
        computeNodeOfEdge<<<blks, NUM_THREADS>>>(graphCSR.numEdges, num_nodes, d_row_ptrs, d_neighbor_nodes, d_node_of_edge); 
    }

    cudaDeviceSynchronize();
    gettimeofday(&stop, 0);
    double init_time = get_time(start, stop);
    printf("Bellman init time:  %3.1f ms \n", init_time);

    // std :: cout << "Node of Edge" << std :: endl;
    // for (int i = 0; i < graphCSR.numEdges; i++) { 
    //     std :: cout << "i: " << d_node_of_edge[i] << std :: endl;
    // }

    float total_BF_time = 0;
    for (int i = 0; i < graphCSR.numNodes - 1; i++) {
        gettimeofday(&start, 0);
        if (i % 10000 == 0) {
            std :: cout << "Iteration i: " << i << std :: endl;
        }

        if (EDGE_WISE_BELLMAN_FORD) {
            EdgeWiseBellmanFord<<<edge_blks, NUM_THREADS>>>(num_nodes, graphCSR.numEdges, d_dists, d_preds, d_node_of_edge, d_neighbor_nodes, d_edge_weights);
        }
        else {
            BellmanFord<<<blks, NUM_THREADS>>>(num_nodes, graphCSR.numEdges, d_dists, d_preds, d_row_ptrs, d_neighbor_nodes, d_edge_weights);
        }

        cudaDeviceSynchronize();
        if (ELEMENT_WISE_TIMING) {
            gettimeofday(&stop, 0);
            total_BF_time += get_time(start, stop);
        }
    }

    if (ELEMENT_WISE_TIMING) {
        printf("Bellman kernels total time:  %3.1f ms \n", total_BF_time);
        printf("Bellman kernels avg time per node:  %3.1f ms \n", total_BF_time / (graphCSR.numNodes-1));
    }
   
    cudaMemcpy(d, d_dists, graphCSR.numNodes * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(p, d_preds, graphCSR.numNodes * sizeof(int), cudaMemcpyDeviceToHost);
}


// ___________________________________________________________________


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


/*
Edge level parallelization
Require: d_Req is sorted using < on ReqElements. 
*/
__global__ void relax(int* d_B, ReqElement* d_Req, int* d_dists, int d_Req_size, int num_edges, int delta) {
    int tid = threadIdx.x + blockDim.x * blockIdx.x;

    // Invariant should always be maintained. 
    if (d_Req_size > num_edges){ 
        assert(0);
    }

    if (tid >= d_Req_size) {
        return;
    }

    /*
    Since d_Req is sorted, for each nodeId, the best distance should be first. 
    Only need to look at the first entry and the entries where the previous element 
    corresponds to an earlier node. 
    */
    if (tid == 0 || d_Req[tid].nodeId > d_Req[tid - 1].nodeId) {
        int v = d_Req[tid].nodeId;
        int new_dist = d_Req[tid].dist;
        if (new_dist < d_dists[v]) {
            
            /*
            By if statement construction, only a single tid 
            will update a given node, since only one tid
            can be the first element for a given node. 
            Thus, there is only one thread writing, 
            and no direct concurrency is needed. 
            */
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
        printf("Constructing heavy edges\n");
        d_heavy[tid] = d_neighbor_nodes[tid];
        d_light[tid] = -1;
    } else {
        d_light[tid] = d_neighbor_nodes[tid];
        d_heavy[tid] = -1;
    }

}

__global__ void initialize_d_Req(ReqElement* d_Req, int num_edges) {
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    if (tid >= num_edges) {
        return;
    }

    d_Req[tid].nodeId = -1; 
    d_Req[tid].dist = -1;
}

/*
Node-based parallelization.
Computes Req = {(w, dist[v] + dist(v, w)) : v \in B[i] && (v, w) \in light(v)}
*/
__global__ void computeLightReq(int* d_B, int* d_light, int* d_row_ptrs, int* d_edge_weights, int num_nodes, int i, int* d_Req_size, int* d_dists, ReqElement* d_Req) {
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    if (tid >= num_nodes) {
        return;
    }
   
    if (d_B[tid] != i) {
        return;
    }

    int v = tid;
    for (int j = d_row_ptrs[v]; j < d_row_ptrs[v + 1]; j++) {
        if (d_light[j] != -1) {
            int old_index = atomicAdd(d_Req_size, 1);
            d_Req[old_index].nodeId = d_light[j];
            d_Req[old_index].dist = d_dists[v] != INT_MAX ? d_dists[v] + d_edge_weights[j] : INT_MAX;
            // printf("old_index: %d, node_id: %d, num_nodes: %d, dist: %d\n", old_index, d_light[i], num_nodes, d_dists[v] + d_edge_weights[i]);
        }
    }



}

__global__ void printB(int* d_B, int num_nodes) {

    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    if (tid >= num_nodes) {
        return;
    }

    printf("d_B element %d: %d      ", tid, d_B[tid]);
    printf("\n");
}

__global__ void print_d_Req(int d_Req_size, ReqElement* d_Req, int num_edges) {

    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    if (tid > d_Req_size) {
        return;
    }

    printf("d_Req element (%d, %d, %d)", tid, d_Req[tid].nodeId, d_Req[tid].dist);
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
            int old_index = atomicAdd(d_Req_size, 1);
            d_Req[old_index].nodeId = d_heavy[i];
            d_Req[old_index].dist = d_dists[v] + d_edge_weights[i];
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
// Node level parallelization
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
        is_empty[0] = false;
    }

}

void initializeDeltaStepping(CSR graphCSR, int source, int max_node, int* d, int* p, int Delta) {

    int num_nodes = max_node + 1; 
    std::cout << "Begin CUDA Delta Stepping with Delta: " << Delta << std::endl;
    int node_blks = (graphCSR.numNodes + NUM_THREADS - 1) / NUM_THREADS;
    int edge_blks = (graphCSR.numEdges + NUM_THREADS - 1) / NUM_THREADS;

    int* d_row_ptrs;
    int* d_neighbor_nodes;
    int* d_edge_weights;

    int* d_S;

    // Invariant: non-dense pack. 
    // Invariant: -1 means element is empty. 
    int* d_B; 

    int* d_dists;
    int* d_preds;

    // Device Req array
    // Elements up to d_Req_size are correct, beyond is garbage
    ReqElement* d_Req; 

    /*
    d_light and d_heavy are arrays of the same length as graphCSR.numEdges
    They contain a valid node id if the neighbor node corresponds to
    a light or heavy edge of the vertex, otherwise -1.
    Essentially a masked version of CSR for only light or only 
    heavy edges. 
    */
    int* d_light;
    int* d_heavy;

    int i = 0; 

    int* d_Req_size; 
    cudaMallocManaged(&d_Req_size, sizeof(int)); 
    d_Req_size[0] = 0; 

    bool* is_empty; 
    cudaMallocManaged(&is_empty, sizeof(bool)); 
    is_empty[0] = false; 


    if (ASYNC_MEMORY) {

        std :: cout << "Using Async Memory" << std :: endl;
        cudaStream_t stream1, stream2, stream3, stream4, stream5, stream6, stream7, stream8, stream9, stream10; 
        cudaStreamCreate(&stream1);
        cudaStreamCreate(&stream2);
        cudaStreamCreate(&stream3);
        cudaStreamCreate(&stream4);
        cudaStreamCreate(&stream5);
        cudaStreamCreate(&stream6);
        cudaStreamCreate(&stream7);
        cudaStreamCreate(&stream8);
        cudaStreamCreate(&stream9);
        cudaStreamCreate(&stream10);

        cudaEvent_t event1, event2, event3, event4, event8;

        cudaEventCreate(&event1);
        cudaEventCreate(&event2);
        cudaEventCreate(&event3); 
        cudaEventCreate(&event4);
        cudaEventCreate(&event8);


        cudaMallocAsync((void**) &d_row_ptrs, (graphCSR.numNodes + 1) * sizeof(int), stream1);

        cudaMallocAsync((void**) &d_neighbor_nodes, (graphCSR.numEdges) * sizeof(int), stream2);

        cudaMallocAsync((void**) &d_edge_weights, (graphCSR.numEdges) * sizeof(int), stream3);

        cudaMallocAsync((void**) &d_S, graphCSR.numNodes * sizeof(int), stream4);

        cudaEventRecord(event4, stream4); 

        cudaMallocAsync((void**) &d_dists, graphCSR.numNodes * sizeof(int), stream5);

        cudaMallocAsync((void**) &d_preds, graphCSR.numNodes * sizeof(int), stream6);

        cudaMallocAsync((void**) &d_Req, graphCSR.numEdges * sizeof(ReqElement), stream7); 

        cudaMallocAsync((void**) &d_B, graphCSR.numNodes * sizeof(int), stream8);

        cudaEventRecord(event8, stream8); 

        cudaMallocAsync((void**) &d_light, graphCSR.numEdges * sizeof(int), stream9);
        
        cudaMallocAsync((void**) &d_heavy, graphCSR.numEdges * sizeof(int), stream10);

        cudaMemcpyAsync(d_row_ptrs, graphCSR.rowPointers, (graphCSR.numNodes + 1) * sizeof(int), cudaMemcpyHostToDevice, stream1);

        cudaEventRecord(event1, stream1); 

        cudaMemcpyAsync(d_neighbor_nodes, graphCSR.neighborNodes, graphCSR.numEdges * sizeof(int), cudaMemcpyHostToDevice, stream2);

        cudaEventRecord(event2, stream2); 

        cudaMemcpyAsync(d_edge_weights, graphCSR.edgeWeights, graphCSR.numEdges * sizeof(int), cudaMemcpyHostToDevice, stream3);    

        cudaEventRecord(event3, stream3); 

        cudaEventSynchronize(event1); 
        cudaEventSynchronize(event2); 
        cudaEventSynchronize(event3); 

        // Does not use dynamic memory
        initialize_light_heavy_arrays<<<edge_blks, NUM_THREADS, 0, stream3>>>(d_light, d_heavy, d_neighbor_nodes, d_edge_weights, graphCSR.numEdges, Delta);

        cudaEventSynchronize(event4); 
        cudaEventSynchronize(event8); 

        /*
        Initializes d_dists to INT_MAX 
        Clears B[i]
        Sets source distance to 0 
        Adds source to B[0]. 
        Does not use dynamic memory.
        */
        delta_stepping_initialize<<<node_blks, NUM_THREADS, 0, stream8>>>(d_dists, d_B, graphCSR.numNodes, source);

        cudaDeviceSynchronize(); 
    }

    else {

        std :: cout << "Using cudaMalloc" << std :: endl;
        cudaMalloc((void**) &d_row_ptrs, (graphCSR.numNodes + 1) * sizeof(int));
        cudaMemcpy(d_row_ptrs, graphCSR.rowPointers, (graphCSR.numNodes + 1) * sizeof(int), cudaMemcpyHostToDevice);

        cudaMalloc((void**) &d_neighbor_nodes, (graphCSR.numEdges) * sizeof(int));
        cudaMemcpy(d_neighbor_nodes, graphCSR.neighborNodes, graphCSR.numEdges * sizeof(int), cudaMemcpyHostToDevice);

        cudaMalloc((void**) &d_edge_weights, (graphCSR.numEdges) * sizeof(int));
        cudaMemcpy(d_edge_weights, graphCSR.edgeWeights, graphCSR.numEdges * sizeof(int), cudaMemcpyHostToDevice);    

        cudaMalloc((void**) &d_S, graphCSR.numNodes * sizeof(int));

        cudaMalloc((void**) &d_B, graphCSR.numNodes * sizeof(int));

        cudaMalloc((void**) &d_dists, graphCSR.numNodes * sizeof(int));
        cudaMalloc((void**) &d_preds, graphCSR.numNodes * sizeof(int));

        cudaMalloc((void**) &d_light, graphCSR.numEdges * sizeof(int));
        cudaMalloc((void**) &d_heavy, graphCSR.numEdges * sizeof(int));
        initialize_light_heavy_arrays<<<edge_blks, NUM_THREADS>>>(d_light, d_heavy, d_neighbor_nodes, d_edge_weights, graphCSR.numEdges, Delta);

        cudaMalloc((void**) &d_Req, graphCSR.numEdges * sizeof(ReqElement)); 

        /*
        Initializes d_dists to INT_MAX 
        Clears B[i]
        Sets source distance to 0 
        Adds source to B[0]. 
        */
        delta_stepping_initialize<<<node_blks, NUM_THREADS>>>(d_dists, d_B, graphCSR.numNodes, source);
    }
    

    while (!is_empty[0]) {
        clear_S<<<node_blks, NUM_THREADS>>>(d_S, num_nodes);
        /* Checks if B_i is not empty. 
        First iteration, B_i never empty. 
        In later iterations, is_empty_Bi will update is_empty. 
        */
        while (!is_empty[0]) {
            
            d_Req_size[0] = 0; 
            computeLightReq<<<node_blks, NUM_THREADS>>>(d_B, d_light, d_row_ptrs, d_edge_weights, num_nodes, i, d_Req_size, d_dists, d_Req);

            // This cudaDeviceSynchronize is required; causes correctness issues to remove it. 
            cudaDeviceSynchronize();

            update_S_clear_B_i<<<node_blks, NUM_THREADS>>>(
                d_B, d_S, i, num_nodes
            );

            thrust::sort(thrust::device, d_Req, d_Req + d_Req_size[0]);

            cudaDeviceSynchronize(); 

            // print_d_Req<<<edge_blks, NUM_THREADS>>>(d_Req_size[0], d_Req, graphCSR.numEdges);
            
            relax<<<edge_blks, NUM_THREADS>>>(d_B, d_Req, d_dists, d_Req_size[0], graphCSR.numEdges, Delta);

            is_empty[0] = true;
            is_empty_B_i<<<node_blks, NUM_THREADS>>>(d_B, num_nodes, is_empty, i);
            
            // Note this line is required for correctness, as otherwise the updates to is_empty 
            // will not propagate to host code. 
            cudaDeviceSynchronize();
        }
        
        cudaDeviceSynchronize(); 
        // Clear Req. 
        d_Req_size[0] = 0; 

        cudaDeviceSynchronize();
        computeHeavyReq<<<node_blks, NUM_THREADS>>>(d_S, d_heavy, d_row_ptrs, d_edge_weights, num_nodes, i, d_Req_size, d_dists, d_Req);

        relax<<<edge_blks, NUM_THREADS>>>(d_B, d_Req, d_dists, *d_Req_size, graphCSR.numEdges, Delta);
        
        i++;
        is_empty[0] = true; 
        is_empty_B<<<node_blks, NUM_THREADS>>>(d_B, num_nodes, is_empty); 

        // Note this line is required for correctness, as otherwise the updates to is_empty 
        // will not propagate to host code. 
        cudaDeviceSynchronize();
    }

    cudaMemcpy(d, d_dists, graphCSR.numNodes * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(p, d_preds, graphCSR.numNodes * sizeof(int), cudaMemcpyDeviceToHost);

}