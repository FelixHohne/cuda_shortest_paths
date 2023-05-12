//
// Created by Felix Hohne on 4/3/23.
//

#ifndef GRAPHCUDA_GRAPH_H
#define GRAPHCUDA_GRAPH_H
#include <unordered_map>
#include <list>
#include <utility> // std::pair


typedef struct CSR {
    int numNodes;
    int numEdges;
    int* rowPointers;
    int* neighborNodes;
    int* edgeWeights;
} CSR;

// adjacency list = u -> list of neighbors represented as (v, edge_weight)
std::unordered_map<int, std::list<std::pair<int, int>>> construct_adj_list(std::list<std::pair<std::pair<int, int>, int>> edge_list);
void print_adj_list(std::unordered_map<int, std::list<std::pair<int, int>>>  adjList);
CSR construct_sparse_CSR(std::unordered_map<int, std::list<std::pair<int, int>>> adjList, int num_nodes, bool is_cuda);

#endif //GRAPHCUDA_GRAPH_H
