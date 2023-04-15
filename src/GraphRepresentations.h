//
// Created by Felix Hohne on 4/3/23.
//

#ifndef GRAPHALGORITHMSWITHCUDA_GRAPHREPRESENTATIONS_H
#define GRAPHALGORITHMSWITHCUDA_GRAPHREPRESENTATIONS_H
#include <unordered_map>
#include <list>


typedef struct CSR {
    int numNodes;
    int numEdges;
    int* rowPointers;
    int* neighborNodes;
    int* edgeWeights;
} CSR;


std::unordered_map<int, std::list<int>> construct_adj_list(std::list<std::pair<int, int>> edge_list);
void print_adj_list(std::unordered_map<int, std::list<int>>  adjList);
CSR construct_sparse_CSR(std::unordered_map<int, std::list<int>> adjList, int numNodes);

#endif //GRAPHALGORITHMSWITHCUDA_GRAPHREPRESENTATIONS_H
