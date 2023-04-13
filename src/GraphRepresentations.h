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
};


std::unordered_map<int, std::list<int>> constructAdjList(std::list<std::pair<int, int>> edge_list);
void printAdjList(std::unordered_map<int, std::list<int>>  adjList);
CSR constructSparseCSR(std::unordered_map<int, std::list<int>> adjList, int numNodes);

#endif //GRAPHALGORITHMSWITHCUDA_GRAPHREPRESENTATIONS_H
