//
// Created by Felix Hohne on 4/3/23.
//

#ifndef GRAPHALGORITHMSWITHCUDA_GRAPHALGORITHMS_H
#define GRAPHALGORITHMSWITHCUDA_GRAPHALGORITHMS_H

#endif //GRAPHALGORITHMSWITHCUDA_GRAPHALGORITHMS_H
#include <vector>
#include <unordered_map>
#include <list>


std::pair<int*, int*> st_dijkstra(std::unordered_map<int, std::list<int>> adjList, int source, int num_nodes);
