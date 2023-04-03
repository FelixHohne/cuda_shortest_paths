//
// Created by Felix Hohne on 4/3/23.
//

#ifndef GRAPHALGORITHMSWITHCUDA_GRAPHALGORITHMS_H
#define GRAPHALGORITHMSWITHCUDA_GRAPHALGORITHMS_H

#endif //GRAPHALGORITHMSWITHCUDA_GRAPHALGORITHMS_H
#include <vector>
#include <unordered_map>
#include <list>


std::pair<std::vector<int>, std::vector<int>> st_dijkstra(std::unordered_map<int, std::list<int>> adjList, int source);
