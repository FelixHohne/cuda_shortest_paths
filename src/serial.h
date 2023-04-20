//
// Created by Felix Hohne on 4/3/23.
//

#ifndef GRAPHCUDA_SERIAL_H
#define GRAPHCUDA_SERIAL_H

#include <vector>
#include <unordered_map>
#include <list>


std::pair<int*, int*> st_dijkstra(std::unordered_map<int, std::list<int>> adjList, int source, int num_nodes);
#endif //GRAPHCUDA_SERIAL_H