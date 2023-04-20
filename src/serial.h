//
// Created by Felix Hohne on 4/3/23.
//

#ifndef GRAPHCUDA_SERIAL_H
#define GRAPHCUDA_SERIAL_H

#include <vector>
#include <unordered_map>
#include <list>

// Single threaded Dijkstra. 
void st_dijkstra(std::unordered_map<int, std::list<int>> adjList, int source, int num_nodes, int* d, int* p);

// Effects: populates dists with shortest distance to corresponding node
// Currently does nothing with preds, maybe we can in the future 
// Uses a delta value of delta
// Requires: length of dists is num_nodes
void delta_stepping(std::unordered_map<int, std::list<int>> adj_list, int source, int num_nodes, int* dists, int* preds, int delta);

#endif //GRAPHCUDA_SERIAL_H