//
// Created by Felix Hohne on 4/3/23.
//

#ifndef GRAPHCUDA_LOADING_H
#define GRAPHCUDA_LOADING_H


#include <list>
#include <string>

// each edge is in the form ((u, v), weight)- note that all edges are undirected
void print_edge_list(std::list<std::pair<std::pair<int, int>, int>> parsed_edge_list);
std::pair<std::list<std::pair<std::pair<int, int>, int>>, int> read_edge_list(std::string filename, bool use_edge_weights);

#endif //GRAPHCUDA_LOADING_H

