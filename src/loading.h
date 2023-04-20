//
// Created by Felix Hohne on 4/3/23.
//

#ifndef GRAPHCUDA_LOADING_H
#define GRAPHCUDA_LOADING_H


#include <list>
#include <string>

void print_edge_list(std::list<std::pair<int, int>> parsed_edge_list);
std::pair<std::list<std::pair<int, int>>, int> read_edge_list(std::string filename);

#endif //GRAPHCUDA_LOADING_H

