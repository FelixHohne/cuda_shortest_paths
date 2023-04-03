//
// Created by Felix Hohne on 4/3/23.
//

#ifndef GRAPHALGORITHMSWITHCUDA_GRAPHLOADING_H
#define GRAPHALGORITHMSWITHCUDA_GRAPHLOADING_H


#include <list>

void print_edge_list(std::list<std::pair<int, int>> parsed_edge_list);
std::list<std::pair<int, int>> read_edge_list(std::string);

#endif //GRAPHALGORITHMSWITHCUDA_GRAPHLOADING_H

