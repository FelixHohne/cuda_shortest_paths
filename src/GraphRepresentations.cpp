//
// Created by Felix Hohne on 4/3/23.
//

#include "GraphRepresentations.h"
#include <vector>
#include <list>
#include <unordered_map>
#include <vector>


std::unordered_map<int, std::list<int>> constructAdjList(std::list<std::pair<int, int>> edge_list) {
    std::unordered_map<int, std::list<int>> adjList;
    for (const auto &token: edge_list) {
        adjList[token.first].push_back(token.second);
        adjList[token.second].push_back(token.first);
    }
    return adjList;
}

