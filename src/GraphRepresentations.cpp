//
// Created by Felix Hohne on 4/3/23.
//

#include "GraphRepresentations.h"
#include <vector>
#include <list>
#include <iostream>
#include <unordered_map>
#include <vector>


void printAdjList(std::unordered_map<int, std::list<int>>  adjList) {
    for (const auto &token: adjList) {
        std::cout << "Adjacency list of " << token.first << std:: endl;
        for (const auto &elem: token.second) {
            std::cout << elem << ", ";
        }
        std :: cout << std :: endl;
    }

}

std::unordered_map<int, std::list<int>> constructAdjList(std::list<std::pair<int, int>> edge_list) {
    std::unordered_map<int, std::list<int>> adjList;
    for (const auto &token: edge_list) {
        adjList[token.first].push_back(token.second);
        adjList[token.second].push_back(token.first);
    }
    return adjList;
}

