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


CSR constructSparseCSR(std::unordered_map<int, std::list<int>> adjList, int numNodes) {

    int* rowPointers = new int[numNodes + 1];
    int numEdges = 0;

    for (auto const p: adjList) {
        numEdges = numEdges + p.second.size();
    }

    int* neighborNodes = new int[numEdges + 1];
    int* edgeWeights = new int[numEdges + 1];

    // TODO: Handle edge weights
    std::fill_n(edgeWeights, numEdges + 1, 1);

    int numEdgesAdded = 0;

    for (int i = 0; i < numNodes + 1; i++) {

        std::list<int> neighbors;
        if (adjList.contains(i)) {
            neighbors = adjList[i];
        }

        rowPointers[i] = numEdgesAdded;


        for (auto const j: neighbors) {
            neighborNodes[numEdgesAdded] = j;
            numEdgesAdded++;
        }

        // TODO: Handle edge weights here.

    }

    CSR graphCSR = {numNodes, numEdges, rowPointers, neighborNodes, edgeWeights};
    return graphCSR;
}



