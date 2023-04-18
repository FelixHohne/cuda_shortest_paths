//
// Created by Felix Hohne on 4/3/23.
//

#include "GraphRepresentations.h"
#include <vector>
#include <list>
#include <iostream>
#include <unordered_map>
#include <vector>


void print_adj_list(std::unordered_map<int, std::list<int>>  adjList) {
    for (const auto &token: adjList) {
        std::cout << "Adjacency list of " << token.first << std:: endl;
        for (const auto &elem: token.second) {
            std::cout << elem << ", ";
        }
        std :: cout << std :: endl;
    }

}

std::unordered_map<int, std::list<int>> construct_adj_list(std::list<std::pair<int, int>> edge_list) {
    std::unordered_map<int, std::list<int>> adjList;
    for (const auto &token: edge_list) {
        adjList[token.first].push_back(token.second);
        adjList[token.second].push_back(token.first);
    }
    return adjList;
}


CSR construct_sparse_CSR(std::unordered_map<int, std::list<int>> adj_list, int num_nodes) {
    int* row_pointers = new int[num_nodes + 1];
    int num_edges = 0;

    for (auto const p: adj_list) {
        num_edges = num_edges + p.second.size();
    }

    // ids of neighbor nodes in adj_list
    int* neighbor_nodes = new int[num_edges];
    int* edge_weights = new int[num_edges];

    // TODO: Handle edge weights
    std::fill_n(edge_weights, num_edges, 1);
    int num_edges_added = 0;

    for (int i = 0; i < num_nodes; i++) {
        // neighbors of node i
        std::list<int> neighbors;
        if (adj_list.contains(i)) {
            neighbors = adj_list[i];
        }

        for (auto const j: neighbors) {
            neighbor_nodes[num_edges_added] = j;
            num_edges_added++;
        }

        // TODO: Handle edge weights here.

        // hopefully update row pointers correctly
        // note row_pointers[0] should always be 0
        row_pointers[i + 1] = num_edges_added;
    }

    CSR graph_CSR = {
        .numNodes = num_nodes,
        .numEdges = num_edges,
        .rowPointers = row_pointers,
        .neighborNodes = neighbor_nodes,
        .edgeWeights = edge_weights
    };
    return graph_CSR;
}



