//
// Created by Felix Hohne on 4/3/23.
//

#include "loading.h"
#include <fstream>
#include <sstream>
#include <utility> // std::pair
#include <stdexcept> // std::runtime_error
#include <list>
#include <iostream>
#include <string>

void print_edge_list(std::list<std::pair<std::pair<int, int>, int>> parsed_edge_list) {
    for (const auto &token: parsed_edge_list) {
        auto edge = token.first;
        std::cout << "(" << edge.first << ", " << edge.second << "), " << token.second << std::endl;
    }
}


std::pair<std::list<std::pair<std::pair<int, int>, int>>, int> read_edge_list(std::string filename, bool use_edge_weights) {
    // Create an input filestream
    std::ifstream myFile(filename);

    // Track the largest node we encounter 
    int max_value = 0; 

    // Make sure the file is open
    if (!myFile.is_open()) throw std::runtime_error("Could not open file");

    std::string line;

    if (myFile.good()) {
        std::getline(myFile, line); // Ignore column descriptions
    }

    std::list<std::pair<std::pair<int, int>, int>> parsed_edge_list;
    while (std::getline(myFile, line)) {
        // Create a stringstream of the current line
        std::stringstream ss(line);

        int source;
        int sink;
        int edge_weight;
        ss >> source;
        ss >> sink;
        if (use_edge_weights) {
            ss >> edge_weight;
        } else {
            edge_weight = 1;
        }

        max_value = std::max(max_value, source);
        max_value = std::max(max_value, sink);

        parsed_edge_list.push_back(std::make_pair(std::make_pair(sink, source), edge_weight));
    }

    myFile.close();
    return std::make_pair(parsed_edge_list, max_value);
}

