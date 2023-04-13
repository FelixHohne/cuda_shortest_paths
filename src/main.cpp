#include <iostream>
#include "GraphLoading.h"
#include "GraphRepresentations.h"
#include "GraphAlgorithms.h"
#include "GraphAlgorithmsGPU.cuh"
#include <vector>
#include <fstream>
#include <iostream>
#include <chrono>
#include <bits/stdc++.h>
#include <string>
#include <cuda.h>

int main(int argc, char *argv[]) {
    bool DEBUG_PRINT = false;
    bool OUTPUT_TO_FILE = true; 

    HelloWorld(); 

    if (argc <= 3) {
        std::cout << "Requires arguments for matrix location, method name, source node" << std::endl;
        std::cout << "Valid method names: \"Dijkstra\", \"Bellman-Ford\"" << std::endl;
        return 1;
    }
    std::string file_location = argv[1];
    std::string graph_algo = argv[2];
    // int source_node = std::stoi(argv[3]);
    int source_node = 0;

    if (source_node < 0) {
        std::cout << "Nodes start at 0" << std::endl;
        return 1;
    }

    std::list<std::pair<int, int>> parsed_edge_list = read_edge_list(file_location);

    if (DEBUG_PRINT)
        print_edge_list(parsed_edge_list);
    std::unordered_map<int, std::list<int>> adjList = constructAdjList(parsed_edge_list);
    CSR graphCSR = constructSparseCSR(adjList, adjList.size());
    // TODO: AdjList.size() is not safe
    
    if (DEBUG_PRINT)
        printAdjList(adjList);

    auto start_algo = std::chrono::steady_clock::now();

    std::pair<int*, int*> results;
    if (graph_algo == "Dijkstra") {
        results = st_dijkstra(adjList, source_node);
    } else if (graph_algo == "Bellman-Ford") {
        results = initializeBellmanFord(graphCSR, source_node);
    } else {
        std::cout << "Valid method names: \"Dijkstra\", \"Bellman-Ford\"" << std::endl;
        return 1;
    }

    // auto end_algo = std::chrono::steady_clock::now();
    // std::chrono::duration<double> get_algo_time = end_algo - start_algo;
    // double algo_time = get_algo_time.count();

    // std::cout << "Time to run Algorithm: " << algo_time << std::endl;

    // auto [min_distances, p] = results;

    // if (DEBUG_PRINT) {
    //     std :: cout << "Reporting minimum distances: " << std::endl;
    //     int counter = 0;
    //     for (auto const distance: min_distances) {
    //         if (distance != INT_MAX) {
    //             std::cout << "min distance for " << counter << ": " << distance << std :: endl;
    //         }
    //         counter++;
    //     }
    // }


    // if (OUTPUT_TO_FILE) {
    //     int counter = 0; 
    //     std::string output_file_name = "../serial_dijkstra.txt"; 
    //     std::ofstream fsave(output_file_name); 

    //     fsave << "source\tdistance" << std :: endl; 
    //     for (auto const distance: min_distances) {
    //         if (distance != INT_MAX) { 
    //             fsave << counter << "\t" << distance << std :: endl;
    //         }
    //         counter++;
    //     }
    //     fsave.close();
    // }
    return 0;
}