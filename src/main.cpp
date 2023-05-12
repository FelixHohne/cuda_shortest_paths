#include "gpu.cuh"
#include "graph.h"
#include "loading.h"
#include "serial.h"
#include <bits/stdc++.h>
#include <chrono>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <cuda_runtime_api.h>
#include <cuda.h>

// commented out since G2 doesn't like it
// #include <cuda.h>


// Returns: index of option in argv, -1 if not found
int find_arg_idx(int argc, char* argv[], const char* option) {
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], option) == 0) {
            return i;
        }
    }
    return -1;
}

// Returns: string following option in argv, default_value if not found
std::string find_string_arg(int argc, char* argv[], const char* option, std::string default_value) {
    int i = find_arg_idx(argc, argv, option);
    if (i > 0 && i < argc - 1) {
        return argv[i + 1];
    }
    return default_value;
}

// Returns: int following option in argv, default_value if not found
int find_int_arg(int argc, char* argv[], const char* option, int default_value) {
    int i = find_arg_idx(argc, argv, option);
    if (i > 0 && i < argc - 1) {
        return std::stoi(argv[i + 1]);
    }
    return default_value;
}

bool DEBUG_PRINT = false;
std::vector<std::string> algos {
    "serial-dijkstra",
    "serial-delta-stepping",
    "gpu-bellman-ford", 
    "gpu-delta-stepping"
};

int main(int argc, char* argv[]) {
    // command-line parsing
    if (find_arg_idx(argc, argv, "-h") > 0) {
        std::cout << "Options:" << std::endl;
        std::cout << "-h: see this help" << std::endl;
        std::cout << "-f <filename>: file to load graph from" << std::endl;
        std::cout << "-a <algo>: set the algorithm to run" << std::endl;
        std::cout << "\tSupported algorithms: ";
        for (int i = 0; i < algos.size(); i++) {
            std::cout << algos[i];
            if (i + 1 < algos.size()) {
                std::cout << ", ";
            } else {
                std::cout << std::endl;
            }
        }
        std::cout << "-s <int>: set source node, defaults to 0 if not specified" << std::endl;
        std::cout << "-o <filename>: set the output file name" << std::endl;
        return 0;
    }

    std::string input_file = find_string_arg(argc, argv, "-f", "");
    if (input_file.empty()) {
        std::cout << "Please specify a graph input file with -f" << std::endl;
        return 0;
    }

    std::string algo = find_string_arg(argc, argv, "-a", "");
    if (algo.empty()) {
        std::cout << "Please specify an algorithm with -a" << std::endl;
        return 0;
    }

    int source_node = find_int_arg(argc, argv, "-s", 0);
    if (source_node < 0) {
        std::cout << "Nodes start at 0" << std::endl;
        return 1;
    }

    std::string output_file = find_string_arg(argc, argv, "-o", "");

    std::cout<< "Parsing edge list" << std::endl;
    std::cout << "Input file is: " << input_file << std :: endl;

    // whether to use edge weights- defaults to a weight of 1 per edge
    bool USE_EDGE_WEIGHTS = false;
    // construct adjacency list
    auto [parsed_edge_list, max_node] = read_edge_list(input_file, USE_EDGE_WEIGHTS);

    // account for 0-indexing when calculating number of nodes
    int num_nodes = max_node + 1;

    if (DEBUG_PRINT) {
        std::cout << "Number of nodes in the graph: " << num_nodes << std::endl;
        std::cout << "Number of edges in the graph: " << parsed_edge_list.size() << std :: endl;
    }

    std::cout<<"Finished parsing edge list" << std::endl;
    if (DEBUG_PRINT) {
        print_edge_list(parsed_edge_list);
    }
    std::unordered_map<int, std::list<std::pair<int, int>>> adjList = construct_adj_list(parsed_edge_list);

    std::cout<<"Constructed adjacency list" << std::endl;

    if (DEBUG_PRINT) {
        std::cout << "Number of edges of node 9721 " << adjList[9721].size() << std::endl;

        for (auto v: adjList[9721]) {
            std::cout << v.first << std :: endl; 
        }
    }
    

    CSR graphCSR;
    if (ASYNC_MEMORY && (algo == "gpu-bellman-ford" || algo == "gpu-delta-stepping")) {
        graphCSR = construct_sparse_CSR(adjList, num_nodes, true);
    } else {
        graphCSR = construct_sparse_CSR(adjList, num_nodes, false);
    }
    std::cout<< "Constructed sparse CSR representation" << std::endl;
    if (DEBUG_PRINT) {
        print_adj_list(adjList);
    }

    // start algorithm
    auto start_algo = std::chrono::steady_clock::now();
    
    int* min_distances = new int[num_nodes];
    int* p = new int[num_nodes];

    // Doesn't seem to be worth it. 
    if (false) {
        cudaMallocHost((void**) &min_distances, (num_nodes) * sizeof(int));
        cudaMallocHost((void**) &p, (num_nodes) * sizeof(int));
    } else {
        min_distances = new int[num_nodes];
        p = new int[num_nodes];
    }

    int delta = find_int_arg(argc, argv, "-D", 50);

    std::cout << "Running algorithm: " << algo << std::endl;
    std :: cout << "Delta: " << delta << std :: endl;
    if (algo == "serial-dijkstra") {
        st_dijkstra(adjList, source_node, num_nodes, min_distances, p);
    } else if (algo == "serial-delta-stepping") {
        delta_stepping(graphCSR, source_node, num_nodes, min_distances, p, delta);
    } else if (algo == "gpu-bellman-ford") {
        initializeBellmanFord(graphCSR, source_node, num_nodes, min_distances, p);
    } else if (algo == "gpu-delta-stepping") {
        initializeDeltaStepping(graphCSR, source_node, num_nodes, min_distances, p, delta);
    } else {
        std::cout<<"Invalid algorithm provided. " << std::endl;
        std::cout << "-a <algo>: set the algorithm to run" << std::endl;
    }

    auto end_algo = std::chrono::steady_clock::now();
    std::chrono::duration<double> get_algo_time = end_algo - start_algo;
    double algo_time = get_algo_time.count();

    std::cout << "Time to run Algorithm: " << algo_time << std::endl;
    
    std::cout << "Min distance vector source is : " << min_distances[source_node] << std::endl;
    if (DEBUG_PRINT) {
        std :: cout << "Reporting minimum distances: " << std::endl;
        int counter = 0;
        for (int i = 0; i < graphCSR.numNodes; i++) {
            int distance = min_distances[i];
            if (distance != INT_MAX) {
                std::cout << "min distance for " << counter << ": " << distance << std :: endl;
            }
            counter++;
        }
    }

    // save output if specified
    if (!output_file.empty()) {
        int counter = 0;
        std::ofstream fsave(output_file);

        fsave << "source\tdistance" << std::endl;
        for (int i = 0; i < graphCSR.numNodes; i++) {
            int distance = min_distances[i];
            if (distance != INT_MAX) { 
                fsave << counter << "\t" << distance << std::endl;
            }
            counter++;
        }
        fsave.close();
    }

    // free memory
    // delete[] min_distances;
    // delete[] p; 
    return 0;
}
