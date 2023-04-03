#include <iostream>
#include "GraphLoading.h"
#include "GraphRepresentations.h"
#include <vector>

int main(int argc, char *argv[]) {
    if (argc <= 1) {
        std::cout << "Requires argument for matrix location" << std::endl;
        return 1;
    }
    std::string file_location = argv[1];

    std::list<std::pair<int, int>> parsed_edge_list = read_edge_list(file_location);

    print_edge_list(parsed_edge_list);
    std::unordered_map<int, std::list<int>> adjList = constructAdjList(parsed_edge_list);
    printAdjList(adjList);




    return 0;
}