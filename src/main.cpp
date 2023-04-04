#include <iostream>
#include "GraphLoading.h"
#include "GraphRepresentations.h"
#include "GraphAlgorithms.h"
#include <vector>

int main(int argc, char *argv[]) {
    bool DEBUG_PRINT = false;

    if (argc <= 1) {
        std::cout << "Requires argument for matrix location" << std::endl;
        return 1;
    }
    std::string file_location = argv[1];

    std::list<std::pair<int, int>> parsed_edge_list = read_edge_list(file_location);

    if (DEBUG_PRINT)
        print_edge_list(parsed_edge_list);
    std::unordered_map<int, std::list<int>> adjList = constructAdjList(parsed_edge_list);

    if (DEBUG_PRINT)
        printAdjList(adjList);

    auto start_dijkstra = std::chrono::steady_clock::now();

    auto [min_distances, p] = st_dijkstra(adjList, 0);

    auto end_dijkstra = std::chrono::steady_clock::now();
    std::chrono::duration<double> get_dijkstra_time = end_dijkstra - start_dijkstra;
    double dijkstra_time = get_dijkstra_time.count();

    std::cout << "Time to run Dijkstra: " << dijkstra_time << std::endl;

    if (DEBUG_PRINT) {
        std :: cout << "Reporting minimum distances: " << std::endl;
        int counter = 0;
        for (auto const distance: min_distances) {
            std::cout << "min distance for " << counter << ": " << distance << std :: endl;
            counter++;
        }
    }




    return 0;
}