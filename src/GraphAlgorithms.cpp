//
// Created by Felix Hohne on 4/3/23.
//

#include "GraphAlgorithms.h"
#include <vector>
#include <list>
#include <iostream>
#include <unordered_map>
#include <vector>
#include <set>
#include <queue>

// Assume no max path length over 1B.
const int INF = 1000000000;

std::pair<std::vector<int>, std::vector<int>> st_dijkstra(std::unordered_map<int, std::list<int>> adjList, int source) {
    /*
     * Note: Implementation using Red-Black Tree via std::set.
     * Current implementation assumes edge weights are always 1.
     */
    int n =  adjList.size();
    std::vector<int> d(n, INF);
    std::vector<int> p(n, -1);

    d[source] = 0;
    std::set<std::pair<int, int>> q; //q.first = cost, q.second is node id
    q.insert({0, source});
    while (!q.empty()) {
        int v = q.begin() -> second;
        q.erase(q.begin());
        for (auto edge: adjList[v]) {
            int to = edge;
            int len = 1;

            if (d[v] + len < d[to]) {
                q.erase({d[to], to});
                d[to] = d[v] + len;
                p[to] = v;
                q.insert({d[to], to});
            }
        }
    }
    return std::make_pair(d, p);
}
