//
// Created by Felix Hohne on 4/3/23.
//

#include "serial.h"
#include <vector>
#include <list>
#include <iostream>
#include <unordered_map>
#include <vector>
#include <set>
#include <queue>
#include <bits/stdc++.h>

// Assume no max path length over INT_MAX.
const int INF = INT_MAX;

void st_dijkstra(std::unordered_map<int, std::list<int>> adjList, int source, 
int num_nodes, int* d, int* p) {
    /*
     * Note: Implementation using Red-Black Tree via std::set.
     * Current implementation assumes edge weights are always 1.
     */

    for (int i = 0; i < num_nodes; i++) {
        d[i] = INF; 
        p[i] = -1;
    }
    
    d[source] = 0;
    std::set<std::pair<int, int>> q; // q.first = cost, q.second is node id
    q.insert({0, source});
    while (!q.empty()) {
        int v = q.begin()->second;
        q.erase(q.begin());
        for (auto edge: adjList[v]) {
            int to = edge;
            // TODO: support graphs with weighted edges
            int len = 1;

            if (d[v] < INT_MAX && d[v] + len < d[to]) {
                q.erase({d[to], to}); 
                d[to] = d[v] + len;
                p[to] = v;
                q.insert({d[to], to});
            }
        }
    }
   
}

void relax(int v, int new_dist, std::unordered_map<int, std::list<int>> buckets, int* dists) {
    // TODO: implement
}

void delta_stepping(std::unordered_map<int, std::list<int>> adj_list, int source, int num_nodes, int* dists, int* preds, int delta) {
    // TODO: implement
    std::unordered_map<int, std::list<int>> heavy;
    std::unordered_map<int, std::list<int>> light;
    std::unordered_map<int, std::list<int>> buckets;
    std::vector<int> S;
    for (int i = 0; i < num_nodes; i++) {
        dists[i] = INT_MAX;
    }
    relax(source, 0, buckets, dists);
    int i = 0;
    while (!buckets.empty()) {
        S.clear();
        i++;
    }
}
