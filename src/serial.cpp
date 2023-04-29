//
// Created by Felix Hohne on 4/3/23.
//

#include "serial.h"
#include <cmath>
#include <vector>
#include <list>
#include <iostream>
#include <unordered_map>
#include <vector>
#include <set>
#include <queue>
#include <bits/stdc++.h>
#include <stdexcept>
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

void relax(int v, int new_dist, std::unordered_map<int, std::list<int>>& B, int* dists, int delta) {
    // std::cout << "relax values: " << dists[v] << ": " << new_dist << std::endl;
    if (new_dist < dists[v]) {
        B[floor(dists[v] / delta)].remove(v);
        int new_bucket = floor(new_dist / delta);
        // std::cout << "new bucket" << new_bucket << std::endl;
        if (!B.contains(new_bucket)) {
            B.insert({new_bucket, std::list<int>()});
        }
        B[new_bucket].push_back(v); 
        dists[v] = new_dist; 
    }

    // std::cout << "B size: " << B.size() << std::endl;
}

void delta_stepping(CSR graph, int source, int num_nodes, int* dists, int* preds, int Delta) {
    // TODO: implement
    std::unordered_map<int, std::list<int>> heavy;
    std::unordered_map<int, std::list<int>> light;
    std::unordered_map<int, std::list<int>> B;
    std::vector<int> S;
    
    // initialize heavy and light
    for (int i = 0; i < num_nodes; i++) {
        std::list<int> heavy_list;
        std::list<int> light_list;
        for (int j = graph.rowPointers[i]; j < graph.rowPointers[i+1]; j++) {
            // TODO: edge weights
            int weight = 1;
            if (weight > Delta) {
                heavy_list.push_back(graph.neighborNodes[j]);
            } else if (weight > 0) {
                light_list.push_back(graph.neighborNodes[j]);
            } else {
                throw std::invalid_argument("Cannot have negative edge weights");
            }
        }
        if (!heavy_list.empty()) {
            heavy.insert({i, heavy_list});
        }
        if (!light_list.empty()) {
            light.insert({i, light_list});
        }
    }

    std::cout << "Light size: " << light.size() << std::endl; 


    // initialize tentative distances
    for (int i = 0; i < num_nodes; i++) {
        dists[i] = INT_MAX;
    }
    
    relax(source, 0, B, dists, Delta);
    std :: cout << "At begin, B has size: " << B.size() << std :: endl;
    int i = 0;


    while (!B.empty()) {
        std :: cout << "i: " << i << "B size: " << B.size() << std :: endl;
        // std::cout << "While B size" << B.size() << std::endl; 
        if (B.find(i) == B.end()) {
            // std:: cout << "B" << "[" << i << "] is empty" << std::endl;
            i++;
            continue; 
        }
        S.clear();
        std::unordered_map<int, int> Req; 

        while (!B[i].empty()) {
            // initialize Req
            for (auto v: B[i]) {
                if (light.contains(v)) {
                    for (auto w: light[v]) {
                        // TODO: Fix edge weights
                        int new_distance = dists[v] + 1;
                        if (Req.contains(w)) {
                            new_distance = std::min(Req[w], new_distance);
                        }
                        Req.insert({w, new_distance});
                    }
                }
                S.push_back(v);
            }
            B.erase(i);
            
            for (const auto &pair: Req) {
                relax(pair.first, pair.second, B, dists, Delta);
            }
        }
        Req.clear();
        for (auto v: S) {
            if (heavy.contains(v)) {
                for (auto w: heavy[v]) {
                    // TODO: Fix edge weights
                    int new_distance = dists[v] + 1;
                    if (Req.contains(w)) {
                        new_distance = std::min(Req[w], new_distance);
                    }
                    Req.insert({w, new_distance});
                }
            }
        }
        // for (const auto &pair: Req) {
        //     relax(pair.first, pair.second, B, dists, Delta);
        // }
        i++;
    }
}
