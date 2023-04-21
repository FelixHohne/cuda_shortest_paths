import numpy as np 
import pandas as pd 
import networkx as nx 
import sys

def main(argv):
   if len(argv) != 1:
      raise ValueError("Must pass in graph representation input")
   
   df = pd.read_csv(argv[0], sep="\t")
   assert len(df.index) > 0

   DEBUG = False

   G = nx.from_pandas_edgelist(
      df, 
      "FromNodeId", 
      "ToNodeId")

   if DEBUG:
      print("G number of nodes: ", G.number_of_nodes())
      print("G max node: ", max(list(G.nodes)))
      print("G number of edges: ", G.number_of_edges())

      print("Number of edges for node 9721: ", G.degree[9721]) 
      print("9721 neighbors: ", G.edges(9721))
   assert G.number_of_edges() > 0 
   p = nx.shortest_path_length(G, source=0)

   p_sorted = dict(sorted(p.items()))
   
   f = open("python_shortest_path_outs.txt", "w")
   f.write("source\tdistance\n")
   for (key, value) in p_sorted.items():
      f.write(f"{key}\t{value}\n")

 
if __name__ == "__main__":
   main(sys.argv[1:])


