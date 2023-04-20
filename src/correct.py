import numpy as np 
import pandas as pd 
import networkx as nx 
import sys

def main(argv):
   if len(argv) != 1:
      raise ValueError("Must pass in graph representation input")
   
   df = pd.read_csv(argv[0], sep="\t")
   assert len(df.index) > 0

   G = nx.from_pandas_edgelist(
      df, 
      "FromNodeId", 
      "ToNodeId")
   
   assert G.number_of_edges() > 0 
   p = nx.shortest_path_length(G, source=0)

   p_sorted = dict(sorted(p.items()))
   

   f = open("python_shortest_path_outs.txt", "w")
   f.write("source\tdistance\n")
   for (key, value) in p_sorted.items():
      f.write(f"{key}\t{value}\n")

 
if __name__ == "__main__":
   main(sys.argv[1:])


