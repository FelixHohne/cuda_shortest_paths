# Scaling Single-Source Shortest Paths with CUDA
## Felix Hohne, Shiyuan Huang, and Adarsh Sriram


## Introduction


In this project, we implemented algorithms for solving the Single-Source
Shortest Paths (SSSP) problem in CUDA. We demonstrate that though
the GPU appears more suitable for dense computation, GPUs will be able
to accelerate SSSP on large graphs, with good performance when compared
to CPUs with many cores. 


## CUDA-Based Delta-Stepping

### Algorithm Description

We implement the delta-stepping algorithm proposed by Meyer and Sanders in their paper
*$\Delta$-stepping: a parallelizable shortest path algorithm*[^3]. The
intuition behind the delta-stepping algorithm is based on the following
observations. While Dijkstra's has a good time complexity, it exposes very little
parallelism, as it only processes one minimum-distance element from a
priority queue at a time. On the other hand, Bellman-Ford has a much
higher time complexity, but exposes
a lot of parallelism, as it processes all edges in the graph in each
iteration. Delta-stepping interpolates these two by utilizing a
coarse-grained priority queue that assigns all nodes whose tentative
distance is in a given range, and
processing all nodes in a single bucket at a time. Setting the bucket size to 1
corresponds to Dijkstra's, while setting the bucket size to infinity corresponds to
Bellman-Ford, which allows us to tune the parallelism in the algorithms.  


### CUDA-Based Implementation 

In addition to the arrays required for the graph's CSR and the resultant
distances, we first initialize the following data-structures and
allocate them on device memory, including: $|E|$ sized arrays $d\_light$
and $d\_heavy$ corresponding to the pseudo-code's Light and Heavy
arrays, which contain a valid node-id if the neighbor node corresponds
to a light/heavy edge of the node, and -1 otherwise; $|E|$ sized array
$d\_Req$ of ReqElement structs containing nodeId and current distance
from the source, that corresponds to Req in the pseudo-code; $|V|$ sized
array $d\_B$ that maps nodes (thread-index) to their bucket-index; and
$d\_S$, corresponding to set S in the pseudo-code. Each array is
initialzed within CUDA-kernels, which are load-balanced depending on the
array-size (i.e. 1 thread per edge or node).

Since data structures corresponding to the buckets and Req are only
modified by CUDA kernels, we needed additional logic to copy the current
size of the relevant arrays from GPU to CPU after every modification,
and have the host perform checks (e.g. boolean $is\_empty$ checks if all
buckets (or a specific one) are empty) / pass data to downstream kernels
(e.g. passing $d\_Req\_size$ to the relax procedure). We allocated these
variables in CUDA's unified memory using cudaMallocManaged to allow host
access. In our implementation, entries in array $d\_Req$ are sorted and
valid up to the size contained in $d\_Req\_size$. At a high-level, we
have an outer while-loop that checks if all buckets are $is\_empty$
(updated as above each iteration), and processes buckets in order of
their indices. At each iteration, we have a kernel to clear $d\_S$ (fill
with -1), and then begin processing Light edges. We re-use $is\_empty$
to check if the current bucket $B_i$ being processed is empty in another
while loop. Within this loop, the kernel computeLightReq updates
$d\_Req$ with Light edges (and updates $d\_Req\_size$), 1 thread per
Light edge. After this, kernel $update\_S\_clear\_B\_i$ updates set S as
in the pseudo-code and clears the current bucket $B_i$, i.e. fills
corresponding elements in $d\_B$ with -1, with a thread per node. We
then sort the first $d\_Req\_size$ elements (ReqElement structs) of
$d\_Req$ using $thrust::sort$ and a custom comparator that first
compares structs by nodeId and breaks ties with current source-distance.
Then, a kernel performs the relax procedure with the sorted Req elements
and updates buckets (1 thread per edge) according to the pseudo-code.
Finally, another kernel checks if the current bucket $B_i$ is empty and
updates $is\_empty$ accordingly - completing the Light edges processing
for this outer-loop iteration.

For processing Heavy edges, the kernel computeHeavyReq updates $d\_Req$
with Heavy edges as in the pseudo-code and recomputes $d\_Req\_size$ (1
thread per node). Then, we again call the relax kernel to process the
updated $d\_Req$ and update buckets according. Finally, a kernel checks
if all buckets are empty and updates $is\_empty$ accordingly, and we
move on to the next outer-loop iteration that will process bucket
$B_{i+1}$.

### Asynchronous Memory

One optimization we considered and implemented was utilizing
asynchronous CUDA memory operations. The origin of this idea was the
fact that the Delta-Stepping implementation requires a large number of
memory allocations, from constructing GPU versions of the sparse CSR
data structure to the various arrays needed for the Delta-Stepping
implementation. Some subsets of these arrays were necessary for certain
initializations to be completed, but others were not. Thus, it seemed
wasteful to block after every single memory operation.

As a result, we decided to use
cudaMemcpyAsync on different streams for the various different arrays,
while using event streams to synchronize the events that needed to be
completed before certain initializations could be conducted. In doing
this, all the memory allocations could be conducted in parallel, without
waiting for each cudaMalloc to return, before initializing the next
cudaMalloc.

### Edge-Wise Req Computation

Based on our timing results, we realized that the Light Req computation
was an expensive portion of the overall run-time, particularly for
graphs in which our algorithm did not provide a high speedup, so accelerating 
Light Req computation would make a major
difference. We also noted that WikiTalk spent a lot of time in Light
Req, even though it had very low diameter. We suspected one cause of
this could be because of load imbalance, in that some threads may be
assigned much more work than other threads.

Our initial assumption was that using shared memory could be a path
towards accelerating this code. However, when analyzing our
implementation, we realized that our implementation was too tied with
the underlying sparse CSR graph representation. In particular, for each
specific node, we look up its edges, and then update Req based on the
values of these edges. This is because CSR does not natively make it
possible to parallelize across edges. Because we must operate on the
edges for each node in parallel, and we do not know a priori how many
edges a single node can have, there was no easy way to allocate the
right amount of shared memory per block. Thus, accelerating this code
via shared memory was not feasible.

As a result, we instead decided to compute a new array, once at the
beginning of the overall execution, that computes the node pointing
backwards given an edge. This means we can instead parallelize across
edges in computeLightReq. Not only does this reduce the number of array
lookups we need to conduct for each iteration of light req, this should
also improve the load distribution, as it is possible that some
super-star nodes have vastly more edges than other nodes, resulting in
inefficient work distributions.

## Timing

All timing results measure the time the overall SSSP takes from
initialization to return, given the correct graph representation input
(Adjacency Matrix or Sparse CSR, always on CPU). We do not time parsing
or graph construction, even though it is a very significant portion of
overall runtime execution. It likely would have been possible to use
OpenMP to accelerate this task, but we felt the parallelization in this
problem was not very interesting, and thus we focused primarily on the
underlying SSSP algorithm. Timing was based on CUDA events, wrapping the
time key kernels took to run. In timing pieces of the execution, we
focused on the kernels that we felt were most important in the
algorithm, and then verified that the remaining time was relatively low.
Certainly, a greater degree of precision could have been useful, but the
timing code results in a lot of code bloat, making it harder to edit and
modify over time.

# Performance Analysis

Firstly, we see that performance speedup of our GPU Delta-Stepping
algorithm depends greatly on the specific graph. We note that each
improvement of the Delta-Stepping algorithm improves performance, with
all optimizations achieving the highest overall speedup. We noted that
the performance did not vary substantially with $\Delta$, and thus we
set $\Delta = 50$ for all experiments.

![image](images/performance_analysis.png){width="100%"}

To better understand our implementation, we analyzed where the algorithm
was spending time in Asynchronous Delta-Stepping and Edge-Wise
Asynchronous Delta-Stepping. For Asynchronous Delta-Stepping, we see
that in the high-diameter graph, roadNet-CA, computing Light Req is very
expensive. For graphs with lower node diameter, it is less critical, but
still expensive. As discussed above, this was a key motivation for
implementing Edge-Wise Delta-Stepping.

More interestingly, we observed that thrust sort could be very fast on
some graphs. This is because for most iterations, the number of elements
in Req is very small, perhaps a few thousand in a graph of millions of
nodes. Thus, by densely packing Req, we only need to sort a relatively
short array, rather than a very long one. This validated our original
approach to favor densely packing Req in particular.

![image](images/async_analysis.png){width="\\textwidth"}

Given the implementation of Edge-Wise Delta-Stepping, we saw major
performance improvement. We note that though roadNet-CA still spends the
majority of its time in Light Req, the time for light Req declined from
1957.73 to 1041.03 miliseconds, with a corresponding decline in thrust
sort that was more unexpected. WikiTalk also saw a dramatic performance
improvement, which strengthened our suspicion that this was a load
imbalance problem in the original implementation.

We also compared our CUDA delta-stepping implementations against serial
Bellman Ford. Since Bellmand Ford runs in $\mathcal{O}(|V||E|)$ time, we
had to extrapolate total runtime based on performance on $10*|E|$
iterations for each dataset to avoid timeouts. We then compared the
ratio of speedup obtained by Asynchronous Delta-Stepping to the speedup
obtained by Edge-Wise Async Delta-Stepping. We found that for
LiveJournal, both implementations achieved similar speedup, but for
Pokec, web-Stanford and roadNet-CA, the ratios of speedups were between
0.6-0.7, and for WikiTalk, the ratio was 0.26. We discuss a possible
explanation for this in section 4.2.

![image](images/edge_wise_analysis.png){width="\\textwidth"}

## Profiling

We also analyzed our implementation using Nvidia Nsight profiling tools,
to gain a better understanding of our kernel implementations. These
analyses were done on the Pokec graph. Unexpectedly, our kernels have
low memory and compute throughput. For example, computing light req has
54.22% memory and 13.76% compute. The low compute utilization was to be
expected, as we assumed from the output that the performance of
Delta-Stepping was memory-bound.

When analyzing computeLightReq, it is expensive because it reads
15,545,368 blocks from Device Memory. We looked over this function
again, and there does not appear a way to easily speed this up, as it is
far too large to be placed in shared memory, and does not have
substantial data reuse. We also noted the high occupancy for the
function, at 88.90%. When using 512 threads, this drops substantially,
and thus we did not explore lower thread counts, as using fewer threads
likely would have resulted in even less saturation of the Streaming
Multiprocessors of the A100. Other functions like update_S\_clear_B\_i
don't write to many elements of the array, but this behavior is required
by the algorithm. Thus, despite the L1 being a 128 byte cache line, only
very little of this data is being used. The is empty checking functions
such as is_empty_B have a large number of branches; however, they appear
to have 0 avg. divergent branches, which appears to suggest that branch
prediction is pretty good and the performance overhead is perhaps less
than originally expected.

To some extent, the profiling suggests that we are running into the
limits of the Delta-Stepping algorithm itself. To improve parallelism
further, we likely need to move to a Linear Algebra based variant, which
can expose more parallelism.

## Effect of Network Topology

To more comprehensively understand the variation in performance, we
analyzed the performance speedups our Delta-Stepping implementations
obtained over Serial Dijkstra w.r.t topological properties of each
graph. Specifically, we looked for trends w.r.t: Diameter (D) (and
90th-percentile effective diameter), average clustering coefficient, and
the relative size of the largest strongly-connected component (SCC) of
each graph. Statistics are as follows:

                 **90th-Percentile Diameter**   **Relative LSCC (nodes)**   **Relative LSCC (edges)**   **Avg Clustering Coeff**
  -------------- ------------------------------ --------------------------- --------------------------- --------------------------
  roadNet-CA     511.01                         0.99                        0.99                        0.46
  LiveJournal    6.18                           0.79                        0.95                        0.12
  Pokec          5.23                           0.79                        0.95                        0.11
  Web-Stanford   9.7                            0.54                        0.68                        0.59
  WikiTalk       3.87                           0.083                       0.29                        0.05

For Edge-wise Asynchronous Delta-Stepping, we saw a definitive trend of
performance speedup decreasing with increase in graph diameter, and saw
no relationship w.r.t other network-properties.

For Asynchronous Delta-Stepping, we also saw performance speedup
decreasing with an increase in diameter, with the exception of the
WikiTalk network (D = 9, speedup = 39.33), which saw a smaller speedup
compared to the Pokec network (D = 11, speedup = 59.50). We also observe
in section 4.1 that the Light-Req computation during Asynchronous
Detla-Stepping for WikiTalk is similar to that of roadNet-CA, despite
the latter having a much larger diameter. A likely explanation for this
outlier is the fact that for WikiTalk, only 29.4% of edges and 8.3% of
nodes are contained in the largest SCC, while for Pokec, 95.3% of edges
and 78.9% of nodes are contained in the largest SCC (similar to other
networks). Thus, we might expect that using asynchronous memory alone
does result in the best memory-access patterns for WikiTalk's Light-Req
since the load is not balanced across its edges. Furthermore, we see
that the Edge-wise computation of Light-Req for WikiTalk is an order of
magnitude faster, while the Light-Req computation time for roadNet-CA is
identical to the Edge-wise version. This can also be explained by the
fact that 99% of roadNet-CA's edges and nodes are contained in its
largest SCC, and thus load-balancing computation across edges does not
do much better than asynchronous memory alone.

Regarding speedups w.r.t serial Bellman-Ford, since the serial run-times
are dominated by the size of node and edge-sets, we were not able to
observe any specific trends w.r.t topological properties. However, as
discussed before, we noticed that on WikiTalk, the Edge-Wise
optimizations gives 4 times the relative speedup compared to Async
memory alone. Again, we believe that the edge distribution in WikiTalk
influences this ratio, and since serial-Bellman Ford sequentially
iterates through all edgdes of the graph $|V|-1$ times, it is an
unfavorable memory access pattern for a graph like WikiTalk.

## Weak Scaling

We could not perform a strong scaling analysis because we did not
implement a multi-GPU algorithm, and we were unable to set the number of
blocks that could execute in parallel by, for instance, restricting the
number of streaming multi-processors used. Thus, we focused on weak
scaling instead. As recommended by Professor Guidi, we measured weak
scaling by sampling a percentage of edges from our input edge lists to
construct smaller graphs. Since we mostly use edge-wise parallelism, the
number of GPU threads used increases proportional to the number of
edges, allowing us to keep a roughly fixed amount of work per thread. We
sampled 25%, 50%, 75%, and 100% of edges from each of the five graphs we
worked with and provide a log-log plot for weak scaling of edge-wise
async delta-stepping below.

::: center
![image](weak.png){width="15cm"}
:::

As expected, we do not get perfect weak scaling for many of the
datasets. One interesting exception to this is the webStanford dataset,
which ran much faster when 50% of edges were sampled instead of 25% of
edges. This might be because sampling edges does not always end up
having a predictable effect on the SSSP problem. For example, it might
be that the percentage of edges sampled doesn't correlate well with the
number of nodes reachable from the source node, and the webStanford weak
scaling behavior might occur because sampling 25% of the edges results
in a sub-graph that has almost as many reachable nodes as the original
graph. Conversely, if sampling 50% of the graph creates a subgraph with
more than twice as many reachable nodes as the subgraph created by
sampling 25% of the graph, this could make weak scaling worse than
expected.

# Difficulties

Our primary difficulty was not always having access to compute with
GPUs. All three team members were using ARM Macs, and thus could not
develop locally. As Perlmutter was frequently down, there were large
stretches were only one group members with Cornell G2 access could run
and debug code, with other team members only able to write code without
the ability to test. Furthermore, Perlmutter's CUDA configuration was
very different from G2, resulting in a number of continuous build issues
when trying to get our codes to run on both machines. Finally, G2's
compiler and run-time environment is quite different from Perlmutter;
for example, most arrays are 0-initialized, and often do not result in
segmentation faults with out of bounds array accesses. As a result,
frequently, our code would appear to be correct when we were testing on
G2, and when Perlmutter resumed operation, we would realize that in
fact, there were still other errors that we needed to track down.
Debugging Delta-Stepping also proved to be quite hard, though this is
not unexpected. Here, again we struggled with CMake to enable the
various flags to get better debugging output.

Ultimately, we were also quite surprised at how great the difference the
performance speedup between various graphs was. I think it is quite
unfortunate that road networks turn out to be very hard to parallelize
using Delta-Stepping, as tools such as Google Maps are a key application
of SSSP. A hypothesis that turned out to be wrong again and again was
that shared memory could lead to performance improvement. This is not
because shared memory isn't faster, but often because we conceived of
shared memory as a crutch to speed up an inefficient approach to
parallelization. Instead, re-writing the parallelism and exposing more
parallelism naively, often with better coalesced reads from global
memory was more effective, as Delta-Stepping, unlike something like
stencils, does not re-use elements of the data array sufficiently to get
GPU speedup.

# Appendix

In the appendix, we report overall runtimes. All datasets were
downloaded from Stanford's SNAP repository of graphs.

::: {#tab:mini_results}
                  Dijkstra   $\Delta$-Stepping   Bellman-Ford   $\Delta$-Stepping   A. $\Delta$-Stepping   EWA $\Delta$-Stepping
  ------------- ---------- ------------------- -------------- ------------------- ---------------------- -----------------------
  roadNet-CA        2.9560              9.7069       251.1790              3.5940                 2.3863                  1.4876
  LiveJournal       3.1087             86.7250              T              1.2841                 0.1844                  0.1775
  Pokec             6.5813             37.5810      2225.2700              1.0603                 0.1106                  0.0806
  webStanford       0.5467              2.9678      2996.7000              1.2988                 0.2101                  0.1309
  wikiTalk          4.4071              8.9862              T              1.1419                 0.1120                  0.0290

  : Run-times of various algorithms. The first two (Dijkstra and
  $\Delta$-Stepping are serial, CPU implementations, while the remainder
  are run on GPU. We record T for if the execution time is longer than
  12 hours on Perlmutter A100 GPU. A. $\Delta$-Stepping is short-hand
  for Asynchronous Delta-Steping and EWA $\Delta$-Stepping stands for
  Asynchronous + Edge-Wise Delta-Stepping. Speedup recorded versus
  Serial Dijkstra.
:::

[^1]: Work-Efficient Parallel GPU Methods for Single-Source Shortest
    Paths by Andrew Davidson, Sean Baxter, Michael Garland, and John D.
    Owens

[^2]: Simd-x: Programming and processing of graph algorithms on GPUs by
    Hang Liu, H. Howie Huang

[^3]: $\Delta$-stepping: a parallelizable shortest path algorithm by U.
    Meyer and P. Sanders

[^4]: Delta-stepping SSSP: from Vertices and Edges to GraphBLAS
    Implementations by Upasana Sridhar, Mark Blanco, Rahul Mayuranath,
    Daniele G. Spampinato, Tze Meng Low, and Scott McMillan
