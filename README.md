# Scaling Single-Source Shortest Paths with CUDA
### Project by Felix Hohne, Shiyuan Huang, and Adarsh Sriram

## Introduction

In this project, we implemented algorithms for solving the Single-Source
Shortest Paths (SSSP) problem in CUDA. We demonstrate that though
the GPU appears more suitable for dense computation, GPUs will be able
to accelerate SSSP on large graphs, with good performance when compared
to CPUs. All experiments were run on the Department of Energy's Perlmutter Computer, 
utilizing Nvidia A100 GPUs. 

## CUDA-Based Delta-Stepping

### Algorithm Description

We implement the delta-stepping algorithm proposed by Meyer and Sanders. The
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

### Primary Implementation

In addition to the arrays required for the graph's CSR and the resultant
distances, we first initialize the following data-structures and
allocate them on device memory, including: $|E|$ sized arrays d_light
and d_heavy corresponding to the pseudo-code's Light and Heavy
arrays, which contain a valid node-id if the neighbor node corresponds
to a light/heavy edge of the node, and -1 otherwise; $|E|$ sized array
d_Req of ReqElement structs containing nodeId and current distance
from the source, that corresponds to Req in the pseudo-code; $|V|$ sized
array d_B that maps nodes (thread-index) to their bucket-index; and
d_S, corresponding to set S in the pseudo-code. Each array is
initialzed within CUDA-kernels, which are load-balanced depending on the
array-size (i.e. 1 thread per edge or node).

Since data structures corresponding to the buckets and Req are only
modified by CUDA kernels, we needed additional logic to copy the current
size of the relevant arrays from GPU to CPU after every modification,
and have the host perform checks (e.g. boolean is_empty checks if all
buckets (or a specific one) are empty) / pass data to downstream kernels
(e.g. passing d_Req_size to the relax procedure). We allocated these
variables in CUDA's unified memory using cudaMallocManaged to allow host
access. In our implementation, entries in array d_Req are sorted and
valid up to the size contained in d_Req_size. At a high-level, we
have an outer while-loop that checks if all buckets are is_empty
(updated as above each iteration), and processes buckets in order of
their indices. At each iteration, we have a kernel to clear d_S (fill
with -1), and then begin processing Light edges. We re-use is_empty
to check if the current bucket $B_i$ being processed is empty in another
while loop. Within this loop, the kernel computeLightReq updates
d_Req with Light edges (and updates d_Req_size), 1 thread per
Light edge. After this, kernel update_S_clear_B_i updates set S as
in the pseudo-code and clears the current bucket B_i, i.e. fills
corresponding elements in d_B with -1, with a thread per node. We
then sort the first d_Req_size elements (ReqElement structs) of
d_Req using $thrust::sort$ and a custom comparator that first
compares structs by nodeId and breaks ties with current source-distance.
Then, a kernel performs the relax procedure with the sorted Req elements
and updates buckets (1 thread per edge) according to the pseudo-code.
Finally, another kernel checks if the current bucket $B_i$ is empty and
updates is_empty accordingly - completing the Light edges processing
for this outer-loop iteration.

For processing Heavy edges, the kernel computeHeavyReq updates d_Req
with Heavy edges as in the pseudo-code and recomputes d_Req_size (1
thread per node). Then, we again call the relax kernel to process the
updated d_Req and update buckets according. Finally, a kernel checks
if all buckets are empty and updates is_empty accordingly, and we
move on to the next outer-loop iteration that will process bucket
$B_{i+1}$.

## Optimizations 

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

##  Performance Analysis

Firstly, we see that performance speedup of our GPU Delta-Stepping
algorithm depends greatly on the specific graph. We note that each
improvement of the Delta-Stepping algorithm improves performance, with
all optimizations achieving the highest overall speedup. We noted that
the performance did not vary substantially with $\Delta$, and thus we
set $\Delta = 50$ for all experiments.

<img width="1083" alt="performance_analysis" src="https://github.com/FelixHohne/cuda_shortest_paths/assets/58995473/2369e41a-3532-4e5d-9b33-0031c65e1d68">

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

<img width="1174" alt="async_analysis" src="https://github.com/FelixHohne/cuda_shortest_paths/assets/58995473/f65211d4-c9cf-4a90-bfa2-f99a01477095">

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

<img width="1022" alt="edge_wise_analysis" src="https://github.com/FelixHohne/cuda_shortest_paths/assets/58995473/cb73364a-aee5-4b64-af64-9871fbaca3e0">

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
