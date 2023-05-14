#!/bin/sh
echo "roadNet-CA.txt"
./graph_cuda -f ../datasets/roadNet-CA.txt -a gpu-delta-stepping -s 0 -w 25 &> ../perf/roadNet-CA-25-out.txt
./graph_cuda -f ../datasets/roadNet-CA.txt -a gpu-delta-stepping -s 0 -w 50 &> ../perf/roadNet-CA-50-out.txt
./graph_cuda -f ../datasets/roadNet-CA.txt -a gpu-delta-stepping -s 0 -w 75 &> ../perf/roadNet-CA-75-out.txt
./graph_cuda -f ../datasets/roadNet-CA.txt -a gpu-delta-stepping -s 0 -w 100 &> ../perf/roadNet-CA-100-out.txt

echo "soc-LiveJournal.txt"
./graph_cuda -f ../datasets/soc-LiveJournal.txt -a gpu-delta-stepping -s 0 -w 25 &> ../perf/soc-LiveJournal-25-out.txt
./graph_cuda -f ../datasets/soc-LiveJournal.txt -a gpu-delta-stepping -s 0 -w 50 &> ../perf/soc-LiveJournal-50-out.txt
./graph_cuda -f ../datasets/soc-LiveJournal.txt -a gpu-delta-stepping -s 0 -w 75 &> ../perf/soc-LiveJournal-75-out.txt
./graph_cuda -f ../datasets/soc-LiveJournal.txt -a gpu-delta-stepping -s 0 -w 100 &> ../perf/soc-LiveJournal-100-out.txt

echo "soc-pokec-relationships.txt"
./graph_cuda -f ../datasets/soc-pokec-relationships.txt -a gpu-delta-stepping -s 0 -w 25 &> ../perf/soc-pokec-relationships-25-out.txt
./graph_cuda -f ../datasets/soc-pokec-relationships.txt -a gpu-delta-stepping -s 0 -w 50 &> ../perf/soc-pokec-relationships-50-out.txt
./graph_cuda -f ../datasets/soc-pokec-relationships.txt -a gpu-delta-stepping -s 0 -w 75 &> ../perf/soc-pokec-relationships-75-out.txt
./graph_cuda -f ../datasets/soc-pokec-relationships.txt -a gpu-delta-stepping -s 0 -w 100 &> ../perf/soc-pokec-relationships-100-out.txt

echo "web-Stanford.txt"
./graph_cuda -f ../datasets/web-Stanford.txt -a gpu-delta-stepping -s 0 -w 25 &> ../perf/web-Stanford-25-out.txt
./graph_cuda -f ../datasets/web-Stanford.txt -a gpu-delta-stepping -s 0 -w 50 &> ../perf/web-Stanford-50-out.txt
./graph_cuda -f ../datasets/web-Stanford.txt -a gpu-delta-stepping -s 0 -w 75 &> ../perf/web-Stanford-75-out.txt
./graph_cuda -f ../datasets/web-Stanford.txt -a gpu-delta-stepping -s 0 -w 100 &> ../perf/web-Stanford-100-out.txt

echo "WikiTalk.txt"
./graph_cuda -f ../datasets/WikiTalk.txt -a gpu-delta-stepping -s 0 -w 25 &> ../perf/WikiTalk-25-out.txt
./graph_cuda -f ../datasets/WikiTalk.txt -a gpu-delta-stepping -s 0 -w 50 &> ../perf/WikiTalk-50-out.txt
./graph_cuda -f ../datasets/WikiTalk.txt -a gpu-delta-stepping -s 0 -w 75 &> ../perf/WikiTalk-75-out.txt
./graph_cuda -f ../datasets/WikiTalk.txt -a gpu-delta-stepping -s 0 -w 100 &> ../perf/WikiTalk-100-out.txt