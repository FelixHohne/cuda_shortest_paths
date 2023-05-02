#!/bin/bash

wget https://snap.stanford.edu/data/soc-LiveJournal1.txt.gz
gunzip soc-LiveJournal1.txt.gz
tail -n +5 soc-LiveJournal1.txt > soc-LiveJournal.txt
rm soc-LiveJournal1.txt