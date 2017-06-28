#!/bin/bash
for i in `seq 1 50`;
do
       mv iterated_filtering_4$i iterated_filtering_$i 
done    

for i in `seq 1 50`;
do
       mv python_bayesopt$i\_4.pkl python_bayesopt$i.pkl
done    

for i in `seq 1 50`;
do
       mv python_bayesopt$i\_4.dat python_bayesopt$i.dat
done    