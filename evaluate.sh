#!/bin/bash

cd src/

# evaluate iavi and iql.
for i in 4 32 128 512
do
    python plot_experiments.py $i
done


