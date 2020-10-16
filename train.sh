#!/bin/bash

cd src/

# collect data.
python collect_data.py 1 4 32 128 512

# train iavi and iql.
for i in 4 32 128 512
do
	python train.py iavi $i
	python train.py iql $i 100
done


