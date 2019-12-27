#!/bin/bash

method="tpe bohb hb smac"

for m in $method; do
	for space in {1..3}; do
		sbatch cluster_scripts/start-${m}.sh $space
	done
done
