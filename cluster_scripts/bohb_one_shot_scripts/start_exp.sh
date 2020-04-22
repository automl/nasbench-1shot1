#!/bin/sh

spaces='1 2 3'
algs='darts gdas pc_darts'
cs='1 2 3'
seed='1 2 3'

for s in $spaces; do
       for a in $algs; do
	       for c in $cs; do
		       for sd in $seed; do
			       sbatch -J ${s}_${c}_${sd}_${a} cluster_scripts/bohb_one_shot_scripts/bohb-darts.sh $s $a $c $sd
			       echo submitted job: space $s, $a, cs $c, seed $sd
		       done
	       done
       done
done
