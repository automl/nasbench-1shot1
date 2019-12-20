#!/bin/sh

spaces='1 2'
algs='darts gdas pc_darts'

for s in $spaces; do
       for a in $algs; do
	       sbatch scripts/bohb-darts.sh $s $a
	       echo submitted job $s $a
       done
done       
