#!/bin/bash
#
# submit to the right queue
#SBATCH -p bosch_gpu-rtx2080
#SBATCH --gres gpu:1
#SBATCH -a 0-499
#SBATCH -J bohb-nasbench
#
# the execution will use the current directory for execution (important for relative paths)
#SBATCH -D .
#
# redirect the output/error to some files
#SBATCH -o ./experiments/cluster_logs/%A_%a.o
#SBATCH -e ./experiments/cluster_logs/%A_%a.e
#
#

source activate tensorflow-stable
PYTHONPATH=$PWD python optimizers/bohb/run_bohb.py --seed $SLURM_ARRAY_TASK_ID --search_space $1

