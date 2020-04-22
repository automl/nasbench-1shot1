#!/bin/bash
#
# submit to the right queue
#SBATCH -p bosch_gpu-rtx2080
#SBATCH --gres gpu:1
#SBATCH --array=1-16
#
# the execution will use the current directory for execution (important for relative paths)
#SBATCH -D .
#
# redirect the output/error to some files
#SBATCH -o ./experiments/bohb_logs/logs/%A-%a.o
#SBATCH -e ./experiments/bohb_logs/logs/%A-%a.e
#
#
source activate tensorflow-stable
python optimizers/bohb_one_shot/master.py --array_id $SLURM_ARRAY_TASK_ID --total_num_workers 16 --num_iterations 64 --run_id $SLURM_ARRAY_JOB_ID --working_directory ./experiments/bohb_output/cs$3 --min_budget 25 --max_budget 100 --space $1 --algorithm $2 --cs $3 --seed $4
