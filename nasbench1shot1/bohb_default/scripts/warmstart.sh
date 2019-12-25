#!/bin/bash
#
# submit to the right queue
#SBATCH -p bosch_gpu-rtx2080
#SBATCH --gres gpu:1
#SBATCH --array=1-16
#SBATCH -J bohb-pc_darts
#
# the execution will use the current directory for execution (important for relative paths)
#SBATCH -D .
#
# redirect the output/error to some files
#SBATCH -o ./logs/%A-%a.o
#SBATCH -e ./logs/%A-%a.e
#
#
source activate tensorflow-stable
python src/warmstart.py --array_id $SLURM_ARRAY_TASK_ID --total_num_workers 16 --num_iterations 64 --run_id $SLURM_ARRAY_JOB_ID --working_directory ./bohb_warm --previous_dir ./bohb_output/search_space_3/pc_darts/run3703171-seed1 --min_budget 25 --max_budget 100 --seed 1 --space 3 --algorithm pc_darts
