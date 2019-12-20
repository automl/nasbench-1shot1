#!/bin/bash
#SBATCH -p bosch_gpu-rtx2080 # partition (queue)
#SBATCH --mem 10000 # memory pool for all cores (8GB)
#SBATCH -t 11-00:00 # time (D-HH:MM)
#SBATCH -c 2 # number of cores
#SBATCH -a 1-6 # array size
#SBATCH --gres=gpu:1  # reserves four GPUs
#SBATCH -D /home/siemsj/projects/darts/cnn # Change working_dir
#SBATCH -o log/log_$USER_%Y-%m-%d.out # STDOUT  (the folder log has to be created prior to running or this won't work)
#SBATCH -e log/err_$USER_%Y-%m-%d.err # STDERR  (the folder log has to be created prior to running or this won't work)
#SBATCH -J DARTS_NASBENCH # sets the job name. If not specified, the file name will be used as job name
# #SBATCH --mail-type=END,FAIL # (recive mails about end and timeouts/crashes of your job)
# Print some information about the job to STDOUT
echo "Workingdir: $PWD";
echo "Started at $(date)";
echo "Running job $SLURM_JOB_NAME using $SLURM_JOB_CPUS_PER_NODE cpus per node with given JID $SLURM_JOB_ID on queue $SLURM_JOB_PARTITION";

# Activate conda environment
source ~/.bashrc
conda activate pytorch1_0_1

# Job to perform
if [ 1 -eq $SLURM_ARRAY_TASK_ID ]; then
   PYTHONPATH=$PWD python optimizers/pc_darts/train_search.py --seed=0 --save=unrolled --unrolled --search_space=1 --epochs=100
   exit $?
fi


if [ 2 -eq $SLURM_ARRAY_TASK_ID ]; then
   PYTHONPATH=$PWD python optimizers/pc_darts/train_search.py --seed=1 --save=unrolled --unrolled --search_space=1 --epochs=100
   exit $?
fi


if [ 3 -eq $SLURM_ARRAY_TASK_ID ]; then
   PYTHONPATH=$PWD python optimizers/pc_darts/train_search.py --seed=2 --save=unrolled --unrolled --search_space=1 --epochs=100
   exit $?
fi


if [ 4 -eq $SLURM_ARRAY_TASK_ID ]; then
   PYTHONPATH=$PWD python optimizers/pc_darts/train_search.py --seed=3 --save=unrolled --unrolled --search_space=1 --epochs=100
   exit $?
fi


if [ 5 -eq $SLURM_ARRAY_TASK_ID ]; then
   PYTHONPATH=$PWD python optimizers/pc_darts/train_search.py --seed=4 --save=unrolled --unrolled --search_space=1 --epochs=100
   exit $?
fi


if [ 6 -eq $SLURM_ARRAY_TASK_ID ]; then
   PYTHONPATH=$PWD python optimizers/pc_darts/train_search.py --seed=5 --save=unrolled --unrolled --search_space=1 --epochs=100
   exit $?
fi
# Print some Information about the end-time to STDOUT
echo "DONE";
echo "Finished at $(date)";