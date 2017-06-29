#!/bin/bash
#SBATCH --job-name=bias_summ
#SBATCH --output=logs/iterated_filtering.%A.%a.out
#SBATCH --error=logs/iterated_filtering.%A.%a.err
#SBATCH --partition=secondgen,fastgen
#SBATCH --array=1-100:1
#SBATCH --mem-per-cpu=3000
#SBATCH --share

# extracting settings from $SLURM_ARRAY_TASK_ID
echo index $SLURM_ARRAY_TASK_ID

# calling script
module load anaconda2
# echo Calling script for subj $SUBJ_ID cond $COND_ID model $MODEL_ID

if [ $SLURM_ARRAY_TASK_ID -le 50 ]
then
    python bayesian_optimisation.py $SLURM_ARRAY_TASK_ID
else
	SLURM_ARRAY_TASK_ID=$(($SLURM_ARRAY_TASK_ID - 50))
	Rscript iterated_filtering.R $SLURM_ARRAY_TASK_ID
fi


