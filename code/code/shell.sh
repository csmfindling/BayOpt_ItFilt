
#!/bin/bash
for SLURM_ARRAY_TASK_ID in {1..100}
do
if [ $SLURM_ARRAY_TASK_ID -le 50 ]
then
    python bayesian_optimisation.py $SLURM_ARRAY_TASK_ID
else
	SLURM_ARRAY_TASK_ID=$(($SLURM_ARRAY_TASK_ID - 50))
	Rscript iterated_filtering.R $SLURM_ARRAY_TASK_ID
fi
done


