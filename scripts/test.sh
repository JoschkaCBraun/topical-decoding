#!/bin/bash
#SBATCH --gres=gpu:rtx2080ti:1  # Specify the type and number of GPUs
#SBATCH --time=00:30:00         # Set the maximum time for the job (6 hours and 30 minutes)
#SBATCH --output=test-%j.out    # Output file (%j will be replaced with the job ID to ensure uniqueness)

# Print information about the current job
scontrol show job $SLURM_JOB_ID

# Insert your commands here
nvidia-smi
echo "Hello ML World!"
