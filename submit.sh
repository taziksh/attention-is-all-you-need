#!/bin/bash
#SBATCH --job-name=tazik_learning      # Job name
#SBATCH --nodes=1                      # Number of nodes. We only have one node at the moment.
#SBATCH --ntasks=1                     # Number of CPU tasks. Typically, one task is started per node.
#SBATCH --cpus-per-task=1              # Specifies the number of CPUs (which might be interpreted as cores or threads) you wish to allocate to each of those tasks.
#SBATCH --mem=32768                    # Request 8Gb of memory
#SBATCH --gres=gpu:3090:1              # Request 1 3090 GPUs. GRE stands for generic resources.
#SBATCH --time=01:00:00                # Time limit hrs:min:sec
#SBATCH --output=job_%j.out            # Standard output and error log. %j interpolates the job ID

# Source your .bashrc to ensure that the environment is properly set up
# source ~/.bashrc

# Activate your environment (if you're using conda or another virtual environment)
conda activate aj

# Use srun to run the job
srun python -u scratch.py > scratch.out