#!/bin/bash
# For running on gilbreth via sbatch
# sbatch --nodes=1 --gpus-per-node=1 --partition=a100-40gb --mem=40GB --account=sundara2 bench.bash

#SBATCH --job-name=pendulum
#SBATCH --cpus-per-task=16
#SBATCH --ntasks=1

# Activate the environment
module load conda
conda activate nclv

which python
python --version
lscpu | grep CPU
echo SLURM CPUs per Task: $SLURM_CPUS_PER_TASK

# Get to correct directory or die trying
cd "${SCRATCH}" || { echo "Error: SCRATCH directory not accessible"; exit 1; }
cd NN-Closed-Loop-Verification
pwd

# Run
date
python -u hpc_pendulum.py
date