#!/bin/bash
# For running on gilbreth via sbatch

# Activate the environment
module load conda
conda activate nclv

# Get to correct directory or die trying
cd "${SCRATCH}" || { echo "Error: SCRATCH directory not accessible"; exit 1; }
cd NN-Closed-Loop-Verification
pwd

# Run
jupyter nbconvert --clear-output --inplace --execute hpc.ipynb