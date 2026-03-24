#!/bin/bash
# For running on gilbreth via sbatch
# sbatch --nodes=1 --gpus-per-node=1 --partition=a100-40gb --account=sundara2 bench.bash

#SBATCH --job-name=duffing
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=20G
#SBATCH --ntasks=1
#SBATCH --time=03:00:00

# Activate the environment
module load conda
conda activate rl

which python
python --version
hostname
lscpu | grep CPU
echo SLURM CPUs per Task: $SLURM_CPUS_PER_TASK

# Get to correct directory or die trying
cd "${SCRATCH}" || { echo "Error: SCRATCH directory not accessible"; exit 1; }
cd NN-Closed-Loop-Verification
pwd

HIDDEN_SIZE=100

echo Statarting: `date`

echo Running warmup...
python -m relu_vnn verify

echo "1 worker"
time python -u -m relu_vnn verify \
    --device cuda \
    --max-workers 1 \
    --hole 0.004 \
    --hidden-size $HIDDEN_SIZE \
    --checkpoint demos/duffing/100n_pass/duffing_100n.pt \
    demos/duffing/100n_pass/duffing_oscillator.py

echo "8 workers"
time python -u -m relu_vnn verify -v \
    --device cuda \
    --max-workers 8 \
    --hole 0.004 \
    --hidden-size $HIDDEN_SIZE \
    --checkpoint demos/duffing/100n_pass/duffing_100n.pt \
    demos/duffing/100n_pass/duffing_oscillator.py

echo "16 workers"
time python -u -m relu_vnn verify -v \
    --device cuda \
    --max-workers 16 \
    --hole 0.004 \
    --hidden-size $HIDDEN_SIZE \
    --checkpoint demos/duffing/100n_pass/duffing_100n.pt \
    demos/duffing/100n_pass/duffing_oscillator.py

echo "32 workers"
time python -u -m relu_vnn verify -v \
    --device cuda \
    --max-workers 32 \
    --hole 0.004 \
    --hidden-size $HIDDEN_SIZE \
    --checkpoint demos/duffing/100n_pass/duffing_100n.pt \
    demos/duffing/100n_pass/duffing_oscillator.py

echo "64 workers"
time python -u -m relu_vnn verify -v \
    --device cuda \
    --max-workers 64 \
    --hole 0.004 \
    --hidden-size $HIDDEN_SIZE \
    --checkpoint demos/duffing/100n_pass/duffing_100n.pt \
    demos/duffing/100n_pass/duffing_oscillator.py

echo "96 workers"
time python -u -m relu_vnn verify -v \
    --device cuda \
    --max-workers 96 \
    --hole 0.004 \
    --hidden-size $HIDDEN_SIZE \
    --checkpoint demos/duffing/100n_pass/duffing_100n.pt \
    demos/duffing/100n_pass/duffing_oscillator.py

echo "128 workers"
time python -u -m relu_vnn verify -v \
    --device cuda \
    --max-workers 128 \
    --hole 0.004 \
    --hidden-size $HIDDEN_SIZE \
    --checkpoint demos/duffing/100n_pass/duffing_100n.pt \
    demos/duffing/100n_pass/duffing_oscillator.py

echo Complete: `date`