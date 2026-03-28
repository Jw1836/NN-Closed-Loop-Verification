#!/bin/bash
# For running on gilbreth via sbatch
# sbatch --nodes=1 --gpus-per-node=1 --partition=a100-40gb --account=sundara2 bench.bash

#SBATCH --job-name=duffing
#SBATCH --cpus-per-task=96
#SBATCH --mem=128G
#SBATCH --ntasks=1
#SBATCH --time=06:00:00

echo Starting: `date`

# Activate the environment
module load conda
conda activate rl

which python
python --version
hostname
lscpu
nvidia-smi
echo SLURM CPUs per Task: $SLURM_CPUS_PER_TASK

# Get to correct directory or die trying
cd $SLURM_SUBMIT_DIR
pwd

echo /////////////////////////////////////////////////////////////

echo Running warmup...
python -m relu_vnn verify --help

echo "////////////////////////////////// 2 ///////////////////////"
time timeout 1800 python -u -m relu_vnn verify -v \
    --max-workers 1 \
    --hidden-size 2 \
    problems/linear_L1.py

echo "////////////////////////////////// 3 ///////////////////////"
time timeout 1800 python -u -m relu_vnn verify -v \
    --max-workers 1 \
    --hidden-size 3 \
    problems/linear_L1.py

echo "////////////////////////////////// 4 ///////////////////////"
time timeout 1800 python -u -m relu_vnn verify -v \
    --max-workers 4 \
    --hidden-size 4 \
    problems/linear_L1.py

echo "////////////////////////////////// 5 ///////////////////////"
time timeout 1800 python -u -m relu_vnn verify -v \
    --max-workers 16 \
    --hidden-size 5 \
    problems/linear_L1.py

echo "////////////////////////////////// 6 ///////////////////////"
time timeout 1800 python -u -m relu_vnn verify -v \
    --max-workers 48 \
    --hidden-size 6 \
    problems/linear_L1.py

echo "////////////////////////////////// 7 ///////////////////////"
time timeout 1800 python -u -m relu_vnn verify -v \
    --max-workers 96 \
    --hidden-size 7 \
    problems/linear_L1.py

echo "////////////////////////////////// 8 ///////////////////////"
time timeout 1800 python -u -m relu_vnn verify -v \
    --max-workers 96 \
    --hidden-size 8 \
    problems/linear_L1.py

echo "////////////////////////////////// 9 ///////////////////////"
time timeout 2400 python -u -m relu_vnn verify -v \
    --max-workers 96 \
    --hidden-size 9 \
    problems/linear_L1.py

echo "////////////////////////////////// 10 ///////////////////////"
time timeout 2400 python -u -m relu_vnn verify -v \
    --max-workers 96 \
    --hidden-size 10 \
    problems/linear_L1.py

echo ////////////////// Switching to abCROWN: $(date) //////////////////

conda deactivate
conda activate abcrown
python verify_crown.py

echo "////////////////////////////////// 2 ///////////////////////"
time timeout 2800 python -u verify_crown.py \
    --hidden-size 2 \
    problems/linear_L1.py

echo "////////////////////////////////// 3 ///////////////////////"
time timeout 2800 python -u verify_crown.py \
    --hidden-size 3 \
    problems/linear_L1.py

echo "////////////////////////////////// 4 ///////////////////////"
time timeout 2800 python -u verify_crown.py \
    --hidden-size 4 \
    problems/linear_L1.py

echo "////////////////////////////////// 5 ///////////////////////"
time timeout 2800 python -u verify_crown.py \
    --hidden-size 5 \
    problems/linear_L1.py

echo "////////////////////////////////// 6 ///////////////////////"
time timeout 2800 python -u verify_crown.py \
    --hidden-size 6 \
    problems/linear_L1.py

echo "////////////////////////////////// 7 ///////////////////////"
time timeout 2800 python -u verify_crown.py \
    --hidden-size 7 \
    problems/linear_L1.py

echo "////////////////////////////////// 8 ///////////////////////"
time timeout 2800 python -u verify_crown.py \
    --hidden-size 8 \
    problems/linear_L1.py

echo "////////////////////////////////// 9 ///////////////////////"
time timeout 3400 python -u verify_crown.py \
    --hidden-size 9
    problems/linear_L1.py

echo "////////////////////////////////// 10 ///////////////////////"
time  python -u verify_crown.py \
    --hidden-size 10 \
    problems/linear_L1.py

echo Complete: `date`