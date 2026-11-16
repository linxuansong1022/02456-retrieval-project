#!/usr/bin/env bash
# SLURM skeleton for DTU HPC. Never executed on the cluster: check the partition,
# module names and venv path against the current HPC docs before submitting.
#SBATCH --job-name=dpr-v1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=4:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=logs/dpr_v1_%j.out
#SBATCH --error=logs/dpr_v1_%j.err

set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs

# adjust to the cluster modules
module load cuda/12.x
module load python/3.10
source /path/to/venv/bin/activate

python train_dpr.py model=hf_encoder train.epochs=3 train.batch_size=32 train.lr=2e-5 device=cuda
