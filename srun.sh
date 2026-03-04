#!/bin/bash
# CLM offload: 1 GPU on Quadro RTX 6000 (single-process, no torchrun)
#SBATCH --output=./slurm/slurm%j.out
#SBATCH --error=./slurm/slurm%j.err
#SBATCH --nodes=1
#SBATCH --gres=gpu:quadro_rtx_6000:1
#SBATCH --mem=128G
#SBATCH --cpus-per-task=16
#SBATCH --partition=gpu
#SBATCH --time=1-00:00:00

set -e

module purge
module load miniforge
module load cuda/12.8.1

cd /u/xna8aw/6501_GPU_architectures/multi_gpu_3dgs
conda activate clm_gs

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0

python train.py \
    -s 'data/rubble-colmap' \
    --clm_offload \
    --bsz 8 \
    --prealloc_capacity 20000000 \
    --eval \
    -m output/rubble_clm
