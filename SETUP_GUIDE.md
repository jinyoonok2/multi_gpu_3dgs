# CLM-GS Setup Guide

## 1. Clone and Initialize

```bash
git clone https://github.com/nyu-systems/CLM-GS.git
cd CLM-GS
git config --global url."https://github.com/".insteadOf git@github.com:
git submodule update --init --recursive
```

## 2. Create Conda Environment

```bash
conda create -n clm_gs python=3.10
conda activate clm_gs
```

## 3. Install PyTorch (CUDA 12.4)

```bash
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
```

## 4. Install Python Dependencies

```bash
pip install tqdm plyfile psutil numba opencv-python scipy matplotlib pandas imageio imageio-ffmpeg requests tabulate
```

## 5. Build & Install CUDA/C++ Submodules

These submodules contain custom CUDA kernels that must be compiled. Make sure CUDA is available (`nvcc --version`) before proceeding.

**Why `--no-build-isolation`?** Four of the five submodules import `torch` in their `setup.py` to use `torch.utils.cpp_extension`. By default, pip builds packages in an isolated virtual environment that does **not** have torch installed, causing the build to fail. The `--no-build-isolation` flag tells pip to use the current environment (where torch is already installed).

```bash
# Load CUDA module (if on SLURM cluster)
module load cuda/12.8.1   # or whichever version is available

# simple-knn: spatial KNN for Gaussian initialization (CUDA, needs torch)
pip install --no-build-isolation submodules/simple-knn

# clm_kernels: custom CUDA kernels for CLM offloading (CUDA, needs torch)
pip install --no-build-isolation submodules/clm_kernels

# cpu-adam: CPU Adam optimizer for offloaded parameters (C++, needs torch)
pip install --no-build-isolation submodules/cpu-adam

# gsplat: Gaussian splatting rasterizer (CUDA, needs torch)
pip install --no-build-isolation submodules/gsplat

# fast-tsp: TSP solver for camera ordering (C++ with pybind11, does NOT need torch)
# Regular install is fine here — it uses scikit-build, not torch extensions
pip install submodules/fast-tsp
```

## 6. Dataset Preparation (Rubble)

Download the Rubble dataset (COLMAP format):

```bash
mkdir -p data
# Download rubble-colmap from the project's data source
# Place it at: data/rubble-colmap/
```

The dataset directory should look like:
```
data/rubble-colmap/
├── images/          # Original images
├── sparse/0/        # COLMAP sparse reconstruction
│   ├── cameras.bin
│   ├── images.bin
│   └── points3D.bin
└── decoded_images/  # Auto-generated on first run
```

## 7. Running Training

### Single GPU (master baseline)

```bash
# Via SLURM (self-submits):
bash single_gpu.sh nvidia_a100    # A100 PCIe (cheetah01)
bash single_gpu.sh a100           # A100 SXM  (cheetah04)
bash single_gpu.sh a40            # A40       (jaguar nodes)

# Or directly on a GPU node:
python train.py \
    -s data/rubble-colmap \
    --clm_offload \
    --bsz 8 \
    --eval \
    -m output/rubble_a100_single_clm
```

### Multi-GPU (2 GPUs) — Unified Script

The `multi_gpu.sh` script handles all three modes (baseline, P2P, overlap).
It auto-detects GPUs, picks a dynamic port, and self-submits to SLURM.

```bash
# Baseline (no optimizations):
bash multi_gpu.sh nvidia_a100 baseline   # A100 PCIe (cheetah01)
bash multi_gpu.sh a100 baseline          # A100 SXM  (cheetah04)

# P2P GPU-to-GPU SH sharing (best on NVLink systems):
bash multi_gpu.sh a100 p2p               # A100 SXM  (cheetah04)

# Overlapped scheduling (dual-stream prefetch/offload):
bash multi_gpu.sh a100 overlap           # A100 SXM  (cheetah04)
bash multi_gpu.sh nvidia_a100 overlap    # A100 PCIe (cheetah01)
```

Output directories are auto-named: `output/rubble_<gpu>_<mode>`
(e.g., `rubble_a100_multi_clm`, `rubble_a100_multi_p2p`, `rubble_a100_multi_overlap`).

## 8. SLURM Job Scripts

### Install submodules (one-time, needs GPU node)

```bash
#!/bin/bash
#SBATCH --output=./slurm/slurm%j.out
#SBATCH --error=./slurm/slurm%j.err
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --partition=gpu
#SBATCH --time=1:00:00

set -e
source ~/.bashrc
conda activate clm_gs
module load cuda/12.8.1

cd /path/to/CLM-GS

pip install --no-build-isolation submodules/simple-knn
pip install --no-build-isolation submodules/clm_kernels
pip install --no-build-isolation submodules/cpu-adam
pip install --no-build-isolation submodules/gsplat
pip install submodules/fast-tsp
```

### Multi-GPU training

The unified `multi_gpu.sh` script self-submits to SLURM — no manual sbatch needed:

```bash
# Baseline multi-GPU:
bash multi_gpu.sh a100 baseline

# P2P (NVLink recommended):
bash multi_gpu.sh a100 p2p

# Overlapped scheduling:
bash multi_gpu.sh a100 overlap
```

The script handles environment setup, dynamic port selection, and GPU detection.
See `multi_gpu.sh` for the full SLURM configuration.

## Troubleshooting

- **`ModuleNotFoundError: No module named 'torch'` during submodule install**: You forgot `--no-build-isolation`. The setup.py imports torch at build time.
- **NCCL timeout**: Increase `NCCL_TIMEOUT` or check that GPUs can communicate (NVLink for P2P, PCIe otherwise).
- **OOM (system RAM)**: The `prealloc_capacity` auto-scales based on available memory. If `--mem` is too high on SLURM, it over-allocates. Use `--mem=200G` for 2 GPUs or set `--prealloc_capacity` manually (e.g., `40000000`).
