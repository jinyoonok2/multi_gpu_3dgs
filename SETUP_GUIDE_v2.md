# multi_gpu_3dgs: 3D Gaussian Splatting — Setup & SLURM Guide (v2)

This guide details the complete workflow for setting up the multi_gpu_3dgs environment, applying required bug fixes, compiling custom CUDA kernels, and running distributed training jobs using SLURM.

> This repo is a fork of [nyu-systems/CLM-GS](https://github.com/nyu-systems/CLM-GS) with minor modifications. The setup process is nearly identical but requires two additional bug fixes before training will succeed (see Part 1, Step 3).

---

## Part 0: Dataset Setup (Run Once)

### Why `/bigtemp`?

Your personal home directory (`/u/xna8aw`) has a limited quota. The `rubble-colmap` dataset alone is ~16GB compressed, and once unzipped plus the predecoded image cache generated during training, it grows to **~82GB total**. This will completely fill your home directory and break VS Code and other tools.

The `/bigtemp` scratch drive is a large shared storage space with no personal quota. **Always store large datasets there.** The symlink trick below makes the training scripts still find the data at the expected relative path without any code changes.

### Step 1: Create Your Scratch Directory

```bash
mkdir -p /bigtemp/xna8aw
cd /bigtemp/xna8aw
```

### Step 2: Download the Dataset

The dataset is hosted on HuggingFace. Use the Python API (with `clm_gs` activated) to download it:

```bash
module purge
module load miniforge
conda activate clm_gs

pip install huggingface_hub

python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='HexuZhao/mega_nerf_rubble_colmap',
    repo_type='dataset',
    local_dir='/bigtemp/xna8aw/rubble-colmap'
)
"
```

This downloads ~16GB and places a `rubble-colmap.zip` inside `/bigtemp/xna8aw/rubble-colmap/`.

### Step 3: Unzip the Dataset

```bash
cd /bigtemp/xna8aw/rubble-colmap
unzip rubble-colmap.zip
```

This extracts to a `rubble-colmap/` subdirectory, so the final dataset path is:
`/bigtemp/xna8aw/rubble-colmap/rubble-colmap/`

### Step 4: Symlink into the Project

```bash
# Create the data directory in the project
mkdir -p /u/xna8aw/6501_GPU_architectures/multi_gpu_3dgs/data

# Create the symlink — train.py finds data at data/rubble-colmap
# but the actual 82GB lives safely on /bigtemp
ln -s /bigtemp/xna8aw/rubble-colmap/rubble-colmap \
      /u/xna8aw/6501_GPU_architectures/multi_gpu_3dgs/data/rubble-colmap

# Verify it works
ls /u/xna8aw/6501_GPU_architectures/multi_gpu_3dgs/data/rubble-colmap
```

> The predecoded image cache (`data/rubble-colmap/decoded_images`) is also written through the symlink during training, so it lands on `/bigtemp` automatically — keeping your home directory quota safe.

---

## Part 1: Initial Environment Creation (Run Once)

Run these commands sequentially from the **portal login node** to pull the repository, create the Conda environment, and compile the necessary CUDA submodules.

### 1. Repository Setup & Base Environment

```bash
# Clone the repository and submodules
git clone https://github.com/Tahuubinh/multi_gpu_3dgs.git
cd multi_gpu_3dgs
git config --global url."https://github.com/".insteadOf git@github.com:
git submodule update --init --recursive

# Load necessary modules for the portal node
module purge
module load miniforge
module load cuda/12.8.1

# Create and activate the conda environment
conda create -n clm_gs python=3.10 -y
conda activate clm_gs

# Install PyTorch and base dependencies
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install tqdm plyfile psutil numba opencv-python scipy matplotlib pandas imageio imageio-ffmpeg requests tabulate
```

### 2. Custom CUDA Kernel Compilation

Because the portal login node does not have attached GPUs, PyTorch auto-detection will fail. You must **explicitly set the target GPU architectures** before compiling the submodules.

```bash
# Set architectures manually (covers Turing, Ampere, Ada, etc.)
export TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;8.9"

# Clean any old CMake caches (prevents Ninja vs. Unix Makefiles conflicts)
rm -rf ./submodules/fast-tsp/_skbuild

# Install all custom submodules using the host environment (--no-build-isolation)
pip install --no-build-isolation ./submodules/simple-knn
pip install --no-build-isolation ./submodules/gsplat
pip install --no-build-isolation ./submodules/clm_kernels
pip install --no-build-isolation ./submodules/cpu-adam
pip install ./submodules/fast-tsp
```

> **Note — `simple_knn` package structure fix (applied 2026-03-03):**
> The original `submodules/simple-knn/simple_knn/` directory had no `__init__.py`,
> so Python did not recognise it as a package when installed in editable mode (`pip install -e .`).
> A `__init__.py` has been added to `submodules/simple-knn/simple_knn/` that imports `distCUDA2` from
> the compiled `_C` extension. Without this file, training crashes with
> `ModuleNotFoundError: No module named 'simple_knn'` on any node except the one the `.so` was built on.
> This fix is already committed to the repo — no manual action needed.

### 3. Apply Required Bug Fixes

> **This step is mandatory.** Without these two fixes, training will crash. They are not present in this repo.

**Fix 1 — Race condition in DDP directory creation (`scene/__init__.py` line 166):**

When running with 2 GPUs, both processes try to create the same directory simultaneously. The original `os.makedirs()` call fails if the directory already exists.

```bash
# From the multi_gpu_3dgs directory:
sed -i 's/os.makedirs(self.decode_dataset_path)$/os.makedirs(self.decode_dataset_path, exist_ok=True)/' scene/__init__.py

# Verify the fix:
grep "decode_dataset_path" scene/__init__.py
# Should show: os.makedirs(self.decode_dataset_path, exist_ok=True)
```

**Fix 2 — GPU memory fragmentation OOM at initialization (`strategies/clm_offload/gaussian_model.py` line ~61):**

After 37+ minutes of image predecoding, the CUDA allocator is fragmented. The original code then allocates a redundant second copy of the 1.4M-point cloud on GPU for `distCUDA2`, which fails on fragmented memory even though total VRAM is sufficient.

Open `strategies/clm_offload/gaussian_model.py` and find this block:

```python
dist2 = torch.clamp_min(
    distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()),
    0.0000001,
)
```

Replace it with:

```python
torch.cuda.empty_cache()  # defragment GPU memory after predecoding phase
dist2 = torch.clamp_min(
    distCUDA2(fused_point_cloud),  # reuse existing GPU tensor instead of allocating a new copy
    0.0000001,
)
```

---

## Part 2: SLURM Batch Scripts

> **Do not run `train.py` directly on the login node.** Instead, use the provided SLURM batch scripts.
>
> The script automatically handles loading `cuda/12.8.1` and activating the `clm_gs` environment for the compute node.

### Option A: Generic 2-GPU Script (`srun.sh`)

Use this to grab any 2 available GPUs to avoid long queue wait times.

```bash
#!/bin/bash
#SBATCH --output=./slurm/slurm%j.out
#SBATCH --error=./slurm/slurm%j.err
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
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
export CUDA_VISIBLE_DEVICES=0,1
export NCCL_TIMEOUT=1800
export NCCL_DEBUG=WARN
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

torchrun \
    --nproc_per_node=2 \
    --master_addr=localhost \
    --master_port=29500 \
    train.py \
    -s 'data/rubble-colmap' \
    --clm_offload \
    --bsz 8 \
    --prealloc_capacity 20000000 \
    --eval \
    -m output/rubble_2gpu_ddp
```

### Option B: A100-Specific Script (`srun-a100.sh`)

Use this if you strictly require A100 GPUs (may result in `PD` status if nodes are busy).

```bash
# Same script as Option A, but replace the --gres line with:
#SBATCH --gres=gpu:a100:2
```

> **Note on `--prealloc_capacity`:** Set to `20000000` (20M) rather than the default `40000000`. The full 40M allocates ~30GB of pinned memory across 2 processes, which competes with CUDA device allocations and can cause OOM. 20M is sufficient for this scene.

---

## Part 3: Running & Monitoring Jobs

When you log into the portal, navigate to the project directory:

```bash
cd ~/6501_GPU_architectures/multi_gpu_3dgs
```

### Submitting a Job

```bash
sbatch srun.sh
```

### Checking Job Status

```bash
squeue -u xna8aw
```

| Status | Meaning |
|--------|---------|
| `PD`   | Pending — waiting for requested resources |
| `R`    | Running — training has started |

### Viewing Live Logs

```bash
tail -f slurm/slurm<JOB_ID>.out
```

> Press `Ctrl + C` to exit. The job continues running in the background.

### Canceling a Job

```bash
scancel <JOB_ID>
```

### What to Expect on First Run

On the **first run only**, training will predecode all 1,468 training images to disk (~37 minutes). This cache is saved at `data/rubble-colmap/decoded_images/` (on `/bigtemp` via symlink). Every subsequent run skips this step entirely and loads from cache.

---

## Differences from Original CLM-GS

| | CLM-GS (`nyu-systems/CLM-GS`) | This repo (`Tahuubinh/multi_gpu_3dgs`) |
|---|---|---|
| Bug Fix 1 (`exist_ok`) | ✅ Applied | ❌ Missing — apply manually (Part 1, Step 3) |
| Bug Fix 2 (`empty_cache`) | ✅ Applied | ❌ Missing — apply manually (Part 1, Step 3) |
| SLURM scripts | ✅ Included | ✅ Added (`srun.sh`, `srun-a100.sh`) |
| Debug test iterations | Not present | Commented-out line in `arguments/__init__.py` |
