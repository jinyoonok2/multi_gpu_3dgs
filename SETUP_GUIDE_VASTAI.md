# CLM-GS Setup Guide — Vast.ai

This guide is specific to Vast.ai instances. No SLURM, no `module load`, no conda environment creation — the instance already provides a working Python environment.

---

## 1. Pick a Vast.ai Instance

When renting on Vast.ai, choose:
- **CUDA version**: 12.4+ (tested on CUDA 13.0 — see note below)
- **GPU**: A100 40GB recommended; 2x for multi-GPU runs
- **RAM**: ≥ 100 GB for single GPU, ≥ 200 GB for 2 GPUs
- **Disk**: ≥ 60 GB (code ~1 GB + dataset ~16 GB + decoded images ~30 GB)
- **Template**: Any PyTorch image (e.g. `pytorch/pytorch:2.x-cuda12.x-cudnn9-devel`)

> **CUDA 13.0 note**: PyTorch wheels are only published up to CUDA 12.4 as of March 2026. CUDA 13.0 is backward-compatible at runtime, but PyTorch's extension compiler raises a version-mismatch error by default. The fix is documented in Step 4.

---

## 2. Clone the Repository

```bash
cd /workspace
git clone -b p2p_overlap https://github.com/jinyoonok2/multi_gpu_3dgs.git multi_gpu_3dgs_p2p_overlap
cd multi_gpu_3dgs_p2p_overlap
```

> The submodules (`simple-knn`, `clm_kernels`, `cpu-adam`, `gsplat`, `fast-tsp`) are committed directly into this repo — no `git submodule update` needed.

---

## 3. Install PyTorch

```bash
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 \
    --index-url https://download.pytorch.org/whl/cu124
```

Verify:
```bash
python3 -c "import torch; print(torch.__version__, torch.version.cuda)"
# Expected: 2.6.0+cu124  12.4
```

---

## 4. Fix CUDA Version Check (CUDA 13.x only)

If `nvcc --version` shows CUDA 13.x, PyTorch's extension builder will refuse to compile submodules due to a major-version mismatch check. Downgrade the check to a warning:

```bash
python3 - <<'EOF'
import torch, pathlib

f = pathlib.Path(torch.__file__).parent / "utils/cpp_extension.py"
text = f.read_text()
old = "            raise RuntimeError(CUDA_MISMATCH_MESSAGE.format(cuda_str_version, torch.version.cuda))"
new = "            warnings.warn(CUDA_MISMATCH_MESSAGE.format(cuda_str_version, torch.version.cuda))"
if old in text:
    f.write_text(text.replace(old, new))
    print("Patched successfully.")
else:
    print("Already patched or not found — check manually.")
EOF
```

Verify it worked:
```bash
grep -n "CUDA_MISMATCH_MESSAGE\|raise RuntimeError" \
    $(python3 -c "import torch, pathlib; print(pathlib.Path(torch.__file__).parent / 'utils/cpp_extension.py')") \
    | grep -A1 "CUDA_MISMATCH_MESSAGE"
# Should show: warnings.warn(CUDA_MISMATCH_MESSAGE...) — NOT raise RuntimeError
```

> **Skip this step entirely** if `nvcc --version` shows CUDA 12.x.

---

## 5. Install Python Dependencies

```bash
pip install tqdm plyfile psutil numba opencv-python scipy matplotlib \
    pandas imageio imageio-ffmpeg requests tabulate
```

---

## 6. Build & Install CUDA/C++ Submodules

All submodule source code is already in the repo under `submodules/`.

**Why `--no-build-isolation`?** Four submodules import `torch` in their `setup.py`. Without this flag, pip builds in an isolated env where torch is missing and the build fails.

```bash
cd /workspace/multi_gpu_3dgs_p2p_overlap

# ~1 min each
pip install --no-build-isolation submodules/simple-knn
pip install --no-build-isolation submodules/clm_kernels
pip install --no-build-isolation submodules/cpu-adam

# ~15–30 min (many CUDA kernels)
pip install --no-build-isolation submodules/gsplat

# C++/pybind11 only, no torch needed
pip install submodules/fast-tsp
```

Verify all installed:
```bash
python3 -c "
import simple_knn, clm_kernels, cpu_adam, gsplat, fast_tsp
print('simple_knn: ok')
print('clm_kernels: ok')
print('cpu_adam: ok')
print('gsplat:', gsplat.__version__)
print('fast_tsp: ok')
"
```

---

## 7. Download the Rubble Dataset

```bash
mkdir -p /workspace/multi_gpu_3dgs_p2p_overlap/data

python3 -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='HexuZhao/mega_nerf_rubble_colmap',
    repo_type='dataset',
    local_dir='/workspace/multi_gpu_3dgs_p2p_overlap/data/rubble-colmap'
)
"
```

Then extract:
```bash
cd /workspace/multi_gpu_3dgs_p2p_overlap/data/rubble-colmap
unzip rubble-colmap.zip
```

The final layout should be:
```
data/rubble-colmap/
├── rubble-colmap.zip          # original download (can delete after extraction)
└── rubble-colmap/             # actual COLMAP dataset used for training
    ├── images/                # ~1400 JPG photos
    └── sparse/0/
        ├── cameras.bin
        ├── images.bin
        ├── points3D.bin
        └── points3D.ply
```

The training scripts use `data/rubble-colmap/rubble-colmap` as the `-s` source path.

---

## 8. Run Training

Ready-to-use scripts are in the repo root. Logs are saved to `logs/` with timestamps. The scripts use `setsid + nohup + disown` so the training process survives terminal disconnection.

### Recommended: Run inside tmux

Vast.ai instances automatically start a `tmux` session called `ssh_tmux`. Always run training inside it so it survives SSH disconnection:

```bash
# If you SSH'd in fresh (not via VS Code):
tmux attach -t ssh_tmux

# Create a new window for training:
# Press: Ctrl+B, then c

cd /workspace/multi_gpu_3dgs_p2p_overlap
bash multi_gpu_vastai.sh p2p

# Detach and let it run while you close your laptop:
# Press: Ctrl+B, then d
```

To reattach later:
```bash
ssh -p <PORT> root@<IP>
tmux attach -t ssh_tmux
# Switch windows: Ctrl+B then w
```

### Single GPU

```bash
cd /workspace/multi_gpu_3dgs_p2p_overlap

bash single_gpu_vastai.sh                  # default: clm_offload
bash single_gpu_vastai.sh naive_offload    # simple offload
bash single_gpu_vastai.sh no_offload       # GPU-only baseline
```

### Multi-GPU

```bash
bash multi_gpu_vastai.sh                   # baseline (all available GPUs)
bash multi_gpu_vastai.sh p2p               # P2P GPU-to-GPU SH sharing
bash multi_gpu_vastai.sh overlap           # dual-stream overlapped schedule
bash multi_gpu_vastai.sh overlap 2         # force exactly 2 GPUs
```

### Monitor / stop a running training

```bash
# Stream live log output:
tail -f logs/multi_p2p_2gpu_<timestamp>.log

# Check if process is alive:
ps -p $(cat logs/train.pid)

# Check GPU usage:
nvidia-smi

# Stop training gracefully:
kill $(cat logs/train.pid)
```

### Manual launch (if you prefer)

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_TIMEOUT=1800
export NCCL_DEBUG=WARN
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export CUDA_VISIBLE_DEVICES=0,1

MASTER_PORT=$(python3 -c "import socket; s=socket.socket(); s.bind(('',0)); print(s.getsockname()[1]); s.close()")

setsid nohup torchrun \
    --nproc_per_node=2 \
    --master_addr=localhost \
    --master_port="${MASTER_PORT}" \
    train_multi.py \
    -s data/rubble-colmap/rubble-colmap \
    --clm_offload \
    --enable_distributed \
    --p2p_fetch \
    --bsz 8 \
    --eval \
    -m output/rubble_multi_p2p \
    > logs/train.log 2>&1 &
disown $!
echo $! > logs/train.pid
```

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `ModuleNotFoundError: No module named 'torch'` during submodule build | You forgot `--no-build-isolation` |
| `RuntimeError: The detected CUDA version (13.0) mismatches...` | Apply the patch in Step 4 |
| `fast-tsp` build fails with `FileNotFoundError: CMakeLists.txt` | A `CMakeLists.txt` was manually created and committed in `submodules/fast-tsp/`; if it's missing, re-clone the branch |
| NCCL timeout during multi-GPU training | Increase `NCCL_TIMEOUT` or verify both GPUs are on NVLink (A100 SXM has NVLink by default) |
| OOM on GPU | Lower `--bsz` (batch size) or switch to `--clm_offload` which keeps fewer params on GPU |
| OOM on CPU RAM | Set `--prealloc_capacity` manually, e.g. `--prealloc_capacity 40000000` for ~40M Gaussians (~320 GB RAM for 100M) |
| `decoded_images/` folder missing | First run auto-generates it — this is normal and takes extra time on the first training run |
