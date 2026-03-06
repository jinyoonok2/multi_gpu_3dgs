# Changelog

All notable changes to this codebase are recorded here.
Format: one entry per development session, newest at the top.

---

## [2026-03-06] — Redesign: Camera Parallelism (M1 → M3 → M2)

**Status:** PLANNED — Implementation starts today

---

### Why the Redesign

The previous multi-GPU approach (spatial partitioning) was fundamentally flawed:

1. **CLM already solves the memory problem.** SH features (77% of per-Gaussian
   storage) live on CPU pinned memory. Spatial params are only ~235 MB for 1.4M
   Gaussians — trivially fits on one GPU. Splitting them across GPUs was pointless.

2. **Spatial partitioning breaks rendering correctness.** Gaussian splatting uses
   alpha blending in depth-sorted order. If each GPU renders only its partition's
   Gaussians, the partial images cannot be simply summed — the global depth order
   is lost. Each GPU's loss(partial_image, full_ground_truth) produces wrong gradients.

3. **The real bottleneck is throughput, not capacity.** CLM processes cameras
   sequentially: visibility → SH fetch → render → backward, one camera at a time.
   Multi-GPU should parallelize this across cameras.

---

### New Architecture: Camera Parallelism with CLM

**Core Principle:** Both GPUs hold ALL Gaussians. The camera batch is split across
GPUs. Each GPU renders mathematically correct, full-scene images for its cameras.

```
Memory Layout (BOTH GPUs identical):
  CPU Pinned (shared):
    SH features          — ALL N Gaussians × 48 floats
    SH gradient buffer   — ALL N Gaussians × 48 floats
    CPU Adam states      — for SH

  GPU 0 and GPU 1 (replicated):
    xyz, opacity, scaling, rotation  — ALL N Gaussians
    GPU Adam states                  — for spatial params

Training Loop (batch of 8 cameras):
  GPU 0: cameras [0,1,2,3]  →  identify visible → fetch SH → render → backward
  GPU 1: cameras [4,5,6,7]  →  identify visible → fetch SH → render → backward
  AllReduce: spatial gradients (xyz, opacity, scaling, rotation)
  CPU Adam: updates SH from both GPUs' gradient contributions
```

**Result:** Identical quality to single-GPU CLM. ~2× speedup from camera parallelism.

---

### Three Implementation Stages (Ablation Study)

#### Stage 1: M1 — Basic Camera Parallelism
- Camera batch divided across GPUs
- Each GPU independently fetches its visible SH from CPU over PCIe
- AllReduce spatial gradients at end of batch
- **Measures:** Baseline multi-GPU speedup, PCIe bandwidth utilization

#### Stage 2: M1 + M3 — P2P Collaborative SH Caching
- Before CPU fetch, GPUs intersect their visible Gaussian indices
- Overlapping Gaussians: one GPU fetches from CPU, P2P sends to the other
- Reduces redundant PCIe traffic when cameras see overlapping 3D regions
- **Measures:** PCIe bandwidth savings, P2P NVLink/PCIe utilization

#### Stage 3: M1 + M3 + M2 — Overlapped Stream Scheduling
- Dual CUDA stream pipeline: compute stream + communication stream
- While compute stream renders camera t, comm stream prefetches SH for camera t+1
- Comm stream also handles P2P exchange for the next camera's overlapping data
- **Measures:** GPU compute utilization, overlap efficiency, end-to-end speedup

---

### What Changes from Current Code

| Component | Current (spatial partition) | New (camera parallelism) |
|---|---|---|
| Gaussian model | Partitioned spatial params | Replicated spatial params |
| SH buffer | Each GPU has its partition | Both GPUs share full CPU buffer |
| Visibility | Project local only | Project ALL Gaussians |
| Rendering | Partial scene per GPU | Full scene per GPU |
| SH fetch | Each GPU fetches own subset | Each GPU fetches for its cameras |
| GPU↔GPU comm | P2P SH exchange + grad exchange | AllReduce spatial grads + P2P SH cache (M3) |
| Correctness | Broken (no global depth sort) | Correct (identical to single-GPU) |

---

### Files to Modify/Create

- `strategies/multi_gpu_clm/gaussian_model.py` — Remove spatial partitioning,
  replicate all params on each GPU, shared CPU pinned SH buffer
- `strategies/multi_gpu_clm/engine.py` — Rewrite for camera parallelism:
  M1 (basic), M3 (P2P cache), M2 (stream overlap)
- `train_multi_gpu_clm.py` — Update training loop for camera-split dispatch
- `srun-mgclm-a40.sh`, `srun-mgclm-a100.sh` — Update as needed

### Evaluation Plan

- Compare PSNR and training time across: CLM baseline (1 GPU), M1, M1+M3, M1+M3+M2
- CLM baseline PSNR: 24.997 on A100 (job 5567867)
- Dataset: rubble-colmap (1,399,033 Gaussians, 1,678 cameras, 4591×3436)

---

## [2026-03-04 to 03-05] — Spatial Partitioning Attempt (Deprecated)

**Commits:** `5882d97` → `654341a`

Built `strategies/multi_gpu_clm/` with spatial partitioning + P2P SH exchange.
Fixed 6 bugs (device mismatch, NCCL deadlock, etc.). Hit fundamental OOM from
rendering all Gaussians per GPU. Switched to local-only rendering which fixed OOM
but produced incorrect gradients (partial scene per GPU). Realized the entire
spatial partitioning approach was unnecessary given CLM's memory design.

**Conclusion:** Spatial partitioning is the wrong parallelism axis for CLM.
Camera parallelism is the correct approach. This code will be replaced.

---

## [2026-03-02] — Multi-GPU Strategy (M3 + M1M3)

**Author:** Jinyoon (jinyoonok@gmail.com)
**Commit:** `25903c0`
**Base:** Original CLM-GS repo (commit `08d1b8d`)

---

### Overview

Added a fourth training mode — `--multi_gpu` — that distributes the Gaussian point cloud across multiple GPUs using spatial partitioning. Two sub-strategies are implemented: M3 (AllGather) and M1M3 (P2P caching). All existing single-GPU strategies (CLM offload, naive offload, no offload) are unchanged.

---

### Modified Files (from original CLM-GS)

#### `arguments/__init__.py`
- Added `self.multi_gpu = False` flag (CLI: `--multi_gpu`)
- Added `self.enable_p2p_caching = False` flag (CLI: `--enable_p2p_caching`, selects M1M3 over M3)
- Extended the mutual-exclusion assert to include `multi_gpu` alongside the three existing offload modes
- Removed `enable_overlap_scheduling` (M2 overlap, was experimental and dropped)

#### `strategies/__init__.py`
- Added import and export of `GaussianModelMultiGPU`, `multi_gpu_train_one_batch`, `multi_gpu_eval_one_cam` from `strategies.multi_gpu`

#### `strategies/clm_offload/gaussian_model.py`
- Minor fix: corrected device string for Quadro RTX 6000 GPU type detection (was causing incorrect GPU assignment on lotus nodes)

#### `scene/__init__.py`
- Added `world_size` / `rank`-aware camera partitioning: each rank receives only its assigned subset of training cameras (modulo partition by rank)

#### `train.py`
- Added `dist.init_process_group(backend='nccl')` initialization block for `--multi_gpu` mode
- Added `LOCAL_RANK` environment variable handling (set by `torchrun`) to assign correct GPU to each process
- Added dispatch branches for `--multi_gpu` in the training loop and eval loop
- Imports: `GaussianModelMultiGPU`, `multi_gpu_train_one_batch`, `multi_gpu_eval_one_cam`

#### `.gitignore`
- Activated `*.sh` rule (was commented out `# *.sh`) — shell scripts are now gitignored
- `output/` and `slurm/` were already ignored

---

### New Directory: `strategies/multi_gpu/`

All multi-GPU logic lives here. Single-GPU strategies are untouched.

---

#### `strategies/multi_gpu/__init__.py`
Public API — re-exports the three symbols used by `train.py`:
```
GaussianModelMultiGPU
multi_gpu_train_one_batch
multi_gpu_eval_one_cam
```

---

#### `strategies/multi_gpu/gaussian_model.py`
`GaussianModelMultiGPU` — the distributed Gaussian model.

Key responsibilities:
- Owns a **local partition** of the full Gaussian point cloud (determined by spatial TSP partitioning at init)
- Maintains `global_xyz`, `global_opacity`, `global_scaling_raw`, `global_rotation_raw` as **non-trainable** AllGathered tensors (updated each iteration)
- Exposes `rank`, `world_size`, `n_local`, `n_global` properties
- Densification and pruning operate **only on the local partition** then re-sync global tensors
- `_parameters` tensor (SH features, `[N_local, 48]`) is the only trainable, rank-local parameter not shared globally

---

#### `strategies/multi_gpu/base_strategy.py`
`BaseMultiGPUStrategy(ABC)` — abstract base shared by both M3 and M1M3.

Contains all logic that is identical between strategies:
- `calculate_filters_global()` — visibility culling using the global Gaussian proxy
- `render_one_camera()` — gsplat projection + rasterization
- `assemble_local_features()` — gather features for locally-owned visible Gaussians
- `train_one_batch()` — full training loop: visibility → assemble → render → loss → backward → optimizer step
- `eval_one_cam()` — always uses AllGather (full scene, no P2P) for clean evaluation metrics

Subclasses implement exactly **one** abstract method:
```python
def assemble_features(self, gaussians, local_idx, remote_global_idx,
                      remote_rank, remote_local_idx,
                      global_scaling, global_rotation):
    ...  # returns (fxyz, fopa, fscl, frot, fshs, n_local)
```

---

#### `strategies/multi_gpu/m3_strategy.py`
`M3Strategy(BaseMultiGPUStrategy)` — **Method 3: AllGather**.

For each camera batch, gathers SH features from all ranks via `dist.all_gather`, then assembles the full visible Gaussian set locally. Simple and robust — no P2P, no deadlock risk. Higher bandwidth than M1M3 because all SH features are transferred even if not needed.

---

#### `strategies/multi_gpu/m1m3_strategy.py`
`M1M3Strategy(BaseMultiGPUStrategy)` — **Method 1 + Method 3: P2P caching**.

Only fetches SH features actually needed for visible remote Gaussians, using symmetric NCCL `isend`/`irecv` per peer rank. Reduces bandwidth vs M3 when each rank only sees a small fraction of remote Gaussians.

**3-phase symmetric protocol per peer:**
1. Exchange counts — `isend(my_count) + irecv(peer_count)`
2. Exchange index lists — send my needed indices, receive peer's needed indices
3. Exchange features — gather local SH for peer, send; receive my SH

**Bug fixed (2026-03-02):** Previous version used `for rk in torch.unique(remote_rank)` which skipped peers when a GPU needed 0 Gaussians from them. This caused an NCCL deadlock (`WorkNCCL SeqNum=2 RECV timeout`) because the peer was waiting in `irecv` with no matching `isend`. Fixed by looping over **all** peer ranks unconditionally, passing empty index tensors when nothing is needed.

**Gradient exchange:** After `backward()`, gradients for remote Gaussians are sent back to their owning rank via the same symmetric protocol.

---

#### `strategies/multi_gpu/engine.py`
Dispatch shim — the only file `train.py` interacts with.

```
--enable_p2p_caching  →  M1M3Strategy
(default)             →  M3Strategy
```

`multi_gpu_eval_one_cam` always uses `M3Strategy` regardless of training mode (AllGather gives deterministic metrics).

To add a new strategy later: create a new `*_strategy.py`, import it here, add a branch in `_get_strategy()`.

---

### Training Modes Summary

| Flag | Strategy | GPUs | Description |
|---|---|---|---|
| `--no_offload` | `GaussianModelNoOffload` | 1 | All Gaussians on GPU, no offload |
| `--clm_offload` | `GaussianModelCLMOffload` | 1 | CLM CPU offload |
| `--naive_offload` | `GaussianModelNaiveOffload` | 1 | Naive CPU offload |
| `--multi_gpu` | `GaussianModelMultiGPU` + M3Strategy | 2+ | Spatial partitioning + AllGather |
| `--multi_gpu --enable_p2p_caching` | `GaussianModelMultiGPU` + M1M3Strategy | 2+ | Spatial partitioning + P2P |

---

### Known Issues / Future Work

- [ ] M1M3 gradient exchange not yet verified end-to-end (forward deadlock was just fixed)
- [ ] Densification sync across ranks uses AllGather of full point cloud — potential bottleneck at high Gaussian counts
- [ ] `eval_one_cam` AllGathers full SH for all ranks — could be optimized for large scenes
- [ ] Only 2-GPU tested; >2 GPU correctness unverified

---
