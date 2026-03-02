# Changelog

All notable changes to this codebase are recorded here.
Format: one entry per development session, newest at the top.

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
