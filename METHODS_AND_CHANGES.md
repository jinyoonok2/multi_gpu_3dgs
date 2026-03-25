# Multi-GPU CLM-GS — Methods & Changes from Master

This document describes the three communication optimization methods implemented on the `p2p_overlap` branch and the code changes made relative to the original `master` branch.

---

## Part 1: Methods

### 1.1 Baseline: Multi-GPU CLM Offloading

CLM-GS uses **camera parallelism** (data parallelism): each GPU holds a complete copy of all Gaussians but renders different camera views. After computing gradients from its own cameras, gradients are averaged across GPUs via `all_reduce` (DDP).

CLM keeps only 4 small parameter tensors on GPU (xyz, opacity, scaling, rotation) and offloads the large **SH coefficients** (48 floats x N Gaussians) to CPU pinned memory. Each iteration processes a batch of camera views as micro-batches, selectively loading only the SH coefficients visible to each camera.

**Per-iteration pipeline:**
1. Filter & Order — determine visible Gaussians per camera; sort cameras via TSP
2. CPU Thread Init — start background CPU Adam thread for SH parameters
3. State Init — zero accumulators, create `comm_stream` for CPU-GPU transfers
4. Micro-batch Loop — for each camera: load SH (CPU->GPU), prefetch next SH, forward, backward, scatter gradients, offload SH gradients (GPU->CPU)
5. Post-training — blocking all-reduce across GPUs, GPU Adam step, join CPU Adam

**Key bottleneck:** SH prefetch (CPU->GPU) and gradient offload (GPU->CPU) share a single `comm_stream`, serializing CPU-GPU transfers.

### 1.2 P2P GPU-to-GPU SH Sharing (`--p2p_fetch`)

**Problem:** In multi-GPU training, GPUs render different cameras but many Gaussians are visible to multiple cameras. Each GPU independently loads overlapping SH from CPU via PCIe (~25 GB/s), causing redundant PCIe traffic.

**Solution:** Exchange visibility filters across GPUs via `all_reduce(SUM)`, then classify Gaussians as:
- **Overlap:** visible on 2+ GPUs -> loaded once by GPU 0, broadcast via NVLink
- **Local-only:** visible on this GPU only -> loaded from CPU normally
- **Peer-only:** visible only on the other GPU -> not loaded

This replaces 2x redundant PCIe loads with 1x PCIe load + 1x NVLink broadcast (600 GB/s on A100 SXM, effectively free).

**Hardware requirement:** NVLink mandatory. On PCIe-only GPUs, GPU-to-GPU transfers share the same PCIe bus, offering no benefit.

### 1.3 Async All-Reduce (`--async_allreduce`)

**Problem:** Blocking all-reduce at the end of each iteration forces all GPUs to idle while gradients synchronize.

**Solution:** Replace blocking all-reduce with `async_op=True`:
- After forward/backward, initiate non-blocking all-reduce
- Immediately begin the next iteration's forward pass
- Wait for completion only before gradients are needed (optimizer step)

This overlaps gradient communication with the next iteration's computation.

### 1.4 Dual-Stream Overlap (`--overlap_schedule`)

**Problem:** A single `comm_stream` serializes SH prefetch (CPU->GPU) and gradient offload (GPU->CPU).

**Solution:** Create a dedicated `offload_stream` for gradient offload, enabling three-way overlap:

```
comm_stream:     [prefetch SH(i+1)]
default_stream:  [fwd(i)] [bwd(i)]
offload_stream:           [offload grad(i-1)]
```

PCIe is full-duplex, so CPU->GPU and GPU->CPU transfers run simultaneously at full bandwidth in each direction.

Three CUDA events coordinate the streams:
- `cpu2gpu_event`: comm_stream -> default_stream (SH loaded, safe to compute)
- `gpu2cpu_event`: default_stream -> offload_stream (backward done, safe to offload)
- `indices_ready_event`: comm_stream -> offload_stream (G indices computed)

### 1.5 All Combined (`--p2p_fetch --overlap_schedule --async_allreduce`)

Applies all three optimizations simultaneously. P2P reduces how much PCIe traffic exists, Overlap hides the latency of remaining traffic, and Async overlaps gradient sync with computation.

### Method Comparison

| Aspect | P2P | Async All-Reduce | Overlap |
|--------|-----|-------------------|---------|
| **What it reduces** | Redundant PCIe loads | Gradient sync idle time | Transfer pipeline stalls |
| **Mechanism** | GPU-GPU cooperative loading | Non-blocking all-reduce | Dual CUDA streams |
| **Hardware requirement** | NVLink mandatory | Any | Any |
| **Applies to** | First micro-batch (shared SH) | End of iteration | Every micro-batch |

---

## Part 2: Code Changes from Master

All changes relative to the original `master` branch.

### Architecture

A **modular strategy pattern** replaces per-method engine copies with a single unified engine and pluggable strategy modules. The engine calls 7 hook points; each strategy overrides only the hooks it needs.

| Mode | Flags | Strategy class | Hooks overridden |
|------|-------|----------------|------------------|
| Baseline | *(none)* | `BaseStrategy` (no-ops) | 0 |
| P2P | `--p2p_fetch` | `P2PStrategy` | 3 |
| Overlap | `--overlap_schedule` | `OverlapStrategy` | 3 |
| Async | `--async_allreduce` | via engine flag | 1 (all-reduce) |
| All | all three flags | Combined | all |

### New Files

| File | Purpose |
|------|---------|
| `strategies/clm_offload/strategy_base.py` | `BaseStrategy` — 7 pipeline hooks with no-op defaults |
| `strategies/clm_offload/p2p_module.py` | `P2PStrategy` — cooperative GPU-GPU SH loading |
| `strategies/clm_offload/overlap_module.py` | `OverlapStrategy` — dedicated CUDA stream for gradient offload |
| `strategies/clm_offload/p2p_comm.py` | `P2PCommManager` — filter partition, SH sharing, gradient sync |

### Modified Files

| File | Change |
|------|--------|
| `strategies/clm_offload/engine_multi.py` | Added `strategy` parameter and 7 hook call sites |
| `train_multi.py` | Creates strategy object, passes to engine; weak scaling (iteration scaling by `1/world_size`) |
| `arguments/__init__.py` | Added `--p2p_fetch`, `--overlap_schedule`, `--async_allreduce` flags |
| `multi_gpu.sh` | Unified SLURM script with `MODE` argument (baseline/p2p/overlap/async/all) |
| `single_gpu.sh` | Fixed OOM with direct `--prealloc_capacity` computation |

### Hook Points in engine_multi.py

| # | Hook | Pipeline Stage | Purpose |
|---|------|----------------|---------|
| 1 | `post_filters` | After filter + ordering | P2P computes overlap partition |
| 2 | `get_offload_stream` | Stream creation | Overlap creates dedicated stream |
| 3 | `load_first_shs` | First micro-batch SH load | P2P does cooperative GPU-GPU loading |
| 4 | `pre_compute_offload` | After H/D indices | Overlap pre-computes Category G indices |
| 5 | `before_forward` | Before forward pass | Available hook (currently no-op) |
| 6 | `pre_gradient_sync` | Before all-reduce | Overlap does full device synchronize |
| 7 | `sync_gradients` | Gradient sync | P2P uses P2PCommManager; Async uses non-blocking |

### Launch

```bash
bash multi_gpu.sh <partition> [baseline|p2p|overlap|async|all]
bash single_gpu.sh <partition>
```
