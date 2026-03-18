# Multi-GPU CLM Optimization — All Changes

Consolidated record of every file added or modified across both methods.
The baseline is `engine_multi.py` (original multi-GPU CLM engine, **never modified**).

---

## At a Glance

| | Method 1 — P2P SH Sharing | Method 2 — Overlapped Scheduling |
|---|---|---|
| **Flag** | `--p2p_fetch` | `--overlap_schedule` |
| **Core idea** | GPU-to-GPU SH sharing via NVLink/NCCL to avoid duplicate PCIe loads | Dual CUDA streams to overlap prefetch, compute, and grad offload |
| **New engine** | `engine_multi_p2p.py` | `engine_multi_overlap.py` |
| **Launch script** | `multi_gpu_p2p.sh` | `multi_gpu_overlap.sh` |
| **Best on** | NVLink systems (A100 SXM, H100 NVL) | Any multi-GPU (PCIe or NVLink) |

---

## New Files

| File | Method | Purpose |
|------|--------|---------|
| `strategies/clm_offload/p2p_comm.py` | 1 | `P2PCommManager`: filter partition, cooperative SH loading, gradient sync |
| `strategies/clm_offload/engine_multi_p2p.py` | 1 | P2P-modified training engine |
| `multi_gpu_p2p.sh` | 1 | SLURM launch script (`--p2p_fetch`) |
| `strategies/clm_offload/engine_multi_overlap.py` | 2 | Dual-stream training engine |
| `multi_gpu_overlap.sh` | 2 | SLURM launch script (`--overlap_schedule`) |
| `CHANGES.md` | — | This file |

## Modified Files

| File | Method | Change |
|------|--------|--------|
| `arguments/__init__.py` | 1 | Added `self.p2p_fetch = False` (line 172) |
| `arguments/__init__.py` | 2 | Added `self.overlap_schedule = False` (line 173) |
| `train_multi.py` | 1+2 | Imports both engines; dispatches via `args.p2p_fetch` / `args.overlap_schedule` |

## Unchanged (read-only reference)

- `strategies/clm_offload/engine_multi.py` — baseline engine (single `comm_stream`)

---

## Method 1 — P2P GPU-to-GPU SH Sharing

### Problem
Both GPUs independently load overlapping SH coefficients from CPU via PCIe
(~25 GB/s each), doubling PCIe traffic for shared Gaussians.

### Solution
Exchange visibility filters, then GPU 0 loads shared SH once and broadcasts
to GPU 1 via NVLink (600–900 GB/s).

### Pipeline changes

| Stage | What changed |
|-------|-------------|
| **1 — Filters** | Exchange filters across GPUs via `all_reduce`; classify Gaussians as *overlap / local_only / peer_only* |
| **4.1 — First SH load** | `P2PCommManager.share_shs_p2p()`: GPU 0 loads overlap from CPU, broadcasts via NCCL; each GPU loads local-only independently |
| **4.2+ — Retention** | Unchanged (already GPU-local optimization) |
| **5.0 — Grad sync** | `P2PCommManager.sync_gradients_p2p()` (same semantics, encapsulated) |

### Expected benefit
- **NVLink (A100, H100)**: Significant PCIe savings — overlap Gaussians loaded 1× instead of 2×.
- **PCIe-only (A40)**: Minimal benefit — GPU↔GPU P2P shares the same bus.

### How to run
```bash
bash multi_gpu_p2p.sh a100
```

---

## Method 2 — Dual-Stream Overlapped Scheduling

### Problem
A single `comm_stream` handles both SH prefetch (CPU→GPU) and gradient
offload (GPU→CPU), serializing them.

### Solution
Split into two CUDA streams so prefetch, compute, and offload all run
concurrently:

```
comm_stream:     [prefetch SH(i+1)]
default_stream:  [fwd(i)]  [bwd(i)]
offload_stream:              [offload grad(i-1)]
```

### Pipeline changes

| Stage | What changed |
|-------|-------------|
| **3 — Init** | Create dedicated `offload_stream` |
| **4.2 — Prefetch** | Pre-compute Category G indices for later offload; record `indices_ready_event` |
| **4.6 — Grad offload** | Runs on `offload_stream` (waits `gpu2cpu_event` + `indices_ready_event`) instead of `comm_stream` |
| **5 — Post-training** | `torch.cuda.synchronize()` drains both streams before all-reduce |

### Expected benefit
- Reduced iteration time when prefetch transfer is large relative to backward.
- Greatest on PCIe-bottlenecked configs (many Gaussians, constrained bandwidth).
- Still helps on NVLink, though absolute gain may be smaller.

### How to run
```bash
bash multi_gpu_overlap.sh a100
```
