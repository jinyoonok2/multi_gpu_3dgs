# p2p_overlap Branch — Changes from Master

All changes relative to the `master` branch (`3f8781e`).

---

## Architecture

A **modular strategy pattern** replaces per-method engine copies with a single
unified engine and pluggable strategy modules. The engine calls 7 hook points;
each strategy overrides only the hooks it needs.

| Mode | Flag | Strategy class | Hooks overridden |
|------|------|----------------|------------------|
| Baseline | *(none)* | `BaseStrategy` (no-ops) | 0 |
| P2P SH Sharing | `--p2p_fetch` | `P2PStrategy` | 3 (`post_filters`, `load_first_shs`, `sync_gradients`) |
| Overlapped Scheduling | `--overlap_schedule` | `OverlapStrategy` | 3 (`get_offload_stream`, `pre_compute_offload`, `pre_gradient_sync`) |

**Launch**: `bash multi_gpu.sh <partition> [baseline|p2p|overlap]`

---

## New Files (6)

| File | Lines | Purpose |
|------|-------|---------|
| `strategies/clm_offload/strategy_base.py` | 123 | `BaseStrategy` — 7 pipeline hooks with no-op defaults |
| `strategies/clm_offload/p2p_module.py` | 80 | `P2PStrategy` — cooperative GPU-GPU SH loading via NVLink/NCCL |
| `strategies/clm_offload/overlap_module.py` | 55 | `OverlapStrategy` — dedicated CUDA stream for gradient offload |
| `strategies/clm_offload/p2p_comm.py` | 250 | `P2PCommManager` — filter partition, SH sharing, gradient sync |
| `CHANGES.md` | — | This file |
| `SETUP_GUIDE.md` | 185 | Environment setup and run instructions |
| `SUMMARY_METHODS.md` | 383 | Detailed method descriptions |

## Modified Files (4)

| File | Change |
|------|--------|
| `strategies/clm_offload/engine_multi.py` | +151/−114 — added `strategy` parameter and 7 hook call sites |
| `train_multi.py` | +33 — creates strategy object, passes to unified engine |
| `arguments/__init__.py` | +2 — added `p2p_fetch` and `overlap_schedule` flags |
| `multi_gpu.sh` | +86/−15 — unified script with `MODE` argument (baseline/p2p/overlap) |

---

## Hook Points in engine_multi.py

| # | Hook | Stage | Purpose |
|---|------|-------|---------|
| 1 | `post_filters` | 1 | After filter + ordering — P2P computes overlap partition |
| 2 | `get_offload_stream` | 3 | Return CUDA stream for grad offload — Overlap creates dedicated stream |
| 3 | `load_first_shs` | 4.1 | Load SH for micro_idx=0 — P2P does cooperative GPU-GPU loading |
| 4 | `pre_compute_offload` | 4.2 | After H/D indices in prefetch — Overlap pre-computes Category G indices |
| 5 | `before_forward` | 4.3 | Before forward pass — available hook (currently no-op for all strategies) |
| 6 | `pre_gradient_sync` | 5.0 | Before all-reduce — Overlap does full device synchronize |
| 7 | `sync_gradients` | 5.0 | Perform gradient sync — P2P uses P2PCommManager |

---

## P2P SH Sharing (`--p2p_fetch`)

**Problem**: Both GPUs independently load overlapping SH coefficients from CPU
via PCIe (~25 GB/s each), doubling PCIe traffic for shared Gaussians.

**Solution**: Exchange visibility filters across GPUs, classify Gaussians as
overlap / local-only / peer-only. GPU 0 loads shared SHs once and broadcasts
to GPU 1 via NVLink (600–900 GB/s).

**Best on**: NVLink systems (A100 SXM, H100 NVL). Minimal benefit on PCIe-only.

---

## Overlapped Scheduling (`--overlap_schedule`)

**Problem**: A single `comm_stream` handles both SH prefetch (CPU→GPU) and
gradient offload (GPU→CPU), serializing them.

**Solution**: Dedicated `offload_stream` enables three-way overlap:

```
comm_stream:     [prefetch SH(i+1)]
default_stream:  [fwd(i)]  [bwd(i)]
offload_stream:              [offload grad(i-1)]
```

**Best on**: Any multi-GPU setup; greatest benefit on PCIe-bottlenecked configs.
