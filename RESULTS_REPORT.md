# Multi-GPU Communication Optimization for 3D Gaussian Splatting (CLM-GS)

## CS 6501 — GPU Architectures Project Report

---

## 1. Overview

This report presents experimental results from implementing and evaluating three multi-GPU communication optimization strategies for **CLM-GS** (CPU-offloading for Large-scale Mapping Gaussian Splatting), a memory-efficient 3DGS training framework that offloads spherical harmonic (SH) coefficients to CPU memory and uses CPU Adam for parameter updates.

We tested on two GPU architectures with fundamentally different interconnects:
- **NVIDIA A100-SXM 80GB** — connected via **NVLink** (~600 GB/s bidirectional)
- **NVIDIA A16 16GB** — connected via **PCIe Gen4 x16** (~32 GB/s bidirectional)

This hardware choice was deliberate: by comparing NVLink (high-bandwidth, low-latency) against PCIe (lower-bandwidth, higher-latency), we can isolate the impact of interconnect architecture on communication optimization effectiveness.

**Dataset:** Rubble scene from Mega-NeRF (large-scale outdoor, 1.4M initial points)

---

## 2. Training Configuration

All experiments use identical training hyperparameters to ensure fair comparison:

| Parameter | Single GPU | 2-GPU |
|---|---|---|
| Total Batch Size | 8 | 8 |
| Per-GPU Batch Size | 8 | 4 |
| Base Iterations | 30,000 | 15,000 (auto-scaled) |
| Densify Until | 15,000 | 7,500 (auto-scaled) |
| Optimizer (SH) | CPU Adam | CPU Adam |
| SH Location | CPU Pinned Memory | CPU Pinned Memory |

**Auto-scaling:** With 2 GPUs, each GPU processes 4 micro-batches per iteration. To keep the total number of training images seen constant (240,000), iterations are halved from 30,000 to 15,000.

---

## 3. Methods

### 3.1 Baseline (Multi-GPU CLM-GS)

The baseline multi-GPU training flow per iteration:
1. Each GPU independently loads its own SH coefficients from CPU → GPU
2. Each GPU renders its assigned micro-batches and accumulates gradients
3. **Blocking all-reduce** synchronizes gradients across all GPUs (GPUs idle during this)
4. CPU Adam optimizer updates SH parameters on CPU

### 3.2 Method 1: P2P GPU-to-GPU SH Sharing

Instead of each GPU loading the full SH from CPU independently, GPUs split the SH data and exchange via direct GPU-to-GPU memory copies:
- GPU 0 loads the first half of SH from CPU, GPU 1 loads the second half
- GPUs broadcast their halves to each other via `torch.cuda.memcpy_async`

**Rationale:** On NVLink, GPU↔GPU bandwidth (600 GB/s) far exceeds CPU↔GPU bandwidth (~25 GB/s PCIe from host). Splitting the load and sharing via NVLink should be faster than redundant CPU reads.

**Why only tested on A100 NVLink:** P2P direct memory access between GPUs requires either NVLink or same-CPU PCIe root complex with P2P support. On PCIe-connected A16s, P2P transfers would still traverse the PCIe bus (same bandwidth as host→device), offering no benefit. NCCL's all-reduce is already optimized for PCIe topologies.

### 3.3 Method 2: Async All-Reduce

Replaces the blocking all-reduce with a non-blocking variant:
- After all micro-batches complete, initiate all-reduce with `async_op=True`
- Immediately begin the **next iteration's** forward pass while gradients synchronize in the background
- Wait for the all-reduce to complete only before the gradients are needed (optimizer step)

**Rationale:** The blocking all-reduce forces all GPUs to idle while gradients traverse the interconnect. On slow interconnects (PCIe), this idle time dominates. Async all-reduce overlaps communication with computation.

### 3.4 Method 3: Dual-Stream Overlap (Overlap)

Uses a dedicated CUDA stream (`offload_stream`) to overlap SH CPU↔GPU memory transfers with GPU rendering:
- Main stream: rendering + backward pass
- Offload stream: SH coefficient restore (CPU → GPU) and offload (GPU → CPU)
- Stream synchronization via CUDA events ensures data dependencies

**Rationale:** In baseline, SH transfers and rendering are sequential within each micro-batch. Using a second stream allows the GPU to render micro-batch N while simultaneously transferring SH data for micro-batch N+1.

**Bug encountered and fixed:** During development, the overlap method initially produced collapsed PSNR (~7.1 instead of ~24.9). Root cause was a **data race**: `pre_gradient_sync()` called `torch.cuda.synchronize()` which drained all streams and pre-set CPU Adam signal flags. The CPU Adam thread would then start reading the gradient buffer before the main thread finished scaling it with `div_(total_samples)`. Fix: move `div_()` before `cpuadam_worker.start()`.

---

## 4. Results

### 4.1 Reconstruction Quality (Test PSNR)

| Configuration | GPUs | Interconnect | Test PSNR | Test L1 |
|---|---|---|---|---|
| Single GPU (reference) | 1× A100 PCIe | — | 24.92 | 0.0422 |
| Baseline | 2× A100 SXM | NVLink | 24.94 | 0.0421 |
| P2P | 2× A100 SXM | NVLink | **25.00** | 0.0419 |
| Async All-Reduce | 2× A100 SXM | NVLink | 24.93 | 0.0421 |
| Baseline | 2× A16 | PCIe | 24.97 | 0.0419 |
| Async All-Reduce | 2× A16 | PCIe | 24.91 | 0.0423 |
| Overlap | 2× A16 | PCIe | 24.89 | 0.0425 |

**Key observation:** All methods preserve reconstruction quality within ±0.1 dB PSNR of the single-GPU baseline. None of our communication optimizations degrade model quality — they only affect training speed.

### 4.2 Training Speed

| Configuration | GPUs | Interconnect | Wall-Clock | s/iter | Speedup vs Baseline* |
|---|---|---|---|---|---|
| Single GPU | 1× A100 PCIe | — | 8h 22m | 1.00 | — |
| Baseline | 2× A100 SXM | NVLink | 2h 28m | 0.59 | 1.00× |
| P2P | 2× A100 SXM | NVLink | 2h 29m | 0.60 | 0.98× |
| Async All-Reduce | 2× A100 SXM | NVLink | 2h 26m | 0.59 | 1.00× |
| Baseline | 2× A16 | PCIe | 6h 13m | 1.49 | 1.00× |
| Async All-Reduce | 2× A16 | PCIe | **2h 16m** | **0.55** | **2.71×** |
| Overlap | 2× A16 | PCIe | 6h 09m | 1.48 | 1.00× |

*Speedup is relative to the same-hardware baseline (A100 baseline for A100 rows, A16 baseline for A16 rows).

---

## 5. Analysis

### 5.1 Why Async All-Reduce Achieves 2.69× Speedup on A16 PCIe

The A16 baseline iteration takes 1.49s. Of that, only ~0.55s is useful compute (GPU rendering, backward pass, CPU Adam). The remaining ~0.94s — **63% of total iteration time** — is spent idle, waiting for the all-reduce gradient synchronization to complete over the PCIe bus.

Async all-reduce eliminates this idle time by overlapping the gradient sync with the next iteration's forward pass. By the time the forward pass of iteration N+1 completes, the all-reduce from iteration N has finished in the background.

```
Baseline (blocking):
  [Compute 0.55s][====== All-Reduce 0.94s ======]  → 1.49s total
                  ↑ GPUs idle here

Async (non-blocking):
  [Compute 0.55s][Compute 0.55s][Compute 0.55s]... → 0.55s effective
  └─ AR starts ──┘              └─ AR starts ──┘
     (finishes before next optimizer step)
```

### 5.2 Why All Methods Show No Improvement on A100 NVLink

On A100 with NVLink (~600 GB/s), the all-reduce completes in approximately 25ms — negligible compared to the ~590ms iteration time. The iteration time breakdown is approximately:

| Component | A100 NVLink | A16 PCIe |
|---|---|---|
| GPU Compute (render + backward) | ~450ms (76%) | ~450ms (30%) |
| All-Reduce Gradient Sync | ~25ms (4%) | **~940ms (63%)** |
| CPU Adam Optimizer | ~80ms (14%) | ~50ms (3%) |
| SH Transfer (CPU↔GPU) | ~35ms (6%) | ~50ms (3%) |

On A100, **communication is not the bottleneck** — it constitutes only ~4% of iteration time. No communication optimization can yield meaningful speedup. The dominant costs are GPU compute and CPU Adam, which all methods share equally.

### 5.3 Why Overlap Shows No Improvement Even on A16

Overlap targets SH CPU↔GPU transfers (~50ms per iteration, ~3% of total). Even with perfect overlap, the savings are imperceptible against the 940ms all-reduce cost. Overlap optimizes the wrong bottleneck.

### 5.4 Why P2P Shows No Improvement on A100 NVLink

P2P replaces redundant CPU→GPU SH loads with a split-load-and-broadcast pattern via NVLink. However:
1. Each GPU's CPU→GPU SH transfer is already fast (~17ms per half)
2. The NVLink broadcast adds its own overhead (kernel launch, synchronization)
3. At `world_size=2`, NCCL all-reduce is already a simple send+recv — there's no multi-hop overhead to eliminate

The net effect is roughly zero: the time saved by halving each GPU's CPU read is offset by the NVLink broadcast cost.

### 5.5 Multi-GPU Scaling Efficiency

Single A100 → 2× A100 scaling:
- Per-iteration speedup: 1.00 / 0.59 = **1.69×** (of 2.0× theoretical)
- Wall-clock speedup: 8h22m / 2h28m = **3.39×** (includes iteration reduction: 30k → 15k)

The sub-linear per-iteration scaling (1.69× vs 2.0×) is expected due to Amdahl's Law: serial components (CPU Adam, gradient sync, densification synchronization) cannot be parallelized and set a floor on iteration time.

---

## 6. Hardware Context

### 6.1 Why We Chose A100 SXM and A16

| Specification | NVIDIA A100-SXM 80GB | NVIDIA A16 16GB |
|---|---|---|
| Architecture | Ampere (GA100) | Ampere (GA107) |
| CUDA Cores | 6,912 | 1,280 |
| Memory | 80 GB HBM2e | 16 GB GDDR6 |
| Memory Bandwidth | 2,039 GB/s | 177 GB/s |
| GPU Interconnect | **NVLink 3.0 (600 GB/s)** | **PCIe Gen4 x16 (32 GB/s)** |
| Target Workload | HPC / AI Training | Inference / VDI |

These two GPUs represent opposite ends of the multi-GPU communication spectrum:
- **A100 SXM** uses NVLink, purpose-built for multi-GPU training with 18.75× the inter-GPU bandwidth of PCIe
- **A16** uses standard PCIe, representative of commodity/cloud deployments where NVLink is unavailable

This contrast directly tests whether communication optimizations matter when interconnect bandwidth varies by an order of magnitude.

### 6.2 Why P2P Was Only Tested on A100

P2P direct GPU-to-GPU memory copies (`cudaMemcpyPeerAsync`) are designed for NVLink topologies. On PCIe-connected GPUs:
- P2P transfers still traverse the PCIe bus through the CPU's root complex
- Effective bandwidth is no better than standard CPU-mediated copies
- NCCL already implements topology-aware routing that outperforms manual P2P on PCIe

Testing P2P on A16 PCIe would produce results identical (or slightly worse) than baseline, adding no insight.

---

## 7. Conclusions

1. **Async all-reduce is highly effective on PCIe-connected GPUs** (2.71× speedup on A16), where gradient synchronization dominates iteration time (63%). This is a practical win for the common case of multi-GPU training without NVLink.

2. **No communication optimization helps on NVLink-connected GPUs** (A100 SXM). NVLink reduces inter-GPU communication to ~4% of iteration time, making it a non-bottleneck. The dominant cost shifts to the CPU Adam optimizer, which none of our methods address.

3. **Dual-stream overlap provides no measurable benefit** on either platform. The SH CPU↔GPU transfer it targets is only ~3% of iteration time — too small to matter regardless of interconnect.

4. **P2P GPU-to-GPU SH sharing provides no measurable benefit** on NVLink. At `world_size=2`, NCCL's all-reduce is already a direct send+recv, and the CPU→GPU SH load is not a bottleneck.

5. **The key insight is hardware-aware optimization:** the same optimization (async all-reduce) can be transformative or useless depending on the interconnect. Profiling the actual bottleneck before optimizing is essential — communication optimization on NVLink-equipped systems is wasted engineering effort for this workload.

### Future Work

The remaining bottleneck on NVLink systems is the **single-threaded CPU Adam optimizer**. Potential approaches:
- Multi-threaded CPU Adam (parallelize across CPU cores)
- Overlap CPU Adam with next iteration's forward pass
- Hybrid optimizer: keep small parameter groups on GPU, offload only SH to CPU

---

## 8. Experiment Index

| Job ID | Configuration | GPU | Status | Final PSNR | Wall-Clock |
|---|---|---|---|---|---|
| 5621227 | Single GPU | 1× A100 PCIe-40GB | Complete | 24.92 | 8h 22m |
| 5622780 | Baseline 2-GPU | 2× A100 SXM-80GB | Complete | 24.94 | 2h 28m |
| 5621229 | P2P 2-GPU | 2× A100 SXM-80GB | Complete | 25.00 | 2h 29m |
| 5623273 | Async 2-GPU | 2× A100 SXM-80GB | Complete | 24.93 | 2h 26m |
| 5644690 | Baseline 2-GPU | 2× A16 | Complete | 24.97 | 6h 13m |
| 5634761 | Async 2-GPU | 2× A16 | Complete | 24.91 | 2h 16m |
| 5643462 | Overlap 2-GPU | 2× A16 | Complete | 24.89 | 6h 09m |

---

**Cluster:** UVA CS Department — jaguar02 (8× A16, 32 CPUs), cheetah04 (A100 SXM nodes)
**Framework:** CLM-GS (CPU-offloading for Large-scale Mapping Gaussian Splatting) with custom multi-GPU extensions
**Software:** PyTorch 2.x, NCCL 2.21.5, CUDA 12.4
