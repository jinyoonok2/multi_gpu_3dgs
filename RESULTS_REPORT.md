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

All experiments use **weak scaling** to ensure fair comparison: each GPU processes the same batch size as the single-GPU baseline so the per-GPU workload is constant. The number of iterations is scaled down proportionally to keep the total training samples seen identical.

| Parameter | Single GPU | 2-GPU (Weak Scaling) |
|---|---|---|
| Total Batch Size | 8 | 16 (8 per GPU) |
| Per-GPU Batch Size | 8 | 8 |
| Iterations | 30,000 | 15,000 (scaled by 1/world_size) |
| Total Images Seen | 240,000 | 240,000 |
| Densify Until | 15,000 | 7,500 (auto-scaled) |
| Optimizer (SH) | CPU Adam | CPU Adam |
| SH Location | CPU Pinned Memory | CPU Pinned Memory |

**Why weak scaling:** The reason to use more GPUs is that each iteration can process more samples in parallel. A single GPU with 16GB VRAM (A16) can handle BSZ=8 but may not handle BSZ=16. With 2 GPUs, we get a global BSZ=16 while each GPU only needs memory for BSZ=8 — enabling larger effective batch sizes. To keep total training work equal, iterations are halved: 2 GPUs × 8 images/GPU × 15k iters = 1 GPU × 8 images × 30k iters = 240k images.

---

## 3. Methods

### 3.1 Baseline (Multi-GPU CLM-GS)

The baseline multi-GPU training flow per iteration:
1. Each GPU independently loads its own SH coefficients from CPU → GPU
2. Each GPU renders its assigned 8-image micro-batch and computes gradients
3. **Blocking all-reduce** synchronizes gradients across all GPUs (GPUs idle during this)
4. CPU Adam optimizer updates SH parameters on CPU

### 3.2 P2P GPU-to-GPU SH Sharing

Instead of each GPU loading the full SH from CPU independently, GPUs split the SH data and exchange via direct GPU-to-GPU memory copies:
- GPU 0 loads the first half of SH from CPU, GPU 1 loads the second half
- GPUs broadcast their halves to each other via `torch.cuda.memcpy_async`

**Rationale:** On NVLink, GPU↔GPU bandwidth (600 GB/s) far exceeds CPU↔GPU bandwidth (~25 GB/s PCIe from host). Splitting the load and sharing via NVLink should be faster than redundant CPU reads.

### 3.3 Async All-Reduce

Replaces the blocking all-reduce with a non-blocking variant:
- After the forward/backward pass, initiate all-reduce with `async_op=True`
- Immediately begin the **next iteration's** forward pass while gradients synchronize in the background
- Wait for the all-reduce to complete only before the gradients are needed (optimizer step)

**Rationale:** The blocking all-reduce forces all GPUs to idle while gradients traverse the interconnect. On slow interconnects (PCIe), this idle time dominates. Async all-reduce overlaps communication with computation.

### 3.4 Dual-Stream Overlap

Uses a dedicated CUDA stream (`offload_stream`) to overlap SH CPU↔GPU memory transfers with GPU rendering:
- Main stream: rendering + backward pass
- Offload stream: SH coefficient restore (CPU → GPU) and offload (GPU → CPU)
- Stream synchronization via CUDA events ensures data dependencies

**Rationale:** In baseline, SH transfers and rendering are sequential. Using a second stream allows the GPU to render while simultaneously transferring SH data for the next micro-batch.

### 3.5 All Combined (P2P + Overlap + Async)

Applies all three optimizations simultaneously. This tests whether the methods are complementary — each targeting a different bottleneck — or whether they interfere with each other.

---

## 4. Results

### 4.1 Reconstruction Quality (Test PSNR)

| Configuration | GPUs | Interconnect | Test PSNR | Test L1 | Status |
|---|---|---|---|---|---|
| Single GPU | 1× A16 | — | **25.01** | 0.0416 | ✅ Complete |
| Single GPU | 1× A100 SXM | — | **24.97** | 0.0418 | ✅ Complete |
| Baseline (CLM) | 2× A16 | PCIe | 24.84 | 0.0425 | ✅ Complete |
| Baseline (CLM) | 2× A100 SXM | NVLink | 24.85 | 0.0424 | ✅ Complete |
| Async | 2× A16 | PCIe | 24.79 | 0.0428 | ✅ Complete |
| Async | 2× A100 SXM | NVLink | 24.86 | 0.0423 | ✅ Complete |
| P2P | 2× A16 | PCIe | — | — | 🔄 Running |
| P2P | 2× A100 SXM | NVLink | 24.84 | 0.0425 | ✅ Complete |
| Overlap | 2× A16 | PCIe | — | — | ⏳ Pending |
| Overlap | 2× A100 SXM | NVLink | 24.88 | 0.0421 | ✅ Complete |
| All Combined | 2× A16 | PCIe | — | — | ⏳ Pending |
| All Combined | 2× A100 SXM | NVLink | — | — | ⏳ Pending |

**Key observation:** All completed methods preserve reconstruction quality within ~0.2 dB PSNR of the single-GPU baseline. The slight drop (~0.15 dB) in 2-GPU runs is expected: with weak scaling, gradient averaging across GPUs introduces a small smoothing effect relative to full single-GPU gradient updates.

### 4.2 Training Speed

| Configuration | GPUs | Interconnect | Wall-Clock | Speedup vs Single GPU | Speedup vs 2-GPU Baseline |
|---|---|---|---|---|---|
| Single GPU | 1× A16 | — | 8h 36m | 1.00× | — |
| Single GPU | 1× A100 SXM | — | 1h 29m | 1.00× | — |
| Baseline (CLM) | 2× A16 | PCIe | 5h 34m | 1.54× | 1.00× |
| Baseline (CLM) | 2× A100 SXM | NVLink | 1h 51m | 0.80× | 1.00× |
| Async | 2× A16 | PCIe | 5h 29m | 1.57× | 1.02× |
| Async | 2× A100 SXM | NVLink | 1h 47m | 0.83× | 1.04× |
| P2P | 2× A16 | PCIe | — | — | — |
| P2P | 2× A100 SXM | NVLink | 1h 35m | 0.94× | 1.17× |
| Overlap | 2× A16 | PCIe | — | — | — |
| Overlap | 2× A100 SXM | NVLink | 1h 39m | 0.90× | 1.12× |
| All Combined | 2× A16 | PCIe | — | — | — |
| All Combined | 2× A100 SXM | NVLink | — | — | — |

---

## 5. Analysis

### 5.1 Multi-GPU Scaling: A16 vs A100

The most striking result is the difference in multi-GPU scaling between the two architectures:

**A16 (PCIe):** 2-GPU achieves a **1.54× speedup** over single-GPU (8h 36m → 5h 34m). With weak scaling, the ideal speedup is 2.0× (half the iterations, same per-iteration time). The 1.54× indicates that communication overhead (all-reduce over PCIe) adds ~30% to per-iteration time, but the halved iteration count more than compensates.

**A100 (NVLink):** 2-GPU is actually **slower** than single-GPU (1h 29m → 1h 51m, 0.80× "speedup"). Despite having NVLink's fast interconnect, the per-iteration overhead from gradient synchronization, process coordination, and distributed training setup adds enough cost that halving the iterations (30k → 15k) is not sufficient to overcome it. Each multi-GPU iteration takes more than 2× the time of a single-GPU iteration.

**Why the difference?** The A100 single-GPU is very fast (~0.18s/iter), so the fixed overhead of distributed coordination (NCCL setup, barrier synchronization, gradient averaging) represents a large fraction of the iteration time. On the slower A16 (~1.03s/iter single-GPU), the same fixed overhead is proportionally smaller.

### 5.2 Method Comparison on A100 NVLink

Among the 2-GPU A100 runs, the optimization methods do show modest improvements over the baseline:

| Method | Wall-Clock | vs Baseline |
|---|---|---|
| Baseline | 1h 51m | 1.00× |
| Async | 1h 47m | 1.04× |
| Overlap | 1h 39m | 1.12× |
| P2P | 1h 35m | **1.17×** |

P2P shows the best improvement (1.17×), suggesting that on NVLink, reducing redundant CPU→GPU SH transfers is the most effective optimization. Overlap (dual-stream) also helps by hiding SH transfer latency behind compute. Async all-reduce provides only a small benefit because NVLink all-reduce is already fast.

However, **none of these bring 2-GPU A100 below the single-GPU time** (1h 29m). The fastest 2-GPU A100 result (P2P, 1h 35m) is still ~7% slower than the single-GPU baseline. This suggests that for this workload at this scale, the overhead of multi-GPU coordination is not fully amortizable with just 2 GPUs on A100.

### 5.3 Method Comparison on A16 PCIe

On A16, async all-reduce provides only a marginal ~1% improvement over baseline (5h 29m vs 5h 34m). This is surprising given that PCIe has much lower bandwidth than NVLink. The likely explanation is that with weak scaling (BSZ=16, 15k iters), the per-iteration all-reduce cost on A16 PCIe is not as dominant as it would be with more iterations — each iteration already has significant compute time from the 8-image per-GPU batch.

(A16 P2P, Overlap, and All Combined results are pending.)

### 5.4 PSNR Quality vs Single-GPU

All 2-GPU methods show a consistent ~0.15 dB PSNR drop compared to single GPU:
- Single GPU A16: 25.01, Single GPU A100: 24.97
- 2-GPU methods: ~24.79–24.88

This small quality difference is inherent to the weak scaling approach: with 2 GPUs, gradients are averaged across mini-batches from different GPUs at each iteration, which introduces slightly different optimization dynamics than processing all 8 images on one GPU sequentially. The difference is negligible for practical purposes.

---

## 6. Hardware Context

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

---

## 7. Conclusions

1. **Multi-GPU scaling depends heavily on single-GPU iteration speed.** On slower GPUs (A16), 2-GPU training achieves 1.54× speedup (strong win). On faster GPUs (A100), 2-GPU training is actually slower than single-GPU (0.80×) because the fixed distributed overhead becomes a larger fraction of the already-short iteration time.

2. **P2P SH sharing is the most effective optimization on A100 NVLink** (1.17× over 2-GPU baseline), followed by dual-stream overlap (1.12×). These target the CPU→GPU SH transfer bottleneck, which is more impactful than all-reduce latency on NVLink.

3. **Communication-hiding optimizations (async, overlap) provide modest benefits** when the interconnect is fast (NVLink). Their value increases when communication is a larger bottleneck — which may manifest more at higher GPU counts or on PCIe-only systems.

4. **Weak scaling preserves model quality.** All 2-GPU configurations stay within ~0.2 dB PSNR of the single-GPU baseline, with identical total training samples (240k images).

5. **The key takeaway is that multi-GPU training is not always faster.** For compute-bound workloads on fast GPUs (A100), the coordination overhead can outweigh the benefit of splitting iterations — especially at small GPU counts (2 GPUs). Multi-GPU is most beneficial on slower GPUs where each iteration is long enough to amortize the communication cost.

### Pending Results

The following experiments are still running or queued. Once complete, they will fill in the missing entries in the tables above:
- 2× A16 P2P (Job 5650959 — running)
- 2× A16 Overlap (Job 5650961 — pending)
- 2× A16 All Combined (Job 5652024 — pending)
- 2× A100 All Combined (Job 5652025 — pending)

---

## 8. Experiment Index

| Job ID | Configuration | GPU | Node | Status | Test PSNR | Wall-Clock |
|---|---|---|---|---|---|---|
| 5650953 | Single GPU | 1× A16 | jaguar02 | ✅ Complete | 25.01 | 8h 36m |
| 5650954 | Single GPU | 1× A100 SXM | cheetah04 | ✅ Complete | 24.97 | 1h 29m |
| 5650955 | Baseline (CLM) 2-GPU | 2× A16 | jaguar02 | ✅ Complete | 24.84 | 5h 34m |
| 5650956 | Baseline (CLM) 2-GPU | 2× A100 SXM | cheetah04 | ✅ Complete | 24.85 | 1h 51m |
| 5650957 | Async 2-GPU | 2× A16 | jaguar02 | ✅ Complete | 24.79 | 5h 29m |
| 5650958 | Async 2-GPU | 2× A100 SXM | cheetah04 | ✅ Complete | 24.86 | 1h 47m |
| 5650959 | P2P 2-GPU | 2× A16 | jaguar02 | Running | — | — |
| 5650960 | P2P 2-GPU | 2× A100 SXM | cheetah04 | ✅ Complete | 24.84 | 1h 35m |
| 5650961 | Overlap 2-GPU | 2× A16 | jaguar02 | Pending | — | — |
| 5650962 | Overlap 2-GPU | 2× A100 SXM | cheetah04 | ✅ Complete | 24.88 | 1h 39m |
| 5652024 | All Combined 2-GPU | 2× A16 | — | Pending | — | — |
| 5652025 | All Combined 2-GPU | 2× A100 SXM | — | Pending | — | — |

---

**Cluster:** UVA CS Department — jaguar02 (8× A16, 32 CPUs), cheetah04 (2× A100 SXM, NVLink)
**Framework:** CLM-GS (CPU-offloading for Large-scale Mapping Gaussian Splatting) with custom multi-GPU extensions
**Software:** PyTorch 2.x, NCCL 2.21.5, CUDA 12.4
