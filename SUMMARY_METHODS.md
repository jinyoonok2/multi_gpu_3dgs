# Multi-GPU CLM Gaussian Splatting — Optimization Methods

## 1. Baseline: Multi-GPU CLM (Cache-Locality-aware Memory) Offloading

### Parallelism Model: Camera Parallelism (not Gaussian Parallelism)

The multi-GPU strategy in this codebase — both the baseline and our methods — is **camera parallelism** (a form of data parallelism), not Gaussian parallelism (model parallelism).

| Approach | What's split across GPUs | What's replicated | Sync mechanism |
|----------|------------------------|-------------------|----------------|
| **Camera parallelism (this codebase)** | Camera views (micro-batches) | All N Gaussians (params + SH) | `all_reduce` on gradients (DDP) |
| Gaussian parallelism | Gaussians (each GPU owns a subset) | Camera views broadcast to all | `all_gather` on rendered outputs |

Each GPU holds a **complete copy** of all N Gaussians but processes **different camera views**. After computing gradients from its own cameras, gradients are averaged across GPUs via `all_reduce` so every GPU applies the same parameter update. This is standard PyTorch DDP (Distributed Data Parallel).

**Why this matters for our methods:** Because every GPU holds the same Gaussians but renders different cameras, the visible Gaussian sets **partially overlap** between GPUs. This overlap is what creates redundant PCIe loads (which Method 1 eliminates) and what creates the prefetch/offload traffic (which Method 2 pipelines).

### CLM Offloading

CLM-GS keeps only **4 small parameter tensors** on GPU (xyz, opacity, scaling, rotation) and offloads the large **Spherical Harmonic (SH) coefficients** (48 floats × N Gaussians) to CPU pinned memory. Each training iteration processes a batch of camera views as micro-batches, selectively loading only the SH coefficients visible to each camera.

**5-stage pipeline per iteration:**

1. **Filter & Order** — Determine which Gaussians are visible per camera; sort cameras via TSP to maximize overlap between consecutive views (better cache reuse).
2. **CPU Thread Init** — Start a background CPU Adam optimizer thread (for SH parameters) that runs concurrently with GPU work.
3. **State Init** — Zero gradient accumulators; create a single `comm_stream` for CPU↔GPU transfers; initialize retention bitmasks.
4. **Micro-batch Loop** — For each camera view:
   - **4.1** Load visible SH from CPU → GPU (on `comm_stream`)
   - **4.2** Prefetch next camera's SH using retention: classify Gaussians as H (Host-load), D (Duplicate/retain), G (Garbage/offload)
   - **4.3** Forward pass (Gaussian splatting → rendered image)
   - **4.4** Backward pass (compute gradients)
   - **4.5** Scatter gradients back to full-size accumulators
   - **4.6** Offload SH gradients from GPU → CPU (on `comm_stream`)
5. **Post-training** — All-reduce gradients across GPUs (DDP), GPU Adam step for the 4 cached params, join CPU Adam thread.

**Key bottleneck:** Steps 4.2 (prefetch) and 4.6 (offload) share a **single `comm_stream`**, serializing CPU→GPU and GPU→CPU transfers.

---

## 2. Method 1: P2P GPU-to-GPU SH Sharing

### Problem

In multi-GPU DDP training, GPUs render different cameras but many Gaussians are visible to multiple cameras. Each GPU independently loads overlapping SH coefficients from CPU via PCIe (~25 GB/s per direction). This means the **same SH data traverses the PCIe bus multiple times** — once per GPU that needs it.

```
          CPU (pinned memory)
          ┌──────────────────┐
          │   SH coefficients│
          └──┬───────────┬───┘
    PCIe ↓ 25 GB/s    PCIe ↓ 25 GB/s    ← redundant!
     ┌────┴────┐    ┌────┴────┐
     │  GPU 0  │    │  GPU 1  │
     └─────────┘    └─────────┘
```

### Solution: Cooperative Loading via NVLink

Instead of both GPUs loading from CPU, we:
1. **Exchange visibility** — Each GPU shares which Gaussians it needs via `all_reduce(SUM)` on the filter bitmasks
2. **Partition** into three categories:
   - **Overlap**: visible on 2+ GPUs → loaded once by GPU 0, broadcast to others
   - **Local-only**: visible on this GPU only → loaded from CPU normally
   - **Peer-only**: visible only on the other GPU → not loaded at all

```
          CPU (pinned memory)
          ┌──────────────────┐
          │   SH coefficients│
          └──┬───────────────┘
    PCIe ↓ (1× for overlap + local)
     ┌────┴────┐  NVLink  ┌─────────┐
     │  GPU 0  │─────────→│  GPU 1  │
     └─────────┘ 600-900  └─────────┘
                  GB/s
```

### Detailed Mechanism

#### Stage 1 — Filter Partition (new)

After the standard `calculate_filters()`:

```python
# Each GPU has a binary visibility mask: shape (N,), 1 = visible
local_mask = torch.zeros(N, device='cuda')
local_mask[my_filter_indices] = 1

# Sum across GPUs: mask > 1 means multiple GPUs need this Gaussian
combined_mask = all_reduce(local_mask, op=SUM)

# Classify
overlap       = (combined_mask > 1) & (local_mask == 1)  # shared, need it
local_only    = (combined_mask == 1) & (local_mask == 1)  # only I need it
peer_only     = (combined_mask > 0) & (local_mask == 0)   # only peer needs it
```

This classification happens once per iteration (cheap — just integer arithmetic on GPU).

#### Stage 4.1 — Cooperative SH Loading (modified first micro-batch)

For the first micro-batch, instead of each GPU calling `send_shs2gpu_stream()` for all its visible Gaussians:

```
Step 1: GPU 0 loads overlap SH from CPU        → PCIe (1×, not 2×)
Step 2: GPU 0 broadcasts overlap SH to GPU 1   → NVLink (600-900 GB/s)
Step 3: Each GPU loads its local_only SH        → PCIe (independent, no redundancy)
Step 4: Each GPU scatters into correct positions in shs tensor
```

For subsequent micro-batches (micro_idx > 0), the existing retention logic (H/D/G categories) handles everything — no P2P needed since retention is already a GPU-local optimization.

#### Stage 5.0 — Gradient Synchronization

Functionally the same as baseline — `all_reduce` on GPU param grads + copy-to-GPU → all_reduce → copy-back for CPU SH grads. Encapsulated in `P2PCommManager.sync_gradients_p2p()`.

### PCIe Traffic Analysis

| Scenario | Overlap Ratio | GPU 0 PCIe Load | GPU 1 PCIe Load | Total PCIe |
|----------|--------------|-----------------|-----------------|------------|
| Baseline (no P2P) | 60% | 100% | 100% | 200% |
| **With P2P** | 60% | 100% (overlap + local) | 40% (local only) | 140% |
| Baseline (no P2P) | 80% | 100% | 100% | 200% |
| **With P2P** | 80% | 100% (overlap + local) | 20% (local only) | 120% |

The higher the overlap ratio (common in outdoor scenes with large Gaussians), the greater the savings.

### When P2P Helps vs. Doesn't Help

| Hardware | NVLink BW | PCIe BW | P2P Benefit |
|----------|-----------|---------|------------|
| A100 SXM 40GB | 600 GB/s | 25 GB/s | **High** — NVLink 24× faster than PCIe |
| H100 NVL 80GB | 900 GB/s | 32 GB/s | **Medium** — Large VRAM reduces CLM need |
| A40 (PCIe) | N/A (PCIe P2P ~25 GB/s) | 25 GB/s | **None** — P2P shares same bus |
| A100 PCIe | N/A (PCIe P2P ~25 GB/s) | 25 GB/s | **None** — No NVLink |

**Key insight:** P2P only helps when NVLink bandwidth >> PCIe bandwidth AND VRAM is constrained enough to require CLM offloading.

### Why GPU Interconnect Topology Matters

P2P relies on one GPU sending data directly to another GPU. The bandwidth of the **interconnect between GPUs** determines whether this is worthwhile:

**NVLink** is a dedicated high-bandwidth link between GPUs that bypasses the PCIe bus entirely:

```
With NVLink:
  CPU ──PCIe (25 GB/s)──→ GPU 0 ──NVLink (600 GB/s)──→ GPU 1
  Total: 1× PCIe load + 1× NVLink broadcast (essentially free)

Without NVLink (PCIe P2P):
  CPU ──PCIe (25 GB/s)──→ GPU 0 ──PCIe (25 GB/s)──→ GPU 1
  Total: 1× PCIe load + 1× PCIe transfer (shares the SAME bus!)
```

When GPUs connect via **PCIe only** (A40, A100-PCIe), the GPU-to-GPU transfer goes through the PCIe switch or CPU chipset — **the exact same bus** used for CPU→GPU transfers. So sending data from GPU 0 to GPU 1 over PCIe consumes the same bandwidth that could have been used to load from CPU. There is no net savings; you're just shifting traffic from one PCIe transaction to another.

With **NVLink** (A100 SXM, H100 NVL), the GPU-to-GPU path is a separate physical link running at 600–900 GB/s — **24–36× faster than PCIe**. The broadcast over NVLink completes in microseconds for the same data that would take milliseconds over PCIe. This is why NVLink is mandatory for Method 1 to be effective.

```
Latency comparison (broadcasting 100 MB of SH data):
  NVLink 600 GB/s:  100 MB / 600 GB/s ≈ 0.17 ms
  PCIe 25 GB/s:     100 MB / 25 GB/s  ≈ 4.0 ms    ← 24× slower
```

### Why VRAM Size Matters — And Why 40GB A100 SXM Is the Sweet Spot

A natural question: *if a GPU has very large VRAM (e.g., 80GB), does CLM offloading even happen? And if not, does P2P matter?*

**VRAM determines how much CLM offloading is needed.** CLM exists because SH coefficients are too large to keep entirely on GPU alongside everything else needed for training:

| Component | Size (10M Gaussians) |
|-----------|---------------------|
| SH coefficients (48 floats × N) | **1.92 GB** |
| SH gradients | 1.92 GB |
| SH optimizer states (Adam: m + v) | 3.84 GB |
| GPU params (xyz, opacity, scaling, rotation) | ~0.5 GB |
| GPU gradients + optimizer states | ~2.0 GB |
| Intermediate buffers (forward/backward) | ~2–4 GB |
| PyTorch/CUDA overhead | ~1–2 GB |
| **Total training footprint** | **~14–16 GB** |

For 10M Gaussians, this fits in 40GB. But real scenes are much larger:

| Scene | Gaussians | SH alone | Full training footprint |
|-------|-----------|----------|------------------------|
| Small indoor | 2–5M | 0.4–1.0 GB | 6–10 GB |
| Rubble (our test) | 10–20M | 2–4 GB | 14–28 GB |
| Urban city-scale | 50–100M | 10–19 GB | 60–120 GB |
| MegaCity / MatrixCity | 200M+ | 38+ GB | 200+ GB |

**With 40GB VRAM (A100 SXM):**
- Rubble (10–20M Gaussians): The training footprint of 14–28 GB technically fits, but when you add optimizer states, gradient accumulators, and CUDA memory fragmentation, the SH coefficients often cannot be kept on GPU. CLM offloading activates for the SH component → **heavy PCIe traffic** → P2P provides big savings by eliminating redundant loads.
- Larger scenes simply don't fit at all. CLM is essential.

**With 80GB VRAM (H100 NVL):**
- Rubble: Fits comfortably. Less CLM offloading needed → less PCIe traffic → less for P2P to optimize. P2P still helps but the absolute time savings is smaller.
- City-scale (50M+): Back to needing CLM even at 80GB → P2P becomes valuable again.

**The key realization: VRAM size determines the threshold at which CLM kicks in, but scene complexity always grows.** Research-scale scenes like Rubble hit 40GB limits today, and next-generation city-scale reconstructions will exceed 80GB. The A100 SXM 40GB is the sweet spot for evaluating our methods because:

1. **VRAM is constrained enough** that CLM offloading is necessary on our Rubble test scene → heavy PCIe traffic to optimize
2. **NVLink is available** (600 GB/s) → P2P broadcast is essentially free
3. **It's a realistic deployment target** — many research clusters run A100 40GB

In summary:

| GPU | VRAM | NVLink | P2P Usefulness |
|-----|------|--------|---------------|
| A100 SXM 40GB | Constrained → CLM active | 600 GB/s | **Best target** — lots of traffic to save + fast inter-GPU link |
| H100 NVL 80GB | Less constrained | 900 GB/s | Moderate — helps on larger scenes |
| A40 48GB | Moderate | None (PCIe only) | **No benefit** — inter-GPU path is PCIe |

---

## 3. Method 2: Dual-Stream Overlapped Scheduling

### Problem

In the baseline, a **single `comm_stream`** handles both:
- **Prefetch**: loading SH(i+1) from CPU → GPU
- **Offload**: sending gradients(i) from GPU → CPU

These are serialized on the same stream:

```
Timeline (baseline — single comm_stream):

default_stream:  [  fwd(i)  ][  bwd(i)  ]          [  fwd(i+1)  ][  bwd(i+1)  ]
comm_stream:         [ prefetch(i+1) ][ offload(i) ]    [ prefetch(i+2) ][ offload(i+1) ]
                                      ↑
                              Must wait for prefetch
                              to finish first!
```

The gradient offload for micro-batch i **cannot start** until the prefetch for i+1 finishes — even though the backward pass may have completed long before the prefetch ends. This creates a pipeline bubble.

### Solution: Split into Two CUDA Streams

Create a dedicated `offload_stream` for gradient transfers:

```
Timeline (method 2 — dual streams):

default_stream:  [  fwd(i)  ][  bwd(i)  ][ fwd(i+1) ][ bwd(i+1) ]
comm_stream:         [ prefetch SH(i+1) ]     [ prefetch SH(i+2) ]
offload_stream:            [ offload grad(i-1) ]    [ offload grad(i) ]
                           ↑                    ↑
                     Runs in parallel       Runs in parallel
                     with prefetch!         with prefetch!
```

**Three-way overlap:** GPU compute, CPU→GPU prefetch, and GPU→CPU offload all proceed concurrently.

### Why This Works at the Hardware Level

PCIe is **full-duplex**: CPU→GPU and GPU→CPU can transfer simultaneously at full bandwidth in each direction. By putting them on separate streams, the CUDA runtime can schedule them concurrently:

```
PCIe Bus:
  CPU → GPU direction:  [====== prefetch SH(i+1) ======]
  GPU → CPU direction:  [==== offload grad(i-1) ====]
  (both at ~25 GB/s simultaneously)

GPU Compute Engine:
  [======= forward(i) + backward(i) =======]
```

### Detailed Mechanism

#### Stage 3 — Stream Creation (new)

```python
default_stream = torch.cuda.current_stream()   # forward/backward
# comm_stream is passed as parameter                # prefetch (CPU→GPU)
offload_stream = torch.cuda.Stream(device=gpu)  # NEW: offload (GPU→CPU)
```

#### Stage 4.2 — Pre-compute Offload Indices (modified)

In the baseline, Category G indices (Gaussians to offload) are computed in section 4.6 on `comm_stream`, right before the offload transfer. This ties the offload to comm_stream.

In method 2, we **pre-compute G indices in section 4.2** alongside the existing H and D indices (which are already computed here for prefetch). Then record an event:

```python
# Already computed on comm_stream in section 4.2:
# Category H: host_indices_to_param, param_indices_from_host  (for prefetch)
# Category D: rtnt_indices_to_param, param_indices_from_rtnt   (for retention)

# NEW: pre-compute Category G for offload_stream
bit_g = this_bit & ~next_bit
idx_g = torch.nonzero_static(bit_g, size=cnt_g[micro_idx]).flatten()
precomp_host_indices_from_grad = idx_g.to(torch.int32)
precomp_grad_indices_to_host = torch.gather(retention_vec, dim=0, index=idx_g)

# Record event so offload_stream knows indices are ready
indices_ready_event = torch.cuda.Event()
indices_ready_event.record(comm_stream)
```

#### Stage 4.6 — Offload on Dedicated Stream (modified)

```python
# BASELINE: runs on comm_stream (serialized after prefetch)
# METHOD 2: runs on offload_stream (parallel with prefetch)

with torch.cuda.stream(offload_stream):
    gpu2cpu_event.wait(offload_stream)      # wait for backward to finish
    indices_ready_event.wait(offload_stream) # wait for indices from comm_stream

    # Now offload gradients — runs in PARALLEL with prefetch on comm_stream
    send_shs2cpu_grad_buffer_stream_retention(
        shs_grad, parameters_grad_buffer, shs_grad_next,
        host_indices_from_grad, rtnt_indices_from_grad,
        grad_indices_to_host, grad_indices_to_rtnt, ...
    )
```

#### Synchronization Events

```
Event Flow:

comm_stream:    ──[compute H,D,G indices]──●indices_ready──[prefetch transfer]──
                                           │
offload_stream: ─────────────wait(indices_ready)──wait(gpu2cpu)──[offload]──────
                                                    │
default_stream: ──[forward]──[backward]──●gpu2cpu───────────────────────────────
```

Three events coordinate the streams:
- `cpu2gpu_event`: comm_stream → default_stream (SH loaded, safe to compute)
- `gpu2cpu_event`: default_stream → offload_stream (backward done, safe to offload)
- `indices_ready_event`: comm_stream → offload_stream (G indices computed, safe to use them)

### Performance Model

Let:
- $T_p$ = prefetch transfer time (CPU→GPU)
- $T_o$ = offload transfer time (GPU→CPU)
- $T_c$ = compute time (forward + backward)

**Baseline (single stream):**
$$T_{iteration} \approx \sum_{i} \max(T_c^{(i)},\; T_p^{(i)} + T_o^{(i-1)})$$

The $T_p + T_o$ sum is the bottleneck because they're serialized on one stream.

**Method 2 (dual stream):**
$$T_{iteration} \approx \sum_{i} \max(T_c^{(i)},\; T_p^{(i)},\; T_o^{(i-1)})$$

Now $T_p$ and $T_o$ overlap — the bottleneck is the max of the three, not a sum.

**Speedup factor per micro-batch:**
$$\text{Speedup} = \frac{\max(T_c, T_p + T_o)}{\max(T_c, T_p, T_o)}$$

When $T_p \approx T_o \approx T_c$: baseline = $\max(T_c, 2T_c) = 2T_c$, method 2 = $T_c$ → **2× speedup** on the comm-bound portion.

### When Overlap Helps vs. Doesn't Help

| Scenario | Benefit |
|----------|---------|
| Large filters (many visible Gaussians per camera) | **High** — more data to transfer, longer $T_p$ and $T_o$ |
| PCIe-bottlenecked (A40, A100-PCIe) | **High** — comm time dominates |
| NVLink systems (A100 SXM, H100 NVL) | **Moderate** — transfers are fast, compute dominates |
| Small scenes (few Gaussians) | **Low** — transfers already negligible |
| bsz=2 (minimum) | **Lower** — fewer micro-batches to pipeline |
| bsz=8+ | **Higher** — more opportunities for prefetch/offload overlap |

---

## 4. Comparison of Methods

| Aspect | Method 1 (P2P) | Method 2 (Overlap) |
|--------|---------------|-------------------|
| **What it reduces** | Total PCIe bytes (avoids redundant loads) | Pipeline stalls (parallelizes transfers) |
| **Mechanism** | GPU↔GPU cooperative loading via NVLink | Dual CUDA streams for concurrent CPU↔GPU |
| **Hardware requirement** | NVLink mandatory | Works on any GPU setup |
| **Applies to** | First micro-batch load (shared Gaussians) | Every micro-batch (prefetch + offload) |
| **Orthogonal?** | Yes — could combine both methods | Yes — could combine both methods |
| **Best scenario** | High overlap ratio + NVLink available | Large transfers + PCIe bandwidth-bound |

### Can They Be Combined?

Yes. Method 1 reduces **how much** PCIe traffic there is. Method 2 hides **the latency** of whatever PCIe traffic remains. A combined approach would use P2P to eliminate redundant loads AND dual-stream to overlap the remaining transfers with computation. This is a potential Method 3.
