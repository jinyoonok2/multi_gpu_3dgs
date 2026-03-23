"""
Overlap Training Strategy Module
===================================

Hooks into the unified CLM engine to add dual-stream overlapping:
a dedicated CUDA stream for gradient offloading (GPU → CPU) enables
three-way overlap of prefetch, compute, and offload.

Overridden hooks
----------------
- ``get_offload_stream``     : Create a dedicated ``offload_stream``.
- ``pre_compute_offload``    : Pre-compute Category G indices during prefetch.
- ``pre_gradient_sync``      : Full device synchronise (both streams).
"""

import torch
from strategies.clm_offload.strategy_base import BaseStrategy


class OverlapStrategy(BaseStrategy):
    """Dual-stream overlapped training — separate stream for grad offloading."""

    def __init__(self, gpu_device=None, **kwargs):
        super().__init__(**kwargs)
        self._gpu_device = gpu_device
        # Create the offload stream ONCE — reusing across iterations prevents
        # record_stream() annotations from becoming dangling references when
        # the CUDA caching allocator reclaims memory after densification.
        self._offload_stream = torch.cuda.Stream(device=gpu_device)

    # ---------- Stage 3: dedicated offload stream ----------
    def get_offload_stream(self, comm_stream, args):
        return self._offload_stream

    # ---------- Stage 4.2: pre-compute Category G indices ----------
    def pre_compute_offload(self, micro_idx, comm_stream,
                             this_bit, next_bit, cnt_g, retention_vec):
        bit_g = this_bit & ~next_bit
        idx_g = torch.nonzero_static(bit_g, size=cnt_g[micro_idx]).flatten()
        host_indices_from_grad = idx_g.to(torch.int32)
        grad_indices_to_host = torch.gather(retention_vec, dim=0, index=idx_g)
        del idx_g, bit_g

        indices_ready_event = torch.cuda.Event()
        indices_ready_event.record(comm_stream)

        return {
            "host_indices_from_grad": host_indices_from_grad,
            "grad_indices_to_host": grad_indices_to_host,
            "indices_ready_event": indices_ready_event,
        }

    # ---------- Stage 4.3: sync offload before forward ----------
    # Diagnostic: ensure previous micro-batch's offload completes before
    # the next forward pass.  If this fixes NaN, there is a latent
    # memory-ordering issue between offload_stream and default_stream.
    def before_forward(self, default_stream):
        if self._offload_stream is not None:
            self._offload_stream.synchronize()

    # ---------- Stage 5.0: full device sync before all-reduce ----------
    def pre_gradient_sync(self):
        torch.cuda.synchronize()
