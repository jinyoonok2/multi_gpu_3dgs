"""
Base Training Strategy for the Unified CLM Engine
===================================================

The unified ``engine_multi.clm_offload_train_one_batch()`` calls hook
methods on a *strategy* object at well-defined points in the pipeline.
The base class provides no-op / default implementations so that
**baseline behaviour is unchanged** when no strategy is supplied.

Subclass and override only the hooks your strategy needs.

Hook points (execution order)
-----------------------------
1. ``post_filters``           — after filter + ordering             (Stage 1)
2. ``get_offload_stream``     — return CUDA stream for grad offload (Stage 3)
3. ``load_first_shs``         — load SH for micro_idx=0            (Stage 4.1)
4. ``pre_compute_offload``    — after H/D indices in prefetch       (Stage 4.2)
5. ``before_forward``         — before forward pass                 (Stage 4.3)
6. ``pre_gradient_sync``      — before all-reduce                   (Stage 5.0)
7. ``sync_gradients``         — perform gradient sync               (Stage 5.0)
"""

import torch
from clm_kernels import send_shs2gpu_stream


class BaseStrategy:
    """Default (baseline) training strategy — all hooks are no-ops."""

    def __init__(self, **kwargs):
        pass  # Accept and ignore keyword args for MRO chaining

    # ------------------------------------------------------------------
    # Stage 1: After filter calculation + camera ordering
    # ------------------------------------------------------------------
    def post_filters(self, args, filters, n_gaussians, iteration, log_file):
        """Called after ``calculate_filters`` and ``order_calculation``.

        Returns
        -------
        dict
            Extra state carried through the training step (available in
            later hooks via *extra_state*).
        """
        return {}

    # ------------------------------------------------------------------
    # Stage 3: CUDA stream for SH gradient offloading
    # ------------------------------------------------------------------
    def get_offload_stream(self, comm_stream, args):
        """Return the CUDA stream used for SH gradient offloading.

        Default: reuses *comm_stream* (same as baseline).
        Override to create a dedicated stream for three-way overlap.
        """
        return comm_stream

    # ------------------------------------------------------------------
    # Stage 4.1: Load SH coefficients for first micro-batch
    # ------------------------------------------------------------------
    def load_first_shs(self, shs, gaussians_params, local_filter,
                        n_gaussians, grid_size, block_size,
                        extra_state, args):
        """Load SH coefficients for the first micro-batch (micro_idx=0).

        Default: standard CPU → GPU transfer via ``send_shs2gpu_stream``.
        Override for cooperative GPU-GPU loading (P2P).
        """
        send_shs2gpu_stream(
            shs, gaussians_params, local_filter, grid_size, block_size,
        )

    # ------------------------------------------------------------------
    # Stage 4.2: Pre-compute offload indices during prefetch
    # ------------------------------------------------------------------
    def pre_compute_offload(self, micro_idx, comm_stream,
                             this_bit, next_bit, cnt_g, retention_vec):
        """Called in Stage 4.2 after H/D indices are computed on *comm_stream*.

        Override to pre-compute Category G indices early, enabling the
        offload stream to start as soon as backward + indices are both ready.

        Returns
        -------
        dict
            May contain ``host_indices_from_grad``, ``grad_indices_to_host``,
            and ``indices_ready_event`` when indices are pre-computed.
            Empty dict means Stage 4.6 will compute G indices inline.
        """
        return {}

    # ------------------------------------------------------------------
    # Stage 4.3: Before forward pass
    # ------------------------------------------------------------------
    def before_forward(self, default_stream):
        """Called just before the forward pass.

        Override to add stream synchronisation (e.g. wait for *offload_stream*
        to finish before running compute kernels on *default_stream*).
        """
        pass

    # ------------------------------------------------------------------
    # Stage 5.0: Before gradient all-reduce
    # ------------------------------------------------------------------
    def pre_gradient_sync(self):
        """Called before the gradient all-reduce.

        Override to add extra synchronisation (e.g. full device sync when
        multiple streams are in flight).
        """
        pass

    # ------------------------------------------------------------------
    # Stage 5.0: Gradient synchronisation across GPUs
    # ------------------------------------------------------------------
    def sync_gradients(self, gaussians, parameters_grad_buffer, N, args):
        """Synchronise gradients across GPUs.

        Returns
        -------
        bool
            ``True``  — this method performed the full sync; skip default.
            ``False`` — use the default ``all_reduce_coalesced`` logic.
        """
        return False
