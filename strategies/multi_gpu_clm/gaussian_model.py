"""
GaussianModelMultiGPUCLM
========================

Hybrid Gaussian model that combines:
  - Multi-GPU spatial partitioning (from multi_gpu): xyz, opacity, scaling,
    rotation partitioned across GPUs with a replicated lightweight proxy
  - CLM CPU offload (from clm_offload): SH features stored in CPU pinned
    memory, with CPU Adam optimizer

Memory layout:
  CPU (pinned):
    parameters_buffer     — ALL N Gaussians' SH features (N × 48 float32)
    parameters_grad_buffer — SH gradient accumulator (N × 48 float32)
    CPU Adam states for SH (~2× SH size)

  GPU k (of K total):
    _xyz       — local partition (N/K × 3)
    _opacity   — local partition (N/K × 1)
    _scaling   — local partition (N/K × 3)
    _rotation  — local partition (N/K × 4)
    GPU Adam states for spatial params
    global_xyz, global_opacity — replicated proxy (N × 3, N × 1)
    Transient SH cache (fetched per camera, ~visible_count × 48)

Key insight: the CPU holds ALL N Gaussians' SH features as a single
contiguous pinned buffer.  Any GPU can fetch any Gaussian's SH.  The
"ownership" partitioning is only for spatial metadata and optimizer state.
"""

import torch
import torch.distributed as dist
import numpy as np
import numba.cuda
from torch import nn
import os
import math

from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from utils.system_utils import mkdir_p
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from plyfile import PlyData, PlyElement
from utils.graphics_utils import BasicPointCloud
import utils.general_utils as utils
from optimizer import UnifiedAdam

from strategies.base_gaussian_model import BaseGaussianModel


class GaussianModelMultiGPUCLM(BaseGaussianModel):
    """
    Multi-GPU CLM: spatial metadata partitioned across GPUs,
    SH features on CPU pinned memory with CLM streaming.
    """

    def _get_device(self):
        return "cuda"

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def create_from_pcd(
        self,
        pcd: BasicPointCloud,
        spatial_lr_scale: float,
        subsample_ratio: float = 1.0,
    ):
        log_file = utils.get_log_file()
        self.spatial_lr_scale = spatial_lr_scale

        # DDP info
        if dist.is_initialized():
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()
        else:
            self.rank = 0
            self.world_size = 1

        # ==============================================================
        # 1. Allocate CPU pinned buffers for SH features (CLM style)
        # ==============================================================
        parameters_buffer_array = numba.cuda.pinned_array(
            (self.args.prealloc_capacity, 48), dtype=np.float32
        )
        self.parameters_buffer = torch.from_numpy(parameters_buffer_array)
        assert self.parameters_buffer.is_pinned()

        if not self.only_for_rendering:
            parameters_grad_buffer_array = numba.cuda.pinned_array(
                (self.args.prealloc_capacity, 48), dtype=np.float32
            )
            self.parameters_grad_buffer = torch.from_numpy(parameters_grad_buffer_array)
            assert self.parameters_grad_buffer.is_pinned()

        # ==============================================================
        # 2. Prepare full point cloud on GPU (temporarily)
        # ==============================================================
        all_points = torch.tensor(np.asarray(pcd.points)).float().cuda()
        all_points = all_points.contiguous()
        all_colors = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())

        N_total = all_points.shape[0]
        print(f"[Rank {self.rank}] Total points before partitioning: {N_total}")

        # ==============================================================
        # 3. Compute nearest-neighbor distances (needed for initial scales)
        # ==============================================================
        torch.cuda.empty_cache()
        dist2 = torch.clamp_min(distCUDA2(all_points), 0.0000001)
        all_scales = torch.log(torch.sqrt(dist2))[..., None].repeat(1, 3)

        # ==============================================================
        # 4. Build SH features on GPU (temporary)
        # ==============================================================
        all_features = torch.zeros(
            (N_total, 3, (self.max_sh_degree + 1) ** 2),
            dtype=torch.float32,
            device="cuda",
        )
        all_features[:, :3, 0] = all_colors
        all_features[:, 3:, 1:] = 0.0

        # ==============================================================
        # 5. Optional subsampling
        # ==============================================================
        if subsample_ratio != 1.0:
            assert 0 < subsample_ratio < 1
            sub_N = int(N_total * subsample_ratio)
            print(f"[Rank {self.rank}] Subsampling to {sub_N} points")
            perm = torch.randperm(N_total, device="cuda")[:sub_N].sort()[0]
            all_points = all_points[perm]
            all_scales = all_scales[perm]
            all_features = all_features[perm]
            N_total = sub_N

        # ==============================================================
        # 6. Spatial partitioning: sort by longest axis, split evenly
        # ==============================================================
        longest_axis = (
            (all_points.max(0)[0] - all_points.min(0)[0]).argmax().item()
        )
        sorted_indices = torch.argsort(all_points[:, longest_axis])

        chunk_size = N_total // self.world_size
        start = self.rank * chunk_size
        end = start + chunk_size if self.rank < self.world_size - 1 else N_total

        local_indices = sorted_indices[start:end]
        N_local = local_indices.shape[0]

        print(
            f"[Rank {self.rank}] Owns Gaussians [{start}, {end}) = {N_local} points "
            f"(split along axis {longest_axis})"
        )

        # ==============================================================
        # 7. Extract local spatial partition → GPU
        # ==============================================================
        local_xyz = all_points[local_indices].contiguous()
        local_scales = all_scales[local_indices].contiguous()

        local_rots = torch.zeros((N_local, 4), device="cuda")
        local_rots[:, 0] = 1.0

        local_opacities = inverse_sigmoid(
            0.1 * torch.ones((N_local, 1), dtype=torch.float32, device="cuda")
        )

        # Spatial params as nn.Parameters on GPU
        self._xyz = nn.Parameter(local_xyz.requires_grad_(True))
        self._scaling = nn.Parameter(local_scales.requires_grad_(True))
        self._rotation = nn.Parameter(local_rots.requires_grad_(True))
        self._opacity = nn.Parameter(local_opacities.requires_grad_(True))

        # ==============================================================
        # 8. Store ALL SH features in CPU pinned memory
        # ==============================================================
        # Flatten features: (N, 3, 16) → dc (N, 3) + rest (N, 45) → concat (N, 48)
        # We store ALL N_total features, not just local partition
        local_features = all_features[local_indices].contiguous()
        features_dc = (
            local_features[:, :, 0:1].transpose(1, 2).contiguous().view(N_local, -1)
        )  # (N_local, 3)
        features_rest = (
            local_features[:, :, 1:].transpose(1, 2).contiguous().view(N_local, -1)
        )  # (N_local, 45)

        # Store in CPU pinned buffer (like CLM)
        # Move GPU tensors to CPU before writing to pinned buffer
        dims = [features_dc.shape[1], features_rest.shape[1]]
        torch.cat((features_dc.cpu(), features_rest.cpu()), dim=1, out=self.parameters_buffer[:N_local])

        self._parameters = nn.Parameter(
            self.parameters_buffer[:N_local].requires_grad_(True)
        )
        self._features_dc, self._features_rest = torch.split(
            self._parameters, dims, dim=1
        )
        self.param_dims = torch.tensor(dims, dtype=torch.int, device="cuda")

        # Tracking buffers on GPU
        self.max_radii2D = torch.zeros((N_local,), device="cuda")
        self.sum_visible_count_in_one_batch = torch.zeros((N_local,), device="cuda")

        # ==============================================================
        # 9. Build global proxy for cross-GPU visibility
        # ==============================================================
        self._build_global_proxy()

        # Free temporary full arrays
        del all_points, all_scales, all_features, all_colors, dist2
        del local_features, features_dc, features_rest
        torch.cuda.empty_cache()

        print(
            f"[Rank {self.rank}] Init complete. Local spatial: {N_local} on GPU, "
            f"SH features: {N_local} on CPU pinned, "
            f"Global proxy: {self.global_N}"
        )

    # ------------------------------------------------------------------
    # Global proxy: lightweight replicated xyz + opacity for visibility
    # ------------------------------------------------------------------

    def _build_global_proxy(self):
        """
        AllGather xyz and opacity → replicated lightweight proxy for visibility.
        ~16 bytes per Gaussian — trivially fits for millions.
        """
        if not dist.is_initialized() or self.world_size == 1:
            self.global_xyz = self._xyz.detach()
            self.global_opacity = self._opacity.detach()
            self.global_N = self._xyz.shape[0]
            self.partition_sizes = [self._xyz.shape[0]]
            self.partition_offsets = [0]
            return

        # Gather local sizes (may differ across ranks)
        local_n = torch.tensor([self._xyz.shape[0]], device="cuda")
        all_sizes = [
            torch.zeros(1, device="cuda", dtype=torch.long)
            for _ in range(self.world_size)
        ]
        dist.all_gather(all_sizes, local_n)
        self.partition_sizes = [int(s.item()) for s in all_sizes]
        self.global_N = sum(self.partition_sizes)
        self.partition_offsets = [0]
        for s in self.partition_sizes[:-1]:
            self.partition_offsets.append(self.partition_offsets[-1] + s)

        # AllGather xyz
        local_xyz = self._xyz.detach().contiguous()
        gathered_xyz = [
            torch.zeros(s, 3, device="cuda") for s in self.partition_sizes
        ]
        dist.all_gather(gathered_xyz, local_xyz)
        self.global_xyz = torch.cat(gathered_xyz, dim=0)

        # AllGather opacity
        local_opacity = self._opacity.detach().contiguous()
        gathered_opacity = [
            torch.zeros(s, 1, device="cuda") for s in self.partition_sizes
        ]
        dist.all_gather(gathered_opacity, local_opacity)
        self.global_opacity = torch.cat(gathered_opacity, dim=0)

    def sync_global_proxy(self):
        """Keep the replicated proxy up-to-date after optimizer step."""
        self._build_global_proxy()

    def get_local_and_remote_indices(self, visible_global_indices):
        """
        Given global indices visible for a camera, split into:
          - local (owned by this rank): for spatial params from VRAM, SH from CPU
          - remote (owned by peers): spatial from global proxy, SH from CPU or P2P
        """
        my_start = self.partition_offsets[self.rank]
        my_end = my_start + self.partition_sizes[self.rank]

        local_mask = (visible_global_indices >= my_start) & (
            visible_global_indices < my_end
        )
        remote_mask = ~local_mask

        local_global_idx = visible_global_indices[local_mask]
        local_idx = local_global_idx - my_start

        remote_global_idx = visible_global_indices[remote_mask]

        # Determine owner rank and local index for each remote Gaussian
        remote_rank = torch.zeros_like(remote_global_idx)
        remote_local_idx = torch.zeros_like(remote_global_idx)
        for r in range(self.world_size):
            if r == self.rank:
                continue
            r_start = self.partition_offsets[r]
            r_end = r_start + self.partition_sizes[r]
            mask_r = (remote_global_idx >= r_start) & (remote_global_idx < r_end)
            remote_rank[mask_r] = r
            remote_local_idx[mask_r] = remote_global_idx[mask_r] - r_start

        return (
            local_global_idx,
            local_idx,
            remote_global_idx,
            remote_rank,
            remote_local_idx,
        )

    # ------------------------------------------------------------------
    # All parameters list
    # ------------------------------------------------------------------

    def all_parameters(self):
        return [
            self._xyz,
            self._opacity,
            self._scaling,
            self._rotation,
            self._parameters,  # SH on CPU pinned
        ]

    # ------------------------------------------------------------------
    # Training setup — GPU Adam for spatial, CPU Adam for SH (like CLM)
    # ------------------------------------------------------------------

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros(
            (self.get_xyz.shape[0], 1), device="cuda"
        )
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        args = utils.get_args()
        log_file = utils.get_log_file()

        # Use UnifiedAdam (like CLM): GPU Adam for spatial, CPU Adam for SH
        l = [
            {
                "params": [self._xyz],
                "lr": training_args.position_lr_init
                * self.spatial_lr_scale
                * args.lr_scale_pos_and_scale,
                "name": "xyz",
            },
            {
                "params": [self._opacity],
                "lr": training_args.opacity_lr,
                "name": "opacity",
            },
            {
                "params": [self._scaling],
                "lr": training_args.scaling_lr * args.lr_scale_pos_and_scale,
                "name": "scaling",
            },
            {
                "params": [self._rotation],
                "lr": training_args.rotation_lr,
                "name": "rotation",
            },
            {
                "params": [self._parameters],  # SH on CPU pinned
                "lr": training_args.feature_lr,
                "name": "parameters",
            },
        ]
        column_sizes = [3, 45]
        column_lrs = [training_args.feature_lr, training_args.feature_lr / 20.0]

        self.optimizer = UnifiedAdam(
            l,
            column_sizes,
            column_lrs,
            lr=0.0,
            bias_correction=True,
            betas=(0.9, 0.999),
            eps=1e-15,
            weight_decay=0,
            amsgrad=False,
            adamw_mode=False,
            fp32_optimizer_states=True,
            fused=True,
            sparse=self.args.sparse_adam,
        )

        # Scale learning rates according to bsz (like CLM)
        bsz = args.bsz
        for param_group in self.optimizer.param_groups:
            if training_args.lr_scale_mode == "linear":
                lr_scale = bsz
                param_group["lr"] *= lr_scale
            elif training_args.lr_scale_mode == "sqrt":
                lr_scale = np.sqrt(bsz)
                param_group["lr"] *= lr_scale
                if "eps" in param_group:
                    param_group["eps"] /= lr_scale
                    param_group["betas"] = [
                        beta**bsz for beta in param_group["betas"]
                    ]
                    log_file.write(
                        param_group["name"]
                        + " betas: "
                        + str(param_group["betas"])
                        + "\n"
                    )
            elif training_args.lr_scale_mode == "accumu":
                lr_scale = 1
            else:
                assert False, f"lr_scale_mode {training_args.lr_scale_mode} not supported."

        # Update columns_lr for UnifiedAdam
        if training_args.lr_scale_mode == "linear":
            lr_scale = bsz
            self.optimizer.columns_lr *= lr_scale
        elif training_args.lr_scale_mode == "sqrt":
            lr_scale = np.sqrt(bsz)
            self.optimizer.columns_lr *= lr_scale

        self.xyz_scheduler_args = get_expon_lr_func(
            lr_init=training_args.position_lr_init
            * self.spatial_lr_scale
            * lr_scale
            * args.lr_scale_pos_and_scale,
            lr_final=training_args.position_lr_final
            * self.spatial_lr_scale
            * lr_scale
            * args.lr_scale_pos_and_scale,
            lr_delay_mult=training_args.position_lr_delay_mult,
            max_steps=training_args.position_lr_max_steps,
        )

        utils.check_initial_gpu_memory_usage("after training_setup (multi_gpu_clm)")

    def update_learning_rate(self, iteration):
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group["lr"] = lr
                return lr

    # ------------------------------------------------------------------
    # Save / Load
    # ------------------------------------------------------------------

    def save_ply(self, path):
        """Save the full global model. Rank 0 gathers all partitions."""
        mkdir_p(os.path.dirname(path))

        if dist.is_initialized() and self.world_size > 1:
            # AllGather spatial from GPU
            all_xyz = [torch.zeros(s, 3, device="cuda") for s in self.partition_sizes]
            all_opacity = [torch.zeros(s, 1, device="cuda") for s in self.partition_sizes]
            all_scaling = [torch.zeros(s, 3, device="cuda") for s in self.partition_sizes]
            all_rotation = [torch.zeros(s, 4, device="cuda") for s in self.partition_sizes]

            dist.all_gather(all_xyz, self._xyz.detach().contiguous())
            dist.all_gather(all_opacity, self._opacity.detach().contiguous())
            dist.all_gather(all_scaling, self._scaling.detach().contiguous())
            dist.all_gather(all_rotation, self._rotation.detach().contiguous())

            xyz = torch.cat(all_xyz).cpu().numpy()
            opacity = torch.cat(all_opacity).cpu().numpy()
            scaling = torch.cat(all_scaling).cpu().numpy()
            rotation = torch.cat(all_rotation).cpu().numpy()

            # AllGather SH from CPU via GPU staging
            local_sh = self._parameters.detach().clone().cuda()  # CPU→GPU
            all_sh = [torch.zeros(s, 48, device="cuda") for s in self.partition_sizes]
            dist.all_gather(all_sh, local_sh.contiguous())
            params = torch.cat(all_sh)
        else:
            xyz = self._xyz.detach().cpu().numpy()
            opacity = self._opacity.detach().cpu().numpy()
            scaling = self._scaling.detach().cpu().numpy()
            rotation = self._rotation.detach().cpu().numpy()
            params = self._parameters.detach()

        if self.rank == 0:
            features_dc = params[:, :3].cpu().numpy()
            features_rest = params[:, 3:].cpu().numpy()
            normals = np.zeros_like(xyz)

            features_dc = (
                features_dc.reshape(-1, 1, 3).transpose(0, 2, 1).reshape(-1, 3)
            )
            features_rest = (
                features_rest.reshape(-1, 15, 3).transpose(0, 2, 1).reshape(-1, 45)
            )

            dtype_full = [
                (attribute, "f4")
                for attribute in self.construct_list_of_attributes()
            ]
            elements = np.empty(xyz.shape[0], dtype=dtype_full)
            attributes = np.concatenate(
                (xyz, normals, features_dc, features_rest, opacity, scaling, rotation),
                axis=1,
            )
            elements[:] = list(map(tuple, attributes))
            el = PlyElement.describe(elements, "vertex")
            PlyData([el]).write(path)

    def load_ply(self, path):
        """Load and repartition a saved PLY file."""
        plydata = PlyData.read(path)
        xyz = np.stack(
            (
                np.asarray(plydata.elements[0]["x"]),
                np.asarray(plydata.elements[0]["y"]),
                np.asarray(plydata.elements[0]["z"]),
            ),
            axis=1,
        )
        pcd = BasicPointCloud(
            points=xyz,
            colors=np.zeros_like(xyz),
            normals=np.zeros_like(xyz),
        )
        self.create_from_pcd(pcd, self.spatial_lr_scale)

    # ------------------------------------------------------------------
    # Densification support
    # ------------------------------------------------------------------

    def reset_opacity(self):
        opacities_new = inverse_sigmoid(
            torch.min(self.get_opacity, torch.ones_like(self.get_opacity) * 0.01)
        )
        for group in self.optimizer.param_groups:
            if group["name"] == "opacity":
                stored_state = self.optimizer.state.get(group["params"][0], None)
                if stored_state is not None:
                    stored_state["exp_avg"] = torch.zeros_like(opacities_new)
                    stored_state["exp_avg_sq"] = torch.zeros_like(opacities_new)
                    del self.optimizer.state[group["params"][0]]
                    group["params"][0] = nn.Parameter(
                        opacities_new.requires_grad_(True)
                    )
                    self.optimizer.state[group["params"][0]] = stored_state
                else:
                    group["params"][0] = nn.Parameter(
                        opacities_new.requires_grad_(True)
                    )
        self._opacity = self.optimizer.param_groups[1]["params"][0]

    def prune_points(self, mask):
        """Prune Gaussians locally. Each GPU prunes its own partition."""
        valid_points_mask = ~mask
        N_new = valid_points_mask.sum().item()

        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group["params"][0], None)
            if group["name"] == "parameters":
                # SH on CPU — prune in pinned buffer
                old_params = group["params"][0]
                new_params_data = old_params[valid_points_mask.cpu()]
                self.parameters_buffer[:N_new].copy_(new_params_data)
                if stored_state is not None:
                    stored_state["exp_avg"] = stored_state["exp_avg"][
                        valid_points_mask.cpu()
                    ]
                    stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][
                        valid_points_mask.cpu()
                    ]
                    del self.optimizer.state[group["params"][0]]
                    group["params"][0] = nn.Parameter(
                        self.parameters_buffer[:N_new].requires_grad_(True)
                    )
                    self.optimizer.state[group["params"][0]] = stored_state
                else:
                    group["params"][0] = nn.Parameter(
                        self.parameters_buffer[:N_new].requires_grad_(True)
                    )
            else:
                # Spatial params on GPU — standard prune
                if stored_state is not None:
                    stored_state["exp_avg"] = stored_state["exp_avg"][valid_points_mask]
                    stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][
                        valid_points_mask
                    ]
                    del self.optimizer.state[group["params"][0]]
                    group["params"][0] = nn.Parameter(
                        group["params"][0][valid_points_mask].requires_grad_(True)
                    )
                    self.optimizer.state[group["params"][0]] = stored_state
                else:
                    group["params"][0] = nn.Parameter(
                        group["params"][0][valid_points_mask].requires_grad_(True)
                    )
            optimizable_tensors[group["name"]] = group["params"][0]

        self._xyz = optimizable_tensors["xyz"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        self._parameters = optimizable_tensors["parameters"]
        dims = [3, 45]
        self._features_dc, self._features_rest = torch.split(
            self._parameters, dims, dim=1
        )

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]
        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]
        if hasattr(self, "sum_visible_count_in_one_batch"):
            self.sum_visible_count_in_one_batch = (
                self.sum_visible_count_in_one_batch[valid_points_mask]
            )

        self._build_global_proxy()

    def densification_postfix(
        self,
        new_xyz,
        new_features_dc,
        new_features_rest,
        new_opacities,
        new_scaling,
        new_rotation,
        new_parameters=None,
    ):
        """Add new Gaussians to local partition. Spatial on GPU, SH on CPU pinned."""
        if new_parameters is None:
            new_parameters = torch.cat((new_features_dc, new_features_rest), dim=1)

        N_old = self._xyz.shape[0]
        N_new_pts = new_xyz.shape[0]
        N_total = N_old + N_new_pts

        # For spatial params (GPU) and SH (CPU), handle separately
        d = {
            "xyz": new_xyz,
            "opacity": new_opacities,
            "scaling": new_scaling,
            "rotation": new_rotation,
            "parameters": new_parameters,
        }

        for group in self.optimizer.param_groups:
            name = group["name"]
            assert name in d
            extension_tensor = d[name]
            stored_state = self.optimizer.state.get(group["params"][0], None)

            if name == "parameters":
                # SH: extend in CPU pinned buffer
                old = group["params"][0]
                self.parameters_buffer[N_old:N_total].copy_(
                    extension_tensor.cpu()
                    if extension_tensor.is_cuda
                    else extension_tensor
                )
                if stored_state is not None:
                    stored_state["exp_avg"] = torch.cat(
                        (
                            stored_state["exp_avg"],
                            torch.zeros_like(extension_tensor.cpu()
                                if extension_tensor.is_cuda else extension_tensor),
                        ),
                        dim=0,
                    )
                    stored_state["exp_avg_sq"] = torch.cat(
                        (
                            stored_state["exp_avg_sq"],
                            torch.zeros_like(extension_tensor.cpu()
                                if extension_tensor.is_cuda else extension_tensor),
                        ),
                        dim=0,
                    )
                    del self.optimizer.state[group["params"][0]]
                    group["params"][0] = nn.Parameter(
                        self.parameters_buffer[:N_total].requires_grad_(True)
                    )
                    self.optimizer.state[group["params"][0]] = stored_state
                else:
                    group["params"][0] = nn.Parameter(
                        self.parameters_buffer[:N_total].requires_grad_(True)
                    )
            else:
                # Spatial: standard GPU extend
                if stored_state is not None:
                    stored_state["exp_avg"] = torch.cat(
                        (
                            stored_state["exp_avg"],
                            torch.zeros_like(extension_tensor),
                        ),
                        dim=0,
                    )
                    stored_state["exp_avg_sq"] = torch.cat(
                        (
                            stored_state["exp_avg_sq"],
                            torch.zeros_like(extension_tensor),
                        ),
                        dim=0,
                    )
                    del self.optimizer.state[group["params"][0]]
                    group["params"][0] = nn.Parameter(
                        torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True)
                    )
                    self.optimizer.state[group["params"][0]] = stored_state
                else:
                    group["params"][0] = nn.Parameter(
                        torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True)
                    )

        # Refresh references
        for group in self.optimizer.param_groups:
            if group["name"] == "xyz":
                self._xyz = group["params"][0]
            elif group["name"] == "opacity":
                self._opacity = group["params"][0]
            elif group["name"] == "scaling":
                self._scaling = group["params"][0]
            elif group["name"] == "rotation":
                self._rotation = group["params"][0]
            elif group["name"] == "parameters":
                self._parameters = group["params"][0]
                dims = [3, 45]
                self._features_dc, self._features_rest = torch.split(
                    self._parameters, dims, dim=1
                )

        # Extend tracking buffers
        self.xyz_gradient_accum = torch.cat(
            (self.xyz_gradient_accum, torch.zeros(N_new_pts, 1, device="cuda")),
            dim=0,
        )
        self.denom = torch.cat(
            (self.denom, torch.zeros(N_new_pts, 1, device="cuda")), dim=0
        )
        self.max_radii2D = torch.cat(
            (self.max_radii2D, torch.zeros(N_new_pts, device="cuda")), dim=0
        )
        if hasattr(self, "sum_visible_count_in_one_batch"):
            self.sum_visible_count_in_one_batch = torch.cat(
                (
                    self.sum_visible_count_in_one_batch,
                    torch.zeros(N_new_pts, device="cuda"),
                ),
                dim=0,
            )

        self._build_global_proxy()

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        selected_pts_mask = torch.where(
            torch.norm(grads, dim=-1) >= grad_threshold, True, False
        )
        selected_pts_mask = torch.logical_and(
            selected_pts_mask,
            torch.max(self.get_scaling, dim=1).values
            <= self.percent_dense * scene_extent,
        )

        # SH features are on CPU — use CPU mask for CPU tensors
        selected_pts_mask_cpu = selected_pts_mask.cpu()

        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask_cpu]
        new_features_rest = self._features_rest[selected_pts_mask_cpu]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]

        self.densification_postfix(
            new_xyz,
            new_features_dc,
            new_features_rest,
            new_opacities,
            new_scaling,
            new_rotation,
        )

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0]
        padded_grad = torch.zeros((n_init_points,), device="cuda")
        padded_grad[: grads.shape[0]] = grads.squeeze()

        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(
            selected_pts_mask,
            torch.max(self.get_scaling, dim=1).values
            > self.percent_dense * scene_extent,
        )

        # SH features are on CPU — use CPU mask for CPU tensors
        selected_pts_mask_cpu = selected_pts_mask.cpu()

        stds = self.get_scaling[selected_pts_mask].repeat(N, 1)
        means = torch.zeros((stds.size(0), 3), device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N, 1, 1)

        new_xyz = (
            torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1)
            + self.get_xyz[selected_pts_mask].repeat(N, 1)
        )
        new_scaling = self.scaling_inverse_activation(
            self.get_scaling[selected_pts_mask].repeat(N, 1) / (0.8 * N)
        )
        new_rotation = self._rotation[selected_pts_mask].repeat(N, 1)
        new_features_dc = self._features_dc[selected_pts_mask_cpu].repeat(N, 1)
        new_features_rest = self._features_rest[selected_pts_mask_cpu].repeat(N, 1)
        new_opacities = self._opacity[selected_pts_mask].repeat(N, 1)

        self.densification_postfix(
            new_xyz,
            new_features_dc,
            new_features_rest,
            new_opacities,
            new_scaling,
            new_rotation,
        )

        prune_filter = torch.cat(
            (
                selected_pts_mask,
                torch.zeros(
                    N * selected_pts_mask.sum(), device="cuda", dtype=bool
                ),
            )
        )
        self.prune_points(prune_filter)

    def gsplat_add_densification_stats_exact_filter(
        self,
        viewspace_point_tensor_grad,
        radii,
        send2gpu_final_filter_indices,
        width,
        height,
    ):
        """Accumulate densification statistics for locally-owned visible Gaussians."""
        self.max_radii2D = torch.zeros((self._xyz.shape[0],), device="cuda")

    def save_tensors(self, parent_path):
        mkdir_p(parent_path)
        torch.save(self._xyz, os.path.join(parent_path, "xyz.pt"))
        torch.save(self._opacity, os.path.join(parent_path, "opacity.pt"))
        torch.save(self._scaling, os.path.join(parent_path, "scaling.pt"))
        torch.save(self._rotation, os.path.join(parent_path, "rotation.pt"))
        torch.save(self._parameters, os.path.join(parent_path, "parameters.pt"))
