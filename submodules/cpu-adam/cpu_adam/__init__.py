# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import torch
from . import _C

class CPUAdam(torch.optim.Optimizer):
    optimizer_id = 0

    def __init__(self,
                 model_params,
                 lr=1e-3,
                 bias_correction=True,
                 betas=(0.9, 0.999),
                 eps=1e-8,
                 weight_decay=0,
                 amsgrad=False,
                 adamw_mode=False,
                 fp32_optimizer_states=True):
        """Fast vectorized implementation of two variations of Adam optimizer on CPU:

        * Adam: A Method for Stochastic Optimization: (https://arxiv.org/abs/1412.6980);
        * AdamW: Fixing Weight Decay Regularization in Adam (https://arxiv.org/abs/1711.05101)

        DeepSpeed CPU Adam(W) provides between 5x to 7x speedup over torch.optim.adam(W).
        In order to apply this optimizer, the model requires to have its master parameter (in FP32)
        reside on the CPU memory.

        To train on a heterogeneous system, such as coordinating CPU and GPU, DeepSpeed offers
        the ZeRO-Offload technology which efficiently offloads the optimizer states into CPU memory,
        with minimal impact on training throughput. CPUAdam plays an important role to minimize
        the overhead of the optimizer's latency on CPU. Please refer to ZeRO-Offload tutorial
        (https://www.deepspeed.ai/tutorials/zero-offload/) for more information on how to enable this technology.

        For calling step function, there are two options available: (1) update optimizer's states and (2) update
        optimizer's states and copy the parameters back to GPU at the same time. We have seen that the second
        option can bring 30% higher throughput than the doing the copy separately using option one.


        .. note::
                We recommend using our `config
                <https://www.deepspeed.ai/docs/config-json/#optimizer-parameters>`_
                to allow :meth:`deepspeed.initialize` to build this optimizer
                for you.


        Arguments:
            model_params (iterable): iterable of parameters to optimize or dicts defining
                parameter groups.
            lr (float, optional): learning rate. (default: 1e-3)
            betas (Tuple[float, float], optional): coefficients used for computing
                running averages of gradient and its square. (default: (0.9, 0.999))
            eps (float, optional): term added to the denominator to improve
                numerical stability. (default: 1e-8)
            weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
            amsgrad (boolean, optional): whether to use the AMSGrad variant of this
                algorithm from the paper `On the Convergence of Adam and Beyond`_
                (default: False) NOT SUPPORTED in DeepSpeed CPUAdam!
            adamw_mode: select between Adam and AdamW implementations (default: AdamW)
            fp32_optimizer_states: creates momentum and variance in full precision regardless of
                        the precision of the parameters (default: True)
        """

        default_args = dict(lr=lr,
                            betas=betas,
                            eps=eps,
                            weight_decay=weight_decay,
                            bias_correction=bias_correction,
                            amsgrad=amsgrad)
        super(CPUAdam, self).__init__(model_params, default_args)

        # cpu_info = get_cpu_info()
        # self.cpu_vendor = cpu_info["vendor_id_raw"].lower() if "vendor_id_raw" in cpu_info else "unknown"
        # if "amd" in self.cpu_vendor:
        #     for group_id, group in enumerate(self.param_groups):
        #         for param_id, p in enumerate(group['params']):
        #             if p.dtype == torch.half:
        #                 logger.warning("FP16 params for CPUAdam may not work on AMD CPUs")
        #                 break
        #         else:
        #             continue
        #         break

        self.opt_id = CPUAdam.optimizer_id
        CPUAdam.optimizer_id = CPUAdam.optimizer_id + 1
        self.adam_w_mode = adamw_mode
        self.fp32_optimizer_states = fp32_optimizer_states
        

        _C.create_adam(self.opt_id, lr, betas[0], betas[1], eps, weight_decay, adamw_mode, False)

    def __del__(self):
        # need to destroy the C++ object explicitly to avoid a memory leak when deepspeed.initialize
        # is used multiple times in the same process (notebook or pytest worker)
        _C.destroy_adam(self.opt_id)

    def __setstate__(self, state):
        super(CPUAdam, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)

    @torch.no_grad()
    def step(self, closure=None):
        """Update the model parameters.

        .. note::
            This method will be called internally by ZeRO-Offload. DeepSpeed
            users should still use ``engine.step()`` as shown in the
            `Getting Started
            <https://www.deepspeed.ai/getting-started/#training>`_ guide.

        Args:
            closure (callable, optional): closure to compute the loss.
                Defaults to ``None``.

        Returns:
            loss: if ``closure`` is provided. Otherwise ``None``.
        """

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # intended device for step
        device = torch.device('cpu')

        for group_id, group in enumerate(self.param_groups):
            for param_id, p in enumerate(group['params']):

                if p.grad is None:
                    continue

                assert p.device == device, f"CPUAdam param is on {p.device} and must be 'cpu', make " \
                        "sure you enabled 'offload_optimizer': 'cpu' in your ZeRO config."

                state = self.state[p]
                # State initialization
                if len(state) == 0:
                    #print(f'group {group_id} param {param_id} = {p.numel()}')
                    state['step'] = 0

                    #use full precision by default unless self.fp32_optimizer_states is off
                    state_dtype = torch.float if self.fp32_optimizer_states else p.dtype

                    # gradient momentums
                    state['exp_avg'] = torch.zeros_like(p.data, dtype=state_dtype, device=device)
                    #memory_format=torch.preserve_format)
                    # gradient variances
                    state['exp_avg_sq'] = torch.zeros_like(p.data, dtype=state_dtype, device=device)
                    #memory_format=torch.preserve_format)

                state['step'] += 1
                beta1, beta2 = group['betas']

                _C.adam_update(self.opt_id, state['step'], group['lr'], beta1, beta2, group['eps'],
                                             group['weight_decay'], group['bias_correction'], p.data, p.grad.data,
                                             state['exp_avg'], state['exp_avg_sq'])
        return loss
    
    @torch.no_grad()
    def sparse_step(self, closure=None, sparse_indices=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        assert sparse_indices is not None, "sparse_indices must be provided for sparse_step"

        # intended device for step
        device = torch.device('cpu')

        for group_id, group in enumerate(self.param_groups):
            for param_id, p in enumerate(group['params']):

                if p.grad is None:
                    continue

                assert p.device == device, f"FusedCPUAdam param is on {p.device} and must be 'cpu', make " \
                        "sure you enabled 'offload_optimizer': 'cpu' in your ZeRO config."

                state = self.state[p]
                # State initialization
                if len(state) == 0:
                    #print(f'group {group_id} param {param_id} = {p.numel()}')
                    state['step'] = 0

                    #use full precision by default unless self.fp32_optimizer_states is off
                    state_dtype = torch.float if self.fp32_optimizer_states else p.dtype

                    # gradient momentums
                    state['exp_avg'] = torch.zeros_like(p.data, dtype=state_dtype, device=device)
                    #memory_format=torch.preserve_format)
                    # gradient variances
                    state['exp_avg_sq'] = torch.zeros_like(p.data, dtype=state_dtype, device=device)
                    #memory_format=torch.preserve_format)

                state['step'] += 1
                beta1, beta2 = group['betas']

                _C.sparse_adam_update(self.opt_id, state['step'], group['lr'], beta1, beta2, group['eps'],
                                      group['weight_decay'], group['bias_correction'], p.data, p.grad.data,
                                      state['exp_avg'], state['exp_avg_sq'],
                                      sparse_indices)

        return loss

class FusedCPUAdam(torch.optim.Optimizer):
    optimizer_id = 0

    def __init__(self,
                 model_params, # (N, sum(columns_sizes))
                 columns_sizes, # (# of column types,)
                 columns_lr, # (# of column types,)
                 lr=1e-3,
                 bias_correction=True,
                 betas=(0.9, 0.999),
                 eps=1e-8,
                 weight_decay=0,
                 amsgrad=False,
                 adamw_mode=False,
                 fp32_optimizer_states=True):

        default_args = dict(lr=lr,
                            betas=betas,
                            eps=eps,
                            weight_decay=weight_decay,
                            bias_correction=bias_correction,
                            amsgrad=amsgrad)
        super(FusedCPUAdam, self).__init__(model_params, default_args)

        self.opt_id = FusedCPUAdam.optimizer_id
        FusedCPUAdam.optimizer_id = FusedCPUAdam.optimizer_id + 1

        assert len(model_params) == 1, "FusedCPUAdam expects a single parameter group"
        assert adamw_mode == False, "FusedCPUAdam does not support AdamW mode"
        self.adam_w_mode = adamw_mode
        self.fp32_optimizer_states = fp32_optimizer_states

        self.columns_n_total = sum(columns_sizes)
        self.columns_sizes = torch.tensor(columns_sizes, dtype=torch.int32)
        self.columns_offsets = torch.cumsum(self.columns_sizes, dim=0, dtype=torch.int32)
        self.columns_lr = torch.tensor(columns_lr, dtype=torch.float32)
        
        _C.create_adam(self.opt_id, lr, betas[0], betas[1], eps, weight_decay, adamw_mode, False)

    def __del__(self):
        # need to destroy the C++ object explicitly to avoid a memory leak when deepspeed.initialize
        # is used multiple times in the same process (notebook or pytest worker)
        _C.destroy_adam(self.opt_id)

    def __setstate__(self, state):
        super(FusedCPUAdam, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)

    @torch.no_grad()
    def step(self, closure=None):
        # NOTE: xyz lr is changing at every step, while other columes_lr are fixed during training.

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # intended device for step
        device = torch.device('cpu')

        p = self.param_groups[0]["params"][0]
        if p.grad is None:
            return
        assert p.device == device, f"FusedCPUAdam param is on {p.device} and must be 'cpu', make " \
                "sure you enabled 'offload_optimizer': 'cpu' in your ZeRO config."
        assert p.data.shape[1] == self.columns_n_total, "FusedCPUAdam expects the model to have the same number of columns as the optimizer"

        state = self.state[p]
        if len(state) == 0:
            state['step'] = 0
            state_dtype = torch.float if self.fp32_optimizer_states else p.dtype
            # gradient momentums
            state['exp_avg'] = torch.zeros_like(p.data, dtype=state_dtype, device=device)
            # gradient variances
            state['exp_avg_sq'] = torch.zeros_like(p.data, dtype=state_dtype, device=device)

        state['step'] += 1
        beta1, beta2 = self.param_groups[0]['betas'] # betas are shared across all columns
        
        xyz_lr = self.param_groups[0]['lr']

        eps = self.param_groups[0]['eps']
        weight_decay = self.param_groups[0]['weight_decay']
        bias_correction = self.param_groups[0]['bias_correction']

        _C.fused_adam_update(self.opt_id, state['step'],
                             self.columns_offsets,
                             xyz_lr, self.columns_lr,
                             beta1, beta2, eps, weight_decay, bias_correction,
                             p.data, p.grad.data, state['exp_avg'], state['exp_avg_sq'])

        return loss

    @torch.no_grad()
    def sparse_step(self, closure=None, sparse_indices=None, version=1, scale=1.0):
        # sparse_indices int
        # NOTE: xyz lr is changing at every step, while other columes_lr are fixed during training.

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        assert sparse_indices is not None, "sparse_indices must be provided for sparse_step"

        # intended device for step
        device = torch.device('cpu')

        p = self.param_groups[0]["params"][0]
        if p.grad is None:
            return
        assert p.device == device, f"FusedCPUAdam param is on {p.device} and must be 'cpu', make " \
                "sure you enabled 'offload_optimizer': 'cpu' in your ZeRO config."
        assert p.data.shape[1] == self.columns_n_total, "FusedCPUAdam expects the model to have the same number of columns as the optimizer"

        state = self.state[p]
        if len(state) == 0:
            state['step'] = 0
            state_dtype = torch.float if self.fp32_optimizer_states else p.dtype
            # gradient momentums
            state['exp_avg'] = torch.zeros_like(p.data, dtype=state_dtype, device=device)
            # gradient variances
            state['exp_avg_sq'] = torch.zeros_like(p.data, dtype=state_dtype, device=device)

        beta1, beta2 = self.param_groups[0]['betas'] # betas are shared across all columns
        
        xyz_lr = self.param_groups[0]['lr']

        eps = self.param_groups[0]['eps']
        weight_decay = self.param_groups[0]['weight_decay']
        bias_correction = self.param_groups[0]['bias_correction']

        _C.sparse_fused_adam_update(self.opt_id, state['step'],
                                self.columns_offsets,
                                xyz_lr, self.columns_lr,
                                beta1, beta2, eps, weight_decay, bias_correction,
                                p.data, p.grad.data, state['exp_avg'], state['exp_avg_sq'],
                                sparse_indices, version, scale)

        return loss

    @torch.no_grad()
    def sparse_adam_inc_step(self):
        p = self.param_groups[0]["params"][0]
        if p.grad is None:
            return
        device = torch.device('cpu')
        assert p.device == device, f"FusedCPUAdam param is on {p.device} and must be 'cpu', make " \
                "sure you enabled 'offload_optimizer': 'cpu' in your ZeRO config."
        assert p.data.shape[1] == self.columns_n_total, "FusedCPUAdam expects the model to have the same number of columns as the optimizer"
        state = self.state[p]
        if len(state) == 0:
            state['step'] = 0
            state_dtype = torch.float if self.fp32_optimizer_states else p.dtype
            # gradient momentums
            state['exp_avg'] = torch.zeros_like(p.data, dtype=state_dtype, device=device)
            # gradient variances
            state['exp_avg_sq'] = torch.zeros_like(p.data, dtype=state_dtype, device=device)
        
        state['step'] += 1

    @torch.no_grad()
    def batched_sparse_step(self, batch_size, batched_sparse_indices=None, signal_tensor_pinned=None, version=1, scale=1.0, sparse_adam=False):
        # sparse_indices int
        # NOTE: xyz lr is changing at every step, while other columes_lr are fixed during training.

        assert batched_sparse_indices is not None, "sparse_indices must be provided for sparse_step"
        assert signal_tensor_pinned is not None, "signal_tensor_pinned must be provided for sparse_step"

        # intended device for step
        device = torch.device('cpu')

        p = self.param_groups[0]["params"][0]
        if p.grad is None:
            return
        assert p.device == device, f"FusedCPUAdam param is on {p.device} and must be 'cpu', make " \
                "sure you enabled 'offload_optimizer': 'cpu' in your ZeRO config."
        assert p.data.shape[1] == self.columns_n_total, "FusedCPUAdam expects the model to have the same number of columns as the optimizer"

        state = self.state[p]
        if len(state) == 0:
            state['step'] = 0
            state_dtype = torch.float if self.fp32_optimizer_states else p.dtype
            # gradient momentums
            state['exp_avg'] = torch.zeros_like(p.data, dtype=state_dtype, device=device)
            # gradient variances
            state['exp_avg_sq'] = torch.zeros_like(p.data, dtype=state_dtype, device=device)

        state['step'] += 1
        beta1, beta2 = self.param_groups[0]['betas'] # betas are shared across all columns
        
        xyz_lr = self.param_groups[0]['lr']

        eps = self.param_groups[0]['eps']
        weight_decay = self.param_groups[0]['weight_decay']
        bias_correction = self.param_groups[0]['bias_correction']

        _C.batched_sparse_fused_adam_update(self.opt_id, state['step'],
                                    self.columns_offsets,
                                    xyz_lr, self.columns_lr,
                                    beta1, beta2, eps, weight_decay, bias_correction,
                                    p.data, p.grad.data, state['exp_avg'], state['exp_avg_sq'],
                                    batch_size,
                                    batched_sparse_indices, # vector of torch long tensors
                                    signal_tensor_pinned, # torch tensor
                                    version,
                                    scale,
                                    1 if sparse_adam else 0)

