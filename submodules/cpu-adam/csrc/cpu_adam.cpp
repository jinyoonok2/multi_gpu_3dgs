// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0

// DeepSpeed Team

#include "cpu_adam.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("adam_update", &ds_adam_step, "CPU Adam update (C++)");
    m.def("sparse_adam_update", &ds_sparse_adam_step, "CPU Adam sparse update (C++)");
    m.def("fused_adam_update", &ds_fused_adam_step, "Fused CPU Adam update (C++)");
    m.def("sparse_fused_adam_update", &ds_sparse_fused_adam_step, "Fused CPU Adam update (C++)");
    m.def("batched_sparse_fused_adam_update", &ds_batched_sparse_fused_adam_step, "Fused CPU Adam update (C++)");
    m.def("create_adam", &create_adam_optimizer, "CPU Adam (C++)");
    m.def("destroy_adam", &destroy_adam_optimizer, "CPU Adam destroy (C++)");
}