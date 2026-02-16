# python tests/test_cpu_adam.py

import torch
from torch import optim
import time
from cpu_adam import CPUAdam, FusedCPUAdam
import gc
import pandas as pd
import os

def test_speed(optimizer_mode, N, sparsity):
    # set seed
    torch.manual_seed(1)

    column_sizes = [3, 45] # sum=59, we have 59 columns in total
    # column_sizes = [3, 3, 45, 1, 3, 4] # sum=59, we have 59 columns in total
    total_columns = sum(column_sizes)
    column_lrs = [0.1, 0.2]

    # random test case
    # params = [torch.randn((N, column_size), requires_grad=True, dtype=torch.float32) for column_size in column_sizes]
    # grads = [torch.randn((N, column_size), dtype=torch.float32) for column_size in column_sizes]

    # constant test case
    params = [torch.ones((N, column_size), requires_grad=True, dtype=torch.float32) for column_size in column_sizes]
    grads = [torch.ones((N, column_size), dtype=torch.float32) for column_size in column_sizes]
    column_lrs = [ x*0.001 for x in column_lrs]

    sum_before_training = [round(torch.sum(param).item(), 5) for param in params]

    # sample sparse indices for adam updating
    n_sampled_indices = int(N*sparsity)
    sparse_indices = torch.randperm(N, dtype=torch.int32)[:n_sampled_indices]
    sparse_indices = torch.sort(sparse_indices).values
    print("Sampled Indices/All Indices=", n_sampled_indices, N)

    if "fused_cpu_adam" in optimizer_mode:
        params = [torch.cat(params, dim=1).detach().requires_grad_()]
        grads = [torch.cat(grads, dim=1)]

        param_list = [
            {
                "params": [params[0]],
                "lr": column_lrs[0],
            }
        ]
        optimizer = FusedCPUAdam(
            param_list,
            columns_sizes=column_sizes,
            columns_lr=column_lrs,
            lr=1e-3,
            bias_correction=True, # This True is required. 
            # betas=(0.9, 0.999),
            betas=(0.8, 0.899),
            eps=1e-8,
            weight_decay=0.01,
            amsgrad=False,
            adamw_mode=False,
            fp32_optimizer_states=True
        )
    elif optimizer_mode == "sparse_cpu_adam":
        param_list = [
            {
                "params": [params[0]],
                "lr": column_lrs[0],
            },
            {
                "params": [params[1]],
                "lr": column_lrs[1],
            },
        ]
        optimizer = CPUAdam(
            param_list,
            lr=1e-3,
            bias_correction=True, # This True is required. 
            # betas=(0.9, 0.999),
            betas=(0.8, 0.899),
            eps=1e-8,
            weight_decay=0.01,
            amsgrad=False,
            adamw_mode=False,
            fp32_optimizer_states=True
        )
    else:
        raise NotImplementedError

    # Function to simulate a step of optimization
    def optimize(optimizer, steps):
        warmup_step = 5
        total_time = 0
        for idx in range(warmup_step + steps):
            # Simulate the backward pass by manually setting the gradients
            # params.grad = grads
            for p, g in zip(params, grads):
                p.grad = g

            gc.collect()
            if idx >= warmup_step:
                start_time = time.perf_counter()
            if optimizer_mode == "fused_cpu_adam":
                optimizer.step()
            elif optimizer_mode == "sparse_fused_cpu_adam_v1":
                optimizer.sparse_step(sparse_indices=sparse_indices, version=1)
            elif optimizer_mode == "sparse_fused_cpu_adam_v2":
                optimizer.sparse_step(sparse_indices=sparse_indices, version=2)
            elif optimizer_mode == "sparse_cpu_adam":
                optimizer.sparse_step(sparse_indices=sparse_indices)
            # optimizer.zero_grad(set_to_none=True)  sparse_fused_cpu_adam# Clear gradients for the next step. Maybe I need to disable zero_grad() operation.
            if idx >= warmup_step:
                elapsed_time = (time.perf_counter() - start_time) * 1000
                total_time += elapsed_time
        average_time = total_time / steps
        print(f"Mode[{optimizer_mode}]: Optimization completed in {average_time:.2f} ms[average over {steps} steps].")
        return average_time

    # Test the speed of the Adam optimizer
    average_time = optimize(optimizer, 10)

    if "fused_cpu_adam" in optimizer_mode:
        params = params[0].split(column_sizes, dim=1)
        params = [param.contiguous() for param in params]
        # NOTE: contiguous() is required other wise it will give wrong results; I think this is a bug in the code.
    elif optimizer_mode == "sparse_cpu_adam":
        pass
    else:
        raise NotImplementedError

    sum_after_training = [round(torch.sum(param).item(), 5) for param in params]

    # print the parameters sum
    print(f"sum of the parameters before training: {sum_before_training}")
    print(f"sum of the parameters after training: {sum_after_training}")

    results = {
        "sample_after_training": params[0].detach().flatten().numpy().tolist(),
        "average_time": average_time,
    }
    return results

def compare(list_A, list_B):
    if len(list_A) != len(list_B):
        return False
    # print mismatched elements
    for i, (a, b) in enumerate(zip(list_A, list_B)):
        if a != b:
            print(f"mismatch at index {i}: {a} != {b}")
            return False
    print("All elements matched.")
    return True

if __name__ == "__main__":

    all_N = [int(2e5), int(2e6), int(1e7)]
    all_sparsity = [1.0/32, 
                    # 1.0/16, 
                    1.0/8,
                    # 1.0/4,
                    # 1.0/2,
                    1.0]
    # all_sparsity = [1.0/32]

    df_columns = ["N", "Sparsity", "Sparse Fused CPU Adam V1", "Sparse Fused CPU Adam V2", "Fused CPU Adam", "Sparse CPU Adam"]
    df = pd.DataFrame(columns=df_columns)

    for N in all_N:
        for sparsity in all_sparsity:
            print(f"N: {N}")
            print(f"N: {N}, sparsity: {sparsity}")
            sparse_fused_cpu_adam_v1_results = test_speed("sparse_fused_cpu_adam_v1", N, sparsity)
            sparse_fused_cpu_adam_v2_results = test_speed("sparse_fused_cpu_adam_v2", N, sparsity)
            sparse_cpu_adam_results = test_speed("sparse_cpu_adam", N, sparsity)
            if sparsity == 1.0:
                fused_cpu_adam_results = test_speed("fused_cpu_adam", N, sparsity)
            else:
                fused_cpu_adam_results = None

            # compare(sparse_fused_cpu_adam_v1_results["sample_after_training"],
            #         sparse_fused_cpu_adam_v2_results["sample_after_training"])
            compare(sparse_fused_cpu_adam_v1_results["sample_after_training"],
                    sparse_cpu_adam_results["sample_after_training"])
            if sparsity == 1.0:
                # compare(sparse_fused_cpu_adam_v1_results["sample_after_training"],
                #         fused_cpu_adam_results["sample_after_training"])
                compare(sparse_cpu_adam_results["sample_after_training"],
                        fused_cpu_adam_results["sample_after_training"])
            
            df = df._append({
                "N": int(N),
                "Sparsity": sparsity,
                "Sparse Fused CPU Adam V1": str(round(sparse_fused_cpu_adam_v1_results["average_time"], 3))+"ms",
                "Sparse Fused CPU Adam V2": str(round(sparse_fused_cpu_adam_v2_results["average_time"], 3))+"ms",
                "Fused CPU Adam": str(round(fused_cpu_adam_results["average_time"], 3))+"ms" if sparsity == 1.0 else "N/A",
                "Sparse CPU Adam": str(round(sparse_cpu_adam_results["average_time"], 3))+"ms",
            }, ignore_index=True)
        print("\n")
    
    # save df in file
    idx = 1
    while os.path.exists(f"speed_test_results_{idx}.csv"):
        idx += 1
    df.to_csv(f"speed_test_results_{idx}.csv", index=False)
    # df.to_csv("speed_test_results_5.csv", index=False)

    pass
