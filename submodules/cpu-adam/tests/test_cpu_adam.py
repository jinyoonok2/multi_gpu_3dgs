# python tests/test_cpu_adam.py

import torch
from torch import optim
import time
from cpu_adam import CPUAdam, FusedCPUAdam

def test_speed(optimizer_mode, N):
    # set seed
    torch.manual_seed(1)

    column_sizes = [3, 3, 45, 1, 3, 4] # sum=59, we have 59 columns in total
    # column_sizes = [3, 3, 45, 1, 3, 4] # sum=59, we have 59 columns in total
    total_columns = sum(column_sizes)
    column_lrs = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]

    # random test case
    # params = [torch.randn((N, column_size), requires_grad=True, dtype=torch.float32) for column_size in column_sizes]
    # grads = [torch.randn((N, column_size), dtype=torch.float32) for column_size in column_sizes]

    # constant test case
    params = [torch.ones((N, column_size), requires_grad=True, dtype=torch.float32) for column_size in column_sizes]
    grads = [torch.ones((N, column_size), dtype=torch.float32) for column_size in column_sizes]
    column_lrs = [ x*0.001 for x in column_lrs]

    sum_before_training = [round(torch.sum(param).item(), 5) for param in params]

    if optimizer_mode == "fused_cpu_adam":
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
    else:
        param_list = [
            {
                "params": [param],
                "lr": lr,
            } for param, lr in zip(params, column_lrs)
        ]
        # Define the optimizer
        if optimizer_mode == "cpu_adam":
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
        elif optimizer_mode == "torch_adam":
            optimizer = optim.Adam(
                param_list,
                lr=0.001, 
                # betas=(0.9, 0.999), 
                betas=(0.8, 0.899),
                eps=1e-8, 
                weight_decay=0.01, 
                amsgrad=False,
            )

    # Function to simulate a step of optimization
    def optimize(optimizer, steps):
        start_time = time.time()
        for _ in range(steps):
            # Simulate the backward pass by manually setting the gradients
            # params.grad = grads
            for p, g in zip(params, grads):
                p.grad = g
            optimizer.step()    # Update parameters
            optimizer.zero_grad(set_to_none=True)  # Clear gradients for the next step
        elapsed_time = time.time() - start_time
        print(f"Mode[{optimizer_mode}]: Optimization completed in {elapsed_time:.2f} seconds for {steps} steps.")

    # Test the speed of the Adam optimizer
    optimize(optimizer, 30)

    if optimizer_mode == "fused_cpu_adam":
        params = params[0].split(column_sizes, dim=1)
        params = [param.contiguous() for param in params]
        # NOTE: contiguous() is required other wise it will give wrong results; I think this is a bug in the code.

    sum_after_training = [round(torch.sum(param).item(), 5) for param in params]

    # print the parameters sum
    print(f"sum of the parameters before training: {sum_before_training}")
    print(f"sum of the parameters after training: {sum_after_training}")

    results = {
        "sample_after_training": params[0].detach().flatten().numpy().tolist(),
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

    all_N = [int(1e5), int(1e6), int(1e7)]
    for N in all_N[:3]:
        print(f"N: {N}")
        torch_adam_results = test_speed("torch_adam", N)
        cpu_adam_results = test_speed("cpu_adam", N)
        fused_cpu_adam_results = test_speed("fused_cpu_adam", N)

        compare(torch_adam_results["sample_after_training"], fused_cpu_adam_results["sample_after_training"])
        compare(cpu_adam_results["sample_after_training"], fused_cpu_adam_results["sample_after_training"])

        print("\n")


    pass
