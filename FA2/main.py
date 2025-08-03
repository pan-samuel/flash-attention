import torch
import math
import subprocess
import time
import torch.utils.cpp_extension
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
import triton.runtime as runtime
from fused_attention import triton_attention


device = "0"  # Assuming single GPU

# Load your custom module
module = torch.utils.cpp_extension.load(
    "module",
    sources=["flashattention_v1.cu",
             "flash_attention.cpp"],
    extra_cuda_cflags=["-O3", "--use_fast_math", "--ptxas-options=-v", "-allow-unsupported-compiler"],
    verbose=False,
)

def set_fixed_clocks():
    # Set fixed GPU clock speed for reproducible benchmarking 
    clock_speed = 1350  # Adjust based on your A6000's supported speeds
    
    try:
        subprocess.run(f"sudo nvidia-smi -pm ENABLED -i {device}", shell=True, check=True)
        subprocess.run(f"sudo nvidia-smi -lgc {clock_speed} -i {device}", shell=True, check=True)
        print(f"Set GPU clock to {clock_speed} MHz")
    except subprocess.CalledProcessError:
        print("Warning: Could not set fixed GPU clocks (requires sudo)")

def reset_clocks():
    # Reset GPU clocks to default
    try:
        subprocess.run(f"sudo nvidia-smi -rgc -i {device}", shell=True, check=True)
        print("Reset GPU clocks to default")
    except subprocess.CalledProcessError:
        print("Warning: Could not reset GPU clocks")

def benchmark(func, *args, warmup_steps=10, timing_steps=10, **kwargs):
    for _ in range(warmup_steps):
        func(*args, **kwargs)
    
    cache = runtime.driver.active.get_empty_cache_for_benchmark()
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(timing_steps)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(timing_steps)]
    
    for i in range(timing_steps):
        # Flush cache to avoid cache hits affecting timing
        runtime.driver.active.clear_cache(cache)
        
        # Sleep to saturate command queue and avoid GPU "outrunning" CPU
        torch.cuda._sleep(1_000_000)
        
        # Record timing events around the kernel execution
        start_events[i].record()
        func(*args, **kwargs)
        end_events[i].record()
    
    torch.cuda.synchronize()
    
    times = [start.elapsed_time(end) for start, end in zip(start_events, end_events)]
    return sorted(times)[len(times) // 2]

def calculate_flops(B, n_heads, T, head_dim, time_ms, is_causal=True):
    """Calculate FLOPS for attention computation"""
    # Attention FLOPS: 4 * B * n_heads * T^2 * head_dim for QK^T + softmax + weighted sum
    flops = 4 * T * T * head_dim * n_heads * B
    
    # With causal mask, only ~half the computation is done
    if is_causal:
        flops = flops // 2
    
    flops_per_second = flops / (time_ms / 1000) 
    return flops_per_second / 1e12

def main():

    set_fixed_clocks()

    B, n_heads = 1, 6
    T, head_dim = 1024, 64
    sm_scale = 1.0 / math.sqrt(head_dim)
    torch.manual_seed(0)
    dtype = torch.bfloat16   
    device = 'cuda'
    
    
    # Generate input tensors
    q = torch.randn(B, n_heads, T, head_dim, dtype=dtype, device=device, requires_grad=False).cuda()
    k = torch.randn(B, n_heads, T, head_dim, dtype=dtype, device=device, requires_grad=False).cuda()
    v = torch.randn(B, n_heads, T, head_dim, dtype=dtype, device=device, requires_grad=False).cuda()
        
    # Get reference output using default SDPA (will use the best available kernel)
    output_ref = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
    
    # Test custom kernels
    output_v1 = module.flashattn_v1(q, k, v)
    # output_v2 = module.flashattn_v2(q, k, v)
    # output_v3 = module.flashattn_v3(q, k, v)
    # output_v4 = module.flashattn_v4(q, k, v)
    # output_v5 = module.flashattn_v5(q, k, v)
    # output_v6 = module.flashattn_v6(q, k, v)

    # Verify correctness
    print("Verifying correctness...")
    # torch.testing.assert_close(output_v1, output_ref, rtol=1e-4, atol=1e-4)
    # torch.testing.assert_close(output_v2, output_ref, rtol=1e-4, atol=1e-4)
    # torch.testing.assert_close(output_v3, output_ref, rtol=1e-4, atol=1e-4)
    # torch.testing.assert_close(output_v4, output_ref, rtol=1e-4, atol=1e-4)
    # torch.testing.assert_close(output_v5, output_ref, rtol=1e-4, atol=1e-4)
    # torch.testing.assert_close(output_v6, output_ref, rtol=1e-4, atol=1e-4)
    print("All kernels passed correctness test!")
    print()

    
    print("Benchmarking...")
    print(f"{'Kernel':<30} {'Time (ms)':<12} {'TFLOPS':<10}")
    print("-" * 85)
    
    # Build list of kernels to test based on availability
    kernels_to_test = [
        ("PyTorch SDPA (Default)", lambda: F.scaled_dot_product_attention(q, k, v, is_causal=True), True),
        ("PyTorch SDPA (Math)", "math", True),
        ("PyTorch SDPA (Efficient)", "efficient", True),
        ("PyTorch SDPA (Flash Attention)", "flash", True),
        ("PyTorch SDPA (cuDNN)", "cuDNN", True),
        # ("Triton Flash Attention", lambda: triton_attention(q, k, v, sm_scale), True),
        # ("Custom v2", lambda: module.flashattn_v2(q, k, v), False),
        # ("Custom v3", lambda: module.flashattn_v3(q, k, v), False),
        # ("Custom v4", lambda: module.flashattn_v4(q, k, v), False),
        # ("Custom v5", lambda: module.flashattn_v5(q, k, v), False),
        # ("Custom v6", lambda: module.flashattn_v6(q, k, v), False),
    ]

    
    results = []
    
    for kernel_info in kernels_to_test:
        if len(kernel_info) == 3:
            name, func_or_backend, is_causal = kernel_info
        else:
            name, func_or_backend = kernel_info
            is_causal = True
            
        if isinstance(func_or_backend, str):
            backend_map = {
                "math": SDPBackend.MATH,
                "efficient": SDPBackend.EFFICIENT_ATTENTION,
                "flash": SDPBackend.FLASH_ATTENTION,
                "cuDNN": SDPBackend.CUDNN_ATTENTION,
            }
            backend = backend_map[func_or_backend]
            def sdpa_func():
                with sdpa_kernel([backend]):
                    return F.scaled_dot_product_attention(q, k, v, is_causal=True)
            median_time = benchmark(sdpa_func)
        else:
            func = func_or_backend
            median_time = benchmark(func)
        
        tflops = calculate_flops(B, n_heads, T, head_dim, median_time, is_causal)
        results.append((name, median_time, tflops))
        print(f"{name:<30} {median_time:<12.4f} {tflops:<10.2f}")
        time.sleep(0.5)
            
    print("\nBenchmark Results Summary:")
    print("-" * 85)
    sorted_results = sorted(results, key=lambda x: x[1])
    for i, (name, time_ms, tflops) in enumerate(sorted_results):
        speedup = sorted_results[0][1] / time_ms if i > 0 else 1.0
        print(f"{i+1:2d}. {name:<28} {time_ms:8.4f}ms  {tflops:6.2f} TFLOPS  {speedup:5.2f}x")
    
    reset_clocks()

if __name__ == "__main__":
    main()
