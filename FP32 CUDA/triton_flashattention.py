# triton_flashattention.py
# Contains the forward pass of a Triton Flash Attention implementation for benchmarking.
# All backward pass code and testing/benchmarking runners have been removed.

import torch
import triton
import triton.language as tl
import math

@triton.jit
def _attn_fwd_inner(
    Q, O, L, M,
    K_ptr, V_ptr,
    K_T_offsets, V_offsets,
    block_index_QO,
    softmax_scale,
    stride_K_N, stride_V_N,
    BLOCK_SIZE_QO: tl.constexpr, BLOCK_SIZE_KV: tl.constexpr,
    DIAGONAL: tl.constexpr,
    offsets_QO_N: tl.constexpr, offsets_KV_N: tl.constexpr,
    N: tl.constexpr, Dh: tl.constexpr,
):
    '''
    arrows indicate direction of this pid's for loop; each arrow is a different PID
                N of K & V
                ------------>
                ------------>
    N of Q      ------------>
                ------------>
                ------------>
    but if we actually take into account the causal mask then really it's more like
                N of K & V
                >
                --->
    N of Q      ------>
                --------->
                ------------>
    and to get even more accurate, we do the diagonal in our second call of this inner kernel
                N of K & V
                x
                   x
    N of Q            x
                         x
                            x
    and then the first call gets all the parts below the diagonal
                N of K & V
                
                -->
    N of Q      ----->
                -------->
                ----------->
    This inner kernel function processes a block of queries against key-value pairs.
    It's called twice by the main kernel:
    1. For all key/value blocks below the diagonal (fully dense attention).
    2. For the single key/value block on the diagonal (causally masked attention).
    '''
    if DIAGONAL:
        # This branch handles blocks on the diagonal, which require causal masking.
        lo = block_index_QO * BLOCK_SIZE_QO
        hi = (block_index_QO + 1) * BLOCK_SIZE_QO
        # Hint to the compiler that `lo` is a multiple of the block size for potential optimizations.
        lo = tl.multiple_of(lo, BLOCK_SIZE_QO)
    else: 
        # This branch processes all blocks below the diagonal, which are fully attended to.
        lo, hi = 0, block_index_QO * BLOCK_SIZE_QO

    # Adjust pointers to the start of the relevant K/V block section.
    K_T_offsets += lo * stride_K_N
    V_offsets += lo * stride_V_N
    offsets_KV_N += lo

    # Iterate over key-value blocks.
    for start_KV in range(lo, hi, BLOCK_SIZE_KV):
        # Again, provide a hint to the compiler about the loop variable's alignment.
        start_KV = tl.multiple_of(start_KV, BLOCK_SIZE_KV)

        # --- Begin attention score calculation ---
        mask_KV_N = offsets_KV_N < N
        # Load a block of K, transposing it on the fly. Mask out any padding tokens.
        K_T = tl.load(K_ptr + K_T_offsets, mask=mask_KV_N[None, :], other=0.)
        # Calculate the scaled dot-product attention scores.
        S = tl.dot(Q, K_T) * softmax_scale

        if DIAGONAL: # If this is a diagonal block, apply the causal mask.
            # Create a mask where attention is only allowed for keys at or before the current query position.
            causal_mask = offsets_QO_N[:, None] >= (offsets_KV_N[None, :])
            # Set scores for future tokens to a large negative number to zero them out after softmax.
            S += tl.where(causal_mask, 0, -1.0e6)
        
        # --- Perform online softmax ---
        # Find the new maximum score for the current block and update the running max.
        M_new = tl.maximum(M, tl.max(S, axis=1))
        # Stabilize scores by subtracting the new maximum value.
        S -= M_new[:, None]
        # Compute the numerator of the softmax. `exp2` is used as a faster alternative to `exp`.
        P = tl.exp2(S)
        
        # Calculate the sum of the new exponentiated scores.
        L_new = tl.sum(P, axis=1)
        # Create a correction factor based on the change in the running maximum.
        alpha = tl.exp2(M - M_new)
        # Update the running denominator accumulator `L`.
        L = L * alpha + L_new

        # --- Update the output block O ---
        # Load the corresponding block of V, masking any padding.
        V = tl.load(V_ptr + V_offsets, mask=mask_KV_N[:, None], other=0.)
        # Apply the correction factor to the existing output accumulator.
        O = O * alpha[:, None]
        # Accumulate the weighted values into the output. `acc=O` performs `O += P @ V`.
        O = tl.dot(P, V, acc=O)

        # Update the running maximum for the next iteration.
        M = M_new

        # Advance pointers for the next block in the loop.
        K_T_offsets += BLOCK_SIZE_KV * stride_K_N
        V_offsets += BLOCK_SIZE_KV * stride_V_N
        offsets_KV_N += BLOCK_SIZE_KV

    return O, L, M


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_QO': 64, 'BLOCK_SIZE_KV': 64, 'num_warps': 4, 'num_stages': 5}),
        triton.Config({'BLOCK_SIZE_QO': 128, 'BLOCK_SIZE_KV': 64, 'num_warps': 8, 'num_stages': 3}),
    ],
    key=['Dh'],
)
@triton.jit
def attn_fwd(
    Q_ptr, K_ptr,  V_ptr,
    O_ptr,
    LSE_ptr,
    softmax_scale,
    stride_Q_B, stride_Q_H, stride_Q_N, stride_Q_Dh,
    stride_K_B, stride_K_H, stride_K_N, stride_K_Dh,
    stride_V_B, stride_V_H, stride_V_N, stride_V_Dh,
    stride_O_B, stride_O_H, stride_O_N, stride_O_Dh,
    stride_LSE_B, stride_LSE_H, stride_LSE_N,
    B, H: tl.constexpr, N: tl.constexpr, Dh: tl.constexpr,
    BLOCK_SIZE_QO: tl.constexpr, BLOCK_SIZE_KV: tl.constexpr,
):
    '''
    We use tl.exp2(x * 1/ln(2)) instead of tl.exp(x) because exp2 is faster.
    The identity is e^x = 2^(x * log2(e)), where log2(e) is 1/ln(2) or rln2.
    So, we scale our input `softmax_scale` by `rln2` at the beginning.
    '''
    rln2: tl.constexpr = 1.4426950408889634
    softmax_scale *= rln2
    
    # This assertion is a performance guard, though its original motivation is not fully clear.
    tl.static_assert(BLOCK_SIZE_KV <= Dh, "BLOCK_SIZE_KV must be less than or equal to head dimension.")

    # Each program instance handles a specific block of Q and a specific batch/head.
    block_index_QO = tl.program_id(0) # ID for the block in the sequence dimension.
    index_BH = tl.program_id(1)      # ID for the batch and head.
    index_B = index_BH // H          # Derive batch index.
    index_H = index_BH % H           # Derive head index.

    # Move pointers to the start of the data for the current batch and head.
    Q_ptr += index_B * stride_Q_B + index_H * stride_Q_H
    K_ptr += index_B * stride_K_B + index_H * stride_K_H
    V_ptr += index_B * stride_V_B + index_H * stride_V_H
    O_ptr += index_B * stride_O_B + index_H * stride_O_H

    # Define the ranges of indices for the current block.
    offsets_QO_N = block_index_QO * BLOCK_SIZE_QO + tl.arange(0, BLOCK_SIZE_QO)
    offsets_KV_N = tl.arange(0, BLOCK_SIZE_KV)
    offsets_Dh = tl.arange(0, Dh)
    
    # Construct 2D offset tensors to load blocks of data from global memory.
    Q_offsets = (offsets_QO_N[:, None] * stride_Q_N + offsets_Dh[None, :] * stride_Q_Dh)
    # K is transposed on-the-fly by swapping the role of offsets and strides.
    K_T_offsets = (offsets_Dh[:, None] * stride_K_Dh + offsets_KV_N[None, :] * stride_K_N)
    V_offsets = (offsets_KV_N[:, None] * stride_V_N + offsets_Dh[None, :] * stride_V_Dh)

    # Load the block of Q that this program will process; it remains in SRAM.
    mask_QO_N = offsets_QO_N < N
    Q = tl.load(Q_ptr + Q_offsets, mask=mask_QO_N[:, None], other=0.)

    # Initialize accumulators for the online softmax algorithm.
    M = tl.full(shape=[BLOCK_SIZE_QO], value=-1e6, dtype=tl.float32) # Running maximum.
    L = tl.full(shape=[BLOCK_SIZE_QO], value=1.0, dtype=tl.float32)  # Running sum (denominator).
    O = tl.zeros([BLOCK_SIZE_QO, Dh], dtype=tl.float32)              # Output accumulator.

    # First, calculate attention for all dense blocks (below the diagonal).
    O, L, M = _attn_fwd_inner(
        Q, O, L, M, K_ptr, V_ptr, K_T_offsets, V_offsets, block_index_QO, softmax_scale,
        stride_K_N, stride_V_N, BLOCK_SIZE_QO, BLOCK_SIZE_KV, False, offsets_QO_N, offsets_KV_N, N, Dh
    )

    # Second, calculate attention for the single sparse block (on the diagonal).
    O, L, M = _attn_fwd_inner(
        Q, O, L, M, K_ptr, V_ptr, K_T_offsets, V_offsets, block_index_QO, softmax_scale,
        stride_K_N, stride_V_N, BLOCK_SIZE_QO, BLOCK_SIZE_KV, True, offsets_QO_N, offsets_KV_N, N, Dh
    )
    
    # Final normalization: divide the output accumulator by the softmax denominator.
    # This is done after accumulating all P@V products, which is valid due to associativity.
    O = O / L[:, None]
    # Compute the log-sum-exp, required for a numerically stable backward pass.
    LSE = M + tl.math.log2(L)

    # --- Store results back to global memory (DRAM) ---
    LSE_offsets = index_BH * stride_LSE_H + offsets_QO_N
    LSE_mask = block_index_QO * BLOCK_SIZE_QO + tl.arange(0, BLOCK_SIZE_QO) < N
    tl.store(LSE_ptr + LSE_offsets, LSE, mask=LSE_mask)
    
    O_offsets = (offsets_QO_N[:, None] * stride_O_N + offsets_Dh[None, :] * stride_O_Dh)
    tl.store(O_ptr + O_offsets, O, mask=mask_QO_N[:, None])


class _flashattention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, scale): 
        # A practical limitation based on typical GPU SRAM capacity.
        assert q.shape[-1] <= 128, \
            f'Flash attention only supports head dimension of 128 or less, but got {q.shape[-1]}'
        B, H, N, Dh = q.shape
        assert q.is_cuda and k.is_cuda and v.is_cuda
        assert q.dtype == torch.float32 and k.dtype == torch.float32 and v.dtype == torch.float32

        # Allocate tensors for the output and intermediate log-sum-exp values.
        O = torch.empty_like(q)
        LSE = torch.empty((B, H, N), device=q.device, dtype=torch.float32)

        # Configure the launch grid for the Triton kernel.
        grid = lambda args: (
            triton.cdiv(N, args["BLOCK_SIZE_QO"]), # Parallelize over the sequence length.
            B * H,                                # Parallelize over batches and heads.
        )
        # Note: The sequence dimension is the first grid axis. This tends to place
        # programs processing adjacent sequence blocks onto the same Streaming Multiprocessor (SM),
        # which can improve performance through better data locality in caches.

        # Execute the Triton kernel.
        attn_fwd[grid](
            q, k, v, O, LSE, 
            scale,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            O.stride(0), O.stride(1), O.stride(2), O.stride(3),
            LSE.stride(0), LSE.stride(1), LSE.stride(2),
            B, H, N, Dh,
        )

        return O

# Public API for the forward pass.
triton_attention = _flashattention.apply
