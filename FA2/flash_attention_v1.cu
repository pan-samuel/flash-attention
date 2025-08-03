#include <cmath>
#include <cuda_runtime.h>
#include <stdio.h>
#include <cuda_bf16.h>
#include <cstdint>
#include "ptx.cuh"

constexpr int WARP_SIZE = 32;
using uint128_t = uint4; 

__host__ __device__ inline constexpr int cdiv(int a, int b) { return (a + b - 1) / b; }

template<int BLOCK_SIZE, int HEIGHT, int WIDTH>
__device__ void smem_load_128b(const nv_bfloat16* in, int in_row_stride, nv_bfloat16* out, int out_row_stride, int tid, int global_offset) {
    using load_type = uint128_t;
    constexpr int num_vals = sizeof(uint128_t) / sizeof(nv_bfloat16);

    for (int offset = 0; offset < HEIGHT * WIDTH; offset += BLOCK_SIZE * num_vals) {
        const int idx = offset + tid * num_vals;
        const int row = idx / WIDTH;
        const int col = idx % WIDTH;
        load_type tmp = reinterpret_cast<const load_type*>(&in[global_offset + row * in_row_stride + col])[0];
        reinterpret_cast<load_type*>(&out[row * out_row_stride + col])[0] = tmp;
    }
}

template<int BLOCK_M, int BLOCK_N, int NUM_WARPS, int d>
__global__ void flashattn_kernel_v1(
    const nv_bfloat16* __restrict__ Q,
    const nv_bfloat16* __restrict__ K,
    const nv_bfloat16* __restrict__ V,
    nv_bfloat16* __restrict__ O,
    int bh, 
    int seqlen_q)
{
    int tidx = threadIdx.x;
    int m_block = blockIdx.x;
    int bid = blockIdx.y;
    int warp_id = tidx / WARP_SIZE;
    int lane_id = tidx % WARP_SIZE;

    const int NThreads = WARP_SIZE * NUM_WARPS;
    const int Q_SHARD = BLOCK_M / NUM_WARPS;

    const int offset_q = (bid * seqlen_q + m_block * BLOCK_M + Q_SHARD * warp_id) * d;
    const int offset_kv = bid * seqlen_q * d;

    extern __shared__ nv_bfloat16 smem[];
    nv_bfloat16* Q_smem = smem;
    nv_bfloat16* K_smem = Q_smem + BLOCK_M * d;
    nv_bfloat16* V_smem = K_smem + BLOCK_N * d;

    constexpr int MMA_M = 16;
    constexpr int MMA_N = 8;
    constexpr int MMA_K = 16;
    constexpr int NUM_REG_A = 4;
    constexpr int NUM_REG_B = 2;
    constexpr int NUM_REG_C = 4;
    
    constexpr int D_TILES = d / MMA_N;
    constexpr int N_TILES = BLOCK_N / MMA_N;
    constexpr int K_TILES = d / MMA_K;
    constexpr int NK_TILES = BLOCK_N / MMA_K;

    uint32_t Q_reg[D_TILES][NUM_REG_A];
    uint32_t K_reg[N_TILES][K_TILES][NUM_REG_B];
    uint32_t V_reg[N_TILES][K_TILES][NUM_REG_B];
    uint32_t S_bf16[D_TILES][NUM_REG_A];
    float O_reg[D_TILES][NUM_REG_A] = {0.f};

    float scores_max[2] = {-INFINITY, -INFINITY};
    float scores_sum[2] = {0.0f, 0.0f};

    smem_load_128b<NThreads, Q_SHARD, d>(Q, d, Q_smem, d, tidx, offset_q);
    __syncthreads();

    uint32_t Q_shard_addr = cvta_shared(Q_smem);
    uint32_t K_shard_addr = cvta_shared(K_smem);
    uint32_t V_shard_addr = cvta_shared(V_smem);

    for (int mma_tile_id_k = 0; mma_tile_id_k < D_TILES; ++mma_tile_id_k) {
        const uint32_t Qs_addr = Q_shard_addr + (warp_id * Q_SHARD * d + mma_tile_id_k * MMA_K) * sizeof(nv_bfloat16);
        ldmatrix<NUM_REG_A>(Q_reg[mma_tile_id_k], Qs_addr);
    }

    for (int k_tile = 0; k_tile <= m_block; ++k_tile) {
        float S_reg[N_TILES][NUM_REG_C] = {0.0f};
        float scores_max_prev[2] = {scores_max[0], scores_max[1]};

        smem_load_128b<NThreads, BLOCK_N, d>(K, d, K_smem, d, tidx, offset_kv + k_tile * BLOCK_M * d);
        __syncthreads();

        for (int mma_tile_id_n = 0; mma_tile_id_n < N_TILES; ++mma_tile_id_n)
            for (int mma_tile_id_k = 0; mma_tile_id_k < K_TILES; ++mma_tile_id_k) {
                uint32_t Ks_addr = K_shard_addr + (mma_tile_id_n * MMA_N * d + mma_tile_id_k * MMA_K) * sizeof(nv_bfloat16);
                ldmatrix<NUM_REG_B>(K_reg[mma_tile_id_n][mma_tile_id_k], Ks_addr);
            }

        for (int mma_tile_id_n = 0; mma_tile_id_n < N_TILES; ++mma_tile_id_n)
            for (int mma_tile_id_k = 0; mma_tile_id_k < K_TILES; ++mma_tile_id_k)
                mma(Q_reg[mma_tile_id_k], K_reg[mma_tile_id_n][mma_tile_id_k], S_reg[mma_tile_id_k]);

        const float scale = 1.0f / sqrtf(static_cast<float>(d));
        const int query_pos = m_block * BLOCK_M + lane_id / 4;

        for (int mma_tile_id_n = 0; mma_tile_id_n < N_TILES; ++mma_tile_id_n) {
            const int key_pos = k_tile * BLOCK_N + mma_tile_id_n * MMA_N + (lane_id * 2) % 8;
            S_reg[mma_tile_id_n][0] = (key_pos <= query_pos) ? S_reg[mma_tile_id_n][0] * scale : -INFINITY;
            S_reg[mma_tile_id_n][1] = (key_pos + 1 <= query_pos) ? S_reg[mma_tile_id_n][1] * scale : -INFINITY;
            S_reg[mma_tile_id_n][2] = (key_pos <= query_pos + 8) ? S_reg[mma_tile_id_n][2] * scale : -INFINITY;
            S_reg[mma_tile_id_n][3] = (key_pos + 1 <= query_pos + 8) ? S_reg[mma_tile_id_n][3] * scale : -INFINITY;
        }

        for (int mma_tile_id_k = 0; mma_tile_id_k < D_TILES; ++mma_tile_id_k) {
            scores_max[0] = fmaxf(scores_max[0], S_reg[mma_tile_id_k][0]);
            scores_max[0] = fmaxf(scores_max[0], S_reg[mma_tile_id_k][1]);
            scores_max[1] = fmaxf(scores_max[1], S_reg[mma_tile_id_k][2]);
            scores_max[1] = fmaxf(scores_max[1], S_reg[mma_tile_id_k][3]);
        }

        scores_max[0] = fmaxf(scores_max[0], __shfl_xor_sync(0xffffffff, scores_max[0], 1));
        scores_max[0] = fmaxf(scores_max[0], __shfl_xor_sync(0xffffffff, scores_max[0], 2));
        scores_max[1] = fmaxf(scores_max[1], __shfl_xor_sync(0xffffffff, scores_max[1], 1));
        scores_max[1] = fmaxf(scores_max[1], __shfl_xor_sync(0xffffffff, scores_max[1], 2));

        float scale_top = __expf(scores_max_prev[0] - scores_max[0]);
        float scale_bottom = __expf(scores_max_prev[1] - scores_max[1]);
        scores_sum[0] *= scale_top;
        scores_sum[1] *= scale_bottom;

        for (int mma_tile_id_k = 0; mma_tile_id_k < D_TILES; ++mma_tile_id_k) {
            O_reg[mma_tile_id_k][0] *= scale_top;
            O_reg[mma_tile_id_k][1] *= scale_top;
            O_reg[mma_tile_id_k][2] *= scale_bottom;
            O_reg[mma_tile_id_k][3] *= scale_bottom;
        }

        smem_load_128b<NThreads, BLOCK_N, d>(V, d, V_smem, d, tidx, offset_kv + k_tile * BLOCK_M * d);
        __syncthreads();

        for (int mma_tile_id_n = 0; mma_tile_id_n < N_TILES; ++mma_tile_id_n) {
            S_reg[mma_tile_id_n][0] = expf(S_reg[mma_tile_id_n][0] - scores_max[0]);
            S_reg[mma_tile_id_n][1] = expf(S_reg[mma_tile_id_n][1] - scores_max[0]);
            S_reg[mma_tile_id_n][2] = expf(S_reg[mma_tile_id_n][2] - scores_max[1]);
            S_reg[mma_tile_id_n][3] = expf(S_reg[mma_tile_id_n][3] - scores_max[1]);
            scores_sum[0] += S_reg[mma_tile_id_n][0] + S_reg[mma_tile_id_n][1];
            scores_sum[1] += S_reg[mma_tile_id_n][2] + S_reg[mma_tile_id_n][3];

            // https://github.com/gau-nernst/learn-cuda/blob/main/07_attention/main.py
            nv_bfloat162 *S16_regs = reinterpret_cast<nv_bfloat162 *>(S_bf16[mma_tile_id_n / 2]);
            S16_regs[(mma_tile_id_n % 2) * 2]     = __float22bfloat162_rn({S_reg[mma_tile_id_n][0], S_reg[mma_tile_id_n][1]});
            S16_regs[(mma_tile_id_n % 2) * 2 + 1] = __float22bfloat162_rn({S_reg[mma_tile_id_n][2], S_reg[mma_tile_id_n][3]});

        }

        scores_sum[0] += __shfl_xor_sync(0xffffffff, scores_sum[0], 1);
        scores_sum[0] += __shfl_xor_sync(0xffffffff, scores_sum[0], 2);
        scores_sum[1] += __shfl_xor_sync(0xffffffff, scores_sum[1], 1);
        scores_sum[1] += __shfl_xor_sync(0xffffffff, scores_sum[1], 2);

        for (int mma_tile_id_n = 0; mma_tile_id_n < NK_TILES; ++mma_tile_id_n)
            for (int mma_tile_id_k = 0; mma_tile_id_k < D_TILES; ++mma_tile_id_k) {
                uint32_t Vs_addr = V_shard_addr + (mma_tile_id_n * MMA_N * d + mma_tile_id_k * MMA_K) * sizeof(nv_bfloat16);
                ldmatrix<NUM_REG_B>(V_reg[mma_tile_id_n][mma_tile_id_k], Vs_addr);
            }

        for (int mma_tile_id_n = 0; mma_tile_id_n < D_TILES; ++mma_tile_id_n)
            for (int mma_tile_id_k = 0; mma_tile_id_k < NK_TILES; ++mma_tile_id_k)
                mma(S_bf16[mma_tile_id_k], V_reg[mma_tile_id_n][mma_tile_id_k], O_reg[mma_tile_id_k]);
    }

    for (int mma_tile_id_k = 0; mma_tile_id_k < D_TILES; ++mma_tile_id_k) {
        O_reg[mma_tile_id_k][0] /= scores_sum[0];
        O_reg[mma_tile_id_k][1] /= scores_sum[0];
        O_reg[mma_tile_id_k][2] /= scores_sum[1];
        O_reg[mma_tile_id_k][3] /= scores_sum[1];
        nv_bfloat16* O_frags = O + (warp_id * Q_SHARD + lane_id / 4) * d + mma_tile_id_k * MMA_N + ((lane_id * 2) % 8);

        ushort2 tmp;
        tmp.x = __bfloat16_as_ushort(__float2bfloat16(O_reg[mma_tile_id_k][0]));
        tmp.y = __bfloat16_as_ushort(__float2bfloat16(O_reg[mma_tile_id_k][1]));
        reinterpret_cast<ushort2*>(O_frags)[0] = tmp;
        
        tmp.x = __bfloat16_as_ushort(__float2bfloat16(O_reg[mma_tile_id_k][2]));
        tmp.y = __bfloat16_as_ushort(__float2bfloat16(O_reg[mma_tile_id_k][3]));
        reinterpret_cast<ushort2*>(O_frags + 8 * d)[0] = tmp;
    }
}

void flashattn_v1(const nv_bfloat16 *Q, 
                const nv_bfloat16 *K, 
                const nv_bfloat16 *V, 
                nv_bfloat16* O, 
                int bh, 
                int seqlen_q, 
                int d) {
    const int BLOCK_M = 64, BLOCK_N = 64;
    const int NUM_WARPS = 4;

    const int NThreads = NUM_WARPS * WARP_SIZE;
    const int smem_size = (BLOCK_M + 2 * BLOCK_N) * d * sizeof(nv_bfloat16);

    dim3 grid(cdiv(seqlen_q, BLOCK_M), bh);

    auto kernel = flashattn_kernel_v1<BLOCK_M, BLOCK_N, NUM_WARPS, 128>;

    if (smem_size >= 48 * 1024) {
        cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
    }
    kernel<<<grid, NThreads, smem_size>>>(Q, K, V, O, bh, seqlen_q);
}
