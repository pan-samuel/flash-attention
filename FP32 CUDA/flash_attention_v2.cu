#include <cmath>
#include <cuda_runtime.h>
#include <stdio.h>

constexpr int WARP_SIZE = 32;
constexpr int d = 64;

__host__ __device__ inline constexpr int cdiv(int a, int b) { return (a + b - 1) / b; }

template<int BLOCK_SIZE, int Br, int Bc> 
__global__ void flashattn_kernel_v2(
    const float *Q, //  [B, nh, T, head_dim]
    const float *K, //  [B, nh, T, head_dim]
    const float *V, //  [B, nh, T, head_dim]
    float *__restrict__ O, //  [B, nh, T, head_dim]
    float *l, // norm intermediate [B, nh, T]
    float *m, // Row max intermediate [B, nh, T]
    int B, int nh, int T, 
    float scale // Scale factor (usually 1/sqrt(head_dim))
) { 
    int tid = threadIdx.x;
    int batch_head_id = blockIdx.x;

    int s_row = tid / Bc;
    int s_col = tid % Bc;

    const int Tc = cdiv(T, Bc); const int Tr = cdiv(T, Br);

    const int QKV_head_offset = batch_head_id * T * d;
    const int lm_head_offset = batch_head_id * T;

    const int d_per_thread = d / Bc;

    __shared__ float Q_smem[Br * d];
    __shared__ float K_smem[Bc * d];
    __shared__ float V_smem[Bc * d];
    __shared__ float S_ij_smem[Br * Bc];
    __shared__ float O_smem[Br * d];
    __shared__ float m_i[Br];
    __shared__ float l_i[Br];

    for (int j = 0; j < Tc; ++j) {
        // Load K and V tiles into shared memory
        for (int k_tile = 0; k_tile < cdiv(Bc * d, BLOCK_SIZE); ++k_tile) {
            int idx = k_tile * BLOCK_SIZE + tid;
            if (idx < Bc * d) {
                int kv_row = idx / d;
                int kv_col = idx % d;
                if (j * Bc + kv_row < T) {
                    K_smem[kv_row * d + kv_col] = K[QKV_head_offset + (j * Bc + kv_row) * d + kv_col];
                    V_smem[kv_row * d + kv_col] = V[QKV_head_offset + (j * Bc + kv_row) * d + kv_col];
                }
            }
        }
        __syncthreads();

        // For causal attention, we only process blocks where i >= j
        for (int i = j; i < Tr; ++i) {
            // Load Q and O tiles into shared memory
            for (int k_tile = 0; k_tile < cdiv(Br * d, BLOCK_SIZE); ++k_tile) {
                int idx = k_tile * BLOCK_SIZE + tid;
                if (idx < Br * d) {
                    int qo_row = idx / d;
                    int qo_col = idx % d; 
                    if (i * Br + qo_row < T) {
                        Q_smem[qo_row * d + qo_col] = Q[QKV_head_offset + (i * Br + qo_row) * d + qo_col];
                        O_smem[qo_row * d + qo_col] = O[QKV_head_offset + (i * Br + qo_row) * d + qo_col];
                    }
                }
            }
            __syncthreads();

            // Load m_i and l_i for current block
            if (s_col == 0) {
                int global_row = i * Br + s_row;
                m_i[s_row] = m[lm_head_offset + global_row];
                l_i[s_row] = l[lm_head_offset + global_row];
            }
            __syncthreads();

            // --- Compute S_ij = Q_i * K_j^T ---
            // Each thread computes one element of the Br x Bc S_ij tile
            float s_ij_val = 0.0f;
            for (int k = 0; k < d; k++) {
                s_ij_val += Q_smem[s_row * d + k] * K_smem[s_col * d + k];
            }
            s_ij_val *= scale;

            // Apply causal masking
            int query_pos = i * Br + s_row;
            int key_pos = j * Bc + s_col;
            if (key_pos > query_pos) {
                s_ij_val = -INFINITY;
            }
            
            S_ij_smem[s_row * Bc + s_col] = s_ij_val;
            __syncthreads();

            float m_ij = -INFINITY;
            for (int k = 0; k < Bc; k++) {
                m_ij = fmaxf(m_ij, S_ij_smem[s_row * Bc + k]);
            }

            float p_ij_val = __expf(S_ij_smem[s_row * Bc + s_col] - m_ij);
            S_ij_smem[s_row * Bc + s_col] = p_ij_val;
            __syncthreads();

            float l_ij = 0.0f;
            for (int k = 0; k < Bc; k++) {
                l_ij += S_ij_smem[s_row * Bc + k];
            }

            float m_ii = m_i[s_row];
            float l_ii = l_i[s_row];
            
            float m_new = fmaxf(m_ij, m_ii);
            float l_new = __expf(m_ii - m_new) * l_ii + __expf(m_ij - m_new) * l_ij; 
            
            for (int k = 0; k < d_per_thread; k++) {
                int col = s_col + k * Bc;
                float pv_acc = 0.0f;
                for (int inner_k = 0; inner_k < Bc; inner_k++) {
                    pv_acc += S_ij_smem[s_row * Bc + inner_k] * V_smem[inner_k * d + col];
                }
                
                // Update output with the running computation
                float o_old = O_smem[s_row * d + col];
                float o_new = (1.0f / l_new) * (l_ii * __expf(m_ii - m_new) * o_old \
                                                + __expf(m_ij- m_new) * pv_acc);
                O[QKV_head_offset + (i * Br + s_row) * d + col] = o_new; 
            }

            m[lm_head_offset + (i * Br) + s_row] = m_new;
            l[lm_head_offset + (i * Br) + s_row] = l_new;
            __syncthreads();
        }
        __syncthreads();
    }
} 

void flashattn_v2(const float *Q, const float *K, const float *V, float *O,
                  float *l, float *m, 
                  int B, int nh, int T, int d) {
    const int Bc = 32; const int Br = 32;
    const int BLOCK_SIZE = Bc * Br;
    const float scale = 1.0 / sqrt(d);    

    dim3 grid_dim(B * nh);  // batch_size x num_heads
    dim3 block_dim(BLOCK_SIZE);

    flashattn_kernel_v2<BLOCK_SIZE, Br, Bc><<<grid_dim, block_dim>>>(
        Q, K, V, O, l, m, B, nh, T, scale
    );
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error after kernel launch: %s\n", cudaGetErrorString(err));
    }
}