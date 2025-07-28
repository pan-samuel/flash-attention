#include <cmath>
#include <cuda_runtime.h>
#include <stdio.h>

constexpr int WARP_SIZE = 32;
constexpr int d = 64;

__host__ __device__ inline constexpr int cdiv(int a, int b) { return (a + b - 1) / b; }

template<int BLOCK_SIZE, int Br, int Bc>
__global__ __launch_bounds__(BLOCK_SIZE, 1)
void flashattn_kernel_v3(
    const float *Q, // [B, nh, T, head_dim]
    const float *K, // [B, nh, T, head_dim]
    const float *V, // [B, nh, T, head_dim]
    float *__restrict__ O, // [B, nh, T, head_dim]
    int B, int nh, int T,
    float scale // Scale factor (usually 1/sqrt(head_dim))
) {
    int tid = threadIdx.x;
    int batch_head_id = blockIdx.x;
    int i = blockIdx.y; // output tile index, this block computes O_i

    int s_row = tid / Bc; // This thread's row within the Br x Bc tile
    int s_col = tid % Bc; // This thread's col within the Br x Bc tile

    const int Tc = cdiv(T, Bc);

    const int batch_head_offset = batch_head_id * T * d;
    const int QO_offset = batch_head_offset + i * Br * d;

    // --- Shared Memory Declarations ---
    __shared__ float Q_smem[Br * d];
    __shared__ float K_smem[Bc * d];
    __shared__ float V_smem[Bc * d];
    __shared__ float S_ij_smem[Br * Bc];
    
    __shared__ float m_i[Br];
    __shared__ float l_i[Br];

    // Use registers to accumulate this thread's output values
    // Assuming d=64, Bc=32. Each thread handles d/Bc = 64/32 = 2 elements of the output vector.
    const int d_per_thread = d / Bc;
    float O_reg[d_per_thread]; 
    for (int k = 0; k < d_per_thread; k++) {
        O_reg[k] = 0.0f;
    }

    // One thread from each row initializes the stats for that row.
    if (s_col == 0) {
        m_i[s_row] = -INFINITY;
        l_i[s_row] = 0.0f;
    }

    for (int tid_offset = 0; tid_offset < cdiv(Br * d, BLOCK_SIZE); ++tid_offset) {
        int idx = tid_offset * BLOCK_SIZE + tid;
        if (idx < Br * d) {
            int q_row = idx / d;
            int q_col = idx % d;
            if (i * Br + q_row < T) {
                Q_smem[q_row * d + q_col] = Q[QO_offset + q_row * d + q_col];
            }
        }
    }
    __syncthreads();

    for (int j = 0; j <= i; ++j) {
        for (int tid_offset = 0; tid_offset < cdiv(Bc * d, BLOCK_SIZE); ++tid_offset) {
            int idx = tid_offset * BLOCK_SIZE + tid;
            if (idx < Bc * d) {
                int kv_row = idx / d;
                int kv_col = idx % d;
                if (j * Bc + kv_row < T) {
                    K_smem[kv_row * d + kv_col] = K[batch_head_offset + (j * Bc + kv_row) * d + kv_col];
                    V_smem[kv_row * d + kv_col] = V[batch_head_offset + (j * Bc + kv_row) * d + kv_col];
                } 
            }
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

        // --- Online Softmax Update ---
        // 1. Find the max of the current tile's row (m_ij)
        float m_ij = -INFINITY;
        for (int k = 0; k < Bc; k++) {
            m_ij = fmaxf(m_ij, S_ij_smem[s_row * Bc + k]);
        }

        // 3. Compute P_ij = exp(S_ij - m_ij) and the row-sum l_ij
        float p_ij_val = __expf(S_ij_smem[s_row * Bc + s_col] - m_ij);
        S_ij_smem[s_row * Bc + s_col] = p_ij_val; // Overwrite S with P
        __syncthreads();
        
        float l_ij = 0.0f;
        for (int k = 0; k < Bc; k++) {
            l_ij += S_ij_smem[s_row * Bc + k];
        }

        // 2. Load old statistics and compute new max
        float m_ii = m_i[s_row];
        float l_ii = l_i[s_row];

        float m_new = fmaxf(m_ii, m_ij);
        float l_new = __expf(m_ii - m_new) * l_ii + __expf(m_ij - m_new) * l_ij; 

        
        // Add the new value component: (1/l_new) * P_ij * V_j
        for (int k = 0; k < d_per_thread; k++) {
            int col = s_col + k * Bc; // This thread's column in the V matrix
            float pv_acc = 0.0f;
            for (int inner_k = 0; inner_k < Bc; inner_k++) { // Dot product of P_ij row and V_j column
                pv_acc += S_ij_smem[s_row * Bc + inner_k] * V_smem[inner_k * d + col];
            }
            float o_old = O_reg[k];
            float o_new = (1.0f / l_new) * (l_ii * __expf(m_ii - m_new) * o_old \
                                            + __expf(m_ij- m_new) * pv_acc);
            O_reg[k] = o_new; 
        }
        if (s_col == 0) {
            m_i[s_row] = m_new;
            l_i[s_row] = l_new;
        }
        __syncthreads();
    } // end of j loop

    // Write final accumulated output to global memory
    int global_row = i * Br + s_row;
    if (global_row < T) {
        for (int k = 0; k < d_per_thread; k++) {
            int global_col = s_col + k * Bc;
            O[batch_head_offset + global_row * d + global_col] = O_reg[k];
        }
    }
}


// Host LAUNCHER function
void flashattn_v3(const float *Q, const float *K, const float *V, float *O,
                  float *l, float *m, 
                  int B, int nh, int T, int d) {
    const int Bc = 32; 
    const int Br = 32;
    const int BLOCK_SIZE = Bc * Br;
    const float scale = 1.0 / sqrt(d);
    
    dim3 grid_dim(B * nh, cdiv(T, Br)); // batch_size x num_heads x (T / Br)
    dim3 block_dim(BLOCK_SIZE);
    
    flashattn_kernel_v3<BLOCK_SIZE, Br, Bc><<<grid_dim, block_dim>>>(
        Q, K, V, O, B, nh, T, scale
    );
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error after kernel launch: %s\n", cudaGetErrorString(err));
    }
}
