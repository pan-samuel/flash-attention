#include <cmath>
#include <cuda_runtime.h>
#include <stdio.h>

constexpr int WARP_SIZE = 32;
constexpr int d = 64;

__host__ __device__ inline constexpr int cdiv(int a, int b) { return (a + b - 1) / b; }

template<int BLOCK_SIZE, int Br, int Bc, int Bk, int TM, int TN, int TK> 
__global__ void flashattn_kernel_v4(
    const float *Q,
    const float *K,
    const float *V,
    float *__restrict__ O,
    int B, int nh, int T,
    float scale
) {
    static_assert(Br % TM == 0);
    static_assert(Bc % TN == 0);

    int tid = threadIdx.x;
    int batch_head_id = blockIdx.x;
    int i = blockIdx.y;

    const int num_threads_per_row = Bc / TN;         // 8
    const int num_threads_v_per_row = d / TK;         // 8
    const int tile_thread_row_id  = tid / num_threads_per_row;
    const int tile_thread_col_id  = tid % num_threads_per_row;
    const int tile_thread_col_v_id = tid % num_threads_v_per_row;
    
    const int rows_per_block = BLOCK_SIZE / Bk;
    const int k_row = tid / d;
    const int k_col = tid % d;
    const int strideK = BLOCK_SIZE / Bk;

    const int Tc = cdiv(T, Bc);

    const int batch_head_offset = batch_head_id * T * d;
    const int QO_offset  = batch_head_offset + i * Br * d;

    __shared__ float Q_smem[Br * d];
    __shared__ float K_smem[Bc * d];
    __shared__ float V_smem[Bc * d];
    __shared__ float S_ij_smem[Br * Bc];

    float m[TM];
    float l[TM]  = {0.f};
    float m_i[TM];
    float l_i[TM] = {0.f};

    #pragma unroll
    for (int ii = 0; ii < TM; ++ii){
        m[ii] = -INFINITY;
        m_i[ii] = -INFINITY;
    }

    float Q_reg[TM];
    float K_reg[TN];
    float V_reg[TK];

    float O_reg[TM * TK];
    for (int j = 0; j < TM * TK; j++)
        O_reg[j] = 0.f;

    const int q_col = tid % d;
    for (int q_row = 0; q_row < Br; ++q_row) {
        int g_idx = q_col + q_row * d;
        Q_smem[q_row * d + q_col] = Q[QO_offset + g_idx];
    }
    __syncthreads();

    /* ================= main causal block-scan over j ================ */
    for (int j = 0; j <= i; ++j) {
        float m_new[TM];
        float l_new[TM];
        float acc[TM][TN] = {0.f};

        for (int k_row = 0; k_row < Br; ++k_row) {
            int k_idx = (k_row + j * Bc) * d + k_col;
            K_smem[k_row * d + k_col] = K[batch_head_offset + k_idx];
        }
        __syncthreads();

        for (int k = 0; k < d; ++k) {
            #pragma unroll
            for (int mm = 0; mm < TM; ++mm)
                Q_reg[mm] = Q_smem[(tile_thread_row_id * TM + mm) * d + k];

            #pragma unroll
            for (int nn = 0; nn < TN; ++nn)
                K_reg[nn] = K_smem[(tile_thread_col_id * TN + nn) * d + k];

            #pragma unroll
            for (int mm = 0; mm < TM; ++mm)
                #pragma unroll
                for (int nn = 0; nn < TN; ++nn)
                    acc[mm][nn] += Q_reg[mm] * K_reg[nn];
        }
        
        /* ---- store to shared memory with causal mask & scale ---- */
        #pragma unroll
        for (int mm = 0; mm < TM; ++mm) {
            int query_pos = i * Br + tile_thread_row_id * TM + mm;
            #pragma unroll
            for (int nn = 0; nn < TN; ++nn) {
                int key_pos = j * Bc + tile_thread_col_id * TN + nn;
                float s_val = (key_pos > query_pos) ? -INFINITY : acc[mm][nn] * scale;
                S_ij_smem[(tile_thread_row_id * TM + mm) * Bc + tile_thread_col_id * TN + nn] = s_val;
            }
        }
        __syncthreads();

        /* ---- per-row softmax (max, exp, sum) ---- */
        #pragma unroll
        for (int mm = 0; mm < TM; ++mm) {
            float m_ij = -INFINITY;
            #pragma unroll
            for (int k = 0; k < Bc; ++k)
                m_ij = fmaxf(m_ij, S_ij_smem[(tile_thread_row_id * TM + mm) * Bc + k]);
            m[mm] = m_ij;
        }

        #pragma unroll
        for (int mm = 0; mm < TM; ++mm) {
            #pragma unroll
            for (int nn = 0; nn < TN; ++nn) {
                float p = __expf( S_ij_smem[(tile_thread_row_id * TM + mm) * Bc +
                                             tile_thread_col_id * TN + nn] - m[mm]);
                S_ij_smem[(tile_thread_row_id * TM + mm) * Bc +
                          tile_thread_col_id * TN + nn] = p;
            }
        }
        __syncthreads();

        #pragma unroll
        for (int mm = 0; mm < TM; ++mm) {
            float l_ij = 0.f;
            #pragma unroll
            for (int k = 0; k < Bc; ++k)
                l_ij += S_ij_smem[(tile_thread_row_id * TM + mm) * Bc + k];
            l[mm] = l_ij;
        }

        /* ---- combine old stats with new stats for this row ---- */
        #pragma unroll
        for (int mm = 0; mm < TM; ++mm) {
            float mm_new = fmaxf(m_i[mm], m[mm]);
            float ll_new = __expf(m_i[mm] - mm_new) * l_i[mm] +
                           __expf(m[mm]   - mm_new) * l[mm];
            m_new[mm] = mm_new;
            l_new[mm] = ll_new;
        }

        /* --- load V tile --- */
        for (int v_row = 0; v_row < Br; ++v_row) {
            int v_idx = (v_row + j * Bc) * d + k_col;
            V_smem[v_row * d + k_col] = V[batch_head_offset + v_idx];
        }
        __syncthreads();
        float pv_acc[TM][TK] = {0.f};

        #pragma unroll
        for (int k = 0; k < Bc; ++k) {
            #pragma unroll
            for (int mm = 0; mm < TM; ++mm) {
                Q_reg[mm] = S_ij_smem[(tile_thread_row_id * TM + mm) * Bc + k]; //Save some registers, don't know the full calcs yet
            }
            #pragma unroll
            for (int nn = 0; nn < TK; ++nn) {
                V_reg[nn] = V_smem[k * d + tile_thread_col_v_id * TK + nn];
            }

            #pragma unroll
            for (int mm = 0; mm < TM; ++mm) {
                #pragma unroll
                for (int nn = 0; nn < TK; ++nn) {
                    pv_acc[mm][nn] += Q_reg[mm] * V_reg[nn];
                }
            }   
        }
        __syncthreads();
        #pragma unroll
        for (int mm = 0; mm < TM; ++mm) {
            float m_ii = m_i[mm];
            float l_ii = l_i[mm];
            float m_ij = m[mm];
            float mm_new = m_new[mm];
            float ll_new = l_new[mm];

            #pragma unroll
            for (int nn = 0; nn < TK; ++nn) {
                float o_old = O_reg[mm * TK + nn];
                float o_new = (1.0f / ll_new) * (l_ii * __expf(m_ii - mm_new) * o_old \
                                        + __expf(m_ij- mm_new) * pv_acc[mm][nn]);
                O_reg[mm * TK + nn] = o_new;
            }
        }
    
        for (int mm = 0; mm < TM; ++mm) {
            m_i[mm] = m_new[mm];
            l_i[mm] = l_new[mm];
        }
        __syncthreads();
    }     

    #pragma unroll
    for (int mm = 0; mm < TM; ++mm) {
        int out_row = i * Br + tile_thread_row_id * TM + mm;
        #pragma unroll
        for (int nn = 0; nn < TK; ++nn) {
            int out_col = tile_thread_col_v_id * TK + nn;
            O[batch_head_offset + out_row * d + out_col] = O_reg[mm * TK + nn];
        }
    }

}


void flashattn_v4(const float *Q, const float *K, const float *V, float *O,
                float *l, float *m, 
                int B, int nh, int T, int d) {
    const int Bc = 32, Br = 32, Bk = 32;
    const int TM = 4, TN = 4, TK = 8;
    const int BLOCK_SIZE = Bc * Br / (TM * TN); //  64
    const float scale = 1.0 / sqrt(d);    

    dim3 grid_dim(B * nh, cdiv(T, Br));  
    dim3 block_dim(BLOCK_SIZE);

    flashattn_kernel_v4<BLOCK_SIZE, Br, Bc, Bk, TM, TN, TK><<<grid_dim, block_dim>>>(
        Q, K, V, O, B, nh, T, scale
    );
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error after kernel launch: %s\n", cudaGetErrorString(err));
    }
}
