#include <cmath>
#include <cuda_runtime.h>
#include <stdio.h>

__host__ __device__ inline constexpr int cdiv(int a, int b) { return (a + b - 1) / b; }

template<int BLOCK_SIZE, int HEIGHT, int WIDTH>
__device__ void load_shmem_vectorized(const float* in, int in_row_stride, float* out, int out_row_stride, int tid, int global_offset) {
    for (int offset = 0; offset < HEIGHT * WIDTH; offset += BLOCK_SIZE * 4) {
        const int idx = offset + tid * 4;
        const int row = idx / WIDTH;
        const int col = idx % WIDTH;
        float4 tmp = reinterpret_cast<const float4*>(&in[global_offset + row * in_row_stride + col])[0];
        float* dst = &out[row * out_row_stride + col];
        dst[0] = tmp.x; dst[1] = tmp.y; dst[2] = tmp.z; dst[3] = tmp.w;
    }
}

template<int BLOCK_SIZE, int Br, int Bc, int Bk, int TM, int TN, int D_TILES = 2>
__global__ void flashattn_kernel_v7(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    float* __restrict__ O,
    int B, int T, int d, float scale)
{
    int tid = threadIdx.x;
    int i = blockIdx.x;
    int batch_head_id = blockIdx.y;

    const int batch_head_offset = batch_head_id * T * d;
    const int QO_offset = batch_head_offset + i * Br * d;

    const int num_threads_per_row = Bc / TN;
    const int tile_thread_row_id = tid / num_threads_per_row;
    const int tile_thread_col_id = tid % num_threads_per_row;

    __shared__ float Q_smem[Br][64];
    __shared__ float K_smem[Bc][Bk + 1];
    __shared__ float V_smem[Bc][Bk];
    __shared__ float S_ij_smem[Br][Bc + 1];

    float m[TM];
    float l[TM];
    float m_i[TM];
    float l_i[TM] = {0.f};
    for (int ii = 0; ii < TM; ++ii) { m[ii] = -INFINITY; m_i[ii] = -INFINITY; l[ii] = 0.f; }

    float Q_reg[TM], K_reg[TN], V_reg[TN];
    float O_reg[D_TILES * TM * TN] = {0.f};

    load_shmem_vectorized<BLOCK_SIZE, Br, 64>(Q, 64, (float*)Q_smem, 64, tid, QO_offset);
    __syncthreads();

    for (int j = 0; j <= i; ++j) {
        for (int mm = 0; mm < TM; ++mm) { m[mm] = -INFINITY; l[mm] = 0.f; } 

        float m_new[TM], l_new[TM];
        float acc[TM][TN] = {0.f};

        for (int k_tile = 0; k_tile < D_TILES; ++k_tile) {
            load_shmem_vectorized<BLOCK_SIZE, Br, Bk>(K, 64, (float*)K_smem, Bk + 1, tid, batch_head_offset + (j * Bc) * d + k_tile * Bk);
            __syncthreads();

            for (int k = 0; k < Bk; ++k) {
                for (int mm = 0; mm < TM; ++mm) Q_reg[mm] = Q_smem[tile_thread_row_id * TM + mm][k + k_tile * Bk];
                for (int nn = 0; nn < TN; ++nn) K_reg[nn] = K_smem[tile_thread_col_id * TN + nn][k];
                for (int mm = 0; mm < TM; ++mm)
                    for (int nn = 0; nn < TN; ++nn)
                        acc[mm][nn] += Q_reg[mm] * K_reg[nn];
            }
            __syncthreads();
        }

        for (int mm = 0; mm < TM; ++mm) {
            int query_pos = i * Br + tile_thread_row_id * TM + mm;
            for (int nn = 0; nn < TN; ++nn) {
                int key_pos = j * Bc + tile_thread_col_id * TN + nn;
                float s = (key_pos > query_pos) ? -INFINITY : acc[mm][nn] * scale;
                m[mm] = fmaxf(m[mm], s);
                acc[mm][nn] = s;
            }
        }

        for (int t = 1; t < 8; t <<= 1) {
            m[0] = fmaxf(m[0], __shfl_xor_sync(0xffffffff, m[0], t));
            m[1] = fmaxf(m[1], __shfl_xor_sync(0xffffffff, m[1], t));
            m[2] = fmaxf(m[2], __shfl_xor_sync(0xffffffff, m[2], t));
            m[3] = fmaxf(m[3], __shfl_xor_sync(0xffffffff, m[3], t));
        }

        for (int mm = 0; mm < TM; ++mm)
            for (int nn = 0; nn < TN; ++nn) {
                float p = __expf(acc[mm][nn] - m[mm]);
                S_ij_smem[tile_thread_row_id * TM + mm][tile_thread_col_id * TN + nn] = p;
                l[mm] += p;
            }

        for (int t = 1; t < 8; t <<= 1) {
            l[0] += __shfl_xor_sync(0xffffffff, l[0], t);
            l[1] += __shfl_xor_sync(0xffffffff, l[1], t);
            l[2] += __shfl_xor_sync(0xffffffff, l[2], t);
            l[3] += __shfl_xor_sync(0xffffffff, l[3], t);
        }

        for (int mm = 0; mm < TM; ++mm) {
            float mm_new = fmaxf(m_i[mm], m[mm]);
            float ll_new = __expf(m_i[mm] - mm_new) * l_i[mm] + __expf(m[mm] - mm_new) * l[mm];
            m_new[mm] = mm_new; l_new[mm] = ll_new;
        }

        for (int k_tile = 0; k_tile < D_TILES; ++k_tile) {
            load_shmem_vectorized<BLOCK_SIZE, Br, Bk>(V, 64, (float*)V_smem, Bk, tid, batch_head_offset + (j * Bc) * d + k_tile * Bk);
            __syncthreads();
            float pv_acc[TM][TN] = {0.f};

            for (int k = 0; k < Bc; ++k) {
                for (int mm = 0; mm < TM; ++mm) Q_reg[mm] = S_ij_smem[tile_thread_row_id * TM + mm][k];
                for (int nn = 0; nn < TN; ++nn) V_reg[nn] = V_smem[k][tile_thread_col_id * TN + nn];
                for (int mm = 0; mm < TM; ++mm)
                    for (int nn = 0; nn < TN; ++nn)
                        pv_acc[mm][nn] += Q_reg[mm] * V_reg[nn];
            }
            __syncthreads();

            for (int mm = 0; mm < TM; ++mm) {
                float m_ii = m_i[mm], l_ii = l_i[mm], m_ij = m[mm], mm_new = m_new[mm], ll_new = l_new[mm];
                for (int nn = 0; nn < TN; ++nn) {
                    float o_old = O_reg[k_tile * TM * TN + mm * TN + nn];
                    O_reg[k_tile * TM * TN + mm * TN + nn] = (1.f / ll_new) * (l_ii * __expf(m_ii - mm_new) * o_old + __expf(m_ij - mm_new) * pv_acc[mm][nn]);
                }
            }
        }

        for (int mm = 0; mm < TM; ++mm) { m_i[mm] = m_new[mm]; l_i[mm] = l_new[mm]; }
        __syncthreads();
    }

    for (int k_tile = 0; k_tile < D_TILES; ++k_tile)
        for (int mm = 0; mm < TM; ++mm) {
            int out_row = i * Br + tile_thread_row_id * TM + mm;
            for (int nn = 0; nn < TN; ++nn) {
                int out_col = k_tile * Bk + tile_thread_col_id * TN + nn;
                O[batch_head_offset + out_row * d + out_col] = O_reg[k_tile * TM * TN + mm * TN + nn];
            }
        }
}

void flashattn_v7(const float* Q, const float* K, const float* V, float* O,
                  float* l, float* m,
                  int B, int nh, int T, int d) {
    const int Bc = 32, Br = 32, Bk = 32;
    const int TM = 4, TN = 4;
    const int BLOCK_SIZE = Bc * Br / (TM * TN);
    const float scale = 1.0 / sqrtf((float)d);

    dim3 grid_dim(cdiv(T, Br), B * nh);
    dim3 block_dim(BLOCK_SIZE);

    flashattn_kernel_v7<BLOCK_SIZE, Br, Bc, Bk, TM, TN><<<grid_dim, block_dim>>>(
        Q, K, V, O, B, T, d, scale);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) printf("CUDA Error after kernel launch: %s\n", cudaGetErrorString(err));
}
