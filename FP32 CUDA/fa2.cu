#include <cuda_runtime.h>
#include <cmath>

__host__ __device__ inline constexpr int cdiv(int a, int b) { return (a + b - 1) / b; }

template <int BLOCK_SIZE, int HEIGHT, int WIDTH>
__device__ void load_shmem_vectorized(const float *in, int in_row_stride, float *out, int out_row_stride, int tid, int global_offset) {
    for (int offset = 0; offset < HEIGHT * WIDTH; offset += BLOCK_SIZE * 4) {
        const int idx = offset + tid * 4;
        const int row = idx / WIDTH;
        const int col = idx % WIDTH;

        float4 tmp = reinterpret_cast<const float4 *>(&in[global_offset + row * in_row_stride + col])[0];
        float* dst = &out[row * out_row_stride + col];
        dst[0] = tmp.x;
        dst[1] = tmp.y;
        dst[2] = tmp.z;
        dst[3] = tmp.w;
    }
}

template <int BLOCK_SIZE, int Br, int Bc, int Bk, int TM, int TN>
__global__ void fa2_kernel(
    const float*__restrict__ Q,
    const float*__restrict__ K,
    const float*__restrict__ V,
    float*__restrict__ O,
    float scale,
    int T,
    int d)
{
    const int tid = threadIdx.x;
    const int batch_head_id = blockIdx.x;
    const int i = blockIdx.y;

    const int batch_head_offset = batch_head_id * T * d;
    const int QO_offset = batch_head_offset + i * Br * d;

    __shared__ float Q_smem[Br][64];
    __shared__ float K_smem[Bc][Bk + 1];
    __shared__ float V_smem[Bc][Bk];
    __shared__ float S_ij_smem[Br][Bc + 1];

    constexpr int num_threads_per_row = Bc / TN;
    int tile_thread_row_id = tid / num_threads_per_row;
    int tile_thread_col_id = tid % num_threads_per_row;

    const int Tc = cdiv(T, Bc);

    float l[TM] = {0.f};
    float m_i[TM];
    float m[TM];
    for (int i = 0; i < TM; ++i) m[i] = -INFINITY;

    float O_i[2 * TM * TN] = {0.f};

    load_shmem_vectorized<BLOCK_SIZE, Br, 64>(Q, 64, (float*)Q_smem, 64, tid, QO_offset);
    __syncthreads();

    float Q_reg[TM];
    float K_reg[TN];
    float V_reg[TN];

    for (int j = 0; j < Tc && j <= i; ++j) {
        float acc[TM][TN] = {0.f};
        for (int tile = 0; tile < 2; ++tile) {
            load_shmem_vectorized<BLOCK_SIZE, Br, Bk>(K, 64, (float*)K_smem, Bk+1, tid, batch_head_offset + (j * Bc) * d + tile * Bk);
            __syncthreads();

            for (int k = 0; k < Bk; ++k) {
                for (int mm = 0; mm < TM; ++mm)
                    Q_reg[mm] = Q_smem[tile_thread_row_id * TM + mm][k + tile * Bk];

                for (int nn = 0; nn < TN; ++nn)
                    K_reg[nn] = K_smem[tile_thread_col_id * TN + nn][k];

                for (int mm = 0; mm < TM; ++mm)
                    for (int nn = 0; nn < TN; ++nn)
                        acc[mm][nn] += Q_reg[mm] * K_reg[nn];
            }
            __syncthreads();, 
        }

        for (int mm = 0; mm < TM; ++mm) {
            for (int nn = 0; nn < TN; ++nn) {
                int key_pos = j * Bc + tile_thread_col_id * TN + nn;
                int query_pos  = i * Br + tile_thread_row_id * TM + mm;
                S_ij_smem[tile_thread_row_id * TM + mm][tile_thread_col_id * TN + nn] =
                    (key_pos <= query_pos) ? acc[mm][nn] * scale : -INFINITY;
            }
        }
        __syncthreads();

        for (int mm = 0; mm < TM; ++mm) {
            m_i[mm] = m[mm];
            float m_ij= m[mm];
            for (int k = 0; k < Bc; ++k) {
                float val = S_ij_smem[tile_thread_row_id * TM + mm][k];
                if (m_ij < val) m_ij = val;
            }
            m[mm] = m_ij;
        }


        for (int tile = 0; tile < 2; ++tile) {
            for (int mm = 0; mm < TM; ++mm) { 
                float scale_ = expf(m_i[mm] - m[mm]);
                if (tile == 0) l[mm] *= scale_;
                for (int nn = 0; nn < TN; ++nn) {
                    O_i[tile * TM * TN + mm * TN + nn] *= scale_;
                }
            }
        }

        for (int tile = 0; tile < 2; ++tile) {
            load_shmem_vectorized<BLOCK_SIZE, Br, Bk>(V, 64, (float*)V_smem, Bk, tid, batch_head_offset + (j * Bc) * d + tile * Bk);
            __syncthreads();

            for (int k = 0; k < Bc; ++k) {
                for (int p_m = 0; p_m < TM; ++p_m) {
                    Q_reg[p_m] = expf(S_ij_smem[tile_thread_row_id * TM + p_m][k] - m[p_m]);
                    if (tile == 0) {
                        l[p_m] += Q_reg[p_m];
                    }
                }
                for (int v_n = 0; v_n < TN; ++v_n)
                    V_reg[v_n] = V_smem[k][tile_thread_col_id * TN + v_n];

                for (int p_m = 0; p_m < TM; ++p_m)
                    for (int v_n = 0; v_n < TN; ++v_n)
                        O_i[tile * TM * TN + p_m * TN + v_n] += Q_reg[p_m] * V_reg[v_n];
            }
            __syncthreads();
        }
    }

    for (int tile = 0; tile < 2; ++tile) {
        for (int mm = 0; mm < TM; ++mm) {
            for (int nn = 0; nn < TN; ++nn) {
                int out_idx = (i * Br + tile_thread_row_id * TM + mm) * d +
                              tile * Bc + tile_thread_col_id * TN + nn;
                if (out_idx < T * d) {
                    O[batch_head_offset + out_idx] = O_i[tile * TM * TN + mm * TN + nn] / l[mm];
                }
            }
        }
    }
}

void flashattn2(const float* Q, const float* K, const float* V, float* O, float* m, float* l, int B, int nh, int T, int d)
{
    constexpr int Bc = 32, Br = 32, Bk = 32;
    constexpr int TM = 4, TN = 4;
    const int BLOCK_SIZE = Bc * Br / (TM * TN);

    dim3 block_dim(BLOCK_SIZE);
    dim3 grid_dim(B * nh, cdiv(T, Br));
    float scale = 1.0 / sqrt(d);

    fa2_kernel<BLOCK_SIZE, Br, Bc, Bk, TM, TN><<<grid_dim, block_dim>>>(Q, K, V, O, scale, T, d);
    cudaError_t err = cudaGetLastError();
}
