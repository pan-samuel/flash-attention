#include <cmath>
#include <cuda_runtime.h>
#include <stdio.h>

__host__ __device__ inline constexpr int cdiv(int a, int b) { return (a + b - 1) / b; }

template<int BLOCK_SIZE, int Br, int Bc> 
__global__ void flashattn_kernel_v1(
    const float *Q, //  [B, nh, T, head_dim]
    const float *K, //  [B, nh, T, head_dim]
    const float *V, //  [B, nh, T, head_dim]
    float *__restrict__ O, //  [B, nh, T, head_dim]
    float *l, // norm intermediate [B, nh, T]
    float *m, // Row max intermediate [B, nh, T]
    int B, int nh, int T, int d, 
    float scale // Scale factor (usually 1/sqrt(head_dim))
) { 
    int tid = threadIdx.x;
    int batch_id = blockIdx.x;
    int head_id = blockIdx.y;

    const int Tc = cdiv(T, Bc); const int Tr = cdiv(T, Br);

    const int QKV_head_offset = batch_id * (nh * T * d) + head_id * (T * d);
    const int lm_head_offset = batch_id * (nh * T) + head_id * T;

    extern __shared__ float smem[];
    int Bc_tile_size = Bc * d;
    int Br_tile_size = Br * d;

    float* Q_smem = smem;
    float* K_smem = &smem[Br_tile_size];
    float* V_smem = &smem[Br_tile_size + Bc_tile_size];
    float* S_ij_smem = &smem[Br_tile_size + 2 * Bc_tile_size];

    for (int j = 0; j < Tc; ++j) {
        int KV_tile_row = QKV_head_offset + j * Bc_tile_size;

        if (j * Bc + tid < T) {
            for (int x = 0; x < d; ++x) {
                K_smem[tid * d + x] = K[KV_tile_row + tid * d + x];
                V_smem[tid * d + x] = V[KV_tile_row + tid * d + x];
            }
        }
        __syncthreads();

        for (int i = j; i < Tr; ++i) {
            if (i * Br + tid >= T) 
                break;
                
            int Q_tile_row = QKV_head_offset + i * Br_tile_size;
            int lm_tile_row = lm_head_offset + i * Br;
            
            for (int x = 0; x < d; ++x) {
                Q_smem[tid * d + x] = Q[Q_tile_row + tid * d + x];
            }
            
            float m_i = m[lm_tile_row + tid];
            float l_i = l[lm_tile_row + tid];

            float m_ij = -INFINITY;
            for (int x = 0; x < Bc; ++x) {
                if (j * Bc + x >= T)
                    break;
                    
                float sum = 0.0f;
                for (int y = 0; y < d; ++y) {
                    sum += Q_smem[tid * d + y] * K_smem[x * d + y];
                }
                sum *= scale;
                
                if (i * Br + tid < j * Bc + x) {
                    sum = -INFINITY;
                }
                
                S_ij_smem[(tid * Bc) + x] = sum;

                if (sum > m_ij) {
                    m_ij = sum;
                }
            }

            float l_ij = 0;
            for (int x = 0; x < Bc; ++x) {
                if (j * Bc + x >= T)
                    break;
                    
                if (i * Br + tid < j * Bc + x) {
                    S_ij_smem[tid * Bc + x] = 0.0f;
                } else {
                    S_ij_smem[tid * Bc + x] = __expf(S_ij_smem[tid * Bc + x] - m_ij);
                }
                l_ij += S_ij_smem[tid * Bc + x];
            }

            float m_new = max(m_i, m_ij);
            float l_new = __expf(m_i - m_new) * l_i + __expf(m_ij - m_new) * l_ij;

            float alpha = __expf(m_i - m_new);
            float beta = __expf(m_ij - m_new);

            for (int x = 0; x < d; ++x) {
                float PV_acc = 0.0f;
                for (int y = 0; y < Bc; ++y) {
                    if (j * Bc + y >= T)
                        break;
                    PV_acc += S_ij_smem[(tid * Bc) + y] * V_smem[(y * d) + x];
                }
                O[Q_tile_row + (tid * d) + x] = (1.0f / l_new) * (l_i * alpha * O[Q_tile_row + (tid * d) + x] + beta * PV_acc);
            }
            m[lm_tile_row + tid] = m_new;
            l[lm_tile_row + tid] = l_new;
        }
        __syncthreads();
    }
} 

// Add this function to check shared memory requirements
bool check_shared_memory_requirements(size_t required_shared_mem) {
    int device;
    cudaError_t err = cudaGetDevice(&device);
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, device);

    // Get shared memory per block
    size_t shared_mem_per_block = deviceProp.sharedMemPerBlock;
    
    if (required_shared_mem > shared_mem_per_block) {
        printf("ERROR: Required shared memory (%zu bytes) exceeds device limit (%zu bytes)\n", 
               required_shared_mem, shared_mem_per_block);
        return false;
    }
    
    return true;
}

// Host LAUNCHER function
void flashattn_v1(const float *Q, const float *K, const float *V, float *O,
                  float *l, float *m, 
                  int B, int nh, int T, int d) {
    const int Bc = 32; const int Br = 32;
    const float scale = 1.0 / sqrt(d);
    const int sram_size = (2 * Bc * d * sizeof(float)) + (Br * d * sizeof(float)) + (Bc * Br * sizeof(float));
    
    if (!check_shared_memory_requirements(sram_size)) {
        printf("Not enough shared memory");
        return;
    }

    dim3 grid_dim(B, nh);  // batch_size x num_heads
    dim3 block_dim(Bc);

    flashattn_kernel_v1<Bc, Br, Bc><<<grid_dim, block_dim, sram_size>>>(
        Q, K, V, O, l, m, B, nh, T, d, scale
    );
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error after kernel launch: %s\n", cudaGetErrorString(err));
        // Consider throwing an exception
    }
}
