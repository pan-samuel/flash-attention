#include <torch/extension.h>
#include <cuda_bf16.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) \
    CHECK_CUDA(x); \
    CHECK_CONTIGUOUS(x)

typedef void FlashAttentionFn(
    const nv_bfloat16 *Q,
    const nv_bfloat16 *K,
    const nv_bfloat16 *V,
    nv_bfloat16* O,
    int bh,
    int seqlen_q,
    int d);

FlashAttentionFn flashattn_v1;

template <FlashAttentionFn flashattn_fn>
torch::Tensor compute_attn(
    const torch::Tensor& Q,
    const torch::Tensor& K,
    const torch::Tensor& V) {
    
    CHECK_INPUT(Q);
    CHECK_INPUT(K);
    CHECK_INPUT(V);
    
    int b = Q.size(0);
    int h = Q.size(1);
    int seqlen_q = Q.size(2);
    int seqlen_k = K.size(2);
    int d = Q.size(3);
    int bh = b * h;
    
    torch::Tensor O = torch::empty_like(Q);
    
    auto Q_ptr = reinterpret_cast<const nv_bfloat16 *>(Q.data_ptr());
    auto K_ptr = reinterpret_cast<const nv_bfloat16 *>(K.data_ptr());
    auto V_ptr = reinterpret_cast<const nv_bfloat16 *>(V.data_ptr());
    auto O_ptr = reinterpret_cast<nv_bfloat16 *>(O.data_ptr());
    
    flashattn_fn(
        reinterpret_cast<const nv_bfloat16 *>(Q.data_ptr()), 
        reinterpret_cast<const nv_bfloat16 *>(K.data_ptr()), 
        reinterpret_cast<const nv_bfloat16 *>(V.data_ptr()), 
        reinterpret_cast<nv_bfloat16 *>(O.data_ptr()), 
        bh, seqlen_q, d
    );
    
    return O;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("flashattn_v1", &compute_attn<flashattn_v1>, "Flash Attention v1");
}