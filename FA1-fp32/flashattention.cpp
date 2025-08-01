#include <torch/extension.h>


#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x)                                                                                                 \
    CHECK_CUDA(x);                                                                                                       \
    CHECK_CONTIGUOUS(x)

typedef void FlashAttentionFn(const float *Q, const float *K, const float *V, float* O, float* m, float* l, int B, int nh, int T, int head_dim);

FlashAttentionFn flashattn_v1;
FlashAttentionFn flashattn_v2;
FlashAttentionFn flashattn_v3;
FlashAttentionFn flashattn_v4;
FlashAttentionFn flashattn_v5;
FlashAttentionFn flashattn_v6;
FlashAttentionFn flashattn_v7;



template <FlashAttentionFn flashattn_fn> torch::Tensor flashattn_pt(torch::Tensor Q, torch::Tensor K, torch::Tensor V) {
    CHECK_INPUT(Q);
    CHECK_INPUT(K);
    CHECK_INPUT(V);
    int B = Q.size(0), nh = Q.size(1), T = Q.size(2), head_dim = Q.size(3);

    torch::Tensor O = torch::empty({B, nh, T, head_dim}, Q.options());
    auto tensor_opts = torch::dtype(torch::kFloat32).device(torch::kCUDA);
    torch::Tensor l = torch::zeros({B, nh, T}, tensor_opts);
    torch::Tensor m = torch::full({B, nh, T}, -INFINITY, tensor_opts);

    flashattn_fn(
        Q.data_ptr<float>(),
        K.data_ptr<float>(),
        V.data_ptr<float>(),
        O.data_ptr<float>(),
        l.data_ptr<float>(), // Pass data pointers
        m.data_ptr<float>(),
        B, nh, T, head_dim
    );
    return O;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("flashattn_v1", &flashattn_pt<flashattn_v1>, "Flash Attention v1");
    m.def("flashattn_v2", &flashattn_pt<flashattn_v2>, "Flash Attention v2");
    m.def("flashattn_v3", &flashattn_pt<flashattn_v3>, "Flash Attention v3");
    m.def("flashattn_v4", &flashattn_pt<flashattn_v4>, "Flash Attention v4");
    m.def("flashattn_v6", &flashattn_pt<flashattn_v6>, "Flash Attention v6");
    m.def("flashattn_v5", &flashattn_pt<flashattn_v5>, "Flash Attention v5");
    m.def("flashattn_v7", &flashattn_pt<flashattn_v7>, "Flash Attention v7");
}
