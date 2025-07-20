#include <torch/extension.h>
#include <torch/python.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <cutlass/numeric_types.h>

Flash_params set_params(const at::Tensor q,
                        const at::Tensor k,
                        const at::Tensor v,
                        const at::Tensor out,
                        const size_t bs,
                        const size_t seq_len,
                        const size_t k_seq_len,
                        const size_t head_dim) {
    
    Flash_params params;
    memset(&params, 0, sizeof(params));

    params.q_ptr = q.data_ptr();
    params.k_ptr = k.data_ptr();
    params.v_ptr = v.data_ptr();

    params.out_ptr = out.data_ptr();

    params.q_batch_stride = q.stride(0);
    params.k_batch_stride = k.stride(0);
    params.v_batch_stride = v.stride(0);

    params.o_batch_stride = out.stride(0);

    params.q_head_stride = q.stride(1);
    params.k_head_stride = k.stride(1);
    params.v_head_stride = v.stride(1);
    params.q_seq_stride = q.stride(2);
    params.k_seq_stride = k.stride(2);
    params.v_seq_stride = v.stride(2);
    
    params.o_head_stride = out.stride(1);
    params.o_seq_stride = out.stride(2);

    params.bs = bs;
    params.seq_len = seq_len;
    params.k_seq_len = k_seq_len;
    params.d = head_dim;

    params.softmax_scale = softmax_scale;
}

std::vector<at::Tensor> flash_attention(const at::Tensor &q, const at::Tensor &k, const at::Tensor &v,
                const float softmax_scale) {
    
    auto dprops = at::cuda::getCurrentDeviceProperties();
    bool is_sm8x = dprops->major=8 && dprops->minor >= 0;
    TORCH_CHECK(is_sm8x, "Only supports Ampere GPUs")

    TORCH_CHECK(q.is_cuda(), "Q tensor must be on CUDA device");
    TORCH_CHECK(k.is_cuda(), "K tensor must be on CUDA device");
    TORCH_CHECK(v.is_cuda(), "V tensor must be on CUDA device");

    TORCH_CHECK(q.stride(-1) == 1, "Q tensor muts be contiguous in last dimension");
    TORCH_CHECK(q.stride(-1) == 1, "K tensor muts be contiguous in last dimension");
    TORCH_CHECK(q.stride(-1) == 1, "V tensor muts be contiguous in last dimension");
    
    auto dtype = q.d
    const auto sizes = q.sizes();
    
    const int bs = sizes[0];
    const int num_heads = sizes[1];
    const int seq_len = sizes[2];
    const int head_dim = sizes[3];
    const int k_seq_len = k.size(2);

    at::Tensor out = at::empty_like(q);

    static Flash_params params = set_params(q, k, v, out, bs, seq_len, k_seq_len, head_dim)
    
    at::cuda::CUDAGuard device_guard{(char)q.get_device()};
    auto stream = at::cuda::getCurrentCUDAStream().stream();



}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("flash_attention", &flash_attention, "Flash Attention Forward Pass")
}
