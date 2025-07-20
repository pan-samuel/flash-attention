#pragma once

#include <cuda.h>

#include <vector>

struct Flash_params {
    using index_t = uint32_t;

    void *__restrict__ q_ptr;
    void *__restrict__ k_ptr;
    void *__restrict__ v_ptr;
    
    void *__restrict out_ptr;

    index_t q_batch_stride;
    index_t k_batch_stride;
    index_t v_batch_stride;
    index_t q_head_stride;
    index_t k_head_stride;
    index_t v_head_stride;
    index_t q_seq_stride;
    index_t k_seq_stride;
    index_t v_seq_stride;

    index_t o_batch_stride;
    index_t o_head_stride;
    index_t o_seq_stride;

    int b, nh, seq_len, k_seq_len, d;
    float softmax_scale;
}
