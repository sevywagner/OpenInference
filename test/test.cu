#include <iostream>
#include "containers/Tensor.h"
#include "gtest/gtest.h"
#include "memory/HostArena.h"
#include "memory/DeviceArena.h"
#include "model/decoder_layer/Attention.h"
#include "cute/tensor.hpp"
#include <numeric>
#include <string.h>

using namespace Memory;

TEST(TensorSuite, SquareMatrixMatmul) {
  HostArena ha(1 << 30);
  DeviceArena da(1 << 20);
  
  ::std::vector<u64> a_dims = { 3, 3 };
  ::std::vector<u64> b_dims = { 3, 3 };
  ::std::vector<u64> c_dims = { 3, 3 };

  ::Tensor<f32> a(ha, a_dims), b(ha, b_dims), c(ha, c_dims);
  
  ::std::iota(a.get_ptr(), a.get_ptr() + a.get_size(), 1);
  ::std::iota(b.get_ptr(), b.get_ptr() + b.get_size(), 1);
  
  f32 *a_dev = a.copy_to_cuda(da);
  f32 *b_dev = b.copy_to_cuda(da);
  f32 *c_dev = c.copy_to_cuda(da);
  
  cublasLtHandle_t handle;
  CUBLAS_CHECK(cublasLtCreate(&handle));

  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));

  u64 workspace_size = 32 * (1 << 20);
  void *workspace = da.allocate(workspace_size);

  cublas_matmul<f32, CUDA_R_32F, CUBLAS_COMPUTE_32F>(a_dev,
						     a_dims,
						     b_dev,
						     b_dims,
						     c_dev,
						     handle,
						     workspace,
						     workspace_size,
						     stream);

  c.copy_to_host(c_dev);

  CUDA_CHECK(cudaStreamDestroy(stream));
  CUBLAS_CHECK(cublasLtDestroy(handle));

  ::std::vector<f32> expected = { 30.0, 36.0, 42.0, 66.0, 81.0, 96.0, 102.0, 126.0, 150.0 };

  ASSERT_EQ(expected.size(), c.get_size());
  
  f32 *res_ptr = c.get_ptr();
  for (u32 i{}; i < c.get_size(); i++)
    ASSERT_EQ(res_ptr[i], expected[i]);
  
}

TEST(AttentionSuite, GQAPrefillNumericalCorrectness) {
  HostArena ha(1 << 30);
  DeviceArena da(12 << 20);
  
  size_t d_model = 4096, n_q_heads = 32, n_kv_heads = 4, d_head = 128, seq_len = 1024;
  auto kv_dim = n_kv_heads * d_head;
  
  ::std::vector<u64> q_dims = { seq_len, d_model };
  ::std::vector<u64> k_dims = { seq_len, kv_dim };
  ::std::vector<u64> v_dims = { seq_len, kv_dim };
  ::std::vector<u64> o_dims = { seq_len, d_model };

  ::Tensor<cute::half_t> Q_h(ha, q_dims);
  ::Tensor<cute::half_t> K_h(ha, k_dims);
  ::Tensor<cute::half_t> V_h(ha, v_dims);
  ::Tensor<cute::half_t> O_h(ha, o_dims);
  ::Tensor<f32> O_cpu(ha, o_dims);
  
  for (auto i{0uz}; i < Q_h.get_size(); i++)
    Q_h.get_ptr()[i] = 2 * (rand() / double(RAND_MAX)) - 1;

  for (auto i{0uz}; i < K_h.get_size(); i++)
    K_h.get_ptr()[i] = 2 * (rand() / double(RAND_MAX)) - 1;

  for (auto i{0uz}; i < V_h.get_size(); i++)
    V_h.get_ptr()[i] = 2 * (rand() / double(RAND_MAX)) - 1;
  
  for (auto i{0uz}; i < O_h.get_size(); i++) {
    O_h.get_ptr()[i] = 0.0;
    O_cpu.get_ptr()[i] = 0.0;
  }
  
  auto Q_d = Q_h.copy_to_cuda(da);
  auto K_d = K_h.copy_to_cuda(da);
  auto V_d = V_h.copy_to_cuda(da);
  auto O_d = O_h.copy_to_cuda(da);

  DecoderLayer::gqa_prefill(Q_d,
			    K_d,
			    V_d,
			    O_d,
			    d_model,
			    n_q_heads,
			    n_kv_heads,
			    d_head,
			    seq_len);

  O_h.copy_to_host(O_d);

  for (u32 i = 0; i < seq_len; i++) {
    cute::half_t *q = Q_h.get_ptr() + d_model * i;
    f32 *attn = O_cpu.get_ptr() + d_model * i;
    
    DecoderLayer::grouped_query_attention(q, K_h.get_ptr(), V_h.get_ptr(),
			    attn,
			    i,
			    n_q_heads,
			    n_kv_heads,
			    d_model);
  }

  auto o_cpu_it = O_cpu.get_ptr();
  auto o_gpu_it = O_h.get_ptr();
  float atol = 1e-3;
  float rtol = 1e-2;
  
  for (auto i{0uz}; i < O_cpu.get_size(); i++)
    ASSERT_LE(::std::abs(o_gpu_it[i] - o_cpu_it[i]), atol + rtol * ::std::abs(o_cpu_it[i]));
}
