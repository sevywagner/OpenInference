#include <iostream>
#include "containers/Tensor.h"
#include "gtest/gtest.h"
#include "memory/HostArena.h"
#include "memory/DeviceArena.h"
#include "model/decoder_layer/Attention.h"
#include "model/decoder_layer/Normalization.h"
#include "cute/tensor.hpp"
#include <numeric>
#include <string.h>

using namespace Memory;

class TestEnvironment : public ::testing::Environment {
  static const size_t two_GB = static_cast<u64>(1) << 31;

public:
  static HostArena &get_host_arena() {
    static HostArena ha(two_GB);
    return ha;
  }

  static DeviceArena &get_device_arena() {
    static DeviceArena da(two_GB);
    return da;
  }
};

TEST(TensorSuite, SquareMatrixMatmul) {
  auto &ha = TestEnvironment::get_host_arena();
  auto &da = TestEnvironment::get_device_arena();
  
  ::std::vector<u64> a_dims = { 3, 3 };
  ::std::vector<u64> b_dims = { 3, 3 };
  ::std::vector<u64> c_dims = { 3, 3 };

  ::Tensor<f32> a(ha, a_dims), b(ha, b_dims), c(ha, c_dims);
  
  ::std::iota(a.get_ptr(), a.get_ptr() + a.size(), 1);
  ::std::iota(b.get_ptr(), b.get_ptr() + b.size(), 1);
  
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

  ASSERT_EQ(expected.size(), c.size());
  
  f32 *res_ptr = c.get_ptr();
  for (u32 i{}; i < c.size(); i++)
    ASSERT_EQ(res_ptr[i], expected[i]);
  
  ha.reset();
  da.reset();
}

TEST(AttentionSuite, GQAPrefillNumericalCorrectness) {
  auto N_TEST_VALUES = 3;
  size_t d_model[] = { 4096, 8192, 16384 };
  size_t n_q_heads[] = { 32, 64, 128 };
  size_t n_kv_heads[] = { 4, 8, 8 };
  size_t d_head = 128;
  size_t seq_len[] = { 19, 127, 1024 };
  size_t kv_dim[] = { n_kv_heads[0] * d_head, n_kv_heads[1] * d_head, n_kv_heads[2] * d_head };

  auto &ha = TestEnvironment::get_host_arena();
  auto &da = TestEnvironment::get_device_arena();

  for (int test{}; test < N_TEST_VALUES; test++) {
    ::std::vector<u64> q_dims = { seq_len[test], d_model[test] };
    ::std::vector<u64> k_dims = { seq_len[test], kv_dim[test] };
    ::std::vector<u64> v_dims = { seq_len[test], kv_dim[test] };
    ::std::vector<u64> o_dims = { seq_len[test], d_model[test] };

    ::Tensor<cute::half_t> Q_h(ha, q_dims);
    ::Tensor<cute::half_t> K_h(ha, k_dims);
    ::Tensor<cute::half_t> V_h(ha, v_dims);
    ::Tensor<cute::half_t> O_h(ha, o_dims);
    ::Tensor<f32> O_cpu(ha, o_dims);
  
    for (auto i{0uz}; i < Q_h.size(); i++)
      Q_h.get_ptr()[i] = 2 * (rand() / double(RAND_MAX)) - 1;

    for (auto i{0uz}; i < K_h.size(); i++)
      K_h.get_ptr()[i] = 2 * (rand() / double(RAND_MAX)) - 1;

    for (auto i{0uz}; i < V_h.size(); i++)
      V_h.get_ptr()[i] = 2 * (rand() / double(RAND_MAX)) - 1;
  
    for (auto i{0uz}; i < O_h.size(); i++) {
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
			      d_model[test],
			      n_q_heads[test],
			      n_kv_heads[test],
			      d_head,
			      seq_len[test]);

    O_h.copy_to_host(O_d);

    for (u32 i = 0; i < seq_len[test]; i++) {
      cute::half_t *q = Q_h.get_ptr() + d_model[test] * i;
      f32 *attn = O_cpu.get_ptr() + d_model[test] * i;
    
      DecoderLayer::grouped_query_attention(q, K_h.get_ptr(), V_h.get_ptr(),
					    attn,
					    i,
					    n_q_heads[test],
					    n_kv_heads[test],
					    d_model[test]);
    }

    auto o_cpu_it = O_cpu.get_ptr();
    auto o_gpu_it = O_h.get_ptr();
    float atol = 1e-3;
    float rtol = 1e-2;

    for (auto i{0uz}; i < O_cpu.size(); i++)
      ASSERT_LE(::std::abs(o_gpu_it[i] - o_cpu_it[i]), atol + rtol * ::std::abs(o_cpu_it[i]));

    ha.reset();
    da.reset();
  }
}

TEST(NormalizationSuite, RMSNormCorrectness) {
  auto N_TEST_VALUES = 3;
  size_t d_model[N_TEST_VALUES] = { 3072, 5120, 16376 };
  size_t seq_len[N_TEST_VALUES] = { 30, 500, 1024 };

  auto &ha = TestEnvironment::get_host_arena();
  auto &da = TestEnvironment::get_device_arena();
    
  for (int i{}; i < N_TEST_VALUES; i++) {
    ::std::vector<u64> x_dims = { seq_len[i], d_model[i] };
    ::std::vector<u64> w_dims = { d_model[i] };

    ::Tensor<half> X_h(ha, x_dims);
    ::Tensor<half> W_h(ha, w_dims);
    ::Tensor<half> O_cpu(ha, x_dims);
    
    for (auto j{0uz}; j < W_h.size(); j++)
      W_h.get_ptr()[j] = 2 * (rand() / double(RAND_MAX)) - 1;
    
    for (auto j{0uz}; j < X_h.size(); j++) {
      X_h.get_ptr()[j] = 2 * (rand() / double(RAND_MAX)) - 1;
      O_cpu.get_ptr()[j] = 0.0;
    }

    auto X_d = X_h.copy_to_cuda(da);
    auto W_d = W_h.copy_to_cuda(da);

    DecoderLayer::rms_norm_cpu(X_h.get_ptr(),
			       O_cpu.get_ptr(),
			       W_h.get_ptr(),
			       d_model[i],
			       seq_len[i],
			       1e-5);
    
    DecoderLayer::rms_norm<half, half>(X_d,
				       X_d,
				       W_d,
				       d_model[i],
				       seq_len[i],
				       1e-5);

    X_h.copy_to_host(X_d);

    float atol = 1e-3, rtol = 1e-2;
    
    for (auto j{0uz}; j < X_h.size(); j++) {
      f32 kernel = X_h.get_ptr()[j];
      f32 cpu = O_cpu.get_ptr()[j];
      ASSERT_LE(::std::abs(static_cast<f32>(kernel - cpu)), atol + rtol * ::std::abs(cpu));
    }
    
    ha.reset();
    da.reset();
  }
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new TestEnvironment);
  return RUN_ALL_TESTS();
}
