#ifndef IE_NORMALIZATION_H
#define IE_NORMALIZATION_H

#include "Common.h"

namespace DecoderLayer {
  const auto WARP_SCALE = 4;
  const auto WARP_SCALE_THRESHOLD = 5120;
  const auto BLOCK_SCALE = 4;
  const auto BLOCK_SCALE_THRESHOLD = 40;

  template <typename TX, typename TW>
  __global__ void rms_norm_kernel(const TX *src,
				  TX *dest,
				  const TW *weights,
				  u64 d_model,
				  u64 seq_len,
				  f32 epsilon) {
    extern __shared__ f32 shared_memory[];
    
    auto rows_per_block = seq_len / gridDim.x;
    auto elements_per_block = d_model / blockDim.x;
    auto warps_per_block = blockDim.x / warpSize;
    
    auto warp_idx = threadIdx.x / warpSize;
    
    for (auto row{0uz}; row < rows_per_block; row++) {
      f32 sum_of_squares{};
      // accumulate thread-local partial sum
      if (d_model % blockDim.x > 0 && threadIdx.x < d_model % blockDim.x) {
	f32 x_i = src[((row * gridDim.x) + blockIdx.x) * d_model + (d_model / blockDim.x) * blockDim.x + threadIdx.x];
	sum_of_squares += x_i * x_i;
      }

      for (auto el{0uz}; el < elements_per_block; el++) {
	f32 x_i = src[((row * gridDim.x) + blockIdx.x) * d_model + threadIdx.x + el * blockDim.x];
	sum_of_squares += x_i * x_i;
      }

      // accumulate warp-local partial sums
      for (u32 offset = 16; offset > 0; offset >>= 1)
	sum_of_squares += __shfl_down_sync(0xFFFFFFFF, sum_of_squares, offset);
      
      // store warp-local partial sums into shared_memory
      // [warps_per_block]
      if (threadIdx.x % warpSize == 0)
	shared_memory[warp_idx] = sum_of_squares;
      __syncthreads();

      // load partial sums from shared_memory
      f32 full_row_sum{};
      if (warp_idx == 0 && threadIdx.x < warps_per_block)
	full_row_sum = shared_memory[threadIdx.x];

      // perform final reduction for the row
      for (int offset = 16; offset > 0; offset >>= 1)
	full_row_sum += __shfl_down_sync(0xFFFFFFFF, full_row_sum, offset);

      // calculate inv rms
      if (threadIdx.x == 0)
	shared_memory[0] = 1.0 / sqrt((full_row_sum / d_model) + epsilon);
      __syncthreads();
      half inv_rms = shared_memory[0];
      
      // perform update
      if (d_model % blockDim.x > 0 && threadIdx.x < d_model % blockDim.x) {
	auto el_idx = (d_model / blockDim.x) * blockDim.x + threadIdx.x;
	dest[((row * gridDim.x) + blockIdx.x) * d_model + el_idx] = src[((row * gridDim.x) + blockIdx.x) * d_model + el_idx] * inv_rms * weights[el_idx];
      }

      for (auto el{0uz}; el < elements_per_block; el++) {
	auto el_idx = threadIdx.x + el * blockDim.x;
	dest[((row * gridDim.x) + blockIdx.x) * d_model + el_idx] = src[((row * gridDim.x) + blockIdx.x) * d_model + el_idx] * inv_rms * weights[el_idx];
      }

      __syncthreads();
    }

    
    // process outstanding rows
    if (seq_len % gridDim.x > 0 && blockIdx.x < seq_len % gridDim.x) {
      auto outstanding_row_start = (((seq_len / gridDim.x) * gridDim.x) + blockIdx.x) * d_model;

      f32 sum_of_squares{};
      // accumulate thread-local partial sum
      if (d_model % blockDim.x > 0 && threadIdx.x < d_model % blockDim.x) {
	f32 x_i = src[outstanding_row_start + (d_model / blockDim.x) * blockDim.x + threadIdx.x];
	sum_of_squares += x_i * x_i;
      }

      for (auto el{0uz}; el < elements_per_block; el++) {
	f32 x_i = src[outstanding_row_start + threadIdx.x + el * blockDim.x];
	sum_of_squares += x_i * x_i;
      }

      // accumulate warp-local partial sums
      for (u32 offset = 16; offset > 0; offset >>= 1)
	sum_of_squares += __shfl_down_sync(0xFFFFFFFF, sum_of_squares, offset);
      
      // store warp-local partial sums into shared_memory
      // [warps_per_block]
      if (threadIdx.x % warpSize == 0)
	shared_memory[warp_idx] = sum_of_squares;
      __syncthreads();

      // load partial sums from shared_memory
      f32 full_row_sum{};
      if (warp_idx == 0 && threadIdx.x < warps_per_block)
	full_row_sum = shared_memory[threadIdx.x];

      // perform final reduction for the row
      for (int offset = 16; offset > 0; offset >>= 1)
	full_row_sum += __shfl_down_sync(0xFFFFFFFF, full_row_sum, offset);

      // calculate inv rms
      if (threadIdx.x == 0)
	shared_memory[0] = 1.0 / sqrt((full_row_sum / d_model) + epsilon);
      __syncthreads();
      half inv_rms = shared_memory[0];
      
      // perform update
      if (d_model % blockDim.x > 0 && threadIdx.x < d_model % blockDim.x) {
	auto el_idx = (d_model / blockDim.x) * blockDim.x + threadIdx.x;
	dest[outstanding_row_start + el_idx] = src[outstanding_row_start + el_idx] * inv_rms * weights[el_idx];
      }

      for (auto el{0uz}; el < elements_per_block; el++) {
	auto el_idx = threadIdx.x + el * blockDim.x;
	dest[outstanding_row_start + el_idx] = src[outstanding_row_start + el_idx] * inv_rms * weights[el_idx];
      }
    }
  }

  template <typename TX, typename TW>
  void rms_norm(const TX *src,
		TX *dest,
		const TW *weights,
		u64 d_model,
		u64 seq_len,
		f32 epsilon,
		cudaStream_t stream = 0) {
    dim3 block(WARP_SCALE * 32 * ((d_model + WARP_SCALE_THRESHOLD - 1) / WARP_SCALE_THRESHOLD));
    dim3 grid(((seq_len + BLOCK_SCALE_THRESHOLD - 1) / BLOCK_SCALE_THRESHOLD) * BLOCK_SCALE);

    u64 smem_size = WARP_SCALE * ((d_model + WARP_SCALE_THRESHOLD - 1) / WARP_SCALE_THRESHOLD) * sizeof(f32);
    
    rms_norm_kernel<<<grid, block, smem_size, stream>>>(src,
							dest,
							weights,
							d_model,
							seq_len,
							epsilon);

    cudaDeviceSynchronize();
  }

  void rms_norm_cpu(half *x, half *out, half *w, u64 dim, u64 n_rows, float epsilon) {
    for (u32 i = 0; i < n_rows; i++) {
      float sum_sq = 0.0;
      u64 x_start_pos = i * dim;
    
      for (u32 j = 0; j < dim; j++) {
	f32 x_i = x[x_start_pos + j];
	sum_sq += x_i * x_i;
      }
	
      f32 rms = sqrtf((sum_sq / dim) + epsilon);
      f32 inv_rms = 1.0f / rms;
      half inv_rms_16 = inv_rms;
      
      for (u32 j = 0; j < dim; j++) {
	out[x_start_pos + j] = x[x_start_pos + j] * inv_rms_16 * w[j];
      }
    }
  }
};

#endif
