#ifndef IE_ATTENTION_H
#define IE_ATTENTION_H

#include "Common.h"

#include "cute/tensor.hpp"
#include <math_constants.h>

#include <sys/param.h>

using namespace cute;

namespace DecoderLayer {
  template<typename TQ,
	   typename TK,
	   typename TV,
	   typename QSMemLayout,
	   typename KSMemLayout,
	   typename VSMemLayout>
  struct SharedStorage {
    ArrayEngine< TQ, cosize_v<QSMemLayout> > Q;
    ArrayEngine< TK, cosize_v<KSMemLayout> > K;
    ArrayEngine< TV, cosize_v<VSMemLayout> > V;
  };
  
  template<typename QKProblemShape, typename PVProblemShape, typename QKCtaTiler, typename PVCtaTiler,
	   typename TQ, typename QStride, typename QSMemLayout, typename CopyQ, typename S2RAtomQ,
	   typename TK, typename KStride, typename KSMemLayout, typename CopyK, typename S2RAtomK,
	   typename TV, typename VStride, typename VSMemLayout, typename CopyV, typename S2RAtomV,
	   typename TO, typename OStride, typename TiledMma,
	   typename Alpha, typename Beta,
	   typename NKvHeads>
  __global__ void gqa_prefill_kernel(QKProblemShape qk_prob_shape, PVProblemShape pv_prob_shape, QKCtaTiler qk_cta_tiler, PVCtaTiler pv_cta_tiler,
				     const TQ *Q, QStride dQ, QSMemLayout sQ_layout, CopyQ copyQ, S2RAtomQ s2r_atomQ,
				     const TK *K, KStride dK, KSMemLayout sK_layout, CopyK copyK, S2RAtomK s2r_atomK,
				     const TV *V, VStride dV, VSMemLayout sV_layout, CopyV copyV, S2RAtomV s2r_atomV,
				     TO *O, OStride dO, TiledMma tiled_mma,
				     Alpha alpha, Beta beta,
				     NKvHeads n_kv_heads) {
    extern __shared__ char smem[];
    auto &shared_storage = *reinterpret_cast< SharedStorage<TQ, TK, TV, QSMemLayout, KSMemLayout, VSMemLayout> * >(smem);
    cute::Tensor sQ = make_tensor(make_smem_ptr(shared_storage.Q.begin()), sQ_layout);
    cute::Tensor sK = make_tensor(make_smem_ptr(shared_storage.K.begin()), sK_layout);
    cute::Tensor sV = make_tensor(make_smem_ptr(shared_storage.V.begin()), sV_layout);

    auto q_head = blockIdx.y;
    auto q_heads_to_kv_heads = gridDim.y / n_kv_heads; // (n_q_heads : n_kv_heads)
    auto kv_head = q_head / q_heads_to_kv_heads;
    
    const TQ *Q_local = Q + q_head * get<2>(qk_prob_shape);
    const TK *K_local = K + kv_head * get<2>(qk_prob_shape);

    const TV *V_local = V + kv_head * get<1>(pv_prob_shape);
    TO *O_local = O + q_head * get<1>(pv_prob_shape);
    
    cute::Tensor mQ = make_tensor(make_gmem_ptr(Q_local), select<0, 2>(qk_prob_shape), dQ);
    cute::Tensor mK = make_tensor(make_gmem_ptr(K_local), select<1, 2>(qk_prob_shape), dK);

    cute::Tensor mV = make_tensor(make_gmem_ptr(V_local), select<1, 2>(pv_prob_shape), dV);
    cute::Tensor mO = make_tensor(make_gmem_ptr(O_local), select<0, 1>(pv_prob_shape), dO);

    cute::Tensor mI = make_identity_tensor(select<0, 1>(qk_prob_shape));

    cute::Tensor cQ = make_identity_tensor(shape(mQ));
    cute::Tensor cK = make_identity_tensor(shape(mK));

    cute::Tensor cV = make_identity_tensor(shape(mV));
    cute::Tensor cO = make_identity_tensor(shape(mO));
    
    auto qk_cta_coord = make_coord(blockIdx.x, _, _);
    // (qk_blkM, qk_blkK, k)
    // k = 128 / 128 = 1
    // (64, 128, 1)
    cute::Tensor gQ = local_tile(mQ, qk_cta_tiler, qk_cta_coord, Step<_1, X, _1>{});
    // (qk_blkN, qk_blkK, n, k)
    // n = 512 / 64 = 2^9 / 2^6 = 2^3 = 8
    // k = 128 / 128 = 1
    // (64, 128, 8, 1)
    cute::Tensor gK = local_tile(mK, qk_cta_tiler, qk_cta_coord, Step<X, _1, _1>{});
    cute::Tensor gI = local_tile(mI, qk_cta_tiler, qk_cta_coord, Step<_1, _1, X>{});

    cute::Tensor cQgQ = local_tile(cQ, qk_cta_tiler, qk_cta_coord, Step<_1, X, _1>{});
    cute::Tensor cKgK = local_tile(cK, qk_cta_tiler, qk_cta_coord, Step<X, _1, _1>{});
    
    auto pv_cta_coord = make_coord(blockIdx.x, _, _);
    // (pv_blkN, pv_blkK, n, k)
    // n = 128 / 128 = 1
    // k = 512 / 64 = 8
    // (128, 64, 1, 8)
    cute::Tensor gV = local_tile(mV, pv_cta_tiler, pv_cta_coord, Step<X, _1, _1>{});
    // (pv_blkM, pv_blkN, n)
    // n = 1
    // (64, 128, 1)
    cute::Tensor gO = local_tile(mO, pv_cta_tiler, pv_cta_coord, Step<_1, _1, X>{});

    cute::Tensor cVgV = local_tile(cV, pv_cta_tiler, pv_cta_coord, Step<X, _1, _1>{});
    cute::Tensor cOgO = local_tile(cO, pv_cta_tiler, pv_cta_coord, Step<_1, _1, X>{});
    
    ThrCopy thr_copyQ = copyQ.get_slice(threadIdx.x);
    // (CPY, CPY_M, CPY_K, k)
    // CPY = 128 / 16 = 8
    // CPY_M = 64 / (16 * 1) = 4
    // CPY_K = 128 / (8 * 8) = 2
    // (8, 4, 2, 1)
    cute::Tensor tQgQ = thr_copyQ.partition_S(gQ);
    // (CPY, CPY_M, CPY_K, pipe)
    // (8, 4, 2, 2)
    cute::Tensor tQsQ = thr_copyQ.partition_D(sQ);
    cute::Tensor tQcQ = thr_copyQ.partition_S(cQgQ);
    
    ThrCopy thr_copyK = copyK.get_slice(threadIdx.x);
    // (CPY, CPY_N, CPY_K, n, k)
    // CPY = 8
    // CPY_N = 64 / (16 * 1) = 4
    // CPY_K = 128 / (8 * 8) = 2
    // (8, 4, 2, 8, 1)
    cute::Tensor tKgK = thr_copyK.partition_S(gK);
    // (CPY, CPY_N, CPY_K, pipe)
    // (8, 4, 2, 2)
    cute::Tensor tKsK = thr_copyK.partition_D(sK);
    cute::Tensor tKcK = thr_copyK.partition_S(cKgK);
    
    ThrCopy thr_copyV = copyV.get_slice(threadIdx.x);
    // (CPY, CPY_N, CPY_K, n, k)
    // CPY = 8
    // CPY_N = 128 / (8 * 8) = 2
    // CPY_k = 64 / (16 * 1) = 4
    // (8, 2, 4, 1, 8)
    cute::Tensor tVgV = thr_copyV.partition_S(gV);
    // (CPY, CPY_N, CPY_K)
    // (8, 2, 4)
    cute::Tensor tVsV = thr_copyV.partition_D(sV);
    cute::Tensor tVcV = thr_copyV.partition_S(cVgV);

    cute::Tensor tQpQ = make_tensor<bool>(make_shape(size<0>(tQcQ), size<1>(tQcQ), size<2>(tQcQ)),
					  make_stride(Int<0>{}, Int<1>{}, size<1>(tQcQ)));

    cute::Tensor tKpK = make_tensor<bool>(make_shape(size<0>(tKcK), size<1>(tKcK), size<2>(tKcK)),
					  make_stride(Int<0>{}, Int<1>{}, size<1>(tKcK)));

    cute::Tensor tVpV = make_tensor<bool>(make_shape(size<0>(tVcV), size<1>(tVcV), size<2>(tVcV)),
					  make_stride(Int<0>{}, Int<1>{}, size<1>(tVcV)));

    int pv_k_tile_ct = size<3>(gV);
    int pv_k_tile_next{};
    
    CUTE_UNROLL
      for (int m = 0; m < size<1>(tVpV); m++) {
	for (int n = 0; n < size<2>(tVpV); n++)
	  tVpV(0, m, n) = elem_less(get<0>(tVcV(0, m, n, 0, pv_k_tile_next)), shape<0>(mV)) && elem_less(get<1>(tVcV(0, m, n, 0, pv_k_tile_next)), shape<1>(mV));
      }
    
    copy_if(copyV, tVpV, tVgV(_, _, _, Int<0>{}, pv_k_tile_next), tVsV(_, _, _));
    cp_async_fence();
    pv_k_tile_ct--;
    if (pv_k_tile_ct > 0)
      pv_k_tile_next++;

    auto K_PIPE_MAX = size<2>(sQ);
    
    int qk_k_tile_ct = size<2>(gQ);
    int qk_k_tile_next{};

    int K_TILE_MAX = size<2>(gQ);

    for (int k_pipe{}; k_pipe < min(K_PIPE_MAX - 1, K_TILE_MAX); k_pipe++) {
      CUTE_UNROLL
	for (int m = 0; m < size<1>(tQpQ); m++) {
	  for (int n = 0; n < size<2>(tQpQ); n++)
	    tQpQ(0, m, n) = elem_less(get<0>(tQcQ(0, m, n, qk_k_tile_next)), shape<0>(mQ)) && elem_less(get<1>(tQcQ(0, m, n, qk_k_tile_next)), shape<1>(mQ));
	}

      CUTE_UNROLL
	for (int m = 0; m < size<1>(tKpK); m++) {
	  for (int n = 0; n < size<2>(tKpK); n++)
	    tKpK(0, m, n) = elem_less(get<0>(tKcK(0, m, n, 0, qk_k_tile_next)), shape<0>(mK)) && elem_less(get<1>(tKcK(0, m, n, 0, qk_k_tile_next)), shape<1>(mK));
	}

      copy_if(copyQ, tQpQ, tQgQ(_, _, _, qk_k_tile_next), tQsQ(_, _, _, k_pipe));
      copy_if(copyK, tKpK, tKgK(_, _, _, Int<0>{}, qk_k_tile_next), tKsK(_, _, _, k_pipe));
      cp_async_fence();
      qk_k_tile_ct--;
      if (qk_k_tile_ct > 0)
	qk_k_tile_next++;
    }

    ThrMMA thr_mma = tiled_mma.get_slice(threadIdx.x);
    // (MMA, MMA_M, MMA_N)
    // 16x8x16
    // 16 * 8 = 128
    // MMA = 128 / 32 = 4
    // MMA_M = 64 / 64 = 1
    // MMA_N = 128 / 8 = 16
    // (4, 1, 16)
    cute::Tensor tOgO = thr_mma.partition_C(gO(_, _, Int<0>{}));
    cute::Tensor tOcO = thr_mma.partition_C(cOgO(_, _, Int<0>{}));

    cute::Tensor tOpO = make_tensor<bool>(make_shape(size<0>(tOcO), size<1>(tOcO), size<2>(tOcO)),
					  make_stride(Int<1>{}, size<0>(tOcO), size<0>(tOcO) * size<1>(tOcO)));

    for (int m{}; m < size<1>(tOpO); m++) {
      for (int n{}; n < size<2>(tOpO); n++) {
	for (int mma{}; mma < size<0>(tOpO); mma++)
	  tOpO(mma, m, n) = elem_less(get<0>(tOcO(mma, m, n)), shape<0>(mO)) && elem_less(get<1>(tOcO(mma, m, n)), shape<1>(mO));
      }
    }

    
    // (MMA, MMA_M, MMA_K)
    // 16 x 16
    // 256 / 32 = 8
    // 64 / 64 = 1
    // 64 / 16 = 4
    // (8, 1, 4)
    cute::Tensor tOrP = thr_mma.make_fragment_A(Shape<_8, _1, _4>{});
    // (MMA, MMA_N, MMA_K)
    cute::Tensor tOrV = thr_mma.partition_fragment_B(sV(_, _));
    cute::Tensor tOrO = thr_mma.make_fragment_C(tOgO);
    
    cute::Tensor tCrQ = thr_mma.partition_fragment_A(sQ(_, _, 0));
    cute::Tensor tCrK = thr_mma.partition_fragment_B(sK(_, _, 0));
    
    // (MMA, MMA_M, MMA_N)
    // (8 * 16) / 32 = 4
    // MMA_M = 64 / 64 = 1
    // MMA_N = 64 / 8 = 8
    // (4, 1, 8)
    cute::Tensor tCrC = thr_mma.make_fragment_C(Shape<_4, _1, _8>{});
    // (MMA, MMA_M, MMA_N)
    cute::Tensor tIgI = thr_mma.partition_C(gI);
    
    TiledCopy s2r_copyQ = make_tiled_copy_A(s2r_atomQ, tiled_mma);
    ThrCopy s2r_thr_copyQ = s2r_copyQ.get_slice(threadIdx.x);
    // (CPY, MMA_M, MMA_K, pipe)
    // CPY = 8
    // 64 / 64 = 1
    // 128 / 16 = 8
    // (8, 1, 8, 2)
    cute::Tensor tXsQ = s2r_thr_copyQ.partition_S(sQ);
    // (CPY, MMA_M, MMA_K)
    cute::Tensor tXrQ = s2r_thr_copyQ.retile_D(tCrQ);

    TiledCopy s2r_copyK = make_tiled_copy_B(s2r_atomK, tiled_mma);
    ThrCopy s2r_thr_copyK = s2r_copyK.get_slice(threadIdx.x);
    // (CPY, MMA_N, MMA_K, pipe)
    // CPY = 4
    // MMA_N = 64 / 8 = 8
    // MMA_K = 128 / 16 = 8
    cute::Tensor tXsK = s2r_thr_copyK.partition_S(sK);
    // (CPY, MMA_N, MMA_K, pipe)
    cute::Tensor tXrK = s2r_thr_copyK.retile_D(tCrK);

    TiledCopy s2r_copyV = make_tiled_copy_B(s2r_atomV, tiled_mma);
    ThrCopy s2r_thr_copyV = s2r_copyV.get_slice(threadIdx.x);
    // (CPY, MMA_N, MMA_K)
    // CPY = 4
    // MMA_N = 128 / 8 = 16
    // MMA_K = 64 / 16 = 4
    // (4, 16, 4)
    cute::Tensor tXsV = s2r_thr_copyV.partition_S(sV);
    cute::Tensor tXrV = s2r_thr_copyV.retile_D(tOrV);

    auto smem_pipe_read = 0;
    auto smem_pipe_write = min(K_PIPE_MAX - 1, K_TILE_MAX);
    
    cute::Tensor tXsQ_p = tXsQ(_, _, _, smem_pipe_read);
    cute::Tensor tXsK_p = tXsK(_, _, _, smem_pipe_read);

    auto QK_K_BLOCK_MAX = size<2>(tXsQ);
    auto PV_K_BLOCK_MAX = size<2>(tXsV);
    auto KV_TILE_MAX = size<2>(gK);

    auto n_softmax_statistics = size<1>(tCrC);
    float max_row1[n_softmax_statistics];
    float max_row2[n_softmax_statistics];

    float correction_factor_row1[n_softmax_statistics];
    float correction_factor_row2[n_softmax_statistics];

    float l_row1[n_softmax_statistics];
    float l_row2[n_softmax_statistics];

    auto sqrt_d_head = sqrtf(get<2>(qk_prob_shape));
    
    for (int m_repitition{}; m_repitition < n_softmax_statistics; m_repitition++) {
      max_row1[m_repitition] = -CUDART_INF_F;
      max_row2[m_repitition] = -CUDART_INF_F;

      correction_factor_row1[m_repitition] = 0;
      correction_factor_row2[m_repitition] = 0;

      l_row1[m_repitition] = 0;
      l_row2[m_repitition] = 0; 
    }
    
    clear(tOrO);
    CUTE_UNROLL
      for (int kv_tile{}; kv_tile < KV_TILE_MAX; kv_tile++) {
	clear(tCrC);

	tXsQ_p = tXsQ(_, _, _, smem_pipe_read);
	tXsK_p = tXsK(_, _, _, smem_pipe_read);
      
	if (QK_K_BLOCK_MAX > 1) {
	  cp_async_wait<0>();
	  __syncthreads();

	  copy(s2r_atomQ, tXsQ_p(_, _, Int<0>{}), tXrQ(_, _, Int<0>{}));
	  copy(s2r_atomK, tXsK_p(_, _, Int<0>{}), tXrK(_, _, Int<0>{}));
	}

	CUTE_NO_UNROLL
	  while (qk_k_tile_ct > -(min(K_PIPE_MAX - 1, K_TILE_MAX))) {
	    CUTE_UNROLL
	      for (int k_block{}; k_block < QK_K_BLOCK_MAX; k_block++) {
		if (k_block == QK_K_BLOCK_MAX - 1) {
		  tXsQ_p = tXsQ(_, _, _, smem_pipe_read);
		  tXsK_p = tXsK(_, _, _, smem_pipe_read);

		  cp_async_wait<0>();
		  __syncthreads();
		}

		auto k_block_next = (k_block + Int<1>{}) % QK_K_BLOCK_MAX;
		copy(s2r_atomQ, tXsQ_p(_, _, k_block_next), tXrQ(_, _, k_block_next));
		copy(s2r_atomK, tXsK_p(_, _, k_block_next), tXrK(_, _, k_block_next));

		if (k_block == 0) {
		  if (qk_k_tile_ct > 0) {
		    CUTE_UNROLL
		      for (int m = 0; m < size<1>(tQpQ); m++) {
			for (int n = 0; n < size<2>(tQpQ); n++)
			  tQpQ(0, m, n) = elem_less(get<0>(tQcQ(0, m, n, qk_k_tile_next)), shape<0>(mQ)) && elem_less(get<1>(tQcQ(0, m, n, qk_k_tile_next)), shape<1>(mQ));
		      }

		    CUTE_UNROLL
		      for (int m = 0; m < size<1>(tKpK); m++) {
			for (int n = 0; n < size<2>(tKpK); n++)
			  tKpK(0, m, n) = elem_less(get<0>(tKcK(0, m, n, kv_tile, qk_k_tile_next)), shape<0>(mK)) && elem_less(get<1>(tKcK(0, m, n, kv_tile, qk_k_tile_next)), shape<1>(mK));
		      }

		    copy_if(copyQ, tQpQ, tQgQ(_, _, _, qk_k_tile_next), tQsQ(_, _, _, smem_pipe_write));
		    copy_if(copyK, tKpK, tKgK(_, _, _, kv_tile, qk_k_tile_next), tKsK(_, _, _, smem_pipe_write));
		    cp_async_fence();
		  }
	    
		  qk_k_tile_ct--;
		  if (qk_k_tile_ct > 0)
		    qk_k_tile_next++;

		  smem_pipe_write = smem_pipe_read;
		  smem_pipe_read = (smem_pipe_read == K_PIPE_MAX - 1) ? 0 : smem_pipe_read + 1;
		}

		gemm(tiled_mma, tCrQ(_, _, k_block), tCrK(_, _, k_block), tCrC);
	      }
	  }

	CUTE_UNROLL
	  for (int m_repitition{}; m_repitition < size<1>(tCrC); m_repitition++) {
	    CUTE_UNROLL
	      for(int n_repitition{}; n_repitition < size<2>(tCrC); n_repitition++) {
		CUTE_UNROLL
		  for (int mma{}; mma < size<0>(tCrC); mma++) {
		    auto coord = tIgI(mma, m_repitition, n_repitition, kv_tile);
		    if (get<1>(coord) > get<0>(coord) || get<1>(coord) >= get<1>(qk_prob_shape))
		      tCrC(mma, m_repitition, n_repitition) = -CUDART_INF_F;
		    else
		      tCrC(mma, m_repitition, n_repitition) = tCrC(mma, m_repitition, n_repitition) / sqrt_d_head;
		  }
	      }

	    auto max_old_row1 = max_row1[m_repitition];
	    auto max_old_row2 = max_row2[m_repitition];
	    CUTE_UNROLL
	      for (int n_repitition{}; n_repitition < size<2>(tCrC); n_repitition++) {
		max_row1[m_repitition] = max(max_row1[m_repitition], max(tCrC(0, m_repitition, n_repitition), tCrC(1, m_repitition, n_repitition)));
		max_row2[m_repitition] = max(max_row2[m_repitition], max(tCrC(2, m_repitition, n_repitition), tCrC(3, m_repitition, n_repitition)));
	      }

	    // partner lane = lane_id ^ offset
	    // 0 ^ 2
	    // 00 ^ 10 = 10
	    // 0 ^ 1 = 1

	    // 1 ^ 2
	    // 01 ^ 10 = 11
	    // 01 ^ 01 = 00

	    // 4 ^ 2
	    // 100 ^ 010 = 110
	    // 100 ^ 001 = 101
	    CUTE_UNROLL
	      for (int offset = 2; offset > 0; offset >>= 1) {
		max_row1[m_repitition] = max(max_row1[m_repitition], __shfl_xor_sync(0xFFFFFFFF, max_row1[m_repitition], offset));
		max_row2[m_repitition] = max(max_row2[m_repitition], __shfl_xor_sync(0xFFFFFFFF, max_row2[m_repitition], offset));
	      }

	    CUTE_UNROLL
	      for (int n_repitition{}; n_repitition < size<2>(tCrC); n_repitition++) {
		correction_factor_row1[m_repitition] = expf(max_old_row1 - max_row1[m_repitition]);
		correction_factor_row2[m_repitition] = expf(max_old_row2 - max_row2[m_repitition]);
	      }

	    float l_new_row1 = 0.0;
	    float l_new_row2 = 0.0;
	    
	    CUTE_UNROLL
	      for (int n_repitition{}; n_repitition < size<2>(tCrC); n_repitition += 2) {
		auto tOrP_n_rep = n_repitition / 2;

		float p_uncast;
		p_uncast = expf(tCrC(0, m_repitition, n_repitition) - max_row1[m_repitition]);
		l_new_row1 += p_uncast;
		tOrP(0, m_repitition, tOrP_n_rep) = static_cast<half_t>(p_uncast);

		p_uncast = expf(tCrC(1, m_repitition, n_repitition) - max_row1[m_repitition]);
		l_new_row1 += p_uncast;
		tOrP(1, m_repitition, tOrP_n_rep) = static_cast<half_t>(p_uncast);
		
		p_uncast = expf(tCrC(2, m_repitition, n_repitition) - max_row2[m_repitition]);
		l_new_row2 += p_uncast;
		tOrP(2, m_repitition, tOrP_n_rep) = static_cast<half_t>(p_uncast);

		p_uncast = expf(tCrC(3, m_repitition, n_repitition) - max_row2[m_repitition]);
		l_new_row2 += p_uncast;
		tOrP(3, m_repitition, tOrP_n_rep) = static_cast<half_t>(p_uncast);

		p_uncast = expf(tCrC(0, m_repitition, n_repitition + 1) - max_row1[m_repitition]);
		l_new_row1 += p_uncast;
		tOrP(4, m_repitition, tOrP_n_rep) = static_cast<half_t>(p_uncast);

		p_uncast = expf(tCrC(1, m_repitition, n_repitition + 1) - max_row1[m_repitition]);
		l_new_row1 += p_uncast;
		tOrP(5, m_repitition, tOrP_n_rep) = static_cast<half_t>(p_uncast);

		p_uncast = expf(tCrC(2, m_repitition, n_repitition + 1) - max_row2[m_repitition]);
		l_new_row2 += p_uncast;
		tOrP(6, m_repitition, tOrP_n_rep) = static_cast<half_t>(p_uncast);

		p_uncast = expf(tCrC(3, m_repitition, n_repitition + 1) - max_row2[m_repitition]);
		l_new_row2 += p_uncast;
		tOrP(7, m_repitition, tOrP_n_rep) = static_cast<half_t>(p_uncast);
	      }

	    for (int offset = 2; offset > 0; offset >>= 1) {
	      l_new_row1 += __shfl_xor_sync(0xFFFFFFFF, l_new_row1, offset);
	      l_new_row2 += __shfl_xor_sync(0xFFFFFFFF, l_new_row2, offset);
	    }

	    l_row1[m_repitition] = l_row1[m_repitition] * correction_factor_row1[m_repitition] + l_new_row1;
	    l_row2[m_repitition] = l_row2[m_repitition] * correction_factor_row2[m_repitition] + l_new_row2;
	    
	  }
	
	for (int m_repitition{}; m_repitition < size<1>(tOrO); m_repitition++) {	
	  CUTE_UNROLL
	    for (int n_repitition{}; n_repitition < size<2>(tOrO); n_repitition++) {
	      tOrO(0, m_repitition, n_repitition) *= correction_factor_row1[m_repitition];
	      tOrO(1, m_repitition, n_repitition) *= correction_factor_row1[m_repitition];
	      tOrO(2, m_repitition, n_repitition) *= correction_factor_row2[m_repitition];
	      tOrO(3, m_repitition, n_repitition) *= correction_factor_row2[m_repitition];
	    }
	}
	
	cp_async_wait<0>();
	__syncthreads();
	
	for (int k_block{}; k_block < PV_K_BLOCK_MAX; k_block++) {
	  copy(s2r_atomV, tXsV(_, _, k_block), tXrV(_, _, k_block));
	  gemm(tiled_mma, tOrP(_, _, k_block), tOrV(_, _, k_block), tOrO);
	}
	  

	qk_k_tile_ct = size<2>(gQ);
	qk_k_tile_next = 0;

	if (kv_tile + 1 < KV_TILE_MAX) {
	  cp_async_wait<0>();
	  __syncthreads();

	  CUTE_UNROLL
	    for (int m = 0; m < size<1>(tVpV); m++) {
	      for (int n = 0; n < size<2>(tVpV); n++)
		tVpV(0, m, n) = elem_less(get<0>(tVcV(0, m, n, 0, pv_k_tile_next)), shape<0>(mV)) && elem_less(get<1>(tVcV(0, m, n, 0, pv_k_tile_next)), shape<1>(mV));
	    }

	  copy_if(copyV, tVpV, tVgV(_, _, _, Int<0>{}, pv_k_tile_next), tVsV(_, _, _));
	  cp_async_fence();
	  pv_k_tile_ct--;
	  if (pv_k_tile_ct > 0)
	    pv_k_tile_next++;

	  for (int k_pipe{}; k_pipe < min(K_PIPE_MAX - 1, K_PIPE_MAX); k_pipe++) {
	    CUTE_UNROLL
	      for (int m = 0; m < size<1>(tQpQ); m++) {
		for (int n = 0; n < size<2>(tQpQ); n++)
		  tQpQ(0, m, n) = elem_less(get<0>(tQcQ(0, m, n, qk_k_tile_next)), shape<0>(mQ)) && elem_less(get<1>(tQcQ(0, m, n, qk_k_tile_next)), shape<1>(mQ));
	      }

	    CUTE_UNROLL
	      for (int m = 0; m < size<1>(tKpK); m++) {
		for (int n = 0; n < size<2>(tKpK); n++)
		  tKpK(0, m, n) = elem_less(get<0>(tKcK(0, m, n, kv_tile + 1, qk_k_tile_next)), shape<0>(mK)) && elem_less(get<1>(tKcK(0, m, n, kv_tile + 1, qk_k_tile_next)), shape<1>(mK));
	      }


	    copy_if(copyQ, tQpQ, tQgQ(_, _, _, qk_k_tile_next), tQsQ(_, _, _, k_pipe));
	    copy_if(copyK, tKpK, tKgK(_, _, _, kv_tile + 1, qk_k_tile_next), tKsK(_, _, _, k_pipe));
	    cp_async_fence();
	    qk_k_tile_ct--;
	    if (qk_k_tile_ct > 0)
	      qk_k_tile_next++;
	  }
	}

	smem_pipe_read = 0;
	smem_pipe_write = min(K_PIPE_MAX - 1, K_TILE_MAX);
      }

    for (int m_repitition{}; m_repitition < size<1>(tOrO); m_repitition++) {
      for (int n_repitition{}; n_repitition < size<2>(tOrO); n_repitition++) {
	tOrO(0, m_repitition, n_repitition) /= l_row1[m_repitition];
	tOrO(1, m_repitition, n_repitition) /= l_row1[m_repitition];
	tOrO(2, m_repitition, n_repitition) /= l_row2[m_repitition];
	tOrO(3, m_repitition, n_repitition) /= l_row2[m_repitition];	
      }
    }

    axpby(alpha, tOrO, beta, tOgO, tOpO);
  }
  
  template <typename TQ, typename TK, typename TV, typename TO>
  void gqa_prefill(const TQ *Q,
		   const TK *K,
		   const TV *V,
		   TO *O,
		   size_t d_model,
		   size_t n_q_heads,
		   size_t n_kv_heads,
		   size_t d_head,
		   size_t seq_len) {
    auto qkM = seq_len;
    auto qkN = seq_len;
    auto qkK = d_head;
    auto qk_prob_shape = make_shape(qkM, qkN, qkK);

    auto ldQ = d_model; // (M, K)
    auto ldK = n_kv_heads * d_head; // (N, K)

    auto dQ = make_stride(ldQ, Int<1>{});
    auto dK = make_stride(ldK, Int<1>{});

    auto pvM = seq_len;
    auto pvN = d_head;
    auto pvK = seq_len;
    auto pv_prob_shape = make_shape(pvM, pvN, pvK);

    auto ldV = n_kv_heads * d_head; // (N, K)
    auto ldO = d_model; // (M, N)

    auto dV = make_stride(Int<1>{}, ldV);
    auto dO = make_stride(ldO, Int<1>{});

    auto qk_blkM = Int<64>{};
    auto qk_blkN = Int<64>{};
    auto qk_blkK = Int<128>{};
    auto qk_cta_tiler = make_shape(qk_blkM, qk_blkN, qk_blkK);
    auto qk_blkP = Int<2>{};

    auto pv_blkM = Int<64>{};
    auto pv_blkN = Int<128>{};
    auto pv_blkK = Int<64>{};
    auto pv_cta_tiler = make_shape(pv_blkM, pv_blkN, pv_blkK);

    // 2^3 = 8
    // 2^0 = 1
    // 2^6 = 64
    // kkkiiijjj
    // kkk - (iii ^ kkk) - jjj
    // x ^ ((x & 0b111000000) >> 3)
    // offset = 8i + j + 64k
    // offset_swizzle = 64k + (8k ^ 8i) + j
    auto qk_swizzle_atom = composition(Swizzle<3, 3, 3>{},
				       Layout< Shape< _8, Shape<_8, _8> >, Stride< _8, Stride<_1, _64> > >{});

    auto pv_swizzle_atom = composition(Swizzle<3, 3, 3>{},
				       Layout< Shape< Shape<_8, _8>, _8 >, Stride< Stride<_1, _8>, _64 > >{});

    auto sQ = tile_to_shape(qk_swizzle_atom, make_shape(qk_blkM, qk_blkK, qk_blkP));
    auto sK = tile_to_shape(qk_swizzle_atom, make_shape(qk_blkN, qk_blkK, qk_blkP));

    auto sV = tile_to_shape(pv_swizzle_atom, make_shape(pv_blkN, pv_blkK));

    TiledCopy copyQ = make_tiled_copy(Copy_Atom< SM80_CP_ASYNC_CACHEALWAYS_ZFILL<uint128_t>, half_t >{},
				      Layout< Shape<_16, _8>, Stride<_8, _1> >{},
				      Layout< Shape<_1, _8> >{});

    TiledCopy copyK = make_tiled_copy(Copy_Atom< SM80_CP_ASYNC_CACHEALWAYS_ZFILL<uint128_t>, half_t >{},
				      Layout< Shape<_16, _8>, Stride<_8, _1> >{},
				      Layout< Shape<_1, _8> >{});

    TiledCopy copyV = make_tiled_copy(Copy_Atom< SM80_CP_ASYNC_CACHEALWAYS_ZFILL<uint128_t>, half_t >{},
				      Layout< Shape<_8, _16>, Stride<_1, _8> >{},
				      Layout< Shape<_8, _1> >{});

    TiledMMA tiled_mma = make_tiled_mma(SM80_16x8x16_F32F16F16F32_TN{},
					Layout< Shape<_4, _1> >{},
					Tile<_64, _8, _16>{});

    // 16 * 16 = 256
    // 256 / 32 = 2^8 / 2^5 = 2^3 = 8
    Copy_Atom<SM75_U32x4_LDSM_N, half_t> s2r_atomQ;

    // 8 * 16 = 128
    // 128 / 32 = 4
    Copy_Atom<SM75_U32x2_LDSM_N, half_t> s2r_atomK;
    Copy_Atom<SM75_U16x4_LDSM_T, half_t> s2r_atomV;

    float alpha = 1.0;
    float beta = 0.0;

    auto smem_size = sizeof(SharedStorage<TQ, TK, TV, decltype(sQ), decltype(sK), decltype(sV)>);
    
    auto fptr = gqa_prefill_kernel<decltype(qk_prob_shape), decltype(pv_prob_shape), decltype(qk_cta_tiler), decltype(pv_cta_tiler),
				   TQ, decltype(dQ), decltype(sQ), decltype(copyQ), decltype(s2r_atomQ),
				   TK, decltype(dK), decltype(sK), decltype(copyK), decltype(s2r_atomK),
				   TV, decltype(dV), decltype(sV), decltype(copyV), decltype(s2r_atomV),
				   TO, decltype(dO), decltype(tiled_mma),
				   decltype(alpha), decltype(beta),
				   decltype(n_kv_heads)>;

    cudaFuncSetAttribute(fptr, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
    cudaFuncSetAttribute(fptr, cudaFuncAttributePreferredSharedMemoryCarveout, 100);
    
    dim3 block(size(tiled_mma));
    dim3 grid(size(ceil_div(qkM, qk_blkM)), n_q_heads);
    fptr<<<grid, block, smem_size, 0>>>(qk_prob_shape, pv_prob_shape, qk_cta_tiler, pv_cta_tiler,
					Q, dQ, sQ, copyQ, s2r_atomQ,
					K, dK, sK, copyK, s2r_atomK,
					V, dV, sV, copyV, s2r_atomV,
					O, dO, tiled_mma,
					alpha, beta,
					n_kv_heads);

    cudaDeviceSynchronize();
  }

  void grouped_query_attention(cute::half_t *q,
			       cute::half_t *k,
			       cute::half_t *v,
			       f32 *o,
			       u32 pos,
			       u32 n_heads,
			       u32 n_kv_heads,
			       u32 d_model) {
    u32 head_dim = d_model / n_heads;
    u32 kv_dim = (n_kv_heads * head_dim);

    memset(o, 0, d_model * sizeof(f32));
  
    f32 attn_scores[n_heads * (pos + 1)];
    f32 softmax_factor[n_heads];
    memset(softmax_factor, 0, sizeof(f32) * n_heads);
  
    f32 max[n_heads];
    for (u32 i = 0; i < n_heads; i++)
      max[i] = -FLT_MAX;

    for (u32 i = 0; i <= pos; i++) {
      u32 start_pos_attn = i * n_heads;
    
      for (u32 j = 0; j < n_heads; j++) {
	attn_scores[start_pos_attn + j] = 0.0f;
	u64 start_pos_q = j * head_dim;
	u64 start_pos_kv = ((j * n_kv_heads) / n_heads) * head_dim;
	for (u32 y = 0; y < head_dim; y++) {
	  attn_scores[start_pos_attn + j] += static_cast<f32>(q[start_pos_q + y]) * static_cast<f32>((k + i * kv_dim)[start_pos_kv + y]);
	}

	attn_scores[start_pos_attn + j] /= sqrtf((f32) head_dim);
	max[j] = MAX(max[j], attn_scores[start_pos_attn + j]);
      }
    }

    for (u32 i = 0; i <= pos; i++) {
      u32 start_pos_attn = i * n_heads;
      for (u32 j = 0; j < n_heads; j++) {
	softmax_factor[j] += expf(attn_scores[start_pos_attn + j] - max[j]);
      }
    }
  
    for (u32 i = 0; i <= pos; i++) {
      u32 start_pos_attn = i * n_heads;

      for (u32 j = 0; j < n_heads; j++) {
	f32 attn_weight = expf(attn_scores[start_pos_attn + j] - max[j]) / softmax_factor[j];

	u64 start_pos_out = j * head_dim;
	u64 start_pos_kv = ((j * n_kv_heads) / n_heads) * head_dim;

	for (u32 y = 0; y < head_dim; y++) {
	  o[start_pos_out + y] += attn_weight * (v + i * kv_dim)[start_pos_kv + y];
	}
      }
    }
  }
};

#endif
