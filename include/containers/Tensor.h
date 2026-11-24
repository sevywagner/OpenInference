#ifndef IE_TENSOR_H
#define IE_TENSOR_H

#include <vector>
#include <numeric>
#include <stdexcept>
#include <stdio.h>
#include <cublasLt.h>

#include "Common.h"
#include "cuda_resources/GlobalHandle.h"
#include "memory/HostArena.h"
#include "memory/DeviceArena.h"

template <typename T>
class Tensor {
 private:
  T *_data;
  ::std::vector<u64> _dims;
  u64 _size;

  void move(Tensor &&other) {
    if (this == &other)
      return;

    _data = ::std::exchange(other._data, nullptr);
    _dims = ::std::move(other._dims);
    _size = ::std::exchange(other._size, 0);
  }
  
 public:
  Tensor() = delete;
 Tensor(::Memory::HostArena &ha, const ::std::vector<u64> &dims) : _dims{dims} {
    _size = ::std::accumulate(dims.begin(), dims.end(), 1, ::std::multiplies<>{});
    _data = (T *) ha.allocate(_size * sizeof(T));
  }
  
 Tensor(T *data, const ::std::vector<u64> &dims) : _data{data}, _dims{dims} {
    _size = ::std::accumulate(dims.begin(), dims.end(), 1, ::std::multiplies<>{});
  }

  Tensor(const Tensor &) = delete;
  Tensor(Tensor &&other) {
    this->move(::std::move(other));
  }

  Tensor &operator=(Tensor &&other) {
    this->move(::std::move(other));
    return *this;
  }
  
  Tensor &operator=(const Tensor &) = delete;

  const ::std::vector<u64> &get_dims() const {
    return _dims;
  }

  u64 size() {
    return _size;
  }

  T *get_ptr() {
    return _data;
  }
  
  T *copy_to_cuda(::Memory::DeviceArena &da) {
    T *res = (T *) da.allocate(_size * sizeof(T));
    CUDA_CHECK(cudaMemcpy(res, _data, _size * sizeof(T), cudaMemcpyHostToDevice));
    return res;
  }

  void copy_to_host(T *dev_ptr) {
    CUDA_CHECK(cudaMemcpy(_data, dev_ptr, _size * sizeof(T), cudaMemcpyDeviceToHost));
  }

  static Tensor matmul(Tensor &a,
		       Tensor &b,
		       ::Memory::HostArena &ha,
		       ::Memory::DeviceArena &da,
		       void *workspace,
		       u64 workspace_size,
		       cudaStream_t stream) {
    const ::std::vector<u64> &a_dims = a.get_dims();
    const ::std::vector<u64> &b_dims = b.get_dims();

    if (a_dims.size() != 2 || b_dims.size() != 2)
      throw ::std::invalid_argument("Matmul only supports 2D tensors\nN_dims received received: a (" + ::std::to_string(a_dims.size()) + "), b (" + ::std::to_string(b_dims.size()) + ")");

    T *a_dev = a.copy_to_cuda(da);
    T *b_dev = b.copy_to_cuda(da);
    
    ::std::vector<u64> out_dims = { a_dims[0], b_dims[1] };
    Tensor out(ha, out_dims);

    T *out_dev = out.copy_to_cuda(da);

    auto &handle = ::GlobalHandle::instance();
    cublas_matmul<f32, CUDA_R_32F, CUBLAS_COMPUTE_32F>(a_dev,
						       a_dims,
						       b_dev,
						       b_dims,
						       out_dev,
						       handle,
						       workspace,
						       workspace_size,
						       stream);

    a.copy_to_host(a_dev);
    b.copy_to_host(b_dev);
    out.copy_to_host(out_dev);

    return out;
  }
  
  static Tensor clone(::Memory::HostArena &ha, const Tensor &other) {
    Tensor res(ha, other._dims);
    ::std::copy(other._data, other._data + other._size, res._data);
    return res;
  }
};

template <typename T, cudaDataType_t CUDA_DT, cublasComputeType_t CUDA_CT>
void cublas_matmul(const T *a,
		   const ::std::vector<u64> &a_dims,
		   const T *b,
		   const ::std::vector<u64> &b_dims,
		   T *out,
		   cublasLtHandle_t &handle,
		   void *workspace,
		   u64 workspace_size,
		   cudaStream_t stream) {
  cublasLtMatmulDesc_t matmul_desc;
  cublasLtMatrixLayout_t a_layout, b_layout, out_layout;
  cublasLtOrder_t row_major;
  cublasLtMatmulPreference_t preference;
  cublasLtMatmulHeuristicResult_t heuristic_res;
  i32 algo_ct;
  T alpha = 1.0, beta = 0.0;
  
  CUBLAS_CHECK(cublasLtMatmulDescCreate(&matmul_desc, CUDA_CT, CUDA_DT));
  CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&a_layout, CUDA_DT, a_dims[0], a_dims[1], a_dims[1]));
  CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&b_layout, CUDA_DT, b_dims[0], b_dims[1], b_dims[1]));
  CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&out_layout, CUDA_DT, a_dims[0], b_dims[1], b_dims[1]));
  
  row_major = CUBLASLT_ORDER_ROW;
  CUBLAS_CHECK(cublasLtMatrixLayoutSetAttribute(a_layout, CUBLASLT_MATRIX_LAYOUT_ORDER, (const void *) &row_major, sizeof(cublasLtOrder_t)));
  CUBLAS_CHECK(cublasLtMatrixLayoutSetAttribute(b_layout, CUBLASLT_MATRIX_LAYOUT_ORDER, (const void *) &row_major, sizeof(cublasLtOrder_t)));
  CUBLAS_CHECK(cublasLtMatrixLayoutSetAttribute(out_layout, CUBLASLT_MATRIX_LAYOUT_ORDER, (const void *) &row_major, sizeof(cublasLtOrder_t)));

  CUBLAS_CHECK(cublasLtMatmulPreferenceCreate(&preference));
  CUBLAS_CHECK(cublasLtMatmulPreferenceSetAttribute(preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, (const void *) &workspace_size, sizeof(u64)));
  
  CUBLAS_CHECK(cublasLtMatmulAlgoGetHeuristic(handle, matmul_desc, a_layout, b_layout, NULL, out_layout, preference, 1, &heuristic_res, &algo_ct));
  CUBLAS_CHECK(cublasLtMatmul(handle,
			      matmul_desc,
			      (const void *) &alpha,
			      (const void *) a,
			      a_layout,
			      (const void *) b,
			      b_layout,
			      (const void *) &beta,
			      NULL,
			      NULL,
			      (void *) out,
			      out_layout,
			      algo_ct > 0 ? &heuristic_res.algo : NULL,
			      workspace,
			      workspace_size,
			      stream));

  cublasLtMatmulDescDestroy(matmul_desc);
  cublasLtMatrixLayoutDestroy(a_layout);
  cublasLtMatrixLayoutDestroy(b_layout);
  cublasLtMatrixLayoutDestroy(out_layout);
  cublasLtMatmulPreferenceDestroy(preference);
}

#endif
