#include "cuda_resources/GlobalHandle.h"
#include "Error.h"

GlobalHandle *GlobalHandle::_global_handle = nullptr;

GlobalHandle::GlobalHandle() {
  CUBLAS_CHECK(cublasLtCreate(&_handle));
}

cublasLtHandle_t &GlobalHandle::instance() {
  if (!_global_handle) {
    _global_handle = new GlobalHandle();
    ::std::atexit([] { delete _global_handle; });
  }

  return _global_handle->_handle;
}

GlobalHandle::~GlobalHandle() {
  cublasLtDestroy(_handle);
}
