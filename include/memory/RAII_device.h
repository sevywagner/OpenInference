#ifndef IE_RAII_DEVICE_H
#define IE_RAII_DEVICE_H

#include "Common.h"
#include <cuda_runtime.h>

namespace Memory {
  template <typename T>
  class RAIIBufferDevice {
  private:
    T *_ptr;
    u64 _size;

    void move(RAIIBufferDevice &&other) {
      if (this == &other)
	return;
    
      this->_ptr = ::std::exchange(other._ptr, nullptr);
      this->_size = ::std::exchange(other._size, 0);
    }
  
  public:
    RAIIBufferDevice() : _ptr{ nullptr }, _size{} {}
    RAIIBufferDevice(u64 size) : _size(size) {
      CUDA_CHECK(cudaMalloc((void **) &_ptr, size));
    }
  
    RAIIBufferDevice(const RAIIBufferDevice &) = delete;
    RAIIBufferDevice(RAIIBufferDevice &&other) {
      this->move(::std::move(other));
    }
  
    RAIIBufferDevice &operator=(const RAIIBufferDevice &) = delete;
    RAIIBufferDevice &operator=(RAIIBufferDevice &&other) {
      this->move(::std::move(other));
      return *this;
    }
  
    ~RAIIBufferDevice() {
      this->reset();
    }

    T *get() {
      return _ptr;
    }

    u64 size() {
      return _size;
    }

    void reset() {
      if (_ptr)
	CUDA_CHECK(cudaFree(_ptr));

      _ptr = nullptr;
      _size = 0;
    }
  };
};
  
#endif
