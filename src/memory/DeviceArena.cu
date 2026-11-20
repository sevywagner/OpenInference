#include "memory/DeviceArena.h"

namespace Memory {
  DeviceArena::DeviceArena(u64 capacity)
    : _memory(capacity), _memory_it{}, _capacity(capacity) {
    CUDA_CHECK(cudaMemset(_memory.get(), 0, _capacity));
  }

  void DeviceArena::free_excess() {
    for (auto &buffer: _excess)
      buffer.reset();

    _excess.clear();
  }

  void DeviceArena::move(DeviceArena &&other) {
    if (this == &other)
      return;

    this->_memory.reset();
    this->free_excess();
  
    this->_memory = ::std::move(other._memory);
    this->_memory_it = ::std::exchange(other._memory_it, 0);
    this->_capacity = ::std::exchange(other._capacity, 0);
    this->_excess = ::std::move(other._excess);
  }

  DeviceArena::DeviceArena(DeviceArena &&other) {
    this->move(::std::move(other));
  }

  DeviceArena &DeviceArena::operator=(DeviceArena &&other) {
    this->move(::std::move(other));
    return *this;
  }

  void *DeviceArena::allocate(u64 size) {
    void *res;
    if (_memory_it + size > _capacity) {
      RAIIBufferDevice<u8> buf(size);
      res = static_cast<void *>(buf.get());
      _excess.push_back(::std::move(buf));
    } else {
      res = static_cast<void *>(_memory.get() + _memory_it);
      _memory_it += size;
    }

    return res;
  }

  void DeviceArena::reset() {
    CUDA_CHECK(cudaMemset(_memory.get(), 0, _memory_it));
  }
};
