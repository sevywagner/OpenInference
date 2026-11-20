#include "memory/HostArena.h"
#include <utility>

namespace Memory {
  HostArena::HostArena(u64 capacity)
    : _memory_it{0}, _capacity(capacity) {
    _memory = ::std::make_unique<u8[]>(capacity);
    ::std::fill(_memory.get(), _memory.get() + _capacity, 0);
  }

  void HostArena::free_excess() {
    for (auto &ptr: _excess)
      ptr.reset();

    _excess.clear();
  }

  void HostArena::move(HostArena &&other) {
    if (this == &other)
      return;

    this->_memory.reset();
    this->free_excess();
  
    this->_memory = ::std::move(other._memory);
    this->_memory_it = ::std::exchange(other._memory_it, 0);
    this->_capacity = ::std::exchange(other._capacity, 0);
    this->_excess = ::std::move(other._excess);
  }

  HostArena::HostArena(HostArena &&other) {
    this->move(::std::move(other));
  }

  HostArena &HostArena::operator=(HostArena &&other) {
    this->move(::std::move(other));
    return *this;
  }

  void *HostArena::allocate(u64 size) {
    void *res;

    if (_memory_it + size > _capacity) {
      auto ptr = ::std::make_unique<u8[]>(size);
      res = static_cast<void *>(ptr.get());
      _excess.push_back(::std::move(ptr));
    } else {
      res = static_cast<void *>(_memory.get() + _memory_it);
      _memory_it += size;
    }

    return res;
  }

  void HostArena::reset() {
    ::std::fill(_memory.get(), _memory.get() + _memory_it, 0);
    this->free_excess();
  }
};
