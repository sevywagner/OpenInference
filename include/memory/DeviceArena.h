#ifndef IE_DEVICE_ARENA_H
#define IE_DEVICE_ARENA_H

#include "Common.h"
#include "RAII_device.h"

#include <vector>
#include <span>

namespace Memory {
  class DeviceArena {
  private:
    RAIIBufferDevice<u8> _memory;
    u64 _memory_it;
    u64 _capacity;

    ::std::vector< RAIIBufferDevice<u8> > _excess;

    void move(DeviceArena &&other);
    void free_excess();
  
  public:
    DeviceArena() = delete;
    DeviceArena(u64 capacity);
    DeviceArena(const DeviceArena &) = delete;
    DeviceArena(DeviceArena &&other);
    DeviceArena &operator=(const DeviceArena &) = delete;
    DeviceArena &operator=(DeviceArena &&other);
    ~DeviceArena() = default;
  
    void *allocate(u64 size);
    void reset();
  };
};
  
#endif
