#ifndef IE_HOST_ARENA_H
#define IE_HOST_ARENA_H

#include "Common.h"

#include <memory>
#include <span>
#include <vector>

namespace Memory {
  class HostArena {
  private:
    ::std::unique_ptr<u8[]> _memory;
    u64 _memory_it;
    u64 _capacity;

    ::std::vector< ::std::unique_ptr<u8[]> > _excess;

    void free_excess();
    void move(HostArena &&other);
  
  public:
    HostArena() = delete;
    HostArena(u64 capacity);
    HostArena(const HostArena &) = delete;
    HostArena(HostArena &&other);
    HostArena &operator=(const HostArena &) = delete;
    HostArena &operator=(HostArena &&other);
    ~HostArena() = default;

    void *allocate(u64 size);
    void reset();
  };
};
  
#endif
