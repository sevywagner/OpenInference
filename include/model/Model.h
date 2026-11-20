#ifndef IE_MODEL_H
#define IE_MODEL_H

#include "model/ggml/GGUF.h"
#include "memory/HostArena.h"
#include "memory/DeviceArena.h"

#include <vector>

class Model {
 private:
  ::GGML::gguf_file _file;
  
  void move(Model &&other);
  
 public:
  Model(::std::string gguf_file_path, ::Memory::HostArena &host_arena, ::Memory::DeviceArena &device_arena);

  Model(const Model &) = delete;
  Model(Model &&other);
  Model &operator=(const Model &) = delete;
  Model &operator=(Model &&other);
  ~Model();
};

#endif
