#include "model/Model.h"

using namespace GGML;
using namespace Memory;

Model::Model(::std::string gguf_file_path,
	     HostArena &host_arena,
	     DeviceArena &device_arena) {
  gguf_open(gguf_file_path, _file, host_arena, device_arena);
}

void Model::move(Model &&other) {
  if (this == &other)
    return;

  this->_file = ::std::move(other._file);
}

Model::Model(Model &&other) {
  this->move(::std::move(other));
}

Model &Model::operator=(Model &&other) {
  this->move(::std::move(other));
  return *this;
}

Model::~Model() {
  gguf_close(_file);
}
