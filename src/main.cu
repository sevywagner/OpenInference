#include "Common.h"
#include "memory/HostArena.h"
#include "memory/DeviceArena.h"
#include "containers/Tensor.h"
#include "model/ggml/GGUF.h"
#include "model/Model.h"
#include "model/decoder_layer/Attention.h"

#include "cute/tensor.hpp"

#include <iostream>
#include <numeric>
#include <limits>
#include <sys/param.h>

#include <string.h>

using namespace GGML;
using namespace Memory;

int main() {
  ::std::cout << "This software is still in development. Driver not yet implemented. " << ::std::endl;
  
  return 0;
}
