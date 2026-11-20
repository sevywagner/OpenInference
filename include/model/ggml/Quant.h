#ifndef IE_QUANT_H
#define IE_QUANT_H

#include "model/ggml/GGUF.h"
#include "containers/Tensor.h"

namespace GGML::Quant {
  Tensor<f32> dequant(::GGML::GGMLTensor &tensor);
};

#endif
