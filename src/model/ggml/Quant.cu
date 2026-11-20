#include "model/ggml/Quant.h"

#include <unordered_map>
#include <functional>

namespace GGML::Quant {
  static Tensor<f32> f32_handler(GGMLTensor &tensor) {
    return Tensor<f32>((f32 *) tensor.data, tensor.dims);
  }
  
  static const ::std::unordered_map< ggml_type, ::std::function< Tensor<f32>(GGMLTensor &) > > dequant_handler_map = {
    { ggml_type::GGML_TYPE_F32, f32_handler }
  };

  Tensor<f32> dequant(GGMLTensor &tensor) {
    return dequant_handler_map.at(tensor.quant_type)(tensor);
  }
};
