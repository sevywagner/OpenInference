#ifndef IE_GGUF_H
#define IE_GGUF_H

#include "Common.h"
#include "memory/HostArena.h"
#include "memory/DeviceArena.h"
#include "containers/Tensor.h"

#include <vector>
#include <functional>


#define MAX_TOKEN_LEN 4096
#define GGUF_MAX_STRING_LEN 65535

namespace GGML {
  enum ggml_type: uint32_t {
    GGML_TYPE_F32     = 0,
    GGML_TYPE_F16     = 1,
    GGML_TYPE_Q4_0    = 2,
    GGML_TYPE_Q4_1    = 3,
    // GGML_TYPE_Q4_2 = 4, support has been removed
    // GGML_TYPE_Q4_3 = 5, support has been removed
    GGML_TYPE_Q5_0    = 6,
    GGML_TYPE_Q5_1    = 7,
    GGML_TYPE_Q8_0    = 8,
    GGML_TYPE_Q8_1    = 9,
    GGML_TYPE_Q2_K    = 10,
    GGML_TYPE_Q3_K    = 11,
    GGML_TYPE_Q4_K    = 12,
    GGML_TYPE_Q5_K    = 13,
    GGML_TYPE_Q6_K    = 14,
    GGML_TYPE_Q8_K    = 15,
    GGML_TYPE_IQ2_XXS = 16,
    GGML_TYPE_IQ2_XS  = 17,
    GGML_TYPE_IQ3_XXS = 18,
    GGML_TYPE_IQ1_S   = 19,
    GGML_TYPE_IQ4_NL  = 20,
    GGML_TYPE_IQ3_S   = 21,
    GGML_TYPE_IQ2_S   = 22,
    GGML_TYPE_IQ4_XS  = 23,
    GGML_TYPE_I8      = 24,
    GGML_TYPE_I16     = 25,
    GGML_TYPE_I32     = 26,
    GGML_TYPE_I64     = 27,
    GGML_TYPE_F64     = 28,
    GGML_TYPE_IQ1_M   = 29,
    GGML_TYPE_BF16    = 30,
    // GGML_TYPE_Q4_0_4_4 = 31, support has been removed from gguf files
    // GGML_TYPE_Q4_0_4_8 = 32,
    // GGML_TYPE_Q4_0_8_8 = 33,
    GGML_TYPE_TQ1_0   = 34,
    GGML_TYPE_TQ2_0   = 35,
    // GGML_TYPE_IQ4_NL_4_4 = 36,
    // GGML_TYPE_IQ4_NL_4_8 = 37,
    // GGML_TYPE_IQ4_NL_8_8 = 38,
    GGML_TYPE_MXFP4   = 39, // MXFP4 (1 block)
    GGML_TYPE_COUNT   = 40,
  };

  struct GGMLTensor {
    u8 *data;
    ggml_type quant_type;
    ::std::vector<u64> dims;
  };

  using ex_graph_fn = ::std::function< Tensor<f32>(Tensor<f32> &,
						   Tensor<f32> &,
						   ::Memory::HostArena &,
						   ::Memory::DeviceArena &,
						   void *,
						   u64,
						   cudaStream_t) >;

  using ex_graph_bound_fn = ::std::function< Tensor<f32>(GGMLTensor &,
							 void *,
							 u64,
							 cudaStream_t) >;
  

  struct gguf_header_t {
    // Magic number to announce that this is a GGUF file.
    // Must be `GGUF` at the byte level: `0x47` `0x47` `0x55` `0x46`.
    // Your executor might do little-endian byte order, so it might be
    // check for 0x46554747 and letting the endianness cancel out.
    // Consider being *very* explicit about the byte order here.
    u32 magic;
    // The version of the format implemented.
    // Must be `3` for version described in this spec, which introduces big-endian support.
    //
    // This version should only be increased for structural changes to the format.
    // Changes that do not affect the structure of the file should instead update the metadata
    // to signify the change.
    u32 version;
    // The number of tensors in the file.
    // This is explicit, instead of being included in the metadata, to ensure it is always present
    // for loading the tensors.
    u64 tensor_count;
    // The number of metadata key-value pairs.
    u64 metadata_kv_count;
  };

  enum gguf_metadata_value_type: uint32_t {
    // The value is a 8-bit unsigned integer.
    GGUF_METADATA_VALUE_TYPE_UINT8 = 0,
    // The value is a 8-bit signed integer.
    GGUF_METADATA_VALUE_TYPE_INT8 = 1,
    // The value is a 16-bit unsigned little-endian integer.
    GGUF_METADATA_VALUE_TYPE_UINT16 = 2,
    // The value is a 16-bit signed little-endian integer.
    GGUF_METADATA_VALUE_TYPE_INT16 = 3,
    // The value is a 32-bit unsigned little-endian integer.
    GGUF_METADATA_VALUE_TYPE_UINT32 = 4,
    // The value is a 32-bit signed little-endian integer.
    GGUF_METADATA_VALUE_TYPE_INT32 = 5,
    // The value is a 32-bit IEEE754 floating point number.
    GGUF_METADATA_VALUE_TYPE_FLOAT32 = 6,
    // The value is a boolean.
    // 1-byte value where 0 is false and 1 is true.
    // Anything else is invalid, and should be treated as either the model being invalid or the reader being buggy.
    GGUF_METADATA_VALUE_TYPE_BOOL = 7,
    // The value is a UTF-8 non-null-terminated string, with length prepended.
    GGUF_METADATA_VALUE_TYPE_STRING = 8,
    // The value is an array of other values, with the length and type prepended.
    ///
    // Arrays can be nested, and the length of the array is the number of elements in the array, not the number of bytes.
    GGUF_METADATA_VALUE_TYPE_ARRAY = 9,
    // The value is a 64-bit unsigned little-endian integer.
    GGUF_METADATA_VALUE_TYPE_UINT64 = 10,
    // The value is a 64-bit signed little-endian integer.
    GGUF_METADATA_VALUE_TYPE_INT64 = 11,
    // The value is a 64-bit IEEE754 floating point number.
    GGUF_METADATA_VALUE_TYPE_FLOAT64 = 12,
  };

  enum rope_scaling_type: u32 {
    NONE,
    LINEAR,
    YARN
  };

  struct gguf_file {
    gguf_header_t header;
    void *data;
    u64 size; 
    i32 fd;

    ::std::string model_name, version, tensor_layout; // replace with c-style
    u64 context_length, embd_len, block_ct, ffn_len, head_ct, kv_head_ct, rope_dim_ct;
    u32 alignment;
  
    u32 expert_ct, expert_used_ct;
    bool use_par_residual;
    ::std::string model_architecture;

    rope_scaling_type rope_scale_type;
    f32 rope_scaling_factor, rope_scale_linear;
    u32 rope_og_context_len, rope_freq_base;

    f32 layer_norm_rms_epsilon;

    ::std::vector<::std::string> tokens;
    u32 bos_token_id, eos_token_id, unknown_token_id, seperator_token_id, padding_token_id;
    ::std::vector<f32> scores;
    ::std::vector<::std::string> merges;
    ::std::string chat_template, token_ggml_model;

    ::std::vector<ex_graph_bound_fn> execution_graph;
  };

  void gguf_open(::std::string gguf_path,
		 gguf_file &res,
		 ::Memory::HostArena &host_arena,
		 ::Memory::DeviceArena &device_arena);

  void gguf_close(gguf_file &file);
};

#endif
