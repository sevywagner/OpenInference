#include "model/ggml/GGUF.h"
#include "model/Model.h"
#include "model/ggml/Quant.h"

#include <fcntl.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>
#include <sys/stat.h>

#include <stdexcept>
#include <iostream>
#include <unordered_map>
#include <sstream>

using namespace ::std::placeholders;
using namespace Memory;

namespace GGML {
  static u64 gguf_dtype_sizes[] = {
    sizeof(u8),
    sizeof(i8),
    sizeof(u16),
    sizeof(i16),
    sizeof(u32),
    sizeof(i32),
    sizeof(f32),
    sizeof(bool),
    0,
    0,
    sizeof(u64),
    sizeof(i64),
    sizeof(f64),
  };

  static void gguf_zero(gguf_file &file) {
    file = (gguf_file) {
      .data = NULL,
      .size = 0,
      .fd = -1
    };
  }

  static void gguf_parse_value(u8 **src_ptr,
			       gguf_metadata_value_type dtype,
			       void *dest) {
    u64 size_bytes;
    switch (dtype) {
    case GGUF_METADATA_VALUE_TYPE_STRING:
      char tmp[GGUF_MAX_STRING_LEN];
    
      u64 str_len;
      memcpy(&str_len, *src_ptr, sizeof(u64));
      *src_ptr += sizeof(u64);

      size_bytes = str_len * sizeof(char);
      memcpy(tmp, *src_ptr, size_bytes);
      *src_ptr += size_bytes;

      tmp[str_len] = '\0';
      *(::std::string *)dest = ::std::string(tmp, str_len);
    
      break;
    default:
      memcpy(dest, *src_ptr, gguf_dtype_sizes[dtype]);
      *src_ptr += gguf_dtype_sizes[dtype];
      break;
    }
  }

  template <typename T>
  static ::std::vector<T> gguf_parse_array(u8 **src_ptr,
					   gguf_metadata_value_type dtype,
					   HostArena &host_arena) {
    gguf_metadata_value_type arr_dtype;
    memcpy(&arr_dtype, *src_ptr, sizeof(u32));
    *src_ptr += sizeof(u32);

    u64 arr_len;
    memcpy(&arr_len, *src_ptr, sizeof(u64));
    *src_ptr += sizeof(u64);

    ::std::vector<T> res;
    res.reserve(arr_len);

    if (arr_dtype == gguf_metadata_value_type::GGUF_METADATA_VALUE_TYPE_ARRAY) {
      // todo
    }

    for (u64 i{}; i < arr_len; i++) {
      T val;
      gguf_parse_value(src_ptr, arr_dtype, &val);
      res.push_back(val);
    }
  
    return res;  
  }

  enum ggml_metadata_handlers: u32 {
    GENERAL_ARCHITECTURE,
    GENERAL_NAME,
    GENERAL_VERSION,
    GENERAL_ALIGNMENT,
    CONTEXT_LENGTH,
    EMBEDDING_LENGTH,
    BLOCK_COUNT,
    FEED_FORWARD_LENGTH,
    USE_PARALLEL_RESIDUAL,
    TENSOR_DATA_LAYOUT,
    EXPERT_COUNT,
    EXPERT_USED_COUNT,
    ATTENTION_HEAD_COUNT,
    ATTENTION_KV_HEAD_COUNT,
    ROPE_DIMENSION_COUNT,
    ROPE_FREQ_BASE,
    ROPE_SCALING_TYPE,
    ROPE_SCALING_FACTOR,
    ROPE_SCALING_ORIGINAL_CONTEXT_LENGTH,
    ROPE_SCALE_LINEAR,
    LAYER_NORM_RMS_EPSILON,
    TOKEN_GGML_MODEL,
    TOKEN_GGML_TOKENS,
    TOKEN_GGML_SCORES,
    TOKEN_GGML_MERGES,
    TOKEN_GGML_BOS_TOKEN,
    TOKEN_GGML_EOS_TOKEN,
    TOKEN_GGML_UNKNOWN_TOKEN,
    TOKEN_GGML_SEPERATOR_TOKEN,
    TOKEN_GGML_PADDING_TOKEN,
    TOKENIZER_CHAT_TEMPLATE
  };

  static const ::std::unordered_map<::std::string, u32> gguf_metadata_handler_map = {
    { "general.architecture", GENERAL_ARCHITECTURE },
    { "general.name", GENERAL_NAME },
    { "general.version", GENERAL_VERSION },
    { "general.alignment", GENERAL_ALIGNMENT },
    { "context_length", CONTEXT_LENGTH },
    { "embedding_length", EMBEDDING_LENGTH },
    { "block_count", BLOCK_COUNT },
    { "feed_forward_length", FEED_FORWARD_LENGTH },
    { "use_parallel_residual", USE_PARALLEL_RESIDUAL },
    { "tensor_data_layout", TENSOR_DATA_LAYOUT },
    { "expert_count", EXPERT_COUNT },
    { "expert_used_count", EXPERT_USED_COUNT },
    { "attention.head_count", ATTENTION_HEAD_COUNT },
    { "attention.kv_head_count", ATTENTION_KV_HEAD_COUNT },
    { "rope.dimension_count", ROPE_DIMENSION_COUNT },
    { "rope.freq_base", ROPE_FREQ_BASE },
    { "attention.layer_norm_rms_epsilon", LAYER_NORM_RMS_EPSILON },
    { "tokenizer.ggml.model", TOKEN_GGML_MODEL },
    { "tokenizer.ggml.tokens", TOKEN_GGML_TOKENS },
    { "tokenizer.ggml.scores", TOKEN_GGML_SCORES },
    { "tokenizer.ggml.merges", TOKEN_GGML_MERGES },
    { "tokenizer.ggml.bos_token_id", TOKEN_GGML_BOS_TOKEN },
    { "tokenizer.ggml.eos_token_id", TOKEN_GGML_EOS_TOKEN },
    { "tokenizer.ggml.unknown_token_id", TOKEN_GGML_UNKNOWN_TOKEN },
    { "tokenizer.ggml.seperator_token_id", TOKEN_GGML_SEPERATOR_TOKEN },
    { "tokenizer.ggml.padding_token_id", TOKEN_GGML_PADDING_TOKEN },
    { "rope.scale_linear", ROPE_SCALE_LINEAR }
  };

  enum ggml_tensor_block_handlers: u32 {
    TOKEN_EMBD,
    POS_EMBD,
    OUTPUT_NORM,
    OUTPUT,
    ATTN_NORM,
    ATTN_NORM_2,
    ATTN_QKV,
    ATTN_Q,
    ATTN_Q_NORM,
    ATTN_K,
    ATTN_K_NORM,
    ATTN_V,
    ATTN_OUTPUT,
    FFN_NORM,
    FFN_UP,
    FFN_GATE,
    FFN_DOWN,
    FFN_GATE_EXP,
    FFN_GATE_INP,
    FFN_DOWN_EXP,
    FFN_UP_EXP
  };

  const static ::std::unordered_map<::std::string, u32> gguf_tensor_handler_map = {
    { "token_embd.weight", TOKEN_EMBD },
    { "pos_embd.weight", POS_EMBD },
    { "output_norm.weight", OUTPUT_NORM },
    { "output.weight", OUTPUT },
    { "attn_norm.weight", ATTN_NORM },
    { "attn_norm_2.weight", ATTN_NORM_2 },
    { "attn_qkv.weight", ATTN_QKV },
    { "attn_q.weight", ATTN_Q },
    { "attn_q_norm.weight", ATTN_Q_NORM },
    { "attn_k.weight", ATTN_K },
    { "attn_k_norm.weight", ATTN_K_NORM },
    { "attn_v.weight", ATTN_V },
    { "attn_output.weight", ATTN_OUTPUT },
    { "ffn_norm.weight", FFN_NORM },
    { "ffn_up.weight", FFN_UP },
    { "ffn_gate.weight", FFN_GATE },
    { "ffn_down.weight", FFN_DOWN },
    { "ffn_gate_exp.weight", FFN_GATE_EXP },
    { "ffn_gate_inp.weight", FFN_GATE_INP },
    { "ffn_down_exp.weight", FFN_GATE_EXP },
    { "ffn_up_exp.weight", FFN_UP_EXP }
  };

  static ::std::unordered_map< ::std::string, ex_graph_fn > decoder_layer_fn_map = {
    { "output.weight", Tensor<f32>::matmul }
  };

  static void skip_str(u8 **src_ptr) {
    u64 str_len;
    gguf_parse_value(src_ptr, GGUF_METADATA_VALUE_TYPE_UINT64, &str_len);
    *src_ptr += str_len;
  }

  static void skip_array(u8 **src_ptr) {
    gguf_metadata_value_type arr_type;
    gguf_parse_value(src_ptr, GGUF_METADATA_VALUE_TYPE_UINT32, &arr_type);
  
    u64 len;
    gguf_parse_value(src_ptr, GGUF_METADATA_VALUE_TYPE_UINT64, &len);

    switch (arr_type) {
    case GGUF_METADATA_VALUE_TYPE_STRING:
      for (u64 i{}; i < len; i++)
	skip_str(src_ptr);

      break;
    case GGUF_METADATA_VALUE_TYPE_ARRAY:
      for (u64 i{}; i < len; i++) // consider iterative approach
	skip_array(src_ptr);
      
      break;
    default:
      *src_ptr += len * gguf_dtype_sizes[arr_type];
    
      break;
    }
  }

  static void skip_metadata(u8 **src_ptr, gguf_metadata_value_type dtype) {
    switch (dtype) {
    case GGUF_METADATA_VALUE_TYPE_STRING:
      skip_str(src_ptr);
      break;
    case GGUF_METADATA_VALUE_TYPE_ARRAY:
      skip_array(src_ptr);
      break;
    default:
      *src_ptr += gguf_dtype_sizes[dtype];
      break;
    }
  }
  
  Tensor<f32> ex_graph_wrapper(ex_graph_fn &fn,
			       GGMLTensor &hidden_state,
			       GGMLTensor &weight,
			       HostArena &ha,
			       DeviceArena &da,
			       void *workspace,
			       u64 workspace_size,
			       cudaStream_t stream) {
    auto hidden_state_tensor = Quant::dequant(hidden_state);
    auto weight_tensor = Quant::dequant(weight);

    return fn(hidden_state_tensor, weight_tensor, ha, da, workspace, workspace_size, stream);
  }

  static void gguf_read(u8 *src_ptr, gguf_file &file, HostArena &host_arena, DeviceArena &device_arena) {
    ::std::string key, rope_scale_type;

    file.alignment = 32;
  
    for (u32 i{}; i < file.header.metadata_kv_count; i++) {
      gguf_parse_value(&src_ptr, GGUF_METADATA_VALUE_TYPE_STRING, &key);
    
      u32 handler_id = ::std::numeric_limits<u32>::max();
      if (gguf_metadata_handler_map.contains(::std::string(key))) {
	handler_id = gguf_metadata_handler_map.at(::std::string(key));
      } else {
	::std::stringstream ss(key);
	::std::string segment;
	for (u8 i{}; i < 2; i++) // skip the [llm] segment, store metadata key
	  getline(ss, segment, '.');
      
	if (gguf_metadata_handler_map.contains(segment))
	  handler_id = gguf_metadata_handler_map.at(::std::string(segment));
      }
   
      gguf_metadata_value_type dtype;
      gguf_parse_value(&src_ptr, GGUF_METADATA_VALUE_TYPE_UINT32, &dtype);

      switch (handler_id) {
      case ggml_metadata_handlers::GENERAL_ARCHITECTURE:
	gguf_parse_value(&src_ptr, dtype, &file.model_architecture);
	break;
      case ggml_metadata_handlers::GENERAL_NAME:
	gguf_parse_value(&src_ptr, dtype, &file.model_name);
	break;
      case ggml_metadata_handlers::GENERAL_VERSION:
	gguf_parse_value(&src_ptr, dtype, &file.version);
	break;
      case ggml_metadata_handlers::GENERAL_ALIGNMENT:
	gguf_parse_value(&src_ptr, dtype, &file.alignment);
	if (file.alignment % 8 != 0)
	  throw ::std::runtime_error("GGUF malformed: alignment, " + ::std::to_string(file.alignment) + ", must be a multiple of 8");
	break;
      case ggml_metadata_handlers::CONTEXT_LENGTH:
	gguf_parse_value(&src_ptr, dtype, &file.context_length);
	break;
      case ggml_metadata_handlers::EMBEDDING_LENGTH:
	gguf_parse_value(&src_ptr, dtype, &file.embd_len);
	break; 
      case ggml_metadata_handlers::BLOCK_COUNT:
	gguf_parse_value(&src_ptr, dtype, &file.block_ct);
	break;
      case ggml_metadata_handlers::FEED_FORWARD_LENGTH:
	gguf_parse_value(&src_ptr, dtype, &file.ffn_len);
	break;
      case ggml_metadata_handlers::USE_PARALLEL_RESIDUAL:
	gguf_parse_value(&src_ptr, dtype, &file.use_par_residual);
	break;
      case ggml_metadata_handlers::TENSOR_DATA_LAYOUT:
	gguf_parse_value(&src_ptr, dtype, &file.tensor_layout);
	break;
      case ggml_metadata_handlers::EXPERT_COUNT:
	gguf_parse_value(&src_ptr, dtype, &file.expert_ct);
	break;
      case ggml_metadata_handlers::EXPERT_USED_COUNT:
	gguf_parse_value(&src_ptr, dtype, &file.expert_used_ct);
	break;
      case ggml_metadata_handlers::ATTENTION_HEAD_COUNT:
	gguf_parse_value(&src_ptr, dtype, &file.head_ct);
	break;
      case ggml_metadata_handlers::ATTENTION_KV_HEAD_COUNT:
	gguf_parse_value(&src_ptr, dtype, &file.kv_head_ct);
	break;
      case ggml_metadata_handlers::ROPE_DIMENSION_COUNT:
	gguf_parse_value(&src_ptr, dtype, &file.rope_dim_ct);
	break;
      case ggml_metadata_handlers::ROPE_SCALING_TYPE:
	gguf_parse_value(&src_ptr, dtype, &rope_scale_type);

	if (rope_scale_type == "none")
	  file.rope_scale_type = rope_scaling_type::NONE;
	else if (rope_scale_type == "linear")
	  file.rope_scale_type = rope_scaling_type::LINEAR;
	else if (rope_scale_type == "yarn")
	  file.rope_scale_type = rope_scaling_type::YARN;
	else
	  file.rope_scale_type = rope_scaling_type::NONE;
      
	break;
      case ggml_metadata_handlers::ROPE_FREQ_BASE:
	gguf_parse_value(&src_ptr, dtype, &file.rope_freq_base);
	break;
      case ggml_metadata_handlers::ROPE_SCALING_FACTOR:
	gguf_parse_value(&src_ptr, dtype, &file.rope_scaling_factor);
	break;  
      case ggml_metadata_handlers::ROPE_SCALING_ORIGINAL_CONTEXT_LENGTH:
	gguf_parse_value(&src_ptr, dtype, &file.rope_og_context_len);
	break;      
      case ggml_metadata_handlers::ROPE_SCALE_LINEAR:
	gguf_parse_value(&src_ptr, dtype, &file.rope_scale_linear);
	break;      
      case ggml_metadata_handlers::LAYER_NORM_RMS_EPSILON:
	gguf_parse_value(&src_ptr, dtype, &file.layer_norm_rms_epsilon);
	break;
      case ggml_metadata_handlers::TOKEN_GGML_MODEL:
	gguf_parse_value(&src_ptr, dtype, &file.token_ggml_model);
	break;
      case ggml_metadata_handlers::TOKEN_GGML_TOKENS:
	if (dtype != GGUF_METADATA_VALUE_TYPE_ARRAY)
	  ::std::runtime_error("Metadata dtype inconsistent");

	file.tokens = gguf_parse_array<::std::string>(&src_ptr, dtype, host_arena);
      	
	break;
      case ggml_metadata_handlers::TOKEN_GGML_SCORES:
	file.scores = gguf_parse_array<f32>(&src_ptr, dtype, host_arena);
	break;
      case ggml_metadata_handlers::TOKEN_GGML_MERGES:
	file.merges = gguf_parse_array<::std::string>(&src_ptr, dtype, host_arena);
	gguf_parse_value(&src_ptr, dtype, &file.layer_norm_rms_epsilon);
	break;
      case ggml_metadata_handlers::TOKEN_GGML_BOS_TOKEN:
	gguf_parse_value(&src_ptr, dtype, &file.bos_token_id);
	break;
      case ggml_metadata_handlers::TOKEN_GGML_EOS_TOKEN:
	gguf_parse_value(&src_ptr, dtype, &file.eos_token_id);
	break;
      case ggml_metadata_handlers::TOKEN_GGML_UNKNOWN_TOKEN:
	gguf_parse_value(&src_ptr, dtype, &file.unknown_token_id);
	break;
      case ggml_metadata_handlers::TOKEN_GGML_SEPERATOR_TOKEN:
	gguf_parse_value(&src_ptr, dtype, &file.seperator_token_id);
	break;
      case ggml_metadata_handlers::TOKEN_GGML_PADDING_TOKEN:
	gguf_parse_value(&src_ptr, dtype, &file.padding_token_id);
	break;
      case ggml_metadata_handlers::TOKENIZER_CHAT_TEMPLATE:
	gguf_parse_value(&src_ptr, dtype, &file.chat_template);
	break;
      default:
	skip_metadata(&src_ptr, dtype);
	break;
      };    
    }

    auto decoder_layer_tensors = ::std::make_unique< ::std::unordered_map<::std::string, GGMLTensor>[] >(file.block_ct);
  
    ::std::unordered_map<::std::string, GGMLTensor> global_tensor_map;
  
    for (u64 i{}; i < file.header.tensor_count; i++) {
      ::std::string tensor_key;
      gguf_parse_value(&src_ptr, gguf_metadata_value_type::GGUF_METADATA_VALUE_TYPE_STRING, &tensor_key);
      
      u32 n_dims;
      gguf_parse_value(&src_ptr, gguf_metadata_value_type::GGUF_METADATA_VALUE_TYPE_UINT32, &n_dims);

      ::std::vector<u64> dims;
      dims.reserve(n_dims);

      for (u32 j{}; j < n_dims; j++) {
	u64 dim;
	gguf_parse_value(&src_ptr, gguf_metadata_value_type::GGUF_METADATA_VALUE_TYPE_UINT64, &dim);
	dims.push_back(dim);
      }

      ggml_type quant_type;
      gguf_parse_value(&src_ptr, gguf_metadata_value_type::GGUF_METADATA_VALUE_TYPE_UINT32, &quant_type);

      u32 handler_id = ::std::numeric_limits<u32>::max();
      u32 blk_idx = ::std::numeric_limits<u32>::max();
      if (gguf_tensor_handler_map.contains(tensor_key))
	handler_id = gguf_tensor_handler_map.at(tensor_key);

    
      if (handler_id == ::std::numeric_limits<u32>::max()) {
	::std::stringstream ss(tensor_key);
	::std::string segment;
	getline(ss, segment, '.');
	if (segment != "blk")
	  throw ::std::runtime_error("Tensor name unexpected: " + tensor_key);

	getline(ss, segment, '.');
	blk_idx = ::std::stoi(segment);
	getline(ss, segment, '.');
	tensor_key = segment;
	getline(ss, segment, '.');
	tensor_key += "." + segment;
	if (!gguf_tensor_handler_map.contains(tensor_key))
	  throw ::std::runtime_error("Tensor name unexpected: " + tensor_key);

	handler_id = gguf_tensor_handler_map.at(tensor_key);
      }

      u64 offset;
      gguf_parse_value(&src_ptr, gguf_metadata_value_type::GGUF_METADATA_VALUE_TYPE_UINT64, &offset);
      
      switch(handler_id) {
      case ggml_tensor_block_handlers::TOKEN_EMBD:
      case ggml_tensor_block_handlers::POS_EMBD:
      case ggml_tensor_block_handlers::OUTPUT_NORM:
      case ggml_tensor_block_handlers::OUTPUT:
	global_tensor_map[tensor_key] = (GGMLTensor) {
	  .data = (u8 *) offset,
	  .quant_type = quant_type,
	  .dims = dims
	};
      
	break;
      case ggml_tensor_block_handlers::ATTN_NORM:
      case ggml_tensor_block_handlers::ATTN_NORM_2:
      case ggml_tensor_block_handlers::ATTN_QKV:
      case ggml_tensor_block_handlers::ATTN_Q:
      case ggml_tensor_block_handlers::ATTN_Q_NORM:
      case ggml_tensor_block_handlers::ATTN_K:
      case ggml_tensor_block_handlers::ATTN_K_NORM:
      case ggml_tensor_block_handlers::ATTN_V:
      case ggml_tensor_block_handlers::ATTN_OUTPUT:
      case ggml_tensor_block_handlers::FFN_NORM:
      case ggml_tensor_block_handlers::FFN_UP:
      case ggml_tensor_block_handlers::FFN_GATE:
      case ggml_tensor_block_handlers::FFN_DOWN:
      case ggml_tensor_block_handlers::FFN_GATE_INP:
      case ggml_tensor_block_handlers::FFN_GATE_EXP:
      case ggml_tensor_block_handlers::FFN_DOWN_EXP:
      case ggml_tensor_block_handlers::FFN_UP_EXP:
	decoder_layer_tensors[blk_idx][tensor_key] = (GGMLTensor) {
	  .data = (u8 *) offset,
	  .quant_type = quant_type,
	  .dims = dims
	};
       
	break;
      default:
	throw ::std::runtime_error("Unexpected tensor name: " + tensor_key);
	break;
      }
    }

    u64 offset = (u64) src_ptr - (u64) file.data;
    auto tensor_offset = offset + (file.alignment - (offset % file.alignment)) % file.alignment;

    for (auto &p: global_tensor_map) {
      p.second.data += tensor_offset;
    }

    for (u32 i{}; i < file.block_ct; i++) {
      for (auto &p: decoder_layer_tensors[i]) {
	p.second.data += tensor_offset;
      }
    }

    ::std::vector<ex_graph_bound_fn> ex_graph;

    for (u64 i{}; i < file.block_ct; i++) {
      for (auto &fn: decoder_layer_fn_map)
	if (decoder_layer_tensors[i].contains(fn.first)) {
	  ex_graph_bound_fn bound_fn = ::std::bind(ex_graph_wrapper,
						   fn.second,
						   _1,
						   decoder_layer_tensors[i].at(fn.first),
						   ::std::ref(host_arena),
						   ::std::ref(device_arena),
						   _2,
						   _3,
						   _4);

	  ex_graph.push_back(bound_fn);
	}
    }
  }

  void gguf_open(::std::string gguf_path, gguf_file &file, HostArena &host_arena, DeviceArena &device_arena) {
    // ensure gguf file not already associated with different gguf
    gguf_zero(file);
  
    if ((file.fd = open(gguf_path.c_str(), 0, S_IRUSR)) == -1)
      throw ::std::runtime_error("GGUF file open failure");

    struct stat st;
    if (fstat(file.fd, &st) == -1)
      throw ::std::invalid_argument("GGUF stat failure");

    file.size = st.st_size;
  
    file.data = mmap(NULL, file.size, PROT_READ, MAP_SHARED, file.fd, 0);
    if (file.data == MAP_FAILED) {
      file.data = NULL;
      throw ::std::runtime_error("GGUF mmap failure");
    }

    u8 *src_ptr = (u8 *) file.data;
  
    if (src_ptr[0] != 0x47 &&
	src_ptr[1] != 0x47 &&
	src_ptr[2] != 0x55 &&
	src_ptr[3] != 0x46)
      throw ::std::runtime_error("Invalid GGUF magic");

    gguf_parse_value(&src_ptr, GGUF_METADATA_VALUE_TYPE_UINT32, &file.header.magic);
    gguf_parse_value(&src_ptr, GGUF_METADATA_VALUE_TYPE_UINT32, &file.header.version);
    gguf_parse_value(&src_ptr, GGUF_METADATA_VALUE_TYPE_UINT64, &file.header.tensor_count);
    gguf_parse_value(&src_ptr, GGUF_METADATA_VALUE_TYPE_UINT64, &file.header.metadata_kv_count);

    gguf_read(src_ptr, file, host_arena, device_arena);
  }

  void gguf_close(gguf_file &file) {
    if (file.fd >= 0) {
      if (file.data)
	munmap(file.data, file.size);

      close(file.fd);
    }

    gguf_zero(file);
  }
};
