add_library(ggml STATIC GGUF.cu Quant.cu)
target_link_libraries(ggml PRIVATE cuda_resources)
target_include_directories(ggml PRIVATE ${INCLUDE_DIR})
