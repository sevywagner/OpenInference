add_library(cuda_resources STATIC GlobalHandle.cu)
target_link_libraries(cuda_resources PRIVATE cublasLt)
target_include_directories(cuda_resources PRIVATE ${INCLUDE_DIR})
