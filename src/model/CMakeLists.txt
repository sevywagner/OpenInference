add_library(model STATIC Model.cu)
target_link_libraries(model PRIVATE ggml)
target_include_directories(model PRIVATE ${INCLUDE_DIR})

add_subdirectory(ggml)