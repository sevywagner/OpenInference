add_library(inference Error.cu)
target_link_libraries(inference PRIVATE memory cuda_resources model)
target_include_directories(inference PRIVATE ${INCLUDE_DIR})

add_executable(main main.cu)
target_link_libraries(main PRIVATE inference)
target_include_directories(main PRIVATE ${INCLUDE_DIR})

add_subdirectory(memory)
add_subdirectory(model)
add_subdirectory(cuda_resources)
