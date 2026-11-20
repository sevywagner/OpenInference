cmake_minimum_required(VERSION 4.0)

if (NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
   set(CMAKE_CUDA_ARCHITECTURES native)
endif()

project(InferenceEngine
	VERSION 0.0.1
	LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)

option(INSTALL_GTEST OFF)

add_subdirectory(src)
add_subdirectory(test)
