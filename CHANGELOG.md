# Changelog

All notable changes to InferenceEngine will be documented in this file.

## [0.0.1] - 2026-11-20

### Added
- Initial project structure and CMake build configuration using C++20 and CUDA.
- GGUF parser for loading model files and accessing tensor metadata.
- Foundational memory management infrastructure, including host and device arenas.
- Tensor containers and supporting utilities for CPU and GPU operations.
- GPU matrix multiplication using NVIDIA cuBLAS.
- **Custom fused FlashAttention-style causal GQA prefill kernel**, designed and implemented from scratch using NVIDIA CuTe and Tensor Core MMA instructions.
  - Fused attention-score computation, online softmax, and attention/value multiplication.
  - Shared-memory tiling and asynchronous memory transfers.
  - SM80-based implementation targeting NVIDIA Ampere and newer architectures.
  - Initial support for 128-dimensional attention heads.
- GoogleTest numerical correctness tests, including comparison of GPU attention output against an independent CPU reference.

### Known Limitations
- Inference execution is not yet end-to-end functional.
- Attention currently supports prefill only and is configured for `head_dim = 128`.
- Tokenization, execution graphs, and the inference driver remain incomplete.

### Development
The entire development process, including kernel design, debugging, and numerical validation, has been livestreamed on [YouTube — @SevyWagner](https://www.youtube.com/@SevyWagner).

[0.0.1]: https://github.com/sevywagner/OpenInference/releases/tag/v0.0.1
