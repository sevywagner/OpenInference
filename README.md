
# InferenceEngine

**v0.0.1** · **CUDA C++** · **Work in progress**

A GPU-first language-model inference engine built from the ground up in CUDA C++. The project focuses on the systems and low-level GPU kernels behind inference, from model loading and memory management to Tensor Core attention.

> **Development livestreams:** The entire development of this software has been streamed on YouTube: **[youtube.com/@SevyWagner](https://www.youtube.com/@SevyWagner)**. The streams cover the design, implementation, debugging, and testing process.

## Current state — v0.0.1

The project currently includes the core infrastructure, a custom attention kernel, and a baseline RMSNorm kernel. **The inference engine is not yet end-to-end functional**; the tokenizer, execution path, and driver are still being developed.

| Component | Status | Description |
| --- | --- | --- |
| **GGUF parser** | Implemented | Infrastructure for reading GGUF model files and tensor metadata. |
| **Memory management** | Implemented | Host and device arenas for managing allocations. |
| **Tensors** | Implemented | Foundational tensor containers and host/device data handling. |
| **Matrix multiplication** | Implemented | GPU matrix multiplication backed by **cuBLAS**. |
| **Fused attention prefill** | Implemented | Custom causal grouped-query attention (GQA) prefill kernel, written with **CuTe** and Tensor Core MMA instructions. |
| **RMSNorm** | Implemented | Custom fused RMS normalization kernel using warp shuffles and shared-memory reduction, supporting in-place and out-of-place output. |
| **Numerical tests** | Available | GoogleTest suite covering matrix multiplication and comparing attention and RMSNorm kernels against CPU references. |

### Custom fused attention kernel

The primary GPU milestone in v0.0.1 is a **fused, FlashAttention-style causal GQA prefill kernel** that I designed and implemented myself using **CuTe**, the low-level tensor programming framework within NVIDIA CUTLASS.

The kernel uses Tensor Core matrix multiply-accumulate instructions for both attention-score computation and the attention/value product, with online softmax to avoid materializing the full attention matrix in global memory.

**Current limitations:**

- Designed around **SM80 Tensor Core instructions** and compilable for **NVIDIA Ampere and newer GPUs** with an appropriate CUDA toolchain.
- Currently configured specifically for **128-dimensional attention heads** (`head_dim = 128`).
- Supports the **prefill** phase; a separate decode kernel is planned.

The current head-dimension restriction is primarily a matter of the host-side problem and CTA tile shapes, rather than a fundamental limitation of the approach. Support for additional head dimensions is an immediate next step.

## Dependencies

| Dependency | Purpose |
| --- | --- |
| **CUDA C++ / CUDA Toolkit** | Core implementation, device memory management, and GPU kernel execution. |
| **cuBLAS (cuBLASLt)** | GPU matrix multiplication. |
| **NVIDIA CUTLASS / CuTe** | Low-level tensor layouts, data movement, and Tensor Core MMA operations used by the custom attention kernel. |
| **CMake 4.0+** | Build configuration. |
| **GoogleTest** | Numerical correctness tests. |

The project uses **C++20**. CUTLASS is used for its CuTe primitives rather than a prebuilt attention implementation.

## Build and test

With the CUDA Toolkit and CUTLASS headers available to the compiler:

```bash
cmake -S . -B build -DINSTALL_GTEST=ON
cmake --build build -j
```

The project defaults to CMake's **native** CUDA architecture selection unless `CMAKE_CUDA_ARCHITECTURES` is specified explicitly. The `INSTALL_GTEST` option fetches GoogleTest during configuration; an existing GoogleTest installation can also be used.

Run the test executable:

```bash
./build/test/test
```

The attention and RMSNorm tests compare GPU outputs against CPU reference implementations, using combined absolute/relative tolerances. The main executable currently reports that the inference driver has not yet been implemented.

## Roadmap

### Immediate plans

- [ ] Extend the existing attention prefill kernel to additional head dimensions by adjusting the host-side problem and CTA tile configurations.
- [ ] Implement a dedicated **attention decode kernel**.
- [ ] Implement **SIMT** versions of the attention kernels.
- [ ] Implement a **RoPE (rotary positional embedding) kernel**.
- [ ] Finish the **execution graphs** that connect model operations.
- [ ] Implement the **tokenizer**.
- [ ] Complete the **inference driver**.

### Longer-term plans

- [ ] Develop more specialized attention kernels for different head sizes and workloads.
- [ ] Introduce architecture-specific implementations, **starting with NVIDIA Blackwell**, to take advantage of newer GPU capabilities.

## Project layout

```text
InferenceEngine/
├── include/
│   ├── containers/             # Tensor infrastructure
│   ├── cuda_resources/         # CUDA resource and handle management
│   ├── memory/                 # Host and device arenas
│   └── model/
│       ├── decoder_layer/      # Attention and RMSNorm kernels
│       └── ggml/               # GGUF and quantization interfaces
├── src/                        # Core implementations and current driver stub
├── test/                       # GoogleTest numerical correctness tests
├── model_arch.md               # Model tensor-layout notes
└── CMakeLists.txt
```

---

**Follow the development:** [YouTube — @SevyWagner](https://www.youtube.com/@SevyWagner)
License: This project is licensed under the Apache License, Version 2.0. See LICENSE for details.
Copyright (c) 2026 Severin Wagner
