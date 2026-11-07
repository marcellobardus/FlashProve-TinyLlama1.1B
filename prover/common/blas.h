// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// Products of matrices of signed bytes with exact 32-bit sums, computed by cuBLAS, and each host
// thread's cuBLAS handle (the library's context object). Callers must meet cuBLAS's conditions on
// addresses and sizes, listed at gemm_tn(), and the rules at handle_of_this_thread() and blas():
// nothing here checks them. If a cuBLAS call fails, these functions print a message and exit with
// code 1.
//
// Public source: int8 products, handles and workspaces: NVIDIA, cuBLAS documentation
// (https://docs.nvidia.com/cuda/cublas/), "Data Layout", "cublasGemmEx()"
// (CUDA_R_8I operands, a CUDA_R_32I result, CUBLAS_COMPUTE_32I),
// "cublasLtMatmul()" (the requirements of the integer tensor-core kernels),
// "cublasCreate()", "cublasSetStream()", "cublasSetWorkspace()", "Thread Safety",
// "Results Reproducibility".
#pragma once
#include "common/end_name.h"
#include <cublas_v2.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <vector>
namespace blas {
// Default size of a workspace (GPU scratch memory for cuBLAS): 32 MiB, which cuBLAS's
// documentation recommends on the Hopper architecture.
constexpr size_t BLAS_WORKSPACE = (size_t)32 << 20;
struct Setting {
  cudaStream_t stream = nullptr;
  void *work = nullptr;
  size_t bytes = 0;
};
// The arguments of the calling thread's last call of blas().
static inline Setting &setting() {
  static thread_local Setting s;
  return s;
}
// Returns a reference to the calling thread's handle, null if the thread has none yet. Each host
// thread has a handle because, by cuBLAS's documentation, several threads should not share one. A
// thread's first call takes a handle that a finished thread has returned, if there is one; a thread
// returns its handle when it finishes. So blas() gives no two live threads the same handle, and a
// program creates no more handles than it has live threads that have called blas() at any one
// time.
// Handles are reused because a handle owns GPU memory (64 MiB, measured on an H100 with CUDA 13.0's
// library), and the prover program cp_prove starts a thread for each batch of requests
// (Prover::fold_begin() in prover/prover/fold_walk.cuh): thread_local handles that are never
// destroyed would leak 64 MiB a batch.
// No handle is destroyed and the list of returned handles is never deleted, not even at exit:
// cublasDestroy() synchronizes the whole device, other batches' work included, and exit() can run
// while another thread finishes.
// A reused handle keeps its last thread's settings, except the stream and the workspace, which
// blas() sets at every call. Callers of blas() must keep four rules:
//   - Code that changes another setting restores it before its thread finishes, as
//     prover/weights_commitment/worst_case.cuh does with the math mode (cublasSetMathMode()).
//   - A thread's work through its handle is complete on the GPU when the thread finishes: the
//     next thread may launch through the handle at once. (Before it finishes, the thread that
//     Prover::fold_begin() starts calls EngineGpu::let_go() in prover/engine/engine_gpu.cuh, which
//     waits for the engine's stream.)
//   - No copy of a handle is used after the thread that got it from blas() has finished.
//   - cublasDestroy() is never called on a handle from blas(): the next thread would get the
//     destroyed handle.
// (The same pattern: sched_streams() in prover/hiding/product_test.cuh.)
static inline cublasHandle_t &handle_of_this_thread() {
  struct GivenBack {
    std::mutex m;
    std::vector<cublasHandle_t> handles;
  };
  static GivenBack *given_back = new GivenBack;
  struct Held {
    cublasHandle_t handle = nullptr;
    Held() {
      std::lock_guard<std::mutex> l(given_back->m);
      if (!given_back->handles.empty()) handle = given_back->handles.back(), given_back->handles.pop_back();
    }
    ~Held() {
      if (!handle) return;
      std::lock_guard<std::mutex> l(given_back->m);
      given_back->handles.push_back(handle);
    }
  };
  static thread_local Held held;
  return held.handle;
}
// Returns the calling thread's handle, set to `stream` (null: cudaStreamPerThread) and to the
// workspace of `bytes` bytes at `work`: GPU memory aligned to 256 bytes. Concurrent streams need a
// workspace each. It can hold hidden values; zeroing it is its owner's task. cp_prove's forward
// pass and sub-proofs do not zero theirs: a restore does, at the latest the one at the end of a
// batch pass (prover/prover/memory_as_set_up.cuh). A thread's first call may allocate host and GPU
// memory (cublasCreate()): make it early, with nullptr and 0 for the handle alone. At every call
// blas() sets the stream first, then the workspace: cublasSetStream() puts cuBLAS's own pool back.
static inline cublasHandle_t blas(cudaStream_t stream, void *work, size_t bytes = BLAS_WORKSPACE) {
  cublasHandle_t &handle = handle_of_this_thread();
  if (!handle && cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) {
    printf("GPU error: cuBLAS could not start (cublasCreate failed); the program stops\n");
    name_the_end("cublas-start"); // the last line of a serving build, whose sentence above is closed
    exit(1);
  }
  if (cublasSetStream(handle, stream ? stream : cudaStreamPerThread) != CUBLAS_STATUS_SUCCESS ||
      cublasSetWorkspace(handle, work, bytes) != CUBLAS_STATUS_SUCCESS) {
    printf("internal error: cuBLAS refused the stream or the work space that it was given; the program stops\n");
    name_the_end("cublas-start"); // the last line of a serving build, whose sentence above is closed
    exit(1);
  }
  setting() = Setting{stream, work, bytes};
  return handle;
}
// For a function that is called between two uses of the handle by its caller: gives the thread's
// handle another stream and workspace, and the destructor restores the earlier ones (setting()).
struct Borrowed {
  const Setting before;
  const cublasHandle_t handle;
  Borrowed(cudaStream_t stream, void *work, size_t bytes) : before(setting()), handle(blas(stream, work, bytes)) {}
  ~Borrowed() { blas(before.stream, before.work, before.bytes); }
  Borrowed(const Borrowed &) = delete;
  Borrowed &operator=(const Borrowed &) = delete;
};
// Sets C[j*ldc + i] to times * (sum over c < k of A[i*lda + c] * B[j*ldb + c]) for i < m and j < n,
// or adds that value to it if `add` is true. In both operands the summed index varies fastest: in
// cuBLAS's column-major terms this is the TN form, op(A) = A^T, op(B) = B. The result is exact
// while it stays below 2^31 in absolute value: the caller must bound k times the largest product,
// and allow for `times` and `add`. lda, ldb and ldc must fit in an int. A, B and C are GPU memory.
// cuBLAS supports these types only if the addresses A and B, lda and ldb are multiples of 4, and
// its integer tensor-core kernels accept only the TN form, with m and k divisible by 4.
static inline void gemm_tn(cublasHandle_t handle, int m, int n, int k, const int8_t *A, size_t lda, const int8_t *B,
                           size_t ldb, int *C, size_t ldc, bool add, int times = 1) {
  const int beta = add ? 1 : 0;
  const cublasStatus_t status =
      cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, m, n, k, &times, A, CUDA_R_8I, (int)lda, B, CUDA_R_8I, (int)ldb,
                   &beta, C, CUDA_R_32I, (int)ldc, CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT);
  if (status != CUBLAS_STATUS_SUCCESS) {
    // n is not printed: in the routed experts' product (grouped_blocks() in
    // prover/weights_blocks/experts.cuh) it depends on the number of tokens routed to an expert,
    // and so on the weights and the prompts.
    printf("GPU error: the cuBLAS int8 matrix multiply failed with status %d (m %d k %d lda %zu ldb %zu ldc %zu); the "
           "program stops\n",
           (int)status, m, k, lda, ldb, ldc);
    exit(1);
  }
}
} // namespace blas
