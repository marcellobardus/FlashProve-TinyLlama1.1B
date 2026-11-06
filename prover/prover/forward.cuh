// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

#pragma once
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cstdint>
#include <functional>
#include "common/index_guard.cuh"
#include "common/lie.cuh"
// The exact integer forward pass of one layer group (LayerLayout::Lam consecutive layers) of the model
// that docs/model.md defines. With `proof` set, ForwardPass::forward() also writes the committed cells and
// lookup keys of the group's sub-proof. A value outside its range sets a flag in ForwardBuffers::err and
// the computation goes on: the caller must read the flags. But no flag guards a norm's quotient
// (norm_words_fit()), prover_gemm()'s sums or k_t2c()'s input; an index out of range sets the failure
// bit INDEX instead (index_below()). All work runs on the given CUDA stream; forward() waits for it
// once at each MoE layer whose routed tensors are kept as quantized blocks (ForwardPass::wblocks), to copy
// routing tables to the host. The lie:: functions are test hooks with no effect in this repository:
// here lie::at() returns its fourth argument. A limb is a balanced base-256 digit (-128 to
// 127); a byte plane is one limb of each element of a matrix. rq(a, k) is a / 2^k rounded to
// nearest, ties up. A segment is a named list of keys of one lookup call. Call c's keys are u16, at
// index (lam * J_c + j) * T + t: layer lam of the group, column j < J_c = 2^Call::lJ, token t.
#include "common/blas.h"
#include "prover/layout.h"
// Include first: prover/common/field_host.h (i8, u16, i64) and prover/attn/output.cuh (AOutput).
#ifndef CK
#define CK(x)                                                                                                          \
  do {                                                                                                                 \
    cudaError_t e_ = (x);                                                                                              \
    if (e_ != cudaSuccess) {                                                                                           \
      printf("CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__);                                  \
      exit(1);                                                                                                         \
    }                                                                                                                  \
  } while (0)
#endif
// The loop form is public: the grid-stride loop: "CUDA Pro Tip: Write Flexible Kernels with Grid-Stride Loops", a post
// on NVIDIA's developer blog, https://developer.nvidia.com/blog/cuda-pro-tip-write-flexible-kernels-grid-stride-loops/.
#define GRID_LOOP(i, n)                                                                                                     \
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < (size_t)(n); i += (size_t)gridDim.x * blockDim.x)
static inline int grid_blocks(size_t n, int t = 256, int cap = 1 << 16) { // thread blocks for n items
  size_t b = (n + t - 1) / t;
  return (int)(b < 1 ? 1 : b > (size_t)cap ? cap : b);
}
__host__ __device__ inline u64 synth_mix(u64 z) { // one step of SplitMix64
  z += 0x9E3779B97F4A7C15ull;
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
  return z ^ (z >> 31);
}
__host__ __device__ inline int synth_weight(u64 seed, u64 st, u64 i, int bits) { // value i of the generator's stream st
  int v = (int)(synth_mix(seed + (st << 40) + i) & ((1u << bits) - 1)) - (1 << (bits - 1));
  if (bits == 16)
    v = v < -32639 || v > 32639 ? 0 : v; // |W| <= 32,639: a draw outside this range gives 0
  else
    v = v == -(1 << (bits - 1)) ? 0 : v;
  return v;
}
// Returns the width in bits of the synthetic weights of the generator's stream st (2 to 12 or 14):
// 13, or 16 for streams 4 to 12 if the parameter file has the line "cw 14". Streams: 2 q, 3 k, 4 v,
// 5 o, 6 gating, 7 up, 8 down, 9 router, 10 to 12 the routed gating, up and down, 14 the head.
__host__ __device__ inline int synth_weight_bits(u64 st, int cw) {
  return cw == 14 && st >= 4 && st <= 12 ? 16 : 13;
}
enum { E_V = 1, E_Q = 2, E_GI = 4, E_U = 8, E_LR = 16, E_XS = 32 }; // range flags in ForwardBuffers::err
// One layer's scales (docs/model.md, "3 Weights and quantization scales"); kr, kgR, kuR: MoE only.
struct LayerScaleSet {
  int kq, kk, kv, ko, kg, ku, kd, M1, K1, M2, K2, kr, kgR, kuR, rp = 1, sh = 8; // M1: first norm
};
// True if the forward pass supports a norm's exponents M and K for residual-stream values of xl
// limbs. No flag guards a norm's quotient rq(x R, K) (docs/model.md, "9 Range conditions", row 2):
// M and K must keep it within +-32,639. R = round(2^M / sig), sig = isqrt(sum of x^2 + eps), so
// |x| <= sig. If R >= 2, |x R| <= 4/3 2^M. If R = 1, |x R| = |x|, at most 2^(M + 1) and, unless a
// flag is set, at most the stream's bound (below 2^31 with four limbs; 547,599,908,735 with five).
// So K >= M - 14 suffices, but five limbs with K <= 23 need K >= M - 13: |x| can reach 2^(K + 15).
static inline bool norm_words_fit(int M, int K, int xl) {
  return M >= 24 && M <= 47 && K >= 17 && K <= 32 && K >= M - (xl == 5 && K <= 23 ? 13 : 14);
}
// Returns whether the forward pass supports layer l's shifts and norm exponents; prints the first
// that it does not. Of s it reads only prof, NE, xl and the digit forms hqk to hr (see write_digits()).
static inline bool layer_words_fit(const LayerScaleSet &w, int l, const LayerShape &s) {
  auto shift = [&](const char *n, int k, int form) {
    const int hi = form == 0 ? 16 : form == 1 ? 24 : 32;
    if (k >= 1 && k <= hi) return true;
    printf("layer constants: layer %d: shift %s = %d is outside the range for digit form %d, which is 1 to %d\n", l, n,
           k, form, hi);
    return false;
  };
  auto norm = [&](int n, int M, int K) {
    if (norm_words_fit(M, K, s.xl)) return true;
    printf("layer constants: layer %d: M%d and K%d refused; they must satisfy 24 <= M%d <= 47, 17 <= K%d <= 32 and K%d "
           ">= M%d - 14 (M%d - 13 with five limbs and K%d <= 23)\n",
           l, n, n, n, n, n, n, n, n);
    return false;
  };
  if (!shift("kg", w.kg, s.hg) || !shift("ku", w.ku, s.hu) || !shift("kd", w.kd, s.hdn) || !norm(2, w.M2, w.K2))
    return false;
  if (s.prof >= 2) {
    if (!shift("kq", w.kq, s.hqk) || !shift("kk", w.kk, s.hqk) || !shift("kv", w.kv, s.hv) || !shift("ko", w.ko, s.ho))
      return false;
    if (w.kq != w.kk && (w.kq < 16 || w.kk < 16)) {
      printf("layer constants: layer %d: kq and kk must be equal when either of them is below 16\n", l);
      return false;
    }
    if (!norm(1, w.M1, w.K1)) return false;
  }
  return !moe_at_layer(s.prof, l, s.NE) || (shift("kr", w.kr, s.hr) && shift("kgR", w.kgR, s.hg) && shift("kuR", w.kuR, s.hu));
}
// GPU buffers. af, ad, ar, qa: cells [feature][t]. key: see the file's top. KH: see k_epi_gate_up().
struct ForwardBuffers {
  i8 *af, *ad, *ar, *qa, *xb0, *xb1;
  short *Cr, *Sr; // RoPE tables [t][dh / 2]
  u16 *key[MAX_CALLS];
  ResidualInt *xst;
  u64 *Rv, *sigv; // each norm's reciprocal R and root sig (see `k_norm_ledger`), [norm][t]
  u16 *KH;
  u32 *err;
  const short *Ts;
  const int *Sig;
  int *es;                        // each token's expert, per MoE layer: [lamM][t]
  int16_t *q16, *k16, *v16, *o16; // attention: copies as arrays [t][j]
  i8 *xT[2], *mk[4];
  // scratch. mT, hT: planes [t][k] for prover_gemm(); hF: limbs of h = s u, [j][t]; Wl: weights; C: sums
  i8 *mT[2], *hT[4], *hF, *Wl[2];
  int *C[5];
  size_t bytes;
};
// Writes the nl limbs of v, lowest first, to dst[0], dst[stride] and so on.
__device__ __forceinline__ void write_limbs(i64 v, int nl, i8 *dst, size_t stride) {
  for (int l = 0; l < nl; l++) {
    int b = (int)(((v + 128) & 255) - 128);
    dst[l * stride] = (i8)b;
    v = (v - b) >> 8;
  }
}
// Writes the digits of r, the remainder of a division by 2^k: the low segment's key lo and, for the
// high digit h = r >> 16, the high segment's key hi (0 if k <= 16) and cells. Digit form 1: a cell,
// h - 128. Form 2: the two limbs of h - 32,896, st apart. Form 0: only lo; if k > 16, as form 2.
__device__ __forceinline__ void write_digits(int form, int k, i64 r, i8 *cell, size_t st, u16 *lo, u16 *hi) {
  if (k <= 16) {
    *lo = (u16)(r << (16 - k));
    if (form == 1) {
      *cell = (i8)-128;
      *hi = 0;
    } else if (form == 2) {
      write_limbs(-32896, 2, cell, st);
      *hi = 0;
    }
    return;
  }
  *lo = (u16)(r & 65535);
  const i64 h = r >> 16;
  if (form == 1) {
    *cell = (i8)(h - 128);
    *hi = (u16)(h << (24 - k));
  } else {
    write_limbs(h - 32896, 2, cell, st);
    *hi = (u16)(h << (32 - k));
  }
}
#include "prover/forward_by_four.cuh"
// Writes the two byte planes of the synthetic matrix W. W[kk][j] is value base + kk * ncol + j of
// the generator's stream st; its limbs go to index (jofs + j) * nrow + kk of w0 and w1.
__global__ void k_synth_weights(u64 seed, u64 st, int bits, u64 base, int nrow, int ncol, int jofs, i8 *w0, i8 *w1) {
  GRID_LOOP(x, (size_t)nrow * ncol) {
    int kk = x % nrow, j = x / nrow;
    int v = synth_weight(seed, st, base + (u64)kk * ncol + j, bits);
    int b0 = ((v + 128) & 255) - 128;
    size_t o = (size_t)(jofs + j) * nrow + kk;
    w0[o] = (i8)b0;
    w1[o] = (i8)((v - b0) >> 8);
  }
}
__global__ void k_fill_u16(u16 *p, size_t n, u16 v) {
  GRID_LOOP(i, n) p[i] = v;
}
// Transposes int8 [R][Cn] to [Cn][R], by the overload with ldb (the pitch of the result's rows) and
// its kernels k_transpose_i8 and k_transpose_i8_words in prover/prover/forward_pieces.cuh, which also cites the
// public source. The forward pass uses it to turn arrays [feature][t] into the byte planes
// [t][feature] that prover_gemm() takes as its operand A.
// The idea is public: a tiled transpose through padded shared memory:
// M. Harris, "An Efficient Matrix Transpose in CUDA C/C++", NVIDIA Technical Blog, 2013,
// https://developer.nvidia.com/blog/efficient-matrix-transpose-cuda-cc/.
// There the elements are of type float.
static inline void transpose_i8(const i8 *a, i8 *b, int R, int Cn, cudaStream_t s, int ldb);
static inline void transpose_i8(const i8 *a, i8 *b, int R, int Cn, cudaStream_t s) {
  transpose_i8(a, b, R, Cn, s, R);
}
// C[j][t] = sum over kk < K of A[t][kk] B[j][kk], added to the old C[j][t] if acc is set. lda is
// the pitch of A's rows.
// The idea is public: one thread for each entry of the result: Cedric Nugteren, "Tutorial: OpenCL
// SGEMM tuning for Kepler", "Kernel 1: Naive implementation": "We basically transform the two
// outer-loops over M and N ... into two dimensions of parallel threads"
// (https://cnugteren.github.io/tutorial/pages/page3.html). Here the entries are counted in one
// dimension, by a grid-stride loop (GRID_LOOP), so a thread may compute several.
__global__ void k_gemm_naive(const i8 *A, const i8 *B, int *C, int T, int N, int K, int acc, int lda) {
  GRID_LOOP(x, (size_t)T * N) {
    int t = x % T, j = x / T;
    int s = acc ? C[x] : 0;
    for (int q = 0; q < K; q++) s += (int)A[(size_t)t * lda + q] * B[(size_t)j * K + q];
    C[x] = s;
  }
}
// The same product; lda = 0 means K. It goes to cuBLAS unless T < 256 or K is not a multiple of 16
// (this file's own limits). Not checked: h's CUDA stream must be s. No caller's 32-bit sum overflows
// under limit_missed()'s rules on H, G, dh and f (prover/common/parameter_file.h).
// Public source: NVIDIA, cuBLAS documentation (https://docs.nvidia.com/cuda/cublas/),
// "cublasGemmEx()": the compute type CUBLAS_COMPUTE_32I with CUDA_R_8I operands and a CUDA_R_32I
// result. cuBLAS then requires the addresses of A and B, lda and ldb to be multiples of 4; prover_gemm()
// ensures that only for ldb = K and for lda if it is given as 0 (prover/common/blas.h has more).
static inline void prover_gemm(cublasHandle_t h, const i8 *A, const i8 *B, int *C, int T, int N, int K, int acc,
                          cudaStream_t s, int lda = 0) {
  if (!lda) lda = K;
  if (T < 256 || (K & 15)) {
    k_gemm_naive<<<grid_blocks((size_t)T * N), 256, 0, s>>>(A, B, C, T, N, K, acc, lda);
    return;
  }
  int one = 1, be = acc;
  cublasStatus_t e = cublasGemmEx(h, CUBLAS_OP_T, CUBLAS_OP_N, T, N, K, &one, A, CUDA_R_8I, lda, B, CUDA_R_8I, K, &be,
                                  C, CUDA_R_32I, T, CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT);
  if (e != CUBLAS_STATUS_SUCCESS) {
    printf("GPU error: the cuBLAS int8 matrix multiply failed with status %d (T %d N %d K %d); the program stops\n",
           (int)e, T, N, K);
    exit(1);
  }
}
// Finds the largest r with r * r <= v; the two loops correct what sqrt() gives in double precision.
__device__ inline u64 isqrt_narrow(u64 v) {
  u64 r = (u64)sqrt((double)v);
  while (r * r > v) r--;
  while ((r + 1) * (r + 1) <= v) r++;
  return r;
}
// Adds to Wg[s][t] the sum W_s, over the features k and over i + j = s, of b_i b_j, where b are the
// NL limbs of x[k][t]. The sum of squares is V = sum over s of 256^s W_s. Launch: see norm_ledger().
template <int NL>
__global__ void __launch_bounds__(256)
    k_norm_sums(const ResidualInt *x /* [k][t] */, int T, int d, int ks /* gridDim.y = ceil(d / ks) */,
             unsigned long long *Wg /* [2 NL - 1][T], zeroed */) {
  constexpr int NS = 2 * NL - 1;
  __shared__ i64 sh[NS][8][33];
  const int tx = threadIdx.x, ty = threadIdx.y, t = blockIdx.x * 32 + tx, k0 = blockIdx.y * ks, k1 = min(d, k0 + ks);
  i64 W[NS] = {};
  if (t < T)
    for (int k = k0 + ty; k < k1; k += 8) {
      i64 v = x[(size_t)k * T + t];
      int b[NL];
#pragma unroll
      for (int l = 0; l < NL; l++) {
        b[l] = (int)(((v + 128) & 255) - 128);
        v = (v - b[l]) >> 8;
      }
#pragma unroll
      for (int i = 0; i < NL; i++)
#pragma unroll
        for (int j = 0; j < NL; j++) W[i + j] += b[i] * b[j];
    }
  for (int s = 0; s < NS; s++) sh[s][ty][tx] = W[s];
  __syncthreads();
  if (ty || t >= T) return;
  for (int s = 0; s < NS; s++) {
    i64 a = 0;
    for (int y = 0; y < 8; y++) a += sh[s][y][tx];
    atomicAdd(Wg + (size_t)s * T + t, (unsigned long long)a);
  }
}
// For residual-stream values of four limbs. Converts each token's sums Wg to V's bytes and carries
// (V >= 2^59 sets E_V), and writes sig = isqrt(V + eps) to sigv and R = round(2^M / sig) to Rv.
// With `proof` it writes the digit row (`ledger` in identifiers): 74 byte digits, each as a cell
// (digit - 128) and a key (mu * digit, mu below): V; six carries + 2^47; sig; l1 = V + eps - sig^2;
// 2 sig - l1; R = Rlo + 2^24 Rhi; Rhi sig; rr = 2^M + (sig >> 1) - R sig; sig - 1 - rr (in order).
// Public sources: integer-only LayerNorm in I-BERT (Kim et al., 2021, section 3.6) and RMSNorm
// in I-LLM (Hu et al., 2024, Algorithm 4); here the floor root is taken of the sum of squares plus epsilon, on wider
// integers than theirs. (S. Kim, A. Gholami, Z. Yao, M. W. Mahoney, K. Keutzer, "I-BERT: Integer-only BERT
// Quantization", ICML 2021 (PMLR 139), section 3.6 "Integer-only LayerNorm" and Algorithm 4,
// https://arxiv.org/abs/2101.01321 ; X. Hu, Y. Cheng, D. Yang, Z. Yuan, J. Yu, C. Xu, S. Zhou, "I-LLM: Efficient
// Integer-Only Inference for Fully-Quantized Low-Bit Large Language Models", 2024, section 3.4.2 and appendix A.2,
// Algorithm 4 "DI-RMSnorm", https://arxiv.org/abs/2405.17849)
// Public sources: Value i of all threads is written side by side, which is a structure of arrays (NVIDIA, "CUDA
// Performance Optimization", Hot Chips 20 tutorial, 2008, on coalescing); CUB's documentation describes the same for
// the items that a thread holds as the striped arrangement, there with the block's number of threads as the stride,
// here with the number of tokens. (P. LeGresley (NVIDIA), "CUDA Performance Optimization", CUDA tutorial at Hot Chips
// 20, 2008, https://old.hotchips.org/wp-content/uploads/hc_archives/hc20/1_Sun/HC20.24.240.pdf ; NVIDIA Research, CUB
// documentation (v1.8.0), main page, section 5.3 "Flexible data arrangement across threads",
// https://nvlabs.github.io/cub/)
// Public source: division by a fixed-point reciprocal (IntDiv): Z. Li, Q. Gu, "I-ViT: Integer-only
// Quantization for Efficient Vision Transformer Inference", ICCV 2023, section 3.3 "Integer-only
// Softmax: Shiftmax", equation (16), https://arxiv.org/abs/2207.01405 (for Softmax's denominator).
// Public source: the same in RMSNorm: I-LLM, Algorithm 4 (above), with an IntDiv that it does not
// define. Here the reciprocal and the quotient (k_norm_out()) are rounded to nearest, not down.
// The compiler hint is public: __launch_bounds__: NVIDIA, "CUDA C++ Best Practices Guide",
// "Occupancy", https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/.
__global__ void __launch_bounds__(1024)
    k_norm_ledger(const unsigned long long *Wg, int T, int M, i64 eps, i8 *row /* instance's first feature */, u16 *keyRow,
               const int *posRow /* 74 columns */, size_t keyBase, u64 *Rv, u64 *sigv, u32 *err, int proof) {
  // Source for the root-mean-square norm, here in integers: Zhang, Sennrich, "Root Mean Square Layer Normalization",
  // arXiv:1910.07467, section 4, eq. (4).
  constexpr int NORM_ROW_LEN = 74;
  const int t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= T) return;
  i64 W[7];
  for (int s = 0; s < 7; s++) W[s] = lie::at(lie::SUM, s, t, (i64)Wg[(size_t)s * T + t]); // a test hook, on the sum read here
  int dg[NORM_ROW_LEN];
  i64 c = 0;
  u64 V = 0;
  for (int s = 0; s < 7; s++) {
    i64 a = W[s] + c;
    int v = (int)(a & 255);
    dg[s] = v;
    V |= (u64)v << (8 * s);
    c = (a - v) >> 8;
    if (s < 6) {
      u64 cc = (u64)(c + (1ll << 47));
      for (int i = 0; i < 6; i++) dg[8 + 6 * s + i] = (int)((cc >> (8 * i)) & 255);
    }
  }
  if (c < 0 || c >= 8) {
    atomicOr(err, (u32)E_V);
    c = 0;
  }
  dg[7] = (int)c;
  V |= (u64)c << 56;
  u64 sig = lie::at(lie::SIGMA, 0, t, isqrt_narrow(V + (u64)eps)), l1 = V + (u64)eps - sig * sig, h1 = 2 * sig - l1,
      R = lie::at(lie::RECIPROCAL, 0, t, ((1ull << M) + (sig >> 1)) / sig), rr = (1ull << M) + (sig >> 1) - R * sig, rq = sig - 1 - rr,
      Rlo = R & 0xFFFFFF, Rhi = R >> 24, p1 = Rhi * sig;
  Rv[t] = R;
  sigv[t] = sig;
  if (!proof) return;
  dg[44] = (int)((sig & 255) >> 1);
  dg[45] = (int)(sig & 1);
  dg[46] = (int)((sig >> 8) & 255);
  dg[47] = (int)((sig >> 16) & 255);
  dg[48] = (int)(sig >> 24);
  for (int i = 0; i < 4; i++) {
    dg[49 + i] = (int)((l1 >> (8 * i)) & 255);
    dg[53 + i] = (int)((h1 >> (8 * i)) & 255);
    dg[66 + i] = (int)((rr >> (8 * i)) & 255);
    dg[70 + i] = (int)((rq >> (8 * i)) & 255);
  }
  for (int i = 0; i < 3; i++) {
    dg[57 + i] = (int)((Rlo >> (8 * i)) & 255);
    dg[60 + i] = (int)((Rhi >> (8 * i)) & 255);
    dg[63 + i] = (int)((p1 >> (8 * i)) & 255);
  }
  for (int i = 0; i < NORM_ROW_LEN; i++) {
    dg[i] = lie::at(lie::DIGIT, i, t, dg[i]);
    row[(size_t)i * T + t] = (i8)(dg[i] - 128);
    int mu = i == 7 ? 32 : i == 44 ? 2 : i == 45 ? 128 : i == 48 ? 4 : 1;
    keyRow[keyBase + (size_t)posRow[i] * T + t] = (u16)(mu * dg[i]);
  }
}
// Does for v < 2^80 what isqrt_narrow() does: it finds the largest r with r * r <= v.
__device__ inline u64 isqrt_wide(unsigned __int128 v) {
  u64 r = (u64)sqrt((double)(u64)(v >> 64) * 18446744073709551616.0 + (double)(u64)v);
  while ((unsigned __int128)r * r > v) r--;
  while ((unsigned __int128)(r + 1) * (r + 1) <= v) r++;
  return r;
}
// k_norm_ledger() for values of five limbs (the line "xl 5" of the parameter file, `policy.txt`): nine
// sums, V < 2^78 in ten bytes, sig < 2^39, 99 digits. The proof checks the relations of root and
// reciprocal in the field, so neither side may reach the field's size. With sig = a + 2^24 b,
// sig^2 = a^2 + 2^25 a b + 2^48 b^2 is compared with V + eps in two halves (Vlo = V mod 2^48) with
// a carry kappa between; R sig is Rlo a + 2^24 (Rlo b + Rhi a) + 2^48 Rhi b, whose last term is 0.
// Digits: 0-9 V; 10-57 eight carries + 2^47, six bytes each; 58-63 sig (58: bits 1-7, 59: bit 0);
// 64-68 l1; 69-73 2 sig - l1; 74-76 Blo, 77-78 Bhi (a b = Blo + 2^23 Bhi); 79 kappa + 4; 80-82 Rlo;
// 83-85 Rhi; 86-88 Rlo b + Rhi a; 89-93 rr; 94-98 sig - 1 - rr (l1, rr: see k_norm_ledger()).
// Public source: division by a reciprocal: I-ViT (Li and Gu, 2023, Eq. 16), cited at k_norm_ledger().
// Public source: integer LayerNorm: I-BERT (Kim et al., 2021, section 3.6), cited at k_norm_ledger().
// Public source: integer RMSNorm: I-LLM (Hu et al., 2024, Algorithm 4), cited at k_norm_ledger().
// Here the sum of squares is kept exact in 128-bit integers, eps is added under the root and the
// reciprocal is rounded to nearest.
// Public source: unsigned __int128 in device code is a documented feature of nvcc (CUDA C++
// Programming Guide, Host Compiler Extensions) if the host compiler has the type. See also: NVIDIA,
// CUDA Programming Guide, "C/C++ Language Extensions", "Host Compiler Type Extensions",
// https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-extensions.html.
// Public source: a lambda defined inside a kernel: CUDA C++ Programming Guide, Lambda Expressions.
// The guide's examples capture nothing; ours captures a local array by reference.
// See also: NVIDIA, CUDA Programming Guide, "C++ Language Support", "Lambda Expressions",
// https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-support.html.
// Public source: The launch (one thread per token) and the order of the stores (coalesced) follow NVIDIA's general
// guidance (CUDA C++ Best Practices Guide, on coalesced access to global memory). The column of a key is looked up in a
// small table. (NVIDIA, CUDA C++ Best Practices Guide,
// https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html)
// Public sources for the order of the stored values: see k_norm_ledger().
// The compiler hint is public: __launch_bounds__: NVIDIA, "CUDA C++ Best Practices Guide", cited at k_norm_ledger.
__global__ void __launch_bounds__(1024)
    k_norm_ledger_wide(const unsigned long long *Wg, int T, int M, i64 eps, i8 *row /* instance's first feature */, u16 *keyRow,
                const int *posRow /* 99 columns */, size_t keyBase, u64 *Rv, u64 *sigv, u32 *err, int proof) {
  const int t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= T) return;
  // dg holds hidden values: a token's sum of squares, its root and the root's reciprocal. An array
  // that is indexed at run time is placed in local memory, which no caller can wipe
  // (prover/common/local_memory.cuh). With the loops over dg unrolled, every index is a constant, so
  // the compiler can keep dg in registers. The choice is the compiler's. A test that is not part of
  // this repository checks that the built kernel has no stack frame and no other local memory.
  int dg[99];
  auto bytes = [&](int at, int n, u64 v) {
#pragma unroll
    for (int i = 0; i < n; i++) dg[at + i] = (int)((v >> (8 * i)) & 255);
  };
  i64 c = 0;
  unsigned __int128 V = 0;
#pragma unroll
  for (int s = 0; s < 9; s++) {
    const i64 a = lie::at(lie::SUM, s, t, (i64)Wg[(size_t)s * T + t]) + c; // a test hook, on the sum read here
    const int v = (int)(a & 255);
    dg[s] = v;
    V |= (unsigned __int128)v << (8 * s);
    c = (a - v) >> 8;
    if (s < 8) bytes(10 + 6 * s, 6, (u64)(c + (1ll << 47)));
  }
  if (c < 0 || c >= 64) {
    atomicOr(err, (u32)E_V);
    c = 0;
  }
  dg[9] = (int)c;
  V |= (unsigned __int128)c << 72;
  if ((V + (u64)eps) >> 78) atomicOr(err, (u32)E_V); // sig would reach 2^39, which its digits cannot represent
  const u64 sig = lie::at(lie::SIGMA, 0, t, isqrt_wide(V + (u64)eps)),
            l1 = (u64)(V + (u64)eps - (unsigned __int128)sig * sig), h1 = 2 * sig - l1,
            R = lie::at(lie::RECIPROCAL, 0, t, ((1ull << M) + (sig >> 1)) / sig), rr = (1ull << M) + (sig >> 1) - R * sig, rq = sig - 1 - rr,
            Rlo = R & 0xFFFFFF, Rhi = R >> 24, a = sig & 0xFFFFFF, b = sig >> 24, Blo = (a * b) & 0x7FFFFF,
            Bhi = (a * b) >> 23, Vlo = (u64)V & 0xFFFFFFFFFFFFull;
  const i64 kappa = ((i64)(a * a) + (i64)(Blo << 25) + (i64)l1 - (i64)Vlo - eps) >> 48;
  Rv[t] = R;
  sigv[t] = sig;
  if (!proof) return;
  dg[58] = (int)((sig & 255) >> 1);
  dg[59] = (int)(sig & 1);
  bytes(60, 4, sig >> 8);
  bytes(64, 5, l1);
  bytes(69, 5, h1);
  bytes(74, 3, Blo);
  bytes(77, 2, Bhi);
  dg[79] = (int)(kappa + 4);
  bytes(80, 3, Rlo);
  bytes(83, 3, Rhi);
  bytes(86, 3, Rlo * b + Rhi * a);
  bytes(89, 5, rr);
  bytes(94, 5, rq);
#pragma unroll
  for (int i = 0; i < 99; i++) {
    dg[i] = lie::at(lie::DIGIT, i, t, dg[i]);
    row[(size_t)i * T + t] = (i8)(dg[i] - 128);
    const int mu = i == 9 ? 4 : i == 58 ? 2 : i == 59 ? 128 : i == 63 ? 2 : i == 76 ? 2 : i == 79 ? 32 : 1;
    keyRow[keyBase + (size_t)posRow[i] * T + t] = (u16)(mu * dg[i]);
  }
}
// Computes one norm's sums and digit row from the values x[k][t]. Wg: scratch of norm_sum_count(xl) * T u64.
static inline void norm_ledger(int xl, const ResidualInt *x, int T, int d, unsigned long long *Wg, int M, i64 eps, i8 *row,
                              u16 *keyRow, const int *posRow, size_t keyBase, u64 *Rv, u64 *sigv, u32 *err, int proof,
                              cudaStream_t s) {
  const int ks = 64;
  const dim3 blocks((T + 31) / 32, (d + ks - 1) / ks), threads(32, 8);
  CK(cudaMemsetAsync(Wg, 0, (size_t)norm_sum_count(xl) * T * 8, s));
  if (xl == 5) {
    k_norm_sums<5><<<blocks, threads, 0, s>>>(x, T, d, ks, Wg);
    k_norm_ledger_wide<<<(T + 255) / 256, 256, 0, s>>>(Wg, T, M, eps, row, keyRow, posRow, keyBase, Rv, sigv, err, proof);
  } else {
    k_norm_sums<4><<<blocks, threads, 0, s>>>(x, T, d, ks, Wg);
    k_norm_ledger<<<(T + 255) / 256, 256, 0, s>>>(Wg, T, M, eps, row, keyRow, posRow, keyBase, Rv, sigv, err, proof);
  }
}
// Maps an index of a segment to its column among the keys of the segment's lookup call. The segment
// has n <= 8 pieces: s[i] is the first index of piece i, and off[i] is that index's column.
struct Pos2 {
  int n;
  int s[8];
  i64 off[8];
  // The compiler hint is public: #pragma unroll: NVIDIA, "CUDA Programming Guide", "C/C++ Language Extensions",
  // "#pragma unroll", https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-extensions.html.
  __device__ __forceinline__ i64 at(int x) const {
    i64 r = off[0] + x;
    // The result comes from the last piece i < n with s[i] <= x,
    // so s must be increasing, with s[0] = 0.
#pragma unroll
    for (int i = 1; i < 8; i++)
      if (i < n && x >= s[i]) r = off[i] + (x - s[i]);
    return r;
  }
};
// With the line gcl of the parameter file (LayerShape::gcl), the gating projection's kernels also write
// the cells gh and the keys of the segments CG and G0. gh points to the kernel's first feature.
struct GateClamp {
  int on = 0;
  int8_t *gh = nullptr;
  size_t st = 0; // the distance from one limb of gh to the next
  uint16_t *kCG = nullptr, *kG0 = nullptr;
  Pos2 pCG{}, pG0{};
};
static inline bool fits_four(const GateClamp &gc) { // for `by_four` (forward_by_four.cuh)
  return fits_four(gc.gh) && fits_four(gc.kCG) && fits_four(gc.kG0);
}
// Computes the key into the SiLU table Ts from the gating projection's quotient q: q, or with gcl,
// 256 clamp(gh, +-127) + g0 for q = g0 + 256 gh, g0 a limb. Returns false if q is out of range.
__device__ __forceinline__ bool gate_key(long long q, int gcl, int &key, int &gh) {
  if (!gcl) {
    key = (int)q;
    gh = 0;
    return q >= -32768 && q <= 32639;
  }
  const int g0 = (int)((q + 128) & 255) - 128;
  const long long h = (q - g0) >> 8;
  gh = (int)h;
  key = 256 * (h < -127 ? -127 : h > 127 ? 127 : (int)h) + g0;
  return h >= -16384 && h <= 16383;
}
// Writes a norm's output: q = rq(x R, K), clamped, to `out`. If `proof` is set, it also writes the
// unclamped q to yw, the remainder's high digit to rh, and the low, high and clamp segments' keys.
// Public sources for the division by a reciprocal: see k_norm_ledger().
// A thread handles W adjacent tokens of a feature; LAUNCH_BY_FOUR (forward_by_four.cuh) chooses W.
template <int W = 1>
__global__ void k_norm_out(const ResidualInt *x, const u64 *Rv, int T, int d, int K, i8 *yw, i8 *rh, i8 *out,
                            size_t stW /* limb stride of yw, rh */, size_t stO, u16 *kLo, u16 *kHi, u16 *kC, Pos2 pLo,
                            Pos2 pHi, Pos2 pC, size_t bLo, size_t bHi, size_t bC, int proof,
                            int CM /* out is clamped to [-CM, CM] */) {
  GRID_LOOP(z, (size_t)T * d / W) {
    const size_t i = W * z;
    const int t = i % T, k = i / T;
    i64 xs[W], R[W], q[W], o[W], high[W];
    u16 lo[W], hi[W], of_q[W];
    ld_w<W>(x + i, xs), ld_w<W>(Rv + t, R);
#pragma unroll
    for (int e = 0; e < W; e++) {
      // |x R| <= 2^M + |x| / 2, because |x| <= sig and R <= (2^M + sig / 2) / sig, so the product
      // does not overflow (unless E_V is set). No flag guards the quotient's 16 bits: see
      // norm_words_fit(). Test hooks get R as it is read here, not the committed R (`second`, the
      // last argument, is 1), and q once its remainder r is computed (QUOTIENT).
      const i64 a = xs[e] * lie::at(lie::RECIPROCAL, 0, t + e, R[e], 1) + (1ll << (K - 1));
      q[e] = lie::at(lie::ROUNDED, k, t + e, a >> K);
      const i64 r = a - (q[e] << K), h = r >> 16;
      q[e] = lie::at(lie::QUOTIENT, k, t + e, q[e]); // after r: a false q fails rounding's equation
      o[e] = lie::at(lie::CLAMPED, k, t + e, q[e] < -CM ? -CM : q[e] > CM ? CM : q[e]);
      high[e] = h - 32896;
      lo[e] = (u16)(r & 65535), hi[e] = (u16)(h << (32 - K)), of_q[e] = (u16)(q[e] + 32768);
    }
    write_limbs_wide<W>(o, 2, out + i, stO);
    if (!proof) continue;
    write_limbs_wide<W>(q, 2, yw + i, stW);
    write_limbs_wide<W>(high, 2, rh + i, stW);
    st_w<W>(kLo + bLo + (size_t)pLo.at(k) * T + t, lo);
    st_w<W>(kHi + bHi + (size_t)pHi.at(k) * T + t, hi);
    st_w<W>(kC + bC + (size_t)pC.at(k) * T + t, of_q);
  }
}
// Rounds the accumulators Acc = C0 + 256 C1 + 65536 C2 (arrays [j][t]) of the gating projection
// and, uo entries later, of the up projection, and writes h = s u: u is the up projection's
// quotient, s the SiLU table Ts at gate_key()'s key. A quotient out of range sets E_GI or E_U and
// becomes 0. Arrays indexed by feature start at the kernel's first feature, key arrays at the
// layer's column 0. KHs, KHu: in ForwardBuffers::KH, at index (lam * Pf + feature) * T + t, Pf the least
// power of two >= f: s + 2^15 for all layers, then u + 2^15. The idea is public: vLLM's SiluAndMul,
// "An activation function for SwiGLU. The function computes x -> silu(x[:d]) * x[d:]": vLLM's API
// documentation, https://docs.vllm.ai/en/v0.11.2/api/vllm/model_executor/layers/activation/. Here
// the values are integers and SiLU is read from the table Ts. W: see k_norm_out().
template <int W = 1>
__global__ void k_epi_gate_up(const int *C0, const int *C1, const int *C2, int T, int n, int j0, size_t uo, int kg,
                            int ku, int hg, int hu, const short *Ts, i8 *gi, i8 *sc, i8 *uc, i8 *rg, i8 *ru, size_t st2,
                            i8 *hF, size_t stH, u16 *kG, u16 *kU, u16 *kGH, u16 *kUH, u16 *kUT, u16 *kF, Pos2 pG,
                            Pos2 pU, Pos2 pGH, Pos2 pUH, Pos2 pUT, Pos2 pF, u16 *KHs, u16 *KHu, GateClamp gc, u32 *err,
                            int proof) {
  GRID_LOOP(z, (size_t)T * n / W) {
    const size_t i = W * z, iu = i + uo;
    const int t = i % T, j = j0 + (int)(i / T);
    int G0[W], G1[W], G2[W], U0[W], U1[W], U2[W];
    ld_w<W>(C0 + i, G0), ld_w<W>(C1 + i, G1), ld_w<W>(C2 + i, G2);
    ld_w<W>(C0 + iu, U0), ld_w<W>(C1 + iu, U1), ld_w<W>(C2 + iu, U2);
    // Source for the gated feed-forward layer (FFN_SwiGLU), here in integers: Shazeer, "GLU Variants Improve
    // Transformer", arXiv:2002.05202, eq. (6).
    i64 key[W], gh[W], s[W], qu[W], h[W], left_g[W], left_u[W];
    u32 bad = 0;
#pragma unroll
    for (int e = 0; e < W; e++) {
      const i64 ag = lie::at(lie::ACCUMULATOR, j, t + e, (i64)G0[e] + ((i64)G1[e] << 8) + ((i64)G2[e] << 16)) + (1ll << (kg - 1)),
                au = lie::at(lie::ACCUMULATOR, j, t + e, (i64)U0[e] + ((i64)U1[e] << 8) + ((i64)U2[e] << 16), 1) + (1ll << (ku - 1));
      i64 qg = lie::at(lie::ROUNDED, j, t + e, ag >> kg), divided = qg; // divided: the quotient that the remainder belongs to
      qu[e] = lie::at(lie::ROUNDED, j, t + e, au >> ku, 1);
      qg = lie::at(lie::GATE, j, t + e, qg);
      int the_key, high;
      if (!gate_key(qg, gc.on, the_key, high)) {
        bad |= E_GI;
        qg = divided = 0;
        the_key = high = 0;
      }
      if (qu[e] < -32768 || qu[e] > 32639) {
        bad |= E_U;
        qu[e] = 0;
      }
      left_g[e] = ag - (divided << kg), left_u[e] = au - (qu[e] << ku);
      if (gc.on) { // A test hook gets gh (CLAMPED with `second` = 1), and the key follows it.
        auto clamped = [](int part) { return part < -127 ? -127 : part > 127 ? 127 : part; };
        const int other = lie::at(lie::CLAMPED, j, t + e, high, 1);
        if (other != high) the_key += 256 * (clamped(other) - clamped(high)), high = other;
        the_key = lie::at(lie::CLAMPED, j, t + e, the_key); // `second` = 0: only the key
      }
      qu[e] = lie::at(lie::UP, j, t + e, qu[e]); // after its remainder, as in `k_norm_out`
      key[e] = the_key, gh[e] = high, s[e] = lie::at(lie::SILU, j, t + e, (i64)Ts[lie::inside(the_key + 32768, 65536)]); // `lie::inside`: a test hook that returns the index
      h[e] = lie::at(lie::HIDDEN, j, t + e, s[e] * qu[e]);
    }
    if (bad) atomicOr(err, bad);
    write_limbs_wide<W>(h, 4, hF + i, stH);
    if (!proof) continue;
    write_limbs_wide<W>(key, 2, gi + i, st2);
    write_limbs_wide<W>(s, 2, sc + i, st2);
    write_limbs_wide<W>(qu, 2, uc + i, st2);
    write_digits_wide<W>(hg, kg, left_g, rg + i, st2, kG + (size_t)pG.at(j) * T + t, kGH + (size_t)pGH.at(j) * T + t);
    write_digits_wide<W>(hu, ku, left_u, ru + i, st2, kU + (size_t)pU.at(j) * T + t, kUH + (size_t)pUH.at(j) * T + t);
    u16 of_u[W], of_key[W], of_s[W];
#pragma unroll
    for (int e = 0; e < W; e++)
      of_u[e] = (u16)(qu[e] + 32768), of_key[e] = (u16)(key[e] + 32768), of_s[e] = (u16)(s[e] + 32768);
    st_w<W>(kUT + (size_t)pUT.at(j) * T + t, of_u);
    st_w<W>(kF + (size_t)pF.at(j) * T + t, of_key);
    st_w<W>(KHs + i, of_s);
    st_w<W>(KHu + i, of_u);
    if (!gc.on) continue;
    write_limbs_wide<W>(gh, 2, gc.gh + i, gc.st);
#pragma unroll
    for (int e = 0; e < W; e++) of_u[e] = (u16)(gh[e] + 49152), of_key[e] = (u16)(((int)key[e] & 255) ^ 128);
    st_w<W>(gc.kCG + (size_t)gc.pCG.at(j) * T + t, of_u);
    st_w<W>(gc.kG0 + (size_t)gc.pG0.at(j) * T + t, of_key);
  }
}
// Updates the residual stream: xnew = xold + rq(Acc, k), with Acc = sum over s of 256^s P_s. Each
// P_s is an array [j < d][t]; P3 and P4 are read only if nP > 3. A result outside the stream's
// range sets E_XS and becomes 0. W: see k_norm_out().
template <int W = 1>
__global__ void k_epi_residual(const int *P0, const int *P1, const int *P2, const int *P3, const int *P4, int nP, int T,
                          int d, int k, int form, int xl, const ResidualInt *xold, ResidualInt *xnew, i8 *xc /* xl limbs */, size_t stX,
                          i8 *rh, size_t stR, u16 *kLo, u16 *kHi, Pos2 pLo, Pos2 pHi, size_t bLo, size_t bHi, u32 *err,
                          int proof) {
  const i64 least = xl == 5 ? -547599908735ll : -2147483647ll, most = xl == 5 ? 547599908735ll : 2139062143ll;
  GRID_LOOP(z, (size_t)T * d / W) {
    const size_t i = W * z;
    const int t = i % T, j = i / T;
    i64 a[W], old[W], xn[W], left[W];
    {
      int p0[W], p1[W], p2[W];
      ld_w<W>(P0 + i, p0), ld_w<W>(P1 + i, p1), ld_w<W>(P2 + i, p2);
#pragma unroll
      for (int e = 0; e < W; e++) a[e] = (i64)p0[e] + ((i64)p1[e] << 8) + ((i64)p2[e] << 16);
    }
    if (nP > 3) {
      int p3[W], p4[W];
      ld_w<W>(P3 + i, p3), ld_w<W>(P4 + i, p4);
#pragma unroll
      for (int e = 0; e < W; e++) a[e] += ((i64)p3[e] << 24) + ((i64)p4[e] << 32);
    }
    ld_w<W>(xold + i, old);
    bool outside = false;
#pragma unroll
    for (int e = 0; e < W; e++) {
      a[e] = lie::at(lie::ACCUMULATOR, j, t + e, a[e]) + (1ll << (k - 1));
      const i64 q = lie::at(lie::ROUNDED, j, t + e, a[e] >> k);
      xn[e] = lie::at(lie::STREAM, j, t + e, old[e] + q), left[e] = a[e] - (q << k);
      if (xn[e] < least || xn[e] > most) outside = true, xn[e] = 0;
    }
    if (outside) atomicOr(err, (u32)E_XS);
    st_w<W>(xnew + i, xn);
    if (!proof) continue;
    write_limbs_wide<W>(xn, xl, xc + i, stX);
    write_digits_wide<W>(form, k, left, rh + i, stR, kLo + bLo + (size_t)pLo.at(j) * T + t, kHi + bHi + (size_t)pHi.at(j) * T + t);
  }
}
#include "prover/forward_pieces.cuh"
// Applies RoPE to q's and k's accumulators (rows j < nqk; Acc as in k_epi_gate_up()), an even row and
// the next at a time, then rounds once: rq(., kq) for q's rows, rq(., kk) for k's. Cr and Sr
// contain cosines and sines times 4096 (4096 and 0 if rope is 0). A quotient outside
// [-32768, 32639] sets E_Q and becomes 0. Rows j < dq go to q, the rest to kx, clamped to +-2047.
// Public source for a layer without RoPE: see k_ropeF() in base.cuh. W: see k_norm_out().
template <int W = 1>
__global__ void k_epi_qk(const int *C0, const int *C1, const int *C2, int T, int nqk, int dq, int dh, int kq, int kk,
                           int form, const short *Cr, const short *Sr, int rope, i8 *qkw, i8 *rqk, size_t stW, i8 *q,
                           size_t stQ, i8 *kx, size_t stK, u16 *kLo, u16 *kHi, u16 *kC, Pos2 pLo, Pos2 pHi, Pos2 pC,
                           size_t bLo, size_t bHi, size_t bC, u32 *err, int proof) {
  GRID_LOOP(u, (size_t)T * (nqk / 2) / W) {
    const size_t z = W * u;
    const int t = z % T, pr = z / T, j0 = 2 * pr, i = (j0 % dh) / 2;
    const size_t a0 = (size_t)j0 * T + t, a1 = a0 + T;
    int x0[W], x1[W], x2[W], y0[W], y1[W], y2[W];
    ld_w<W>(C0 + a0, x0), ld_w<W>(C1 + a0, x1), ld_w<W>(C2 + a0, x2);
    ld_w<W>(C0 + a1, y0), ld_w<W>(C1 + a1, y1), ld_w<W>(C2 + a1, y2);
    i64 R[2][W];
#pragma unroll
    for (int e = 0; e < W; e++) {
      const i64 A0 = lie::at(lie::ACCUMULATOR, j0, t + e, (i64)x0[e] + ((i64)x1[e] << 8) + ((i64)x2[e] << 16)),
                A1 = lie::at(lie::ACCUMULATOR, j0 + 1, t + e, (i64)y0[e] + ((i64)y1[e] << 8) + ((i64)y2[e] << 16));
      const i64 c = rope ? Cr[(size_t)(t + e) * (dh / 2) + i] : 4096,
                sn = rope ? Sr[(size_t)(t + e) * (dh / 2) + i] : 0;
      R[0][e] = lie::at(lie::ROTATED, j0, t + e, c * A0 - sn * A1), R[1][e] = lie::at(lie::ROTATED, j0 + 1, t + e, sn * A0 + c * A1);
    }
    const int k = j0 < dq ? kq : kk;
#pragma unroll
    for (int row = 0; row < 2; row++) {
      const int j = j0 + row;
      const size_t at = (size_t)j * T + t;
      i64 w[W], y[W], left[W];
      u16 of_w[W];
      bool outside = false;
#pragma unroll
      for (int e = 0; e < W; e++) {
        const i64 a = R[row][e] + (1ll << (k - 1));
        w[e] = lie::at(lie::ROUNDED, j, t + e, a >> k);
        if (w[e] < -32768 || w[e] > 32639) outside = true, w[e] = 0;
        left[e] = a - (w[e] << k);
        w[e] = lie::at(lie::QUOTIENT, j, t + e, w[e]); // after its remainder, as in `k_norm_out`
        y[e] = lie::at(lie::CLAMPED, j, t + e, w[e] < -2047 ? -2047 : w[e] > 2047 ? 2047 : w[e]);
        of_w[e] = (u16)(w[e] + 32768);
      }
      if (outside) atomicOr(err, (u32)E_Q);
      if (j < dq)
        write_limbs_wide<W>(y, 2, q + at, stQ);
      else
        write_limbs_wide<W>(y, 2, kx + (size_t)(j - dq) * T + t, stK);
      if (!proof) continue;
      write_limbs_wide<W>(w, 2, qkw + at, stW);
      write_digits_wide<W>(form, k, left, rqk + at, stW, kLo + bLo + (size_t)pLo.at(j) * T + t,
                kHi + bHi + (size_t)pHi.at(j) * T + t);
      st_w<W>(kC + bC + (size_t)pC.at(j) * T + t, of_w);
    }
  }
}
// Writes v = rq(Acc, k), clamped to [-CM, CM]; Acc as in k_epi_gate_up(), W as in k_norm_out(). A
// quotient outside [-32768, 32639] sets E_Q and becomes 0. In this kernel's calls of lie::at(), the
// last argument (`second`) is 1: it marks a row of v, not of q or k.
template <int W = 1>
__global__ void k_epi_v(const int *C0, const int *C1, const int *C2, int T, int n, int k, int CM, int form, i8 *vw,
                          size_t stW, i8 *rv, i8 *v, size_t stV, u16 *kLo, u16 *kHi, u16 *kC, Pos2 pLo, Pos2 pHi,
                          Pos2 pC, size_t bLo, size_t bHi, size_t bC, u32 *err, int proof) {
  GRID_LOOP(z, (size_t)T * n / W) {
    const size_t i = W * z;
    const int t = i % T, j = i / T;
    int x0[W], x1[W], x2[W];
    ld_w<W>(C0 + i, x0), ld_w<W>(C1 + i, x1), ld_w<W>(C2 + i, x2);
    i64 w[W], y[W], left[W];
    u16 of_w[W];
    bool outside = false;
#pragma unroll
    for (int e = 0; e < W; e++) {
      const i64 a = lie::at(lie::ACCUMULATOR, j, t + e, (i64)x0[e] + ((i64)x1[e] << 8) + ((i64)x2[e] << 16), 1) + (1ll << (k - 1));
      w[e] = lie::at(lie::ROUNDED, j, t + e, a >> k, 1);
      if (w[e] < -32768 || w[e] > 32639) outside = true, w[e] = 0;
      left[e] = a - (w[e] << k);
      w[e] = lie::at(lie::QUOTIENT, j, t + e, w[e], 1); // after its remainder, as in `k_norm_out`
      y[e] = lie::at(lie::CLAMPED, j, t + e, w[e] < -CM ? -CM : w[e] > CM ? CM : w[e], 1);
      of_w[e] = (u16)(w[e] + 32768);
    }
    if (outside) atomicOr(err, (u32)E_Q);
    write_limbs_wide<W>(y, 2, v + i, stV);
    if (!proof) continue;
    write_limbs_wide<W>(w, 2, vw + i, stW);
    write_digits_wide<W>(form, k, left, rv + i, stW, kLo + bLo + (size_t)pLo.at(j) * T + t, kHi + bHi + (size_t)pHi.at(j) * T + t);
    st_w<W>(kC + bC + (size_t)pC.at(j) * T + t, of_w);
  }
}
// With the line rcl (LayerShape::rcl), k_epi_router() also writes the cells ly, l3 and the keys of CLR, L3.
struct RouterClamp {
  int on = 0;
  i8 *ly = nullptr, *l3 = nullptr;
  size_t st = 0; // the distance from one limb of ly to the next
  u16 *kC = nullptr, *k3 = nullptr;
  i64 pC = 0, p3 = 0;
  size_t bC = 0, b3 = 0;
};
// The router's logits lrv[e][t], from w = rq(Acc, k): w, or with rc.on, 8 clamp(w >> 3, +-2047) +
// (w & 7). A w outside [-16384, 16383], with rc.on [-262144, 261119], sets E_LR and becomes 0.
__global__ void k_epi_router(const int *C0, const int *C1, const int *C2, int T, int NE, int k, int form, int *lrv,
                           i8 *lr, size_t st, i8 *rl, u16 *kLo, u16 *kHi, u16 *kT, i64 pLo, i64 pHi, i64 pT, size_t bLo,
                           size_t bHi, size_t bT, RouterClamp rc, u32 *err, int proof) {
  GRID_LOOP(i, (size_t)T * NE) {
    i64 a = lie::at(lie::ACCUMULATOR, (long long)(i / T), (long long)(i % T), (i64)C0[i] + ((i64)C1[i] << 8) + ((i64)C2[i] << 16)) +
            (1ll << (k - 1)),
        w = lie::at(lie::ROUNDED, (long long)(i / T), (long long)(i % T), a >> k);
    if (rc.on ? (w >> 3) < -32768 || (w >> 3) > 32639 : w < -16384 || w > 16383) {
      atomicOr(err, (u32)E_LR);
      w = 0;
    }
    const i64 left = a - (w << k);
    w = lie::at(lie::LOGIT, (long long)(i / T), (long long)(i % T), w); // after its remainder
    const i64 y = w >> 3, w3 = w & 7, q = lie::at(lie::CLAMPED, (long long)(i / T), (long long)(i % T), rc.on ? 8 * (y < -2047 ? -2047 : y > 2047 ? 2047 : y) + w3 : w);
    lrv[i] = (int)q;
    if (!proof) continue;
    write_limbs(q, 2, lr + i, st);
    write_digits(form, k, left, rl + i, st, kLo + bLo + (size_t)pLo * T + i, kHi + bHi + (size_t)pHi * T + i);
    kT[bT + (size_t)pT * T + i] = (u16)(2 * (q + 16384));
    if (!rc.on) continue;
    write_limbs(y, 2, rc.ly + i, rc.st);
    rc.l3[i] = (i8)w3;
    rc.kC[rc.bC + (size_t)rc.pC * T + i] = (u16)(y + 32768);
    rc.k3[rc.b3 + (size_t)rc.p3 * T + i] = (u16)(32 * w3);
  }
}
// Top-1 MoE routing: for each token, picks the expert with the largest logit (lowest index
// on ties), and writes its index to es and its score Sig[logit + 32768] to scv.
// Source for proving a maximum by non-negative differences to all candidates (the keys kDD): Liu, Xie, Zhang, "zkCNN:
// Zero Knowledge Proofs for Convolutional Neural Network Predictions and Accuracy", Cryptology ePrint Archive
// 2021/673, section 4.3. There by bits and a product that must be 0; here the differences are looked up, a one-hot
// vector (sel) selects the maximum, and a term in the key breaks ties. The condition on the differences was not
// checked against zkCNN's paper.
__global__ void k_route(const int *lrv /* [e][t] */, int T, int NE, const int *Sig, int *es, int *scv, i8 *sel,
                          i8 *sc, size_t stS, u16 *kSel, u16 *kDD, u16 *kMR, i64 pSel, i64 pDD, i64 pMR, size_t bSel,
                          size_t bDD, size_t bMR, int mr_offset /* 2^15; 2^14 with the line gcl */, int proof) {
  GRID_LOOP(t, T) {
    int be = 0, mr = lrv[t];
    for (int e = 1; e < NE; e++) {
      int v = lrv[(size_t)e * T + t];
      if (v > mr) {
        mr = v;
        be = e;
      }
    }
    es[t] = be;
    {
      const int named = lie::at(lie::EXPERT, 0, t, be); // a test hook, to name another expert
      if (named != be && named >= 0 && named < NE) be = named, mr = lrv[(size_t)be * T + t], es[t] = be;
    }
    if (lie::wants_the_second(lie::EXPERT, 0, t)) { // test hook, false here: no runner-up is chosen
      const int first = be;
      be = first ? 0 : 1, mr = lrv[(size_t)be * T + t];
      for (int e = be + 1; e < NE; e++) {
        const int v = lrv[(size_t)e * T + t];
        if (e != first && v > mr) mr = v, be = e;
      }
      lie::chosen(lie::EXPERT, 0, t, first, be);
      es[t] = be;
    }
    const int row = lie::at(lie::ROW, 0, t, mr); // a test hook, on the row of score and key
    int s = lie::at(lie::SCORE, 0, t, Sig[index_below((long long)row + 32768, 65536, INDEX_ROUTER_LOGIT)]);
    scv[t] = s;
    if (!proof) continue;
    write_limbs(s, 3, sc + t, stS);
    kMR[bMR + (size_t)pMR * T + t] = (u16)(row + mr_offset);
    for (int e = 0; e < NE; e++) {
      const int one = lie::at(lie::SELECTOR, e, t, (int)(e == be)); // a test hook on cell and key
      sel[(size_t)e * T + t] = (i8)one;
      kSel[bSel + ((size_t)pSel + e) * T + t] = (u16)(128 * one);
      kDD[bDD + ((size_t)pDD + e) * T + t] = (u16)(mr - lrv[(size_t)e * T + t] - (be > e ? 1 : 0));
    }
  }
}
__global__ void k_xr(const i8 *m0, const i8 *m1 /* cells [k][t] */, const int *scv, int T, int d, i8 *xr, size_t st,
                       u16 *kX, Pos2 pX, size_t bX, int proof) { // xr = rq(scv[t] m, 16); keys of segment XR
  GRID_LOOP(i, (size_t)T * d) {
    int t = i % T, k = i / T;
    // `second` = 1: a test hook gets the score read here, at every feature, not the committed one
    i64 a = (i64)lie::at(lie::SCORE, 0, t, scv[t], 1) * ((int)m0[i] + 256 * (int)m1[i]) + 32768, q = lie::at(lie::SCALED, k, t, a >> 16);
    write_limbs(q, 2, xr + i, st);
    if (proof) kX[bX + (size_t)pX.at(k) * T + t] = (u16)(a & 65535);
  }
}
// dst[t][x] = es[t] == e ? src[t * lds + x] : 0. K, lds and both addresses must be multiples of 16.
// The idea is public: vectorized memory access, for the load and the store: a post on NVIDIA's
// developer blog, "CUDA Pro Tip: Increase Performance with Vectorized Memory Access",
// https://developer.nvidia.com/blog/cuda-pro-tip-increase-performance-with-vectorized-memory-access/.
__global__ void k_expert_mask(const i8 *src, int lds, i8 *dst, int T, int K, const int *es, int e) {
  GRID_LOOP(i, (size_t)T * (K / 16)) {
    int t = i / (K / 16), c = i % (K / 16);
    uint4 v = make_uint4(0, 0, 0, 0);
    if (es[t] == e) v = *(const uint4 *)(src + (size_t)t * lds + c * 16);
    *(uint4 *)(dst + (size_t)t * K + c * 16) = v;
  }
}
#include "prover/forward_ffn.cuh"
#include "prover/forward_moe.cuh"
#include "weights_blocks/blocks.h"
#include "weights_blocks/experts.cuh"
// Converts two-limb cells [j][t] (the high limb's plane starts st bytes after the low limb's) to
// int16 [t][j]. It needs blocks of 256 threads. Each pass of a block's loop handles 32 tokens by 32
// features (one tile): the threads run along t, the contiguous index of c, to read and along j, the
// contiguous index of o, to write.
// The idea is public: the tiled transposition through a padded tile in shared memory is the method of
// M. Harris, "An Efficient Matrix Transpose in CUDA C/C++" (NVIDIA Technical Blog, 2013; cited at transpose_i8());
// there the elements are floats and nothing is computed on the way. The loop over tiles is the
// grid-stride loop of the post cited at GRID_LOOP, a block for a thread and a tile for an element.
__global__ void k_c2t(const i8 *c, size_t st, int T, int n, int16_t *o) {
  __shared__ int16_t tile[32][33];
  const int x = threadIdx.x & 31, y0 = threadIdx.x >> 5, across = (n + 31) / 32;
  const size_t tiles = (size_t)across * ((T + 31) / 32);
  for (size_t k = blockIdx.x; k < tiles; k += gridDim.x) {
    const int j0 = (int)(k % across) * 32, t0 = (int)(k / across) * 32;
    for (int y = y0; y < 32; y += 8) {
      if (j0 + y >= n || t0 + x >= T) continue;
      const size_t a = (size_t)(j0 + y) * T + t0 + x;
      tile[y][x] = (int16_t)((int)c[a] + 256 * (int)c[st + a]);
    }
    __syncthreads();
    for (int y = y0; y < 32; y += 8)
      if (t0 + y < T && j0 + x < n) o[(size_t)(t0 + y) * n + j0 + x] = tile[x][y];
    __syncthreads();
  }
}
// Converts int16 [t][j] to two-limb cells [j][t] and to the limbs' planes p0, p1 [t][j]. It needs
// blocks of 256 threads and goes by tiles, as k_c2t() does: it reads a tile and writes the planes
// along j, the cells along t. The idea is public: see k_c2t(). Not checked: s[i] <= 32,639.
__global__ void k_t2c(const int16_t *s, int T, int n, i8 *c, size_t st, i8 *p0, i8 *p1) {
  __shared__ int16_t tile[32][33];
  const int x = threadIdx.x & 31, y0 = threadIdx.x >> 5, across = (n + 31) / 32;
  const size_t tiles = (size_t)across * ((T + 31) / 32);
  for (size_t k = blockIdx.x; k < tiles; k += gridDim.x) {
    const int j0 = (int)(k % across) * 32, t0 = (int)(k / across) * 32;
    for (int y = y0; y < 32; y += 8) {
      if (t0 + y >= T || j0 + x >= n) continue;
      const size_t i = (size_t)(t0 + y) * n + j0 + x;
      const int v = s[i], b0 = ((v + 128) & 255) - 128;
      p0[i] = (i8)b0;
      p1[i] = (i8)((v - b0) >> 8);
      tile[y][x] = (int16_t)v;
    }
    __syncthreads();
    for (int y = y0; y < 32; y += 8) {
      if (j0 + y >= n || t0 + x >= T) continue;
      const int v = tile[x][y], b0 = ((v + 128) & 255) - 128;
      const size_t a = (size_t)(j0 + y) * T + t0 + x;
      c[a] = (i8)b0;
      c[st + a] = (i8)((v - b0) >> 8);
    }
    __syncthreads();
  }
}
// Allocates the buffers that are not needed once ForwardPass::forward() has returned. A program may point
// it at 256-byte aligned memory with other uses between passes. alloc_scratch() zeroes the buffers.
static cudaError_t (*forward_scratch_alloc)(void **, size_t) = [](void **p, size_t n) {
  return cudaMalloc(p, n);
};
// Allocates the keys, KH and the residual stream, which a pass writes for the group's sub-proof and
// which are not needed once it is made. ForwardPass::alloc() zeroes them. A program that reuses the memory
// must set forward_pass_buffers_last to false and, outside the keys, restore the zeros before a pass.
static cudaError_t (*forward_pass_alloc)(void **, size_t) = [](void **p, size_t n) {
  return cudaMalloc(p, n);
};
// True if the memory from forward_pass_alloc keeps its contents between two passes. Set it to false
// if other code writes to that memory in between: every proof pass then rewrites the padding keys.
// Public source: see ForwardPass::forward().
static bool forward_pass_buffers_last = true;
struct ForwardPass {
  AOutput *fa = nullptr; // computes a layer's attention output; must be set unless attention (below) is set
  // If set, it is called instead of fa->run(), with the model's layer number as its first argument.
  std::function<void(int l, const int16_t *q, const int16_t *k, const int16_t *v, int sh, int16_t *o, cudaStream_t s)>
      attention;
  // If set, it is called at a MoE layer whose routed tensors are kept as blocks, before they are
  // read. es[t] (device memory) is token t's expert, valid once the work queued on s has run.
  std::function<void(int l, const int *es, cudaStream_t s)> experts_wanted;
  // Optional weights that stay in GPU memory. wres(cls, l) returns device pointers to the two byte
  // planes [row][in] of class cls at layer l: 0, rows q | k | v (in = d); 1, o (d rows, in = dq);
  // 2, rows gating | up (in = d); 3, down (d rows, in = f). Classes 2 and 3: at dense layers only.
  struct WP {
    const i8 *w0 = nullptr, *w1 = nullptr;
  };
  std::function<WP(int cls, int l)> wres;
  // Optional, in place of the generator. wplanes writes to w0 and w1 the byte planes [row][in] of
  // layer l's tensor with stream number st (for a routed tensor, of its expert e). wblocks, if set,
  // returns a routed tensor (st 10 .. 12) as quantized blocks instead: it is "kept as blocks".
  std::function<void(int st, int l, int e, i8 *w0, i8 *w1, cudaStream_t s)> wplanes;
  std::function<const wb::Blocks *(int st, int l)> wblocks;
  // Optional callbacks, with wblocks, for a source that loads routed tensors layer by layer or one
  // by one. They are called while kernels are queued: routed_wanted(l) before any that reads layer
  // l's routed gating or up tensor, routed_down_wanted(l) after wblocks(12, l) and before any that
  // reads the down tensor, routed_read(l) after the last that reads any of the three, whichever
  // experts the tokens chose. With only routed_wanted set, all three must be ready when it returns.
  std::function<void(int l)> routed_wanted, routed_down_wanted, routed_read;
  WP cur;
  LayerLayout L;
  std::vector<LayerScaleSet> kl; // one entry per model layer, index l = g * Lam + lam
  u64 seed;             // seed of the weight generator
  ForwardBuffers b;
  // A pass sets its range flags in the 32-bit device word b.err. By default that is alloc()'s word,
  // and every pass first zeroes it. After raise_flags_in(word) it is the caller's word, which the
  // caller must zero. A ForwardPass is copied by value (layer_setup.cuh): the call changes no earlier copy.
  bool zeroes_its_flags = true;
  void raise_flags_in(u32 *word) { b.err = word, zeroes_its_flags = false; }
  cublasHandle_t h;
  // cuBLAS's workspace for forward()'s products, from alloc_scratch(). It can hold hidden values,
  // so the code that zeroes forward()'s other scratch must zero it too (prover/common/blas.h).
  char *blas_room = nullptr;
  int *posRow, *lrv, *scv;
  unsigned long long *Wg = nullptr; // scratch: one norm's sums W_s (see `k_norm_sums`), [s][t]
  std::vector<size_t> keyN;
  // Returns segment sg's Pos2 (s and off of each of its at most eight pieces); empty if absent.
  Pos2 pos2(const char *sg) const {
    int id = L.segid(sg);
    if (id < 0) return Pos2{};
    const Seg &S = L.seg[id];
    const Call &c = L.call[S.call];
    if (S.pc.size() > 8) {
      printf("internal error: segment %s is split into more than eight pieces, but the layout allows at most eight\n",
             sg);
      exit(1);
    }
    Pos2 p{};
    p.n = (int)S.pc.size();
    for (int i = 0; i < p.n; i++) {
      p.s[i] = (int)c.pc[S.pc[i]].s;
      p.off[i] = c.pc[S.pc[i]].off;
    }
    return p;
  }
  int callOf(const char *sg) const {
    int id = L.segid(sg);
    return id < 0 ? 0 : L.seg[id].call;
  }
  // Returns a pointer to layer lam's column 0 in the key buffer of sg's call; null if sg is absent.
  u16 *kp(const char *sg, int lam) const {
    int id = L.segid(sg);
    if (id < 0) return nullptr;
    int c = L.seg[id].call;
    return b.key[c] + kbase(c, lam);
  }
  size_t kbase(int c, int lam) const { return ((size_t)lam << L.call[c].lJ) * L.T; }
  i8 *cell(TenId X, int l, i64 i) const {
    const Ten &t = L.t[X];
    if (!t.on) return nullptr;
    i8 *m = t.mat == 0 ? b.af : t.mat == 1 ? b.ad : b.ar;
    return m + (size_t)limb_feature(t, l, i, 0) * L.T;
  }
  size_t lst(TenId X) const { return (size_t)L.t[X].cnt * L.t[X].J * L.T; }
  // Returns layer lam's GateClamp for a kernel that starts o cells after the layer's first feature.
  GateClamp gate_clamp(int lam, size_t o, int proof) const {
    GateClamp gc;
    gc.on = L.s.gcl;
    if (!gc.on || !proof) return gc;
    gc.gh = cell(X_gh, 0, lam) + o;
    gc.st = lst(X_gh);
    gc.kCG = kp("CG", lam);
    gc.kG0 = kp("G0", lam);
    gc.pCG = pos2("CG");
    gc.pG0 = pos2("G0");
    return gc;
  }
  // Allocates the buffers, zeroed. With scratch_too = false the caller must call alloc_scratch().
  void alloc(bool scratch_too = true) {
    const size_t T = L.T, d = L.s.d, f = L.s.f;
    b.bytes = 0;
    auto al = [&](auto *&p, size_t n) {
      CK(cudaMalloc(&p, n * sizeof(*p) + 256));
      CK(cudaMemset(p, 0, n * sizeof(*p) + 256));
      b.bytes += n * sizeof(*p);
    };
    auto pass = [&](auto *&p, size_t n) {
      CK(forward_pass_alloc((void **)&p, n * sizeof(*p) + 256));
      CK(cudaMemset(p, 0, n * sizeof(*p) + 256));
      b.bytes += n * sizeof(*p);
    };
    al(b.af, (size_t)L.G[0] * T);
    al(b.ad, (size_t)L.G[1] * T);
    al(b.ar, (size_t)L.G[2] * T);
    al(b.qa, (size_t)std::max(L.Gqa, 1ll) * T);
    al(b.Cr, T * (size_t)(L.s.dh / 2));
    al(b.Sr, T * (size_t)(L.s.dh / 2));
    al(b.xb0, L.s.xl * d * T);
    al(b.xb1, L.s.xl * d * T);
    keyN.clear();
    for (size_t c = 0; c < L.call.size(); c++) {
      size_t n = ((size_t)L.Lam << L.call[c].lJ) * T;
      keyN.push_back(n);
      pass(b.key[c], n);
    }
    pass(b.xst, (size_t)(L.NW * L.Lam + 1) * d * T);
    al(b.Rv, (size_t)L.NN * L.Lam * T);
    al(b.sigv, (size_t)L.NN * L.Lam * T);
    pass(b.KH, 2 * (size_t)L.Lam * L.Pf * T);
    al(b.err, 1);
    for (int i = 0; i < 2; i++)
      al(b.Wl[i], std::max({(size_t)2 * f, (size_t)(L.nqk + L.dkv), (size_t)L.s.NE}) * std::max(d, (size_t)L.dq));
    al(b.es, (size_t)std::max(L.LamM, 1) * T);
    std::vector<int> pr(norm_row_len(L.s.xl) * L.NN);
    int id = L.segid("ROW");
    for (size_t x = 0; x < pr.size(); x++) pr[x] = (int)L.pos(id, x);
    CK(cudaMalloc(&posRow, pr.size() * 4));
    CK(cudaMemcpy(posRow, pr.data(), pr.size() * 4, cudaMemcpyHostToDevice));
    if (scratch_too) alloc_scratch();
  }
  // Allocates, zeroed, what only forward() uses. Call alloc() and set wblocks, if used, before it.
  void alloc_scratch() {
    const size_t T = L.T, d = L.s.d, f = L.s.f;
    auto scratch = [&](auto *&p, size_t n) {
      CK(forward_scratch_alloc((void **)&p, n * sizeof(*p) + 256));
      CK(cudaMemset(p, 0, n * sizeof(*p) + 256));
      b.bytes += n * sizeof(*p);
    };
    if (L.s.att) {
      scratch(b.q16, T * (size_t)L.dq);
      scratch(b.k16, T * (size_t)L.dkv);
      scratch(b.v16, T * (size_t)L.dkv);
      scratch(b.o16, T * (size_t)L.dq);
    }
    for (int i = 0; i < 2; i++) {
      scratch(b.mT[i], T * d);
      scratch(b.xT[i], T * d);
    }
    for (int i = 0; i < 4; i++) scratch(b.mk[i], T * std::max(d, f / 2));
    scratch(lrv, (size_t)L.s.NE * T);
    scratch(scv, T);
    scratch(Wg, (size_t)NORM_SUM_COUNT_MOST * T);
    for (int i = 0; i < 4; i++) scratch(b.hT[i], T * f);
    scratch(b.hF, 4 * T * f);
    for (int i = 0; i < 5; i++)
      scratch(b.C[i], std::max({(size_t)2 * f, (size_t)(L.nqk + L.dkv), d, (size_t)L.s.NE}) *
                          T); // at small shapes q, k and v together may have more rows than gate and up
    if (L.s.prof >= 3 && wblocks) { // only for routed tensors kept as blocks (ffn_moe)
      const size_t S = T + 64 * (size_t)L.s.NE, fe = L.s.fe;
      scratch(moeI, MoeTables::ints((int)T, L.s.NE));
      for (int i = 0; i < 2; i++) scratch(xQ[i], S * d);
      for (int i = 0; i < 4; i++) scratch(hQ[i], S * fe);
      scratch(accG, fe * T);
      scratch(accU, fe * T);
      scratch(tq, fe * T);
      scratch(t1, fe * T);
      scratch(t2, fe * T);
      if (L.s.gcl) scratch(th, fe * T);
      scratch(accM, d * T);
    }
    scratch(blas_room, blas::BLAS_WORKSPACE);
    // Gets the calling host thread's cuBLAS handle at set-up, since creating one allocates memory.
    // Do not set or destroy h. Call forward() on the same host thread; on another it may allocate.
    h = blas::blas(cudaStreamPerThread, blas_room);
    if (L.s.prof >= 3 && wblocks) {
      // Warm-up on zeroed buffers: every size of cuBLAS call for the routed products is made once,
      // so that neither the loaded kernels nor free GPU memory depends on a request: an aim, not
      // checked here (grouped_blocks_warm_up() in prover/weights_blocks/experts.cuh cites NVIDIA's
      // section on lazy loading). With EXPERTS_IN_ONE_LAUNCH at 1 a request makes these calls only for
      // routed tensors of a wider type (Q8_0, F16, BF16), never for Q4_K or Q5_K.
      lend_experts_work();
      grouped_blocks_warm_up<2>(L.s.fe, (int)d, (int)T, experts_work);
      grouped_blocks_warm_up<4>((int)d, L.s.fe, (int)T, experts_work);
      CK(cudaStreamSynchronize(cudaStreamPerThread));
    }
  }
  // Gives the routed products b.Wl and b.C[0] as work buffers: nothing else uses them between the
  // shared expert's k_epi_gate_up and the writing of its down weights. pk_grouped_blocks() uses none.
  void lend_experts_work() {
    const size_t dense = std::max((size_t)2 * L.s.f, (size_t)(L.nqk + L.dkv));
    ExpertsWork &work = experts_work;
    work.blas = h, work.planes = b.Wl[0], work.rows = b.Wl[1], work.sums = b.C[0];
    work.planes_bytes = work.rows_bytes = dense * std::max((size_t)L.s.d, (size_t)L.dq);
    work.sums_ints = std::max(dense, (size_t)L.s.d) * L.T;
  }
  // Fills every call's key buffer with its padding key t0, and KH with 2^15 if
  // Pf != f. forward() calls it before the real keys are written over them.
  void pad(cudaStream_t s) {
    for (size_t c = 0; c < L.call.size(); c++) {
      u16 t0 = L.call[c].kind >= K_FC ? 32768 : 0;
      if (t0)
        k_fill_u16<<<grid_blocks(keyN[c]), 256, 0, s>>>(b.key[c], keyN[c], t0);
      else
        CK(cudaMemsetAsync(b.key[c], 0, keyN[c] * 2, s));
    }
    if (L.Pf != L.s.f)
      k_fill_u16<<<grid_blocks(2 * (size_t)L.Lam * L.Pf * L.T), 256, 0, s>>>(b.KH, 2 * (size_t)L.Lam * L.Pf * L.T, 32768);
  }
  ResidualInt *xs(int sg) const { return b.xst + (size_t)sg * L.s.d * L.T; }
  // The clamp bound of the two norms' outputs a and m and of v only: 2^(cw - 1) - 1, that is 2,047
  // at cw = 12 and 8,191 at cw = 14. q and kx keep the bound 2,047 whatever cw is.
  int clampW() const { return (1 << (L.s.cw - 1)) - 1; }
  void wg(int st, int l, int e, u64 base, int nrow, int ncol, int jofs, cudaStream_t s, i8 *const *dst = nullptr) {
    i8 *w0 = dst ? dst[0] : b.Wl[0], *w1 = dst ? dst[1] : b.Wl[1];
    if (wplanes) {
      wplanes(st, l, e, w0 + (size_t)jofs * nrow, w1 + (size_t)jofs * nrow, s);
      return;
    }
    k_synth_weights<<<grid_blocks((size_t)nrow * ncol), 256, 0, s>>>(seed, st, synth_weight_bits(st, L.s.cw), base, nrow, ncol, jofs, w0,
                                                       w1);
  }
  void rope_tables(double theta0) {
    const int T = L.T, hd = L.s.dh / 2;
    std::vector<short> c((size_t)T * hd), sn((size_t)T * hd);
    auto rha = [](double v) {
      return (short)(v < 0 ? -floor(-v + 0.5) : floor(v + 0.5));
    };
    for (int t = 0; t < T; t++)
      for (int i = 0; i < hd; i++) {
        double an = (double)t * pow(theta0, -2.0 * i / L.s.dh);
        c[(size_t)t * hd + i] = rha(4096.0 * cos(an));
        sn[(size_t)t * hd + i] = rha(4096.0 * sin(an));
      }
    CK(cudaMemcpy(b.Cr, c.data(), c.size() * 2, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(b.Sr, sn.data(), sn.size() * 2, cudaMemcpyHostToDevice));
  }
  i8 *qcell(int X, int l, i64 i) const { return b.qa + (size_t)limb_feature(L.qa[X], l, i, 0) * L.T; }
  size_t qlst(int X) const { return (size_t)L.qa[X].cnt * L.qa[X].J * L.T; }
  // Norm n of layer lam on stream position sg, the residual stream after sg updates. Writes the
  // output's cells, planes [t][k] in b.mT and, with `proof`, the digit row, other cells and keys.
  void norm(int n /* 0, 1 */, int lam, int sg, int inst, int proof, cudaStream_t s) {
    const int T = L.T, d = L.s.d;
    const LayerScaleSet &w = kl[L.s.g * L.Lam + lam];
    const int Mw = n ? w.M2 : w.M1, Kw = n ? w.K2 : w.K1;
    const char *sLo = n ? "N2" : "N1", *sHi = n ? "N2H" : "N1H", *sC = n ? "CN2" : "CN1";
    const TenId Xw = n ? X_mw : X_aw, Xr = n ? X_rm : X_ra, Xo = n ? X_m : X_a;
    const int cR = callOf("ROW"), c1 = callOf(sLo), c2 = callOf(sHi), c3 = callOf(sC);
    norm_ledger(L.s.xl, xs(sg), T, d, Wg, Mw, L.s.eps[n], cell(X_row, 0, inst), b.key[cR],
               posRow + norm_row_len(L.s.xl) * (inst % L.NN), kbase(cR, lam), b.Rv + (size_t)inst * T, b.sigv + (size_t)inst * T,
               b.err, proof, s);
    LAUNCH_BY_FOUR(k_norm_out, T, (size_t)T * d, s, //
                   xs(sg), b.Rv + (size_t)inst * T, T, d, Kw, cell(Xw, 0, lam), cell(Xr, 0, lam), cell(Xo, 0, lam),
                   lst(Xw), lst(Xo), b.key[c1], b.key[c2], b.key[c3], pos2(sLo), pos2(sHi), pos2(sC), kbase(c1, lam),
                   kbase(c2, lam), kbase(c3, lam), proof, clampW());
    for (int i = 0; i < 2; i++) transpose_i8(cell(Xo, i, lam), b.mT[i], d, T, s);
  }
  void write(const char *sLo, const char *sHi, TenId Xr, int nP, int k, int lam, int sgOld, int proof, cudaStream_t s) {
    const int T = L.T, d = L.s.d, sgN = sgOld + 1;
    const bool last = sgN == L.NW * L.Lam;
    const int c1 = callOf(sLo), c2 = callOf(sHi);
    LAUNCH_BY_FOUR(k_epi_residual, T, (size_t)T * d, s, //
                   b.C[0], b.C[1], b.C[2], b.C[3], b.C[4], nP, T, d, k, L.t[Xr].nl, L.s.xl, xs(sgOld), xs(sgN),
                   last ? b.xb1 : cell(X_xs, 0, sgN - 1), last ? (size_t)d * T : lst(X_xs), cell(Xr, 0, lam), lst(Xr),
                   b.key[c1], b.key[c2], pos2(sLo), pos2(sHi), kbase(c1, lam), kbase(c2, lam), b.err, proof);
  }
  // From alloc_scratch(), for experts kept as blocks. rank: a token's index when sorted by expert.
  int *moeI = nullptr;
  i8 *xQ[2] = {nullptr, nullptr}, *hQ[4] = {nullptr, nullptr, nullptr, nullptr};
  ExpertsWork experts_work;
  u32 *tq = nullptr, *t1 = nullptr, *t2 = nullptr;
  u16 *th = nullptr; // with the line gcl: gh of the routed features, indexed like tq
  long long *accM = nullptr;
  long long *accG = nullptr, *accU = nullptr; // the routed gate's and up projection's exact sums, [j < fe][rank]
  bool padded = false;
  // b.C[0] + 256 b.C[1] + 65536 b.C[2] becomes the exact product [j < N][t] of the activations
  // A[0] + 256 A[1] with the weights cur.w0 + 256 cur.w1. All four are planes of signed bytes. The
  // two middle byte products share an array of sums. cur must be set before the call, and
  // prover_gemm()'s unchecked conditions apply.
  // The idea is public: a wide integer product from byte products: Fan et al., "TensorFHE"
  // (arXiv:2212.14191), IV-C: "We use a segment-fusion scheme to fulfill the GEMM without precision
  // loss". There the 8-bit segments are unsigned, each product has its own output, and the fused
  // result is reduced modulo q for a number-theoretic transform. Here the bytes are signed, each
  // side has two (in ffn_dense()'s down projection four and two), products of equal weight share an
  // array of sums, and nothing is reduced.
  void gemm3(const i8 *const *A, int N, int K, cudaStream_t s) {
    const int T = L.T;
    prover_gemm(h, A[0], cur.w0, b.C[0], T, N, K, 0, s);
    prover_gemm(h, A[1], cur.w0, b.C[1], T, N, K, 0, s);
    prover_gemm(h, A[0], cur.w1, b.C[1], T, N, K, 1, s);
    prover_gemm(h, A[1], cur.w1, b.C[2], T, N, K, 0, s);
  }
  // A dense feed-forward layer. The up projection's rows follow the gating projection's, so one
  // product of 2 f rows makes both. It leaves the down projection as five arrays of sums, b.C[0] to
  // b.C[4], for write(). The idea is public: merged gating and up projections: vLLM's
  // documentation, "Basic Model" (https://docs.vllm.ai/en/v0.9.2/contributing/model/basic.html),
  // among its tensor-parallel layers: "MergedColumnParallelLinear". Here there is one GPU and
  // nothing is split; only the merging is the same.
  void ffn_dense(int lam, int l, int proof, cudaStream_t s) {
    const int T = L.T, d = L.s.d, f = L.s.f, Lam = L.Lam;
    if (wres)
      cur = wres(2, l);
    else {
      wg(6, l, 0, (u64)l * d * f, d, f, 0, s);
      wg(7, l, 0, (u64)l * d * f, d, f, f, s);
      cur = WP{b.Wl[0], b.Wl[1]};
    }
    gemm3(b.mT, 2 * f, d, s);
    LAUNCH_BY_FOUR(k_epi_gate_up, T, (size_t)T * f, s, //
                   b.C[0], b.C[1], b.C[2], T, f, 0, (size_t)f * T, kl[l].kg, kl[l].ku, L.s.hg, L.s.hu, b.Ts,
                   cell(X_gi, 0, lam), cell(X_s, 0, lam), cell(X_u, 0, lam), cell(X_rg, 0, lam), cell(X_ru, 0, lam),
                   lst(X_gi), b.hF, (size_t)T * f, kp("G", lam), kp("U", lam), kp("GH", lam), kp("UH", lam),
                   kp("UT", lam), kp("F", lam), pos2("G"), pos2("U"), pos2("GH"), pos2("UH"), pos2("UT"), pos2("F"),
                   b.KH + (size_t)lam * L.Pf * T, b.KH + ((size_t)Lam + lam) * L.Pf * T, gate_clamp(lam, 0, proof),
                   b.err, proof);
    for (int i = 0; i < 4; i++) transpose_i8(b.hF + (size_t)i * T * f, b.hT[i], f, T, s);
    if (wres)
      cur = wres(3, l);
    else
      wg(8, l, 0, (u64)l * f * d, f, d, 0, s);
    for (int sI = 0; sI < 5; sI++) {
      int first = 1;
      for (int a = 0; a < 4; a++) {
        int w = sI - a;
        if (w < 0 || w > 1) continue;
        prover_gemm(h, b.hT[a], w ? cur.w1 : cur.w0, b.C[sI], T, d, f, !first, s);
        first = 0;
      }
    }
  }
  // gemm3() with the weights in b.Wl from row wofs on and the sums in b.C from row cofs on.
  void gemm3at(const i8 *const *A, int lda, size_t wofs, int K, int N, size_t cofs, int acc, cudaStream_t s) {
    const int T = L.T;
    const i8 *w0 = b.Wl[0] + wofs * K, *w1 = b.Wl[1] + wofs * K;
    prover_gemm(h, A[0], w0, b.C[0] + cofs * T, T, N, K, acc, s, lda);
    prover_gemm(h, A[1], w0, b.C[1] + cofs * T, T, N, K, acc, s, lda);
    prover_gemm(h, A[0], w1, b.C[1] + cofs * T, T, N, K, 1, s, lda);
    prover_gemm(h, A[1], w1, b.C[2] + cofs * T, T, N, K, acc, s, lda);
  }
  // The end of a MoE layer whose routed tensors are kept as blocks. The shared expert's down
  // projection (weights in b.Wl, h in b.hT, pitch f) and the routed sums accM[j][rank] are added,
  // the total is rounded and the quotient added to stream position sgOld (k_epi_residual()). The routed
  // sums are brought to token order in xs(sgOld + 1), which k_epi_residual() reads, then overwrites.
  void shared_down(const MoeTables &M, int lam, int l, int sgOld, int proof, cudaStream_t s) {
    const int T = L.T, d = L.s.d, f = L.s.f, fe = L.s.fe;
    const int sgN = sgOld + 1;
    const bool last = sgN == L.NW * L.Lam;
    const int c1 = callOf("D"), c2 = callOf("DH");
    for (int sI = 0; sI < 5; sI++) {
      int first = 1;
      for (int a = 0; a < 4; a++) {
        int w = sI - a;
        if (w < 0 || w > 1) continue;
        prover_gemm(h, b.hT[a], b.Wl[w], b.C[sI], T, d, fe, !first, s, f);
        first = 0;
      }
    }
    ResidualInt *accT = xs(sgN);
    unpermute(accM, M.rank_of, T, d, accT, s);
    lie::here(l, lie::D, s);
    LAUNCH_BY_FOUR(k_epi_residual, T, (size_t)T * d, s, //
                   b.C[0], b.C[1], b.C[2], b.C[3], b.C[4], 5, T, d, kl[l].kd, L.t[X_rd].nl, L.s.xl, xs(sgOld), xs(sgN),
                   last ? b.xb1 : cell(X_xs, 0, sgN - 1), last ? (size_t)d * T : lst(X_xs), cell(X_rd, 0, lam),
                   lst(X_rd), b.key[c1], b.key[c2], pos2("D"), pos2("DH"), kbase(c1, lam), kbase(c2, lam), b.err, proof,
                   accT);
    CK(cudaGetLastError());
  }
  // A MoE layer after its second norm (m in b.mT). Without wblocks each expert's product runs over
  // all tokens, with other experts' rows zeroed. Returns true if it wrote the new stream position.
  bool ffn_moe(int lam, int l, int sgOld, int proof, cudaStream_t s) {
    cur = WP{b.Wl[0], b.Wl[1]};
    const int T = L.T, d = L.s.d, f = L.s.f, fe = L.s.fe, NE = L.s.NE, Lam = L.Lam, lamM = lam >> 1;
    int *es = b.es + (size_t)lamM * T;
    auto P0 = [&](const char *sg) {
      int id = L.segid(sg);
      return id < 0 ? 0ll : L.pos(id, 0);
    };
    wg(9, l, 0, (u64)l * d * NE, d, NE, 0, s);
    gemm3at(b.mT, d, 0, d, NE, 0, 0, s);
    {
      const int c1 = callOf("LR"), c2 = callOf("LH"), c3 = callOf("LT");
      RouterClamp rc;
      rc.on = L.s.rcl;
      if (rc.on && proof) {
        const int c4 = callOf("CLR"), c5 = callOf("L3");
        rc.ly = cell(X_ly, 0, lamM);
        rc.l3 = cell(X_l3, 0, lamM);
        rc.st = lst(X_ly);
        rc.kC = b.key[c4];
        rc.k3 = b.key[c5];
        rc.pC = P0("CLR");
        rc.p3 = P0("L3");
        rc.bC = kbase(c4, lam);
        rc.b3 = kbase(c5, lam);
      }
      k_epi_router<<<grid_blocks((size_t)T * NE), 256, 0, s>>>(b.C[0], b.C[1], b.C[2], T, NE, kl[l].kr, L.s.hr, lrv,
                                                      cell(X_lr, 0, lamM), lst(X_lr), cell(X_rl, 0, lamM), b.key[c1],
                                                      b.key[c2], b.key[c3], P0("LR"), P0("LH"), P0("LT"),
                                                      kbase(c1, lam), kbase(c2, lam), kbase(c3, lam), rc, b.err, proof);
    }
    {
      const int c1 = callOf("SEL"), c2 = callOf("DD"), c3 = callOf("MR");
      k_route<<<grid_blocks(T), 256, 0, s>>>(lrv, T, NE, b.Sig, es, scv, cell(X_sel, 0, lamM), cell(X_sc, 0, lamM),
                                        lst(X_sc), b.key[c1], b.key[c2], b.key[c3], P0("SEL"), P0("DD"), P0("MR"),
                                        kbase(c1, lam), kbase(c2, lam), kbase(c3, lam), L.s.gcl ? 16384 : 32768, proof);
    }
    {
      const int c1 = callOf("XR");
      k_xr<<<grid_blocks((size_t)T * d), 256, 0, s>>>(cell(X_m, 0, lam), cell(X_m, 1, lam), scv, T, d, cell(X_xr, 0, lamM),
                                                 lst(X_xr), b.key[c1], pos2("XR"), kbase(c1, lam), proof);
      for (int i = 0; i < 2; i++) transpose_i8(cell(X_xr, i, lamM), b.xT[i], d, T, s);
    }
    lie::here(l, lie::FFN, s); // the shared part
    // Experts kept as blocks: the tokens are grouped by expert. prover_gemm() and element-wise kernels
    // compute the shared expert, and prover/weights_blocks/experts.cuh the routed experts.
    if (wblocks && T % 64 == 0 && fe % 128 == 0 && d % 128 == 0 && NE <= 1024) {
      const size_t o = (size_t)fe * T;
      auto offc = [&](i8 *p) {
        return p ? p + o : p;
      };
      MoeTables M;
      M.bind(moeI, T, NE);
      u16 *KHs = b.KH + (size_t)lam * L.Pf * T, *KHu = b.KH + ((size_t)Lam + lam) * L.Pf * T;
      wg(6, l, 0, (u64)l * d * fe, d, fe, 0, s);
      wg(7, l, 0, (u64)l * d * fe, d, fe, fe, s);
      // The shared expert's gating and up projections: weights in rows [0, fe) and [fe, 2 fe) of
      // b.Wl, sums in the same rows of b.C. Then h is transposed to planes [t] of pitch f.
      gemm3at(b.mT, d, 0, d, 2 * fe, 0, 0, s);
      LAUNCH_BY_FOUR(k_epi_gate_up, T, (size_t)T * fe, s, //
                     b.C[0], b.C[1], b.C[2], T, fe, 0, (size_t)fe * T, kl[l].kg, kl[l].ku, L.s.hg, L.s.hu, b.Ts,
                     cell(X_gi, 0, lam), cell(X_s, 0, lam), cell(X_u, 0, lam), cell(X_rg, 0, lam), cell(X_ru, 0, lam),
                     lst(X_gi), b.hF, (size_t)T * f, kp("G", lam), kp("U", lam), kp("GH", lam), kp("UH", lam),
                     kp("UT", lam), kp("F", lam), pos2("G"), pos2("U"), pos2("GH"), pos2("UH"), pos2("UT"), pos2("F"),
                     KHs, KHu, gate_clamp(lam, 0, proof), b.err, proof);
      CK(cudaGetLastError());
      for (int i = 0; i < 4; i++) transpose_i8(b.hF + (size_t)i * T * f, b.hT[i], fe, T, s, f);
      // In a test build that is not part of this repository (prover/common/lie.cuh), a fault alters
      // es[token] while the routing tables are made, so only the routed products are false.
      auto chosen_of = [&](long long, long long token, long long) -> int * { return token >= 0 && token < T ? es + token : nullptr; };
      lie::in_memory<int>(l, lie::EXPERTS, lie::EXPERT, s, chosen_of);
      launch_moe_prep(M, es, s);
      if (experts_wanted) experts_wanted(l, es, s);
      lie::back_in_memory<int>(l, lie::EXPERTS, lie::EXPERT, s, chosen_of);
      if (routed_wanted && s != cudaStreamPerThread) { // the callbacks use the calling host thread's stream
        printf(
            "internal error: the forward pass runs on a different stream from its caller's while experts are loaded "
            "layer by layer\n");
        exit(4);
      }
      if (routed_wanted) routed_wanted(l);
      const wb::Blocks &Bg = *wblocks(10, l), &Bu = *wblocks(11, l), &Bd = *wblocks(12, l);
      // The products below write rows * T sums, with rows from the blocks' own shape, into
      // buffers whose sizes come from the layer's shape, so the two shapes must agree.
      if (Bg.rows != (size_t)fe || Bu.rows != (size_t)fe || Bd.rows != (size_t)d || Bg.inputs != d || Bu.inputs != d ||
          Bd.inputs != fe) {
        printf("internal error: a routed expert's tensor has different numbers of rows or inputs from its layer of the "
               "model; the program stops\n");
        exit(4);
      }
      for (int i = 0; i < 2; i++) launch_moe_rows(M, b.xT[i], d, xQ[i], d, d, s);
      lend_experts_work();
      ExpertsWork &work = experts_work;
      work.fetch(M, s);
      grouped_blocks<2>(M, Bg, xQ, d, accG, work, s);
      grouped_blocks<2>(M, Bu, xQ, d, accU, work, s);
      lie::here(l, lie::EXPERTS, s);
      gate_up(M, accG, accU, fe, kl[l].kgR, kl[l].kuR, b.Ts, hQ, fe, tq, t1, t2, b.err, proof, s, L.s.gcl, th);
      if (proof)
        launch_moe_gu_out(M, tq, t1, t2, fe, fe, kl[l].kgR, kl[l].kuR, L.s.hg, L.s.hu, b.Ts, cell(X_gi, 0, lam) + o,
                       cell(X_s, 0, lam) + o, cell(X_u, 0, lam) + o, offc(cell(X_rg, 0, lam)),
                       offc(cell(X_ru, 0, lam)), lst(X_gi), kp("G", lam), kp("U", lam), kp("GH", lam), kp("UH", lam),
                       kp("UT", lam), kp("F", lam), pos2("G"), pos2("U"), pos2("GH"), pos2("UH"), pos2("UT"),
                       pos2("F"), KHs + o, KHu + o, s, gate_clamp(lam, o, proof), th);
      // The routed down product comes before the shared expert's: it uses b.Wl as a work buffer
      // (except in pk_grouped_blocks()), and the shared expert's down weights go to b.Wl next.
      // A tensor of a wider type (Q8_0, F16, BF16) goes through grouped_blocks()'s own body at either setting.
      if (routed_down_wanted) routed_down_wanted(l);
      grouped_blocks<4>(M, Bd, hQ, fe, accM, work, s);
      work.forget();
      if (routed_read) routed_read(l);
      wg(8, l, 0, (u64)l * f * d, fe, d, 0, s);
      shared_down(M, lam, l, sgOld, proof, s);
      return true;
    }
    if (wblocks) {
      printf("weights: experts kept as blocks need a number of tokens that is a multiple of 64, and at most 1024 "
             "experts\n");
      exit(2);
    }
    // The shared expert: rows [0, fe) of C are its gating, and rows [f, f + fe) its up projection.
    wg(6, l, 0, (u64)l * d * fe, d, fe, 0, s);
    wg(7, l, 0, (u64)l * d * fe, d, fe, fe, s);
    gemm3at(b.mT, d, 0, d, fe, 0, 0, s);
    gemm3at(b.mT, d, fe, d, fe, f, 0, s);
    for (int e = 0; e < NE; e++) {
      for (int i = 0; i < 2; i++) k_expert_mask<<<grid_blocks((size_t)T * d / 16), 256, 0, s>>>(b.xT[i], d, b.mk[i], T, d, es, e);
      wg(10, l, e, ((u64)l * NE + e) * d * fe, d, fe, 0, s);
      wg(11, l, e, ((u64)l * NE + e) * d * fe, d, fe, fe, s);
      gemm3at(b.mk, d, 0, d, fe, fe, e > 0, s);
      gemm3at(b.mk, d, fe, d, fe, f + fe, e > 0, s);
    }
    for (int half = 0; half < 2; half++) {
      const size_t o = (size_t)half * fe * T;
      auto offc = [&](i8 *p) {
        return p ? p + o : p;
      };
      LAUNCH_BY_FOUR(k_epi_gate_up, T, (size_t)T * fe, s, //
                     b.C[0] + o, b.C[1] + o, b.C[2] + o, T, fe, half * fe, (size_t)f * T, half ? kl[l].kgR : kl[l].kg,
                     half ? kl[l].kuR : kl[l].ku, L.s.hg, L.s.hu, b.Ts, cell(X_gi, 0, lam) + o, cell(X_s, 0, lam) + o,
                     cell(X_u, 0, lam) + o, offc(cell(X_rg, 0, lam)), offc(cell(X_ru, 0, lam)), lst(X_gi), b.hF + o,
                     (size_t)T * f, kp("G", lam), kp("U", lam), kp("GH", lam), kp("UH", lam), kp("UT", lam),
                     kp("F", lam), pos2("G"), pos2("U"), pos2("GH"), pos2("UH"), pos2("UT"), pos2("F"),
                     b.KH + (size_t)lam * L.Pf * T + o, b.KH + ((size_t)Lam + lam) * L.Pf * T + o,
                     gate_clamp(lam, o, proof), b.err, proof);
    }
    for (int i = 0; i < 4; i++) transpose_i8(b.hF + (size_t)i * T * f, b.hT[i], f, T, s);
    auto down = [&](const i8 *const *A, int lda, int acc0) {
      for (int sI = 0; sI < 5; sI++) {
        int first = !acc0;
        for (int a = 0; a < 4; a++) {
          int w = sI - a;
          if (w < 0 || w > 1) continue;
          prover_gemm(h, A[a], b.Wl[w], b.C[sI], T, d, fe, !first, s, lda);
          first = 0;
        }
      }
    };
    wg(8, l, 0, (u64)l * f * d, fe, d, 0, s);
    down(b.hT, f, 0);
    for (int e = 0; e < NE; e++) {
      for (int i = 0; i < 4; i++)
        k_expert_mask<<<grid_blocks((size_t)T * fe / 16), 256, 0, s>>>(b.hT[i] + fe, f, b.mk[i], T, fe, es, e);
      wg(12, l, e, ((u64)l * NE + e) * fe * d, fe, d, 0, s);
      down(b.mk, fe, 1);
    }
    return false;
  }
  // Runs this group's layers on stream s. Not checked: L.s.att is 1 and xs(0) holds the input.
  void forward(int proof, cudaStream_t s = cudaStreamPerThread) {
    const int T = L.T, d = L.s.d, Lam = L.Lam, pf = L.s.prof;
    // The stream and the workspace are set at every pass, because other code sets its own on the
    // host thread's handle in between (prover/prover/layer_prove.cuh). Code that forward() calls
    // must restore both for later products: attention uses blas::Borrowed (prover/common/blas.h).
    h = blas::blas(s, blas_room);
    if (zeroes_its_flags) CK(cudaMemsetAsync(b.err, 0, 4, s));
    // If the buffers keep their contents between passes (forward_pass_buffers_last is true), the
    // padding keys are written only once. That relies on no kernel of the forward pass or of the
    // prover writing to them; this file does not check it. Otherwise every proof pass writes them.
    // Public source: clearing a buffer when it is allocated, not at each use: FFmpeg documentation
    // (Doxygen, trunk), libavcodec, group lavc_misc (miscellaneous utility functions):
    // av_fast_padded_malloc and av_fast_padded_mallocz,
    // https://ffmpeg.org/doxygen/trunk/group__lavc__misc.html ; with libavutil, group
    // lavu_mem_funcs (functions for allocating, freeing and copying memory):
    // av_fast_malloc and av_fast_mallocz,
    // https://ffmpeg.org/doxygen/trunk/group__lavu__mem__funcs.html.
    if (proof && (!padded || !forward_pass_buffers_last)) {
      pad(s);
      padded = true;
    }

    for (int lam = 0; lam < Lam; lam++) {
      const int l = L.s.g * Lam + lam;
      int sg = L.NW * lam;
      {
        const int dq = (int)L.dq, dkv = (int)L.dkv, nqk = (int)L.nqk;
        lie::here(l, lie::NORM1, s);
        norm(0, lam, sg, 2 * lam, proof, s);
        lie::here(l, lie::QKV, s);
        if (wres)
          cur = wres(0, l);
        else {
          wg(2, l, 0, (u64)l * d * dq, d, dq, 0, s);
          wg(3, l, 0, (u64)l * d * dkv, d, dkv, dq, s);
          wg(4, l, 0, (u64)l * d * dkv, d, dkv, nqk, s);
          cur = WP{b.Wl[0], b.Wl[1]};
        }
        gemm3(b.mT, nqk + dkv, d, s);
        const int c1 = callOf("QK"), c2 = callOf("QKH"), c3 = callOf("CQK"), v1 = callOf("V"), v2 = callOf("VH"),
                  v3 = callOf("CV");
        const size_t vo = (size_t)nqk * T;
        {
          LAUNCH_BY_FOUR(k_epi_qk, T, (size_t)T * nqk / 2, s, //
                         b.C[0], b.C[1], b.C[2], T, nqk, dq, L.s.dh, kl[l].kq, kl[l].kk, L.s.hqk, b.Cr, b.Sr, kl[l].rp,
                         cell(X_qkw, 0, lam), cell(X_rqk, 0, lam), lst(X_qkw), qcell(0, 0, lam), qlst(0),
                         qcell(1, 0, lam), qlst(1), b.key[c1], b.key[c2], b.key[c3], pos2("QK"), pos2("QKH"),
                         pos2("CQK"), kbase(c1, lam), kbase(c2, lam), kbase(c3, lam), b.err, proof);
          LAUNCH_BY_FOUR(k_epi_v, T, (size_t)T * dkv, s, //
                         b.C[0] + vo, b.C[1] + vo, b.C[2] + vo, T, dkv, kl[l].kv, clampW(), L.s.hv, cell(X_vw, 0, lam),
                         lst(X_vw), cell(X_rv, 0, lam), qcell(2, 0, lam), qlst(2), b.key[v1], b.key[v2], b.key[v3],
                         pos2("V"), pos2("VH"), pos2("CV"), kbase(v1, lam), kbase(v2, lam), kbase(v3, lam), b.err,
                         proof);
        }
        {
          k_c2t<<<grid_blocks((size_t)T * dq), 256, 0, s>>>(qcell(0, 0, lam), qlst(0), T, dq, b.q16);
          k_c2t<<<grid_blocks((size_t)T * dkv), 256, 0, s>>>(qcell(1, 0, lam), qlst(1), T, dkv, b.k16);
          k_c2t<<<grid_blocks((size_t)T * dkv), 256, 0, s>>>(qcell(2, 0, lam), qlst(2), T, dkv, b.v16);
          lie::here(l, lie::ATTENTION, s);
          if (attention)
            attention(l, b.q16, b.k16, b.v16, kl[l].sh, b.o16, s);
          else
            fa->run(b.q16, b.k16, b.v16, kl[l].sh, b.o16, s);
          lie::here(l, lie::QKV, s);
          k_t2c<<<grid_blocks((size_t)T * dq), 256, 0, s>>>(b.o16, T, dq, qcell(3, 0, lam), qlst(3), b.mT[0], b.mT[1]);
        }
        if (wres)
          cur = wres(1, l);
        else {
          wg(5, l, 0, (u64)l * dq * d, dq, d, 0, s);
          cur = WP{b.Wl[0], b.Wl[1]};
        }
        gemm3(b.mT, d, dq, s);
        lie::here(l, lie::O, s);
        write("O", "OH", X_ro, 3, kl[l].ko, lam, sg, proof, s);
        sg++;
      }
      lie::here(l, lie::NORM2, s);
      norm(1, lam, sg, L.NN * lam + L.NN - 1, proof, s);
      const bool mo = moe_at_layer(pf, l, L.s.NE);
      bool wrote = false;
      lie::here(l, mo ? lie::ROUTER : lie::FFN, s);
      if (mo)
        wrote = ffn_moe(lam, l, sg, proof, s);
      else
        ffn_dense(lam, l, proof, s);
      lie::here(l, lie::D, s);
      if (!wrote) write("D", "DH", X_rd, 5, kl[l].kd, lam, sg, proof, s);
    }
  }
};
