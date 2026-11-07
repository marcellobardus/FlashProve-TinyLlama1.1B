// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// Kernels and helpers shared by the sub-proofs of cp_prove, the GPU prover. A sub-proof (`fork` in
// names) covers one part of the forward pass. Headers included mid-file use definitions above them.
#pragma once
// T is an element of the cubic extension of the field of size p = 2^64 - 2^32 + 1 (`FIELD_P`); F is its
// arithmetic. T_ or NT is the number of tokens. A limb is a base-256 digit from -128 to 127.
#include "common/end_name.h" // name_the_end (M5g)
#include "prover/fork_memory.h"

#include "common/numa_pin.h"
#include "hiding/product_test.cuh"
#include "prover/layer_points.cuh"
#include "common/row_pass_core.cuh"
#include "common/hash_tree.cuh"
#include <cublas_v2.h>
#include <chrono>
#include <algorithm>
typedef int8_t i8;
typedef uint16_t u16;
// Prints the line "setting:": compiled constants; "hb" and hb_word, the parameter file's target in
// bits (line `sec`; pass -1 if unknown); "a serving build" (production) or "built for tests alone".
static void print_setting(int hb_word) {
  printf("setting: field degree %d; requests: fold width %d, %d repetitions mod q; opening: fold width %d, %d "
         "repetitions mod q; identities up to 2^%d keys; folds inside 2^%d; version word %d",
         EDEG, ETAU, ENJ, ETAUF, ENJW, EIDB, ELBZ, VERSION_WORD);
  if (hb_word >= 0) printf("; hb %d", hb_word);
  printf(SERVING_OR("; a serving build\n", BUILT_SETTING "\n"));
}
// The loop form is public: the grid-stride loop: "CUDA Pro Tip: Write Flexible Kernels with Grid-Stride Loops", a post
// on NVIDIA's developer blog, https://developer.nvidia.com/blog/cuda-pro-tip-write-flexible-kernels-grid-stride-loops/.
#define GSL(i, n)                                                                                                      \
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < (n); i += (size_t)gridDim.x * blockDim.x)
static int nblk(size_t n, int t = 256, int cap = 1 << 20) {
  size_t b = (n + t - 1) / t;
  return (int)std::min<size_t>(b, cap);
}
static int gsl_blocks_of(size_t n, int t = 256, int cap = 1 << 20) {
  return nblk(n, t, cap);
}
typedef int16_t i16;
typedef long long i64_;
// The eq table of z: o[x] = prod_{i<n} (x_i ? z[i] : 1 - z[i]) for x < 2^n; x_0 is x's top bit.
__global__ void k_eq(const T *z, int n, T *o) {
  GSL(x, (size_t)1 << n) {
    T v = F::one();
    for (int i = 0; i < n; i++) {
      T r = z[i];
      v = F::mul(v, ((x >> (n - 1 - i)) & 1) ? r : F::sub(F::one(), r));
    }
    o[x] = v;
  }
}
//   out[q][row] = sum_{y<L} phi[q][y] (M[row L + y] + 128) for each of the NQ vectors phi[q]
// M has `rows` rows of L signed bytes; phi and out are GPU arrays of NQ pointers. The +128 makes
// the entries unsigned for F::smac, which adds an element times a u32 to F::SAcc, an unreduced sum;
// k_phisum and k_rowfix remove the +128. F::sred reduces a sum after at most 64 products, far
// inside F::SAcc's bound, (terms) x (largest factor) <= 2^32 (prover/hiding/pt_field.h).
// Requires blocks of 256 threads. No kernel in this file checks the launch that it requires.
// Public sources for the techniques of this kernel:
// Delayed reduction is a public idea ([FFLAS]), specific to modular arithmetic, here modulo p:
// J.-G. Dumas, P. Giorgi, C. Pernet, "Dense Linear Algebra over Word-Size Prime Fields: the FFLAS
// and FFPACK Packages", ACM TOMS 35(3), 2008, arXiv:cs/0601133, section 2.1.2 (delayed reduction
// of sums of products), https://arxiv.org/pdf/cs/0601133.
// The idea is public: block reduction by warp shuffles and shared memory: J. Luitjens, "Faster
// Parallel Reductions on Kepler", NVIDIA Technical Blog, 2014, "Block Reduce" (here a loop adds the
// warps' sums), https://developer.nvidia.com/blog/faster-parallel-reductions-kepler/.
// The idea is public: several vectors in one pass over the matrix: E.-J. Im, K. Yelick, "Optimizing
// Sparse Matrix Computations for Register Reuse in SPARSITY", section 2,
// https://ftp.eecs.berkeley.edu/~yelick/papers/iccs01.pdf. There the matrix is sparse.
// The idea is public: a count as a template parameter, instances picked by a switch: M. Harris,
// "Optimizing Parallel Reduction in CUDA", NVIDIA Developer Technology, slides "Complete
// Unrolling", "Unrolling with Templates" and "Invoking Template Kernels",
// https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf. There the size is the
// block size; here the switch is in RowJob::run().
// The loop form is public: the block-stride loop (here split into two nested loops): shown in a
// listing in NVIDIA, "CUDA Programming Guide", page "Writing SIMT Kernels", "Distributed Shared
// Memory", https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/writing-cuda-kernels.html.
template <int NQ> __global__ void k_rows(const i8 *M, size_t L, size_t rows, const T *const *phi, T *const *out) {
  __shared__ T sh[NQ][8];
  for (size_t r = blockIdx.x; r < rows; r += gridDim.x) {
    T acc[NQ];
    for (int q = 0; q < NQ; q++) {
      acc[q] = F::zero();
    }
    for (size_t y0 = threadIdx.x; y0 < L; y0 += 256 * 64) {
      F::SAcc a[NQ];
      for (int q = 0; q < NQ; q++) {
        a[q] = F::szero();
      }
      for (int s = 0; s < 64; s++) {
        size_t y = y0 + (size_t)s * 256;
        if (y >= L) {
          break;
        }
        u32 v = (u32)(M[r * L + y] + 128);
        for (int q = 0; q < NQ; q++) {
          F::smac(a[q], F::sprep(phi[q][y]), v);
        }
      }
      for (int q = 0; q < NQ; q++) {
        acc[q] = F::add(acc[q], F::sred(a[q]));
      }
    }
    for (int q = 0; q < NQ; q++) {
      T v = acc[q];
      for (int o = 16; o; o >>= 1) {
        v = F::add(v, shfl_down<F>(v, o));
      }
      if ((threadIdx.x & 31) == 0) {
        sh[q][threadIdx.x >> 5] = v;
      }
    }
    __syncthreads();
    if (threadIdx.x < NQ) {
      T v = F::zero();
      for (int w = 0; w < 8; w++) {
        v = F::add(v, sh[threadIdx.x][w]);
      }
      out[threadIdx.x][r] = v;
    }
    __syncthreads();
  }
}
// s[q] = 128 * the sum of phi[q][0..L).
// Requires a launch of NQ blocks of 256 threads.
// The idea is public: block reduction by warp shuffles and shared memory: J. Luitjens, "Faster
// Parallel Reductions on Kepler" (cited at k_rows()).
// The loop form is public: the block-stride loop: CUDA Programming Guide (cited at k_rows()).
template <int NQ> __global__ void k_phisum(size_t L, const T *const *phi, T *s) {
  __shared__ T sh[8];
  int q = blockIdx.x;
  T v = F::zero();
  for (size_t y = threadIdx.x; y < L; y += 256) {
    v = F::add(v, phi[q][y]);
  }
  for (int o = 16; o; o >>= 1) {
    v = F::add(v, shfl_down<F>(v, o));
  }
  if ((threadIdx.x & 31) == 0) {
    sh[threadIdx.x >> 5] = v;
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    T t = F::zero();
    for (int w = 0; w < 8; w++) {
      t = F::add(t, sh[w]);
    }
    s[q] = F::mul(t, F::small(128));
  }
}
// out[q][r] -= s[q] for every q < NQ and r < rows: removes what the +128 in k_rows() added.
template <int NQ> __global__ void k_rowfix(size_t rows, T *const *out, const T *s) {
  GSL(i, rows * NQ) {
    int q = (int)(i / rows);
    size_t r = i % rows;
    out[q][r] = F::sub(out[q][r], s[q]);
  }
}
static u32 *g_th[2];
// Returns the tree hash, the same as TH() (prover/common/host_common.h), of the n words at `in`
// (GPU memory): a pointer to DW words in g_th, valid until the next call. Calls exit(2) if n < DW.
// Not checked: that g_th's two buffers (from Prover::setup_buffers()) are large enough.
static const u32 *treehash(const u32 *in, size_t n) {
  if (n < (size_t)DW) {
    fprintf(stderr,
            "internal error: the tree hash of %zu words was requested; it is defined for %d words or more; the program "
            "ends\n",
            n, DW);
    name_the_end("guard"); // serving: p258 end 1 guard (M5g: an internal-consistency stop)
    exit(2);
  }
  int p = 0;
  u32 num = TH_LIST;
  for (;;) {
    const size_t nc = (n + THPW - 1) / THPW;
    k_th<<<(int)((nc + 127) / 128), 128>>>(in, n, g_th[p], num);
    in = g_th[p];
    p ^= 1;
    n = nc * DW;
    if (nc == 1) return in;
    num = TH_UPPER;
  }
}

#include "hiding/token_sumcheck.cuh"
#include "hiding/trees.cuh"
#include "hiding/tail.cuh"
#include "hiding/walk.cuh"
#include "hiding/eqs.cuh"
#include "attn/output.cuh"
#include "prover/forward.cuh"
#include "prover/weights_generated.cuh"
#include "weights_blocks/block_weights.cuh"
#include "weights_blocks/made_up_source.cuh"
#include "weights_blocks/gguf_source.cuh"
#include "prover/arguments.h"
#include "common/test_words.h"
#include "prover/ends.h"
#include <memory>
#include "attn/fork.cuh"
#include <map>
#include <set>
__device__ __forceinline__ T ext_from_signed(long long v) {
  return ext_base(v < 0 ? FIELD_P - (u64)(-v) : (u64)v);
}
__global__ void k_gather2(const T *e, Pos2 p, int n, T *o) {
  GSL(x, (size_t)n) o[x] = e[p.at((int)x)];
}
__global__ void k_cat(const T *a, int na, const T *b, int nb, T *o) {
  int i = threadIdx.x;
  if (i < na) {
    o[i] = a[i];
  } else if (i < na + nb) {
    o[i] = b[i - na];
  }
}
// Sums of limb products for a norm's sum of squares. With b_0 .. b_(NL-1) the limbs of x[i][k][t] =
// xst[(i*d + k)*T_ + t]: Ws[(i*T_ + t)*NS + s] = the sum over k and over a + c = s of b_a * b_c.
template <int NL, int LANES>
__global__ void __launch_bounds__(32 * LANES) k_ws(const long long *xst, int T_, int d, long long *Ws) {
  constexpr int NS = 2 * NL - 1; // the number of sums: one for each s = a + c
  __shared__ long long sh[NS][LANES][33];
  const int tx = threadIdx.x, ty = threadIdx.y, t = blockIdx.x * 32 + tx, i = blockIdx.y;
  const long long *x = xst + (size_t)i * d * T_;
  long long W[NS] = {};
  if (t < T_) {
    for (int k = ty; k < d; k += LANES) {
      long long v = x[(size_t)k * T_ + t];
      int b[NL];
      for (int l = 0; l < NL; l++) {
        b[l] = (int)(((v + 128) & 255) - 128);
        v = (v - b[l]) >> 8;
      }
      for (int a = 0; a < NL; a++) {
        for (int c = 0; c < NL; c++) {
          W[a + c] += b[a] * b[c];
        }
      }
    }
  }
  for (int s = 0; s < NS; s++) {
    sh[s][ty][tx] = W[s];
  }
  __syncthreads();
  if (ty || t >= T_) {
    return;
  }
  for (int s = 0; s < NS; s++) {
    long long a = 0;
    for (int y = 0; y < LANES; y++) {
      a += sh[s][y][tx];
    }
    Ws[((size_t)i * T_ + t) * NS + s] = a;
  }
}
// most[0] = max(most[0], |x[i]|) over n values; the caller initializes most[0]. Launched only in
// Prover::g_norm_sums() (prover/prover/layer_prove.cuh), under an environment variable that a
// production build does not read.
// The idea is public: privatization, with one atomic merge per thread: M. Bertini, "Parallel
// Computing", slides "CUDA: atomic operations, privatization, algorithms", slide "Privatization",
// https://www.micc.unifi.it/bertini/download/parallel/2016-2017/14_gpu_cuda_5.pdf.
// The idea is public: a thread-local partial result merged by an atomic: T. Mattson, "A 'Hands-on'
// Introduction to OpenMP", slide "Example: Using an atomic to remove impact of false sharing",
// https://www.openmp.org/wp-content/uploads/Intro_To_OpenMP_Mattson.pdf.
// There the operation is a sum, here a maximum.
__global__ void k_absmost(const long long *x, size_t n, unsigned long long *most) {
  unsigned long long m = 0;
  GSL(i, n) m = max(m, (unsigned long long)llabs(x[i]));
  atomicMax(most, m);
}
// Launches k_ws for NI norm instances whose values have xl = 5 limbs; any other xl is taken as 4.
static inline void ws_of(int xl, const long long *xst, int T_, int d, int NI, long long *Ws) {
  if (xl == 5)
    k_ws<5, 16><<<dim3((T_ + 31) / 32, NI), dim3(32, 16)>>>(xst, T_, d, Ws);
  else
    k_ws<4, 32><<<dim3((T_ + 31) / 32, NI), dim3(32, 32)>>>(xst, T_, d, Ws);
}
// o[q] = sum_t e[t] * fn(q, t), masked. o must point into the sub-proof's message array: what is
// stored is the sum plus the mask entry of the slot o + q, a random element (sc_door::claim(),
// prover/hiding/sumcheck_door.cuh). The sum is never stored in global memory without it, so a
// launch or a store that a faulty GPU skips leaves in the slot what was there before. Requires one
// block of 256 threads for each q.
// The idea is public: block reduction by warp shuffles and shared memory: J. Luitjens, "Faster
// Parallel Reductions on Kepler" (cited at k_rows()).
// The loop form is public: the block-stride loop: CUDA Programming Guide (cited at k_rows()).
template <class Fn> __global__ void k_tsum(int T_, const T *e, T *o, Fn fn) {
  __shared__ T sh[8];
  const int q = blockIdx.x;
  T a = F::zero();
  for (int t = threadIdx.x; t < T_; t += 256) {
    a = F::add(a, F::mul(e[t], fn(q, t)));
  }
  for (int s = 16; s; s >>= 1) {
    a = F::add(a, shfl_down<F>(a, s));
  }
  if ((threadIdx.x & 31) == 0) {
    sh[threadIdx.x >> 5] = a;
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    T v = F::zero();
    for (int w = 0; w < 8; w++) {
      v = F::add(v, sh[w]);
    }
    sc_door::claim(sc_door::Site::TOKEN_SUMS, v, o + q);
  }
}
#include "hiding/rotation.h" // gf_times_2_to
// Returns x times 2^s, for a canonical x and any s >= 0, coordinate by coordinate. Modulo p,
// 2^192 = 1 (grm::TURNS = 192), so the exponent is reduced here and a caller need not reduce it.
__device__ __forceinline__ T times_2_to(T x, int s) {
  T y{};
  for (int c = 0; c < EDEG; c++) set_coord(y, c, gf_times_2_to(coord(x, c), s % grm::TURNS));
  return y;
}
//   A[(i*Pd + k)*T_ + t] = sum_l gam^l limb_l(x[i][k][t]) over the xl (4 or 5) limbs, 0 for k >= d
//   HV[i*Pd + k] = sum_t eV[t] * A[(i*Pd + k)*T_ + t]^2, for the norm instances i < Lam and k < Pd
// x is indexed as in k_ws. The row stride is applied as << lt, so lt must be log2 T_. Requires
// blocks of 256 threads. The limbs are taken plus 128, unsigned for F::smac, which adds a limb's
// multiple of a power of gam at a sixth of the cost of a full product of two field elements, with
// one reduction for all limbs; `shift`, 128 times the sum of the powers of gam, removes the 128.
// prover/tests/base_kernels_test.cu compares A and HV byte for byte with what full products give.
__global__ void k_bg(int xl, const long long *xst, int T_, int lt, int d, int Pd, int Lam, const T *gam, const T *eV,
                     T *A, T *HV) {
  __shared__ T sh[8];
  const T g1 = gam[0], g2 = F::mul(g1, g1), g3 = F::mul(g2, g1), g4 = F::mul(g2, g2);
  const F::SP p1 = F::sprep(g1), p2 = F::sprep(g2), p3 = F::sprep(g3), p4 = F::sprep(g4);
  T powers = F::add(F::add(F::one(), g1), F::add(g2, g3));
  if (xl == 5) powers = F::add(powers, g4);
  const T shift = times_2_to(powers, 7);
  for (size_t q = blockIdx.x; q < (size_t)Lam * Pd; q += gridDim.x) {
    const int i = (int)(q / Pd), k = (int)(q % Pd);
    T acc = F::zero();
    for (int t = threadIdx.x; t < T_; t += 256) {
      T b = F::zero();
      if (k < d) {
        long long v = xst[((size_t)i * d + k) * T_ + t];
        u32 l[5]; // the limbs plus 128
        for (int c = 0; c < 5; c++) { // with four limbs the fifth is not read
          l[c] = (u32)((v + 128) & 255);
          v = (v - ((long long)l[c] - 128)) >> 8;
        }
        F::SAcc s = F::szero();
        s.l[0] = l[0]; // the limb of gam^0 = 1
        F::smac(s, p1, l[1]);
        F::smac(s, p2, l[2]);
        F::smac(s, p3, l[3]);
        if (xl == 5) F::smac(s, p4, l[4]);
        b = F::sub(F::sred(s), shift);
        acc = F::add(acc, F::mul(eV[t], F::mul(b, b)));
      }
      A[(q << lt) + t] = b;
    }
    for (int s = 16; s; s >>= 1) {
      acc = F::add(acc, shfl_down<F>(acc, s));
    }
    if ((threadIdx.x & 31) == 0) {
      sh[threadIdx.x >> 5] = acc;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
      T v = F::zero();
      for (int w = 0; w < 8; w++) {
        v = F::add(v, sh[w]);
      }
      HV[q] = v;
    }
    __syncthreads();
  }
}
// Writes the factors of sigma^2 and R sigma as rows of Fv and Gv: 7 pairs per norm instance if
// xl == 5, else 3. sg holds sigma, a norm's integer square root; Rv holds R, fixed-point 1 / sigma.
// Public source: operands written out, duplicated, for one general routine: S. Chetlur, C. Woolley,
// P. Vandermersch, J. Cohen, J. Tran, B. Catanzaro, E. Shelhamer, "cuDNN: Efficient Primitives for
// Deep Learning", 2014, section 3 (Implementation), https://arxiv.org/abs/1410.0759; there a
// convolution's input is written out for matrix multiplication, here the factors are written out in
// the same way for the token sumcheck, Tsc::prove() (prover/hiding/token_sumcheck.cuh).
// The idea is public: thread coarsening: Wen-mei W. Hwu, David Kirk, "Advanced Algorithmic
// Techniques for GPUs, Lecture 4: Thread Coarsening and more on Tiling/Blocking", PASI Summer
// School, Chile, January 5-7, 2011, slide "Outline of Technique",
// https://www.bu.edu/pasi/files/2011/01/WenMeiHwu4-06-1600.pdf.
__global__ void k_rowterms(int xl, const u64 *Rv, const u64 *sg, int n /* Lam * T_ */, int T_, T *Fv, T *Gv) {
  GSL(x, (size_t)n) {
    const int i = (int)(x / T_), t = (int)(x % T_);
    const u64 R = Rv[x];
    if (xl == 5) { // a a, a b, b b, Rlo a, Rlo b, Rhi a, Rhi b, with sigma = a + 2^24 b
      const T a = ext_base(sg[x] & 0xFFFFFF), b = ext_base(sg[x] >> 24), lo = ext_base(R & 0xFFFFFF), hi = ext_base(R >> 24);
      const T left[7] = {a, a, b, lo, lo, hi, hi}, right[7] = {a, b, b, a, b, a, b};
      for (int c = 0; c < 7; c++) {
        Fv[((size_t)7 * i + c) * T_ + t] = left[c];
        Gv[((size_t)7 * i + c) * T_ + t] = right[c];
      }
      continue;
    }
    const T s = ext_base(sg[x]);
    Fv[((size_t)3 * i) * T_ + t] = s;
    Fv[((size_t)3 * i + 1) * T_ + t] = ext_base(R & 0xFFFFFF);
    Fv[((size_t)3 * i + 2) * T_ + t] = ext_base(R >> 24);
    for (int c = 0; c < 3; c++) {
      Gv[((size_t)3 * i + c) * T_ + t] = s;
    }
  }
}
// pv[q] = sum_t eV[t] Fv[q][t] Gv[q][t], masked as in k_tsum(): pv must point into the sub-proof's
// message array, and the sum is stored with its slot's mask entry added. Fv and Gv hold, as [q][t],
// the two factors of term q of a token sumcheck. Requires one block of 256 threads for each q.
// The idea is public: block reduction by warp shuffles and shared memory: J. Luitjens, "Faster
// Parallel Reductions on Kepler" (cited at k_rows()).
// The loop form is public: the block-stride loop: CUDA Programming Guide (cited at k_rows()).
__global__ void k_pvsum(int T_, const T *eV, const T *Fv, const T *Gv, T *pv) {
  __shared__ T sh[8];
  const int q = blockIdx.x;
  T a = F::zero();
  for (int t = threadIdx.x; t < T_; t += 256) {
    a = F::add(a, F::mul(eV[t], F::mul(Fv[(size_t)q * T_ + t], Gv[(size_t)q * T_ + t])));
  }
  for (int s = 16; s; s >>= 1) {
    a = F::add(a, shfl_down<F>(a, s));
  }
  if ((threadIdx.x & 31) == 0) {
    sh[threadIdx.x >> 5] = a;
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    T v = F::zero();
    for (int w = 0; w < 8; w++) {
      v = F::add(v, sh[w]);
    }
    sc_door::claim(sc_door::Site::TOKEN_SUMS, v, pv + q);
  }
}
// Gv[i] = the sum over z < nz of part[z n + i], k_gn1's sums; Fv[i] = Rv[i] mod p. Fv and Gv are
// the two factors of a norm's term of the token sumcheck; Rv holds R (see k_rowterms). The two
// computations are independent; they share a loop because they have the same index range.
// The idea is public: loop fusion: Wikipedia, "Loop fission and fusion",
// https://en.wikipedia.org/wiki/Loop_fission_and_fusion.
// The idea is public: fusion of kernels that index data the same way: J. Filipovic, M. Madzin,
// J. Fousek, L. Matyska, "Optimizing CUDA Code by Kernel Fusion - Application on BLAS",
// arXiv:1305.1183, section 3.1, https://arxiv.org/pdf/1305.1183.
__global__ void k_gn2(const T *part, int nz, const u64 *Rv, size_t n, T *Fv, T *Gv) {
  GSL(i, n) {
    T a = F::zero();
    for (int z = 0; z < nz; z++) {
      a = F::add(a, part[(size_t)z * n + i]);
    }
    Gv[i] = a;
    Fv[i] = ext_base(Rv[i] % FIELD_P);
  }
}
__global__ void k_rowadd(T *a, const T *b, size_t n) {
  GSL(i, n) {
    a[i] = F::add(a[i], b[i]);
  }
}
// For N = 1 to 9 vectors phi[q] of length L: out[q][r] = sum_y phi[q][y] src[r L + y], r < rows.
struct RowJob {
  int N = 0;
  const i8 *src;
  size_t L, rows;
  T *phi[9], *out[9];
  const T **dphi;
  T **dout;
  T *ps;
  void prep() {
    CK(cudaMalloc(&dphi, 9 * 8));
    CK(cudaMalloc(&dout, 9 * 8));
    CK(cudaMalloc(&ps, 9 * sizeof(T)));
    CK(cudaMemcpy(dphi, phi, N * 8, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dout, out, N * 8, cudaMemcpyHostToDevice));
  }
  template <int NQ> void go(void *rsc) {
    // row_core_i8() (prover/common/row_pass_core.cuh) takes rows of any length K that is a multiple
    // of 16, up to 65,536. A longer row whose length is a multiple of 65,536 is passed in pieces of
    // 65,536, and the pieces' sums are added in the field, which gives the same values. Any other L
    // uses k_rows, k_phisum and k_rowfix. rows must fit an int (cast, not checked).
    const bool fast = L % 16 == 0 && (L <= 65536 || L % 65536 == 0);
    if (fast) {
      const size_t K = piece();
      T *later = (T *)rsc;
      char *rs = (char *)rsc + later_bytes();
      for (size_t s = 0; s * K < L; s++) {
        const T *ph[NQ];
        T *o[NQ];
        for (int q = 0; q < NQ; q++) {
          ph[q] = phi[q] + s * K;
          o[q] = s ? later + (size_t)q * rows : out[q];
        }
        row_core_i8<NQ>(src + s * K, L, (int)rows, (int)K, ph, o, rs);
        if (s)
          for (int q = 0; q < NQ; q++) k_rowadd<<<nblk(rows), 256>>>(out[q], o[q], rows);
      }
    } else {
      k_rows<NQ><<<(unsigned)std::min<size_t>(rows, 8192), 256>>>(src, L, rows, dphi, dout);
      k_phisum<NQ><<<NQ, 256>>>(L, dphi, ps);
      k_rowfix<NQ><<<nblk(NQ * rows), 256>>>(rows, dout, ps);
    }
  }
  // rsc: scratch() bytes of GPU memory. Call prep() first; RowJob never frees prep()'s buffers.
  void run(void *rsc) {
    switch (N) {
    case 1:
      go<1>(rsc);
      break;
    case 2:
      go<2>(rsc);
      break;
    case 3:
      go<3>(rsc);
      break;
    case 4:
      go<4>(rsc);
      break;
    case 5:
      go<5>(rsc);
      break;
    case 6:
      go<6>(rsc);
      break;
    case 7:
      go<7>(rsc);
      break;
    case 8:
      go<8>(rsc);
      break;
    case 9:
      go<9>(rsc);
      break;
    default:
      printf("internal error: the row job size N = %d does not match any compiled kernel\n", N);
      exit(1);
    }
  }
  size_t piece() const { return L < 65536 ? L : 65536; }
  // Bytes for the sums of a long row's later pieces; they precede row_core_i8()'s scratch memory.
  size_t later_bytes() const { return L > 65536 ? ((size_t)N * rows * sizeof(T) + 255) / 256 * 256 : 0; }
  size_t scratch() const { return later_bytes() + row_core_scratch(N, (int)piece(), (int)rows); }
};
struct SecD {
  std::string name;
  const void *p;
  size_t words;
};
static double now() {
  return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}
// ---- RoPE's token sumcheck terms ----
// Coefficient vectors of layer lam, for even dh (not checked). Cm[(2 i + e) d + kk] is, for e = 0,
// lf[2i] bq[kk][2i] + lf[2i+1] bq[kk][2i+1]; for e = 1, lf[2i+1] bq[kk][2i] - lf[2i] bq[kk][2i+1].
// bq[kk][c] = BQ[rest] + 256 BQ[n1 + rest], rest = (lam d + kk) dh + c. BQ: see B at k_bjoin.
__global__ void k_bcs(const T *BQ, size_t n1, int lam, int d, int dh, const T *lf, T *Cm) {
  GSL(x, (size_t)d * (dh / 2)) {
    const int kk = (int)(x % d), i = (int)(x / d);
    const size_t r = ((size_t)lam * d + kk) * dh + 2 * i;
    const T b0 = F::add(BQ[r], F::mul(BQ[n1 + r], ext_base(256))),
            b1 = F::add(BQ[r + 1], F::mul(BQ[n1 + r + 1], ext_base(256)));
    Cm[((size_t)2 * i) * d + kk] = F::add(F::mul(lf[2 * i], b0), F::mul(lf[2 * i + 1], b1));
    Cm[((size_t)2 * i + 1) * d + kk] = F::sub(F::mul(lf[2 * i + 1], b0), F::mul(lf[2 * i], b1));
  }
}
// k_limb9: Bl[((v EDEG + comp) 8 + lb) d + kk] = limb lb < 8 of coordinate comp of Cm[v d + kk].
// Modulo p, the coordinate is the sum over lb of 256^lb times limb lb.
// Public source: elements cut into 8-bit integers: Fan et al., "TensorFHE" (arXiv:2212.14191),
// IV-C, stage 1. Those integers are unsigned; the digits here are signed.
// Public source: a digit s_i >= L/2 becomes s_i - L and carries 1: drouyang, "Signed Bucket Indexes
// for Multi-Scalar Multiplication (MSM)" (https://hackmd.io/@drouyang/signed-bucket-index). Here
// L = 256. A coordinate x too large for eight signed bytes is cut as x - p (grm::signed_digits()).
// The loop form is public: the grid-stride loop, cited at GSL.
// If d is a multiple of 4 and Bl is 4-byte aligned, a thread takes four neighbouring kk of one v
// and stores their limbs as 24 words (grm::put_digits(), prover/hiding/gram_products.cuh);
// otherwise a thread writes one byte at a time. A launch of any size works; limb9_blocks(n) blocks
// of 256 threads give one thread to every four entries.
static inline unsigned limb9_blocks(size_t n) {
  return (unsigned)std::max<size_t>(1, (n / 4 + 255) / 256);
}
__global__ void k_limb9(const T *Cm, size_t n /* V d */, int d, i8 *Bl) {
  static_assert(EDEG == 3, "grm::put_digits writes the 24 rows of three coordinates");
  if (d % 4 == 0 && (uintptr_t)Bl % 4 == 0) {
    GSL(x, n / 4) {
      const size_t at = 4 * x;
      const T e[4] = {Cm[at], Cm[at + 1], Cm[at + 2], Cm[at + 3]};
      grm::put_digits(Bl + at % d, (size_t)d, (int)(at / d), e);
    }
    return;
  }
  GSL(x, n * EDEG * 8) {
    const size_t kk = x % d, row = x / d, v = row / (8 * EDEG);
    const int lb = (int)(row % 8), comp = (int)(row / 8 % EDEG);
    // The same limbs as in the branch above. A loop that carries from byte to byte would give a
    // coordinate x above 0x7F7F7F7F7F7F7F7F the limbs of x - 2^64, which is not x modulo p.
    Bl[x] = (i8)(grm::signed_digits(coord(Cm[v * d + kk], comp)) >> (8 * lb));
  }
}
// G[v][t] coordinate comp = sum_lb 256^lb (C0 + 256 C1)[((v EDEG + comp) 8 + lb)][t] mod p.
// C_i is the product of k_limb9's rows of limbs with limb i of the caller's integer matrix. Each
// thread writes one element of G, so a launch needs at least V T_ threads. The sum over lb is
// evaluated by Horner's rule, from lb = 7 down, and reduced modulo p at every step.
// Public source: one thread per element: NVIDIA, CUDA C++ Programming Guide, "Thread Hierarchy".
// Public source: 32-bit partial sums fused, then reduced: "TensorFHE", IV-C, stages 3 and 5 (cited
// at limb9_blocks()).
// Public source: Horner's rule: Wikipedia, "Horner's method",
// https://en.wikipedia.org/wiki/Horner%27s_method.
// The idea is public: a reduction at every step of Horner's rule: the page "3.4 Hash Tables",
// paragraph "Strings", https://algs4.cs.princeton.edu/34hash/.
// The idea is public: the same in base 256, modulo a prime: Wikipedia, "Rabin-Karp algorithm",
// https://en.wikipedia.org/wiki/Rabin%E2%80%93Karp_algorithm.
__global__ void k_recomb(const int *C0, const int *C1, int V, int T_, T *G) {
  const size_t x = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  if (x < (size_t)V * T_) {
    const size_t t = x % T_, v = x / T_;
    T g{};
    for (int comp = 0; comp < EDEG; comp++) {
      u64 sum = 0;
      for (int lb = 7; lb >= 0; lb--) {
        const size_t at = ((v * EDEG + comp) * 8 + lb) * T_ + t;
        const long long s = C0[at] + 256ll * C1[at];
        sum = gf_add(gf_mul(sum, 256), s < 0 ? FIELD_P - (u64)(-s) : (u64)s);
      }
      set_coord(g, comp, sum);
    }
    G[x] = g;
  }
}
// Fv[(2 i + e) T_ + t] = Cr[t hd + i] at e = 0, Sr[t hd + i] at e = 1: cosine and sine of token t's
// rotation i < hd, where 4096 is 1. A layer without rotary encoding (rope == 0) takes 4096 and 0.
// Public source: the identity rotation where nothing is rotated: The MaxText authors, MaxText
// documentation (release v0.2.2), API reference, page 'maxtext.layers.embeddings module', class
// Gemma4PartialRotaryEmbedding (there for some dimensions, here for a whole layer);
// https://maxtext.readthedocs.io/en/maxtext-v0.2.2/reference/api_generated/maxtext.layers.embeddings.html.
__global__ void k_ropeF(const short *Cr, const short *Sr, int rope, int T_, int hd, T *Fv) {
  GSL(x, (size_t)2 * hd * T_) {
    const int t = (int)(x % T_), v = (int)(x / T_), i = v >> 1, e = v & 1;
    int c = rope ? (e ? Sr : Cr)[(size_t)t * hd + i] : (e ? 0 : 4096);
    Fv[x] = ext_from_signed(c);
  }
}
__global__ void k_scale(const T *el, int lam, int sh, T *co, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    co[i] = F::mul(el[lam], ext_base(1ull << sh));
  }
}
// Hh[q] = sum_t et[t] (KH[row][t] - 32768) (KH[Lam Pf + row][t] - 32768), row = (q / f) Pf + q % f
// KH holds s + 2^15 for Lam layers, then u + 2^15: s is the SiLU table's entry, u the up projection
// rounded to nearest. Requires blocks of whole warps. Public source for the sum over a warp's lanes
// by shuffles: J. Luitjens, "Faster Parallel Reductions on Kepler" (cited at k_rows()).
// A warp takes ROWS rows at once, so that it reads each et[t] once for all of them. The product
// s * u is from -2^30 + 2^15 to 2^30. Plus 2^30 it is unsigned and is split into its 16 low bits
// and the rest; F::smac adds each part times et[t], cheaper than a full product (see k_bg).
// 2^30 times the sum of et is subtracted at the end. Hh is byte for byte what full products give
// (tested by prover/tests/base_kernels_test.cu). F::SAcc does not overflow while (terms) x (largest
// part) <= 2^32 (prover/hiding/pt_field.h): a lane adds about NT / 32 terms, the parts are below
// 2^16, and the parameter file has NT <= 2^16, which gives 2^27.
template <int ROWS>
__device__ __forceinline__ void hh3_rows(const u16 *KH, size_t NT, int Lam, int f, int Pf, const T *et, T *Hh) {
  const size_t half = (size_t)Lam * Pf, all = (size_t)Lam * f;
  const int lane = threadIdx.x & 31;
  const size_t warp = blockIdx.x * (size_t)(blockDim.x >> 5) + (threadIdx.x >> 5),
               warps = (size_t)gridDim.x * (blockDim.x >> 5);
  const bool fours = NT % 4 == 0 && (uintptr_t)KH % 8 == 0;
  for (size_t q0 = warp * ROWS; q0 < all; q0 += warps * ROWS) {
    const u16 *a[ROWS];
#pragma unroll
    for (int r = 0; r < ROWS; r++) {
      const size_t q = min(q0 + r, all - 1); // a row past the last repeats the last; not written
      a[r] = KH + ((q / f) * Pf + q % f) * NT;
    }
    F::SAcc low[ROWS], high[ROWS];
#pragma unroll
    for (int r = 0; r < ROWS; r++) low[r] = high[r] = F::szero();
    T sum = F::zero(); // of the lane's et[t]
    auto term = [&](int r, const F::SP &e, u32 x, u32 y) {
      const u32 w = (u32)(((int)x - 32768) * ((int)y - 32768)) + (1u << 30); // <= 2^31: fits a u32
      F::smac(low[r], e, w & 65535);
      F::smac(high[r], e, w >> 16);
    };
    if (fours) {
      for (size_t t = 4 * lane; t < NT; t += 128) {
        uint2 x[ROWS], y[ROWS];
#pragma unroll
        for (int r = 0; r < ROWS; r++) x[r] = *(const uint2 *)(a[r] + t), y[r] = *(const uint2 *)(a[r] + half * NT + t);
#pragma unroll
        for (int i = 0; i < 4; i++) {
          const T e = et[t + i];
          const F::SP s = F::sprep(e);
          sum = F::add(sum, e);
#pragma unroll
          for (int r = 0; r < ROWS; r++)
            term(r, s, ((i < 2 ? x[r].x : x[r].y) >> 16 * (i & 1)) & 65535,
                 ((i < 2 ? y[r].x : y[r].y) >> 16 * (i & 1)) & 65535);
        }
      }
    } else {
      for (size_t t = lane; t < NT; t += 32) {
        const T e = et[t];
        const F::SP s = F::sprep(e);
        sum = F::add(sum, e);
#pragma unroll
        for (int r = 0; r < ROWS; r++) term(r, s, a[r][t], a[r][half * NT + t]);
      }
    }
    const T less = times_2_to(sum, 30);
#pragma unroll
    for (int r = 0; r < ROWS; r++) {
      T v = F::sub(F::add(F::sred(low[r]), times_2_to(F::sred(high[r]), 16)), less);
      for (int o = 16; o; o >>= 1) v = F::add(v, shfl_down<F>(v, o));
      if (lane == 0 && q0 + r < all) Hh[q0 + r] = v;
    }
  }
}
__global__ void k_hh3(const u16 *KH, size_t NT, int Lam, int f, int Pf, const T *et, T *Hh) {
  hh3_rows<2>(KH, NT, Lam, f, Pf, et, Hh);
}
__global__ void k_gatherh(const T *e, Pos2 p, int dh, int nh, T *o) {
  int h = threadIdx.x;
  if (h < 64) {
    o[h] = h < nh ? e[p.at(h * dh) / dh] : F::zero();
  }
}
// For a norm's term of a token sumcheck: part[z T_ + t] = the sum of ew[k] x[k][t] over the z-th
// of gridDim.z consecutive ranges of k < d. k_gn2 adds these sums. The launch's x dimension must
// cover all t < T_. x[k][t] plus 2^63 is unsigned and is split into 24, 24 and 16 bits for F::smac
// (see k_bg); 2^63 times the sum of ew is subtracted at the end. The sums are reduced after at most
// 256 terms: 256 x (2^24 - 1) is just within F::SAcc's bound of 2^32. The loop is unrolled so that
// a thread can have eight loads in flight. The result is byte for byte what full products give
// (tested by prover/tests/base_kernels_test.cu). The callers' `part` has space for 32 ranges only,
// and one thread for each t and range is too few to fill a GPU. So a block may have several rows
// of threads: blockDim.y of them, at most four, of at most 128 threads each (not checked). A
// block's rows share its range of k; row 0 adds the others' sums from shared memory.
// times_2_to, k_bg, k_limb9, limb9_blocks, hh3_rows, k_hh3, k_gn1: Ours, with what the tree has:
// F::smac, F::sred (hiding/pt_field.h; the offset as in k_rows of prover/base.cuh), gf_times_2_to
// (hiding/rotation.h), grm::put_digits (hiding/gram_products.cuh; called, not copied).
__global__ void __launch_bounds__(512) k_gn1(const long long *x, int T_, int d, const T *ew, T *part) {
  __shared__ T others[3][128];
  const int t = blockIdx.x * blockDim.x + threadIdx.x, per = (d + gridDim.z - 1) / gridDim.z,
            end = min(d, (int)(blockIdx.z + 1) * per), share = (per + blockDim.y - 1) / blockDim.y,
            k0 = blockIdx.z * per + threadIdx.y * share, k1 = t < T_ ? min(end, k0 + share) : k0;
  T a = F::zero(), sum = F::zero(); // sum: of the ew[k]
  for (int from = k0; from < k1; from += 256) {
    F::SAcc part0 = F::szero(), part1 = F::szero(), part2 = F::szero();
#pragma unroll 8
    for (int k = from; k < min(k1, from + 256); k++) {
      const u64 v = (u64)x[(size_t)k * T_ + t] + (1ull << 63);
      const T e = ew[k];
      const F::SP s = F::sprep(e);
      sum = F::add(sum, e);
      F::smac(part0, s, (u32)v & 0xFFFFFF);
      F::smac(part1, s, (u32)(v >> 24) & 0xFFFFFF);
      F::smac(part2, s, (u32)(v >> 48));
    }
    a = F::add(a, F::add(F::sred(part0), F::add(times_2_to(F::sred(part1), 24), times_2_to(F::sred(part2), 48))));
  }
  a = F::sub(a, times_2_to(sum, 63)); // each v above is x + 2^63
  if (threadIdx.y) others[threadIdx.y - 1][threadIdx.x] = a;
  __syncthreads();
  if (!threadIdx.y && t < T_) {
    for (int y = 1; y < (int)blockDim.y; y++) a = F::add(a, others[y - 1][threadIdx.x]);
    part[(size_t)blockIdx.z * T_ + t] = a;
  }
  __syncthreads();
  // Zeroes the sums in shared memory: what a block leaves there cannot be wiped after the kernel's
  // end. Nothing reads the zeros, so the stores are volatile: the compiler may not leave them out.
  if (threadIdx.y)
    for (int c = 0; c < EDEG; c++) ((volatile u64 *)&others[threadIdx.y - 1][threadIdx.x])[c] = 0;
}
__global__ void k_gather2o(const T *e, Pos2 p, int off, int n, T *o) {
  GSL(x, (size_t)n) o[x] = e[p.at(off + (int)x)];
}
// Cm[x] = B[base + x] + 256 B[n1 + base + x]. A weight is committed as two limbs (specific to this
// proof system): B holds weighted sums of the low limbs and, n1 elements later, of the high limbs.
__global__ void k_bjoin(const T *B, size_t n1, size_t base, size_t n, T *Cm) {
  GSL(x, n) Cm[x] = F::add(B[base + x], F::mul(B[n1 + base + x], ext_base(256)));
}
// G[v][t] coordinate comp = sum_lb 256^lb (sum_i 256^i C_i)[((v EDEG + comp) 8 + lb)][t] mod p.
// This is k_recomb for an integer matrix of four limbs (np == 4: C0 to C3) or, at any other np, of
// two limbs (C0, C1; C2 and C3 are not read).
// The public sources and the number of threads are the same as for k_recomb.
__global__ void k_recombN(const int *C0, const int *C1, const int *C2, const int *C3, int np, int V, int T_, T *G) {
  const size_t x = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  if (x < (size_t)V * T_) {
    const size_t t = x % T_, v = x / T_;
    T g{};
    for (int comp = 0; comp < EDEG; comp++) {
      u64 sum = 0;
      for (int lb = 7; lb >= 0; lb--) {
        const size_t at = ((v * EDEG + comp) * 8 + lb) * T_ + t;
        long long s = C0[at] + 256ll * C1[at];
        if (np == 4) {
          s += 65536ll * C2[at] + 16777216ll * C3[at];
        }
        sum = gf_add(gf_mul(sum, 256), s < 0 ? FIELD_P - (u64)(-s) : (u64)s);
      }
      set_coord(g, comp, sum);
    }
    G[x] = g;
  }
}
__global__ void k_selF(const int *es, int T_, int NE, T *Fv) {
  GSL(x, (size_t)NE * T_) Fv[x] = es[x % T_] == (int)(x / T_) ? F::one() : F::zero();
}
// o[i] = sum_{l < nl} 256^l c[l st + i], i < n. c: cells (committed bytes). nl <= 7 (not checked).
__global__ void k_cellv(const i8 *c, size_t st, int nl, size_t n, T *o) {
  GSL(i, n) {
    long long v = 0;
    for (int l = nl - 1; l >= 0; l--) {
      v = v * 256 + c[l * st + i];
    }
    o[i] = ext_from_signed(v);
  }
}
// k_gn1's sums for x[k][t] = c[k T_ + t] + 256 c[st + k T_ + t], two cells. k_gsum adds them.
__global__ void k_gn1c(const i8 *c, size_t st, int T_, int d, const T *ew, T *part) {
  const int t = blockIdx.x * blockDim.x + threadIdx.x, per = (d + gridDim.z - 1) / gridDim.z, k0 = blockIdx.z * per,
            k1 = min(d, k0 + per);
  if (t >= T_) {
    return;
  }
  T a = F::zero();
  for (int k = k0; k < k1; k++) {
    a = F::add(a, F::mul(ew[k], ext_from_signed((int)c[(size_t)k * T_ + t] + 256 * (int)c[st + (size_t)k * T_ + t])));
  }
  part[(size_t)blockIdx.z * T_ + t] = a;
}
__global__ void k_gsum(const T *part, int nz, size_t n, T *Gv) {
  GSL(i, n) {
    T a = F::zero();
    for (int z = 0; z < nz; z++) {
      a = F::add(a, part[(size_t)z * n + i]);
    }
    Gv[i] = a;
  }
}
// co[i] = el[lam] sum_{x < n} e[p.at(x)] for i < m. Pos2::at() (prover/prover/forward.cuh) maps an
// index of a lookup segment to its column among its call's keys. Requires one block of 256 threads.
// The idea is public: block reduction by warp shuffles and shared memory: J. Luitjens, "Faster
// Parallel Reductions on Kepler" (cited at k_rows()).
// The loop form is public: the block-stride loop: CUDA Programming Guide (cited at k_rows()).
__global__ void k_scalew(const T *el, int lam, const T *e, Pos2 p, int n, T *co, int m) {
  __shared__ T sh[8];
  T a = F::zero();
  for (int x = threadIdx.x; x < n; x += 256) {
    a = F::add(a, e[p.at(x)]);
  }
  for (int s = 16; s; s >>= 1) {
    a = F::add(a, shfl_down<F>(a, s));
  }
  if ((threadIdx.x & 31) == 0) {
    sh[threadIdx.x >> 5] = a;
  }
  __syncthreads();
  __shared__ T w;
  if (threadIdx.x == 0) {
    T v = F::zero();
    for (int q = 0; q < 8; q++) {
      v = F::add(v, sh[q]);
    }
    w = F::mul(v, el[lam]);
  }
  __syncthreads();
  for (int i = threadIdx.x; i < m; i += 256) {
    co[i] = w;
  }
}
// h = s * u, the routed expert's gated product. KHs and KHu hold s + 2^15 and u + 2^15 as [fe][T_];
// h0 to h3 get the four limbs of h as [T_][fe]. A launch needs blockDim = (32, 8) and
// gridDim = (T_ / 32, fe / 32); 32 must divide T_ and fe.
// The idea is public: a transpose through a padded tile in shared memory: M. Harris, "An Efficient
// Matrix Transpose in CUDA C/C++", NVIDIA Technical Blog, 2013,
// https://developer.nvidia.com/blog/efficient-matrix-transpose-cuda-cc/.
__global__ void k_hl4(const u16 *KHs, const u16 *KHu, int T_, int fe, i8 *h0, i8 *h1, i8 *h2, i8 *h3) {
  __shared__ int tile[32][33];
  const int t0 = blockIdx.x * 32, k0 = blockIdx.y * 32;
  for (int r = threadIdx.y; r < 32; r += 8) {
    const size_t a = (size_t)(k0 + r) * T_ + t0 + threadIdx.x;
    tile[r][threadIdx.x] = ((int)KHs[a] - 32768) * ((int)KHu[a] - 32768);
  }
  __syncthreads();
  for (int r = threadIdx.y; r < 32; r += 8) {
    long long v = tile[threadIdx.x][r];
    const size_t o = (size_t)(t0 + r) * fe + k0 + threadIdx.x;
    i8 *h[4] = {h0, h1, h2, h3};
    for (int l = 0; l < 4; l++) {
      const int b = (int)(((v + 128) & 255) - 128);
      h[l][o] = (i8)b;
      v = (v - b) >> 8;
    }
  }
}
// Gathers the factors of the routed experts' gated products for k_hh3: the first half of o holds
// s + 2^15, the second half u + 2^15. Each half is indexed (lm * fe + kk) * T_ + t, where lm < LamM
// counts the layers with routed experts: layer 2 lm + 1 of KH, in which rows fe to 2 fe - 1 are the
// routed expert's and the rows below fe the shared expert's.
// The idea is public: loop fusion: Wikipedia, "Loop fission and fusion" (cited at k_gn2()).
// The idea is public: kernel fusion: Filipovic and others, "Optimizing CUDA Code by Kernel Fusion",
// section 3.1 (cited at k_gn2()).
__global__ void k_kh2(const u16 *KH, int T_, int Lam, int Pf, int LamM, int fe, u16 *o) {
  const size_t n = (size_t)LamM * fe * T_;
  GSL(i, n) {
    const size_t t = i % T_, r = i / T_, kk = r % fe, lm = r / fe;
    o[i] = KH[((2 * lm + 1) * Pf + fe + kk) * T_ + t];
    o[n + i] = KH[(((size_t)Lam + 2 * lm + 1) * Pf + fe + kk) * T_ + t];
  }
}
