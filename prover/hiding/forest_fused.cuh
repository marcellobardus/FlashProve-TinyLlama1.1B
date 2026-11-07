// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Gram passes for the lookup forest that keep their digit matrices on the multiprocessor.
//
// The lookup forest (prover/hiding/forest.cuh) is the product trees of a group of lookup calls. A
// Gram pass computes the 64 weighted inner products of a layer's 8 left and 8 right parts (slabs).
// prover/hiding/gram_products.cuh (namespace grm) defines it. An entry is a slab's element at one
// index and has 24 byte digits; a chunk is a run of indices whose weights share a factor; a piece
// is the part of a chunk that one 32-bit sum covers. In grm, one kernel writes two matrices of
// digits, a column per index, to GPU memory, a library multiplies them, and another kernel joins
// the 32-bit sums into field elements. Written and read once, matrices of 192 rows cost 768 bytes
// of traffic per index, but a pass reads 384 bytes of stored entries or as little as 32 bytes of
// lookup keys. Here one kernel does the first two steps: a thread block computes the digits of 128
// indices (a tile) in shared memory, multiplies the tiles on the tensor cores, and keeps the sums
// in registers until a piece ends. k_pass writes them in grm::product()'s layout for the joining
// kernels; k_stored joins them itself. All sums are exact, so both methods give the same elements.
//
// Public sources. The titles and addresses of the works cited below are unchecked. The same holds
// for the works cited in the other files prover/hiding/forest_fused_*.cuh.
// [mma] NVIDIA, Parallel Thread Execution ISA, "Warp-level matrix instructions", "Matrix Fragments for mma.m16n8k32": the
//   instruction mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 and which thread of a warp holds which entries.
// [ldmatrix] the same document, "Warp-level matrix load instruction: ldmatrix".
// [CUTLASS] NVIDIA, CUTLASS documentation, "Efficient GEMM in CUDA",
//   https://docs.nvidia.com/cutlass/media/docs/cpp/efficient_gemm.html: a block's tile is staged in shared memory, each warp
//   computes a part of it from fragments in registers, and the accumulators stay in registers over the loop along the
//   summed dimension; in the scheme named sliced K, warps of a block share the summed dimension.
// [Padding] M. Harris, "An Efficient Matrix Transpose in CUDA C/C++", NVIDIA Technical Blog,
//   https://developer.nvidia.com/blog/efficient-matrix-transpose-cuda-cc/: a padded row length in shared memory removes bank
//   conflicts between rows.
// Making an operand of a matrix product inside the kernel that multiplies it is public too: [Ma] (cited in
// hiding/gram_products.cuh) concatenates its 1-bit matrices in shared memory within the kernel.
// Include after prover/hiding/forest_keys.cuh (namespace fk).
#pragma once
#include "common/end_name.h" // name_the_end(): names an internal stop before a direct exit (M5e3)
#include <cuda_pipeline.h>
// ff: the lookup forest's passes and folds on the tensor cores (prover/hiding/forest_fused*.cuh).
namespace ff {
typedef grm::i8 i8;
constexpr int COLS = 128;     // the indices of a tile
constexpr int LD = COLS + 16; // the bytes from one row of a tile to the next: 36 words, so that
                              // eight rows' words at one column lie in eight different groups of
                              // four banks [Padding]
constexpr int THREADS = 256;  // eight warps; here warp a computes the digits of slabs a and 8 + a
// ---- Bounds on the integer sums, for all files forest_fused*.cuh.
// The tensor cores add products of bytes into 32-bit integers and do not detect overflow. So a
// static_assert, here or at its kernel, bounds each such sum for any keys, tables, weights and
// challenges. Where a caller sets a sum's length, the launching function exits if it is too long.
// grm::signed_digits() gives bytes in [-128, 127], and so does lowering an unsigned byte by 128.
constexpr long long DIGIT_MOST = 128;        // |a signed digit|
constexpr long long BYTE_MOST = 255;         // an unsigned byte
constexpr long long INT_MOST = 2147483647;   // a 32-bit sum
constexpr long long TURN_MOST = 1ll << 61;   // `grm::signed_turn` needs |v| below this
constexpr int PIECE_MOST = 65536;            // the largest piece of k_pass and k_stored, in indices
static_assert(DIGIT_MOST * DIGIT_MOST * PIECE_MOST <= INT_MOST, "a piece's sum of products of two signed digits fits 32 bits");
// How the eight warps share the product of a tile of 192 rows with a tile of NARROW rows. Warps w
// and w + 4, w < 4, take 48 rows of the first tile. They halve the second tile's rows or, if these
// are at most 40, the columns: each half's sums are then a separate piece for the joining kernels.
template <int NARROW> struct Tiles {
  static constexpr int NT = (NARROW + 7) / 8;   // blocks of eight rows of the second matrix
  static constexpr int WN = NT > 5 ? 2 : 1;     // warps along the second matrix's rows
  static constexpr int WK = 2 / WN;             // warps along the columns
  static constexpr int NW = (NT + WN - 1) / WN; // a warp's blocks of eight rows
  static constexpr int ROWS_B = 8 * NW * WN;    // the second tile's rows; those from NARROW on are
                                                // never written, and the sums made from them are
                                                // not stored
  static constexpr int SUMS = 12 * NW;          // a thread's sums
  static constexpr int BLOCKS = SUMS <= 48 ? 2 : 1; // blocks on a multiprocessor that the
                                                    // registers are budgeted for
  static constexpr size_t BYTES = (size_t)(192 + ROWS_B) * LD;
};
// Tiles<narrow>::WK, for host code: how many matrices of sums k_pass writes for each piece.
inline int split(int narrow) {
  return (narrow + 7) / 8 > 5 ? 1 : 2;
}
// d += a b^T: a has 16 rows and b has 8, each of 32 signed bytes; d has 16 x 8 sums [mma].
// The asm template and the order of its operands are the same as in ggml,
// ggml/src/ggml-cuda/mma.cuh (llama.cpp), MIT; that project's notice is in NOTICE.
__device__ __forceinline__ void multiply(int (&d)[4], const u32 (&a)[4], const u32 (&b)[2]) {
  asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"
               : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}
// Loads four (load_two(): two) blocks of 8 rows x 16 bytes from shared memory [ldmatrix]. Thread l
// gives the 16-byte aligned address of row l & 7 of block l >> 3 (load_two(): l < 16 only). Thread
// 4 g + t4, t4 < 4, gets bytes 4 t4 .. 4 t4 + 3 of row g of each block, as multiply() expects.
__device__ __forceinline__ void load_four(u32 (&r)[4], const i8 *mine) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
               : "r"((u32)__cvta_generic_to_shared(mine))
               : "memory");
}
__device__ __forceinline__ void load_two(u32 (&r)[2], const i8 *mine) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];"
               : "=r"(r[0]), "=r"(r[1])
               : "r"((u32)__cvta_generic_to_shared(mine))
               : "memory");
}
// sum[m][n] += (rows 16 m .. of A) x (rows 8 n .. of B) over 32 columns, m < 3, n < NW. A and B
// point to the warp's first row and the first of the columns, rows LD bytes apart.
template <int NW> __device__ __forceinline__ void multiply_step(int (&sum)[3][NW][4], const i8 *A, const i8 *B, int lane) {
  const int block = lane >> 3, row = lane & 7;
  u32 a[3][4];
#pragma unroll
  for (int m = 0; m < 3; m++) load_four(a[m], A + (m * 16 + (block & 1) * 8 + row) * LD + (block >> 1) * 16);
#pragma unroll
  for (int n = 0; n + 1 < NW; n += 2) {
    u32 b[4];
    load_four(b, B + ((n + (block >> 1)) * 8 + row) * LD + (block & 1) * 16);
    const u32 first[2] = {b[0], b[1]}, second[2] = {b[2], b[3]};
#pragma unroll
    for (int m = 0; m < 3; m++) multiply(sum[m][n], a[m], first), multiply(sum[m][n + 1], a[m], second);
  }
  if (NW & 1) {
    u32 b[2];
    load_two(b, B + ((NW - 1) * 8 + row) * LD + (block & 1) * 16);
#pragma unroll
    for (int m = 0; m < 3; m++) multiply(sum[m][NW - 1], a[m], b);
  }
}
__device__ __forceinline__ void put_word(i8 *p, u32 w) {
  *reinterpret_cast<u32 *>(p) = w;
}
// 'pieces' pieces of a Gram pass from index c0 on, each of k = 128 2^ltiles indices. P supplies:
//   P::NARROW, P::Own    the rows of the second matrix; a thread's state, passed as own;
//   p.shared()           (host) the bytes of shared memory that P needs after the tiles;
//   p.begin(own, room)   by all threads once, before a barrier; room: that memory, 16-byte aligned;
//   p.stage(own, c)      by all threads, before a barrier: copies what all warps read in make();
//   p.make(own, c, A, B) by each warp, no barrier inside: warp a writes the digits of slabs a and
//                        8 + a of indices c .. c + 127 to the tiles A and B, rows LD bytes apart.
// Computing digits and multiplying alternate. Overlapping them on two pairs of tiles, as k_stored
// does, was a few per cent faster for range tables and slower for function tables.
// C receives, for piece i and half h < WK, 192 x NARROW sums, column-major as from grm::product(),
// at (i WK + h) 192 NARROW. No sum overflows if k <= PIECE_MOST, which pass() checks.
// Needs blocks of THREADS threads, which the kernel does not check.
template <class P>
__global__ void __launch_bounds__(THREADS, Tiles<P::NARROW>::BLOCKS)
    k_pass(const P p, size_t c0, int pieces, int ltiles, int *__restrict__ C) {
  typedef Tiles<P::NARROW> X;
  extern __shared__ __align__(16) unsigned char room[];
  i8 *const digits = reinterpret_cast<i8 *>(room);
  typename P::Own own;
  p.begin(own, room + X::BYTES);
  __syncthreads();
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t4 = lane & 3;
  const int wm = warp & 3, wn = X::WN == 2 ? warp >> 2 : 0, wk = X::WK == 2 ? warp >> 2 : 0;
  const int tiles = ((pieces - (int)blockIdx.x + (int)gridDim.x - 1) / (int)gridDim.x) << ltiles;
  auto piece_of = [&](int t) { return (int)blockIdx.x + (t >> ltiles) * (int)gridDim.x; };
  auto index_of = [&](int t) { return c0 + (((size_t)piece_of(t) << ltiles) + (t & ((1 << ltiles) - 1))) * COLS; };
  int sum[3][X::NW][4];
  auto clear = [&] {
#pragma unroll
    for (int m = 0; m < 3; m++)
#pragma unroll
      for (int n = 0; n < X::NW; n++)
#pragma unroll
        for (int e = 0; e < 4; e++) sum[m][n][e] = 0;
  };
  auto make = [&](int t) {
    p.make(own, index_of(t), digits, digits + 192 * LD);
  };
  auto product = [&] {
    const i8 *mine_a = digits + wm * 48 * LD + wk * (COLS / X::WK), *mine_b = digits + (192 + wn * X::NW * 8) * LD + wk * (COLS / X::WK);
#pragma unroll
    for (int s = 0; s < COLS / 32 / X::WK; s++) multiply_step<X::NW>(sum, mine_a + 32 * s, mine_b + 32 * s, lane);
  };
  auto piece_ends = [&](int t) {
    if ((t & ((1 << ltiles) - 1)) != (1 << ltiles) - 1) return;
    int *out = C + ((size_t)piece_of(t) * X::WK + wk) * (192 * P::NARROW);
#pragma unroll
    for (int n = 0; n < X::NW; n++)
#pragma unroll
      for (int e = 0; e < 4; e++) {
        const int q = (wn * X::NW + n) * 8 + 2 * t4 + (e & 1);
        if (q < P::NARROW) {
#pragma unroll
          for (int m = 0; m < 3; m++) out[q * 192 + wm * 48 + m * 16 + g + 8 * (e >> 1)] = sum[m][n][e];
        }
      }
    clear();
  };
  clear();
  for (int t = 0; t < tiles; t++) {
    p.stage(own, index_of(t));
    __syncthreads();
    make(t);
    __syncthreads();
    product();
    piece_ends(t);
    __syncthreads();
  }
}
// Launches k_pass for n indices from c0 on, in pieces of k. A kernel's first launch with over 48 KB
// of shared memory raises the kernel's limit, which must not happen during a stream capture.
template <class P> inline void pass(const P &p, size_t c0, size_t n, int k, int *C, cudaStream_t s) {
  typedef Tiles<P::NARROW> X;
  auto kernel = k_pass<P>;
  const size_t bytes = X::BYTES + p.shared();
  int ltiles = 0;
  while ((COLS << ltiles) < k) ltiles++;
  if (bytes > 227 * 1024 || (COLS << ltiles) != k || n % k || k > PIECE_MOST) {
    printf("internal error: look-up forest: a pass on the multiprocessor with %zu bytes of shared memory, %zu indices in pieces of "
           "%d\n", bytes, n, k);
    fflush(stdout);
    name_the_end("forest-pass-shape"); // serving: p258 end 1 forest-pass-shape (kept as an exit(1))
    exit(1);
  }
  if (bytes > 49152) {
    static std::mutex m;
    static bool asked = false; // one for each P
    std::lock_guard<std::mutex> lock(m);
    if (!asked) CK_ATTRIBUTE(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)bytes);
    asked = true;
  }
  const int pieces = (int)(n / k), at_once = (int)std::min<size_t>(X::BLOCKS, 227 * 1024 / (bytes + 1024));
  kernel<<<std::min(pieces, g_sms * at_once), THREADS, bytes, s>>>(p, c0, pieces, ltiles, C);
  CK(cudaGetLastError());
}
inline size_t rounded16(size_t bytes) {
  return (bytes + 15) / 16 * 16;
}
// k_pass's P, layer LV <= 2 levels above the leaves: digits and arguments of fk::k_digits_poly().
template <int LV, class S> struct Keys {
  typedef fk::Poly<LV, S> P;
  static constexpr int NARROW = P::NARROW;
  const uint16_t *keys;
  int dT;
  const T *alpha;
  S src;
  int chunk;
  const u32 *W;
  struct Own {
    fk::Form<S> f;
    u32 *near;
    fk::Powers<P::D> *powers;
  };
  static constexpr size_t NEAR = (size_t)(6 * P::D + 6) * 128 * 4;
  size_t shared() const { return rounded16(fk::Form<S>::shared(src)) + NEAR + rounded16(sizeof(fk::Powers<P::D>)); }
  __device__ __forceinline__ void begin(Own &o, unsigned char *room) const {
    o.f.init_values(src, room);
    o.near = reinterpret_cast<u32 *>(room + (fk::Form<S>::shared(src) + 15) / 16 * 16);
    o.powers = reinterpret_cast<fk::Powers<P::D> *>(reinterpret_cast<unsigned char *>(o.near) + NEAR);
    if (!P::BYTE_FORM && threadIdx.x == 0) o.powers->set(*alpha, F::one());
  }
  __device__ __forceinline__ void stage(Own &o, size_t c) const {
    fk::table_near<P::D>(W, chunk, (int)(c % chunk), o.near, threadIdx.x, THREADS);
  }
  __device__ __forceinline__ void make(Own &o, size_t c, i8 *A, i8 *B) const {
    const int a = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const fk::KeySets<LV> left = fk::key_sets<LV>(keys, dT, a, c, lane), right = fk::key_sets<LV>(keys, dT, 8 + a, c, lane);
    fk::rows_of_slab<LV>(o.f, left, a, lane, o.near, o.powers,
                         [&](int row, u32 w) { put_word(A + (a * 24 + row) * LD + 4 * lane, w); });
    fk::rows_of_slab<LV>(o.f, right, 8 + a, lane, o.near, o.powers, [&](int row, u32 w) {
      put_word(B + (a * (P::BYTE_FORM ? P::NBT : 24) + row) * LD + 4 * lane, w);
    });
    if (P::BYTE_FORM && a < 4) put_word(B + (8 * P::NBT + a) * LD + 4 * lane, a ? 0u : 0x01010101u);
  }
};
// k_pass's P, pass 2 of the last layer: digits and arguments of fk::k_digits_second().
template <class S> struct Second {
  typedef fk::SecondRows<S> P;
  static constexpr int NARROW = P::NARROW;
  const uint16_t *keys;
  int dT;
  const T *alpha;
  S src;
  const T *k8;
  int chunk;
  const u32 *W;
  struct Own {
    fk::Form<S> f;
    fk::Powers<8> *constants;
  };
  size_t shared() const { return rounded16(fk::Form<S>::shared(src)) + rounded16(sizeof(fk::Powers<8>)); }
  __device__ __forceinline__ void begin(Own &o, unsigned char *room) const {
    o.f.init_values(src, room);
    o.constants = reinterpret_cast<fk::Powers<8> *>(room + (fk::Form<S>::shared(src) + 15) / 16 * 16);
    if (!P::BYTE_FORM && threadIdx.x == 0) fk::second_constants(*o.constants, *alpha, k8, F::one());
  }
  __device__ __forceinline__ void stage(Own &, size_t) const {}
  __device__ __forceinline__ void make(Own &o, size_t c, i8 *A, i8 *B) const {
    const int a = threadIdx.x >> 5, lane = threadIdx.x & 31;
    uint4 q[4];
    fk::eight_keys<fk::IN_FOURS, 64>(keys, dT, false, a, c, lane, q);
    fk::rows_of_second(o.f, q, a, c, lane, chunk, W, o.constants,
                       [&](int row, u32 w) { put_word(A + (a * 24 + row) * LD + 4 * lane, w); });
    fk::eight_keys<fk::IN_FOURS, 64>(keys, dT, true, a, c, lane, q);
    fk::rows_of_second(o.f, q, 8 + a, c, lane, chunk, W, o.constants, [&](int row, u32 w) {
      put_word(B + (a * (P::BYTE_FORM ? P::NBT : 24) + row) * LD + 4 * lane, w);
    });
    if (P::BYTE_FORM && a < 4) put_word(B + (8 * P::NBT + a) * LD + 4 * lane, a ? 0u : 0x01010101u);
  }
};
// The work of fk::digits() and grm::product() for LV <= 2; `c0`, `n`, `k` and `C` as in pass() and
// k_pass. fk::digits()'s conditions on `c0` and `chunk` apply here too; nothing checks them.
inline void keys_pass(int LV, const fk::Tree &t, int dT, const T *alpha, size_t c0, size_t n, int chunk, const u32 *W, int k, int *C,
                      cudaStream_t s) {
  fk::with_src(t.tab, [&](auto src) {
    fk::with_constant<0, 2>(LV, [&](auto lv) {
      pass(Keys<decltype(lv)::value, decltype(src)>{t.keys, dT, alpha, src, chunk, W}, c0, n, k, C, s);
    });
  });
}
// The work of fk::second_digits() and grm::product(); c0 and chunk as for keys_pass().
inline void second_pass(const fk::Tree &t, int dT, const T *alpha, const T *k8, size_t c0, size_t n, int chunk, const u32 *W, int k,
                        int *C, cudaStream_t s) {
  fk::with_src(t.tab, [&](auto src) { pass(Second<decltype(src)>{t.keys, dT, alpha, src, k8, chunk, W}, c0, n, k, C, s); });
}
// ---- Stored tables: the block works on two tiles at a time, and joins its sums itself.
// While the tensor cores multiply tile t, other units compute the digits of tile t + 1, and the
// entries of tile t + 2 are copied to shared memory. Warps 0 to 3 multiply first and then compute
// digits; warps 4 to 7 do the reverse, assuming that a scheduler holds warps w and w + 4 and so has
// one in either kind of work. At the end of a piece a warp joins its sums as grm::k_join() does,
// multiplies by the chunk's weight and adds to a running total. Each thread block writes 64 totals,
// which grm::k_sum() adds up. Only stored() checks that k <= PIECE_MOST and J.lsc >= 7.
// Nothing checks that a block has THREADS threads, or that J.L, J.R and J.eqlo are 16-byte aligned
// and J.ts is even, which the 16-byte asynchronous copies need.
// Public sources. [Async] NVIDIA, CUDA Programming Guide, "Asynchronous Data Copies" (the pipeline primitives
// __pipeline_memcpy_async(), __pipeline_commit(), __pipeline_wait_prior() of cuda_pipeline.h): fetch() starts a tile's
// copy out of global into shared memory, the warps compute on while it runs, and the copy takes no register on its way.
// [CUTLASS] (at the file's head): software pipelining with several stages in shared memory hides the loads' latency.
// In tree q, L's and R's slab a begins at entry q ts + a 2^lsc.
struct StoredJob {
  const K3 *L, *R; // as in fr::k_digits()
  size_t ts;
  int lsc, lchunk;   // log2 of a slab's entries in a tree, and of a chunk's indices
  const M3L *eqlo;   // a weight for each index of a chunk, prepared for `FK2::mulp`
  const T *eqhi;     // a weight for each chunk
};
// The bytes of one tile's entries, weights and digits.
constexpr int RAW = 16 * COLS * 24, WEIGHTS = COLS * 40, DIGITS = 384 * LD;
constexpr size_t STORED_BYTES = 2 * (size_t)(RAW + WEIGHTS + DIGITS);
__device__ __forceinline__ K3 entry_at(const unsigned char *p) {
  const u64 *q = reinterpret_cast<const u64 *>(p);
  return K3{q[0], q[1], q[2]};
}
__device__ __forceinline__ M3L weight_at(const unsigned char *p) {
  M3L w;
  const u64 *q = reinterpret_cast<const u64 *>(p);
  u64 x[5];
#pragma unroll
  for (int i = 0; i < 5; i++) x[i] = q[i];
  memcpy(&w, x, 40);
  return w;
}
// 'pieces' pieces of k = 128 2^ltiles indices, piece i from index i k on. k must divide a chunk:
// join() gives a piece one chunk's weight. out: 64 elements for each block.
__global__ void __launch_bounds__(THREADS, 1) k_stored(const StoredJob J, int pieces, int ltiles, T *__restrict__ out) {
  extern __shared__ __align__(16) unsigned char room[];
  unsigned char *const raw = room, *const weights = raw + 2 * RAW;
  i8 *const digits = reinterpret_cast<i8 *>(weights + 2 * WEIGHTS);
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t4 = lane & 3;
  const int wm = warp & 3, wn = warp >> 2;
  const int tiles = ((pieces - (int)blockIdx.x + (int)gridDim.x - 1) / (int)gridDim.x) << ltiles;
  const size_t SC = (size_t)1 << J.lsc;
  auto piece_of = [&](int t) { return (int)blockIdx.x + (t >> ltiles) * (int)gridDim.x; };
  auto index_of = [&](int t) { return (((size_t)piece_of(t) << ltiles) + (t & ((1 << ltiles) - 1))) * COLS; };
  // Starts tile t's copies; commits a group even with no tile t, because the waits count groups.
  auto fetch = [&](int t) {
    if (t < tiles) {
      const size_t g0 = index_of(t), at = (g0 >> J.lsc) * J.ts + (g0 & (SC - 1));
      // Warp a copies slabs a and 8 + a: 128 entries of 24 bytes each.
      unsigned char *to = raw + (t & 1) * RAW + warp * (COLS * 24) + lane * 16;
      const unsigned char *left = reinterpret_cast<const unsigned char *>(J.L + at + (size_t)warp * SC) + lane * 16,
                          *right = reinterpret_cast<const unsigned char *>(J.R + at + (size_t)warp * SC) + lane * 16;
#pragma unroll
      for (int u = 0; u < COLS * 24 / 512; u++) {
        __pipeline_memcpy_async(to + u * 512, left + u * 512, 16);
        __pipeline_memcpy_async(to + 8 * (COLS * 24) + u * 512, right + u * 512, 16);
      }
      const unsigned char *w = reinterpret_cast<const unsigned char *>(J.eqlo + (g0 & (((size_t)1 << J.lchunk) - 1)));
      for (int u = threadIdx.x; u < WEIGHTS / 16; u += THREADS)
        __pipeline_memcpy_async(weights + (t & 1) * WEIGHTS + u * 16, w + u * 16, 16);
    }
    __pipeline_commit();
  };
  // A lane's word of a row holds the indices lane, lane + 32, lane + 64, lane + 96 of the tile, in
  // both matrices: the order within a tile does not show in the sums.
  auto make = [&](int t) {
    if (t >= tiles) return;
    const unsigned char *from = raw + (t & 1) * RAW + warp * (COLS * 24) + lane * 24, *w = weights + (t & 1) * WEIGHTS + lane * 40;
    i8 *A = digits + (t & 1) * DIGITS + 4 * lane, *B = A + 192 * LD;
    K3 e[4];
#pragma unroll
    for (int i = 0; i < 4; i++) e[i] = FK2::mulp(weight_at(w + i * 32 * 40), entry_at(from + i * 32 * 24));
    grm::put_digits(A, LD, warp, e);
#pragma unroll
    for (int i = 0; i < 4; i++) e[i] = entry_at(from + 8 * (COLS * 24) + i * 32 * 24);
    grm::put_digits(B, LD, warp, e);
  };
  int sum[3][12][4];
  auto clear = [&] {
#pragma unroll
    for (int m = 0; m < 3; m++)
#pragma unroll
      for (int n = 0; n < 12; n++)
#pragma unroll
        for (int e = 0; e < 4; e++) sum[m][n][e] = 0;
  };
  auto product = [&](int t) {
    const i8 *mine_a = digits + (t & 1) * DIGITS + wm * 48 * LD, *mine_b = digits + (t & 1) * DIGITS + (192 + wn * 96) * LD;
#pragma unroll
    for (int s = 0; s < COLS / 32; s++) multiply_step<12>(sum, mine_a + 32 * s, mine_b + 32 * s, lane);
  };
  // The warp's sums belong to the pairs of slabs (a, a') = (2 wm + x, 4 wn + y), x < 2, y < 4. With
  // h = e >> 1, sum[m][n][e] belongs to digit g of coordinate (2 m + h) % 3 of slab
  // 2 wm + (2 m + h) / 3 and digit 2 t4 + (e & 1) of coordinate n % 3 of slab 4 wn + n / 3; its
  // weight is 2^(8 (g + 2 t4 + (e & 1))). join() first adds integers: even and odd digits as
  // v[0] + 2^8 v[1], then the five degrees in X into three coordinates (X^3 = w = FIELD_W). A lane
  // then has three numbers for each pair, each below 2^43: 2^30 (a sum) x 3 (sums of a degree)
  // x 257 x (1 + w), w <= 5. The lanes, lane = 4 g + t4, are added up bit by bit of `lane`: for a
  // bit of weight 2^S, (the value where it is 0) + 2^S (the value where it is 1). In the step over
  // bit i of g a lane keeps the pairs p = 4 x + y whose bit i equals its own and sends the others:
  // 12 + 6 + 3 exchanges, not 3 x 24. Pair p ends in lanes 4 p .. 4 p + 3, and two steps over t4
  // bring it to lane 4 p, which keeps the pair's running total. The result is exact and canonical,
  // like grm::k_join()'s.
  // The exchange of halves is the usual way to add up many values over a warp at once; no source
  // was looked for.
  T total = F::zero();
  auto join = [&](int t) {
    long long c[8][3];
#pragma unroll
    for (int x = 0; x < 2; x++)
#pragma unroll
      for (int y = 0; y < 4; y++) {
        long long u[5];
#pragma unroll
        for (int d = 0; d < 5; d++) {
          long long v[2] = {0, 0};
#pragma unroll
          for (int j = 0; j < 3; j++) {
            const int jp = d - j, row = 3 * x + j; // row = 2 m + h
            if (jp < 0 || jp > 2) continue;
#pragma unroll
            for (int odd = 0; odd < 2; odd++) v[odd] += sum[row >> 1][3 * y + jp][2 * (row & 1) + odd];
          }
          u[d] = v[0] + 256 * v[1];
        }
        c[4 * x + y][0] = u[0] + (long long)FIELD_W * u[3], c[4 * x + y][1] = u[1] + (long long)FIELD_W * u[4], c[4 * x + y][2] = u[2];
      }
    static_assert((long long)PIECE_MOST * DIGIT_MOST * DIGIT_MOST * 3 * 257 * (1 + (long long)FIELD_W) * 257 < (1ll << 51),
                  "k_stored's join: the whole numbers of its first step must stay below 2^51, far below p and 2^63: take smaller pieces (PIECE_MOST)");
    auto turned = [](auto shift, u64 lo, u64 hi) { return gf_add(lo, blk::lazy::shifted<decltype(shift)::value>(blk::F{hi}).v); };
    const bool up2 = lane & 4, up3 = lane & 8, up4 = lane & 16;
    u64 a[4][3], b[2][3], e[3];
#pragma unroll
    for (int q = 0; q < 4; q++)
#pragma unroll
      for (int i = 0; i < 3; i++) { // bit 0 of g: 2^8
        const long long keep = up2 ? c[2 * q + 1][i] : c[2 * q][i], got = __shfl_xor_sync(~0u, up2 ? c[2 * q][i] : c[2 * q + 1][i], 4);
        const long long s = (up2 ? got : keep) + 256 * (up2 ? keep : got);
        a[q][i] = s < 0 ? FIELD_P - (u64)(-s) : (u64)s;
      }
#pragma unroll
    for (int r = 0; r < 2; r++)
#pragma unroll
      for (int i = 0; i < 3; i++) { // bit 1 of g: 2^16
        const u64 keep = up3 ? a[2 * r + 1][i] : a[2 * r][i], got = __shfl_xor_sync(~0u, up3 ? a[2 * r][i] : a[2 * r + 1][i], 8);
        b[r][i] = turned(std::integral_constant<int, 16>{}, up3 ? got : keep, up3 ? keep : got);
      }
#pragma unroll
    for (int i = 0; i < 3; i++) { // bit 2 of g: 2^32; then the two bits of t4: 2^16, 2^32
      const u64 keep = up4 ? b[1][i] : b[0][i], got = __shfl_xor_sync(~0u, up4 ? b[0][i] : b[1][i], 16);
      e[i] = turned(std::integral_constant<int, 32>{}, up4 ? got : keep, up4 ? keep : got);
      e[i] = turned(std::integral_constant<int, 16>{}, e[i], __shfl_down_sync(~0u, e[i], 1));
      e[i] = turned(std::integral_constant<int, 32>{}, e[i], __shfl_down_sync(~0u, e[i], 2));
    }
    if ((lane & 3) == 0) total = F::add(total, F::mul(K3{e[0], e[1], e[2]}, J.eqhi[index_of(t) >> J.lchunk]));
  };
  fetch(0), fetch(1);
  __pipeline_wait_prior(1);
  __syncthreads();
  make(0);
  clear();
  __syncthreads();
  for (int t = 0; t < tiles; t++) {
    fetch(t + 2); // overwrites tile t's entries, whose digits were made before the last barrier
    __pipeline_wait_prior(1);
    __syncthreads(); // tile t + 1's entries have arrived, for every thread
    if (warp < 4) {
      product(t);
      make(t + 1);
    } else {
      make(t + 1);
      product(t);
    }
    if ((t & ((1 << ltiles) - 1)) == (1 << ltiles) - 1) {
      join(t);
      clear();
    }
    __syncthreads();
  }
  if ((lane & 3) == 0) out[(size_t)blockIdx.x * 64 + (2 * wm + (lane >> 4)) * 8 + 4 * wn + ((lane >> 2) & 3)] = total;
}
// One whole pass over 'total' indices. Writes 64 elements for each block to out and returns the
// number of blocks (at most g_sms). The first call raises k_stored's limit of shared memory: not
// during a stream capture (see pass()).
inline int stored(const StoredJob &J, size_t total, T *out, cudaStream_t s) {
  static const bool once = [] {
    CK_ATTRIBUTE(k_stored, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)STORED_BYTES);
    return true;
  }();
  (void)once;
  size_t k = (size_t)1 << J.lchunk;
  while (total / k < (size_t)4 * g_sms && k > 1024) k /= 2;
  int ltiles = 0;
  while (((size_t)COLS << ltiles) < k) ltiles++;
  if (((size_t)COLS << ltiles) != k || total % k || J.lsc < 7 || k > (size_t)PIECE_MOST) {
    printf("internal error: look-up forest: a pass over stored tables of %zu indices in pieces of %zu\n", total, k);
    fflush(stdout);
    name_the_end("forest-pass-shape"); // serving: p258 end 1 forest-pass-shape (kept as an exit(1))
    exit(1);
  }
  const int pieces = (int)(total / k), blocks = std::min(pieces, g_sms);
  k_stored<<<blocks, THREADS, STORED_BYTES, s>>>(J, pieces, ltiles, out);
  CK(cudaGetLastError());
  return blocks;
}
} // namespace ff
