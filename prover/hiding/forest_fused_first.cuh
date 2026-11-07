// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Lookup forest (the product trees of a group of lookup calls): the first pass of the last layer,
// the one that reads the leaves (`LV` 0 in prover/hiding/forest_keys.cuh), for the trees of range
// tables, where the table's value for lookup key k is k. It runs on the tensor cores and gives
// these trees' share of the 8 x 8 matrix G that the layer's first three sumcheck rounds use.
// Call order: k_first_table(), first_range(), a sum over the records written, k_first_finish().
//
// A tree has N = 2^dT 16-bit keys. Index c has the left keys k_a(c) = keys[8 c + a] and the
// right keys k'_a(c) = keys[N / 2 + 8 c + a], a < 8. With the tree's challenge alpha,
//   G[a][a'] = sum_c w(c) (alpha - k_a(c)) (alpha - k'_a'(c))
//            = alpha^2 sum_c w - alpha (sum_c w k_a + sum_c w k'_a') + sum_c w k_a k'_a',
// added over the trees. A tree has 2^nh chunks of 2^lc indices; c runs over the chunks in
// FirstTrees, and w(c) = (the chunk's weight, from eqhi) eqlo[c mod 2^lc].
// A chunk's weights eqlo must add up to 1, as equality weights eq(r, .) do. This is not checked:
// for sum_c w the code adds up only eqhi.
//
// Public sources: as in prover/hiding/forest_fused_folds.cuh. Keeping the large factor apart and
// multiplying the small ones first is the small-value idea of Bagad, Domb, Thaler, "The Sum-Check
// Protocol over Fields of Small Characteristic", ePrint 2024/1046; there on a CPU, in machine
// words. [Async] (the copies into `rings`): see prover/hiding/forest_fused.cuh.
// Include after prover/hiding/forest_fused_folds.cuh.
#pragma once
#include "common/end_name.h" // name_the_end(): names an internal stop before a direct exit (M5e3)
namespace ff {
// A thread block's record: 64 sums of products, 8 of left keys and 8 of right keys (times alpha),
// the sum of eqhi (times alpha^2).
constexpr int FIRST_WARPS = 16, FIRST_SUMS = 81; // warps in a block; elements (`T`) in a record
// Dynamic shared memory in bytes: a ring of keys and running totals for each warp.
constexpr int FIRST_BYTES = FIRST_WARPS * (4096 + FIRST_SUMS * 24);
constexpr int FIRST_PART_BITS = 12; // a warp sums over one part of a chunk: at most 2^12 indices
static_assert((4ll << FIRST_PART_BITS) * BYTE_MOST * DIGIT_MOST <= INT_MOST,
              "a part's sum fits 32 bits: four unsigned bytes an index (of a product of two keys), each against a signed digit of a weight");
static_assert(INT_MOST < TURN_MOST, "each 32-bit sum is turned by itself");
// The digit table, for a chunk of 2^lc indices; it serves every chunk and tree. An element of the
// cubic extension field has 24 signed byte digits, eight for each of three coordinates. For
// each group s of eight indices and each coordinate j the table has 32 pairs of 32-bit words, one
// for each thread 4 g + t4 of a warp. Byte b of the first word is digit g of coordinate j of
// eqlo[8 s + t4] 2^(8 b) mod p; the second word has the same for eqlo[8 s + 4 + t4].
inline size_t first_table_words(int lc) {
  return (size_t)24 << lc;
}
__global__ void k_first_table(const T *__restrict__ eqlo, int n, u32 *__restrict__ table) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= n) return;
  const T w = eqlo[c];
#pragma unroll
  for (int j = 0; j < 3; j++) {
    const u64 x[4] = {coord(w, j), grm::gram_times_2_to(coord(w, j), 8), grm::gram_times_2_to(coord(w, j), 16), grm::gram_times_2_to(coord(w, j), 24)};
    u32 digit[8];
    grm::digit_words(x, digit);
#pragma unroll
    for (int g = 0; g < 8; g++) table[((((size_t)(c >> 3) * 3 + j) * 32 + 4 * g + (c & 3)) << 1) + ((c >> 2) & 1)] = digit[g];
  }
}
// The trees of a launch. Chunk i of the launch belongs to the tree q with
// first[q] <= i < first[q + 1]; it is that tree's chunk i - first[q], and its weight is
// eqhi[(place[q] << nh) + i - first[q]].
struct FirstTrees {
  const uint16_t *keys[256];
  int first[257];
  unsigned char place[256];
  const T *alpha[256]; // the tree's alpha, GPU memory
  int n;
};
// The other kernels for this layer multiply eqlo into each left factor (fk::rows_of_dot() in
// prover/hiding/forest_keys.cuh), which costs a reduction modulo p and a split into digits for
// every left key. Here one side of the matrix product holds only eqlo, as digits from `table`. The
// other side holds 32-bit integers: the 64 products k_a k'_a' and the 16 keys. Such an integer is
// four neighbouring bytes of the summed dimension, which therefore runs over (index, byte), so a
// product goes from the multiplier's register into the tensor cores as it is.
// A warp's 80 rows by 24 digits are 60 sums a thread. The sums are exact and the elements made
// from them canonical, so each entry of G is the element that the other kernels give.
__global__ void __launch_bounds__(32 * FIRST_WARPS, 1)
    k_first_range(const __grid_constant__ FirstTrees trees, int dT, int lc, int nh, const uint2 *__restrict__ table,
                  const T *__restrict__ eqhi, T *__restrict__ out) {
  // A warp's ring: four batches of 32 indices, each 512 bytes of left keys, then 512 of right keys.
  extern __shared__ __align__(16) unsigned char rings[];
  T *const totals = reinterpret_cast<T *>(rings + FIRST_WARPS * 4096) + (threadIdx.x >> 5) * FIRST_SUMS; // the warp's running totals
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t4 = lane & 3, ga = g >> 1, gb = g & 1;
  // An item is one part of one chunk. Items are ordered by part, then by chunk: a thread block's
  // warps take neighbouring items, so they mostly read the same part of the digit table.
  const int chunks = trees.first[trees.n], lp = lc < FIRST_PART_BITS ? lc : FIRST_PART_BITS, batches = 1 << (lp - 5);
  const int items = chunks << (lc - lp);
  for (int x = lane; x < FIRST_SUMS; x += 32) totals[x] = F::zero();
  __syncwarp();
  for (int i0 = blockIdx.x * FIRST_WARPS; i0 < items; i0 += gridDim.x * FIRST_WARPS) {
    const bool works = i0 + warp < items;
    const int part = works ? (i0 + warp) / chunks : 0, i = works ? (i0 + warp) % chunks : 0;
    int q = 0;
    if (works)
      while (trees.first[q + 1] <= i) q++;
    const size_t c0 = works ? ((size_t)(i - trees.first[q]) << lc) + ((size_t)part << lp) : 0;
    const uint2 *weights_of = table + ((size_t)part << (lp - 3)) * 96;
    const unsigned char *left = reinterpret_cast<const unsigned char *>(trees.keys[q] + 8 * c0),
                        *right = left + ((size_t)1 << dT); // N / 2 keys further
    auto fetch = [&](int batch) {
      if (works && batch < batches) {
        unsigned char *to = rings + (warp * 4 + (batch & 3)) * 1024;
        __pipeline_memcpy_async(to + 16 * lane, left + (size_t)batch * 512 + 16 * lane, 16);
        __pipeline_memcpy_async(to + 512 + 16 * lane, right + (size_t)batch * 512 + 16 * lane, 16);
      }
      __pipeline_commit();
    };
    int sum[5][3][4];
#pragma unroll
    for (int x = 0; x < 5; x++)
#pragma unroll
      for (int j = 0; j < 3; j++)
#pragma unroll
        for (int e = 0; e < 4; e++) sum[x][j][e] = 0;
    fetch(0), fetch(1), fetch(2);
    for (int batch = 0; batch < batches; batch++) {
      // Keeps the block's warps about in step, so that one fetch of a part of the digit table can
      // serve them all. A warp without an item (`works` false) runs this loop too, for the barrier.
      if ((batch & 15) == 0) __syncthreads();
      __pipeline_wait_prior(2);
      __syncwarp();
      if (works) {
        const unsigned char *from = rings + (warp * 4 + (batch & 3)) * 1024;
#pragma unroll
        for (int s = 0; s < 4; s++) {
          uint2 b[3];
#pragma unroll
          for (int j = 0; j < 3; j++) b[j] = __ldg(weights_of + ((size_t)(batch * 4 + s) * 3 + j) * 32 + lane);
          u32 l[2][2], r[2][4], one[2][2]; // [index 8 s + t4 + 4 h]
#pragma unroll
          for (int h = 0; h < 2; h++) {
            const int at = (8 * s + t4 + 4 * h) * 16;
            const u32 lw = *reinterpret_cast<const u32 *>(from + at + 4 * ga);
            const uint2 rw = *reinterpret_cast<const uint2 *>(from + 512 + at + 8 * gb);
            l[h][0] = lw & 0xFFFFu, l[h][1] = lw >> 16;
            r[h][0] = rw.x & 0xFFFFu, r[h][1] = rw.x >> 16, r[h][2] = rw.y & 0xFFFFu, r[h][3] = rw.y >> 16;
            one[h][0] = gb ? l[h][1] : l[h][0];                                          // left key 2 ga + gb
            one[h][1] = ga & 2 ? (ga & 1 ? r[h][3] : r[h][2]) : (ga & 1 ? r[h][1] : r[h][0]); // right key 4 gb + ga
          }
          // Tile x < 4: row g is the product for (a, a') = (2 ga, 4 gb + x), row g + 8 that for
          // (2 ga + 1, 4 gb + x). Tile 4: the two single keys.
#pragma unroll
          for (int x = 0; x < 5; x++) {
            const u32 a[4] = {x < 4 ? l[0][0] * r[0][x & 3] : one[0][0], x < 4 ? l[0][1] * r[0][x & 3] : one[0][1],
                              x < 4 ? l[1][0] * r[1][x & 3] : one[1][0], x < 4 ? l[1][1] * r[1][x & 3] : one[1][1]};
#pragma unroll
            for (int j = 0; j < 3; j++) {
              const u32 bb[2] = {b[j].x, b[j].y};
              multiply_unsigned(sum[x][j], a, bb);
            }
          }
        }
      }
      __syncwarp();
      fetch(batch + 3);
    }
    __pipeline_wait_prior(0);
    if (works) {
      // Value v = 2 x + h is row g + 8 h of tile x; sum[x][j][2 h + e] is its sum for digit
      // 2 t4 + e of coordinate j. The weights hi1 and hi2 carry the tree's powers of alpha, so
      // trees with different alphas add into the same totals.
      const T hi = eqhi[((size_t)trees.place[q] << nh) + (i - trees.first[q])], hi1 = F::mul(hi, *trees.alpha[q]),
              hi2 = F::mul(hi1, *trees.alpha[q]);
      T mine[3] = {F::zero(), F::zero(), F::zero()}; // values t4, t4 + 4, t4 + 8
#pragma unroll
      for (int v = 0; v < 10; v++) {
        u64 x[3];
#pragma unroll
        for (int j = 0; j < 3; j++) {
          x[j] = gf_add(grm::signed_turn(sum[v >> 1][j][2 * (v & 1)], (16 * t4) % grm::TURNS),
                         grm::signed_turn(sum[v >> 1][j][2 * (v & 1) + 1], (16 * t4 + 8) % grm::TURNS));
          x[j] = gf_add(x[j], __shfl_xor_sync(~0u, x[j], 1));
          x[j] = gf_add(x[j], __shfl_xor_sync(~0u, x[j], 2));
        }
        if ((v & 3) == t4) mine[v >> 2] = T{x[0], x[1], x[2]};
      }
#pragma unroll
      for (int u = 0; u < 3; u++) {
        const int v = 4 * u + t4, x = v >> 1, h = v & 1; // no two lanes get the same `at`
        if (v < 10) {
          const int at = x < 4 ? (2 * ga + h) * 8 + 4 * gb + x : h ? 72 + 4 * gb + ga : 64 + 2 * ga + gb;
          totals[at] = F::add(totals[at], F::mul(mine[u], x < 4 ? hi : hi1));
        }
      }
      if (!part && lane == 0) totals[80] = F::add(totals[80], hi2);
    }
  }
  __syncthreads();
  for (int x = threadIdx.x; x < FIRST_SUMS; x += blockDim.x) {
    const T *of_warps = reinterpret_cast<const T *>(rings + FIRST_WARPS * 4096) + x;
    T sum_ = of_warps[0];
    for (int w = 1; w < FIRST_WARPS; w++) sum_ = F::add(sum_, of_warps[w * FIRST_SUMS]);
    out[(size_t)blockIdx.x * FIRST_SUMS + x] = sum_;
  }
}
// Makes G[8 a + a'] from H, the sum of all blocks' records (see FIRST_SUMS). One block of 64
// threads.
__global__ void k_first_finish(const T *H, T *G) {
  const int t = threadIdx.x;
  G[t] = F::add(F::sub(H[80], F::add(H[64 + (t >> 3)], H[72 + (t & 7)])), H[t]);
}
// Launches k_first_range() and returns the number of records written to out, at most g_sms (the
// GPU's count of multiprocessors, prover/common/common.h). Exits with code 1 if the sizes fail the
// test below. Not checked: that out can hold g_sms records of FIRST_SUMS elements, that `table` was
// made for this lc, that trees.first rises from 0 in steps of at most 2^nh, and that each
// trees.keys[q] is 16-byte and `table` 8-byte aligned.
inline int first_range(const FirstTrees &trees, int dT, int lc, int nh, const u32 *table, const T *eqhi, T *out, cudaStream_t s) {
  // lc >= 5: a part is read in batches of 32 indices.
  if (lc < 5 || trees.n < 1 || trees.n > 256 || nh < 0 || lc + nh + 4 != dT) {
    printf("internal error: look-up forest: the last layer's first pass over %d trees of 2^%d keys in 2^%d chunks of 2^%d indices\n", trees.n, dT, nh, lc);
    fflush(stdout);
    name_the_end("forest-pass-shape"); // serving: p258 end 1 forest-pass-shape (kept as an exit(1))
    exit(1);
  }
  const int items = trees.first[trees.n] << std::max(lc - FIRST_PART_BITS, 0), blocks = std::min((items + FIRST_WARPS - 1) / FIRST_WARPS, g_sms);
  // Runs at the first call only, which must be outside a stream capture: forest_prove_graph()
  // (prover/hiding/forest.cuh) launches its first proof directly.
  static const bool once = [] {
    CK_ATTRIBUTE(k_first_range, cudaFuncAttributeMaxDynamicSharedMemorySize, FIRST_BYTES);
    return true;
  }();
  (void)once;
  k_first_range<<<blocks, 32 * FIRST_WARPS, FIRST_BYTES, s>>>(trees, dT, lc, nh, reinterpret_cast<const uint2 *>(table), eqhi, out);
  CK(cudaGetLastError());
  return blocks;
}
} // namespace ff
