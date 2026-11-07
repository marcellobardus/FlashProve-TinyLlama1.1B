// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Lookup forest (the product trees of a group of lookup calls): pass 2 (sumcheck rounds 4 to 6) of
// the last layer, which reads the leaves, for range tables (a key is its own table value), on the
// tensor cores. From prover/hiding/gram_products.cuh: slab, one of a side's eight parts; chunk,
// indices that share an eqhi weight; piece, part of a chunk summed in 32 bits.
//
// Defined in prover/hiding/forest_keys.cuh, "Pass 2 of the last layer":
// the left entry of slab a at index c is
//   w U_a = W_0(c) + sum over j < 8 of t_j W_(j+1)(c),
// with w = eqlo[c mod chunk], 16-bit keys t_j = keys[64 c + 8 a + j] and nine field elements W_i
// from a table; the right entries are the bytes of keys[N / 2 + 64 c + 8 a + j], for N = 2^dT keys.
// That file reduces each of a left entry's three coordinates modulo p = 2^64 - 2^32 + 1 and cuts it
// into eight signed digits, which costs more instructions than the multiplications. Here it stays a
// 9-byte integer.
//
// Public sources: [TensorFHE], [Ozaki-int8], [FFLAS] (delayed reduction) and [Dakkak], section 4
// and Figure 5 (a row of ones in a product gives sums; here among the second matrix's rows), as
// cited in prover/hiding/gram_products.cuh; [mma], [ldmatrix], [CUTLASS], [Async] as in
// prover/hiding/forest_fused.cuh. For 2^64 modulo p, see reduce128() in blocks/field/gl64.h.
// Include after prover/hiding/forest_fused.cuh. Called by the engine, prover/hiding/forest.cuh.
#pragma once
#include "common/end_name.h" // name_the_end(): names an internal stop before a direct exit (M5e3)
namespace ff {
namespace second {
constexpr int BYTES_LEFT = 9, ROWS_A = 8 * 3 * BYTES_LEFT, TILES_A = (ROWS_A + 15) / 16; // 216 rows, 14 blocks of 16
constexpr int ROWS_B = 129, NW = 5;                                                    // 128 rows of bytes, a row of ones; 4 warps x 5 blocks of 8
constexpr int MW = TILES_A / 2;                                                        // 2 warps x 7 blocks of 16
constexpr int PAIR = (16 * TILES_A + 32 * NW) * LD;
constexpr int PLANES = 54, NEAR = PLANES * 16 * 4; // a warp's part of the table: 54 planes of 16 words
constexpr size_t BYTES = 2 * (size_t)PAIR + 2 * 8 * (size_t)NEAR;
constexpr int SUMS = 520;
constexpr u64 TWO_64 = 0xFFFFFFFFull; // 2^64 modulo p
// Per coordinate, with W_i = lo_i + 2^32 hi_i, a left entry is (lower mod 2^32) + 2^32 b < 2^84:
//   lower = lo_0 + sum t_j lo_(j+1) < 2^52,  upper = hi_0 + sum t_j hi_(j+1),
//   b = upper + (lower >> 32) < 2^52.
// Its bits from 2^64 on, fewer than 20, are added back as TWO_64 = 2^32 - 1 times their value,
// which is below 2^52. With the low 64 bits that makes 65 bits: nine bytes.
// Bounds for any keys and table:
constexpr unsigned __int128 KEY_MOST = 65535, WORD_MOST = 0xFFFFFFFFull, U64_MOST = ~0ull;
constexpr unsigned __int128 HALF_MOST = WORD_MOST * (1 + 8 * KEY_MOST);          // 'lower' and 'upper': a word, and eight keys times a word
constexpr unsigned __int128 B_MOST = HALF_MOST + (HALF_MOST >> 32);              // b = upper + (lower >> 32)
static_assert(B_MOST <= U64_MOST, "lower, upper and b fit 64 bits");
static_assert(TWO_64 * (B_MOST >> 32) <= U64_MOST, "what comes back for the bits from 2^64 on fits 64 bits, so low + that has a carry of 0 or 1: nine bytes");
constexpr int PIECE_MOST = 16384; // the largest piece, in indices: `second_range` exits at more
static_assert(BYTE_MOST * BYTE_MOST * PIECE_MOST <= INT_MOST, "a piece's sum of products of two unsigned bytes fits 32 bits");
// k_join_second() adds a chunk's pieces in 64 bits: at most 2^16 indices (far more than the
// engine's 2^13) in pieces of at least 128.
static_assert((65536 / COLS) * INT_MOST < TURN_MOST, "the sum over a chunk's pieces can be turned");
} // namespace second
// d += a b as multiply() (forest_fused.cuh), but both hold unsigned bytes: no offset of 128 has to
// be corrected later.
// But for the operand types (u8 for s8), the asm template and the order of its operands are the
// same as in ggml, ggml/src/ggml-cuda/mma.cuh (llama.cpp), MIT; that project's notice is in NOTICE.
__device__ __forceinline__ void multiply_both_unsigned(int (&d)[4], const u32 (&a)[4], const u32 (&b)[2]) {
  asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.u8.u8.s32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"
               : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}
// The table for `chunk` indices: plane 2 (3 i + j) holds the low 32 bits of coordinate j of W_i,
// the next plane the bits above them; a plane has `chunk` words. W_0 = alpha kappa eqlo,
// W_(j+1) = -k8[j] eqlo, kappa = sum_j k8[j], as fk::second_constants() (forest_keys.cuh).
inline size_t second_table_words(int chunk) {
  return (size_t)second::PLANES * chunk;
}
__global__ void k_second_table(const T *alpha, const T *__restrict__ k8, int chunk, const T *__restrict__ eqlo, u32 *__restrict__ table) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= chunk) return;
  const T w = eqlo[c];
  T kappa = F::zero();
  for (int i = 0; i < 8; i++) kappa = F::add(kappa, k8[i]);
  for (int i = 0; i < 9; i++) {
    const T v = i ? F::sub(F::zero(), F::mul(k8[i - 1], w)) : F::mul(F::mul(*alpha, kappa), w);
    for (int j = 0; j < 3; j++) {
      table[(size_t)(2 * (3 * i + j)) * chunk + c] = (u32)coord(v, j);
      table[(size_t)(2 * (3 * i + j) + 1) * chunk + c] = (u32)(coord(v, j) >> 32);
    }
  }
}
// Handles `pieces` pieces of 128 2^ltiles indices of one tree, from index c0 on. Per piece, C gets
// 216 x 129 sums, column-major: entry 216 q + r is row r = 27 a + 9 j + m of the first matrix (byte
// m of coordinate j of w U_a) times row q = 16 a' + 2 j' + b of the second (byte b of key j' of
// slab a'; q = 128: ones). Not checked: a piece's length divides `chunk`, which divides c0; every
// index is below N / 128; `keys` and `table` are 16-byte aligned. Launch it through second_range():
// a piece has at most second::PIECE_MOST = 2^14 indices, so no sum overflows: 255^2 2^14 < 2^31.
// A warp builds the byte rows for 16 of a tile's 128 indices. Its eight threads with the same four
// indices, one per slab, read the same table words, so a word is fetched once per index, not per
// slab. Warps 0 to 3 multiply tile t first, warps 4 to 7 build tile t + 1 first: the tasks overlap.
__global__ void __launch_bounds__(THREADS, 1)
    k_second_range(const uint16_t *__restrict__ keys, int dT, int chunk, const u32 *__restrict__ table, size_t c0, int pieces,
                   int ltiles, int *__restrict__ C) {
  using namespace second;
  extern __shared__ __align__(16) unsigned char room[];
  i8 *const digits = reinterpret_cast<i8 *>(room);
  unsigned char *const near = room + 2 * PAIR;
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int tiles = ((pieces - (int)blockIdx.x + (int)gridDim.x - 1) / (int)gridDim.x) << ltiles;
  auto piece_of = [&](int t) { return (int)blockIdx.x + (t >> ltiles) * (int)gridDim.x; };
  auto index_of = [&](int t) { return c0 + (((size_t)piece_of(t) << ltiles) + (t & ((1 << ltiles) - 1))) * COLS; };
  // The row of ones [Dakkak], in both pairs of tiles.
  // Public source: one more line of an operand, so that the same product also gives the sum: A. Shafiee, A. Nag, N.
  // Muralimanohar, R. Balasubramonian, J. P. Strachan, M. Hu, R. S. Williams, V. Srikumar, "ISAAC: A Convolutional
  // Neural Network Accelerator with In-Situ Analog Arithmetic in Crossbars", ISCA 2016, section "Managing Bits, ADCs,
  // and Signed Arithmetic", heading "Encoding to Reduce ADC Size", https://users.cs.utah.edu/~rajeev/pubs/isca16.pdf
  // (there a unit column of a crossbar array; here a row of ones in a matrix of bytes).
  for (int x = threadIdx.x; x < 2 * (COLS / 4); x += THREADS)
    put_word(digits + (x / (COLS / 4)) * PAIR + (16 * TILES_A + 128) * LD + 4 * (x % (COLS / 4)), 0x01010101u);
  // Starts copying the warp's part of the table for tile t. Commits a group even past the last
  // tile: __pipeline_wait_prior() counts groups.
  auto fetch = [&](int t) {
    if (t < tiles) {
      const u32 *from = table + (index_of(t) + 16 * warp) % chunk;
      unsigned char *to = near + ((t & 1) * 8 + warp) * NEAR;
      for (int u = lane; u < PLANES * 4; u += 32)
        __pipeline_memcpy_async(to + 16 * u, from + (size_t)(u >> 2) * chunk + 4 * (u & 3), 16);
    }
    __pipeline_commit();
  };
  const int quad = lane >> 3, slab = lane & 7; // the thread's four indices within the warp's 16, and its slab
  auto make = [&](int t) {
    if (t >= tiles) return;
    const size_t c = index_of(t) + 16 * warp + 4 * quad;
    i8 *A = digits + (t & 1) * PAIR + 16 * warp + 4 * quad, *B = A + 16 * TILES_A * LD;
    const uint4 *left = reinterpret_cast<const uint4 *>(keys + 64 * c) + slab, *right = left + ((size_t)1 << (dT - 4)); // N / 2 keys further
    uint4 q[4];
#pragma unroll
    for (int i = 0; i < 4; i++) q[i] = __ldg(right + 8 * i);
#pragma unroll
    for (int x = 0; x < 4; x++) { // 32-bit word x of the four indices' keys: keys 2 x and 2 x + 1
      const u32 in[4] = {x == 0 ? q[0].x : x == 1 ? q[0].y : x == 2 ? q[0].z : q[0].w, x == 0 ? q[1].x : x == 1 ? q[1].y : x == 2 ? q[1].z : q[1].w,
                         x == 0 ? q[2].x : x == 1 ? q[2].y : x == 2 ? q[2].z : q[2].w, x == 0 ? q[3].x : x == 1 ? q[3].y : x == 2 ? q[3].z : q[3].w};
      u32 out[4];
      grm::transpose_bytes(in, out);
#pragma unroll
      for (int y = 0; y < 4; y++) put_word(B + (16 * slab + 4 * x + y) * LD, out[y]);
    }
#pragma unroll
    for (int i = 0; i < 4; i++) q[i] = __ldg(left + 8 * i);
    const uint4 *w = reinterpret_cast<const uint4 *>(near + ((t & 1) * 8 + warp) * NEAR) + quad; // plane x at w[4 x]
#pragma unroll
    for (int j = 0; j < 3; j++) {
      u64 lower[4], upper[4];
      {
        const uint4 lo = w[4 * (2 * j)], hi = w[4 * (2 * j + 1)];
        lower[0] = lo.x, lower[1] = lo.y, lower[2] = lo.z, lower[3] = lo.w;
        upper[0] = hi.x, upper[1] = hi.y, upper[2] = hi.z, upper[3] = hi.w;
      }
#pragma unroll
      for (int e = 0; e < 8; e++) {
        const uint4 lo = w[4 * (2 * (3 * (e + 1) + j))], hi = w[4 * (2 * (3 * (e + 1) + j) + 1)];
        const u32 l[4] = {lo.x, lo.y, lo.z, lo.w}, h[4] = {hi.x, hi.y, hi.z, hi.w};
#pragma unroll
        for (int i = 0; i < 4; i++) {
          const u32 key = grm::key_of(q[i], e);
          lower[i] = wm(key, l[i], lower[i]), upper[i] = wm(key, h[i], upper[i]);
        }
      }
      u32 w0[4], w1[4], w2[4];
#pragma unroll
      for (int i = 0; i < 4; i++) {
        const u64 b = upper[i] + (lower[i] >> 32), low = (lower[i] & 0xFFFFFFFFull) | b << 32, sum = low + TWO_64 * (b >> 32);
        w0[i] = (u32)sum, w1[i] = (u32)(sum >> 32), w2[i] = sum < low;
      }
      u32 out[8];
      grm::transpose_bytes(w0, out);
      grm::transpose_bytes(w1, out + 4);
      i8 *rows = A + (27 * slab + 9 * j) * LD;
#pragma unroll
      for (int y = 0; y < 8; y++) put_word(rows + y * LD, out[y]);
      put_word(rows + 8 * LD, w2[0] | w2[1] << 8 | w2[2] << 16 | w2[3] << 24);
    }
  };
  const int wmr = warp & 1, wn = warp >> 1, g = lane >> 2, t4 = lane & 3;
  int sum[MW][NW][4];
  auto clear = [&] {
#pragma unroll
    for (int m = 0; m < MW; m++)
#pragma unroll
      for (int n = 0; n < NW; n++)
#pragma unroll
        for (int e = 0; e < 4; e++) sum[m][n][e] = 0;
  };
  auto product = [&](int t) {
    const int block = lane >> 3, row = lane & 7;
    const i8 *A = digits + (t & 1) * PAIR + wmr * MW * 16 * LD, *B = digits + (t & 1) * PAIR + (16 * TILES_A + wn * NW * 8) * LD;
#pragma unroll
    for (int s = 0; s < COLS / 32; s++) {
      u32 b[NW][2];
#pragma unroll
      for (int n = 0; n + 1 < NW; n += 2) {
        u32 x[4];
        load_four(x, B + ((n + (block >> 1)) * 8 + row) * LD + (block & 1) * 16 + 32 * s);
        b[n][0] = x[0], b[n][1] = x[1], b[n + 1][0] = x[2], b[n + 1][1] = x[3];
      }
      load_two(b[NW - 1], B + ((NW - 1) * 8 + row) * LD + (block & 1) * 16 + 32 * s);
#pragma unroll
      for (int m = 0; m < MW; m++) {
        u32 a[4];
        load_four(a, A + (m * 16 + (block & 1) * 8 + row) * LD + (block >> 1) * 16 + 32 * s);
#pragma unroll
        for (int n = 0; n < NW; n++) multiply_both_unsigned(sum[m][n], a, b[n]);
      }
    }
  };
  auto piece_ends = [&](int t) {
    if ((t & ((1 << ltiles) - 1)) != (1 << ltiles) - 1) return;
    int *out = C + (size_t)piece_of(t) * (ROWS_A * ROWS_B);
#pragma unroll
    for (int n = 0; n < NW; n++)
#pragma unroll
      for (int e = 0; e < 4; e++) {
        const int q = (wn * NW + n) * 8 + 2 * t4 + (e & 1);
        if (q < ROWS_B) {
#pragma unroll
          for (int m = 0; m < MW; m++) {
            const int r = (wmr * MW + m) * 16 + g + 8 * (e >> 1);
            if (r < ROWS_A) out[q * ROWS_A + r] = sum[m][n][e];
          }
        }
      }
    clear();
  };
  clear();
  fetch(0), fetch(1);
  __pipeline_wait_prior(1);
  __syncwarp(); // a warp reads only its own part of the table
  make(0);
  __syncthreads();
  for (int t = 0; t < tiles; t++) {
    __syncwarp();
    fetch(t + 2); // overwrites tile t's part of the table, which the warp's `make` has read
    __pipeline_wait_prior(1); // the thread's own copies for tile t + 1 have arrived
    __syncwarp();
    if (warp < 4) {
      product(t);
      make(t + 1);
    } else {
      make(t + 1);
      product(t);
    }
    piece_ends(t);
    __syncthreads();
  }
}
// Reduces the exact sums to canonical field elements, the same 520 per chunk as
// fk::k_join_poly<fk::SecondRows<fk::RangeSrc>>. out[520 ch + p] is eqhi[ch] times chunk ch's
// sum of w U_p for p < 8, and of w U_a times right key j' of slab a' for
// p = 8 + 64 j' + 8 a + a'. `pieces`: pieces per chunk. A warp computes one element: lane 9 j + m
// adds up the sums of byte m of coordinate j.
__global__ void __launch_bounds__(128)
    k_join_second(const int *__restrict__ C, int pieces, int nchunks, const T *__restrict__ eqhi, T *__restrict__ out) {
  using namespace second;
  const int id = (blockIdx.x * blockDim.x + threadIdx.x) >> 5, l = threadIdx.x & 31;
  if (id >= nchunks * SUMS) return;
  const int ch = id / SUMS, p = id % SUMS, q = p - 8;
  const bool alone = p < 8;
  const int a = alone ? p : (q >> 3) & 7, first = alone ? 128 : 16 * (q & 7) + 2 * (q >> 6);
  u64 s = 0;
  if (l < 27) {
    const int *r = C + (size_t)ch * pieces * (ROWS_A * ROWS_B) + 27 * a + l;
    for (int b = 0; b < (alone ? 1 : 2); b++) {
      long long v = 0;
      for (int x = 0; x < pieces; x++) v += r[(size_t)x * (ROWS_A * ROWS_B) + (first + b) * ROWS_A];
      s = gf_add(s, grm::signed_turn(v, (8 * (l % 9) + 8 * b) % grm::TURNS));
    }
  }
  u64 whole = s;
  for (int k = 1; k < 9; k++) whole = gf_add(whole, __shfl_sync(~0u, s, (l < 27 ? l / 9 * 9 : 0) + k));
  const u64 x0 = __shfl_sync(~0u, whole, 0), x1 = __shfl_sync(~0u, whole, 9), x2 = __shfl_sync(~0u, whole, 18);
  if (l == 0) out[id] = F::mul(K3{x0, x1, x2}, eqhi[ch]);
}
inline void second_range(const fk::Tree &t, int dT, int chunk, const u32 *table, size_t c0, size_t n, int k, int *C, cudaStream_t s) {
  static const bool once = [] { // not during a stream capture: the engine's first proof is launched directly
    CK_ATTRIBUTE(k_second_range, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)second::BYTES);
    return true;
  }();
  (void)once;
  int ltiles = 0;
  while ((COLS << ltiles) < k) ltiles++;
  if ((COLS << ltiles) != k || n % k || k > second::PIECE_MOST || chunk % COLS || chunk > 65536) {
    printf("internal error: look-up forest: the last layer's second pass over %zu indices in pieces of %d\n", n, k);
    fflush(stdout);
    name_the_end("forest-pass-shape"); // serving: p258 end 1 forest-pass-shape (kept as an exit(1))
    exit(1);
  }
  const int pieces = (int)(n / k);
  k_second_range<<<std::min(pieces, g_sms), THREADS, second::BYTES, s>>>(t.keys, dT, chunk, table, c0, pieces, ltiles, C);
  CK(cudaGetLastError());
}
inline void join_second(const int *C, int pieces, int nc, const T *eqhi, T *out, cudaStream_t s) {
  k_join_second<<<(nc * second::SUMS + 3) / 4, 128, 0, s>>>(C, pieces, nc, eqhi, out);
  CK(cudaGetLastError());
}
} // namespace ff
