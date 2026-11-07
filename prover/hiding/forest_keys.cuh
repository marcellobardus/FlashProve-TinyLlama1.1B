// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Key side of the lookup forest (the product trees of a group of lookup calls). struct Forest
// (forest.cuh) does not store the levels nearest a tree's leaves. For a product layer (one level's
// sumcheck) that reads one, this file computes two digit matrices from the tree's keys, the tree's
// share of the layer's sums from their product (the caller's), and the folded tables. build() fills
// the deepest stored levels. No function synchronizes with the host, allocates GPU memory or reads
// Table::lut; each launcher uses the stream it is given. Canonical inputs (coordinates < p) give
// canonical results. Tree::keys: 16-byte aligned (not checked). Include after product_test.cuh.
// A tree has N = 2^dT keys. A key is a table row number k of 16 bits; tau(k) is the row's value.
// leaf(i) = alpha - tau(keys[i]); `alpha` points to the challenge of the tree's call (GPU memory).
//   Level d:   node_d(x) = the product over h < 2^(dT - d) of leaf(h 2^d + x),   x < 2^d.
// Layer LV is the product layer that reads level d = dT - LV. Its tables are L(x') = node_d(x') and
// R(x') = node_d(2^(d-1) + x'), x' < 2^(d-1). In the default order its sumcheck binds x' from the
// top bit: rounds 1 to 3 bind the slab number a. With SC = 2^(d-4), the slabs (eighths of L, R) are
//   L_a(c) = node_d(a SC + c),   R_a(c) = node_d((8 + a) SC + c),   a < 8, c < SC.
// gram_products.cuh defines pass, index, entry, digit, digit matrix, chunk, piece, sum, eqlo, eqhi.
// Only third_fold() needs weights that sum to 1: it computes alpha - sum_j K64[j] tau_j.
#pragma once
#include "hiding/forest_plan.h"
#include "common/end_name.h" // name_the_end(): names an internal stop before a direct exit (M5e3)
#include <mutex>
#include <set>
#include <type_traits>
namespace fk { // the lookup forest's key side: this file, and Table and Tree in forest_plan.h
typedef grm::i8 i8;
// Row counts of a pass's first and second matrix; number of elements join() writes per chunk.
struct Shape {
  int wide, narrow, sums;
};
// ---- A node of D = 2^LV <= 4 leaves as a polynomial in alpha.
// With t_1 .. t_D the values of the node's keys and e_i their elementary symmetric polynomials
// (e_0 = 1),   node = prod_j (alpha - t_j) = sum_{i = 0 .. D} (-1)^i e_i alpha^(D - i).
// The e_i are in the base field and depend on the keys only; alpha is in the cubic extension (`T`).
// The layers LV <= 2 use this on both sides of G[a][a'] = sum_c w(c) L_a(c) R_a'(c), w = eqhi eqlo:
// - First matrix: eqlo L_a = W_0 + sum_(i >= 1) e_i W_i with W_i = (-1)^i alpha^(D - i) eqlo, which
//   weights() tabulates. An entry costs D products e_i W_i and one reduction for each coordinate.
// - Second matrix, byte form: the bytes of the e_i of R_a', not the digits of R_a'. A pass gives
//   S_0[a] = sum_c w L_a and S_i[a][a'] = sum_c w L_a e_i(R_a'); k_finish_poly turns them into G.
//   The bytes are stored minus 128, beside a row of ones. Public sources: [gemmlowp], [Dakkak];
//   compare [oneDNN]. The tags are explained at the top of gram_products.cuh.
// - Second matrix, digit form, for four leaves of a function table (any table but a range table),
//   whose e_i would take more rows than the digits: R_a' is computed as eqlo L_a is, with eqlo = 1.
// A "loose" number is any 64-bit word, standing for its residue mod p (`FIELD_P`). Digits and bytes may
// be those of any representative, because join() reduces the sums mod p. No public source was
// looked for: worked out from U4<FE>::get() (pt_transcript.h).
constexpr int LOOSE_BITS = 64;
__device__ __forceinline__ u64 loose(u64 x) { // any x
  return x;
}
__device__ __forceinline__ u64 mul_loose(u64 a, u64 b) { // any two words
  return blk::lazy::mul(blk::lazy::Loose{a}, blk::lazy::Loose{b}).w;
}
// c[i - 1] = e_i(t). SMALL: t < 2^16, and the e_i are exact integers of PolyRows::bits(i) bits.
// Otherwise t < 2^61, which table_of() (forest_plan.h) tests: e_1 is exact, the other e_i loose.
template <int D, bool SMALL> struct Coefficients;
template <bool SMALL> struct Coefficients<1, SMALL> {
  static __device__ __forceinline__ void of(const u64 (&t)[1], u64 (&c)[1]) { c[0] = t[0]; }
};
template <> struct Coefficients<2, true> {
  static __device__ __forceinline__ void of(const u64 (&t)[2], u64 (&c)[2]) {
    c[0] = (u32)t[0] + (u32)t[1], c[1] = (u32)t[0] * (u32)t[1];
  }
};
template <> struct Coefficients<2, false> {
  static __device__ __forceinline__ void of(const u64 (&t)[2], u64 (&c)[2]) {
    c[0] = t[0] + t[1], c[1] = mul_loose(t[0], t[1]);
  }
};
template <> struct Coefficients<4, true> {
  static __device__ __forceinline__ void of(const u64 (&t)[4], u64 (&c)[4]) {
    const u32 a = (u32)t[0] * (u32)t[1], b = (u32)t[2] * (u32)t[3], s = (u32)t[0] + (u32)t[1], u = (u32)t[2] + (u32)t[3];
    c[0] = s + u, c[1] = (u64)a + b + (u64)s * u, c[2] = (u64)a * u + (u64)b * s, c[3] = (u64)a * b;
  }
};
template <> struct Coefficients<4, false> {
  static __device__ __forceinline__ void of(const u64 (&t)[4], u64 (&c)[4]) {
    const u64 a = mul_loose(t[0], t[1]), b = mul_loose(t[2], t[3]), s = t[0] + t[1], u = t[2] + t[3];
    // s + u < 2^63 is exact. Adding loose words can overflow: each sum is kept wide, reduced once.
    using namespace blk::lazy;
    AccF z = zero_f();
    mac_words(z, a, 1), mac_words(z, b, 1), mac_words(z, s, u);
    c[0] = s + u, c[1] = reduce_loose(z);
    z = zero_f();
    mac_words(z, a, u), mac_words(z, b, s);
    c[2] = reduce_loose(z);
    c[3] = mul_loose(a, b);
  }
};
// W_1 .. W_D, each coordinate split by l3() (pt_field.h) into two 32-bit limbs; W_0 unsplit.
template <int D> struct Powers {
  u32 l[D][3], h[D][3];
  u64 w0[3];
  __device__ void put(int i, const T &v) { // W_i = v, 1 <= i <= D
    const L3 limbs = l3(v);
    for (int j = 0; j < 3; j++) l[i - 1][j] = limbs.l[j], h[i - 1][j] = limbs.h[j];
  }
  __device__ void set(const T &alpha, const T &times) { // W_i = (-1)^i alpha^(D - i) 'times', 0 <= i <= D
    T u = times;
    for (int i = D; i >= 1; i--) {
      put(i, (i & 1) ? F::sub(F::zero(), u) : u);
      u = F::mul(u, alpha);
    }
    for (int j = 0; j < 3; j++) w0[j] = coord(u, j);
  }
};
// A source of the W_i for add_dots(): the same constants for all four elements.
template <int D> struct FromPowers {
  const Powers<D> *p;
  __device__ __forceinline__ void limbs(int i, int j, u32 (&l)[4], u32 (&h)[4]) const {
#pragma unroll
    for (int x = 0; x < 4; x++) l[x] = p->l[i][j], h[x] = p->h[i][j];
  }
  __device__ __forceinline__ void zeroth(int j, u64 (&w)[4]) const {
#pragma unroll
    for (int x = 0; x < 4; x++) w[x] = p->w0[j];
  }
};
// x[k] += sum_i cf[k][i - 1] (coordinate j of W_i), FROM < i <= TO; x is loose before and after.
// W = l + 2^32 h with l, h < 2^32. A coefficient c < 2^28 adds c l < 2^60 to the 64-bit sum LL and
// c h < 2^60 to MID, so eight fit. x, LL, MID and the products c W of the longer coefficients are
// added in one wide sum (blk::lazy::AccF), which is reduced once.
template <class P, int FROM, int TO> __host__ __device__ constexpr bool some_long_coefficient() {
  for (int i = FROM; i < TO; i++)
    if (P::bits(i + 1) > 28) return true;
  return false;
}
template <class P, int FROM, int TO, class From>
__device__ __forceinline__ void add_some_dots(const u64 (&cf)[4][P::D], const From &from, int j, u64 (&x)[4]) {
  static_assert(TO - FROM <= 8, "the two sums of the short coefficients would overflow");
  using namespace blk::lazy;
  u64 LL[4] = {0, 0, 0, 0}, MID[4] = {0, 0, 0, 0};
  AccF wide[4];
#pragma unroll
  for (int i = FROM; i < TO; i++)
    if (P::bits(i + 1) <= 28) {
      u32 l[4], h[4];
      from.limbs(i, j, l, h);
#pragma unroll
      for (int k = 0; k < 4; k++) LL[k] = wm((u32)cf[k][i], l[k], LL[k]), MID[k] = wm((u32)cf[k][i], h[k], MID[k]);
    }
#pragma unroll
  for (int k = 0; k < 4; k++) { // x + LL + 2^32 MID, below 2^97
    const unsigned __int128 t = (unsigned __int128)x[k] + LL[k] + ((unsigned __int128)MID[k] << 32);
    wide[k] = AccF{{(u32)t, (u32)(t >> 32), (u32)(t >> 64), (u32)(t >> 96), 0}, {0, 0, 0}};
  }
  if constexpr (some_long_coefficient<P, FROM, TO>()) {
#pragma unroll
    for (int i = FROM; i < TO; i++)
      if (P::bits(i + 1) > 28) {
        u32 l[4], h[4];
        from.limbs(i, j, l, h);
#pragma unroll
        for (int k = 0; k < 4; k++) mac_words(wide[k], cf[k][i], l[k] | ((u64)h[k] << 32));
      }
  }
#pragma unroll
  for (int k = 0; k < 4; k++) x[k] = reduce_loose(wide[k]);
}
template <class P, class From>
__device__ __forceinline__ void add_dots(const u64 (&cf)[4][P::D], const From &from, int j, u64 (&x)[4]) {
  add_some_dots<P, 0, P::D>(cf, from, j, x);
}
__device__ __forceinline__ u64 canonical(u64 x) { // of a loose x
  return x >= FIELD_P ? x - FIELD_P : x;
}
// For add_dots(): every coefficient may be any loose number.
template <int D_> struct LooseRows {
  static constexpr int D = D_;
  static __host__ __device__ constexpr int bits(int) { return LOOSE_BITS; }
};
// The canonical node of D keys with values t < 2^61 (see Coefficients). `powers`: set(alpha, 1).
template <int D> __device__ __forceinline__ T node_of_values(const u64 (&t)[D], const Powers<D> *powers) {
  u64 cf[4][D], x[3][4];
  Coefficients<D, false>::of(t, cf[0]);
#pragma unroll
  for (int k = 1; k < 4; k++)
#pragma unroll
    for (int i = 0; i < D; i++) cf[k][i] = 0; // add_dots() takes four nodes; three are zero here
#pragma unroll
  for (int j = 0; j < 3; j++) {
    FromPowers<D>{powers}.zeroth(j, x[j]);
    add_dots<LooseRows<D>>(cf, FromPowers<D>{powers}, j, x[j]);
  }
  return {canonical(x[0][0]), canonical(x[1][0]), canonical(x[2][0])};
}
// ---- Where the kernels take the keys' values from. For each kind of table (Kind, forest_plan.h):
// a Src, which the host passes to a kernel, and its Form, which every thread sets up once.
//   value(k) = tau(k), below 2^BITS;   bytes0(k): NB0 bytes with tau(k) = sum_b byte b 2^turn0(b);
//   quad(k0, .., k3): the product of the four leaves alpha - tau(k), canonical.
// init() sets up all three; a kernel that does not call quad() uses init_values(), without alpha.
// THREADS is the kernels' block size, shared(src) the bytes of dynamic shared memory that a block
// needs: it copies a small table there, because neighbouring threads read it at unrelated places.
//
// Public sources for these techniques.
// The idea is public: a lookup table staged in shared memory:
// A. Di Biagio, A. Barenghi, G. Agosta, G. Pelosi, "Design of a Parallel AES for Graphics Hardware
// using the CUDA framework", section 3.3 and Figure 4 (there: AES's T-boxes, read at
// input-dependent, scattered indices), https://barenghi.faculty.polimi.it/files/IPDPS2009.pdf.
// The idea is public: a vectorized copy, here by uint4:
// "CUDA Pro Tip: Increase Performance with Vectorized Memory Access",
// a post on NVIDIA's developer blog (a vector load or store needs aligned data),
// https://developer.nvidia.com/blog/cuda-pro-tip-increase-performance-with-vectorized-memory-access/.
// The idea is public: read-only data read with __ldg():
// NVIDIA, Kepler Tuning Guide, "Memory Throughput", part "Read-Only Data Cache",
// https://docs.nvidia.com/cuda/kepler-tuning-guide/.
// The guide speaks of the GPU GK110.
// The loop form is public: the block-stride loop:
// R. Van der Wijngaart and F. Oh, "Boosting Application Performance with GPU Memory Prefetching",
// NVIDIA Technical Blog, 2022, "Unrolling",
// https://developer.nvidia.com/blog/boosting-application-performance-with-gpu-memory-prefetching/.
struct RangeSrc {};
struct LimbSrc {
  const u8 *tab8;
};
struct PackedSrc {
  const uint16_t *ctab;
};
template <int BYTES> struct RowSrc { // the row values are below 2^(8 BYTES) and below 2^61; BYTES = 6 or 8
  const u64 *tau;
  u32 rows;
};
template <class S> struct Form;
// Copies `bytes` bytes (a multiple of 16) to shared memory, then synchronizes: all threads call it.
__device__ __forceinline__ void stage(void *to, const void *from, int bytes) {
  for (int w = threadIdx.x; w < bytes / 16; w += blockDim.x)
    reinterpret_cast<uint4 *>(to)[w] = __ldg(reinterpret_cast<const uint4 *>(from) + w);
  __syncthreads();
}
// quad() of a function table, as a polynomial in alpha. Its constants are in the first POWERS_BYTES
// bytes of dynamic shared memory: set_powers() fills them, and a __syncthreads() must follow.
constexpr int POWERS_BYTES = 256;
template <class Derived> struct ByValues {
  const Powers<4> *four;
  __device__ __forceinline__ void set_powers(const T &alpha, void *sh) {
    Powers<4> *p = reinterpret_cast<Powers<4> *>(sh);
    if (threadIdx.x == 0) p->set(alpha, F::one());
    four = p;
  }
  __device__ __forceinline__ T quad(u32 k0, u32 k1, u32 k2, u32 k3) const {
    const Derived &f = *static_cast<const Derived *>(this);
    const u64 t[4] = {f.value(k0), f.value(k1), f.value(k2), f.value(k3)};
    return node_of_values<4>(t, four);
  }
};
// A range table: tau(k) = k. quad() is U4::get() (pt_transcript.h).
template <> struct Form<RangeSrc> {
  static constexpr int THREADS = 256;
  static __host__ __device__ int shared(const RangeSrc &) { return 0; }
  U4<F> u4;
  __device__ __forceinline__ void init(const RangeSrc &, const T &alpha, void *) { u4.prep(alpha); }
  __device__ __forceinline__ T quad(u32 k0, u32 k1, u32 k2, u32 k3) const { return u4.get(k0, k1, k2, k3); }
  static constexpr int BITS = 16, NB0 = 2;
  __device__ __forceinline__ void init_values(const RangeSrc &, void *) {}
  __device__ __forceinline__ u64 value(u32 k) const { return k; }
  __device__ __forceinline__ u64 bytes0(u32 k) const { return k; }
  static __host__ __device__ constexpr int turn0(int b) { return 8 * b; }
};
// A byte-limb table: its 2^16 bytes in shared memory. tab8: 16-byte aligned (not checked).
template <> struct Form<LimbSrc> : ByValues<Form<LimbSrc>> {
  static constexpr int THREADS = 256;
  static __host__ __device__ int shared(const LimbSrc &) { return POWERS_BYTES + 65536; }
  const u8 *tab;
  __device__ __forceinline__ void init(const LimbSrc &s, const T &alpha, void *sh) {
    set_powers(alpha, sh);
    init_values(s, sh);
  }
  static constexpr int BITS = 48, NB0 = 3;
  __device__ __forceinline__ void init_values(const LimbSrc &s, void *sh) {
    tab = (const u8 *)sh + POWERS_BYTES;
    stage((u8 *)sh + POWERS_BYTES, s.tab8, 65536);
  }
  __device__ __forceinline__ u64 value(u32 k) const {
    return (u64)(k & 255) + ((u64)(k >> 8) << 20) + ((u64)tab[k] << 40);
  }
  __device__ __forceinline__ u64 bytes0(u32 k) const { return k | (u32)tab[k] << 16; }
  static __host__ __device__ constexpr int turn0(int b) { return 20 * b; }
};
// A packed table: its 2^16 16-bit entries in shared memory, so one block per multiprocessor. A key
// not below the row count is not reported: its entry is 0, so its value is the key, no row's value,
// and lookup_tie::k_tie() raises OWN_TEST (no proof). Same for a range table of under 2^16 rows.
template <> struct Form<PackedSrc> : ByValues<Form<PackedSrc>> {
  static constexpr int THREADS = 512;
  static __host__ __device__ int shared(const PackedSrc &) { return POWERS_BYTES + 131072; }
  const uint16_t *tab;
  __device__ __forceinline__ void init(const PackedSrc &s, const T &alpha, void *sh) {
    set_powers(alpha, sh);
    init_values(s, sh);
  }
  static constexpr int BITS = 44, NB0 = 4;
  __device__ __forceinline__ void init_values(const PackedSrc &s, void *sh) {
    tab = (const uint16_t *)((u8 *)sh + POWERS_BYTES);
    stage((u8 *)sh + POWERS_BYTES, s.ctab, 131072);
  }
  __device__ __forceinline__ u64 value(u32 k) const { return (u64)k + ((u64)tab[k] << 28); }
  __device__ __forceinline__ u64 bytes0(u32 k) const { return k | (u32)tab[k] << 16; }
  static __host__ __device__ constexpr int turn0(int b) { return b < 2 ? 8 * b : 12 + 8 * b; }
};
// Any other table (GENERAL): the row values are read from GPU memory, 8 bytes per key. A key not
// below `rows` is reported (index_below(), prover/common/index_guard.cuh) and treated as key 0.
template <int BYTES> struct Form<RowSrc<BYTES>> : ByValues<Form<RowSrc<BYTES>>> {
  static constexpr int THREADS = 256;
  static __host__ __device__ int shared(const RowSrc<BYTES> &) { return POWERS_BYTES; }
  RowSrc<BYTES> from;
  __device__ __forceinline__ void init(const RowSrc<BYTES> &s, const T &alpha, void *sh) {
    from = s;
    this->set_powers(alpha, sh);
    __syncthreads();
  }
  static constexpr int BITS = BYTES == 8 ? 61 : 8 * BYTES, NB0 = BYTES;
  __device__ __forceinline__ void init_values(const RowSrc<BYTES> &s, void *) { from = s; }
  __device__ __forceinline__ u64 value(u32 k) const {
    return __ldg(from.tau + index_below(k, from.rows, INDEX_FUNCTION_KEY));
  }
  __device__ __forceinline__ u64 bytes0(u32 k) const { return value(k); }
  static __host__ __device__ constexpr int turn0(int b) { return 8 * b; }
};
// Calls fn with the table's Src.
template <class Fn> inline void with_src(const Table &tab, const Fn &fn) {
  if (tab.kind == RANGE) return fn(RangeSrc{});
  if (tab.kind == BYTE_LIMBS) return fn(LimbSrc{tab.tab8});
  if (tab.kind == PACKED) return fn(PackedSrc{tab.ctab});
  if (tab.bits <= 48) return fn(RowSrc<6>{tab.tau, 1u << tab.k});
  fn(RowSrc<8>{tab.tau, 1u << tab.k});
}
// Calls fn with `number` as a compile-time constant. Exits with code 1 unless FROM <= number <= TO.
template <int FROM, int TO, class Fn> inline void with_constant(int number, const Fn &fn) {
  if (number == FROM) return fn(std::integral_constant<int, FROM>{});
  if constexpr (FROM < TO) return with_constant<FROM + 1, TO>(number, fn);
  printf("internal error: look-up forest (forest_keys.cuh): %d is above %d, the last value its kernels are compiled for, or "
         "below the first\n",
         number, TO);
  fflush(stdout);
  name_the_end("forest-pass-shape"); // serving: p258 end 1 forest-pass-shape (kept as an exit(1))
  exit(1);
}
// launch_of(): the launch of a kernel that sets up a Form<S> and has work for `threads` threads: no more
// blocks than an H100 runs at once if each multiprocessor is counted with 2,048 threads and 227 KB for its
// blocks' shared memory (registers and static shared memory are not counted). The first call for a
// kernel that needs over 48 KB opts in; as with cuBLASLt's plans, this must not happen in a stream capture.
//
// Public sources: the shared-memory opt-in above 48 KB:
// CUDA C++ Programming Guide, "Compute Capabilities", "Compute Capability 9.0", "Shared Memory".
// See also, for the 48 KB: NVIDIA, CUDA Programming Guide, "Advanced Kernel Programming",
// "Configuring L1/Shared Memory Balance" (a kernel's opt-in is cudaFuncSetAttribute(), as below;
// only dynamic memory will do, no fixed-size array; whether more is usable varies by architecture),
// https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/advanced-kernel-programming.html.
// The idea is public: a limit on a grid-stride loop's blocks:
// M. Harris, "CUDA Pro Tip: Write Flexible Kernels with Grid-Stride Loops",
// NVIDIA Technical Blog,
// https://developer.nvidia.com/blog/cuda-pro-tip-write-flexible-kernels-grid-stride-loops/.
struct Launch {
  int blocks, threads, shared;
};
template <class S> inline Launch launch_of(const S &src, const void *kernel, size_t threads) {
  Launch l{0, Form<S>::THREADS, Form<S>::shared(src)};
  if (l.shared > 49152) {
    static std::mutex m;
    static std::set<const void *> asked;
    std::lock_guard<std::mutex> lock(m);
    if (asked.insert(kernel).second)
      CK_ATTRIBUTE(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, l.shared);
  }
  const int at_once = std::min(227 * 1024 / (l.shared + 1024), 2048 / l.threads);
  l.blocks = (int)std::min<size_t>((threads + l.threads - 1) / l.threads, (size_t)g_sms * at_once);
  return l;
}
__device__ __forceinline__ uint2 four_keys(const uint16_t *at) {
  return __ldg(reinterpret_cast<const uint2 *>(at));
}
__device__ __forceinline__ u32 key_of(const uint2 &q, int k) {
  const u32 w = k < 2 ? q.x : q.y;
  return (k & 1) ? w >> 16 : w & 0xFFFFu;
}
// e[k] = node_d(x + k), k < 4, d = dT - LV, LV >= 2: each is the product of 2^LV leaves whose keys
// are 2^d apart. `at` points to keys[x]; x is a multiple of 4.
template <int LV, class Fm>
__device__ __forceinline__ void four_nodes(const Fm &f, const uint16_t *__restrict__ at, int d, T *e) {
#pragma unroll
  for (int g = 0; g < (1 << LV) / 4; g++) {
    uint2 q[4];
#pragma unroll
    for (int j = 0; j < 4; j++) q[j] = four_keys(at + ((size_t)(4 * g + j) << d));
#pragma unroll
    for (int k = 0; k < 4; k++) {
      const T v = f.quad(key_of(q[0], k), key_of(q[1], k), key_of(q[2], k), key_of(q[3], k));
      e[k] = g ? F::mul(e[k], v) : v;
    }
  }
}

// Sizes of a layer LV <= 2 of a table with Src S. BYTE_FORM: the second matrix is in byte form.
template <int LV, class S> struct PolyRows {
  static constexpr int D = 1 << LV, BITS = Form<S>::BITS;
  static constexpr bool SMALL = BITS == 16, BYTE_FORM = SMALL || LV < 2;
  static __host__ __device__ constexpr int bits(int i) { // of e_i as Coefficients gives it, 1 <= i <= D
    if (SMALL) return D == 1 ? 16 : D == 2 ? (i == 1 ? 17 : 32) : i == 1 ? 18 : i == 2 ? 35 : i == 3 ? 50 : 64;
    return D == 1 ? BITS : D == 2 && i == 1 ? BITS + 1 : LOOSE_BITS;
  }
  // Byte form: e_i takes nb(i) of a slab's NBT rows, starting at row off(i); its row b is scaled by
  // 2^turn(b). At LV 0 the rows are the Form's bytes0().
  static __host__ __device__ constexpr int nb(int i) { return LV == 0 ? Form<S>::NB0 : (bits(i) + 7) / 8; }
  static __host__ __device__ constexpr int off(int i) {
    int o = 0;
    for (int j = 1; j < i; j++) o += nb(j);
    return o;
  }
  static __host__ __device__ constexpr int turn(int b) { return LV == 0 ? Form<S>::turn0(b) : 8 * b; }
};
template <int LV, class S> struct Poly : PolyRows<LV, S> {
  using PolyRows<LV, S>::D;
  using PolyRows<LV, S>::BYTE_FORM;
  static constexpr int NBT = PolyRows<LV, S>::off(D + 1);
  // The first matrix has 24 rows per slab. The second has NBT rows of bytes minus 128 per slab and
  // then a row of ones and three of zeros; or, in digit form, 24 rows of digits per slab.
  static constexpr int NARROW = BYTE_FORM ? 8 * NBT + 4 : 192, SUMS = BYTE_FORM ? 8 + 64 * D : 64;
};
// The table of weights(): 6 D planes of `chunk` 32-bit words (plane 2 (3 (i - 1) + j): the low
// limbs of coordinate j of W_i; the next plane: the high limbs), then 3 of 64-bit words (W_0).
inline size_t weights_words(int LV, int chunk) {
  return LV <= 2 ? (size_t)(6 * (1 << LV) + 6) * chunk : 0;
}
template <int D> __device__ void to_table(const Powers<D> &p, int chunk, int c, u32 *__restrict__ W) {
  for (int i = 0; i < D; i++)
    for (int j = 0; j < 3; j++) {
      W[(size_t)((3 * i + j) * 2) * chunk + c] = p.l[i][j];
      W[(size_t)((3 * i + j) * 2 + 1) * chunk + c] = p.h[i][j];
    }
  for (int j = 0; j < 3; j++) reinterpret_cast<u64 *>(W + (size_t)6 * D * chunk)[(size_t)j * chunk + c] = p.w0[j];
}
template <int D> __global__ void k_weights(const T *alpha, int chunk, const T *__restrict__ eqlo, u32 *__restrict__ W) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= chunk) return;
  Powers<D> p;
  p.set(*alpha, eqlo[c]);
  to_table(p, chunk, c, W);
}
// Fills W (weights_words(LV, chunk) words, 16-byte aligned) for a layer LV <= 2 from its weights
// eqlo. One table serves all trees that share alpha (the trees of one call). LV > 2: does nothing.
inline void weights(int LV, const T *alpha, int chunk, const T *eqlo, u32 *W, cudaStream_t s) {
  if (LV > 2) return;
  with_constant<0, 2>(LV, [&](auto lv) {
    k_weights<(1 << decltype(lv)::value)><<<(chunk + 127) / 128, 128, 0, s>>>(alpha, chunk, eqlo, W);
    CK(cudaGetLastError());
  });
}
// A warp handles 128 indices, from index c on. A lane produces one 32-bit word of each row: the
// bytes of the indices c + place(lane, i), i < 4. At LV 0, index c of slab a is key 8 c + a of its
// half of the tree, and with lane + 32 i adjacent lanes read adjacent indices. Matrices and weights
// share the order, so no sum depends on it. Code with no LV passes IN_FOURS or BY_LANES instead.
constexpr int BY_LANES = 0, IN_FOURS = 1;
template <int LV> __device__ __forceinline__ int place(int lane, int i) {
  return LV ? 4 * lane + i : lane + 32 * i;
}
template <bool NEAR, class X> __device__ __forceinline__ X read_one(const X *p) { // NEAR: from shared memory
  if constexpr (NEAR)
    return *p;
  else
    return __ldg(p);
}
// x[i] = plane[place(lane, i)]. `plane`: 16-byte aligned; shared memory if NEAR, else GPU memory.
template <int LV, bool NEAR> __device__ __forceinline__ void four_of(const u32 *plane, int lane, u32 (&x)[4]) {
  if constexpr (LV == 0) {
#pragma unroll
    for (int i = 0; i < 4; i++) x[i] = read_one<NEAR>(plane + place<LV>(lane, i));
  } else {
    const uint4 v = read_one<NEAR>(reinterpret_cast<const uint4 *>(plane) + lane);
    x[0] = v.x, x[1] = v.y, x[2] = v.z, x[3] = v.w;
  }
}
template <int LV, bool NEAR> __device__ __forceinline__ void four_of(const u64 *plane, int lane, u64 (&x)[4]) {
  if constexpr (LV == 0) {
#pragma unroll
    for (int i = 0; i < 4; i++) x[i] = read_one<NEAR>(plane + place<LV>(lane, i));
  } else {
    const ulonglong2 v = read_one<NEAR>(reinterpret_cast<const ulonglong2 *>(plane) + 2 * lane),
                     w = read_one<NEAR>(reinterpret_cast<const ulonglong2 *>(plane) + 2 * lane + 1);
    x[0] = v.x, x[1] = v.y, x[2] = w.x, x[3] = w.y;
  }
}
// The keys of a lane's four nodes of slab `slab` (0 to 7: L; 8 to 15: R): key_of(q[j], i) is key j
// of the lane's node i. LV >= 1: default order; LV 0: range order (see "A layer made from keys").
template <int LV> struct KeySets {
  uint2 q[1 << LV];
};
template <int LV>
__device__ __forceinline__ KeySets<LV> key_sets(const uint16_t *__restrict__ keys, int dT, int slab, size_t c, int lane) {
  KeySets<LV> k;
  if constexpr (LV == 0) {
    const uint16_t *at = keys + ((size_t)(slab >> 3) << (dT - 1)) + 8 * (c + lane) + (slab & 7);
    k.q[0].x = (u32)__ldg(at) | (u32)__ldg(at + 256) << 16, k.q[0].y = (u32)__ldg(at + 512) | (u32)__ldg(at + 768) << 16;
  } else {
    const int d = dT - LV;
    const uint16_t *at = keys + ((size_t)slab << (d - 4)) + c + 4 * lane;
#pragma unroll
    for (int j = 0; j < (1 << LV); j++) k.q[j] = four_keys(at + ((size_t)j << d));
  }
  return k;
}
// cf[i] = the D coefficients of the lane's node i. FOR_DOT: all below 2^LOOSE_BITS, for add_dots().
template <class P, bool FOR_DOT, class Fm, class Keys>
__device__ __forceinline__ void four_coefficients(const Fm &f, const Keys &k, u64 (&cf)[4][P::D]) {
#pragma unroll
  for (int i = 0; i < 4; i++) {
    u64 t[P::D];
#pragma unroll
    for (int j = 0; j < P::D; j++) t[j] = f.value(key_of(k.q[j], i));
    Coefficients<P::D, P::SMALL>::of(t, cf[i]);
    if constexpr (FOR_DOT && P::SMALL && P::D == 4) cf[i][3] = loose(cf[i][3]);
  }
}
// A source of the W_i for add_dots(): the table of weights() at the lane's four indices.
template <int LV, bool NEAR = false> struct FromTable {
  const u32 *W; // the table, from the warp's first index on
  int chunk, lane;
  const u64 *w0; // coordinate 0 of W_0 at the warp's first index: the 64-bit planes begin 6 D chunk words after the table's start
  __device__ __forceinline__ void limbs(int i, int j, u32 (&l)[4], u32 (&h)[4]) const {
    four_of<LV, NEAR>(W + (size_t)((3 * i + j) * 2) * chunk, lane, l);
    four_of<LV, NEAR>(W + (size_t)((3 * i + j) * 2 + 1) * chunk, lane, h);
  }
  __device__ __forceinline__ void zeroth(int j, u64 (&w)[4]) const { four_of<LV, NEAR>(w0 + (size_t)j * chunk, lane, w); }
};
// Copies to shared memory the part of table W for the 128 indices from index cc of a chunk on, as a
// table for a chunk of 128 indices ((6 D + 6) 128 words). `threads` threads share the copy; this
// thread is number `thread`. The caller synchronizes. One GPU memory read then serves eight warps.
// The idea is public: a block's threads share loaded data:
// "Using Shared Memory in CUDA C/C++",
// a post on NVIDIA's developer blog,
// https://developer.nvidia.com/blog/using-shared-memory-cuda-cc/.
template <int D>
__device__ __forceinline__ void table_near(const u32 *__restrict__ W, int chunk, int cc, u32 *near, int thread, int threads) {
  for (int u = thread; u < 6 * D * 32 + 192; u += threads) {
    const int v = u - 6 * D * 32;
    const u32 *from = v < 0 ? W + (size_t)(u >> 5) * chunk + cc + 4 * (u & 31)
                            : W + (size_t)6 * D * chunk + 2 * ((size_t)(v >> 6) * chunk + cc) + 4 * (v & 63);
    reinterpret_cast<uint4 *>(near)[u] = __ldg(reinterpret_cast<const uint4 *>(from));
  }
}
// Passes the 24 digit rows of the four elements W_0 + sum_i cf[.][i - 1] W_i to put(row, word).
template <class P, class From, class Put>
__device__ __forceinline__ void rows_of_dot(const u64 (&cf)[4][P::D], const From &from, const Put &put) {
#pragma unroll
  for (int j = 0; j < 3; j++) {
    u64 x[4];
    from.zeroth(j, x);
    add_dots<P>(cf, from, j, x);
    u32 w[8];
    grm::digit_words(x, w);
#pragma unroll
    for (int m = 0; m < 8; m++) put(8 * j + m, w[m]);
  }
}
// Passes the NBT byte rows of four nodes' coefficients, each byte minus 128, to put(row, word).
template <class P, class Put> __device__ __forceinline__ void rows_of_bytes(const u64 (&cf)[4][P::D], const Put &put) {
#pragma unroll
  for (int i = 1; i <= P::D; i++) {
    u32 w[8];
    const u32 lo[4] = {(u32)cf[0][i - 1], (u32)cf[1][i - 1], (u32)cf[2][i - 1], (u32)cf[3][i - 1]};
    grm::transpose_bytes(lo, w);
    if (P::nb(i) > 4) {
      const u32 hi[4] = {(u32)(cf[0][i - 1] >> 32), (u32)(cf[1][i - 1] >> 32), (u32)(cf[2][i - 1] >> 32),
                         (u32)(cf[3][i - 1] >> 32)};
      grm::transpose_bytes(hi, w + 4);
    }
#pragma unroll
    for (int b = 0; b < P::nb(i); b++) put(P::off(i) + b, w[b] ^ 0x80808080u);
  }
}
// Passes all rows of slab `slab` (0 to 7: the 24 rows of eqlo L_slab; 8 to 15: the rows of
// R_(slab - 8)) for a warp's 128 indices to put(row within the slab's rows, word). `k`: their keys
// (key_sets()). `near`: table_near()'s copy. `powers`: set(alpha, 1), read only in digit form.
template <int LV, class S, class Put>
__device__ __forceinline__ void rows_of_slab(const Form<S> &f, const KeySets<LV> &k, int slab, int lane, const u32 *near,
                                             const Powers<(1 << LV)> *powers, const Put &put) {
  typedef Poly<LV, S> P;
  u64 cf[4][P::D];
  if (slab < 8) {
    four_coefficients<P, true>(f, k, cf);
    rows_of_dot<P>(cf, FromTable<LV, true>{near, 128, lane, reinterpret_cast<const u64 *>(near + 6 * P::D * 128)}, put);
  } else if (!P::BYTE_FORM) {
    four_coefficients<P, true>(f, k, cf);
    rows_of_dot<P>(cf, FromPowers<P::D>{powers}, put);
  } else {
    if constexpr (LV == 0) {
#pragma unroll
      for (int i = 0; i < 4; i++) cf[i][0] = f.bytes0(key_of(k.q[0], i));
    } else {
      four_coefficients<P, false>(f, k, cf);
    }
    rows_of_bytes<P>(cf, put);
  }
}
// Kernel of digits() for LV <= 2. Eight warps share 128 indices: warp a writes slabs a and 8 + a.
template <int LV, class S>
__global__ void __launch_bounds__(Form<S>::THREADS)
    k_digits_poly(const uint16_t *__restrict__ keys, int dT, const T *alpha, const S src, size_t c0, int n, int chunk,
                  const u32 *__restrict__ W, i8 *__restrict__ A, i8 *__restrict__ B, size_t ld) {
  typedef Poly<LV, S> P;
  extern __shared__ __align__(16) unsigned char staged[];
  __shared__ Powers<P::D> powers;
  Form<S> f;
  f.init_values(src, staged);
  if (!P::BYTE_FORM) {
    if (threadIdx.x == 0) powers.set(*alpha, F::one());
    __syncthreads();
  }
  constexpr int GROUPS = Form<S>::THREADS / 256;
  __shared__ __align__(16) u32 near[GROUPS][(6 * P::D + 6) * 128];
  const int a = (threadIdx.x >> 5) & 7, lane = threadIdx.x & 31, group = threadIdx.x >> 8;
  for (int first = blockIdx.x * GROUPS * 128; first < n; first += gridDim.x * GROUPS * 128) {
    const int c = first + group * 128;
    const bool works = c < n;
    KeySets<LV> left, right;
    if (works) {
      left = key_sets<LV>(keys, dT, a, c0 + c, lane), right = key_sets<LV>(keys, dT, 8 + a, c0 + c, lane);
      table_near<P::D>(W, chunk, (int)((c0 + c) % chunk), near[group], threadIdx.x & 255, 256);
    }
    __syncthreads();
    if (works) {
      rows_of_slab<LV>(f, left, a, lane, near[group], &powers, [&](int row, u32 w) {
        *reinterpret_cast<u32 *>(A + (size_t)(a * 24 + row) * ld + c + 4 * lane) = w;
      });
      rows_of_slab<LV>(f, right, 8 + a, lane, near[group], &powers, [&](int row, u32 w) {
        *reinterpret_cast<u32 *>(B + (size_t)(a * (P::BYTE_FORM ? P::NBT : 24) + row) * ld + c + 4 * lane) = w;
      });
      if (P::BYTE_FORM && a < 4)
        *reinterpret_cast<u32 *>(B + (size_t)(8 * P::NBT + a) * ld + c + 4 * lane) = a ? 0u : 0x01010101u;
    }
    __syncthreads();
  }
}
// join() for the byte form. A chunk has 8 + 64 D outputs, out[chunk * SUMS + p]: p < 8 is S_0[p];
// p = 8 + 64 (i - 1) + 8 a + a' is S_i[a][a']. C, `pieces`, eqhi: see join(). 128 times the product
// with the row of ones undoes the bytes' offset of -128. Tags: see the top of gram_products.cuh.
// Public sources: [TensorFHE], [CUTLASS] (split K), [CUDA guide] (shuffles), as at grm::k_join;
// [gemmlowp] (128 * ones). The compiler hint is public: __launch_bounds__ [Best practices].
template <class P>
__global__ void __launch_bounds__(128)
    k_join_poly(const int *__restrict__ C, int pieces, int nchunks, const T *__restrict__ eqhi, T *__restrict__ out) {
  constexpr int size = 192 * P::NARROW;
  const int id = (blockIdx.x * blockDim.x + threadIdx.x) >> 5, l = threadIdx.x & 31;
  if (id >= nchunks * P::SUMS) return;
  const int ch = id / P::SUMS, p = id % P::SUMS, m = l & 7, q = p - 8;
  const bool alone = p < 8; // the right factor is 1
  const int i = alone ? 0 : (q >> 6) + 1, a = alone ? p : (q >> 3) & 7, ap = q & 7;
  const int *r = C + (size_t)ch * pieces * size + a * 24 + l;
  auto column = [&](int col) {
    long long v = 0;
    for (int x = 0; x < pieces; x++) v += r[(size_t)x * size + col * 192];
    return v;
  };
  u64 s = 0;
  // A warp has 24 rows of digits. Its last eight lanes read nothing, and their sum is not used.
  if (l < 24) {
    const long long ones = column(8 * P::NBT);
    if (alone)
      s = grm::signed_turn(ones, 8 * m);
    else
      for (int b = 0; b < P::nb(i); b++)
        s = gf_add(s, grm::signed_turn(column(ap * P::NBT + P::off(i) + b) + 128 * ones, (8 * m + P::turn(b)) % grm::TURNS));
  }
  for (int o = 4; o; o >>= 1) s = gf_add(s, __shfl_down_sync(~0u, s, o, 8));
  const u64 c0 = __shfl_sync(~0u, s, 0), c1 = __shfl_sync(~0u, s, 8), c2 = __shfl_sync(~0u, s, 16);
  if (l == 0) out[id] = F::mul(K3{c0, c1, c2}, eqhi[ch]);
}
// finish() for the byte form: G[8 a + a'] = alpha^D S_0[a] + sum_i (-1)^i alpha^(D - i) S_i[a][a'].
__global__ void k_finish_poly(const T *H, int D, const T *alpha, T *G) {
  const int t = threadIdx.x;
  const T alpha_here = *alpha;
  T r = H[t >> 3];
  for (int i = 1; i <= D; i++) {
    const T x = H[8 + 64 * (i - 1) + t];
    r = F::mul(r, alpha_here);
    r = (i & 1) ? F::sub(r, x) : F::add(r, x);
  }
  G[t] = r;
}

// ---- Eight neighbouring keys at a time, for the last layer (LV 0).
// A source of the W_i for add_dots(): W_1 and W_0 at a lane's four indices, in registers.
struct FromRegisters {
  u32 l[3][4], h[3][4];
  u64 w0[3][4];
  __device__ __forceinline__ void limbs(int, int j, u32 (&lo)[4], u32 (&hi)[4]) const {
#pragma unroll
    for (int x = 0; x < 4; x++) lo[x] = l[j][x], hi[x] = h[j][x];
  }
  __device__ __forceinline__ void zeroth(int j, u64 (&w)[4]) const {
#pragma unroll
    for (int x = 0; x < 4; x++) w[x] = w0[j][x];
  }
};
// q[i] = the eight keys from key side + STRIDE c' + 8 a on, read in one 128-bit load.
// c' = c + place<ORDER>(lane, i); side is N / 2 if `right`, else 0.
// The idea is public: "CUDA Pro Tip: Increase Performance with Vectorized Memory Access", cited
// above struct RangeSrc.
template <int ORDER, int STRIDE>
__device__ __forceinline__ void eight_keys(const uint16_t *__restrict__ keys, int dT, bool right, int a, size_t c, int lane,
                                           uint4 (&q)[4]) {
#pragma unroll
  for (int i = 0; i < 4; i++)
    q[i] = __ldg(reinterpret_cast<const uint4 *>(keys + (right ? (size_t)1 << (dT - 1) : 0) +
                                                 STRIDE * (c + place<ORDER>(lane, i))) + a);
}
// Passes the byte rows of the eight keys to put(j NB0 + b, word): row b of key j.
template <class S, class Put>
__device__ __forceinline__ void bytes_of_eight(const Form<S> &f, const uint4 (&q)[4], const Put &put) {
  typedef Poly<0, S> P;
  u64 cf[4][1];
#pragma unroll
  for (int j = 0; j < 8; j++) {
#pragma unroll
    for (int i = 0; i < 4; i++) cf[i][0] = f.bytes0(grm::key_of(q[i], j));
    rows_of_bytes<P>(cf, [&](int row, u32 w) { put(j * P::NBT + row, w); });
  }
}
// LV 0: passes to put(row, word) all rows of one side's 8 slabs at a warp's 128 indices from c on.
template <class S, class Put>
__device__ __forceinline__ void rows_of_eight(const Form<S> &f, const uint16_t *__restrict__ keys, int dT, bool right, size_t c,
                                              int lane, int chunk, const u32 *__restrict__ W, const Put &put) {
  typedef Poly<0, S> P;
  uint4 q[4];
  eight_keys<BY_LANES, 8>(keys, dT, right, 0, c, lane, q);
  if (right) return bytes_of_eight(f, q, put);
  u64 cf[4][1];
  const int cc = (int)(c % chunk);
  const FromTable<0> table{W + cc, chunk, lane, reinterpret_cast<const u64 *>(W + (size_t)6 * chunk) + cc};
  FromRegisters w;
#pragma unroll
  for (int x = 0; x < 3; x++) table.limbs(0, x, w.l[x], w.h[x]), table.zeroth(x, w.w0[x]);
#pragma unroll
  for (int j = 0; j < 8; j++) {
#pragma unroll
    for (int i = 0; i < 4; i++) cf[i][0] = f.value(grm::key_of(q[i], j));
    rows_of_dot<P>(cf, w, [&](int row, u32 x) { put(j * 24 + row, x); });
  }
}
// Kernel of digits() for LV 0 with many indices. A warp writes all rows of its 128 indices: it
// reads an index's eight keys (one per slab) in one load, and the weights once.
template <class S>
__global__ void __launch_bounds__(Form<S>::THREADS)
    k_digits_all_slabs(const uint16_t *__restrict__ keys, int dT, const S src, size_t c0, int n, int chunk,
                       const u32 *__restrict__ W, i8 *__restrict__ A, i8 *__restrict__ B, size_t ld) {
  typedef Poly<0, S> P;
  extern __shared__ __align__(16) unsigned char staged[];
  Form<S> f;
  f.init_values(src, staged);
  constexpr int WARPS = Form<S>::THREADS / 32;
  const int lane = threadIdx.x & 31;
  for (int c = (blockIdx.x * WARPS + (threadIdx.x >> 5)) * 128; c < n; c += gridDim.x * WARPS * 128) {
    rows_of_eight(f, keys, dT, false, c0 + c, lane, chunk, W,
                  [&](int row, u32 w) { *reinterpret_cast<u32 *>(A + (size_t)row * ld + c + 4 * lane) = w; });
    rows_of_eight(f, keys, dT, true, c0 + c, lane, chunk, W,
                  [&](int row, u32 w) { *reinterpret_cast<u32 *>(B + (size_t)row * ld + c + 4 * lane) = w; });
#pragma unroll
    for (int x = 0; x < 4; x++)
      *reinterpret_cast<u32 *>(B + (size_t)(8 * P::NBT + x) * ld + c + 4 * lane) = x ? 0u : 0x01010101u;
  }
}

// ---- Building the stored levels.
// For each i, a thread computes the 8 nodes of level dT - 2 whose product is node i of level
// dT - 5, then the 4, 2 and 1 nodes of levels dT - 3 to dT - 5. Level dT - LV is written if
// LV >= REB. REB is a template parameter: each instance is compiled without the others' branches,
// and build() chooses the instance at the launch.
// Public sources for the techniques of this routine.
// The idea is public: a branch on a template parameter:
// M. Harris, "Optimizing Parallel Reduction in CUDA", NVIDIA Developer Technology,
// https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf,
// slides "Complete Unrolling" to "Invoking Template Kernels".
// There the parameter is the block size, here the first level that is written. The table staged
// in shared memory: cited above struct RangeSrc. Read-only data are read with __ldg(), through
// pointers marked const and __restrict__: the Kepler Tuning Guide, cited above struct RangeSrc.
// According to the guide, the compiler uses the read-only cache by itself only with both marks.
// The compiler hints and the loop form are public: __launch_bounds__ with one argument: NVIDIA,
// CUDA C++ Best Practices Guide, "Occupancy",
// https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/; #pragma unroll: M. Milakov, "Fast
// Dynamic Indexing of Private Arrays in CUDA", cited at h_idk (product_test.cuh); the grid-stride
// loop: "CUDA Pro Tip: Write Flexible Kernels with Grid-Stride Loops", cited at struct Launch.
template <int REB, class S>
__global__ void __launch_bounds__(Form<S>::THREADS)
    k_build(const uint16_t *__restrict__ keys, int dT, const T *alpha, const S src, T *__restrict__ o2,
            T *__restrict__ o3, T *__restrict__ o4, T *__restrict__ o5) {
  extern __shared__ __align__(16) unsigned char staged[];
  Form<S> f;
  f.init(src, *alpha, staged);
  const size_t n8 = (size_t)1 << (dT - 5), M = 8 * n8;
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n8; i += (size_t)gridDim.x * blockDim.x) {
    T a[4];
#pragma unroll
    for (int k = 0; k < 4; k++) {
      T x[2];
#pragma unroll
      for (int h = 0; h < 2; h++) {
        const uint16_t *at = keys + i + (k + 4 * h) * n8;
        x[h] = f.quad(__ldg(at), __ldg(at + M), __ldg(at + 2 * M), __ldg(at + 3 * M));
        if (REB <= 2) st16(o2 + i + (k + 4 * h) * n8, x[h]);
      }
      a[k] = F::mul(x[0], x[1]);
      if (REB <= 3) st16(o3 + i + k * n8, a[k]);
    }
    const T b0 = F::mul(a[0], a[2]), b1 = F::mul(a[1], a[3]);
    if (REB <= 4) {
      st16(o4 + i, b0);
      st16(o4 + i + n8, b1);
    }
    st16(o5 + i, F::mul(b0, b1));
  }
}
// Writes the tree's levels d = dT - reb, dT - reb - 1, .., dT - 5 to lev[d] (2^d elements each).
// 2 <= reb <= 5 (otherwise exits with code 1). lev: a host array of GPU pointers, indexed by d.
inline void build(const Tree &t, int dT, int reb, const T *alpha, T *const *lev, cudaStream_t s) {
  with_src(t.tab, [&](auto src) {
    with_constant<2, 5>(reb, [&](auto REB) {
      auto kernel = k_build<decltype(REB)::value, decltype(src)>;
      const Launch l = launch_of(src, (const void *)kernel, (size_t)1 << (dT - 5));
      kernel<<<l.blocks, l.threads, l.shared, s>>>(t.keys, dT, alpha, src, reb <= 2 ? lev[dT - 2] : nullptr,
                                                   reb <= 3 ? lev[dT - 3] : nullptr, reb <= 4 ? lev[dT - 4] : nullptr,
                                                   lev[dT - 5]);
      CK(cudaGetLastError());
    });
  });
}

// ---- A layer made from keys, 0 <= LV <= 4: the layer that reads level d = dT - LV.
// LV >= 1 runs in the default order (top of this file). LV 0, the last layer, runs in the range
// order, whose rounds bind bits 2 1 0, 5 4 3, 8 7 6 of x' first. For LV 0 this section computes
// pass 1 (rounds 1 to 3), with SC = N / 16, L_a(c) = leaf(8 c + a), R_a(c) = leaf(N / 2 + 8 c + a).
// A pass gives the tree's share of the layer's Gram matrix G[a][a'] = sum_c w(c) L_a(c) R_a'(c),
// w(c) = eqhi[c / chunk] eqlo[c mod chunk]. shape(): the first matrix has 192 rows, the digits
// of eqlo L_a. {192, 192, 64}: the second has the digits of R_a, in the plain form (LV 3 and 4)
// and in Poly's digit form (LV 2 of a function table). All other layers have Poly's byte form.
inline Shape shape(int LV, const Table &tab) {
  Shape sh{grm::ROWS, grm::ROWS, 64};
  if (LV <= 2)
    with_src(tab, [&](auto src) {
      with_constant<0, 2>(LV, [&](auto lv) {
        typedef Poly<decltype(lv)::value, decltype(src)> P;
        sh.narrow = P::NARROW, sh.sums = P::SUMS;
      });
    });
  return sh;
}
// Kernel of digits() for the plain form, LV 3 and 4: an entry e is a node of 2^LV leaves. A gets
// the digits of eqlo e for slabs 0 to 7, B those of e for slabs 8 to 15. The warps share the work
// as in k_digits_poly.
// Public sources: [TensorFHE], [Ozaki-int8]; the loop: [Grid-stride].
// A and B both hold stacked digits: see [Multiword].
// Public sources for its other techniques.
// - The keys are read by __ldg(). The idea is public: [Kepler guide], which speaks of GK110.
// - The weights eqlo are read from a table in the prepared form M3L. The idea is public: [Harvey],
//   [Poly1305], [NEON Crypto].
// - The compiler hints are public: __launch_bounds__ [Best practices]; #pragma unroll [Milakov].
// The tags are explained at the top of gram_products.cuh.
template <int LV, class S>
__global__ void __launch_bounds__(Form<S>::THREADS)
    k_digits_plain(const uint16_t *__restrict__ keys, int d, const T *alpha, const S src, size_t c0, int n, int chunk,
                  const M3L *__restrict__ eqlo, i8 *__restrict__ A, i8 *__restrict__ B, size_t ld) {
  extern __shared__ __align__(16) unsigned char staged[];
  Form<S> f;
  f.init(src, *alpha, staged);
  constexpr int GROUPS = Form<S>::THREADS / 256;
  const int a = (threadIdx.x >> 5) & 7, lane = threadIdx.x & 31;
  const size_t SC = (size_t)1 << (d - 4);
  for (int c = (blockIdx.x * GROUPS + (threadIdx.x >> 8)) * 128 + lane * 4; c < n; c += gridDim.x * GROUPS * 128) {
    T e[4];
#pragma unroll
    for (int sd = 0; sd < 2; sd++) {
      four_nodes<LV>(f, keys + (size_t)(8 * sd + a) * SC + c0 + c, d, e);
      if (!sd) {
#pragma unroll
        for (int k = 0; k < 4; k++) e[k] = FK2::mulp(eqlo[(c0 + c) % chunk + k], e[k]);
      }
      grm::put_digits((sd ? B : A) + c, ld, a, e);
    }
  }
}
// Writes the digits of n indices from c0 on to A (192 rows) and B (shape().narrow rows): column 0
// is index c0, rows are ld bytes apart. Not checked: c0, n multiples of 128; chunk = 2^lc >= 128;
// A, B, ld multiples of 4. Within 128 columns the indices may be permuted, alike in A and B. W: the
// table of weights(LV, ..), read at LV <= 2. eqprep: eqlo in limbs, read at LV > 2.
inline void digits(int LV, const Tree &t, int dT, const T *alpha, size_t c0, int n, int chunk, const M3L *eqprep,
                   const u32 *W, i8 *A, i8 *B, size_t ld, cudaStream_t s) {
  with_src(t.tab, [&](auto src) {
    // k_digits_all_slabs does less work per index, but below 2^17 indices it has too few warps to
    // fill the GPU; k_digits_poly then runs, as for LV 1 and 2.
    if (LV == 0 && n >= 1 << 17) {
      auto kernel = k_digits_all_slabs<decltype(src)>;
      const Launch l = launch_of(src, (const void *)kernel, (size_t)n / 4);
      kernel<<<l.blocks, l.threads, l.shared, s>>>(t.keys, dT, src, c0, n, chunk, W, A, B, ld);
      CK(cudaGetLastError());
    } else if (LV <= 2)
      with_constant<0, 2>(LV, [&](auto lv) {
        auto kernel = k_digits_poly<decltype(lv)::value, decltype(src)>;
        const Launch l = launch_of(src, (const void *)kernel, (size_t)n * 2);
        kernel<<<l.blocks, l.threads, l.shared, s>>>(t.keys, dT, alpha, src, c0, n, chunk, W, A, B, ld);
        CK(cudaGetLastError());
      });
    else
      with_constant<3, 4>(LV, [&](auto lv) {
        auto kernel = k_digits_plain<decltype(lv)::value, decltype(src)>;
        const Launch l = launch_of(src, (const void *)kernel, (size_t)n * 2);
        kernel<<<l.blocks, l.threads, l.shared, s>>>(t.keys, dT - LV, alpha, src, c0, n, chunk, eqprep, A, B, ld);
        CK(cudaGetLastError());
      });
  });
}
// Combines the 32-bit sums of nc chunks into shape().sums elements per chunk, times eqhi[chunk]:
// out[chunk * sums + p]. C: per chunk, `pieces` column-major matrices of 192 x shape().narrow sums.
inline void join(int LV, const Table &tab, const int *C, int pieces, int nc, const T *eqhi, T *out, cudaStream_t s) {
  if (shape(LV, tab).sums == 64) return grm::join(C, pieces, nc, eqhi, out, s);
  with_src(tab, [&](auto src) {
    with_constant<0, 2>(LV, [&](auto lv) {
      typedef Poly<decltype(lv)::value, decltype(src)> P;
      if constexpr (P::BYTE_FORM) {
        k_join_poly<P><<<(nc * P::SUMS + 3) / 4, 128, 0, s>>>(C, pieces, nc, eqhi, out);
        CK(cudaGetLastError());
      }
    });
  });
}
// Computes G's 64 entries from H: join()'s `sums` outputs, added by the caller over chunks and over
// trees that share alpha. sums = 64: a copy. sums = 8 + 64 D: the byte form for nodes of D leaves.
inline void finish(int sums, const T *H, const T *alpha, T *G, cudaStream_t s) {
  if (sums == 64) {
    CK(cudaMemcpyAsync(G, H, 64 * sizeof(T), cudaMemcpyDeviceToDevice, s));
    return;
  }
  k_finish_poly<<<1, 64, 0, s>>>(H, (sums - 8) / 64, alpha, G);
  CK(cudaGetLastError());
}
// Kernel of fold() for LV 3 and 4, whose entries e are computed from the keys:
//   Lo[c] = sum_(a < 8) k8[a] e(a SC + c),   Ro[c] = sum_(a < 8) k8[a] e((8 + a) SC + c),   c < SC.
// k8: eight elements in GPU memory, the eq weights of three sumcheck rounds. Output i < SC is
// Lo[i]; output SC + i is Ro[i]. A thread computes four neighbouring outputs.
// Public sources for the techniques of this routine.
// The table staged in shared memory and the keys read with __ldg(): cited above struct RangeSrc.
// The compiler hints and the loop forms are public: __launch_bounds__, and #pragma unroll: NVIDIA,
// CUDA C++ Best Practices Guide, "Occupancy" and "Branch Predication",
// https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/; unrolling a loop only in part: Van der
// Wijngaart and Oh, "Boosting Application Performance with GPU Memory Prefetching", "Unrolling",
// cited above struct RangeSrc (here the loop over the eight slabs is not unrolled, and the loops
// over a thread's four outputs are); the grid-stride loop over the outputs: "CUDA Pro Tip: Write
// Flexible Kernels with Grid-Stride Loops", cited at struct Launch.
template <int LV, class S>
__global__ void __launch_bounds__(Form<S>::THREADS)
    k_fold_plain(const uint16_t *__restrict__ keys, int d, const T *alpha, const S src, const T *__restrict__ k8,
                T *__restrict__ Lo, T *__restrict__ Ro) {
  extern __shared__ __align__(16) unsigned char staged[];
  __shared__ M3L weight[8];
  if (threadIdx.x < 8) weight[threadIdx.x] = FK2::prep(k8[threadIdx.x]);
  Form<S> f;
  f.init(src, *alpha, staged);
  __syncthreads();
  const size_t SC = (size_t)1 << (d - 4), step = (size_t)gridDim.x * blockDim.x * 4;
  for (size_t i = (blockIdx.x * (size_t)blockDim.x + threadIdx.x) * 4; i < 2 * SC; i += step) {
    const bool second = i >= SC;
    const uint16_t *at = keys + (second ? i + 7 * SC : i);
    T sum[4], e[4];
#pragma unroll 1
    for (int a = 0; a < 8; a++) {
      four_nodes<LV>(f, at + a * SC, d, e);
#pragma unroll
      for (int k = 0; k < 4; k++) {
        const T x = FK2::mulp(weight[a], e[k]);
        sum[k] = a ? F::add(sum[k], x) : x;
      }
    }
    T *out = second ? Ro + (i - SC) : Lo + i;
#pragma unroll
    for (int k = 0; k < 4; k++) st16(out + k, sum[k]);
  }
}
// Kernel of fold() for LV 1 and 2. Each node is a polynomial in alpha, so
//   sum_a k8[a] node_a = sum_a K_(a,0) + sum_(a, i >= 1) e_i(node_a) K_(a,i)
// with K_(a,i) = (-1)^i alpha^(D - i) k8[a]: only products of base-field numbers with constants.
template <int LV, class S>
__global__ void __launch_bounds__(Form<S>::THREADS)
    k_fold_poly(const uint16_t *__restrict__ keys, int dT, const T *alpha, const S src, const T *__restrict__ k8,
                T *__restrict__ Lo, T *__restrict__ Ro) {
  typedef Poly<LV, S> P;
  extern __shared__ __align__(16) unsigned char staged[];
  __shared__ Powers<P::D> weight[8];
  if (threadIdx.x < 8) weight[threadIdx.x].set(*alpha, k8[threadIdx.x]);
  Form<S> f;
  f.init_values(src, staged);
  __syncthreads();
  u64 start[3]; // sum_a K_(a,0)
#pragma unroll
  for (int j = 0; j < 3; j++) {
    start[j] = 0;
    for (int a = 0; a < 8; a++) start[j] = gf_add(start[j], weight[a].w0[j]);
  }
  const size_t SC = (size_t)1 << (dT - LV - 4), step = (size_t)gridDim.x * blockDim.x * 4;
  for (size_t i = (blockIdx.x * (size_t)blockDim.x + threadIdx.x) * 4; i < 2 * SC; i += step) {
    const bool second = i >= SC;
    const size_t c = second ? i - SC : i;
    u64 x[3][4];
#pragma unroll
    for (int j = 0; j < 3; j++)
#pragma unroll
      for (int k = 0; k < 4; k++) x[j][k] = start[j];
#pragma unroll 1
    for (int a = 0; a < 8; a++) {
      u64 cf[4][P::D];
      four_coefficients<P, true>(f, key_sets<LV>(keys, dT, (second ? 8 : 0) + a, c, 0), cf);
#pragma unroll
      for (int j = 0; j < 3; j++) add_dots<P>(cf, FromPowers<P::D>{&weight[a]}, j, x[j]);
    }
    T *out = second ? Ro + c : Lo + c;
#pragma unroll
    for (int k = 0; k < 4; k++) st16(out + k, T{canonical(x[0][k]), canonical(x[1][k]), canonical(x[2][k])});
  }
}
// Lo[c] = sum_a k8[a] L_a(c), Ro[c] = sum_a k8[a] R_a(c), c < SC = 2^(dT - LV - 4), 1 <= LV <= 4.
// k8: 8 elements in GPU memory. Needs SC >= 4 (not checked here; forest_plan() rejects dT < 17).
inline void fold(int LV, const Tree &t, int dT, const T *alpha, const T *k8, T *Lo, T *Ro, cudaStream_t s) {
  const size_t threads = (size_t)1 << (dT - LV - 5);
  with_src(t.tab, [&](auto src) {
    if (LV <= 2)
      with_constant<1, 2>(LV, [&](auto lv) {
        auto kernel = k_fold_poly<decltype(lv)::value, decltype(src)>;
        const Launch l = launch_of(src, (const void *)kernel, threads);
        kernel<<<l.blocks, l.threads, l.shared, s>>>(t.keys, dT, alpha, src, k8, Lo, Ro);
        CK(cudaGetLastError());
      });
    else
      with_constant<3, 4>(LV, [&](auto lv) {
        auto kernel = k_fold_plain<decltype(lv)::value, decltype(src)>;
        const Launch l = launch_of(src, (const void *)kernel, threads);
        kernel<<<l.blocks, l.threads, l.shared, s>>>(t.keys, dT - LV, alpha, src, k8, Lo, Ro);
        CK(cudaGetLastError());
      });
  });
}

// ---- Pass 2 of the last layer (rounds 4 to 6, bits 5 4 3). With kappa = sum_j k8[j], an entry is
//   U_a(c) = sum_(j < 8) k8[j] leaf(side + 64 c + 8 a + j) = alpha kappa - sum_j k8[j] t_j.
// Here c < N / 128, side = 0 (left) or N / 2 (right), and t_j is key j's value. k8: the eq weights
// of the challenges of rounds 1 to 3; their kappa is 1, but these functions accept any k8. The pass
// runs like a layer of Poly with D = 8 and coefficients t_j. First matrix (see second_constants()):
// eqlo U_a = W_0 + sum_j t_j W_(j+1). Second matrix: the digits of U_a' or, in the byte form, the
// keys' bytes, which give S_0[a] = sum_c w U_a and S_(j+1)[a][a'] = sum_c w U_a t_j(U_a').
// No public source was looked for: follows the machinery of Poly.
template <class S> struct SecondRows {
  static constexpr int D = 8, BITS = Form<S>::BITS, NB0 = Form<S>::NB0;
  // Range, byte-limb: <= 256 byte rows, cheaper to make; cuBLASLt is as fast as on 192 digit rows.
  static constexpr bool BYTE_FORM = 64 * NB0 + 4 <= 256;
  static __host__ __device__ constexpr int bits(int) { return BITS; }
  static __host__ __device__ constexpr int nb(int) { return NB0; }
  static __host__ __device__ constexpr int off(int i) { return (i - 1) * NB0; }
  static __host__ __device__ constexpr int turn(int b) { return Form<S>::turn0(b); }
  static constexpr int NBT = 8 * NB0, NARROW = BYTE_FORM ? 8 * NBT + 4 : 192, SUMS = BYTE_FORM ? 8 + 64 * D : 64;
};
inline Shape shape_second(const Table &tab) {
  Shape sh{0, 0, 0};
  with_src(tab, [&](auto src) {
    typedef SecondRows<decltype(src)> P;
    sh = {192, P::NARROW, P::SUMS};
  });
  return sh;
}
// W_0 = alpha kappa `times`, W_(j+1) = -k8[j] `times`.
__device__ inline void second_constants(Powers<8> &p, const T &alpha, const T *__restrict__ k8, const T &times) {
  T kappa = F::zero();
  for (int j = 0; j < 8; j++) {
    kappa = F::add(kappa, k8[j]);
    p.put(j + 1, F::sub(F::zero(), F::mul(k8[j], times)));
  }
  const T w0 = F::mul(F::mul(alpha, kappa), times);
  for (int j = 0; j < 3; j++) p.w0[j] = coord(w0, j);
}
__global__ void k_second_weights(const T *alpha, const T *__restrict__ k8, int chunk, const T *__restrict__ eqlo,
                                 u32 *__restrict__ W) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= chunk) return;
  Powers<8> p;
  second_constants(p, *alpha, k8, eqlo[c]);
  to_table(p, chunk, c, W);
}
// Pass 2's table of weights, 16-byte aligned: 54 * chunk 32-bit words, in weights_words()'s layout
// with D = 8. Fill it after the challenges of rounds 1 to 3. It serves all trees that share alpha.
inline size_t second_words(int chunk) {
  return (size_t)54 * chunk;
}
inline void second_weights(const T *alpha, const T *k8, int chunk, const T *eqlo, u32 *W, cudaStream_t s) {
  k_second_weights<<<(chunk + 127) / 128, 128, 0, s>>>(alpha, k8, chunk, eqlo, W);
  CK(cudaGetLastError());
}
// Like rows_of_slab(): slab 0 to 7 is eqlo U_slab of the left side, 8 to 15 is U_(slab - 8) of the
// right side. q: eight_keys<IN_FOURS, 64>(..) of the slab, for the warp's 128 indices from c on.
// `constants`: second_constants(.., 1), read only in the digit form.
template <class S, class Put>
__device__ __forceinline__ void rows_of_second(const Form<S> &f, const uint4 (&q)[4], int slab, size_t c, int lane, int chunk,
                                               const u32 *__restrict__ W, const Powers<8> *constants, const Put &put) {
  typedef SecondRows<S> P;
  if (slab >= 8 && P::BYTE_FORM) return bytes_of_eight(f, q, put);
  u64 cf[4][8];
#pragma unroll
  for (int i = 0; i < 4; i++)
#pragma unroll
    for (int j = 0; j < 8; j++) cf[i][j] = f.value(grm::key_of(q[i], j));
  if (slab < 8) {
    const int cc = (int)(c % chunk);
    rows_of_dot<P>(cf, FromTable<IN_FOURS>{W + cc, chunk, lane, reinterpret_cast<const u64 *>(W + (size_t)48 * chunk) + cc}, put);
  } else {
    rows_of_dot<P>(cf, FromPowers<8>{constants}, put);
  }
}
template <class S>
__global__ void __launch_bounds__(Form<S>::THREADS)
    k_digits_second(const uint16_t *__restrict__ keys, int dT, const T *alpha, const S src, const T *__restrict__ k8, size_t c0,
                    int n, int chunk, const u32 *__restrict__ W, i8 *__restrict__ A, i8 *__restrict__ B, size_t ld) {
  typedef SecondRows<S> P;
  extern __shared__ __align__(16) unsigned char staged[];
  __shared__ Powers<8> constants;
  Form<S> f;
  f.init_values(src, staged);
  if (!P::BYTE_FORM) {
    if (threadIdx.x == 0) second_constants(constants, *alpha, k8, F::one());
    __syncthreads();
  }
  constexpr int GROUPS = Form<S>::THREADS / 256;
  const int a = (threadIdx.x >> 5) & 7, lane = threadIdx.x & 31;
  for (int c = (blockIdx.x * GROUPS + (threadIdx.x >> 8)) * 128; c < n; c += gridDim.x * GROUPS * 128) {
    uint4 q[4];
    eight_keys<IN_FOURS, 64>(keys, dT, false, a, c0 + c, lane, q);
    rows_of_second(f, q, a, c0 + c, lane, chunk, W, &constants, [&](int row, u32 w) {
      *reinterpret_cast<u32 *>(A + (size_t)(a * 24 + row) * ld + c + 4 * lane) = w;
    });
    eight_keys<IN_FOURS, 64>(keys, dT, true, a, c0 + c, lane, q);
    rows_of_second(f, q, 8 + a, c0 + c, lane, chunk, W, &constants, [&](int row, u32 w) {
      *reinterpret_cast<u32 *>(B + (size_t)(a * (P::BYTE_FORM ? P::NBT : 24) + row) * ld + c + 4 * lane) = w;
    });
    if (P::BYTE_FORM && a < 4)
      *reinterpret_cast<u32 *>(B + (size_t)(8 * P::NBT + a) * ld + c + 4 * lane) = a ? 0u : 0x01010101u;
  }
}
// Like digits(), for n of pass 2's N / 128 indices, with the same conditions on `c0`, `n`, `chunk`,
// `A`, `B` and `ld`. B has shape_second().narrow rows. W: the table of second_weights(). In the
// kernel, k_digits_second, the warps share the work as in k_digits_poly.
inline void second_digits(const Tree &t, int dT, const T *alpha, const T *k8, size_t c0, int n, int chunk, const u32 *W, i8 *A,
                          i8 *B, size_t ld, cudaStream_t s) {
  with_src(t.tab, [&](auto src) {
    auto kernel = k_digits_second<decltype(src)>;
    const Launch l = launch_of(src, (const void *)kernel, (size_t)n * 2);
    kernel<<<l.blocks, l.threads, l.shared, s>>>(t.keys, dT, alpha, src, k8, c0, n, chunk, W, A, B, ld);
    CK(cudaGetLastError());
  });
}
// Like join().
inline void second_join(const Table &tab, const int *C, int pieces, int nc, const T *eqhi, T *out, cudaStream_t s) {
  if (shape_second(tab).sums == 64) return grm::join(C, pieces, nc, eqhi, out, s);
  with_src(tab, [&](auto src) {
    typedef SecondRows<decltype(src)> P;
    if constexpr (P::BYTE_FORM) {
      k_join_poly<P><<<(nc * P::SUMS + 3) / 4, 128, 0, s>>>(C, pieces, nc, eqhi, out);
      CK(cudaGetLastError());
    }
  });
}
__global__ void k_second_finish(const T *H, const T *alpha, const T *k8, T *G) {
  const int t = threadIdx.x;
  T kappa = F::zero(), r = F::zero();
  for (int j = 0; j < 8; j++) kappa = F::add(kappa, k8[j]), r = F::add(r, F::mul(k8[j], H[8 + 64 * j + t]));
  G[t] = F::sub(F::mul(F::mul(*alpha, kappa), H[t >> 3]), r);
}
// Like finish(). Byte form (520 sums): G[a][a'] = alpha kappa S_0[a] - sum_j k8[j] S_(j+1)[a][a'].
inline void second_finish(int sums, const T *H, const T *alpha, const T *k8, T *G, cudaStream_t s) {
  if (sums == 64) {
    CK(cudaMemcpyAsync(G, H, 64 * sizeof(T), cudaMemcpyDeviceToDevice, s));
    return;
  }
  k_second_finish<<<1, 64, 0, s>>>(H, alpha, k8, G);
  CK(cudaGetLastError());
}

// ---- Pass 3's fold (before rounds 7 to 9, bits 8 7 6), as in gram_products.cuh.
// tau(k) comes from grm::Key (RANGE), grm::Tau (BYTE_LIMBS), `Packed` (PACKED) or `Rows` (GENERAL).
// Packed: value(k) = tau(k) = k + 2^28 ctab[k] = lo(k) + 2^24 hi(k), below 2^44.
struct Packed {
  const uint16_t *ctab;
  static constexpr bool HI = true;
  __device__ __forceinline__ u64 value(u32 k) const { return (u64)k + ((u64)__ldg(ctab + k) << 28); }
  __device__ __forceinline__ u32 lo(u32 k) const { return (u32)value(k) & 0xFFFFFF; }
  __device__ __forceinline__ u32 hi(u32 k) const { return (u32)(value(k) >> 24); }
};
// GENERAL: row values below 2^(8 BYTES), BYTES = 6 or 8 (and below 2^61: table_of()), read from GPU
// memory and split into PARTS parts of 24 bits. A key not below `rows`: as at Form<RowSrc<BYTES>>.
template <int BYTES> struct Rows {
  const u64 *tau;
  u32 rows;
  static constexpr int PARTS = (BYTES + 2) / 3;
  __device__ __forceinline__ u64 value(u32 k) const { return __ldg(tau + index_below(k, rows, INDEX_FUNCTION_KEY)); }
};
// Like grm::fold_keys() and grm::fold_end(), for a GENERAL table's PARTS parts: fold_rows() adds
// sum_j K[j] part_i(key j of q) to acc[i]; fold_rows_end(): alpha - sum_i 2^(24 i) acc[i].
template <class V>
__device__ __forceinline__ void fold_rows(FE::SAcc (&acc)[V::PARTS], const uint4 &q, const FE::SP *K, const V &val) {
#pragma unroll
  for (int j = 0; j < 8; j++) {
    const u64 v = val.value(grm::key_of(q, j));
#pragma unroll
    for (int i = 0; i < V::PARTS; i++) FE::smac(acc[i], K[j], (u32)(v >> (24 * i)) & 0xFFFFFF);
  }
}
template <class V> __device__ __forceinline__ T fold_rows_end(const T &alpha, const FE::SAcc (&acc)[V::PARTS]) {
  T s = FE::sred(acc[0]);
#pragma unroll
  for (int i = 1; i < V::PARTS; i++) {
    const T h = FE::sred(acc[i]);
    s = F::add(s, T{grm::gram_times_2_to(h.k0, 24 * i), grm::gram_times_2_to(h.k1, 24 * i), grm::gram_times_2_to(h.k2, 24 * i)});
  }
  return F::sub(alpha, s);
}
// Calls fn with the table's struct for grm::k_fold64. A GENERAL table has none: returns false.
template <class Fn> inline bool with_two_parts(const Table &tab, const Fn &fn) {
  if (tab.kind == RANGE) return fn(grm::Key{}), true;
  if (tab.kind == BYTE_LIMBS) return fn(grm::Tau{tab.tab8}), true;
  if (tab.kind == PACKED) return fn(Packed{tab.ctab}), true;
  return false;
}
template <class Fn> inline void with_rows(const Table &tab, const Fn &fn) {
  if (tab.bits <= 48) return fn(Rows<6>{tab.tau, 1u << tab.k});
  fn(Rows<8>{tab.tau, 1u << tab.k});
}
// Like grm::k_fold64, for a GENERAL table. Needs blocks of 256 threads (not checked).
// UL, UR: outputs for the keys' first and second half, each eight slabs of CO entries; rows = 8 CO.
template <class V>
__global__ void __launch_bounds__(256)
    k_fold64_rows(const uint4 *__restrict__ key, size_t rows, size_t CO, const T *__restrict__ K,
                  const T *__restrict__ alpha, T *__restrict__ UL, T *__restrict__ UR, const V val) {
  const T al = *alpha;
  const int jj = threadIdx.x & 7;
  FE::SP Ks[8]; // the weights K[8 jj .. 8 jj + 7]
#pragma unroll
  for (int j = 0; j < 8; j++) Ks[j] = FE::sprep(K[jj * 8 + j]);
  for (size_t g = blockIdx.x * 32 + (threadIdx.x >> 3); g < 2 * rows; g += (size_t)gridDim.x * 32) {
    FE::SAcc acc[V::PARTS];
#pragma unroll
    for (int i = 0; i < V::PARTS; i++) acc[i] = FE::szero();
    fold_rows(acc, key[g * 8 + jj], Ks, val);
#pragma unroll
    for (int o = 4; o; o >>= 1)
#pragma unroll
      for (int p = 0; p < V::PARTS; p++)
#pragma unroll
        for (int i = 0; i < 3; i++) {
          acc[p].l[i] += __shfl_down_sync(~0u, acc[p].l[i], o, 8);
          acc[p].h[i] += __shfl_down_sync(~0u, acc[p].h[i], o, 8);
        }
    if (jj == 0) {
      const size_t r = g >= rows ? g - rows : g;
      (g >= rows ? UR : UL)[(r & 7) * CO + (r >> 3)] = fold_rows_end<V>(al, acc);
    }
  }
}
// Writes U_a(c) = sum_(j < 64) K64[j] leaf(side + 64 (8 c + a) + j), a < 8, c < CO = N / 1024, to
// UL[a CO + c] (side = 0) and UR[a CO + c] (side = N / 2): eight slabs U_a of CO entries in each.
// K64 (from k_k64): the 64 eq weights of the challenges of rounds 1 to 6; they sum to 1.
inline void third_fold(const Tree &t, int dT, const T *alpha, const T *K64, T *UL, T *UR, cudaStream_t s) {
  const size_t N = (size_t)1 << dT;
  const int grid = (int)std::min<size_t>((N / 64 + 31) / 32, (size_t)g_sms * 8);
  const bool done = with_two_parts(t.tab, [&](auto val) {
    grm::k_fold64<<<grid, 256, 0, s>>>((const uint4 *)t.keys, N / 128, N / 1024, K64, alpha, UL, UR, val);
  });
  if (!done)
    with_rows(t.tab, [&](auto val) {
      k_fold64_rows<<<grid, 256, 0, s>>>((const uint4 *)t.keys, N / 128, N / 1024, K64, alpha, UL, UR, val);
    });
  CK(cudaGetLastError());
}
} // namespace fk
