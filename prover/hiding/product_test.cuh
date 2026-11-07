// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// Product test (`Pt`, `pt` in names), the lookup argument's grand-product check: sumchecks (`PSC`
// in names) of sum_c eq(rho, c) L(c) R(c), the layers of a lookup's product trees (LKP:
// blocks/iop/lkp.h). prover/hiding/forest.cuh proves them with the kernels, Eng (a chain of
// layers), Sched (tail launches) and pt_graphed() here. PtPL, PtWPL prove one outside a lookup.
#pragma once
#include "hiding/transcript.cuh" // T: extension-field element; F: its operations; Tr; sc_door
// hiding/pt_gram.cuh has the 8-to-1 fold of a stored layer.
#include "hiding/pt_gram.cuh"
#include "common/index_guard.cuh" // for hiding/forest_keys.cuh, which is included after this file
#include "hiding/pt_layer_a.cuh"
// State of a chain of layers (Eng). endv[id]: layer id's end values, L and R at its challenges.
// claim[id] = endv0 + tau (endv1 - endv0), tau sampled after them, goes to the next layer.
struct St {
  Tr tr;
  T alpha;
  u32 err;
  T endv[128][2], claim[128];
  T start[128]; // start[id]: the true value of the sum that layer id claims
  T add0;       // the running layer's summand: add0 + L R; 1 in the table side's leaf layer, else 0
};
__global__ void k_trout(const St *st, Tr *t) {
  *t = st->tr;
}
__global__ void k_err0(St *st) {
  st->err = 0;
}
// The loop form is public: the grid-stride loop: "CUDA Pro Tip: Write Flexible Kernels with Grid-Stride Loops", a post
// on NVIDIA's developer blog, https://developer.nvidia.com/blog/cuda-pro-tip-write-flexible-kernels-grid-stride-loops/.
__global__ void k_bits(const u32 *cnt, int lR, int lK, int lI, u8 *b) {
  const size_t n = (size_t)1 << (lR + lK + lI);
  for (size_t X = blockIdx.x * (size_t)blockDim.x + threadIdx.x; X < n; X += (size_t)gridDim.x * blockDim.x) {
    const u32 t = (u32)(X >> (lK + lI)), ib = (u32)(X >> lK) & ((1u << lI) - 1), j = (u32)X & ((1u << lK) - 1);
    b[X] = j < 32 ? (cnt[((size_t)ib << lR) + t] >> j) & 1 : 0;
  }
}
// One level of the tree hash (verifier/HASHES.md, "TH(x), the tree hash"), one thread per chunk of
// THPW words. num: TH_LIST if `in` is the hashed list itself, TH_UPPER if it is a level of digests.
__global__ void k_th(const u32 *in, size_t n, u32 *out, u32 num) {
  size_t c = blockIdx.x * (size_t)blockDim.x + threadIdx.x, nc = (n + THPW - 1) / THPW;
  if (c >= nc) return;
  u32 h[DW];
  earlier_hash_h_digest(h, num, in + c * THPW, n - c * THPW < THPW ? n - c * THPW : THPW);
  for (int i = 0; i < DW; i++) out[c * DW + i] = h[i];
}
#include "hiding/count_keys16.cuh"
// Histograms of 16-bit keys. An identity is a part of a lookup's keys that is counted separately:
// identity b has nvb words of 8 keys, shared by bpi blocks, each with REP bins per value t. Adds
// the number of its keys equal to t to outAll[b * 2^k + t]; only a key's low k < 16 bits count.
// Not checked: the launch has bpi blocks per identity and 4 REP 2^k bytes of dynamic shared memory.
// Public sources: see hiding/count_keys16.cuh.
// The idea is public: per-block shared-memory histograms, merged atomically:
// NVIDIA, "CUDA Programming Guide", page "Writing SIMT Kernels", "Distributed Shared Memory",
// https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/writing-cuda-kernels.html.
// The idea is public: the counting by atomics: N. Sakharnykh, "GPU Pro Tip: Fast Histograms Using Shared Atomics
// on Maxwell", NVIDIA Technical Blog, 2015,
// https://developer.nvidia.com/blog/gpu-pro-tip-fast-histograms-using-shared-atomics-maxwell/.
// The idea is public: replicated bins: S. Koppaka, D. Mudigere, S. Narasimhan, B. Narayanan, "Fast
// Histograms using Adaptive CUDA Streams", arXiv:1011.0235, section III.A,
// https://arxiv.org/pdf/1011.0235. There the number of sub-bins differs from bin to bin.
// The idea is public: vectorized loads of packed narrow elements: "CUDA Pro Tip: Increase
// Performance with Vectorized Memory Access", a post on NVIDIA's developer blog,
// https://developer.nvidia.com/blog/cuda-pro-tip-increase-performance-with-vectorized-memory-access/.
// The idea is public: eight 16-bit elements in one aligned 128-bit access: NVIDIA, CUTLASS
// documentation, "Fundamental Types", "AlignedArray",
// https://docs.nvidia.com/cutlass/media/docs/cpp/fundamental_types.html.
// The compiler hint is public: #pragma unroll: NVIDIA, CUDA Programming Guide, "C/C++ Language
// Extensions", "#pragma unroll",
// https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-extensions.html.
// The compiler hint is public: x's four words can live in registers if unrolling fixes each a >> 1:
// M. Milakov, "Fast Dynamic Indexing of Private Arrays in CUDA", NVIDIA Technical Blog, 2015,
// https://developer.nvidia.com/blog/fast-dynamic-indexing-private-arrays-cuda/, "Static indexing".
// The loop form is public: the block-stride loop: shown in a listing in the CUDA Programming Guide,
// page "Writing SIMT Kernels", "Distributed Shared Memory", cited above.
template <int REP> __global__ void h_idk(const uint4 *keys, size_t nvb, int bpi, int k, u32 *outAll) {
  extern __shared__ u32 bins[];
  const int nb = 1 << k;
  for (int i = threadIdx.x; i < nb * REP; i += blockDim.x) bins[i] = 0;
  __syncthreads();
  const int b = blockIdx.x / bpi, w = blockIdx.x % bpi, rp = threadIdx.x % REP;
  const uint4 *kb = keys + (size_t)b * nvb;
  u32 *out = outAll + ((size_t)b << k);
  for (size_t i = w * (size_t)blockDim.x + threadIdx.x; i < nvb; i += (size_t)bpi * blockDim.x) {
    uint4 v = kb[i];
    u32 x[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
    for (int a = 0; a < 8; a++) atomicAdd(&bins[((x[a >> 1] >> (16 * (a & 1))) & (nb - 1)) * REP + rp], 1u);
  }
  __syncthreads();
  for (int c = threadIdx.x; c < nb; c += blockDim.x) {
    u32 s = 0;
    for (int r = 0; r < REP; r++) s += bins[c * REP + (r + threadIdx.x) % REP];
    atomicAdd(&out[c], s);
  }
}
// The 8-to-1 fold after the three rounds that bind the slab number a: k8[a] = eq(challenges, a).
// Bit a of BmK[c] is a multiplicity bit of table row t = a 2^(lR-3) + c / nlo. out[c], c < n = nlo
// 2^(lR-3), is the sum of k8[a] over the bits a set in BmK[c], read from a table of the 256
// possible sums. out[n + c] is out[c] - 1 (KIND 1, bit layer) or the fold of alpha - 1 - t (KIND 0,
// a range table's leaf layer). Block size: 256.
// Public source: one-bit weights for the same inputs, packed as a table index: Y. Jeon, B. Park,
// S. J. Kwon, B. Kim, J. Yun, D. Lee, "BiQGEMM: Matrix Multiplication with Lookup Table For
// Binary-Coding-based Quantized DNNs", arXiv:2005.09904 (2020), section III-B.
// Public source: the same packing as above, for each bit plane: J. Wei, S. Cao, T. Cao, L. Ma,
// L. Wang, Y. Zhang, M. Yang, "T-MAC: CPU Renaissance via Table Lookup for Low-Bit LLM Deployment
// on Edge", arXiv:2407.00088 (2024), section 3.1.
// There the table has partial sums of a matrix product's inputs; here the group is the eight
// entries that one 8-to-1 fold combines, and the sums are of field elements.
// The loop form is public: the grid-stride loop: the post on grid-stride loops, cited at k_bits.
// The compiler hint is public: __launch_bounds__: NVIDIA, "CUDA C++ Best Practices Guide",
// "Occupancy", https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/.
// The idea is public: a kernel's template parameter, set by the host: M. Harris, "Optimizing
// Parallel Reduction in CUDA", NVIDIA Developer Technology,
// https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf, slides "Complete
// Unrolling" to "Invoking Template Kernels". There the parameter is the block size, here KIND.
// The idea is public: const __restrict__ pointers to read-only data: NVIDIA, Kepler Tuning Guide,
// "Memory Throughput", part "Read-Only Data Cache",
// https://docs.nvidia.com/cuda/kepler-tuning-guide/. It names both marks as a necessary condition
// for the compiler to use that cache by itself.
template <int KIND>
__global__ void __launch_bounds__(256)
    k_foldmaskN(const u8 *__restrict__ BmK, u32 nlo, int lR, const T *k8, const St *st, T *__restrict__ out) {
  __shared__ T tb[256];
  __shared__ T base;
  const int t = threadIdx.x;
  {
    T s = F::zero();
    for (int a = 0; a < 8; a++)
      if ((t >> a) & 1) s = F::add(s, k8[a]);
    tb[t] = s;
  }
  if (t == 0 && KIND == 0) {
    T s = F::zero();
    for (int a = 1; a < 8; a++) s = F::add(s, F::mul(k8[a], F::small((u64)a << (lR - 3))));
    base = F::sub(F::sub(st->alpha, F::one()), s);
  }
  __syncthreads();
  const size_t n = (size_t)nlo << (lR - 3);
  const T one = F::one();
  for (size_t c = blockIdx.x * (size_t)blockDim.x + t; c < n; c += (size_t)gridDim.x * blockDim.x) {
    const T l = tb[BmK[c]];
    st16(out + c, l);
    st16(out + n + c, KIND ? F::sub(l, one) : F::sub(base, F::small(c / nlo)));
  }
}

// lo + r (hi - lo)
__device__ __forceinline__
T fold_at(T r, T lo, T hi) {
  return F::add(lo, F::mul(r, F::sub(hi, lo)));
}
// Runs the three sumcheck rounds that bind a layer's slab number, from the 8 x 8 Gram matrix of
// its slabs, A[a][a'] = sum_c eq(c) L_a(c) R_a'(c) (gram_products.cuh). Writes the challenges to
// ch->r[j .. j + 2], the transcript to st->tr and eq of the challenges to k8[0 .. 7]. iz2, iz3:
// where ch->rho has the coordinates of the slab number's second and third bit. out is not read.
// Required: the layer's sumcheck has begun (k_layer_begin(), k_leaves_begin()); with none at a
// round, psc::round() traps. Not checked: a block barrier since A (in shared memory in k_g2msgT)
// was written. Only the block's first warp works; it sends through sc_door (sumcheck_door.cuh).
// The lanes share M, Q and R in shared memory, with __syncwarp() between one lane's write and
// another's read. To overwrite Q while lanes still read it: read into y, barrier, write, barrier.
// Public sources for the techniques of this function.
// The idea is public: __syncwarp() between shared-memory accesses: NVIDIA Technical Blog,
// "Using CUDA Warp-Level Primitives", "Warp Synchronization",
// https://developer.nvidia.com/blog/using-cuda-warp-level-primitives/.
// The idea is public: read, barrier, write, barrier: the same part of the post, and its Listing 8.
__device__ void g2msg_dev(const T *A, St *st, Chal<F> *ch, int j, int iz2, int iz3, T *out, T *k8) {
  constexpr int BANKS = 2 + PSC_CHECK_CLAIM; // the coefficients of X and X^2 (psc::Form::EQC); in a checking build also the constant
  __shared__ T M[64], Q[4 * BANKS], R[3];
  TR_IN_SHARED(trs);
  const int t = threadIdx.x;
  (void)out; // the running sumcheck's state (Tr::p) has the address of its next scalar
  if (t >= 32) return;
  // z2, z3: the layer's point at the slab number's second and third bit
  const T z2 = ch->rho[iz2], z3 = ch->rho[iz3];
  sc_door::tr_copy_by_lanes(&trs, &st->tr, t, 32);
  M[t] = A[t];
  M[t + 32] = A[t + 32];
  __syncwarp();
  const int i = t >> 3, c = t & 7;
  for (int rd = 0; rd < 3; rd++) {
    const int h = 4 >> rd; // the matrix of this round is 2h by 2h, stored in M with a row stride of 8
    if (t < h) { // with lo = row or column t, hi = h + t, D = hi - lo: <lo, D> + <D, lo> and <D, D>
      const T d = M[t * 8 + t], a = M[(h + t) * 8 + t], b = M[t * 8 + h + t];
      Q[t] = F::sub(F::add(a, b), F::add(d, d));
      Q[4 + t] = F::add(F::sub(F::sub(M[(h + t) * 8 + h + t], a), b), d);
#if PSC_CHECK_CLAIM
      Q[8 + t] = d;
#endif
    }
    __syncwarp();
    // Evaluates the vectors of diagonal terms at the point, the third bit's coordinate (z3) first.
    for (int w = h >> 1, s = 0; w; w >>= 1, s++) {
      const bool on = t < BANKS * w;
      const int q = 4 * (t / w), x = t % w;
      T y = F::zero();
      if (on) y = fold_at(s ? z2 : z3, Q[q + 2 * x], Q[q + 2 * x + 1]);
      __syncwarp();
      if (on) Q[q + x] = y;
      __syncwarp();
    }
    if (t < SEND2_N) {
#if PSC_CHECK_CLAIM
      const T r = sc_door::round_by_lanes<sc_door::Form::EQC>(sc_door::Site::LAYER_GRAM, j + rd, &trs, Q[0], Q[4], F::add(Q[8], st->add0));
#else
      const T r = sc_door::round_by_lanes<sc_door::Form::EQC>(sc_door::Site::LAYER_GRAM, j + rd, &trs, Q[0], Q[4]);
#endif
      if (t == 0) {
        R[rd] = r;
        ch->r[j + rd] = r;
      }
    }
    __syncwarp();
    if (rd == 2) break;
    const T r = R[rd];
    if (i < h && c < 2 * h) M[i * 8 + c] = fold_at(r, M[i * 8 + c], M[(i + h) * 8 + c]);
    __syncwarp();
    if (i < h && c < h) M[i * 8 + c] = fold_at(r, M[i * 8 + c], M[i * 8 + c + h]);
    __syncwarp();
  }
  __syncwarp();
  sc_door::tr_copy_by_lanes(&st->tr, &trs, t, 32);
  if (t < 8) {
    const T one = F::one(), zero = F::zero(), r1 = R[0], r2 = R[1], r3 = R[2];
    const T a = fold_at((t & 4) ? r1 : F::sub(one, r1), zero, (t & 2) ? r2 : F::sub(one, r2));
    k8[t] = fold_at(a, zero, (t & 1) ? r3 : F::sub(one, r3));
  }
}
// Runs g2msg_dev() on the 8 x 8 Gram matrix G. One block of 64 threads.
__global__ void k_g2msgT(const T *G, St *st, Chal<F> *ch, int j, int iz2, int iz3, T *out, T *k8) {
  __shared__ T A[64];
  A[threadIdx.x] = G[threadIdx.x];
  __syncthreads();
  g2msg_dev(A, st, ch, j, iz2, iz3, out, k8);
}
// A batch of up to 16 eq tables that one launch of k_eqjobs fills; rho is that kernel's ch->rho.
//   ka = 0: o[x] = eq(rho[a .. a + nb), x) for x < 2^nb
//   ka > 0: o[x ka + j] = eq(rho[a .. a + nb), x) eq(rho[a + nb .. a + nb + lk), j), planes j < ka
// If op[i] is set, table i goes to op[i], not o[i], in the form M3L (pt_field.h), which a Gram pass
// (see pt_tail_cap) needs for its low-bit eq table. add() does not check n < 16.
struct EqJobs {
  int n = 0, a[16] = {}, nb[16] = {}, ka[16] = {}, lk[16] = {}; // 0 where unused: all go to the GPU
  T *o[16] = {};
  M3L *op[16] = {};
  void add(int a_, int nb_, T *o_, int ka_ = 0, int lk_ = 0, M3L *op_ = nullptr) {
    a[n] = a_;
    nb[n] = nb_;
    o[n] = o_;
    ka[n] = ka_;
    lk[n] = lk_;
    op[n] = op_;
    n++;
  }
};
// Builds the tables from which eq_at() reads eq(rho[0 .. n), x) as a product of one factor for each
// group of five coordinates (the last group may have fewer): about a fifth of the multiplications
// that one factor for each coordinate would cost. Coordinate 0 goes with the most significant of
// the n bits of x. tb[g][y] is the eq value of coordinates 5 g .. 5 g + 4 at the bit pattern y.
// Each warp builds one table at a time, and lane y computes entry y. Returns the number of groups.
// Required, and not checked:
// - tb has room for (n + 4) / 5 groups;
// - every thread of the block calls this function;
// - the block size is a multiple of 32;
// - the block passes a barrier before any thread calls eq_at().
__device__ __forceinline__ int eq_groups(const T *rho, int n, T (*tb)[32]) {
  const int ng = (n + 4) / 5, y = threadIdx.x & 31;
  for (int g = threadIdx.x >> 5; g < ng; g += blockDim.x >> 5) {
    const int len = min(5, n - 5 * g);
    T v = F::one();
    for (int i = 0; i < len; i++) {
      const T rh = rho[5 * g + i], f = ((y >> (len - 1 - i)) & 1) ? rh : F::sub(F::one(), rh);
      v = i ? F::mul(v, f) : f;
    }
    tb[g][y] = v;
  }
  return ng;
}
// Returns eq(rho[0 .. n), x) from the tables tb and the number of groups ng of eq_groups().
// Public sources for the techniques of this function.
// A value of eq is read as a product of entries of split tables. This is a public idea that is
// specific to the sumcheck's weights eq: Q. Dao, J. Thaler, "More Optimizations to Sum-Check
// Proving", IACR ePrint 2024/1210, section 1.1, equation (4) and footnote 2,
// https://eprint.iacr.org/2024/1210.pdf.
__device__ __forceinline__ T eq_at(const T (*tb)[32], int n, int ng, size_t x) {
  T v = F::one();
  for (int g = 0; g < ng; g++) {
    const int len = min(5, n - 5 * g);
    const T f = tb[g][(x >> (n - 5 * g - len)) & ((1 << len) - 1)];
    v = g ? F::mul(v, f) : f;
  }
  return v;
}
// See EqJobs; blockIdx.y selects the table. Not checked: nb <= 35. M3L has each element split into
// limbs, as the left operand of many products. This is specific to the field's arithmetic.
// The loop form is public: the grid-stride loop: the post on grid-stride loops, cited at k_bits.
__global__ void k_eqjobs(const Chal<F> *ch, const EqJobs J) {
  __shared__ T tb[7][32], tp[2][32];
  const int a = J.a[blockIdx.y], n = J.nb[blockIdx.y], ka = J.ka[blockIdx.y], lk = J.lk[blockIdx.y];
  T *o = J.o[blockIdx.y];
  M3L *op = J.op[blockIdx.y];
  const size_t tot = ka ? (size_t)ka << n : (size_t)1 << n;
  if (blockIdx.x * (size_t)blockDim.x >= tot) return;
  const int ng = eq_groups(ch->rho + a, n, tb);
  const int ngp = ka ? eq_groups(ch->rho + a + n, lk, tp) : 0; /* lk <= 10 */
  __syncthreads();
  for (size_t e = blockIdx.x * (size_t)blockDim.x + threadIdx.x; e < tot; e += (size_t)gridDim.x * blockDim.x) {
    const size_t x = ka ? e / ka : e;
    T v = eq_at(tb, n, ng, x);
    if (ngp) v = F::mul(v, eq_at(tp, lk, ngp, e - x * ka));
    if (op)
      op[e] = m3l(v);
    else
      o[e] = v;
  }
}
// Expands tables that store only planes j < ka to all 2^lk planes. src: L, then R, nh ka entries
// each; dst: L, then R, nh 2^lk each. The planes not stored get: mode 0 (a tree layer) L = R = 1;
// mode 1 (the leaf layer) L = 0, R = the row's plane 0; mode 2 (the bit layer) L = 0, R = -1;
// mode 3 (a single table) 1; mode 4 (a lookup forest's empty trees, forest_table.cuh) L = R = 0.
// The loop form is public: the grid-stride loop: the post on grid-stride loops, cited at k_bits.
__global__ void k_expand(const T *__restrict__ src, T *__restrict__ dst, size_t nh, u32 ka, int lk, int mode) {
  const size_t nf = nh << lk, tot = mode == 3 ? nf : 2 * nf;
  const T one = F::one();
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < tot; i += (size_t)gridDim.x * blockDim.x) {
    const int sd = i >= nf;
    const size_t y = sd ? i - nf : i, h = y >> lk;
    const u32 j = (u32)y & ((1u << lk) - 1);
    const T *p = src + (sd ? nh * ka : 0) + h * ka;
    T v;
    if (j < ka)
      v = ld16(p + j);
    else if (mode == 0 || mode == 3)
      v = one;
    else if (!sd || mode == 4)
      v = F::zero();
    else
      v = mode == 1 ? ld16(p) : F::sub(F::zero(), one);
    st16(dst + i, v);
  }
}
__global__ void k_finA80(const T *G80, const St *st, T *G) {
  const int t = threadIdx.x;
  const T al = st->alpha;
  G[t] = F::add(F::mul(al, F::sub(al, F::add(G80[64 + (t >> 3)], G80[72 + (t & 7)]))), G80[t]);
}
__global__ void k_k64(const T *ka, const T *kb, T *K64) {
  const int j = threadIdx.x;
  K64[j] = F::mul(ka[j & 7], kb[j >> 3]);
}
__global__ void k_cprho(const Chal<F> *a, Chal<F> *b, int n) {
  if (threadIdx.x < n) b->rho[threadIdx.x] = a->rho[threadIdx.x];
}
// Range order (see k_layer_begin()): stores the challenges r[0 .. 8] by index bit, bit 8 first.
__global__ void k_lowrho(const Chal<F> *a, Chal<F> *nx, int mue) {
  const int i = threadIdx.x;
  if (i < 9) nx->rho[1 + mue + i] = a->r[6 - 3 * (i / 3) + i % 3];
}
// Shared memory of tail_layer() (tails.cuh), which runs a layer's last rounds, its tail. Only
// blk_red() uses sh. sums: a wide round's sums over the launch's blocks (wide: tables above 2^NL).
struct ShT {
  T e[E3S], sh[32 * (2 + PSC_CHECK_CLAIM)], tab[2 << NL], rsh[MAXV], rho[MAXV], sums[2 + PSC_CHECK_CLAIM];
  Tr tr;
};
// PRODUCT: sum eq L R; ends with two end values and the challenge that joins them. LEAVES: sum
// eq (1 + L R), of degree 1 in the last five variables (the plane), on which R does not depend; L
// is kept back, so no end value is sent. BITS: sum eq L R, R = L - 1; only L's end value is sent.
enum TailKind : short { TAIL_PRODUCT = 0, TAIL_LEAVES = 1, TAIL_BITS = 2 };
struct LL {
  const T *src[24];
  T *out[24];
  short mu[24], j0[24], id[24], idn[24];
  short begins[24]; // the layer's sumcheck begins with this entry's first round: its first slot is out
  short kind[24];   // TailKind
  int n;
};
// The argument of k_tails (tails.cuh), for one chain (struct Eng): its tail list (in LL, the
// arguments of tail_layer() for n layers), transcript state, challenge arrays and scratch memory.
struct ChainD {
  LL ll;
  St *st;
  Chal<F> *chd;
  T *B0, *B1, *part;
};

__global__ void k_tr2st(const Tr *t, St *s) {
  s->tr = *t;
}
__global__ void k_st2st(const St *a, St *b) {
  b->tr = a->tr;
}
// The loop form is public: the grid-stride loop: the post on grid-stride loops, cited at k_bits.
__global__ void k_eqtabT(const Chal<F> *ch, int a, int n, T *o) {
  __shared__ T tb[7][32];
  if (blockIdx.x * (size_t)blockDim.x >= ((size_t)1 << n)) return;
  const int ng = eq_groups(ch->rho + a, n, tb);
  __syncthreads();
  for (size_t x = blockIdx.x * (size_t)blockDim.x + threadIdx.x; x < ((size_t)1 << n);
       x += (size_t)gridDim.x * blockDim.x)
    o[x] = eq_at(tb, n, ng, x);
}
// A product layer's output. r[0] is the top index bit's challenge, tau the one after vL and vR.
struct PtPLOut {
  T r[32], vL, vR, tau;
};
// claim: the true value of the sum that the layer claims; it becomes st->start[id].
__global__ void k_plin(St *st, const Tr *t, T cst, Chal<F> *ch, const T *rho, int mu, const T *claim, int id) {
  if (threadIdx.x == 0) {
    st->tr = *t;
    st->alpha = cst;
    st->start[id] = *claim;
  }
  if (threadIdx.x < mu) ch->rho[threadIdx.x] = rho[threadIdx.x];
}
// Begins layer id's sumcheck. (A layer whose first round tail_layer() runs begins there instead.)
// slots: the layer's first message slot. One warp. order 0: round j binds the variable of the
// point's coordinate j. order 1 (range order; index bit v has coordinate mu - 1 - v; mu - low >= 9,
// not checked): above the lowest `low` bits, which number a tree, the rounds bind bits 2 1 0,
// 5 4 3, 8 7 6, then the rest from the top down; then the low bits from the top down.
__global__ void k_layer_begin(St *st, const Chal<F> *ch, int mu, int id, int order, T *slots, int low) {
  if (threadIdx.x == 0) st->add0 = F::zero();
  (void)sc_door::begin_by_warp<sc_door::Form::EQ2>(
      sc_door::Site::LAYER_GRAM, &st->tr, slots, mu,
      [=](int j) {
        const int bit = j < 9 ? low + (j / 3 * 3 + 2 - j % 3) : j < mu - low ? mu - 1 - (j - 9) : mu - 1 - j;
        return ch->rho[order == 0 ? j : mu - 1 - bit];
      },
      st->start[id], 2);
}
// Begins the table side's leaf layer (TAIL_LEAVES): round j binds point coordinate j. One warp.
__global__ void k_leaves_begin(St *st, const Chal<F> *ch, int mu, int id, T *slots) {
  if (threadIdx.x == 0) st->add0 = F::one();
  (void)sc_door::begin_by_warp<sc_door::Form::EQ>(
      sc_door::Site::LAYER_GRAM, &st->tr, slots, mu, [=](int j) { return ch->rho[j]; }, st->start[id], 0,
      [=](int j) { return j < mu - 5 ? 2 : 1; });
}
__global__ void k_plout(const St *st, const Chal<F> *nx, int idend, int mu, PtPLOut *o, Tr *t) {
  if (threadIdx.x < mu) o->r[threadIdx.x] = nx->rho[1 + threadIdx.x];
  if (threadIdx.x == 0) {
    o->tau = nx->rho[0];
    o->vL = st->endv[idend][0];
    o->vR = st->endv[idend][1];
    *t = st->tr;
  }
}
#include <thread>
#include <mutex>
#include <condition_variable>
#include <functional>
#include <deque>
#include <unordered_map>
#include <map>
#include <memory>
// The hook for every GPU allocation by PA() or PAN(). The third argument is the buffer's kind:
// 0: lives as long as its object, and nothing else may overwrite it: for example transcript states,
//    challenge arrays, Eng::part, small tables and claims.
// 1: ForestTable::counts() and planes() (forest_table.cuh) write it; its later steps read it.
// 2: scratch memory of Forest::bits() (forest.cuh) or of a prove(); means nothing after prove().
// 3: like 0, for launch descriptors, each written once and then only read until the process exits.
//    They hold layers' sizes and ids and GPU addresses, no value of a request and no random value.
//    Code that zeroes memory between runs must skip them: Sched::slot_of() uploads each only once.
// Public source for sorting buffers by lifetime: see hiding/place.cuh, Place.
// Public source for kind 3: memory labelled as not sensitive: J. Chow, B. Pfaff, T. Garfinkel,
// M. Rosenblum, "Shredding Your Garbage: Reducing Data Lifetime Through Secure Deallocation", 14th
// USENIX Security Symposium, 2005, section 6.2 and appendix A. There freed memory is zeroed only if
// labelled sensitive; here the label is given at allocation and exempts a buffer from zeroing.
static cudaError_t (*pt_alloc)(void **, size_t, int) = [](void **p, size_t n, int) {
  return cudaMalloc(p, n);
};
// Allocates an array of T. n is 16 times the number of elements; each element gets 8 * EDEG bytes,
// and the array 256 bytes more.
#define PA(p, n, kind) CK(pt_alloc((void **)&(p), ((size_t)(n) * EDEG + 1) / 2 + 256, kind))
// Allocates exactly n bytes.
#define PAN(p, n, kind) CK(pt_alloc((void **)&(p), (size_t)(n), kind))
#ifndef SERVING_BUILD
// Test hook: receives memory whose content is no longer needed. Eng::gram_slab_in() calls it twice
// before a Gram pass: with the part of S[] that its argument declares free (own = false), whether
// that pass uses it or not, then with the work buffer sc (own = true), of which no pass reads an
// earlier pass's data. Forest::dead() and lanes_dead() (forest.cuh) call it too. A test can set it
// to overwrite the memory on the caller's stream g_st: if a pass read what it had not written, a
// compared value would change. The memory then counts as written for Compute Sanitizer's initcheck,
// which therefore cannot report a read of never-written bytes there. No file in this tree sets it.
static void (*pt_free_memory)(void *, size_t, bool own) = nullptr;
#define PT_FREE_MEMORY(p, n, own) (pt_free_memory ? pt_free_memory(p, n, own) : (void)0)
#else
#define PT_FREE_MEMORY(p, n, own) ((void)0)
#endif
// A caller's memory region for kind-2 buffers; one PtTurn after another can give out the
// same bytes. While a PtTurn(r, at, bytes) exists, a kind-2 buffer comes from r's bytes [at, at +
// bytes) if it fits; every other buffer comes from the pt_alloc that was installed before.
struct PtRegion {
  char *base = nullptr;
  size_t bytes = 0, pos = 0, end = 0; // pos and end are the current position and the limit of the current PtTurn
  // 1024 bytes is ForkMemory::step, the allocation step in prover/prover/fork_memory.h.
  static size_t rounded(size_t n) { return (n + 1023) & ~(size_t)1023; }
};
static PtRegion *pt_region = nullptr;
static cudaError_t (*pt_alloc_under)(void **, size_t, int) = nullptr;
// See PtRegion. Only one PtTurn may exist at a time (not checked).
struct PtTurn {
  PtTurn(PtRegion &r, size_t at, size_t bytes) {
    pt_region = &r;
    r.pos = at;
    r.end = std::min(at + bytes, r.bytes);
    pt_alloc_under = pt_alloc;
    pt_alloc = [](void **p, size_t n, int kind) {
      PtRegion &r = *pt_region;
      const size_t m = PtRegion::rounded(n);
      if (kind != 2 || !r.base || r.pos + m > r.end) return pt_alloc_under(p, n, kind);
      *p = r.base + r.pos;
      r.pos += m;
      return cudaSuccess;
    };
  }
  ~PtTurn() {
    pt_alloc = pt_alloc_under;
    pt_region = nullptr;
  }
};
// A Gram pass (gram_products.cuh) runs three rounds of a layer. A layer's tables go through Gram
// passes until they have at most 2^lmid entries (or fewer than 2^11); the persistent kernel k_tails
// (tails.cuh) runs the remaining rounds, the layer's tail. The messages do not depend on lmid.
// pt_tail_bits() caps lmid: above 2^16 entries a Gram pass is quicker.
constexpr int pt_tail_cap = 16;
// A chain (struct Eng) is a sequence of layers on one transcript state, such as one side of a
// lookup call. The chains of one Sched::run() run independently of each other: while one chain
// waits for the hashing of its tail, the large kernels of the others can use the GPU. A tail is
// then one launch of at most PT_FREE_BLOCKS blocks.
// Public source: streams whose kernels run at the same time: M. Harris, "GPU Pro Tip: CUDA 7
// Streams Simplify Concurrency", NVIDIA Technical Blog
// (developer.nvidia.com/blog/gpu-pro-tip-cuda-7-streams-simplify-concurrency/).
// Here each chain takes a non-blocking stream from its proving thread's set (struct SchedStreams),
// and only one of the chains' host threads runs at a time.
constexpr int PT_FREE_BLOCKS = 8;
static inline int pt_tail_bits(int asked) {
  return std::min(asked, pt_tail_cap);
}
// The highest stream priority, for kernels that a chain waits for and that mostly wait for the
// transcript's hash: pt_hi() runs k_g2msgT, tside::k_end and others at it, and grm::k_sum, which
// precedes k_g2msgT in a stored layer's Gram pass; launch_tail() uses it if Sched::isfree. A CUDA
// graph respects its nodes' priority if instantiated with cudaGraphInstantiateFlagUseNodePriority
// (NVIDIA, CUDA Runtime API, "Graph Management", cudaGraphInstantiate()).
static inline int pt_high() {
  static int hi = [] {
    int lo = 0, h = 0;
    CK(cudaDeviceGetStreamPriorityRange(&lo, &h));
    return h;
  }();
  return hi;
}
template <class... KA, class... A>
static inline void pt_hi(void (*k)(KA...), dim3 g, dim3 b, size_t sm, cudaStream_t s, A &&...a) {
  cudaLaunchConfig_t c = {};
  c.gridDim = g;
  c.blockDim = b;
  c.dynamicSmemBytes = sm;
  c.stream = s;
  cudaLaunchAttribute at[1];
  at[0].id = cudaLaunchAttributePriority;
  at[0].val.priority = pt_high();
  c.attrs = at;
  c.numAttrs = 1;
  CK(cudaLaunchKernelEx(&c, k, std::forward<A>(a)...));
}
static int g_grid = 0;
// The calling thread's main stream for kernel launches; the cudaMemset() in Eng::init() does not
// use it. The programs are compiled with --default-stream per-thread, so the default, 0, is that
// thread's default stream. In Sched::run() each chain's host thread sets it to the chain's stream.
static thread_local cudaStream_t g_st = 0;
static inline bool capturing() {
  cudaStreamCaptureStatus cs = cudaStreamCaptureStatusNone;
  cudaStreamIsCapturing(g_st, &cs);
  return cs != cudaStreamCaptureStatusNone;
}
struct Eng;
// One host thread's CUDA streams and events: cs[q] with its event ce[q]; cm for the main stream,
// g_st. Sched::run() and Forest (forest.cuh) share them: they run one after another, and each
// joins what it used to g_st. side_streams::join(s) makes s wait for every cs[q].
struct SchedStreams {
  cudaStream_t cs[32];
  cudaEvent_t ce[32], cm;
  SchedStreams() {
    CK(cudaEventCreateWithFlags(&cm, cudaEventDisableTiming));
    for (int q = 0; q < 32; q++) {
      CK(cudaStreamCreateWithFlags(&cs[q], cudaStreamNonBlocking));
      CK(side_streams::enter(cs[q]));
      CK(cudaEventCreateWithFlags(&ce[q], cudaEventDisableTiming));
    }
  }
};
// Returns the calling thread's set. At its first call a thread takes a set that a finished thread
// has returned, or creates one; it returns the set when it ends. So no two live threads hold the
// same set. Sets are reused and never destroyed, as exit() may come while a scheduler runs.
static const SchedStreams &sched_streams() {
  struct GivenBack {
    std::mutex m;
    std::vector<const SchedStreams *> sets;
  };
  static GivenBack *given_back = new GivenBack;
  struct Held {
    const SchedStreams *set = nullptr;
    Held() {
      {
        std::lock_guard<std::mutex> l(given_back->m);
        if (!given_back->sets.empty()) set = given_back->sets.back(), given_back->sets.pop_back();
      }
      if (!set) set = new SchedStreams;
    }
    ~Held() {
      std::lock_guard<std::mutex> l(given_back->m);
      given_back->sets.push_back(set);
    }
  };
  static thread_local Held held;
  return *held.set;
}
// The stream for all schedulers' uploads. Threads can share it: each upload is waited for at once.
static cudaStream_t upload_stream() {
  static const cudaStream_t us = [] {
    cudaStream_t made;
    CK(cudaStreamCreateWithFlags(&made, cudaStreamNonBlocking));
    CK(side_streams::enter(made));
    return made;
  }();
  return us;
}
// launch() runs a chain's tail list in one k_tails launch. run() runs several chains' host code in
// turns: `turn` is the running chain, or -1 for run() itself. Both are defined after struct Eng.
struct Sched {
  std::mutex m;
  std::condition_variable cv;
  int turn = -1;
  // The device copies of the tail launches' descriptors, keyed by content and never overwritten: a
  // captured graph or a launch in flight keeps pointing at the descriptor that it was issued with.
  std::unordered_map<std::string, u8 *> store;
  u8 *chunk = nullptr;
  size_t coff = 0, ccap = 0;
  // If true, launch() uses at most PT_FREE_BLOCKS blocks, at high priority, so that large kernels
  // on other streams can use the GPU while a tail waits for hashes. If false, a tail gets as many
  // blocks as SMs. run() sets it; Forest::prove() (forest.cuh) copies forest_small_tails into it.
  bool isfree = false;
  // Returns the device copy of the descriptor bytes b; uploads them when it first sees them.
  u8 *slot_of(const std::string &b) {
    auto it = store.find(b);
    if (it != store.end()) return it->second;
    const size_t len = (b.size() + 15) & ~(size_t)15;
    if (coff + len > ccap) {
      ccap = std::max<size_t>((size_t)1 << 20, len);
      PAN(chunk, ccap, 3);
      coff = 0;
    }
    u8 *d = chunk + coff;
    coff += len;
    const cudaStream_t us = upload_stream();
    CK(cudaMemcpyAsync(d, b.data(), b.size(), cudaMemcpyHostToDevice, us));
    // The copy must be on the GPU before any reader is launched, on any stream. b is pageable, so a
    // cudaMemcpy()'s DMA may outlast the call, and cudaStreamNonBlocking streams do not wait for it.
    CK(cudaStreamSynchronize(us));
    store.emplace(b, d);
    return d;
  }
  void give(int q) {
    std::unique_lock<std::mutex> l(m);
    turn = q;
    cv.notify_all();
    cv.wait(l, [&] { return turn == -1; });
  }
  void yield(int me) {
    std::unique_lock<std::mutex> l(m);
    turn = -1;
    cv.notify_all();
    cv.wait(l, [&] { return turn == me; });
  }
  void launch(Eng *e);
  void run(std::vector<std::function<void()>> f);
};
#include "hiding/gram_products.cuh" // grm::
// A table-side layer has 2^lk planes, selected by the lowest lk index bits. If ka > 0, only the
// planes j < ka are stored, entry (h, j) at h * ka + j; `mode` is k_expand's mode for the rest.
struct Cmp {
  int ka = 0, lk = 0, mode = 0;
};
// A chain: its transcript state, scratch memory and tail list (ll: layers awaiting a tail launch).
struct Eng {
  St *st;
  Chal<F> *chd;
  T *S[2], *B[2], *part, *eqlo, *eqhi, *G, *k8, *out = nullptr;
  void *sc;
  size_t scb = 0, s_bytes = 0, own_cap = 0; // bytes of sc; of each S[i]; sc's digit-matrix columns
  T *Gcb, *X = nullptr;
  M3L *eqprep = nullptr;
  int lmid;
  LL ll;
  Sched *sch = nullptr, *solo = nullptr;
  int slot = 0;
#include "hiding/gram_engine.inc" // gram_slab, gram_slab_bytes(), gram_slab_lay(), gram_slab_in(), lib_pass()
  // Allocates the state and scratch memory. big: entries of one table of the largest stored layer.
  void init(size_t big, int lmid_, Chal<F> *chd_) {
    lmid = lmid_;
    chd = chd_;
    ll.n = 0;
    PAN(st, sizeof(St), 0);
    // A multiple of 32 entries, so that S[1] is 256-byte aligned as S[0] is.
    const size_t m = (std::max<size_t>(big / 4, 1 << 16) + 31) / 32 * 32;
    PA(S[0], 32 * m, 2);
    S[1] = S[0] + m;
    s_bytes = m * sizeof(T);
    PA(B[0], (size_t)64 << lmid, 2);
    PA(B[1], (size_t)64 << lmid, 2);
    PA(part, 16 * 8 * 4096, 0);
    CK(cudaMemset(part, 0, 8 * EDEG * 8 * 4096));
    PA(eqlo, 16 << 16, 2);
    PA(eqhi, 16 << 14, 2);
    PA(G, 64 * 16, 2);
    PA(k8, 8 * 16, 2);
    // The work buffer of the Gram passes: digits and sums.
    grm::set_up();
    scb = gram_slab_bytes(big);
    own_cap = grm::Slab::cap_for(big);
    PAN(sc, scb, 2);
    gram_slab_lay(sc, big);
    // 2^16 entries of 40 bytes hold a layer's low-bit eq tables, of distinct sizes up to 2^15.
    PAN(eqprep, (size_t)40 << 16, 2);
    // 64 sums per chunk
    PA(Gcb, ((std::max<size_t>(big / 8, 1 << 15) >> 15) + 1) * 1024, 2);
  }
  void mid();
  void flush() {
    if (ll.n) mid();
  }
  T *take(int mu) {
    T *o = out;
    out += sc_door::layer_slots(mu);
    return o;
  }
  // Runs Gram passes on stored tables (L and then R, ent(ls) entries each) and puts the remaining
  // rounds on the tail list. A layer that used S[] (or X) must be finished before that buffer is
  // written again. continues: the rounds continue a sumcheck that has begun but are numbered from
  // 0 (mu: only the variables left). pp0: the first fold writes S[pp0]; no table may be there.
  void gram_rest(const T *tab, int mu, int j0, int id, int idn, T *o, const T *Rsep = nullptr, int pp0 = 0,
                 Cmp cp = Cmp(), bool continues = false, TailKind kind = TAIL_PRODUCT) {
    int j = j0, ls = mu - j0, pp = pp0;
    const int ka = cp.ka, kk = ka ? ka : 1, lk = ka ? cp.lk : 0;
    int mx = 0;
    while ((kk << (mx + 1)) <= 32768) mx++; // a chunk has at most 2^15 entries = 2^mx rows of kk planes
    // The number of stored entries of one table whose full size is 2^l.
    auto ent = [&](int l) {
      return (size_t)kk << (l - lk);
    };
    auto lx = [&](int l) {
      return std::min(l - 3 - lk, mx);
    };
    auto more = [&](int l) {
      return l > lmid && l - 3 >= 8 && l - 3 - lk >= (ka ? 4 : 0);
    };
    // R: at Rsep if given, else after L. Rsep needs a Gram pass to follow (more(ls)): not checked.
    const T *L = tab, *R = Rsep ? Rsep : tab + ent(ls);
    Chal<F> *cd = chd + id;
    const M3L *lo[8];
    const T *hi[8];
    {
      EqJobs J;
      J.n = 0;
      int np = 0, xcur = -1;
      size_t olo = 0, ohi = 0;
      const M3L *cl = nullptr;
      for (int l2 = ls, j2 = j; more(l2); l2 -= 3, j2 += 3, np++) {
        const int x = lx(l2), nh = l2 - 3 - lk - x;
        if (x != xcur) {
          J.add(mu - lk - x, x, nullptr, ka, lk, eqprep + olo);
          cl = eqprep + olo;
          olo += (size_t)kk << x;
          xcur = x;
        }
        lo[np] = cl;
        J.add(j2 + 3, nh, eqhi + ohi);
        hi[np] = eqhi + ohi;
        ohi += (size_t)1 << nh;
      }
      // 128 * 256 threads for each table: a table has at most 2^15 entries.
      if (J.n) k_eqjobs<<<dim3(128, J.n), 256, 0, g_st>>>(cd, J);
    }

    const bool begins = j0 == 0 && !continues;
    if (begins && more(ls)) pt_hi(k_layer_begin, 1, 32, 0, g_st, st, cd, mu, id, 0, o, 0);
    for (int p = 0; more(ls); p++) {
      const int x = lx(ls);
      const size_t SC = ent(ls - 3), chunk = (size_t)kk << x;
      // S[pp] is free until fold8_k3() below writes to it (no table of a pass is stored there), and
      // so is S[pp ^ 1] unless a table of this pass is stored there.
      auto in_s = [&](const T *t) {
        return (uintptr_t)t >= (uintptr_t)S[0] && (uintptr_t)t < (uintptr_t)S[1] + s_bytes;
      };
      gram_slab_in(in_s(L) || in_s(R) ? pp : 2);
      lib_pass(L, R, SC, chunk, lo[p], hi[p], cd, j, o, ka && cp.mode == 0 ? ka : 0, lk, mu - lk, g_st);
      T *Ln = S[pp], *Rn = Ln + SC;
      fold8_k3(L, R, SC, SC, k8, Ln, Rn, g_st);
      L = Ln;
      R = Rn;
      pp ^= 1;
      o += 6;
      j += 3;
      ls -= 3;
    }
    if (ka) {
      if (ls > lmid) {
        printf("product test: a layer with omitted planes still has %d index bits after the "
               "Gram passes, more than lmid = %d (the second argument of init)\n",
               ls, lmid);
        name_the_end("forest-table-rest"); // the forest table side's only caller of gram_rest: p258 end 1 forest-table-rest
        exit(1);
      }
      const size_t nh = (size_t)1 << (ls - lk);
      k_expand<<<(int)std::min<size_t>(((nh << (lk + 1)) + 255) / 256, 1056), 256, 0, g_st>>>(L, X, nh, ka, lk,
                                                                                              cp.mode);
      L = X;
    }
    int i = ll.n++;
    ll.src[i] = L;
    ll.out[i] = o;
    ll.mu[i] = mu;
    ll.j0[i] = j;
    ll.id[i] = id;
    ll.idn[i] = idn;
    ll.begins[i] = begins && j == j0;
    ll.kind[i] = kind;
    if (j > j0 || mu > lmid || ll.n == 24 || ka) mid();
  }
  void stored(const T *tab, int mu, int id, int idn, Cmp cp = Cmp()) {
    T *o = take(mu);
    if (mu > lmid || cp.ka) flush();
    gram_rest(tab, mu, 0, id, idn, o, nullptr, 0, cp);
  }
};
// struct KeyPasses: the layers that are computed from 16-bit keys.
#include "hiding/gram_keys.inc"
#include "hiding/tails.cuh" // k_tails, launch_tail()
// Runs chain e's tail list in one k_tails launch of as many blocks as SMs, and empties it. If
// isfree, the launch instead has high priority and w / 1024 blocks, at least 1 and at most
// PT_FREE_BLOCKS, where w = 2^r for the most rounds r that a layer of the list has left. The
// descriptor (ChainD) starts as zeros, and unused entries stay 0: slot_of() compares all its bytes.
inline void Sched::launch(Eng *e) {
  std::string b(sizeof(ChainD), '\0');
  ChainD &c = *reinterpret_cast<ChainD *>(&b[0]);
  size_t largest = 1;
  c.ll.n = e->ll.n;
  for (int i = 0; i < c.ll.n; i++) {
    largest = std::max(largest, (size_t)1 << (e->ll.mu[i] - e->ll.j0[i]));
    c.ll.src[i] = e->ll.src[i];
    c.ll.out[i] = e->ll.out[i];
    c.ll.mu[i] = e->ll.mu[i];
    c.ll.j0[i] = e->ll.j0[i];
    c.ll.id[i] = e->ll.id[i];
    c.ll.idn[i] = e->ll.idn[i];
    c.ll.begins[i] = e->ll.begins[i];
    c.ll.kind[i] = e->ll.kind[i];
  }
  c.st = e->st;
  c.chd = e->chd;
  c.B0 = e->B[0];
  c.B1 = e->B[1];
  c.part = e->part;
  e->ll.n = 0;
  const int blocks = isfree ? (int)std::max<size_t>(1, std::min<size_t>(PT_FREE_BLOCKS, largest / 1024)) : g_sms;
  launch_tail(reinterpret_cast<const ChainD *>(slot_of(b)), blocks, isfree, g_st);
}
// Runs the functions f, one per chain (at most 32, not checked), each on a separate host thread and
// a separate stream of the caller's SchedStreams, one at a time, in turns. Events order each stream
// after the work that g_st holds at the call, and g_st after every stream. Only a test calls run().
inline void Sched::run(std::vector<std::function<void()>> f) {
  const int n = (int)f.size();
  std::vector<char> done(n, 0);
  std::vector<std::thread> th;
  turn = -1;
  const cudaStream_t mainS = g_st;
  // This thread's set of streams. The chains' threads below use it and take no set of their own.
  const SchedStreams *s = &sched_streams();
  for (int q = 0; q < n; q++)
    th.emplace_back([&, q] {
      g_st = s->cs[q];
      {
        std::unique_lock<std::mutex> l(m);
        cv.wait(l, [&] { return turn == q; });
      }
      f[q]();
      // CUDA holds the latest error of a chain's rejected launches for the chain's own host thread
      // (NVIDIA, CUDA Runtime API, Error Handling), so each chain's thread reads it itself.
      CK(cudaGetLastError());
      std::unique_lock<std::mutex> l(m);
      done[q] = 1;
      turn = -1;
      cv.notify_all();
    });
  isfree = true;
  CK(cudaEventRecord(s->cm, mainS));
  for (int it = 0;; it++) {
    bool any = false;
    for (int q = 0; q < n; q++)
      if (!done[q]) {
        any = true;
        if (it == 0) CK(cudaStreamWaitEvent(s->cs[q], s->cm, 0));
        give(q);
        if (done[q]) {
          CK(cudaEventRecord(s->ce[q], s->cs[q]));
          CK(cudaStreamWaitEvent(mainS, s->ce[q], 0));
        }
      }
    if (!any) break;
  }
  for (auto &t : th) t.join();
}
// Launches this chain's tail list: through sch, and then yields its turn to the scheduler, or, if
// sch is null, through solo. Not checked: under sch, slot is the chain's index in run()'s f.
inline void Eng::mid() {
  if (sch) {
    sch->launch(this);
    sch->yield(slot);
  } else
    solo->launch(this);
}
#include "hiding/table_side.cuh"
// A cache of instantiated CUDA graphs. pt_graphed() says what the map's key must determine. No
// code destroys a cached graph.
// Public source for the graphs: NVIDIA, CUDA C++ Programming Guide, "CUDA Graphs".
// See also: a graph captured from a stream and launched many times: NVIDIA, CUDA Programming
// Guide, "CUDA Graphs",
// https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html.
struct PtGraph {
  struct Entry {
    cudaGraphExec_t first = nullptr; // the graph, once captured
    int second = 0;                  // calls so far that ran without a graph
  };
  std::map<std::vector<const void *>, Entry> m;
};
// Runs f() as a CUDA graph cached in G under key, which must determine every address and number in
// f()'s launches. f() may launch only into g_st or streams forked from and joined to it by events;
// it must not synchronize with the host, allocate or depend on host data. The first `warm` calls
// for a key, and calls inside another capture, run f() directly.
inline void pt_graphed(PtGraph &G, const std::vector<const void *> &key, const std::function<void()> &f,
                         int warm = 1) {
  if (capturing()) {
    f();
    return;
  }
  auto &e = G.m[key];
  if (e.first) {
    CK(cudaGraphLaunch(e.first, g_st));
    return;
  }
  if (e.second++ < warm) {
    f();
    return;
  }
  cudaGraph_t g;
  CK(cudaStreamBeginCapture(g_st, cudaStreamCaptureModeRelaxed));
  f();
  CK(cudaStreamEndCapture(g_st, &g));
  CK(cudaGraphInstantiate(&e.first, g, cudaGraphInstantiateFlagUseNodePriority));
  CK(cudaGraphDestroy(g));
  CK(cudaGraphLaunch(e.first, g_st));
}
// A single product layer on 2^(mu+1) 16-bit keys, mu >= 16: the sumcheck for claim = sum_c
// eq(rho, c) (cst - key[c]) (cst - key[2^mu + c]). Its rounds bind the bits of c in order 1 of
// k_layer_begin(). It runs on the caller's transcript and sends 2 mu + 5 elements; result: *out.
struct PtPL {
  int mu, lN;
  size_t N;
  Eng E;
  Sched solo;
  Chal<F> *chd;
  T *Fo, *G80, *k8b, *K64;
  void *scA;
  PtPLOut *out;
  static int clampi(int x, int a, int b) { return x < a ? a : x > b ? b : x; }
  // Bytes of scA, scratch memory of KeyPasses (gram_keys.inc): the chunks' sums, 80 elements per
  // chunk in pass 1 and 64 in passes 2 and 3, for the chunk sizes that prove() uses.
  static size_t scA_bytes(size_t N, int lN) {
    const int lc1 = clampi(lN - 12, 10, 15), lc2 = clampi(lN - 12, 7, 15), lc3 = clampi(lN - 13, 7, 15);
    const size_t a = ((N / 16) >> lc1) * 80, b = ((N / 128) >> lc2) * 64, c = ((N / 1024) >> lc3) * 64;
    return std::max(a, std::max(b, c)) * sizeof(T) + 256;
  }
  // Allocates the buffers. The program exits if mu_ < 16. lmid: see pt_tail_cap.
  void init(int mu_, int lmid = 18) {
    if (!g_sms) dev_info();
    g_grid = g_sms * 8;
    mu = mu_;
    lN = mu + 1;
    N = (size_t)1 << lN;
    if (lN < 17) {
      printf("product test: rejected: PtPL needs mu >= 16 (mu = %d)\n", mu);
      fflush(stdout);
      exit(2);
    }
    PAN(chd, sizeof(Chal<F>) * 4, 0);
    E.init(N / 16, pt_tail_bits(lmid), chd);
    E.solo = &solo;
    PA(Fo, 16 * (N / 512), 2);
    PA(G80, 80 * 16, 2);
    PA(k8b, 8 * 16, 2);
    PA(K64, 64 * 16, 2);
    PAN(scA, scA_bytes(N, lN), 2);
    PAN(out, sizeof(PtPLOut), 0);
  }
  size_t nmsg() const { return sc_door::layer_slots(mu); }
  // Device pointers: tr, keys, rho_dev (mu coordinates), msg (gets nmsg() elements). Fills *out.
  // claim_dev: the true value of the claimed sum, on the GPU. Not checked: keys is 16-byte aligned.
  void prove(Tr *tr, const uint16_t *keys, const T *rho_dev, T cst, const T *claim_dev, T *msg) {
    E.sch = nullptr;
    E.out = msg;
    Chal<F> *cd = chd;
    k_plin<<<1, 32, 0, g_st>>>(E.st, tr, cst, cd, rho_dev, mu, claim_dev, 0);
    T *o = E.take(mu);
    pt_hi(k_layer_begin, 1, 32, 0, g_st, E.st, cd, mu, 0, 1, o, 0);
    const size_t SC = N / 1024;
    T *UL = E.S[1], *UR = E.S[1] + 8 * SC;
    auto eqs = [&](int nbits, int lc) {
      k_eqtabT<<<64, 256, 0, g_st>>>(cd, nbits - lc, lc, E.eqlo);
      k_eqtabT<<<4, 256, 0, g_st>>>(cd, 0, nbits - lc, E.eqhi);
    };
    const int lc1 = clampi(lN - 12, 10, 15), lc2 = clampi(lN - 12, 7, 15), lc3 = clampi(lN - 13, 7, 15);
    KeyPasses lib{E, keys, N, scA, G80, K64, g_st};
    eqs(mu - 3, lc1);
    lib.a1(nullptr, lc1);
    k_finA80<<<1, 64, 0, g_st>>>(G80, E.st, E.G);
    pt_hi(k_g2msgT, 1, 64, 0, g_st, E.G, E.st, cd, 0, mu - 2, mu - 1, o, E.k8);
    eqs(mu - 6, lc2);
    lib.a2(nullptr, lc2);
    pt_hi(k_g2msgT, 1, 64, 0, g_st, E.G, E.st, cd, 3, mu - 5, mu - 4, o + 6, k8b);
    k_k64<<<1, 64, 0, g_st>>>(E.k8, k8b, K64);
    eqs(mu - 9, lc3);
    lib.a3(nullptr, lc3, UL, UR);
    pt_hi(k_g2msgT, 1, 64, 0, g_st, E.G, E.st, cd, 6, mu - 8, mu - 7, o + 12, E.k8);
    fold8_k3(UL, UR, SC, SC, E.k8, Fo, Fo + SC, g_st);
    k_cprho<<<1, 32, 0, g_st>>>(cd, chd + 1, mu - 9);
    E.gram_rest(Fo, mu - 9, 0, 1, 2, o + 18, nullptr, 0, Cmp(), true);
    E.flush();
    k_lowrho<<<1, 32, 0, g_st>>>(cd, chd + 2, mu - 9);
    k_plout<<<1, 32, 0, g_st>>>(E.st, chd + 2, 1, mu, out, tr);
  }
};
// A single product layer on arrays of field elements: the sumcheck for claim = sum_c eq(rho, c)
// L(c) R(c), for 16-byte aligned (not checked) device arrays L and R of 2^mu elements of T, which
// may be the same array. It binds the top bit first. Transcript, nmsg() and result: as for PtPL.
struct PtWPL {
  int mu;
  Eng E;
  Sched solo;
  Chal<F> *chd;
  T *cat = nullptr;
  PtPLOut *out;
  void init(int mu_, int lmid = 18) {
    if (!g_sms) dev_info();
    g_grid = g_sms * 8;
    mu = mu_;
    lmid = pt_tail_bits(lmid);
    PAN(chd, sizeof(Chal<F>) * 4, 0);
    E.init((size_t)1 << mu, lmid, chd);
    E.solo = &solo;
    if (mu <= lmid) PA(cat, 32ull << mu, 2);
    PAN(out, sizeof(PtPLOut), 0);
  }
  size_t nmsg() const { return sc_door::layer_slots(mu); }
  // Eng::gram_rest() takes a separate R only if a Gram pass follows. So if mu <= E.lmid, L and R
  // are first copied into cat, one after the other. E.lmid must be at least 10 (not checked), or a
  // larger mu may get no Gram pass. On the GPU: tr, rho_dev, msg, claim_dev (the sum's true value).
  void prove(Tr *tr, const T *L, const T *R, const T *rho_dev, const T *claim_dev, T *msg) {
    E.sch = nullptr;
    E.out = msg;
    k_plin<<<1, 32, 0, g_st>>>(E.st, tr, F::zero(), chd, rho_dev, mu, claim_dev, 0);
    T *o = E.take(mu);
    if (mu <= E.lmid) {
      CK(cudaMemcpyAsync(cat, L, sizeof(T) << mu, cudaMemcpyDeviceToDevice, g_st));
      CK(cudaMemcpyAsync(cat + ((size_t)1 << mu), R, sizeof(T) << mu, cudaMemcpyDeviceToDevice, g_st));
      E.gram_rest(cat, mu, 0, 0, 1, o);
    } else
      E.gram_rest(L, mu, 0, 0, 1, o, R);
    E.flush();
    k_plout<<<1, 32, 0, g_st>>>(E.st, chd + 1, 0, mu, out, tr);
  }
};