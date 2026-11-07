// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// Proves a group of lookup calls on the GPU as one lookup (LKP, blocks/iop/lkp.h: every lookup key
// occurs in a public lookup table), by product trees over the keys (key side) and over the tables'
// rows (table side). Forest::prove() draws the shift theta (none for one call) and one challenge
// alpha; call c's trees use alpha - c theta. No root is sent unmasked: sumchecks on hidden values
// tie the two sides' roots. One sumcheck per level (a layer) covers all trees, the key side first,
// on the caller's one transcript (`Tr`, prover/hiding/transcript.cuh). Storage, lanes (parallel
// streams) and the choice of kernels do not change the proof's bytes. Entry points: Forest,
// forest_scratch_bytes(), forest_prove_graph(). Include after prover/hiding/product_test.cuh.
// Leaving out the trees that hold padding alone ("Padding", below) was prompted by advice of the
// researchers whom README.md thanks to lessen the lookups' work
// (README.md, "Licence and acknowledgments").
// Sizes (prover/hiding/forest_plan.h): call c has 2^n_c keys (table row numbers), cut into 2^iota_c
// trees of 2^dT keys, numbered from first_c. The NC calls have NR trees; numbers NR .. NT - 1,
// NT = 2^rK, are trees of padding only (copies of call 0's public padding key). The flat order
// (x | y) is the index within a tree (high bits), then the tree's number. A leaf is
// alpha - tau(key), where tau(t) is row t's value. Level d of a tree has 2^d nodes, each the
// product of 2^LV leaves, LV = dT - d; its halves are the tables L and R of the layer that reads
// it. A Gram pass (prover/hiding/gram_products.cuh, namespace grm) computes the 64 sums that three
// rounds of a layer need: three index bits cut L and R into 8 slabs each, and a slab's indices are
// cut into chunks. The tail kernel (prover/hiding/tails.cuh) runs a layer's last rounds.
#pragma once
#include "hiding/forest_plan.h"
#include "hiding/forest_keys.cuh"
#include "hiding/forest_fused.cuh"
#include "hiding/forest_fused_folds.cuh"
#include "hiding/forest_fused_first.cuh"
#include "hiding/forest_fused_second.cuh"
#include "hiding/forest_table.cuh"
#include "hiding/lookup_tie.cuh"
#include "common/end_name.h" // name_the_end(): names an internal stop before a direct exit (M5e3)
// The claims that a group's proof leaves to the caller: filled by Forest::prove(), in GPU memory.
// (zx | zy) is the key side's end point. Coordinate 0 of a point belongs to the top bit.
// (t | c | j): table row (kT bits), call (cK bits), bit of the multiplicity; plane j holds bit j of
// every multiplicity. The table side's last two sumchecks are the leaves' (summand
// 1 + b (alpha - 1 - tau), b a multiplicity bit) and the bits' (summand b (b - 1)). vL and vR are a
// layer's two end values, zeta the challenge drawn after them. err: the table side's St::err
// (product_test.cuh). theta and alpha must stay adjacent: lookup_tie::k_draws writes both through
// &theta.
struct ForestClaims {
  T theta, alpha; // the shift (0 if the group has one call), then the lookup's challenge
  T a;            // the product of the leaves of all padding keys; public
  T rho5[5];      // the point that joins the planes' roots
  T zx[32];       // dT coordinates
  T zy[8];        // rK coordinates
  T e[16];        // e_c = eq(zy[0 .. rK - iota_c), first_c >> iota_c)
  T empty;        // the sum of eq(zy, y) over the trees of padding alone
  T zk[16][32];   // zk_c = (zy[rK - iota_c .. rK) | zx): n_c coordinates
  T cend;         // the key side's leaves at (zx | zy): its last layer's (1 - zeta) vL + zeta vR
  T zeta_end;     // that layer's zeta
  T zb[2][32];    // the leaves' and the bits' end points: kT + cK + 5 coordinates, (t | c | j)
  T vb[2];        // the planes' values there; only the second is sent, under a mask
  T leaves_factor; // alpha - 1 - (the table at zb[0]): what multiplies vb[0] in the leaves' summand
  u32 err;
};
#ifndef SERVING_BUILD
// Test hook: called on the host at each phase boundary of prove(), with the new phase's name.
static void (*forest_mark)(const char *) = nullptr;
#define FOREST_MARK(name) (forest_mark ? forest_mark(name) : (void)0)
#else
#define FOREST_MARK(name) ((void)0)
#endif
// Number of lanes (ForestLane), clamped to 1 .. 8. A Forest reads this and the next two settings
// when it is set up.
static int forest_lanes = 2;
// log2 of the columns of a lane's two digit matrices of grm::ROWS rows, clamped to 15 .. 20;
// 384 bytes per column. A digit matrix contains table entries cut into bytes, one index per column.
static int forest_lane_bits = 17;
// A lane's memory for sums in MiB, clamped to 10 .. 512.
static int forest_lane_sums_mb = 32;
// false: a tail launch has as many blocks as the GPU has SMs. true: at most PT_FREE_BLOCKS
// blocks, at high priority (Sched::isfree in product_test.cuh).
static bool forest_small_tails = false;
// true: the kernels of forest_fused*.cuh (namespace ff) run where they exist, and a Gram pass then
// computes and multiplies its digits in one kernel. false: a pass writes its digits to GPU memory,
// and cublasLtMatmul multiplies them (gram_products.cuh). -DFOREST_FUSED=0 makes false the initial
// value; one benchmark, prover/tests/forest_lookup_bench.cu, flips it to compare the two (no test
// that SUITE.md lists runs it).
#ifndef FOREST_FUSED
#define FOREST_FUSED 1
#endif
static bool forest_fused = FOREST_FUSED != 0;
// Whether the key side leaves out the padding declared to Forest::counts(). A test can set it to
// false to compare; no file of this repository does.
static bool forest_leaves_padding_out = true;
namespace fr {
typedef grm::i8 i8;
// The trees that the key side walks (reads key by key): walked tree q < n is tree number tree[q].
struct Walked {
  int n = 0;
  unsigned char tree[256] = {};
};
// Up to 16 eq tables, one per blockIdx.y of k_eqs, each written to o as elements, to op in the
// prepared form M3L (pt_field.h), or to both.
//   ay < 0:   entry x is eq(rho[a .. a + nb), x), x < 2^nb
//   ay >= 0:  entry q 2^nb + x is that value times eq(rho[ay .. ay + rK), V.tree[q]), q < V.n
// The second form gives a pass its weights per chunk: a layer's point is (rho_x | rho_y), and tree
// y's sums enter under eq(rho_y, y).
struct Eqs {
  int n = 0, a[16] = {}, nb[16] = {}, ay[16] = {}; // 0 where unused: a launch copies all entries
  T *o[16] = {};
  M3L *op[16] = {};
  void add(int a_, int nb_, int ay_, T *o_, M3L *op_) {
    a[n] = a_, nb[n] = nb_, ay[n] = ay_, o[n] = o_, op[n] = op_;
    n++;
  }
};
__global__ void k_eqs(const Chal<F> *ch, const Eqs J, int rK, const Walked V) {
  __shared__ T tb[7][32], ty[2][32];
  const int q = blockIdx.y, a = J.a[q], n = J.nb[q], ay = J.ay[q];
  const size_t tot = (size_t)(ay < 0 ? 1 : V.n) << n;
  if (blockIdx.x * (size_t)blockDim.x >= tot) return;
  const int ng = eq_groups(ch->rho + a, n, tb);
  const int ngy = ay < 0 ? 0 : eq_groups(ch->rho + ay, rK, ty);
  __syncthreads();
  for (size_t e = blockIdx.x * (size_t)blockDim.x + threadIdx.x; e < tot; e += (size_t)gridDim.x * blockDim.x) {
    T v = eq_at(tb, n, ng, e & (((size_t)1 << n) - 1));
    if (ngy) v = F::mul(v, eq_at(ty, rK, ngy, V.tree[e >> n]));
    if (J.o[q]) J.o[q][e] = v;
    if (J.op[q]) J.op[q][e] = m3l(v);
  }
}
// grm::k_digits for tables stored tree by tree: A receives the digits of eqlo L, B those of R, for
// n indices from g0 on. Index g = (q | c), c < SC = 2^lsc, belongs to walked tree q; slab a of that
// tree begins at offset q ts + a SC of L and of R. g0, n and `chunk` must be multiples of 4, and
// `chunk` must divide SC. Blocks of 256 threads: warp a writes slab a.
__global__ void __launch_bounds__(256)
    k_digits(const K3 *__restrict__ L, const K3 *__restrict__ R, size_t ts, int lsc, size_t g0, int n, int chunk,
             const M3L *__restrict__ eqlo, i8 *__restrict__ A, i8 *__restrict__ B, size_t ld) {
  const int a = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const size_t SC = (size_t)1 << lsc;
  for (int c = blockIdx.x * 128 + lane * 4; c < n; c += gridDim.x * 128) {
    const size_t g = g0 + c, at = (g >> lsc) * ts + a * SC + (g & (SC - 1));
    const K3 *l = L + at, *r = R + at;
    const M3L *w = eqlo + g % chunk;
    K3 e[4];
#pragma unroll
    for (int k = 0; k < 4; k++) e[k] = FK2::mulp(w[k], l[k]);
    grm::put_digits(A + c, ld, a, e);
#pragma unroll
    for (int k = 0; k < 4; k++) e[k] = r[k];
    grm::put_digits(B + c, ld, a, e);
  }
}
// kg3::k_fold8 (pt_gram.cuh) with the addressing of k_digits:
// Lo[g] = sum_a k[a] L(tree q, slab a, c) for g = (q | c) < total, and Ro likewise. ts, SC and
// `total` must be even and the four tables 16-byte aligned: a thread loads and stores two
// neighbouring entries. Blocks of kg3::FOLD8_THREADS threads; gridDim.y must be 2.
__global__ void __launch_bounds__(kg3::FOLD8_THREADS)
    k_fold8(const K3 *__restrict__ L, const K3 *__restrict__ R, size_t ts, int lsc, size_t total,
            const K3 *__restrict__ k, K3 *__restrict__ Lo, K3 *__restrict__ Ro) {
  __shared__ M3L weight[8];
  if (threadIdx.x < 8) weight[threadIdx.x] = FK2::prep(k[threadIdx.x]);
  __syncthreads();
  const K3 *from = blockIdx.y ? R : L;
  K3 *to = blockIdx.y ? Ro : Lo;
  const size_t SC = (size_t)1 << lsc, step = 2 * (size_t)gridDim.x * kg3::FOLD8_THREADS;
  for (size_t g = 2 * (blockIdx.x * (size_t)kg3::FOLD8_THREADS + threadIdx.x); g < total; g += step) {
    const K3 *p = from + (g >> lsc) * ts + (g & (SC - 1));
    FK2::Acc even = FK2::azero(), odd = FK2::azero();
#pragma unroll 2
    for (int slab = 0; slab < 8; slab++) {
      const kg3::Two entries = kg3::load_two(p + slab * SC);
      FK2::mac(even, weight[slab], entries.first);
      FK2::mac(odd, weight[slab], entries.second);
    }
    kg3::store_two(to + g, FK2::ared(even), FK2::ared(odd));
  }
}
// v^(2^times).
__device__ inline T squared(T v, int times) {
  for (int i = 0; i < times; i++) v = F::mul(v, v);
  return v;
}
// at[y] = q >= 0: tree number y < NT is walked tree q. at[y] = -1 - c: the tree is not walked,
// because all its keys are declared to be call c's padding key, row t0_c; leaf[c] then points to
// that key's leaf, (call c's alpha) - tau_c(t0_c), in GPU memory. `unused` is there so that no byte
// of a kernel's argument is left unwritten.
struct TreePlaces {
  const T *leaf[16] = {};
  short at[256] = {};
  int all = 1, unused = 0; // all: every tree is walked
};
// Fills a call's table of leaves: lut[t] = alpha - tau[t] for t < R, or alpha - t if tau is null (a
// range table: row t has the value t). alpha is the call's. The row values tau[t] are below 2^61
// (fk::table_of() rejects any other table). Needs at least R threads.
// Public source: a table by the rule "store precomputed results": SGI, "Origin 2000 and Onyx2
// Performance Tuning and Optimization Guide" (document 007-3430-003), Appendix A "Bentley's Rules
// Updated", Space-for-Time Rules, "Store Precomputed Results", paraphrasing Appendix C of
// J. Bentley, "Writing Efficient Programs" (1982);
// https://tech-pubs.net/sgi_developer/books/007-3430-003/apa.html.
// Public source: the table is filled at run time, as in the OpenCV documentation (4.13.0), tutorial
// "How to scan images, lookup tables and time measurement with OpenCV", section "Our test case";
// https://docs.opencv.org/4.x/db/da5/tutorial_how_to_scan_images.html.
// Both speak of a costly computation; here it is one subtraction.
__global__ void k_lut_at(const T *alpha, const u64 *tau, int R, T *lut) {
  const int t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t < R) lut[t] = F::sub(*alpha, F::small(tau ? tau[t] : (u64)t));
}
// Copies `tables` tables (1 or 2: L, then R) stored tree by tree, 2^s entries per tree, walked tree
// q at offset q ts, to dst in the flat order, 2^(s + rK) entries each. The tail kernel needs that
// order: it binds the top bit of its tables first. Entries at y >= NT are 0 and would make the
// product 0, so Forest passes NT = 2^rK. An entry is a node over 2^LV leaves, or a sum of such
// nodes under weights that add up to 1, so in a tree that is not walked it is leaf^(2^LV).
// Blocks need at least 16 threads.
__global__ void k_flat(const T *__restrict__ L, const T *__restrict__ R, size_t ts, int s, int rK, int NT, int tables,
                       T *__restrict__ dst, const TreePlaces P, int LV) {
  __shared__ T constant[16]; // by call
  if (!P.all) {
    if (threadIdx.x < 16 && P.leaf[threadIdx.x]) constant[threadIdx.x] = squared(*P.leaf[threadIdx.x], LV);
    __syncthreads();
  }
  const size_t one = (size_t)1 << (s + rK), tot = one * tables;
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < tot; i += (size_t)gridDim.x * blockDim.x) {
    const size_t X = i & (one - 1), y = X & (((size_t)1 << rK) - 1), x = X >> rK;
    T v = F::zero();
    if (y < (size_t)NT) {
      const int q = P.at[y];
      v = q >= 0 ? ld16((i >= one ? R : L) + q * ts + x) : constant[-1 - q];
    }
    st16(dst + i, v);
  }
}
// What a Gram pass left out: part i < n is chunks from[i] .. 2^nb - 1 of tree number tree[i], a
// tree of call call[i].
struct LeftOut {
  const T *leaf[16] = {}; // as in TreePlaces
  u32 from[256] = {};
  unsigned char tree[256] = {}, call[256] = {};
  int n = 0, unused = 0; // unused: as in TreePlaces
};
// Adds to each of the Gram matrix's 64 entries, G[a][a'] = sum_(y, c) w(y, c) L_a(y, c) R_a'(y, c),
// what a full pass would have added for the left-out parts, where every entry of both tables is
// leaf^(2^LV) and the weights within a chunk add up to 1:
//   sum_i eq(rho[ay .. ay + rK), tree[i]) leaf_i^(2^(LV + 1)) E_i,
//   E_i = sum_(from[i] <= k < 2^nb) eq(rho[0 .. nb), k).
// E_i is summed bit by bit from the top: where from[i] has a 0, the k that agree with it so far and
// have a 1 there are larger. One block of 64 threads.
__global__ void k_left_out(const Chal<F> *ch, int ay, int rK, int nb, int LV, const LeftOut P, T *G) {
  __shared__ T part[64];
  const int t = threadIdx.x;
  const T one = F::one();
  T sum = F::zero();
  for (int i = t; i < P.n; i += 64) {
    T v = squared(*P.leaf[P.call[i]], LV + 1);
    for (int b = 0; b < rK; b++) {
      const T r = ch->rho[ay + b];
      v = F::mul(v, ((P.tree[i] >> (rK - 1 - b)) & 1) ? r : F::sub(one, r));
    }
    T larger = F::zero(), same = one; // eq summed over the k above, or equal to, from[i] so far
    for (int b = 0; b < nb; b++) {
      const T r = ch->rho[b];
      if ((P.from[i] >> (nb - 1 - b)) & 1) {
        same = F::mul(same, r);
      } else {
        larger = F::add(larger, F::mul(same, r));
        same = F::mul(same, F::sub(one, r));
      }
    }
    sum = F::add(sum, F::mul(v, F::add(larger, same)));
  }
  part[t] = sum;
  __syncthreads();
  sum = part[0];
  for (int i = 1; i < 64; i++) sum = F::add(sum, part[i]);
  G[t] = F::add(G[t], sum);
}
// k_up3 and k_up1 of pt_transcript.h for levels stored tree by tree; blockIdx.y is the walked tree.
__global__ void k_up3(const T *__restrict__ in, T *__restrict__ o1, T *__restrict__ o2, T *__restrict__ o3, size_t n8) {
  const size_t y = blockIdx.y;
  in += y * 8 * n8, o1 += y * 4 * n8, o2 += y * 2 * n8, o3 += y * n8;
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n8; i += (size_t)gridDim.x * blockDim.x) {
    T a[4], b[2];
#pragma unroll
    for (int k = 0; k < 4; k++) {
      a[k] = F::mul(ld16(in + i + k * n8), ld16(in + i + (k + 4) * n8));
      st16(o1 + i + k * n8, a[k]);
    }
#pragma unroll
    for (int k = 0; k < 2; k++) {
      b[k] = F::mul(a[k], a[k + 2]);
      st16(o2 + i + k * n8, b[k]);
    }
    st16(o3 + i, F::mul(b[0], b[1]));
  }
}
__global__ void k_up1(const T *__restrict__ in, T *__restrict__ o, size_t n) {
  in += blockIdx.y * 2 * n, o += blockIdx.y * n;
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x)
    st16(o + i, F::mul(ld16(in + i), ld16(in + i + n)));
}
// G[p] = the sum of parts[64 q + p] over q < n. One block of 64 threads.
__global__ void k_add64(const T *parts, int n, T *G) {
  T s = parts[threadIdx.x];
  for (int q = 1; q < n; q++) s = F::add(s, parts[64 * q + threadIdx.x]);
  G[threadIdx.x] = s;
}
// The last key layer runs its first nine rounds on the point a->rho: mt coordinates of x', the
// index within a half of a tree, then rK of y. Its other rounds run as a separate layer of
// mt - 9 + rK variables, on the point
//   b->rho = (a->rho[0 .. mt - 9) | a->rho[mt ..)).
// One block of 32 threads.
__global__ void k_rest_point(const Chal<F> *a, Chal<F> *b, int mt, int rK) {
  const int i = threadIdx.x;
  if (i < mt - 9) b->rho[i] = a->rho[i];
  if (i < rK) b->rho[mt - 9 + i] = a->rho[mt + i];
}
// nx->rho: the point that the last key layer passes on, by bit position: (zeta | the challenges of
// x' bits mt - 1 .. 9 | those of bits 8 .. 0 | those of y). got->rho is that point without bits
// 8 .. 0, as the tail left it; a->r[0 .. 9) are the challenges of the rounds that bound bits 2 1 0,
// 5 4 3, 8 7 6. One block of 32 threads.
__global__ void k_end_point(const Chal<F> *a, const Chal<F> *got, Chal<F> *nx, int mt, int rK) {
  const int i = threadIdx.x;
  if (i < 1 + mt - 9) nx->rho[i] = got->rho[i];
  if (i < 9) nx->rho[1 + mt - 9 + i] = a->r[6 - 3 * (i / 3) + i % 3];
  if (i < rK) nx->rho[1 + mt + i] = got->rho[1 + mt - 9 + i];
}
// The numbers of a ForestPlan that k_claims reads.
struct Cuts {
  int NC = 0, NR = 0, dT = 0, rK = 0, iota[16] = {}, first[16] = {}; // 0 where unused, as in Eqs
};
// Fills *c, except theta, alpha, a and rho5. One block of 32 threads.
__global__ void k_claims(const St *key, const St *table, const Chal<F> *chd, int idz, int idend, int idzeta, int idb0,
                         int idb1, int mub, const Cuts P, ForestClaims *c) {
  const int t = threadIdx.x;
  const T *z = chd[idz].rho;
  if (t < P.dT) c->zx[t] = z[t];
  if (t < P.rK) c->zy[t] = z[P.dT + t];
  if (t < mub) {
    c->zb[0][t] = chd[idb0].r[t];
    c->zb[1][t] = chd[idb1].r[t];
  }
  for (int q = 0; q < P.NC; q++) {
    const int io = P.iota[q];
    if (t < io) c->zk[q][t] = z[P.dT + P.rK - io + t];
    if (t < P.dT) c->zk[q][io + t] = z[t];
  }
  if (t < P.NC) {
    const int nb = P.rK - P.iota[t], pre = P.first[t] >> P.iota[t];
    T v = F::one();
    for (int i = 0; i < nb; i++) {
      const T zi = z[P.dT + i];
      v = F::mul(v, ((pre >> (nb - 1 - i)) & 1) ? zi : F::sub(F::one(), zi));
    }
    c->e[t] = v;
  }
  if (t == 0) {
    c->cend = key->claim[idend];
    c->zeta_end = chd[idzeta].rho[0];
    c->vb[0] = table->endv[idb0][0];
    c->vb[1] = table->endv[idb1][0];
    c->leaves_factor = table->endv[idb0][1];
    c->err = table->err;
    T empty = F::zero();
    for (int y = P.NR; y < 1 << P.rK; y++) {
      T w = F::one();
      for (int i = 0; i < P.rK; i++) w = F::mul(w, ((y >> (P.rK - 1 - i)) & 1) ? z[P.dT + i] : F::sub(F::one(), z[P.dT + i]));
      empty = F::add(empty, w);
    }
    c->empty = empty;
  }
}
} // namespace fr
// A lane: a stream with separate digit matrices and sums. Forest::spread() deals a pass's chunks to
// the lanes in turn; on one stream the GPU would be idle between the short kernels. Lane 0 is the
// calling thread's stream g_st.
struct ForestLane {
  cudaStream_t st = 0;
  grm::i8 *A = nullptr; // the digit matrices
  int *C = nullptr;     // the sums, right after the digit matrices
};
// forest_kind2_bytes() runs f() under an allocation hook and returns the bytes of kind 2 (scratch
// memory; pt_alloc in product_test.cuh defines each kind) that f requests, each buffer rounded as
// in a PtRegion. Buffers of kind 1 or 2 get no memory, so f must not access them. Those of kind 0
// or 3 share forest_counting_room (32 MiB, reused by each call), which is allocated through
// pt_alloc (kind 0) on first use and never freed. It outlives every Forest, so a program whose
// pt_alloc returns borrowed memory for kind 0 must reset the pointer to nullptr when it returns
// that memory. While f() runs, no other thread may allocate through pt_alloc.
static char *forest_counting_room = nullptr;
static size_t forest_counting_at = 0, forest_counted = 0;
template <class Fn> inline size_t forest_kind2_bytes(const Fn &f) {
  static constexpr size_t room_bytes = (size_t)32 << 20;
  if (!forest_counting_room) PAN(forest_counting_room, room_bytes, 0);
  forest_counting_at = forest_counted = 0;
  auto under = pt_alloc;
  pt_alloc = [](void **p, size_t n, int kind) {
    if (kind == 0 || kind == 3) {
      n = PtRegion::rounded(n);
      if (forest_counting_at + n > room_bytes) return cudaErrorMemoryAllocation;
      *p = forest_counting_room + forest_counting_at;
      forest_counting_at += n;
    } else {
      if (kind == 2) forest_counted += PtRegion::rounded(n);
      *p = (void *)(size_t)512;
    }
    return cudaSuccess;
  };
  f();
  CK(cudaDeviceSynchronize());
  pt_alloc = under;
  return forest_counted;
}
// Scope guard: while it exists, kind-2 buffers come from [from, from + bytes). A buffer that does
// not fit makes the program exit with code 1. Only one may exist at a time.
struct ForestWindow {
  static inline char *at = nullptr, *end = nullptr;
  static inline cudaError_t (*under)(void **, size_t, int) = nullptr;
  ForestWindow(char *from, size_t bytes) {
    at = from, end = from + bytes, under = pt_alloc;
    pt_alloc = [](void **p, size_t n, int kind) {
      if (kind != 2) return under(p, n, kind);
      const size_t m = PtRegion::rounded(n);
      if (at + m > end) {
        printf("internal error: look-up forest: the set-up asks for more scratch memory than it counted\n");
        exit(1);
      }
      *p = at, at += m;
      return cudaSuccess;
    };
  }
  ~ForestWindow() { pt_alloc = under; }
};
// Order of use: init(); then, for each proof, counts() for every call, planes(), bits() if wanted,
// and prove() or forest_prove_graph(). A Forest has no destructor: the GPU memory that init() takes
// through pt_alloc is not freed here.
struct Forest {
  ForestPlan plan;
  ForestClaims *claims = nullptr;
  std::vector<ForestCall> calls;
  std::vector<fk::Table> tabs;
  std::vector<fk::Tree> trees;
  ForestTable table;
  // Tables of at most 2^lmid entries go to the tail. reb: init()'s `rebuilt`. NL: number of lanes.
  int lmid = 16, reb = 3, NL = 2;
  int dT = 0, rK = 0, NT = 0;
  int dS = 0; // the deepest stored level: dT - reb
  int dF = 0; // the levels d <= dF are kept in the flat order, for the tails
  int dB = 0; // the smallest level that is stored tree by tree
  // The key side's chain: a sequence of layers on one transcript state (`Eng`, product_test.cuh).
  Eng ev;     // state, tail list, tail buffers. Its Gram passes are this struct's, not Eng's
  Sched solo;
  // [128], GPU memory: entry id has a layer's point rho and round challenges r (pt_transcript.h).
  Chal<F> *chd = nullptr;
  char *base = nullptr; // the one kind-2 allocation
  // measure() fills `at` with byte offsets from `base`, using a `Lay` as its cursor.
  struct Lay {
    size_t pos = 0;
    size_t take(size_t bytes) {
      const size_t at = pos;
      pos += PtRegion::rounded(bytes);
      return at;
    }
  };
  struct Places {
    size_t hooked = 0, hooked_bytes = 0, B0 = 0, B1 = 0, G = 0, k8 = 0, k8b = 0, K64 = 0, H = 0, parts = 0, X = 0,
           eqlo = 0, eqprep = 0, eqhi = 0, sums = 0, flat = 0, lanes = 0, W = 0, arena = 0, total = 0;
    size_t lane_bytes = 0, lane_digits = 0, lane_sums = 0, arena_elements = 0, eqhi_elements = 0, sums_elements = 0;
  } at;
  T *arena = nullptr, *X = nullptr, *eqlo = nullptr, *eqhi = nullptr, *sums = nullptr, *H = nullptr, *parts = nullptr,
    *k8b = nullptr, *K64 = nullptr;
  T *alpha_of_tree = nullptr; // [2^rK] entry y: tree y's alpha, filled when alpha is drawn
  M3L *eqprep = nullptr;
  u32 *W = nullptr; // the present pass's tables of weights: `table_of`, `first_table`
  // A level of the key side is stored tree by tree, walked tree q at TL[d] + q 2^d, so that the
  // kernels of forest_keys.cuh can work on one tree. The small levels are stored in the flat order
  // (see fr::k_flat). The lowest reb levels are not stored: a layer that reads one computes its
  // entries from the keys, once for the Gram sums and once for the fold (last(): twice and once).
  // Up[u], u <= rK: the levels above the trees' roots, of 2^u entries; Up[0] is the one root.
  std::vector<T *> TL, Uf; // TL[d], dB <= d <= dS: tree by tree. Uf[d], d <= dF: in the flat order
  std::vector<T *> Up;     // Up[rK] = Uf[0]
  // Padding. A padding key is a public value at a public index of its call's keys, so a node over
  // padding only is a power of that key's leaf. Where the caller declares the padding to counts(),
  // the key side walks only the trees that have another key (lists()), and the first two passes of
  // its last layer stop where both halves of a tree contain only padding (kept()). fr::k_left_out
  // and fr::k_flat supply the rest in closed form. The sums are exact in the field, so every sent
  // element is the same either way. These members are public numbers on the host. `told`:
  // forest_padded_from()'s value for each tree.
  std::vector<size_t> told, padded_from; // [NT]
  std::vector<int> walk;                 // the numbers of the walked trees, rising
  int NV = 0;                            // how many they are
  fr::Walked walked;
  fr::TreePlaces places;
  ForestLane lane[8];
  const SchedStreams *streams = nullptr;
  int turn_ = 0;
  unsigned used_ = 0;
  size_t lane_cap = (size_t)1 << 17; // the columns of a lane's digit matrices of grm::ROWS rows
  static size_t r32(size_t elements) { return (elements + 31) / 32 * 32; } // 32 elements are 768 bytes
  // Called after each phase of the key side and at the end of prove(): a launch that the driver
  // rejected leaves only an error code, so the run ends here.
  void check(const char *) { CK(cudaGetLastError()); }
  // ---- Set-up.
  void numbers(const std::vector<ForestCall> &calls_, int lidb, int lmid_, int rebuilt) {
    calls = calls_;
    plan = forest_plan(calls, lidb);
    dT = plan.dT, rK = plan.rK, NT = plan.NT;
    if (rebuilt < 2 || rebuilt > 5) {
      printf("look-up forest: rejected: 2 to 5 levels are rebuilt from the keys (%d)\n", rebuilt);
      exit(2);
    }
    reb = rebuilt;
    NL = std::min(std::max(forest_lanes, 1), 8);
    lane_cap = (size_t)1 << std::min(std::max(forest_lane_bits, 15), 20);
    // lmid >= 10 + rK: rest() runs a Gram pass only on tables of more than 2^10 entries per tree,
    // so each such pass has chunks of at least 2^8 entries within a tree.
    lmid = std::max(pt_tail_bits(lmid_), 10 + rK);
    dS = dT - reb;
    dF = std::min(lmid + 1 - rK, dS);
    dB = std::min(dF, dT - 5);
  }
  size_t level_elements() const {
    size_t e = 0;
    for (int d = dB; d <= dS; d++) e += r32((size_t)plan.NR << d); // the trees of padding alone are never walked
    return e;
  }
  // Sets up the parts that allocate kind-2 buffers themselves: measure() counts these buffers, and
  // init() serves them from a ForestWindow.
  void parts_set_up(std::vector<fk::Table> &tabs_, ForestTable &table_, Chal<F> *chd_, Sched *solo_) {
    tabs_.clear();
    for (const ForestCall &c : calls) tabs_.push_back(fk::table_of(c));
    // max(lmid, 16) has no effect: ForestTable::init() ignores it and always uses pt_tail_cap.
    table_.init(plan.of_the_table_side(), plan, calls, tabs_, std::max(lmid, 16), chd_, solo_);
  }
  void measure() {
    {
      // These three contain addresses from the counting hook. None of them frees GPU memory when
      // destroyed.
      std::vector<fk::Table> t;
      ForestTable tb;
      Sched s;
      at.hooked_bytes = forest_kind2_bytes([&] { parts_set_up(t, tb, nullptr, &s); });
      at.arena_elements = tb.room_elements();
    }
    Lay l;
    const size_t e = sizeof(T);
    at.hooked = l.take(at.hooked_bytes);
    at.B0 = l.take((e * 4) << lmid), at.B1 = l.take((e * 4) << lmid);
    at.G = l.take(e * 64), at.k8 = l.take(e * 8), at.k8b = l.take(e * 8), at.K64 = l.take(e * 64);
    at.H = l.take(e * GROUPS * FORM), at.parts = l.take(e * GROUPS * 64);
    at.X = l.take((e * 2) << lmid);
    at.eqlo = l.take(e << 15), at.eqprep = l.take(sizeof(M3L) << 16);
    // The largest need: pass 1 of last(), at most NR max(2^8, 2^(dT - 4 - MOST_CHUNK)) weights.
    // rest() and last() check that theirs fit; rebuilt() needs less and does not check.
    at.eqhi_elements = 4 * ((size_t)plan.NR << std::max(dT - 19, 8)); // weights per chunk and tree
    at.eqhi = l.take(e * at.eqhi_elements);
    // Space for 289 NR max(2^8, 2^(dT - 19)) sums in all. 289 does not bound a chunk's sums (FORM
    // does): trees_pass() checks that its sums fit in total. stored_pass() does not check its sums.
    at.sums_elements = 289 * ((size_t)plan.NR << std::max(dT - 19, 8));
    at.sums = l.take(e * at.sums_elements);
    at.flat = l.take((e * 2) << (dF + rK));
    at.lane_digits = 2 * grm::ROWS * lane_cap, at.lane_sums = (size_t)std::min(std::max(forest_lane_sums_mb, 10), 512) << 20;
    at.lane_bytes = at.lane_digits + at.lane_sums;
    at.lanes = l.take(at.lane_bytes * NL);
    // One table of weights per call, because it contains powers of the call's alpha. For chunks of
    // at most 2^13 indices, fk::weights_words() is at most 30 2^13 words, fk::second_words() and
    // ff::second_table_words() 54 2^13 (tables_fit() checks). After them comes the one table of
    // ff::k_first_table(), 24 2^13 words.
    at.W = l.take(4 * ((size_t)plan.NC * TABLE_WORDS + ((size_t)24 << MOST_CHUNK)));
    // The arena contains, at different times: the stored levels; a rebuilt layer's folded tables
    // (NR 2^(dT - 4) entries at LV 1) and the two buffers of the folds that follow, or the last key
    // layer's (NR 2^(dT - 6), then an eighth); the table side's memory; the bytes of bits(). Layers
    // run from the roots down, so a layer's fold buffers overwrite the smaller levels. The table
    // side needs the arena only after the key side: ForestTable::roots() multiplies its roots from
    // the counts, and ForestTable::prove() builds its tree.
    const size_t f1 = r32((size_t)plan.NR << (dT - 4));
    at.arena_elements = std::max(at.arena_elements, level_elements());
    at.arena_elements = std::max(at.arena_elements, f1 + r32(f1 / 8) + r32(f1 / 64) + 64);
    at.arena_elements = std::max(at.arena_elements, (((size_t)NT << (plan.kT + 5)) + e - 1) / e);
    at.arena = l.take(e * at.arena_elements);
    at.total = l.pos;
  }
  // rebuilt: the number of lowest levels of the key trees that are not stored but computed from the
  // keys (2 to 5). lidb: an upper bound on dT. lmid_: the requested lmid, which numbers() clamps.
  // Exits with code 2 if forest_plan(), fk::table_of() or ForestTable::init() rejects the group, or
  // if `rebuilt` is out of range.
  void init(const std::vector<ForestCall> &calls_, int lidb, int lmid_ = 18, int rebuilt = 3) {
    if (!g_sms) dev_info();
    g_grid = g_sms * 8;
    grm::set_up();
    numbers(calls_, lidb, lmid_, rebuilt);
    measure();
    PAN(base, at.total, 2);
    PAN(chd, sizeof(Chal<F>) * 128, 0);
    PAN(claims, sizeof(ForestClaims), 0);
    PAN(alpha_of_tree, sizeof(T) << rK, 0);
    {
      ForestWindow w(base + at.hooked, at.hooked_bytes);
      parts_set_up(tabs, table, chd, &solo);
    }
    auto el = [&](size_t off) { return reinterpret_cast<T *>(base + off); };
    ev.lmid = lmid, ev.chd = chd, ev.ll.n = 0, ev.solo = &solo, ev.sch = nullptr;
    PAN(ev.st, sizeof(St), 0);
    PA(ev.part, 16 * 8 * 4096, 0);
    CK(cudaMemset(ev.part, 0, 8 * EDEG * 8 * 4096));
    ev.B[0] = el(at.B0), ev.B[1] = el(at.B1), ev.G = el(at.G), ev.k8 = el(at.k8);
    k8b = el(at.k8b), K64 = el(at.K64), H = el(at.H), parts = el(at.parts), X = el(at.X);
    eqlo = el(at.eqlo), eqprep = reinterpret_cast<M3L *>(base + at.eqprep), eqhi = el(at.eqhi), sums = el(at.sums);
    arena = el(at.arena);
    for (int q = 0; q < NL; q++) {
      lane[q].A = reinterpret_cast<grm::i8 *>(base + at.lanes + q * at.lane_bytes);
      lane[q].C = reinterpret_cast<int *>(lane[q].A + at.lane_digits);
    }
    W = reinterpret_cast<u32 *>(base + at.W);
    Uf.assign(dF + 1, nullptr);
    {
      T *p = el(at.flat);
      for (int d = dF; d >= 0; d--) Uf[d] = p, p += (size_t)1 << (d + rK);
      // The 2^rK - 1 entries above the roots fill the rest of the region's 2^(dF + rK + 1)
      // elements, except one.
      Up.assign(rK + 1, nullptr);
      Up[rK] = Uf[0];
      for (int u = rK - 1; u >= 0; u--) Up[u] = p, p += (size_t)1 << u;
    }
    TL.assign(dS + 1, nullptr);
    {
      T *p = arena;
      for (int d = dB; d <= dS; d++) TL[d] = p, p += r32((size_t)plan.NR << d);
    }
    trees.assign(NT, fk::Tree());
    for (int y = 0; y < NT; y++) trees[y].tab = tabs[plan.call_of[y]], trees[y].call = plan.call_of[y], trees[y].alpha = alpha_of_tree + y;
    told.assign(NT, (size_t)1 << (dT - 1));
    lists();
  }
  // Derives padded_from, `walk`, NV, `walked` and `places` from `told`. If
  // forest_leaves_padding_out is false, or all NR trees are declared as padding only, the NR trees
  // are walked in full.
  void lists() {
    const size_t half = (size_t)1 << (dT - 1);
    padded_from = told;
    if (!forest_leaves_padding_out || *std::max_element(told.begin(), told.begin() + plan.NR) == 0) padded_from.assign(NT, half);
    for (int y = plan.NR; y < NT; y++) padded_from[y] = 0; // trees of padding alone: they have no keys in memory
    walk.clear();
    places = fr::TreePlaces();
    walked = fr::Walked();
    for (int y = 0; y < NT; y++) {
      const int c = plan.call_of[y];
      if (padded_from[y]) {
        places.at[y] = (short)walk.size();
        walk.push_back(y);
      } else {
        places.at[y] = (short)(-1 - c), places.all = 0;
        places.leaf[c] = tabs[c].lut + calls[c].t0;
      }
    }
    NV = (int)walk.size();
    walked.n = NV;
    for (int q = 0; q < NV; q++) walked.tree[q] = (unsigned char)walk[q];
  }
  // Returns how many chunks a pass of the last key layer walks in walked tree q. A half of a tree
  // has 2^nb chunks of 2^span keys; from index padded_from on, both halves contain only padding.
  int kept(int q, int span, int nb) const {
    return (int)std::min<size_t>((size_t)1 << nb, (padded_from[walk[q]] + (((size_t)1 << span) - 1)) >> span);
  }
  // Adds to ev.G, on g_st, what the preceding pass left out. cd: the layer's point; ay: where its
  // coordinates of y begin. LV: as in fr::k_flat. `span` and nb: as in kept(); span < 0 if that
  // pass walked the walked trees in full.
  void left_out(const Chal<F> *cd, int ay, int LV, int span = -1, int nb = 0) {
    fr::LeftOut P;
    for (int c = 0; c < plan.NC; c++) P.leaf[c] = tabs[c].lut + calls[c].t0;
    auto part = [&](int y, u32 from) {
      P.tree[P.n] = (unsigned char)y, P.call[P.n] = (unsigned char)plan.call_of[y], P.from[P.n] = from;
      P.n++;
    };
    for (int y = 0; y < NT; y++)
      if (places.at[y] < 0)
        part(y, 0);
      else if (span >= 0 && kept(places.at[y], span, nb) < 1 << nb)
        part(y, (u32)kept(places.at[y], span, nb));
    if (!P.n) return;
    pt_hi(fr::k_left_out, 1, 64, 0, g_st, cd, ay, rK, span < 0 ? 0 : nb, LV, P, ev.G);
  }
  // ---- Before alpha.
  // keys: GPU memory, call c's 2^n_c row numbers; its tree b is keys[b 2^dT .. (b + 1) 2^dT).
  // `keys` must be 16-byte aligned: the kernels read it as uint4. No call's keys may change until
  // prove()'s GPU work is done. pub: GPU memory, the number of padding keys in each of the call's
  // 2^iota_c trees, or nullptr (none). padded: nullptr, or host memory, forest_padded_from()'s
  // value for each of these trees: the key side may then skip the keys that these values declare as
  // padding and assume the padding key there; the table side reads all keys regardless. `padded`
  // must come from the caller's plan alone, never from a request's data: the launches depend on it.
  void counts(int c, const uint16_t *keys, const u32 *pub, const size_t *padded = nullptr) {
    for (int b = 0; b < (1 << plan.iota[c]); b++) {
      trees[plan.first[c] + b].keys = keys + ((size_t)b << dT);
      told[plan.first[c] + b] = padded ? padded[b] : (size_t)1 << (dT - 1);
    }
    table.counts(c, keys, pub);
  }
  void planes() { table.planes(); }
  // The multiplicity bits, one byte each: bit j of the multiplicity of row t of call c < 2^cK.
  // order 0: at (c 2^kT + t) 32 + j, so the NC calls' plan.plane_bits() bytes come first. order 1:
  // at (t 2^cK + c) 32 + j, the table side's index. Call planes() first: it zeroes the counts for
  // c >= NC. The bytes are in `arena`, valid until prove().
  const u8 *bits(int order = 0) {
    table.bits(order, reinterpret_cast<u8 *>(arena));
    return reinterpret_cast<const u8 *>(arena);
  }
  const u32 *pub_dev() const { return table.pub_dev(); }
  // One sumcheck (see layers()). Its addresses point into chd. They are valid from init() on;
  // prove() writes the values.
  struct Layer {
    int mu;                         // its number of rounds
    std::vector<const T *> r_round; // [mu] the challenge of each round, in the order of sending
    const T *zeta;                  // the challenge after (vL, vR); nullptr for leaves and bits
    size_t slot;                    // its first message slot: index into `prove`'s msg
  };
  // The index in chd of table layer l < plan.d(); the leaves have table_id(plan.d()), the bits 80.
  int table_id(int l) const { return 40 - plan.cK + l; }
  // In the order of the proof: plan.v() key layers, plan.d() table layers, the leaves, the bits.
  std::vector<Layer> layers() const {
    std::vector<Layer> all;
    auto plain = [&](int id, int mu, const T *zeta, size_t slot) {
      Layer l{mu, {}, zeta, slot};
      for (int v = 0; v < mu; v++) l.r_round.push_back(chd[id].r + v);
      all.push_back(l);
    };
    for (int mu = 0; mu + 1 < plan.v(); mu++) plain(mu, mu, chd[mu + 1].rho, plan.at_key_layer(mu));
    {
      // The last key layer: its first nine rounds (see last()) use chd[mu], the others chd[100].
      const int mu = plan.v() - 1;
      Layer l{mu, {}, chd[mu + 1].rho, plan.at_key_layer(mu)};
      for (int v = 0; v < 9; v++) l.r_round.push_back(chd[mu].r + v);
      for (int v = 0; v < mu - 9; v++) l.r_round.push_back(chd[100].r + v);
      all.push_back(l);
    }
    for (int l = 0; l < plan.d(); l++) plain(table_id(l), 5 + l, chd[table_id(l) + 1].rho, plan.at_table_layer(l));
    plain(table_id(plan.d()), 5 + plan.d(), nullptr, plan.at_leaves());
    plain(80, 5 + plan.d(), nullptr, plan.at_bits());
    return all;
  }
  // ---- Lanes.
  // open() orders the lanes' work after the work already queued on g_st. turn() returns the next
  // lane. close() makes g_st wait for every lane that turn() has returned since open().
  // The idea is public: concurrent kernels on several streams: M. Harris, "GPU Pro Tip: CUDA 7
  // Streams Simplify Concurrency", cited at PT_FREE_BLOCKS in prover/hiding/product_test.cuh.
  // One event serves every open(): cudaStreamWaitEvent() waits for the cudaEventRecord() of that
  // event that came last before it (NVIDIA, CUDA 13.0, cuda_runtime_api.h, the comment at
  // cudaEventRecordWithFlags()).
  void open() {
    lane[0].st = g_st;
    for (int q = 1; q < NL; q++) lane[q].st = streams->cs[q];
    turn_ = 0, used_ = 0;
    if (NL > 1) CK(cudaEventRecord(streams->cm, g_st));
  }
  ForestLane &turn() {
    const int q = turn_++ % NL;
    if (q && !(used_ >> q & 1)) CK(cudaStreamWaitEvent(lane[q].st, streams->cm, 0));
    used_ |= 1u << q;
    return lane[q];
  }
  void close() {
    for (int q = 1; q < NL; q++)
      if (used_ >> q & 1) {
        CK(cudaEventRecord(streams->ce[q], lane[q].st));
        CK(cudaStreamWaitEvent(g_st, streams->ce[q], 0));
      }
    used_ = 0;
  }
  // One pass over SC indices (a multiple of 2^lc) in chunks of 2^lc, with digit matrices of `wide`
  // and `narrow` rows. Call between open() and close(). The chunks go to the lanes in turn, in jobs
  // of at most ceil(nchunks / parts) consecutive chunks. digits(lane, c0, n, A, B, ld) writes the
  // digits of n <= ld indices from c0 on. join(lane, C, pieces, nc, chunk0) combines the sums of nc
  // chunks from chunk0 on, once per job: a join of few chunks costs nearly as much as one of many.
  // A piece is the k indices that one 32-bit sum covers.
  // both(lane, c0, n, k, C) replaces digits() and the product by one kernel, for n indices in
  // pieces of k, if forest_fused and `one_kernel` are set.
  template <class D, class B2, class J>
  void spread(int wide, int narrow, size_t SC, int lc, int parts, const D &digits, bool one_kernel, const B2 &both, const J &join) {
    const size_t chunk = (size_t)1 << lc;
    if (forest_fused && one_kernel) return spread_fused(wide, narrow, SC, lc, parts, both, join);
    const int nchunks = (int)(SC >> lc), k = grm::Slab::piece(chunk, SC), pieces = (int)(chunk / k);
    const size_t ld = at.lane_digits / (wide + narrow) / k * k, hold = at.lane_sums / ((size_t)wide * narrow * sizeof(int)) / pieces;
    // grm::product() needs A, B and C at multiples of 256 bytes.
    if (!ld || !hold || !nchunks || (size_t)wide * ld % 256 || (size_t)wide * narrow * sizeof(int) % 256) {
      printf("internal error: look-up forest: a lane's memory does not serve a pass of %d and %d rows\n", wide, narrow);
      fflush(stdout);
      name_the_end("forest-lane-room"); // serving: p258 end 1 forest-lane-room (kept as an exit(1))
      exit(1);
    }
    // With many sums per piece a join's cost is per piece, and a job has at most ld indices (or one
    // chunk), so that its sums stay in cache.
    const size_t most = narrow >= 128 ? std::max<size_t>(1, ld / chunk) : hold;
    const int per = (int)std::max<size_t>(1, std::min(std::min(hold, most), (size_t)(nchunks + parts - 1) / parts));
    for (int c0 = 0; c0 < nchunks; c0 += per) {
      ForestLane &ln = turn();
      const int nc = std::min(per, nchunks - c0);
      const size_t n = nc * chunk;
      fr::i8 *A = ln.A, *B = A + (size_t)wide * ld;
      for (size_t x = 0; x < n; x += ld) {
        const int some = (int)std::min(ld, n - x);
        digits(ln, c0 * chunk + x, some, A, B, ld);
        grm::product(A, B, ln.C + x / k * wide * narrow, wide, narrow, k, some / k, ld, k, ln.st);
      }
      join(ln, ln.C, pieces, nc, c0);
    }
  }
  // spread() where one kernel makes and multiplies the digits. A piece writes only its sums, so it
  // is a whole chunk, halved (down to 512 indices) while there are fewer than 4 per SM.
  // Where ff::split(narrow) is 2, the kernel stores the sums of k indices as two pieces.
  template <class B2, class J>
  void spread_fused(int wide, int narrow, size_t SC, int lc, int parts, const B2 &both, const J &join) {
    const size_t chunk = (size_t)1 << lc;
    const int nchunks = (int)(SC >> lc);
    size_t k = chunk;
    while (SC / k < (size_t)4 * g_sms && k > 512) k /= 2;
    const int pieces = (int)(chunk / k) * ff::split(narrow);
    const size_t hold = at.lane_bytes / ((size_t)wide * narrow * sizeof(int)) / pieces;
    if (!hold || !nchunks || k % ff::COLS) {
      printf("internal error: look-up forest: a pass of %d and %d rows: %d chunks, pieces of %zu indices, room for the sums of %zu chunks in a lane's memory\n",
             wide, narrow, nchunks, k, hold);
      fflush(stdout);
      name_the_end("forest-lane-room"); // serving: p258 end 1 forest-lane-room (same lane-memory reason as spread())
      exit(1);
    }
    const int per = (int)std::max<size_t>(1, std::min(hold, (size_t)(nchunks + parts - 1) / parts));
    for (int c0 = 0; c0 < nchunks; c0 += per) {
      ForestLane &ln = turn();
      const int nc = std::min(per, nchunks - c0);
      int *C = reinterpret_cast<int *>(ln.A); // all of the lane's memory: no digit matrices here
      both(ln, c0 * chunk, nc * chunk, (int)k, C);
      join(ln, C, pieces, nc, c0);
    }
  }
  // One Gram pass over tables stored tree by tree (see rest()), into ev.G: 2^lsc indices per walked
  // tree, in chunks of 2^x. lo, hi: the weights within a chunk and per chunk.
  void stored_pass(const T *L, const T *R, size_t ts, int lsc, int x, const M3L *lo, const T *hi) {
    const size_t total = (size_t)NV << lsc;
    if (forest_fused) {
      const int blocks = ff::stored(ff::StoredJob{L, R, ts, lsc, x, lo, hi}, total, sums, g_st);
      pt_hi(grm::k_sum, 64, 128, 0, g_st, (const T *)sums, blocks, ev.G, (const Chal<F> *)nullptr, 0, 0, 0);
      return;
    }
    auto join = [&](ForestLane &ln, const int *C, int pieces, int nc, int c0) {
      grm::join(C, pieces, nc, hi + c0, sums + (size_t)c0 * 64, ln.st);
    };
    open();
    spread(grm::ROWS, grm::ROWS, total, x, 2 * NL, [&](ForestLane &ln, size_t g0, int n, fr::i8 *A, fr::i8 *B, size_t ld) {
      fr::k_digits<<<std::min((n + 127) / 128, g_sms * 8), 256, 0, ln.st>>>(L, R, ts, lsc, g0, n, 1 << x, lo, A, B, ld);
      CK(cudaGetLastError());
    }, false, [](ForestLane &, size_t, size_t, int, int *) {}, join);
    close();
    pt_hi(grm::k_sum, 64, 128, 0, g_st, (const T *)sums, (int)(total >> x), ev.G, (const Chal<F> *)nullptr, 0, 0, 0);
  }
  // One Gram pass computed from the keys, into ev.G. A tree has 2^nh chunks of 2^lc indices; only
  // the first chunks_of(q) >= 1 of walked tree q are processed. eqhi has a weight for each walked
  // tree and chunk. shape_of(table) returns the rows of a tree's two digit matrices and the number
  // of sums per chunk (`form` in identifiers). digits(), both() and join() are those of spread(),
  // for one tree. finish(sums, call, H, G, stream) computes a Gram matrix's 64 entries from the
  // totals H of one form over the chunks and trees of one call, whose trees share their alpha.
  // `call` is -1 for the form of ff::first_range(), whose totals contain the alphas.
  static constexpr int FORM = 576, GROUPS = 32; // upper limits: sums per form; groups per pass
  template <class Ch, class Sh, class D, class B2, class J, class Fi>
  void trees_pass(int lc, int nh, const Ch &chunks_of, const Sh &shape_of, const D &digits, bool one_kernel, const B2 &both,
                  const J &join, const Fi &finish) {
    // The trees' sums are stored in `sums`, grouped by form and call (`group` in identifiers).
    std::vector<int> forms, call; // a group's form and call
    std::vector<int> group(NV);
    std::vector<fk::Shape> shape(NV);
    size_t all = 0;
    for (int q = 0; q < NV; q++) {
      shape[q] = shape_of(trees[walk[q]].tab);
      // A form without digit matrices (`wide` 0) is that of ff::first_range(): one launch for all
      // its trees, which writes one set of sums per thread block (at most g_sms).
      const int c = shape[q].wide ? trees[walk[q]].call : -1;
      size_t f = 0;
      while (f < forms.size() && (forms[f] != shape[q].sums || call[f] != c)) f++;
      if (f == forms.size()) {
        forms.push_back(shape[q].sums), call.push_back(c);
        if (!shape[q].wide) all += (size_t)shape[q].sums * g_sms;
      }
      group[q] = (int)f;
      all += shape[q].wide ? (size_t)shape[q].sums * chunks_of(q) : 0;
    }
    if (forms.size() > GROUPS || *std::max_element(forms.begin(), forms.end()) > FORM || all > at.sums_elements) {
      printf("internal error: look-up forest: a pass's sums are of more forms, longer or more than the engine has room for\n");
      fflush(stdout);
      name_the_end("forest-pass-sums"); // serving: p258 end 1 forest-pass-sums (kept as an exit(1))
      exit(1);
    }
    std::vector<T *> from(forms.size());
    std::vector<int> chunks(forms.size(), 0);
    const int parts_ = std::max(1, (2 * NL + NV - 1) / NV);
    open();
    {
      T *p = sums;
      for (size_t f = 0; f < forms.size(); f++) {
        from[f] = p;
        if (call[f] < 0) { // the form without digit matrices
          ff::FirstTrees list;
          list.n = 0, list.first[0] = 0;
          for (int q = 0; q < NV; q++)
            if (group[q] == (int)f) {
              list.keys[list.n] = trees[walk[q]].keys, list.place[list.n] = (unsigned char)q, list.alpha[list.n] = trees[walk[q]].alpha;
              list.first[list.n + 1] = list.first[list.n] + chunks_of(q);
              list.n++;
            }
          for (int q = list.n; q < 256; q++) // no array entry unwritten
            list.keys[q] = nullptr, list.place[q] = 0, list.alpha[q] = nullptr, list.first[q + 1] = 0;
          chunks[f] = ff::first_range(list, dT, lc, nh, first_table(), eqhi, p, turn().st);
          p += (size_t)forms[f] * chunks[f];
          continue;
        }
        for (int q = 0; q < NV; q++) {
          if (group[q] != (int)f) continue;
          const fk::Tree &t = trees[walk[q]];
          const T *hi = eqhi + ((size_t)q << nh);
          T *out = p;
          spread(shape[q].wide, shape[q].narrow, (size_t)chunks_of(q) << lc, lc, parts_,
                 [&](ForestLane &ln, size_t c0, int n, fr::i8 *A, fr::i8 *B, size_t ld) { digits(t, c0, n, A, B, ld, ln.st); },
                 one_kernel, [&](ForestLane &ln, size_t c0, size_t n, int k, int *C) { both(t, c0, n, k, C, ln.st); },
                 [&](ForestLane &ln, const int *C, int pieces, int nc, int c0) {
                   join(t.tab, C, pieces, nc, hi + c0, out + (size_t)c0 * forms[f], ln.st);
                 });
          p += (size_t)forms[f] * chunks_of(q), chunks[f] += chunks_of(q);
        }
      }
    }
    close();
    for (size_t f = 0; f < forms.size(); f++) {
      t3k::addchunks<<<forms[f], t3k::ADD_THREADS, 0, g_st>>>(from[f], H + FORM * f, chunks[f], forms[f]);
      CK(cudaGetLastError());
      finish(forms[f], call[f], H + FORM * f, forms.size() == 1 ? ev.G : parts + 64 * f, g_st);
    }
    if (forms.size() > 1) fr::k_add64<<<1, 64, 0, g_st>>>(parts, (int)forms.size(), ev.G);
    CK(cudaGetLastError());
  }
  static constexpr int MOST_CHUNK = 13; // log2 of the largest chunk in a pass computed from keys
  static constexpr size_t TABLE_WORDS = (size_t)54 << MOST_CHUNK;
  u32 *table_of(int c) const { return W + (size_t)c * TABLE_WORDS; }
  u32 *first_table() const { return W + (size_t)plan.NC * TABLE_WORDS; }
  const T *alpha_of_call(int c) const { return alpha_of_tree + plan.first[c]; }
  // Calls fn(c) for each call that has a walked tree.
  template <class Fn> void walked_calls(const Fn &fn) const {
    std::vector<bool> seen(plan.NC, false);
    for (int q = 0; q < NV; q++)
      if (!seen[trees[walk[q]].call]) seen[trees[walk[q]].call] = true, fn(trees[walk[q]].call);
  }
  // Exits with code 1 if a pass's tables of weights for chunks of 2^lc indices, `words` words per
  // call, do not fit their parts of W (see measure()).
  void tables_fit(int lc, size_t words) const {
    if (lc >= 0 && lc <= MOST_CHUNK && words <= TABLE_WORDS && ff::first_table_words(lc) <= (size_t)24 << MOST_CHUNK) return;
    printf("internal error: look-up forest: a table of weights for chunks of 2^%d indices has %zu words, more than its room\n", lc, words);
    fflush(stdout);
    name_the_end("forest-table-room"); // serving: p258 end 1 forest-table-room (kept as an exit(1))
    exit(1);
  }
  // Computes from the keys the Gram matrix of a layer's first three rounds, into ev.G. LV >= 1: the
  // layer that reads level dT - LV, in top-bit-first order. LV = 0: pass 1 of the last key layer,
  // where an index covers eight keys of each half. lc, nh: as in trees_pass(). The caller has
  // filled eqlo and eqprep (weights within a chunk) and eqhi (per chunk and tree).
  void keys_pass(int LV, int lc, int nh) {
    if (LV <= 2) tables_fit(lc, fk::weights_words(LV, 1 << lc)); // LV > 2 has no table of weights
    const bool products = forest_fused && LV == 0; // range tables (row t = t) use `ff::first_range`
    walked_calls([&](int c) {
      if (!(products && tabs[c].kind == fk::RANGE)) fk::weights(LV, alpha_of_call(c), 1 << lc, eqlo, table_of(c), g_st);
    });
    if (products) {
      ff::k_first_table<<<((1 << lc) + 127) / 128, 128, 0, g_st>>>(eqlo, 1 << lc, first_table());
      CK(cudaGetLastError());
    }
    trees_pass(
        lc, nh, [&](int q) { return LV ? 1 << nh : kept(q, lc + 3, nh); },
        [&](const fk::Table &tab) {
          return products && tab.kind == fk::RANGE ? fk::Shape{0, 0, ff::FIRST_SUMS} : fk::shape(LV, tab);
        },
        [&](const fk::Tree &t, size_t c0, int n, fr::i8 *A, fr::i8 *B, size_t ld, cudaStream_t s) {
          fk::digits(LV, t, dT, t.alpha, c0, n, 1 << lc, eqprep, table_of(t.call), A, B, ld, s);
        },
        LV <= 2, [&](const fk::Tree &t, size_t c0, size_t n, int k, int *C, cudaStream_t s) {
          ff::keys_pass(LV, t, dT, t.alpha, c0, n, 1 << lc, table_of(t.call), k, C, s);
        },
        [&](const fk::Table &tab, const int *C, int pieces, int nc, const T *hi, T *out, cudaStream_t s) {
          fk::join(LV, tab, C, pieces, nc, hi, out, s);
        },
        [&](int sums_, int c, const T *totals, T *G, cudaStream_t s) {
          if (c >= 0) return fk::finish(sums_, totals, alpha_of_call(c), G, s);
          ff::k_first_finish<<<1, 64, 0, s>>>(totals, G);
          CK(cudaGetLastError());
        });
  }
  // Fills J's eq tables from the point cd->rho, on g_st.
  void eqs(const Chal<F> *cd, const fr::Eqs &J) {
    if (J.n) fr::k_eqs<<<dim3(128, J.n), 256, 0, g_st>>>(cd, J, rK, walked);
    CK(cudaGetLastError());
  }
  // Test hook: marks [p, p + elements) as free, so that a test can overwrite it.
  void dead(T *p, size_t elements) { PT_FREE_MEMORY(p, elements * sizeof(T), false); }
  void lanes_dead() {
    for (int q = 0; q < NL; q++) PT_FREE_MEMORY(lane[q].A, at.lane_bytes, true);
  }
  // Runs three sumcheck rounds on ev.G (k_g2msgT): writes the challenges to cd->r[j .. j + 2] and
  // their eq values to k8[0 .. 7]. iz2, iz3: where cd->rho has the second and third bound variable.
  // The layer's sumcheck must have begun (k_layer_begin). o is not read: the messages go to that
  // sumcheck's next message slots.
  void three_rounds(Chal<F> *cd, int j, int iz2, int iz3, T *o, T *k8) {
    pt_hi(k_g2msgT, 1, 64, 0, g_st, (const T *)ev.G, ev.st, cd, j, iz2, iz3, o, k8);
  }
  // ---- The rounds of a layer from stored tables.
  // The tables are stored tree by tree: 2^s entries per tree, those of walked tree q at L + q ts
  // and R + q ts. id, idn: the indices in chd of this layer and of the next layer's point. LV: as
  // in fr::k_flat. Sa and Sb: free memory for the folded tables, used alternately:
  // 2 NV 2^(s - 3) entries at Sa, an eighth of that at Sb. The layer has mu
  // variables, of which the first j0 rounds have bound the top bits of x, so mu - j0 = s + rK. Runs Gram passes until the tables
  // have at most 2^lmid entries, then passes them to the tail in the flat order. The layer's
  // sumcheck begins here, with o as its first message slot, unless earlier rounds have bound
  // variables or `continues` is set (it has begun, and mu counts only the variables left).
  void rest(const T *L, const T *R, size_t ts, int s, int mu, int j0, int id, int idn, T *o, T *Sa, T *Sb, int LV,
            bool continues = false) {
    Chal<F> *cd = chd + id;
    bool begun = j0 > 0 || continues;
    const int mt = mu - rK; // rho[0 .. mt) belongs to x, rho[mt .. mu) to y
    int j = j0;
    const M3L *lo[12];
    const T *hi[12];
    {
      fr::Eqs J;
      int np = 0, xcur = -1;
      size_t olo = 0, ohi = 0;
      const M3L *cl = nullptr;
      for (int s2 = s, j2 = j; s2 + rK > lmid; s2 -= 3, j2 += 3, np++) {
        const int x = std::min(s2 - 3, 15), nh = s2 - 3 - x;
        if (x != xcur) {
          J.add(mt - x, x, -1, nullptr, eqprep + olo);
          cl = eqprep + olo, olo += (size_t)1 << x, xcur = x;
        }
        lo[np] = cl, hi[np] = eqhi + ohi;
        J.add(j2 + 3, nh, mt, eqhi + ohi, nullptr);
        ohi += (size_t)NV << nh;
      }
      if (ohi > at.eqhi_elements) {
        printf("internal error: look-up forest: the table of weights per chunk is too small\n");
        name_the_end("forest-weights-room"); // serving: p258 end 1 forest-weights-room (kept as an exit(1))
        exit(1);
      }
      eqs(cd, J);
    }
    for (int p = 0; s + rK > lmid; p++) {
      const int lsc = s - 3, x = std::min(lsc, 15);
      const size_t total = (size_t)NV << lsc;
      lanes_dead();
      stored_pass(L, R, ts, lsc, x, lo[p], hi[p]);
      left_out(cd, mt, LV);
      if (!begun) pt_hi(k_layer_begin, 1, 32, 0, g_st, ev.st, (const Chal<F> *)cd, mu, id, 0, o, 0);
      begun = true;
      three_rounds(cd, j, j + 1, j + 2, o, ev.k8);
      const size_t wanted = (total / 2 + kg3::FOLD8_THREADS - 1) / kg3::FOLD8_THREADS;
      fr::k_fold8<<<dim3((unsigned)std::min<size_t>(wanted, (size_t)g_sms * 4), 2), kg3::FOLD8_THREADS, 0, g_st>>>(
          L, R, ts, lsc, total, ev.k8, Sa, Sa + total);
      CK(cudaGetLastError());
      L = Sa, R = Sa + total, ts = (size_t)1 << lsc;
      std::swap(Sa, Sb);
      o += 6, j += 3, s -= 3;
    }
    fr::k_flat<<<(int)std::min<size_t>((((size_t)2 << (s + rK)) + 255) / 256, 1056), 256, 0, g_st>>>(L, R, ts, s, rK, NT, 2,
                                                                                                   X, places, LV);
    CK(cudaGetLastError());
    tail(X, mu, j, id, idn, o, !begun);
    ev.mid();
  }
  // Appends a layer's rounds after the first j to the tail list `ev.ll`, which ev.mid() launches.
  // begins: they start its sumcheck, whose first message slot is o.
  void tail(const T *src, int mu, int j, int id, int idn, T *o, bool begins) {
    const int i = ev.ll.n++;
    ev.ll.src[i] = src, ev.ll.out[i] = o, ev.ll.mu[i] = mu, ev.ll.j0[i] = j, ev.ll.id[i] = id, ev.ll.idn[i] = idn;
    ev.ll.begins[i] = begins, ev.ll.kind[i] = TAIL_PRODUCT;
    if (ev.ll.n == 24) ev.mid();
  }
  // ---- The key side.
  // Builds every stored level from the keys, then the trees' roots and the levels above them, up to
  // the one root of all keys, Up[0].
  void build() {
    dead(arena, at.arena_elements);
    open();
    for (int q = 0; q < NV; q++) {
      std::vector<T *> lev(dT + 1, nullptr);
      for (int d = std::max(dT - 5, dB); d <= dS; d++) lev[d] = TL[d] + ((size_t)q << d);
      fk::build(trees[walk[q]], dT, reb, trees[walk[q]].alpha, lev.data(), turn().st);
    }
    close();
    int d = dT - 5;
    for (; d - 3 >= dF; d -= 3) {
      const size_t n8 = (size_t)1 << (d - 3);
      fr::k_up3<<<dim3((unsigned)std::min<size_t>((n8 + 255) / 256, (size_t)g_grid), NV), 256, 0, g_st>>>(
          TL[d], TL[d - 1], TL[d - 2], TL[d - 3], n8);
    }
    for (; d > dF; d--) {
      const size_t n = (size_t)1 << (d - 1);
      fr::k_up1<<<dim3((unsigned)std::min<size_t>((n + 255) / 256, (size_t)g_grid), NV), 256, 0, g_st>>>(TL[d], TL[d - 1], n);
    }
    fr::k_flat<<<(int)std::min<size_t>((((size_t)1 << (dF + rK)) + 255) / 256, 1056), 256, 0, g_st>>>(
        TL[dF], TL[dF], (size_t)1 << dF, dF, rK, NT, 1, Uf[dF], places, dT - dF);
    for (d = dF; d >= 1; d--) {
      const size_t n = (size_t)1 << (d - 1 + rK);
      k_up1<F><<<(int)std::min<size_t>((n + 255) / 256, g_grid), 256, 0, g_st>>>(Uf[d], Uf[d - 1], n);
    }
    // Above the trees' roots the index is the tree's number. As in every level, entry i is the
    // product of entries i and i + n of the level below, where n is the upper level's length.
    for (int u = rK; u >= 1; u--) k_up1<F><<<1, 256, 0, g_st>>>(Up[u], Up[u - 1], (size_t)1 << (u - 1));
    check("key side: building the levels");
  }
  // Runs the layer that reads the stored level d.
  void stored(int d) {
    const int mu = rK + d - 1;
    T *o = ev.take(mu);
    if (d <= dF) {
      tail(Uf[d], mu, 0, mu, mu + 1, o, true);
      return;
    }
    ev.flush();
    const size_t folded = r32((size_t)plan.NR << (d - 3));
    dead(arena, TL[d] - arena);
    rest(TL[d], TL[d] + ((size_t)1 << (d - 1)), (size_t)1 << d, d - 1, mu, 0, mu, mu + 1, o, arena, arena + folded, dT - d);
  }
  // Runs the layer whose level dT - LV, LV >= 1, is rebuilt from the keys, in top-bit-first order.
  // The idea is public, computing again what was not stored: Chen et al., arXiv:1604.06174, cited
  // with its title at gram_slab_in() in prover/hiding/gram_engine.inc.
  void rebuilt(int LV) {
    ev.flush();
    const int d = dT - LV, mt = d - 1, mu = mt + rK, nb = mt - 3, lc = std::min(nb, MOST_CHUNK), nh = nb - lc;
    const size_t SC = (size_t)1 << nb;
    Chal<F> *cd = chd + mu;
    T *o = ev.take(mu);
    dead(arena, at.arena_elements);
    lanes_dead();
    {
      fr::Eqs J;
      J.add(mt - lc, lc, -1, eqlo, eqprep);
      J.add(3, nh, mt, eqhi, nullptr);
      eqs(cd, J);
    }
    keys_pass(LV, lc, nh);
    left_out(cd, mt, LV);
    pt_hi(k_layer_begin, 1, 32, 0, g_st, ev.st, (const Chal<F> *)cd, mu, mu, 0, o, 0);
    three_rounds(cd, 0, 1, 2, o, ev.k8);
    T *Lo = arena, *Ro = arena + NV * SC, *Sa = arena + r32(2 * NV * SC), *Sb = Sa + r32(2 * NV * SC / 8);
    open();
    for (int q = 0; q < NV; q++) {
      const fk::Tree &t = trees[walk[q]];
      if (forest_fused && LV <= 2)
        ff::fold(LV, t, dT, t.alpha, ev.k8, Lo + q * SC, Ro + q * SC, turn().st);
      else
        fk::fold(LV, t, dT, t.alpha, ev.k8, Lo + q * SC, Ro + q * SC, turn().st);
    }
    close();
    rest(Lo, Ro, SC, nb, mu, 3, mu, mu + 1, o + 6, Sa, Sb, LV);
    check("key side: a layer rebuilt from the keys");
  }
  static int clampi(int x, int a, int b) { return x < a ? a : x > b ? b : x; }
  // Runs the last key layer from the keys, in order 1 of k_layer_begin() (product_test.cuh): three
  // passes bind bits 2 1 0, 5 4 3 and 8 7 6 of x' (see fr::k_rest_point); then come its other bits,
  // top bit first, then those of y.
  void last() {
    ev.flush();
    const int mt = dT - 1, mu = mt + rK;
    const size_t N = (size_t)1 << dT, CO = N / 1024;
    Chal<F> *cd = chd + mu;
    T *o = ev.take(mu);
    dead(arena, at.arena_elements);
    const int lc1 = clampi(dT - 12, 10, MOST_CHUNK), lc2 = clampi(dT - 12, 7, MOST_CHUNK), lc3 = clampi(dT - 13, 7, 15);
    // Fills the eq weights of a pass over nbits-bit indices: eqlo, eqprep (low lc bits) and eqhi.
    auto weights = [&](int nbits, int lc) {
      if (((size_t)NV << (nbits - lc)) > at.eqhi_elements) {
        printf("internal error: look-up forest: the last layer's passes want more weights per chunk than the set-up made room for\n");
        fflush(stdout);
        name_the_end("forest-weights-room"); // serving: p258 end 1 forest-weights-room (same per-chunk weights room as rest())
        exit(1);
      }
      fr::Eqs J;
      J.add(nbits - lc, lc, -1, eqlo, eqprep);
      J.add(0, nbits - lc, mt, eqhi, nullptr);
      eqs(cd, J);
    };
    // Pass 1.
    weights(mt - 3, lc1);
    lanes_dead();
    keys_pass(0, lc1, mt - 3 - lc1);
    left_out(cd, mt, 0, lc1 + 3, mt - 3 - lc1);
    pt_hi(k_layer_begin, 1, 32, 0, g_st, ev.st, (const Chal<F> *)cd, mu, mu, 1, o, rK);
    three_rounds(cd, 0, mt - 2, mt - 1, o, ev.k8);
    // Pass 2: an index covers 64 keys of each half. Its tables of weights depend on the challenges
    // of rounds 1 to 3, through ev.k8, and overwrite those of pass 1, which are no longer needed.
    weights(mt - 6, lc2);
    lanes_dead();
    tables_fit(lc2, std::max(fk::second_words(1 << lc2), ff::second_table_words(1 << lc2)));
    walked_calls([&](int c) {
      if (forest_fused && tabs[c].kind == fk::RANGE) // range tables use `ff::second_range`
        ff::k_second_table<<<((1 << lc2) + 127) / 128, 128, 0, g_st>>>(alpha_of_call(c), ev.k8, 1 << lc2, eqlo, table_of(c));
      else
        fk::second_weights(alpha_of_call(c), ev.k8, 1 << lc2, eqlo, table_of(c), g_st);
      CK(cudaGetLastError());
    });
    trees_pass(
        lc2, mt - 6 - lc2, [&](int q) { return kept(q, lc2 + 6, mt - 6 - lc2); },
        [&](const fk::Table &tab) {
          return forest_fused && tab.kind == fk::RANGE ? fk::Shape{ff::second::ROWS_A, ff::second::ROWS_B, ff::second::SUMS}
                                                       : fk::shape_second(tab);
        },
        [&](const fk::Tree &t, size_t c0, int n, fr::i8 *A, fr::i8 *B, size_t ld, cudaStream_t s) {
          fk::second_digits(t, dT, t.alpha, ev.k8, c0, n, 1 << lc2, table_of(t.call), A, B, ld, s);
        },
        true, [&](const fk::Tree &t, size_t c0, size_t n, int k, int *C, cudaStream_t s) {
          if (t.tab.kind == fk::RANGE) return ff::second_range(t, dT, 1 << lc2, table_of(t.call), c0, n, k, C, s);
          ff::second_pass(t, dT, t.alpha, ev.k8, c0, n, 1 << lc2, table_of(t.call), k, C, s);
        },
        [&](const fk::Table &tab, const int *C, int pieces, int nc, const T *hi, T *out, cudaStream_t s) {
          if (forest_fused && tab.kind == fk::RANGE) return ff::join_second(C, pieces, nc, hi, out, s);
          fk::second_join(tab, C, pieces, nc, hi, out, s);
        },
        [&](int sums_, int c, const T *totals, T *G, cudaStream_t s) {
          fk::second_finish(sums_, totals, alpha_of_call(c), ev.k8, G, s);
        });
    left_out(cd, mt, 0, lc2 + 6, mt - 6 - lc2);
    three_rounds(cd, 3, mt - 5, mt - 4, o + 6, k8b);
    k_k64<<<1, 64, 0, g_st>>>(ev.k8, k8b, K64);
    // Pass 3: the tables folded by the first six challenges, eight slabs of CO entries per tree in
    // each of UL and UR. From here on the walked trees are walked in full: the tables are small.
    T *UL = arena, *UR = UL + NV * 8 * CO, *Fo = UR + NV * 8 * CO, *Sa = Fo + r32(2 * NV * CO), *Sb = Sa + r32(2 * NV * CO / 8);
    weights(mt - 9, lc3);
    lanes_dead();
    open();
    for (int q = 0; q < NV; q++) {
      const fk::Tree &t = trees[walk[q]];
      if (forest_fused)
        ff::third_fold(t, dT, t.alpha, K64, UL + q * 8 * CO, UR + q * 8 * CO, turn().st);
      else
        fk::third_fold(t, dT, t.alpha, K64, UL + q * 8 * CO, UR + q * 8 * CO, turn().st);
    }
    close();
    stored_pass(UL, UR, 8 * CO, mt - 9, lc3, eqprep, eqhi);
    left_out(cd, mt, 0);
    three_rounds(cd, 6, mt - 8, mt - 7, o + 12, ev.k8);
    {
      const size_t total = NV * CO, wanted = (total / 2 + kg3::FOLD8_THREADS - 1) / kg3::FOLD8_THREADS;
      fr::k_fold8<<<dim3((unsigned)std::min<size_t>(wanted, (size_t)g_sms * 4), 2), kg3::FOLD8_THREADS, 0, g_st>>>(
          UL, UR, 8 * CO, mt - 9, total, ev.k8, Fo, Fo + total);
      CK(cudaGetLastError());
    }
    fr::k_rest_point<<<1, 32, 0, g_st>>>(cd, chd + 100, mt, rK);
    rest(Fo, Fo + NV * CO, CO, mt - 9, mu - 9, 0, 100, 101, o + 18, Sa, Sb, 0, true);
    fr::k_end_point<<<1, 32, 0, g_st>>>(cd, chd + 101, chd + mu + 1, mt, rK);
    check("key side: the last layer");
  }
  // Runs the key layers; layer mu, of mu rounds, uses chd[mu]. The last layer also uses chd[100],
  // for its rounds after the ninth, and chd[101], for the point that these rounds leave. The caller
  // has set ev.st->start[0] to the root.
  void key_side(T *msg) {
    ev.out = msg + plan.at_key_layer(0);
    FOREST_MARK("key side: stored layers");
    for (int u = 1; u <= rK; u++) tail(Up[u], u - 1, 0, u - 1, u, ev.take(u - 1), true); // above the trees' roots
    for (int d = 1; d <= dS; d++) stored(d);
    ev.flush();
    check("key side: stored layers");
    for (int LV = reb - 1; LV >= 1; LV--) {
      FOREST_MARK(LV == 4 ? "key side: layer of products of 16" : LV == 3 ? "key side: layer of products of 8"
                  : LV == 2 ? "key side: layer of products of 4" : "key side: layer of products of 2");
      rebuilt(LV);
    }
    FOREST_MARK("key side: last layer");
    last();
  }
  // Proves the group's lookup on the transcript *tr, in the order of blocks/iop/lkp.h: theta (if
  // the group has several calls), alpha, the tie of the two sides' roots, the point rho5, the key
  // side, the table side. Writes plan.nmsg() message slots to msg and fills `claims`. The mask
  // entry of msg[i] (its entry of the committed mask row) is read from msg[i + PADOFF]
  // (prover/hiding/sumcheck_door.cuh), unchecked. The sumchecks also write scratch values at
  // msg[i + 2 PADOFF].
  // Absorbs nothing public: what the verifier knows of the group (sizes, tables, padding) must
  // already be in the transcript. Does not wait for its GPU work. Its first run must not be
  // captured into a CUDA graph: a shape's first product creates its plan, and some kernels'
  // attributes are set at their first launch.
  void prove(Tr *tr, T *msg) {
    for (int y = 0; y < plan.NR; y++)
      if (!trees[y].keys) {
        printf("internal error: look-up forest: prove() before counts() of every call\n");
        name_the_end("forest-keys"); // serving: p258 end 1 forest-keys (kept as an exit(1))
        exit(1);
      }
    lists();
    streams = &sched_streams();
    ev.sch = nullptr;
    solo.isfree = forest_small_tails;
    FOREST_MARK("alpha, tables, the table side's roots");
    {
      lookup_tie::Trees g;
      g.NC = plan.NC, g.NT = plan.NR, g.rK = rK, g.dT = dT; // g.NT: the calls' trees, NR here
      for (int y = 0; y < plan.NR; y++) g.call_of[y] = (unsigned char)plan.call_of[y];
      for (int c = 0; c < plan.NC; c++) g.table[c] = tabs[c].tau, g.pad_row[c] = calls[c].t0;
      lookup_tie::k_draws<<<1, 1, 0, g_st>>>(tr, g, table.pub_dev(), &claims->theta, alpha_of_tree, &claims->a, &ev.st->alpha,
                                             &table.et.st->alpha);
    }
    for (int c = 0; c < plan.NC; c++)
      fr::k_lut_at<<<((1 << tabs[c].k) + 127) / 128, 128, 0, g_st>>>(alpha_of_call(c), tabs[c].tau, 1 << tabs[c].k, tabs[c].lut);
    dead(arena, at.arena_elements);
    table.roots();
    FOREST_MARK("key side: building the levels");
    build();
    FOREST_MARK("the tie");
    lookup_tie::k_tie<<<1, 32, 0, g_st>>>(tr, Up[0], table.plane_roots(), &claims->a, msg + plan.at_tie());
    lookup_tie::k_join<<<1, 1, 0, g_st>>>(tr, Up[0], table.plane_roots(), claims->rho5, &table.et.st->start[table_id(0)], &ev.st->start[0]);
    CK(cudaMemcpyAsync(chd[table_id(0)].rho, claims->rho5, 5 * sizeof(T), cudaMemcpyDeviceToDevice, g_st));
    k_tr2st<<<1, 1, 0, g_st>>>(tr, ev.st);
    key_side(msg);
    FOREST_MARK("table side");
    k_st2st<<<1, 1, 0, g_st>>>(ev.st, table.et.st);
    dead(arena, at.arena_elements);
    table.prove(arena, msg + plan.at_table_layer(0));
    FOREST_MARK("claims");
    {
      fr::Cuts P;
      P.NC = plan.NC, P.NR = plan.NR, P.dT = dT, P.rK = rK;
      for (int c = 0; c < plan.NC; c++) P.iota[c] = plan.iota[c], P.first[c] = plan.first[c];
      fr::k_claims<<<1, 32, 0, g_st>>>(ev.st, table.et.st, chd, dT + rK, 100, 101, table_id(plan.d()), 80, plan.d() + 5, P, claims);
    }
    k_trout<<<1, 1, 0, g_st>>>(table.et.st, tr);
    FOREST_MARK("");
    check("the end");
  }
};
// The size in bytes of the one kind-2 allocation of Forest::init() with these arguments.
inline size_t forest_scratch_bytes(const std::vector<ForestCall> &calls, int lidb, int lmid, int rebuilt) {
  if (!g_sms) dev_info();
  g_grid = g_sms * 8;
  grm::set_up();
  Forest f; // it allocates nothing of its own: see measure()
  f.numbers(calls, lidb, lmid, rebuilt);
  f.measure();
  return f.at.total;
}
// Runs Forest::prove() through pt_graphed() (product_test.cuh), with the CUDA graphs cached in G:
// the first run with given addresses and padding (k below) is direct, the second is captured and
// launched, later ones launch the graph. The proof's bytes are those of a direct run. g_st must not
// be stream 0.
inline void forest_prove_graph(PtGraph &G, Forest &f, Tr *tr, T *msg) {
  std::vector<const void *> k{tr, msg, (const void *)g_st, &f, (const void *)(size_t)forest_fused};
  for (const fk::Tree &t : f.trees) k.push_back(t.keys);
  // The launches also depend on the padding left out, which may differ between one Forest's groups.
  f.lists();
  for (size_t from : f.padded_from) k.push_back((const void *)from);
  pt_graphed(G, k, [&] { f.prove(tr, msg); });
}
