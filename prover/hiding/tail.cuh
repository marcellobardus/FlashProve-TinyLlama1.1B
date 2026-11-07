// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

/* Parts of a sub-proof's approximate range check on committed cells ("smallness test" in messages,
   `tail` in identifiers), meant to show |cell| < 2^(HT_LBZ + 1), a looser bound than honest cells
   meet. A sub-proof covers one part of the forward pass. HiddenTail (prover/hiding/tail_f.cuh) is
   the check itself. Include this file after prover/hiding/product_test.cuh,
   prover/hiding/trees.cuh and the helpers GSL, `nblk` and `k_eq` of prover/prover/base.cuh. */
#pragma once
#include "common/end_name.h" // name_the_end (M5g)
#include "common/element_from_hash.h"
#include "hiding/forest.cuh"
#include "hiding/late.cuh"
#include "hiding/tail_gam.cuh"
#include <functional>
#include <memory>
#include <string>
#include <vector>
#undef TT // a macro of hiding/pt_transcript.h

constexpr int HT_TAU = ETAU; // tau, the number of random combinations u; a multiple of 32
/* A combination Z (see MatD) must lie in the interval [-2^HT_LBZ, 2^HT_LBZ), or the request gets
   no proof. It is split as Z + 2^HT_LBZ = d0 + 2^10 d1 with d1 < 2^HT_D1; one range lookup checks
   all d0, another all d1. A matrix has at most HT_KMAX columns of signed bytes. If its
   coefficients are modelled as independent (0 with probability 1/2, 1 and -1 with 1/4 each), each
   Z leaves the interval with probability at most 2 exp(-2^(2 HT_LBZ) / (2^14 HT_KMAX)) < 2 e^-128
   (Chernoff bound). */
constexpr int HT_LBZ = ELBZ, HT_D1 = HT_LBZ - 9, HT_KMAX = 8191;

/* One committed matrix M of a sub-proof: m rows and k columns of cells (v, column by column).
   Z[i][u] = sum_c M[i][c] Cf[c][u] are its combinations, Cf the coefficients (each -1, 0 or 1),
   CfT Cf's zero-padded transpose. Nothing writes support and sign. */
struct MatD {
  size_t m = 0, k = 0;
  i8 *v = nullptr, *Cf = nullptr, *CfT = nullptr;
  int *Z = nullptr;
  uint4 *support = nullptr, *sign = nullptr; // the coefficients' support and sign bits; never written
};

// ceil(log2(x)); 0 for x = 0
static int l2i(size_t x) {
  int l = 0;
  while (((size_t)1 << l) < x) l++;
  return l;
}

/* Where an array's rows are stored among a lookup's keys (its looked-up values): rows below r1
   in piece 0, the others, if any, in piece 1 (see HPieceG). p[b] points to piece b's first key; CP
   is the row pitch. */
struct HRows {
  u16 *p[2] = {nullptr, nullptr};
  u32 r1 = 0;
  size_t CP = 0;
  __host__ __device__ u16 *row(u32 r) const { return r < r1 ? p[0] + (size_t)r * CP : p[1] + (size_t)(r - r1) * CP; }
};
// c: the claims of a lookup forest (forest.cuh), which runs several lookups as one argument.
// c->err is always 0 (ForestTable::planes(), forest_table.cuh).
__global__ void k_forest_err(const ForestClaims *c, u32 *e) {
  if (c->err) *e |= 1;
}
// Writes the lookups' multiplicity bit X = r 2^lw + x to cell (r, cb + x) of BZ, the committed
// array of these bits; kb is BZ's row stride.
__global__ void kh_bz(const u8 *bits, size_t B, int lw, size_t cb, size_t kb, u16 *cB) {
  GSL(X, B) cB[(X >> lw) * kb + cb + (X & (((size_t)1 << lw) - 1))] = bits[X];
}

__global__ void kh_ones(T *o, size_t n) {
  GSL(i, n) o[i] = F::one();
}
// A piece: Rp rows (a power of two, from r0) and Cp columns (from c0) of an array, stored among a
// lookup's keys row by row from key `off`. Each row is padded to CP keys, a power of two, so the
// piece has 2^b keys.
struct HPieceG {
  int row = -1; // index in HiddenTail::vrows of the piece's virtual row
  size_t r0 = 0, Rp = 0, c0 = 0, Cp = 0, CP = 0, off = 0;
  int b = 0;
};
// A virtual row (HiddenTail::vrows) of the committed array T: sum_r cv[r] T[r][.].
struct HVRowG {
  std::string nm;
  hz::TreeN *T = nullptr;
  const ::T *cv = nullptr; // T->R GPU coefficients; valid after `make_cv` and `make_powers`
};
// HiddenTail::init() arranges each range lookup's keys with these helpers.
struct TailPieces {
  // Splits R rows into blocks (first row, number of rows), each the largest power of two that fits.
  static std::vector<std::pair<size_t, size_t>> blocks(size_t R) {
    std::vector<std::pair<size_t, size_t>> b;
    for (size_t r0 = 0; r0 < R;) {
      size_t Rp = 1;
      while (2 * Rp <= R - r0) Rp *= 2;
      if (r0 % Rp) {
        printf("internal error: smallness test: a block of %zu rows begins at row %zu, which is "
               "not a multiple of its size\n", Rp, r0);
        name_the_end("tail-setup"); // serving: p258 end 1 tail-setup (M5g)
        exit(2);
      }
      b.push_back({r0, Rp});
      r0 += Rp;
    }
    return b;
  }
  static HPieceG mkp(size_t r0, size_t Rp, size_t c0, size_t Cp) {
    HPieceG p;
    p.r0 = r0;
    p.Rp = Rp;
    p.c0 = c0;
    p.Cp = Cp;
    return p;
  }
  // Sets each piece's CP, b and `off`: largest piece first, from key 0 on, equal sizes in P's
  // order. Sets n (log2 of the key count, at least 10) and io_: the lookup's keys are cut into
  // 2^io_ product trees of 2^(n - io_) consecutive keys, at most 2^identity_cap() each
  // (prover/common/setting.h). pub[t] counts tree t's padding keys: the rows' padding and the keys
  // after the last piece.
  static void placeP(std::vector<HPieceG> &P, int &n, int &io_, std::vector<u32> &pub) {
    for (auto &p : P) {
      p.CP = hz::p2c(p.Cp);
      p.b = l2i(p.Rp) + l2i(p.CP);
    }
    std::vector<int> ord(P.size());
    for (size_t i = 0; i < ord.size(); i++) ord[i] = (int)i;
    std::stable_sort(ord.begin(), ord.end(), [&](int a, int b) { return P[a].b > P[b].b; });
    size_t o = 0;
    for (int i : ord) {
      P[i].off = o;
      o += (size_t)1 << P[i].b;
    }
    n = std::max(l2i(hz::p2c(o)), 10);
    io_ = std::max(0, n - identity_cap());
    const size_t N = (size_t)1 << n, I = (size_t)1 << io_;
    const int dT = n - io_;
    pub.assign(I, 0);
    for (auto &p : P)
      for (size_t rr = 0; rr < p.Rp; rr++)
        for (size_t cc = p.Cp; cc < p.CP; cc++) pub[(p.off + rr * p.CP + cc) >> dT]++;
    for (size_t ix = o; ix < N; ix++) pub[ix >> dT]++;
  }
  // Allocates a range lookup's 2^n keys; the padding keys stay 0.
  static void mkkeys(int n, u16 *&keys) {
    const size_t N = (size_t)1 << n;
    if (keys) return; // a key array that the calling code passes in is used as it is
    hz::alloc(keys, N * 2, hz::FORK);
    CK(cudaMemset(keys, 0, N * 2));
  }
};
#include "hiding/moments.cuh"
#include "hiding/tail_f.cuh"
