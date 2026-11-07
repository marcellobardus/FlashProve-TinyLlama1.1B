// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// The plan of the attention sub-proof: its public parameters, most from the parameter file
// (`policy.txt`), and its lookup calls. A call has 2^n keys, each a row number of its public table.
// The gap of score (i, j) is row i's largest score over j <= i minus the score. gfk::k_records()
// (prover/attn/forward_kernels.cuh) splits a gap below AShD::C into 2^sh f + r, r < 2^sh (sh: the
// layer's shift), flag d = 0; others have d = 1. Include prover/common/common.h first (u32, u64).
#pragma once
#include "common/end_name.h" // name_the_end (M5g)
#include "common/setting.h"
#include <algorithm>
#include <cmath>
#include <string>
#include <vector>
typedef int8_t i8;
typedef int16_t i16;
typedef uint16_t u16;
typedef long long I64;
// Grid-stride loop over i < n. The loop form is public: the grid-stride loop: a post on NVIDIA's
// developer blog, "CUDA Pro Tip: Write Flexible Kernels with Grid-Stride Loops",
// https://developer.nvidia.com/blog/cuda-pro-tip-write-flexible-kernels-grid-stride-loops/.
#define GSA(i, n)                                                                                                      \
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < (n); i += (size_t)gridDim.x * blockDim.x)
// Number of blocks of t threads for n items, at least 1 and at most cap.
static inline int nbk(size_t n, int t = 256, int cap = 1 << 18) {
  size_t b = (n + t - 1) / t;
  return (int)std::min<size_t>(std::max<size_t>(b, 1), cap);
}
// Lookup segments. RM, RD: r in region M or D (see AP). FM, FD: f. WL: the low 16 bits of remainder
// rem or divisor - 1 - rem. WH: a higher byte of each. IX: (i - row i's argmax) 2^(16 - lt).
enum ASeg { A_RM = 0, A_RD = 1, A_FM = 2, A_FD = 3, A_WL = 4, A_WH = 5, A_IX = 6 };
// The request matrices in the approximate range check's order; qa is the layer sub-proof's.
enum AMat { A_AS = 0, A_AH = 1, A_AO = 2, A_BA = 3, A_QA = 4 };
// Kind of a call's table (tauOf()). A_TWH is for WH. Only makeCallList()'s list 1 uses A_KFM.
enum ATab { A_KR = 0, A_KFM = 1, A_KFD = 2, A_R16 = 3, A_TWH = 4 };
// The public parameters and the sizes derived from them. Kernels take the struct by value. A head's
// scores are cut into tiles: region D is the nT diagonal tiles, region M is the tiles below them.
struct AP {
  int lt;  // T = 2^lt tokens
  int lB;  // tiles of B x B scores, B = 2^lB
  int H;   // query heads
  int lgk; // G = 2^lgk key-value heads
  int ldh; // dh = 2^ldh channels per head
  int ll;  // the sub-proof covers one layer group of Lam = 2^ll layers
  // Before build(): the parameter file's S + 32 * (wd - 3). After: S; softmax weights are <= 2^S.
  int S;
  int rx;       // the step of the exponential table is 2^-rx
  int kFM, kFD; // tables of kind A_KFM and A_KFD have 2^kFM and 2^kFD rows
  int D;        // the parameter file's line D; only 2,048 is supported (fork.cuh)
  int lmS;      // the matrix `as` has columns of 2^lmS cells
  int lwH, lwO, lwQ; // ah, ao and qa have columns of T * 2^lwH, T * 2^lwO and T * 2^lwQ cells
  int la;            // tables of AFront (front.cuh) on ao and qa have one entry per 2^la cells
  int lmb;           // bA has columns of 2^lmb cells
  // The parameter file's line iotaA_0. AFront::plan() exits unless it equals 32 + APlan::dT;
  // nothing else in prover/attn reads it.
  int iota;
  int nmin; // a call has at least 2^nmin keys
  // Set by APlan::build(). nT: tiles per side. HA: query heads per key-value head. u0: Tt's size.
  int wd = 0; // bytes of a division remainder: 3 or 4
  int Sw = 0; // the value of the line S as the parameter file has it
  int T, B, nT, lnT, hP, G, HA, dh, Lam, u0, ls, dq, dkv;
  int nM, nD; // a piece of RM or FM has 2^nM keys, and a piece of RD or FD has 2^nD keys
  I64 Smax;   // the largest |score|
};
// For shift sh. A gap >= C gets weight 0; adding Om where j > i makes it >= C if |score| <= Smax.
struct AShD {
  I64 C, rho, nHM, nHD, Om;
};
// A piece (seg, layer lam, query head a of each key-value head): 2^b keys of its call from key off.
struct APiece {
  int seg, lam, a, b;
  size_t off;
};
struct ACall {
  std::string name;
  int kind;              // ATab
  int sh;                // the shift, for table kinds that have one
  int n, iota, k;        // 2^n keys in 2^iota trees (tree b starts at key b 2^dT); 2^k table rows
  int fn;                // 0 range table, 1 general function table, 2 table of byte limbs
  u32 pk0;               // the table row of the padding keys
  std::vector<int> segs; // segment kind of each entry of the call
  std::vector<APiece> pc;
  std::vector<u64> tau; // the table's row values
  std::vector<u32> pub; // the number of padding keys in each tree
};
// The caller sets the parameters in p, form and sh; build() derives the rest. The calls in C run as
// one lookup forest (prover/hiding/forest_plan.h): each call's keys form trees of 2^dT keys.
struct APlan {
  AP p;
  int form = 1;        // which call list: 1 or 3, see `makeCallList`; other values act as 1
  std::vector<int> sh; // one shift per layer
  std::vector<int> Tt; // the exponential table Texp[u], u < u0
  std::vector<ACall> C;
  int dT = 0; // min of `identity_cap` (the parameter file's line lidb) and the smallest call's n

  // Splits the value of the parameter file's line S into S and wd. Returns false if it is invalid.
  static bool decodeS(int word, int &S, int &wd) {
    S = word % 32;
    wd = 3 + word / 32;
    return word >= 0 && wd <= 4 && S >= 1 && S <= 14;
  }
  // Returns Texp[u] = round(2^S exp(-u / 2^rx)) for u = 0, 1, ... while it is >= 1. Larger u have
  // weight 0. A public work that states this technique: Sun, Li, Zhang, "zkLLM", arXiv:2404.16109,
  // section 5.1.1 (forward_kernels.cuh has the whole credit and says what differs).
  static std::vector<int> expTable(int S, int rx) {
    std::vector<int> t;
    for (int u = 0;; u++) {
      const I64 x = (I64)nearbyint(ldexp(exp(-(double)u / (double)(1 << rx)), S));
      if (x < 1) return t;
      t.push_back((int)x);
    }
  }
  AShD shd(int s) const {
    AShD d;
    d.C = (I64)p.u0 << s;
    d.rho = 65536 - (1 << s);
    d.nHM = (2 * p.Smax + d.rho) / d.rho;
    d.nHD = (4 * p.Smax + d.rho) / d.rho;
    d.Om = 2 * p.Smax + d.C;
    return d;
  }
  // log2 of the number of keys in a piece of segment kind seg.
  int segB(int seg) const {
    switch (seg) {
    case A_RM:
    case A_FM:
      return p.nM;
    case A_RD:
    case A_FD:
      return p.nD;
    case A_WL:
      return 1 + p.lgk + p.lt + p.ldh;
    case A_WH:
      return p.lgk + p.lt + p.ldh + (p.wd == 4 ? 1 : 0);
    default:
      return p.lgk + p.lt;
    }
  }
  // A_KFM, A_KFD: row f is gap - r + 2^32 d + 2^33 * weight, then filler rows. A_KR: bit 40 is d.
  std::vector<u64> tauOf(int kind, int s) const {
    if (kind == A_KR || kind == A_TWH) {
      std::vector<u64> t(65536);
      for (u32 j = 0; j < 65536; j++)
        t[j] = (u64)(j & 255) + ((u64)(j >> 8) << 20) + (kind == A_KR ? ((u64)(j >= (1u << s)) << 40) : 0);
      return t;
    }
    if (kind == A_R16) {
      std::vector<u64> t(65536);
      for (u32 j = 0; j < 65536; j++) t[j] = j;
      return t;
    }
    const AShD d = shd(s);
    const int k = kind == A_KFD ? p.kFD : p.kFM;
    const I64 nH = kind == A_KFD ? d.nHD : d.nHM;
    std::vector<u64> t((size_t)1 << k);
    for (size_t j = 0; j < t.size(); j++) {
      if ((I64)j < p.u0)
        t[j] = ((u64)j << s) + ((u64)Tt[j] << 33);
      else if ((I64)j < p.u0 + nH)
        t[j] = (u64)(d.C - (1 << s) + d.rho * ((I64)j - p.u0)) + (1ull << 32);
      else
        t[j] = (u64)(d.C - (1 << s) + d.rho * ((I64)j - p.u0 - nH) + d.rho / 2) + (1ull << 32);
    }
    return t;
  }

  // Exits with code 2 if it rejects the parameters. Call it once: it appends to C.
  void build() {
    if (!p.wd) {
      p.Sw = p.S;
      if (!decodeS(p.Sw, p.S, p.wd)) {
        printf("attention: the value %d of the parameter file's line S is not accepted: it must be s + 32 * (wd - 3) "
               "with s from 1 to 14 and wd 3 or 4\n",
               p.Sw);
        name_the_end("attn-s-line"); // serving: p258 end 1 attn-s-line (M5i)
        exit(2);
      }
    }
    p.T = 1 << p.lt;
    p.B = 1 << p.lB;
    p.lnT = p.lt - p.lB;
    p.nT = 1 << p.lnT;
    p.hP = p.nT / 2;
    p.G = 1 << p.lgk;
    p.HA = p.H / p.G;
    p.dh = 1 << p.ldh;
    p.Lam = 1 << p.ll;
    p.ls = p.la - p.lt;
    p.dq = p.H * p.dh;
    p.dkv = p.G * p.dh;
    p.Smax = (I64)p.dh * 2047 * 2047;
    p.nM = p.lgk + 2 * p.lt - 1;
    p.nD = p.lgk + p.lt + p.lB;
    Tt = expTable(p.S, p.rx);
    p.u0 = (int)Tt.size();
    if (p.H % p.G || p.nT < 4 || p.ls > p.ldh || p.ls < 0 || p.lt + p.S + 1 > 8 * p.wd || (int)sh.size() != p.Lam) {
      printf("attention: this shape is not supported. Required: G divides H, T / B >= 4, 0 <= la - lt <= ldh, lt + S < "
             "8 wd, and one shift for each of the Lam layers\n");
      name_the_end("attn-shape"); // serving: p258 end 1 attn-shape (M5i)
      exit(2);
    }
    if (p.lmS > p.nD) { // In each plane of `as` a piece of RD or FD has 2^nD cells, read as
                        // 2^(nD - lmS) whole columns (planeA() in fork.cuh).
      printf(
          "policy: af-shape-lmS: lmS = %d is above lgk + lt + lB = %d, so a column of the matrix as_g would be longer "
          "than one plane of a piece\n",
          p.lmS, p.nD);
      name_the_end("attn-lms"); // serving: p258 end 1 attn-lms (M5i)
      exit(2);
    }
    for (int s : sh) {
      const AShD d = shd(s);
      const I64 nFM = ((I64)1 << p.kFM) - p.u0 - d.nHM, nFD = ((I64)1 << p.kFD) - p.u0 - d.nHD;
      if (nFM < 0 || nFD < 0 || nFM > d.nHM || nFD > d.nHD || d.C + 65536 > (1 << 24) || s < 4 || s > 12) {
        printf("attention: the shift sh = %d is not supported: sh must be from 4 to 12, u0 * 2^sh + 65536 at most "
               "2^24, and both 0 <= 2^kFM - u0 - nHM <= nHM and 0 <= 2^kFD - u0 - nHD <= nHD must hold; "
               "kFM = %d, kFD = %d\n",
               s, p.kFM, p.kFD);
        name_the_end("attn-shift"); // serving: p258 end 1 attn-shift (M5i)
        exit(2);
      }
    }
    makeCallList();
    for (auto &c : C) placePieces(c);
    dT = identity_cap();
    for (auto &c : C) dT = std::min(dT, c.n);
    for (auto &c : C) setTrees(c, dT);
  }

  const APiece *piece(int ci, int seg, int lam, int a) const {
    for (auto &x : C[ci].pc)
      if (x.seg == seg && x.lam == lam && x.a == a) return &x;
    return nullptr;
  }
  // Returns the index of the call that contains the pieces of this segment kind and layer, or -1.
  int callOf(int seg, int lam) const {
    for (size_t i = 0; i < C.size(); i++)
      for (int s : C[i].segs)
        if (s == seg && (C[i].kind >= A_R16 || C[i].sh == sh[lam])) return (int)i;
    return -1;
  }

private:
  void add(std::string name, int kind, int s, std::vector<int> segs) {
    ACall c;
    c.name = name;
    c.kind = kind;
    c.sh = s;
    c.segs = segs;
    C.push_back(c);
  }
  // KR, KFM and KFD get a call per distinct shift in sh, ascending. List 1: KR {RM RD}; KFM {FM};
  // KFD {FD}; R16 {WL IX}; WH {WH}. List 3 (the program cp_prove's proofs) has KFD {FM FD}, no KFM.
  void makeCallList() {
    std::vector<int> shs = sh;
    std::sort(shs.begin(), shs.end());
    shs.erase(std::unique(shs.begin(), shs.end()), shs.end());
    auto perShift = [&](const char *name, int kind, std::vector<int> segs) {
      for (int s : shs) add(name + std::to_string(s), kind, s, segs);
    };
    perShift("KR", A_KR, {A_RM, A_RD});
    if (form != 3) perShift("KFM", A_KFM, {A_FM});
    perShift("KFD", A_KFD, form == 3 ? std::vector<int>{A_FM, A_FD} : std::vector<int>{A_FD});
    add("R16", A_R16, 0, {A_WL, A_IX});
    add("WH", A_TWH, 0, {A_WH});
  }
  // Places the pieces: in the order (entry of segs, lam, a), stable-sorted by size with the largest
  // first, at consecutive offsets, so off is a multiple of 2^b. Also sets n, k, fn, pk0 and tau.
  void placePieces(ACall &c) {
    for (int seg : c.segs)
      for (int lam = 0; lam < p.Lam; lam++)
        if (c.kind >= A_R16 || sh[lam] == c.sh)
          for (int a = 0; a < p.HA; a++) c.pc.push_back({seg, lam, a, segB(seg), 0});
    std::stable_sort(c.pc.begin(), c.pc.end(), [](const APiece &x, const APiece &y) { return x.b > y.b; });
    size_t o = 0;
    for (auto &x : c.pc) {
      x.off = o;
      o += (size_t)1 << x.b;
    }
    int l = 0;
    while (((size_t)1 << l) < o) l++;
    c.n = std::max(l, p.nmin);
    c.k = c.kind == A_KFM ? p.kFM : c.kind == A_KFD ? p.kFD : 16;
    c.fn = c.kind == A_KR || c.kind == A_TWH ? 2 : c.kind == A_R16 ? 0 : 1;
    c.pk0 = c.kind == A_KFM || c.kind == A_KFD ? p.u0 : 0;
    c.tau = tauOf(c.kind, c.sh);
  }

public:
  // Returns the number of keys that pieces cover. Keys from this index up to 2^n are padding keys.
  static size_t realKeys(const ACall &c) {
    size_t o = 0;
    for (auto &x : c.pc) o += (size_t)1 << x.b;
    return o;
  }
  // Sets c.iota and c.pub for trees of 2^depth keys.
  // Requires c.n and c.pc to be set, and depth <= c.n.
  static void setTrees(ACall &c, int depth) {
    c.iota = c.n - depth;
    const size_t real = realKeys(c), keys = (size_t)1 << depth;
    c.pub.assign((size_t)1 << c.iota, 0);
    for (size_t b = 0; b < c.pub.size(); b++)
      c.pub[b] = (u32)((b + 1) * keys - std::min(std::max(real, b * keys), (b + 1) * keys));
  }
};
