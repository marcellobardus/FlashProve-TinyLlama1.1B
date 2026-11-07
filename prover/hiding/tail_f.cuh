// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

/* The prover's side of the approximate range check (RNG, blocks/iop/rng.h) of a sub-proof (`fork`
   in identifiers): it is to show |cell| < 2^(HT_LBZ + 1) for the cells of its matrices. Messages
   say "smallness test", identifiers "tail". Three adjacent cells of a column are committed as one
   field element, a "packed entry". The prover commits to the digits of each matrix M's combinations
   Z = M Cf under the verifier's random matrix Cf (array ZZ) and to two range lookups' multiplicity
   bits (BZ). Linear claims bind Z to packed entries ("ties") and tables of partial sums ("reads").
   Per proof: `draw_matrix`; the caller fills Z; `fold_to_group2`, `group2`, `make_cv`. */
#pragma once
// tail.cuh includes this file at its end: include tail.cuh, not this file.
#include "common/end_name.h" // name_the_end (M5g)
#include "hiding/eqs.cuh"
#include "hiding/tie_h.cuh"
#include "hiding/door.cuh"
#include "common/lie.cuh"
#include <atomic>
#include "frame/hiding_conditions.h" // blk::RangeCheck
static_assert(EZD0 == 10 && EZD1 == HT_D1, "kf_zz_plain splits a folded value into 10 low bits and HT_D1 high bits");


/* Splits Z[y][u] (u < HT_TAU) as Z + 2^HT_LBZ = d0 + 2^10 d1. It writes d0 and d1 to zz (HT_TAU
   rows of KZ = 2 mz cells) at row u, columns col + y and mz + col + y, and to the lookup keys' rows
   k10 (d0) and k8 (d1). If a Z is outside its window [-2^HT_LBZ, 2^HT_LBZ), it sets *err = 3, for
   which the request gets no proof, and goes on with truncated digits. Launch it on a grid of m / 32
   by HT_TAU / 32 blocks (x by y), each 32 threads wide and 8 high: 32 must divide m and HT_TAU. */
// Public sources for the techniques of this routine.
// The idea is public: a tiled transpose through padded shared memory: M. Harris,
// "An Efficient Matrix Transpose in CUDA C/C++", NVIDIA Technical Blog, 2013, "Matrix Transpose",
// "Coalesced Transpose Via Shared Memory" and "Shared Memory Bank Conflicts",
// https://developer.nvidia.com/blog/efficient-matrix-transpose-cuda-cc/.
// The flag err: each thread that finds a fault writes it with an ordinary store. The last two works
// below count unsynchronized writes of two threads to one location as undefined behavior.
// Public source: the flag is_bad: Oracle Developer Studio 12.5, "Thread Analyzer User's Guide"
// (updated June 2016), "Benign Data Races", example "A Program that Verifies Array-Value Types";
// https://docs.oracle.com/cd/E60778_01/html/E60750/gecqt.html.
// Public source: the flag g_over: AMD, HIP documentation, "Multi-kernel programming: breadth-first
// search tutorial", "Best practices", "Design principles", item 4 "Flag-based control";
// https://rocm.docs.amd.com/projects/HIP/en/develop/_sources/tutorial/programming-patterns/multikernel_bfs.rst.
// Public source: NVIDIA, CUDA C++ Programming Guide, section 7.5 "Memory Fence Functions", first
// paragraph; https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html.
// See also: the same rule: NVIDIA, CUDA Programming Guide,
// "C/C++ Language Extensions", "Memory Fence Functions",
// https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-extensions.html.
__global__ void kf_zz_plain(const int *Z, size_t col, size_t mz, u16 *zz, const HRows k10, const HRows k8, u32 *err) {
  __shared__ int tile[32][33];
  const size_t y0 = (size_t)blockIdx.x * 32;
  const int u0 = blockIdx.y * 32;
  for (int r = threadIdx.y; r < 32; r += 8) {
    const int z = Z[(y0 + r) * HT_TAU + (u32)(u0 + threadIdx.x)];
    tile[r][threadIdx.x] = z;
    if (z < -(1 << HT_LBZ) || z >= (1 << HT_LBZ)) *err = 3;
  }
  __syncthreads();
  for (int r = threadIdx.y; r < 32; r += 8) {
    const u32 zp = (u32)(tile[threadIdx.x][r] + (1 << HT_LBZ));
    const u32 u = (u32)(u0 + r);
    const size_t c = col + y0 + threadIdx.x;
    const u16 d0 = (u16)(zp & 1023), d1 = (u16)((zp >> 10) & ((1 << HT_D1) - 1));
    zz[(size_t)u * 2 * mz + c] = d0;
    zz[(size_t)u * 2 * mz + mz + c] = d1;
    k10.row(u)[c] = d0;
    k8.row(u)[c] = d1;
  }
}
// One entry of HiddenTail::list: a committed array, its commitment number and its proximity tag.
struct HListF {
  hz::TreeN *T = nullptr;
  u32 num = 0, tag = 0;
};
// Launches use the per-thread default stream. Forest F uses g_st: assumed the same, not checked.
// M: NM matrices, each "own" if committed by this sub-proof, "foreign" if by another, its owner.
struct HiddenTail {
  static constexpr int NCZ = 2; // the range lookups: 0 for the low digits, 1 for the high digits
  int NM = 0, bits = 0, n9 = 0 /* both lookups have 2^n9 keys */, kz[NCZ] = {EZD0, EZD1};
  u32 sid0 = 0;
  MatD *M = nullptr;
  std::vector<int> cls /* 0 own, 1 foreign */, shr /* > 0: a "border", read by other sub-proofs */,
      pcOf /* the matrix's space in hw::Walk (walk.cuh), counted from spP0 of `walk_tail_f` */;
  std::vector<size_t> off;
  size_t mz = 0, KZ = 0, kb = 0;
  hz::TreeN ZZ, BZ;
  // The two range lookups ("calls" in identifiers) as one lookup forest (forest.cuh): call 0 has
  // the low digits, call 1 the high ones. It fills F.plan.nmsg() slots of the sub-proof's message.
  Forest F;
  PtGraph graph;
  int rebuilt = 3; // set before `init`: low key-tree levels recomputed, not stored; proof unchanged
  std::vector<HPieceG> pc[NCZ];
  // A virtual row of an array X is sum_r cv[r] X[r][.]. ZZ: one per block of rows (both lookups'
  // keys), rowX4 (the reads), rowTie (the ties). BZ, from rowBZ0: bit claims 0 and 1, then ones.
  std::vector<HVRowG> vrows;
  std::vector<HListF> list;
  int rowX4 = -1, rowTie = -1, rowBZ0 = -1;
  u32 *pubZ = nullptr /* each tree's number of padding keys, the same for both lookups */,
      *r2 = nullptr;
  u16 *kk[NCZ] = {nullptr, nullptr}, *cBZ = nullptr;
  // ZZ's cells (HT_TAU rows of KZ); must not change from the commit point ("moment") to hd::close()
  u16 *zzPlain = nullptr;
  int chainMoment = 3; // the number of ZZ's and BZ's commit point (moments.cuh); the callers set 2
  // Describes that commit point's two arrays (ch::ArrayAt): pointer 0 is zzPlain, pointer 1 is cBZ.
  void chain_arrays(std::vector<ch::ArrayAt> &o) const {
    ch::ArrayAt z, b;
    z.name = "ZZ", z.cells = ch::ArrayAt::Cell16, z.R = HT_TAU, z.K = KZ, z.row_stride = KZ, z.pointer = 0;
    b.name = "BZ", b.cells = ch::ArrayAt::Cell16, b.R = 32, b.K = kb, b.row_stride = kb, b.pointer = 1;
    o.push_back(z);
    o.push_back(b);
  }
  /* The range check's equations, numbered as in blocks/iop/rng.h and the verifier's HTailF2
     (verifier/src/vf_smallness_test_members.inc): eq_first[mu] + i' NP + u for packed row i' of matrix mu and
     combination u (m' such rows take NP 2^ceil(log2 m') numbers); eq_first[NM] + q NP + u for the
     q-th run of L rows of the reads (hw::TableRead). Equation j has weight theta^(n + j): hw::Walk
     draws theta and weighs claim c of n by theta^c; a tie or a run is one claim, times `lead`. */
  static constexpr size_t NP = (size_t)1 << (HT_TAU <= 128 ? 7 : 8);
  static_assert(HT_TAU > 64 && HT_TAU <= 256, "NP is HT_TAU rounded up to a power of two");
  std::vector<unsigned long long> eq_first;
  struct Lead {
    u32 claim;                 // the claim's number in the walk
    unsigned long long first;  // the number of its first equation
  };
  std::vector<Lead> leads;                    // filled by walk_tail_f, in the order of the claims
  T *lead = nullptr, *step = nullptr;         // lead[i] for leads[i]; after them step = theta^NP
  unsigned long long *lead_exponent = nullptr;
  std::vector<unsigned long long> lead_exponent_host;
  size_t most_leads = 0;
  // vt_point = theta^(2^(vt_bits - 1)), .., theta^2, theta: vt[u] = theta^u in hd::SET_BITS form.
  static constexpr int vt_bits = HT_TAU <= 128 ? 7 : 8;
  T *vt_point = nullptr;
  /* Marks each pair of D whose row vector cv is one of this check's vectors as a product over the
     bits of the row number (hd::RowForm); `hd::close_by_frame` fails on a ZZ or BZ pair without a
     form. ZZ's blocks of rows and BZ's two bit claims have equality vectors, vt powers of theta,
     BZ's last vector ones. Call once at set-up, after `add_eqs` on D: the points do not move. */
  void say_forms(hd::Door &D) const {
    auto mark = [&D](const T *cv, hd::RowForm form, const T *point, int bits, size_t first) {
      for (hd::Term &x : D.t)
        for (hd::Pair &p : x.v)
          if (p.cv == cv) p.form = form, p.point = point, p.bits = bits, p.first = first;
    };
    for (size_t i = 0; i < pc[0].size(); i++) mark(vrows[pc[0][i].row].cv, hd::EQUALITY, zin((int)i), l2i(pc[0][i].Rp), pc[0][i].r0);
    mark(vt, hd::SET_BITS, vt_point, vt_bits, 0);
    for (int e = 0; e < 2; e++) mark(wBZ + e * 32, hd::EQUALITY, F.claims->zb[e], 5, 0);
    mark(wBZ + 2 * 32, hd::ONES, nullptr, 0, 0);
  }
  size_t equations(size_t reads) const { return (size_t)eq_first[NM] + reads * NP; }
  // Returns the number of claims to plan the batching round with: `claims` (all of hw::Walk's)
  // plus the range check's equation numbers (equations()). Call it after walk_tail_f().
  size_t batched_rows(u32 claims) const { return (size_t)claims + equations(most_leads - (size_t)NM); }
  T *vt = nullptr,
    *vp = nullptr /* the ties' weights on the combinations: vt */, *wZZ = nullptr, *wBZ = nullptr;
  std::vector<T *> gamT /* Cf_mu vt: a read's weights on its table */, gamP /* the same for ties */,
      wtie /* the tie's weights on Z: m_mu elements Er[i / 3] B^(i % 3) (tie_h.cuh) */,
      tieR /* its weights on the packed entries' columns: gamP, k_mu elements */;
  std::vector<size_t> tieX0 /* the matrix's first index in its space of hw::Walk: 0 */,
      tieN /* k_mu */;
  /* One hc::TieH per matrix (tie_h.cuh defines Er and B). Its point r_mu has one coordinate per bit
     of the packed-row index. Matrix mu's space in hw::Walk has one variable per column:
     pcOf[mu] = mu, tieX0 = 0, tieN = k, tieR = gamP. */
  std::vector<hc::TieH> TH;
  // Where lookup q's keys are stored, for the kernels that write the digits (HRows: tail.cuh).
  HRows rows(int q) const {
    if (pc[q].empty() || pc[q].size() > 2 || pc[q][0].r0 || (pc[q].size() == 2 && pc[q][1].CP != pc[q][0].CP)) {
      printf("internal error: smallness test: the lookup keys of ZZ are in %zu pieces of a "
             "call (1 or 2 are allowed), or they do not begin at row 0\n", pc[q].size());
      exit(2);
    }
    HRows r;
    for (size_t b = 0; b < pc[q].size(); b++) r.p[b] = kk[q] + pc[q][b].off;
    r.r1 = (u32)pc[q][0].Rp;
    r.CP = pc[q][0].CP;
    return r;
  }
  /* Reads M[mu].m and .k and keeps the pointer M; later set-up reads .v through it. cls_[mu]: 0 for
     an own matrix, 1 for a foreign one; shr_: see shr; sid0_: see `stream_ids`. Allocates every Z
     that is null. If owner is given, matrix muForeign is the owner's border matrix muOwner. */
  void init(MatD *M_, int NM_, const int *cls_, const int *shr_, int bits_, u32 sid0_, HiddenTail *owner, int muForeign,
            int muOwner) {
    M = M_;
    NM = NM_;
    bits = bits_;
    sid0 = sid0_;
    cls.assign(cls_, cls_ + NM);
    shr.assign(NM, 0);
    if (shr_)
      for (int mu = 0; mu < NM; mu++)
        if (cls[mu] == 0) shr[mu] = shr_[mu];
    pcOf.assign(NM, -1);
    off.assign(NM, 0);
    mz = 0;
    for (int mu = 0; mu < NM; mu++) {
      if (M[mu].k > (size_t)HT_KMAX) {
        printf("policy: a folded matrix has %zu columns; at most %d are supported\n", (size_t)M[mu].k, HT_KMAX);
        name_the_end("tail-setup"); // serving: p258 end 1 tail-setup (M5g)
        exit(2);
      }
      if (M[mu].m % 32) {
        printf("smallness test: a folded matrix has %zu rows; the number must be a multiple of "
               "32\n", (size_t)M[mu].m);
        name_the_end("tail-setup"); // serving: p258 end 1 tail-setup (M5g)
        exit(2);
      }
      off[mu] = mz;
      mz += M[mu].m;
      if (!M[mu].Z) hz::alloc(M[mu].Z, M[mu].m * HT_TAU * 4, hz::FORK);
    }
    KZ = 2 * mz;
    if (NM > 32) {
      printf("smallness test: at most 32 matrices\n");
      name_the_end("tail-setup"); // serving: p258 end 1 tail-setup (M5g)
      exit(2);
    }
    {
      hz::Late late; // the lookups' keys and engine are first used in fold_to_group2()
      std::vector<u32> pub[NCZ];
      int n[NCZ], io[NCZ];
      for (int q = 0; q < NCZ; q++) {
        pc[q].clear();
        for (auto &b : TailPieces::blocks(HT_TAU))
          pc[q].push_back(TailPieces::mkp(b.first, b.second, q * mz, mz));
        TailPieces::placeP(pc[q], n[q], io[q], pub[q]);
        TailPieces::mkkeys(n[q], kk[q]);
      }
      if (n[0] != n[1] || pub[0] != pub[1]) {
        printf("internal error: smallness test: its two lookups differ in their keys' number or padding\n");
        name_the_end("tail-setup"); // serving: p258 end 1 tail-setup (M5g)
        exit(2);
      }
      n9 = n[0];
      hz::alloc(pubZ, pub[0].size() * 4, hz::SHAPE);
      CK(cudaMemcpy(pubZ, pub[0].data(), pub[0].size() * 4, cudaMemcpyHostToDevice));
      F.init({ForestCall{n9, kz[0], 0, nullptr, 0}, ForestCall{n9, kz[1], 0, nullptr, 0}}, identity_cap(), 18, rebuilt);
      if (F.plan.NT != 1 << F.plan.rK || F.plan.iota[0] != io[0] || F.plan.NC != 2 || F.plan.cK != 1) {
        printf("internal error: smallness test: the forest of its two lookups has not the expected trees or calls\n");
        name_the_end("tail-setup"); // serving: p258 end 1 tail-setup (M5g)
        exit(2);
      }
      kb = (size_t)1 << (F.plan.kT + F.plan.cK); // bit j of table row t's multiplicity in call c: X = (t | c | j) = (t 2^cK + c) 32 + j, at BZ[X / kb][X mod kb]
    }
    {
      hz::Late late; // cBZ and zzPlain are first written in fold_to_group2()
      {
        // The digits and the multiplicity bits go to ch::moment() in the buffers zzPlain and cBZ,
        // so the objects ZZ and BZ are set up for their shape only.
        hz::ShapeOnly numbers_alone;
        ZZ.setup("ZZ", HT_TAU, KZ, 1, 1, sid0 + 12, bits);
        BZ.setup("BZ", 32, kb, 1, 1, sid0 + 24, bits);
      }
      hz::alloc(cBZ, 32 * kb * 2, hz::FORK);
      hz::alloc(zzPlain, (size_t)HT_TAU * KZ * 2, hz::FORK);
    }
    auto al = [&](T *&p, size_t n) {
      hz::alloc(p, sizeof(T) * n, hz::SHAPE);
      CK(cudaMemset(p, 0, sizeof(T) * n));
    };
    hz::alloc(r2, (2 * DW + 2 * SW) * 4, hz::SHAPE);
    const int nb = (int)pc[0].size(); // the number of blocks of the HT_TAU rows: 2 at HT_TAU = 160
    const int nvZ = nb + 2;
    al(wZZ, (size_t)nb * HT_TAU);
    al(wBZ, 3 * 32);
    eq_first.assign(NM + 1, 0);
    for (int mu = 0; mu < NM; mu++) eq_first[mu + 1] = eq_first[mu] + ((unsigned long long)NP << l2i((M[mu].m + 2) / 3));
    al(vt, HT_TAU); // theta^u
    al(vt_point, vt_bits);
    vp = vt;        // the ties weigh the combinations as the reads do
    kh_ones<<<1, 32>>>(wBZ + 2 * 32, 32);
    gamT.assign(NM, nullptr);
    gamP.assign(NM, nullptr);
    wtie.assign(NM, nullptr);
    tieR.assign(NM, nullptr);
    tieX0.assign(NM, 0);
    tieN.assign(NM, 0);
    for (int mu = 0; mu < NM; mu++) {
      al(gamT[mu], M[mu].k);
      gamP[mu] = gamT[mu]; // one vector: the ties weigh the combinations as the reads do
    }
    if (owner) add_foreign(muForeign, owner, muOwner);
    // The virtual rows, in the order in which hw::walk_tail_f() (walk_calls.cuh) expects them.
    {
      auto reg = [&](const std::string &n, hz::TreeN *X, const T *cv) {
        HVRowG r;
        r.nm = n;
        r.T = X;
        r.cv = cv;
        vrows.push_back(r);
        return (int)vrows.size() - 1;
      };
      vrows.clear();
      for (int b = 0; b < nb; b++)
        pc[0][b].row = pc[1][b].row = reg("Z" + std::to_string(b), &ZZ, wZZ + (size_t)b * HT_TAU);
      rowX4 = reg("Z" + std::to_string(nb), &ZZ, vt);
      rowTie = reg("Z" + std::to_string(nb + 1), &ZZ, vp);
      rowBZ0 = reg("BZ0", &BZ, wBZ);
      reg("BZ1", &BZ, wBZ + 32);
      reg("BZ2", &BZ, wBZ + 2 * 32);
      if ((int)vrows.size() != nvZ + 3) {
        printf("internal error: smallness test: the number of virtual rows is not the expected "
               "one\n");
        name_the_end("tail-setup"); // serving: p258 end 1 tail-setup (M5g)
        exit(2);
      }
    }
    seal_list();
    CK(cudaDeviceSynchronize());
  }
  // Builds the range check's entries of the sub-proof's list of committed arrays and sets up each
  // matrix's tie (TH, wtie, tieR, tieN). init() calls it once, after add_foreign().
  void seal_list() {
    list.clear();
    auto addN = [&](hz::TreeN *X, u32 num, u32 tag) {
      HListF e;
      e.T = X;
      e.num = num;
      e.tag = tag;
      list.push_back(e);
    };
    addN(&ZZ, 6, 7);
    addN(&BZ, 9, 10);
    if (TH.empty()) {
      TH.resize(NM);
      hz::Late late; // first written in `make_powers`
      for (int mu = 0; mu < NM; mu++) TH[mu].setup((u32)mu, (u32)M[mu].m, (u32)M[mu].k);
    }
    for (int mu = 0; mu < NM; mu++) {
      pcOf[mu] = mu, tieX0[mu] = 0, tieN[mu] = M[mu].k, tieR[mu] = gamP[mu];
      hz::Late late;
      if (!wtie[mu]) hz::alloc(wtie[mu], sizeof(T) * M[mu].m, hz::FORK);
    }
  }
  // Only checks: exits unless the owner's matrix muOwner is a border matrix of matrix mu's size.
  void add_foreign(int mu, HiddenTail *ow, int muOwner) {
    if (!ow->shr[muOwner]) {
      printf("smallness test: the owner's matrix is not a border matrix, so it has no tree "
             "that another sub-proof can read\n");
      name_the_end("tail-setup"); // serving: p258 end 1 tail-setup (M5g)
      exit(2);
    }
    if (ow->M[muOwner].m != M[mu].m || ow->M[muOwner].k != M[mu].k) {
      printf("smallness test: the owner and the reader of a border matrix disagree on its "
             "number of rows or columns\n");
      name_the_end("tail-setup"); // serving: p258 end 1 tail-setup (M5g)
      exit(2);
    }
  }
  // Appends the randomness stream numbers ("coin streams" in messages) of ZZ and BZ; a number
  // id reserves streams id to id + 3. Prover::plan_stream_ids() (prover/prover/stream_plan.cuh)
  // checks that no two overlap. As offsets from the sub-proof's base, sid0 - 64: ZZ 76, BZ 88.
  void stream_ids(std::vector<std::pair<std::string, u32>> &o) const {
    o.push_back({"ZZ", sid0 + 12});
    o.push_back({"BZ", sid0 + 24});
  }
  size_t hash_words() const { return 16; } // the longest tree-hash input, in words: no long message is hashed here
  // Adds the range check's rounds to the hash chain's plan, as the verifier's HTailF2::planRounds()
  // does. part 0, after the reads' sample: the matrix Cf. part 1, after ZZ's and BZ's sample: the
  // lookup. The equations have no round: plan the batching round with batched_rows(). After init().
  void plan_rounds(blk::Plan &p, const std::string &where, unsigned part) const {
    if (part == 0) {
      size_t columns = 0;
      for (int mu = 0; mu < NM; mu++) columns += M[mu].k;
      if (blk::rng_combinations(p) != (unsigned)HT_TAU) {
        printf("internal error: smallness test: the plan asks for %u combinations, this program was built for %d\n", blk::rng_combinations(p), HT_TAU);
        name_the_end("tail-setup"); // serving: p258 end 1 tail-setup (M5g)
        exit(2);
      }
      blk::plan_rng(p, where, columns);
      return;
    }
    blk::LookupSize x;
    x.v = (unsigned)F.plan.v(), x.d = (unsigned)F.plan.d(), x.mu = 5, x.shifted = F.plan.NC > 1;
    blk::plan_lookup(p, where + ", digits", x);
  }
  // One blk::RangeCheck per matrix for the conditions on hiding (blocks/frame/hiding_conditions.h):
  // the integer matrix of unpacked cells (m x k). A cell is a signed byte, so below 129 in absolute
  // value for every request; a combination must lie in [-2^HT_LBZ, 2^HT_LBZ). After init().
  std::vector<blk::RangeCheck> range_checks() const {
    static_assert(EZD0 + HT_D1 == HT_LBZ + 1, "the two digits represent exactly the window of a combination");
    std::vector<blk::RangeCheck> checks;
    for (int mu = 0; mu < NM; mu++) {
      blk::RangeCheck c;
      c.rows = M[mu].m, c.columns = M[mu].k, c.combinations = HT_TAU, c.bound = 129, c.range = (u64)1 << HT_LBZ;
      c.digit_bits[0] = EZD0, c.digit_bits[1] = HT_D1;
      checks.push_back(c);
    }
    return checks;
  }
  /* Draws the verifier's matrix into M[mu].Cf: Cf[c][u] in {-1, 0, 1} for column c of matrix mu and
     combination u < HT_TAU. The CPU runs this round, between rounds that kernels run. One draw per
     sub-proof, in the order of blocks/iop/rng.h: matrix by matrix, column by column, HT_TAU entries
     per column. Call it only after every message with this sub-proof's packed entries or reads has
     had its sample answered and the reads' weight vectors are fixed: a prover that knows the matrix
     could alter a read in a way that the matrix does not detect. `close_bits` ends the process if
     this is not the plan's next round or a sample's answer is due. Not checked: that the plan puts
     this round after those messages, and that the weight vectors are fixed. */
  void draw_matrix() {
    size_t columns = 0;
    for (int mu = 0; mu < NM; mu++) columns += M[mu].k;
    blk::ProverRounds &channel = ch::host.take_back();
    auto from = channel.close_bits(blk::R_RNG);
    channel.require_combinations(HT_TAU); // before any entry: the plan's error must be for HT_TAU
    const std::vector<signed char> entries = from.trits(columns * HT_TAU);
    size_t at = 0;
    for (int mu = 0; mu < NM; at += M[mu].k * HT_TAU, mu++)
      CK(cudaMemcpyAsync(M[mu].Cf, entries.data() + at, M[mu].k * HT_TAU, cudaMemcpyHostToDevice, cudaStreamPerThread));
    CK(cudaStreamSynchronize(cudaStreamPerThread)); // 'entries' goes out of scope
    ch::host.lend();
  }
  // Given the combinations M[mu].Z, writes the digits and the lookups' multiplicity bits, then
  // commits to ZZ and BZ (ch::moment()). oZZBZ gets DW zero words: ch::host.words() gives zeros.
  void fold_to_group2(Tr *tr, u32 *err, u32 *oZZBZ) {
    const HRows k10 = rows(0), k8 = rows(1);
    for (int mu = 0; mu < NM; mu++)
      kf_zz_plain<<<dim3((unsigned)(M[mu].m / 32), HT_TAU / 32), dim3(32, 8)>>>(M[mu].Z, off[mu], mz, zzPlain, k10, k8,
                                                                                err);
    for (int q = 0; q < NCZ; q++) F.counts(q, kk[q], pubZ);
    F.planes();
    kh_bz<<<nblk(32 * kb), 256>>>(F.bits(1), 32 * kb, F.plan.kT + F.plan.cK, 0, kb, cBZ); // bits(1): in the order (t | c | j)
    ch::moment(chainMoment, {zzPlain, cBZ}, tr);
    CK(cudaMemcpyAsync(oZZBZ, ch::host.words(chainMoment) + ROOTW - DW, 4 * DW,
                       cudaMemcpyDeviceToDevice)); // zeros: see above
  }
  // Runs the lookup forest, cached as a CUDA graph in `graph`: F.plan.nmsg() slots from msg on.
  void group2(Tr *tr, T *msg) { forest_prove_graph(graph, F, tr, msg); }
  // No challenge of the check's own follows the lookup: make_powers() takes powers of theta.
  const T *zk() const { return F.claims->zk[0]; } // the two lookups have one point
  const T *zin(int i) const { return zk() + (n9 - pc[0][i].b); }
  // Fills the coefficient vectors (cv) that depend on the lookup's points. Call after group2().
  void make_cv() {
    for (size_t i = 0; i < pc[0].size(); i++) {
      const HPieceG &p = pc[0][i];
      k_eq<<<1, 128>>>(zin((int)i), l2i(p.Rp), (T *)vrows[p.row].cv + p.r0);
    }
    for (int e = 0; e < 2; e++) k_eq<<<1, 32>>>(F.claims->zb[e], 5, wBZ + e * 32);
  }
  // Allocates `lead` and `lead_exponent` for n claims. hw::walk_tail_f() calls it once.
  void room_for_leads(size_t n) {
    most_leads = n;
    hz::alloc(lead, sizeof(T) * (n + 1), hz::SHAPE);
    hz::alloc(lead_exponent, 8 * (n + 1), hz::SHAPE);
    step = lead + n;
  }
  /* Computes what depends on the batching challenge theta (GPU memory): vt[u] = theta^u, gamT, each
     matrix's Er[i'] = (theta^NP)^i' and wtie, and `lead` for each entry of `leads`. hw::Walk runs
     it as a preparation job once theta is drawn. claims: the number of all its claims. If an entry
     of gamT is 0, `kh_gam` (tail_gam.cuh) sets the flag ZERO_ELEMENT: the request gets no proof. */
  void make_powers(const T *theta, u32 claims) {
    if (leads.size() != most_leads) {
      printf("internal error: smallness test: %zu claims have a lead, room was made for %zu\n", leads.size(), most_leads);
      exit(2);
    }
    kf_tie_powers<<<1, 256>>>(theta, vt_bits, HT_TAU, vt_point, vt); // defined in tail_gam.cuh
    lead_exponent_host.resize(leads.size() + 1);
    for (size_t i = 0; i < leads.size(); i++) lead_exponent_host[i] = (unsigned long long)(claims - leads[i].claim) + leads[i].first;
    lead_exponent_host[leads.size()] = NP;
    CK(cudaMemcpyAsync(lead_exponent, lead_exponent_host.data(), 8 * lead_exponent_host.size(), cudaMemcpyHostToDevice, cudaStreamPerThread));
    kf_powers<<<nblk(lead_exponent_host.size()), 256>>>(theta, lead_exponent, lead_exponent_host.size(), lead);
    for (int mu = 0; mu < NM; mu++)
      kh_gam<HT_TAU><<<nblk(M[mu].k * 32, 256, 1056), 256>>>(M[mu].Cf, M[mu].k, vt, gamT[mu]);
    for (int mu = 0; mu < NM; mu++) {
      TH[mu].powers(step);
      TH[mu].left(wtie[mu]);
    }
  }
};
