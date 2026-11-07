// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// The GPU prover's attention sub-proof (`fork` in identifiers) for layer group g: scores, causal
// softmax, weighted sum. AFront (front.cuh) has the forward pass, cells (`AMat`, plan.cuh) and
// messages; AForkHid adds each sent value's mask entry (`pad`), the commitment to the evaluation
// tables, an approximate range check (`tail`; "smallness test" in messages) and the linear claims.
// One object serves all groups with the same shifts, one at a time: setup(), add_fork() per further
// group, then begin_fork(), phase1(), prove() per sub-proof. The caller sets the private
// randomness's key, starts each attempt, writes PAD_STRIDE to PADOFF, and defines GSL, nblk, k_eq
// before the include. m, ix, Z: a row's largest score, its index, the sum of its weights e.
#pragma once
#include "common/element_from_hash.h"
#include "attn/front.cuh"
#include "attn/plan_rounds.h"
#include "common/zfold_turned.cuh"
#include "hiding/tail.cuh"
#include "hiding/walk.cuh"
#include "hiding/walk_calls.cuh"
#include "hiding/eqs.cuh"
// o[i] = element i (type T: the extension field E) of a row stored as EDEG planes of coordinates.
__global__ void kA_pl2e(const u64 *plane0, long long planes, size_t n, T *o) {
  GSL(i, n) o[i] = hw::at_planes(plane0, planes, (long long)i);
}
// ORs one sub-proof's three error words into words that are kept after the sub-proof has ended.
__global__ void kA_keep(u32 *k, const u32 *a, const u32 *b, const u32 *c) {
  k[0] |= *a;
  k[1] |= *b;
  k[2] |= *c;
}
// Sets *differ to 1 if a and b differ in any of their n 16-byte words; otherwise leaves it
// unchanged. a and b must be 16-byte aligned (uint4 loads): this is not checked.
// Public source: a shared flag set by ordinary stores: Oracle Developer Studio 12.6: Thread
// Analyzer User's Guide, 2017, "Benign Data Races", "A Program that Verifies Array-Value Types",
// https://docs.oracle.com/cd/E77782_01/html/E77800/gecqt.html. That program's threads are CPU
// threads; here they are GPU threads, and none reads the flag inside the kernel. NVIDIA's guides
// on unsynchronized writes to one location are cited at kf_zz_plain() (prover/hiding/tail_f.cuh).
// The idea is public: vectorized loads (uint4): "CUDA Pro Tip: Increase Performance with Vectorized
// Memory Access", a post on NVIDIA's developer blog,
// https://developer.nvidia.com/blog/cuda-pro-tip-increase-performance-with-vectorized-memory-access/.
__global__ void kA_differ(const uint4 *a, const uint4 *b, size_t n, u32 *differ) {
  GSL(i, n) {
    const uint4 x = a[i], y = b[i];
    if ((x.x ^ y.x) | (x.y ^ y.y) | (x.z ^ y.z) | (x.w ^ y.w)) *differ = 1;
  }
}
// Short names (AP, plan.cuh). Piece lam * HA + a: layer lam, query head a of each key-value head.
#define AH_LOCALS                                                                                                      \
  [[maybe_unused]] AP &p = A.P.p;                                                                                      \
  [[maybe_unused]] const int Tn = p.T, H = p.H, G = p.G, HA = p.HA, Lam = p.Lam, NC = A.NC, D = p.D, NP = A.NI,        \
                             lt = p.lt, lg = p.lgk, lB = p.lB, lnT = p.lnT, ldh = p.ldh, ls = p.ls, dh = p.dh,         \
                             LS = 1 << ls, NM = NAs + 4, DR = rank * p.D;                                              \
  [[maybe_unused]] const size_t dq = p.dq, dkv = p.dkv;
// A part of the sub-proof's section of the proof file. The name "+" continues the part before it.
struct ASecD {
  std::string name;
  const void *p;
  size_t words;
};
struct AForkHid {
  AFront A;
  HiddenTail HT;        // the range check; the caller sets HT.chainMoment, its commit point, to 2
  hz::Tree S;           // draws the mask entries ("pad tree" in a message)
  hz::TreeN U3; // the evaluation tables as one row of elements of E; commit point 1 (`moment`)
  he::Eqs EQ;
  hw::Walk W;
  std::vector<MatD> M; // HT's matrices: group g's as_g (NAs parts), ah_g, ao_g, bA_g, qa_g (`AMat`)
  std::vector<u32> gids;
  int NAs = 1, hb = 100;
  // Used only for DR = rank * D (AH_LOCALS), which no function of this file reads.
  int rank = 0;
  size_t poolElems = (size_t)24 << 20; // elements of the walk's pool

  // After prove(): the sub-proof's section, nw words in file order, and its parts.
  u32 *out = nullptr;
  size_t nw = 0;
  std::vector<ASecD> SC;
  // roots has 8 DW + SW words. Only the root of U3 is written to it (DW words from word 2 DW); the
  // other words stay zero, as setupTreesAndBuffers() leaves them.
  u32 *roots = nullptr;

  // Message slot j (below: slot): the j-th masked element in the order of sending; ns = nmsg. First
  // slots: slPT, the lookup forest A.FO (product trees of the sub-proof's lookup calls); slCl, the
  // A.nSend other values that A.run() sends; slPZ, the forest HT.F of the range check's lookups.
  size_t nmsg = 0, ns = 0;
  size_t slPT = 0, slCl = 0, slPZ = 0;

  HiddenTail OW; // a placeholder for qa_g's owner: see `setupTail`
  MatD OM[1];
  int cur = -1;
  u32 gid = 0, base = 0;
  size_t mM[5] = {0, 0, 0, 0, 0}, kM_[5] = {0, 0, 0, 0, 0}; // rows and columns of the matrices, by AMat
  // The evaluation tables, nv3 entries. A table: inner products of a vector phi with runs of cells.
  size_t nv3 = 0;
  T *vec3 = nullptr;
  T *MSG = nullptr; // planes of NSM elements: slots (ns in use), mask entries, sumchecks' scratch
  size_t NSM = 0;
  T *Lall = nullptr, *L3 = nullptr; // W's coefficient vectors on committed arrays: L3 (U3's) first
  std::vector<T *> Lrow;
  size_t nLall = 0;
  T *one1 = nullptr;
  hd::Door door; // the sub-proof's one equation, for `hd::close` (prover/hiding/door.cuh)
  u32 *zsb = nullptr, *sd2 = nullptr;
  u32 *err = nullptr, *errc = nullptr; // this sub-proof's error words: the range check's, A.FO's
  u32 *errK = nullptr;                 // all sub-proofs' flags; none overwrites them: see `errors`
  char *zfold_work = nullptr; // zfold_turned_i8()'s work buffer (common/zfold_turned.cuh)
  int spP0 = 0;              // W's first space for a matrix of M

  // Before the call, fill A.P.p from the parameter file, A.P.sh (one shift per layer) and A.P.form.
  // NAs_: as_g's parts. hb_: `bits` of the trees' set-up. gid0, MIDqa0: add_fork()'s, for handle 0.
  void setup(int NAs_, int hb_, u32 gid0, u32 MIDqa0) {
    NAs = NAs_;
    hb = hb_;
    gid = gid0;
    base = 4u * (4096u + 64u * gid);
    if (A.P.p.D != 2048 || (A.P.form != 1 && A.P.form != 3)) {
      printf("attention: this prover supports only the ring degree D = 2,048 and attention call lists of form 1 or "
             "3\n");
      exit(2);
    }
    A.plan();
    layoutMatrices();
    setupTail(gid0, MIDqa0);
    countSlots();
    setupTreesAndBuffers();
    buildWalk();
    feedEqs();
    layoutProof();
  }

private:
  // Rounds each of the five matrices (AMat) up to full columns (as_g's number of columns to a
  // multiple of NAs), allocates the cells and builds M.
  void layoutMatrices() {
    AH_LOCALS {
      const size_t mm[5] = {(size_t)1 << p.lmS, (size_t)Tn << p.lwH, (size_t)Tn << p.lwO, (size_t)1 << p.lmb,
                            (size_t)Tn << p.lwQ};
      for (int i = 0; i < 5; i++) mM[i] = mm[i];
    }
    for (int i = 0; i < 5; i++) {
      kM_[i] = (A.ncell[i] + mM[i] - 1) / mM[i];
      if (i == 0) kM_[i] = (kM_[i] + NAs - 1) / NAs * NAs;
      A.zero_fill[i] = kM_[i] * mM[i] - A.ncell[i];
      A.ncell[i] = kM_[i] * mM[i];
      if (mM[i] % D) {
        printf("attention: the column length of a committed matrix (2^lmS, 2^lmh, 2^lmo, 2^lmbA or 2^lmq) is not a "
               "multiple of the ring degree D\n");
        exit(2);
      }
    }
    if (p.lmb < 5 || p.lmb > A.FO.plan.kT + 5) {
      printf(
          "policy: forest-planes: 5 <= lmbA <= kT + 5 must hold, where the largest table of attention's lookup calls "
          "has 2^kT rows; lmbA = %d, kT = %d\n",
          p.lmb, A.FO.plan.kT);
      exit(2);
    }
    A.allocate();
    M.assign(NM, MatD());
    {
      const size_t kp = kM_[0] / NAs;
      for (int x = 0; x < NAs; x++) {
        M[x].m = mM[0];
        M[x].k = kp;
        M[x].v = A.cell[A_AS] + (size_t)x * kp * mM[0];
      }
      const int src[4] = {1, 2, 3, 4};
      for (int x = 0; x < 4; x++) {
        M[NAs + x].m = mM[src[x]];
        M[NAs + x].k = kM_[src[x]];
        M[NAs + x].v = A.cell[src[x]];
      }
    }
    for (int mu = 0; mu < NM; mu++) {
      hz::Late late; // HT's coefficients Cf, CfT, combinations Z: used only after `hz::late_hook`
      hz::alloc(M[mu].Z, M[mu].m * HT_TAU * 4, hz::FORK);
      M[mu].Cf = dalA<i8>(M[mu].k * ETAU);
      M[mu].CfT = dalA<i8>((size_t)zfold_kp((int)M[mu].k) * ETAU);
    }
    {
      hz::Late late;
      zfold_work = dalA<char>(ZFOLD_TURNED_WORKSPACE);
    }
  }
  // The group's layer sub-proof commits to qa_g. OW, a placeholder for that owner, has qa_g's size
  // and border flag (`shr`) 1, so HT.init()'s two checks on the owner (add_foreign()) cannot fail.
  void setupTail(u32 gid0, u32 MIDqa0) {
    AH_LOCALS {
      OM[0] = M[NM - 1];
      OW.M = OM;
      OW.shr = {1};
    }
    gids.clear();
    add_fork(gid0, MIDqa0);
    {
      std::vector<int> cls(NM, 0);
      cls[NM - 1] = 1;
      HT.init(M.data(), NM, cls.data(), nullptr, hb, base + 64, &OW, NM - 1, 0);
    }
  }
  // Assigns the slots, in the order of sending.
  void countSlots() {
    AH_LOCALS;
    // SCD (a_scd(), engines.cuh: a sumcheck of varying round degree) needs lt <= ga::SCD_MOST_LT,
    // and its 2 lt + lg challenges must fit A.rps (64 elements). The check below is stricter.
    static_assert(ga::SCD_MOST_LT >= 14, "the check below admits lt = 14");
    if (lt > 14 || 2 * lt + lg > 32) {
      printf("attention: this prover needs lt <= 14 (at most 2^14 tokens) and 2 lt + lg <= 32, where 2^lg is the "
             "number of key-value heads; this parameter file has lt = %d and lg = %d\n",
             lt, lg);
      exit(2);
    }
    size_t s = 0;
    slPT = s;
    s += A.FO.plan.nmsg(); // the slots of the lookup forest A.FO
    slCl = s;
    s += A.nSend;
    slPZ = s;
    s += HT.F.plan.nmsg();
    nmsg = s;
  }
  // Sizes and allocates the trees U3 and S and the device scratch buffers, after countSlots(). The
  // caller repeats these memsets, except errK's, before each sub-proof: ForkMemory::run_begin().
  void setupTreesAndBuffers() {
    AH_LOCALS nv3 = A.nWe;
    vec3 = dalA<T>(nv3);
    U3.setup("U3", 1, nv3, EDEG, 1, base + 12, hb);
    // EQ.list: U3, then the range check's arrays ZZ and BZ.
    {
      {
        he::Ent e;
        e.Tn = &U3;
        e.num = 2;
        e.tag = 15;
        EQ.list.push_back(e);
      }
      for (auto &l : HT.list) {
        he::Ent e;
        e.Tn = l.T;
        e.num = l.num;
        e.tag = l.tag;
        EQ.list.push_back(e);
      }
    }
    ns = nmsg;
    S.setup("S_A", 2, 0, ns, 1, 2, base, hb);
    // roots, zsb and `out` (layoutProof()) are sent, whole or in part: they are zeroed so that
    // earlier contents of this memory do not leave the program through them.
    roots = dalA<u32>(8 * DW + SW);
    CK(cudaMemset(roots, 0, (8 * DW + SW) * sizeof(u32)));
    // zsb has room for four digests.
    zsb = dalA<u32>(4 * DW);
    CK(cudaMemset(zsb, 0, 4 * DW * sizeof(u32)));
    err = dalA<u32>(1);
    CK(cudaMemset(err, 0, 4));
    errc = dalA<u32>(1);
    CK(cudaMemset(errc, 0, 4));
    sd2 = dalA<u32>(2 * SW);
    hz::alloc(errK, 16, hz::OWN);
    CK(cudaMemset(errK, 0, 16));
    NSM = PAD_STRIDE;
    if (S.sh.Kc > NSM) {
      printf("attention: the fork's pad tree needs %zu coefficients; this build holds at most %zu\n", S.sh.Kc, NSM);
      exit(2);
    }
    // The three planes (prover/hiding/psc_device.cuh); a build with SLOT_COUNT=1 adds its counters.
    MSG = dalA<T>((3 + SLOT_COUNT) * NSM);
    CK(cudaMemset(MSG, 0, (3 + SLOT_COUNT) * NSM * sizeof(T)));
  }
  // Builds W ("the walk" in messages): every linear claim as an affine form (hw::AF). Variables, by
  // space: 0, the slots' hidden values; 3, vec3; 5 + r, virtual row r (sum_i cv[i] X[i][.], X = ZZ
  // or BZ); spP0 + mu, M[mu]'s columns. Claim c gets the factor theta^c, so the order matters.
  void buildWalk() {
    using namespace hw;
    AH_LOCALS {
      hz::Late late; // W's pool: apart from set-up's writes, filled and read after `hz::late_hook`
      W.init(poolElems);
    }
    W.scratch = MSG + 2 * NSM; // the sumchecks' scratch plane
    // Space 5 + nrows stays empty: in every sub-proof the matrices' spaces begin at 6 + nrows.
    const int nrows = (int)HT.vrows.size();
    W.sp.resize(6 + nrows);
    // The coefficients for hd::close() (prover/hiding/door.cuh): door.LwM has one for each column
    // of each matrix, door.LS0 one for each slot's mask entry. LS0 has door.nC = 1 further entry,
    // LS0[ns], to which no claim of attention adds: it stays 0.
    spP0 = 6 + nrows;
    W.sp.resize(6 + nrows + HT.NM);
    {
      hz::Late late;
      door.ns = ns, door.nLwM = 0;
      door.atM.assign(HT.NM, 0);
      for (int mu = 0; mu < HT.NM; mu++) door.atM[mu] = door.nLwM, door.nLwM += HT.M[mu].k;
      door.LwM = dalA<T>(door.nLwM);
      door.LS0 = dalA<T>(ns + door.nC);
      for (int mu = 0; mu < HT.NM; mu++) {
        Space &z = W.sp[spP0 + mu];
        z.lam = door.LwM + door.atM[mu];
        z.len = HT.M[mu].k;
      }
      W.sp[0].lam = door.LS0;
    }
    {
      Space &z = W.sp[0];
      z.len = ns;
      z.neg = 1;
    }
    nLall = nv3;
    for (auto &x : HT.vrows) nLall += x.T->K;
    {
      hz::Late late; // W's coefficient vectors, in one array: filled and read after `hz::late_hook`
      Lall = dalA<T>(nLall);
    }
    {
      T *x = Lall;
      L3 = x;
      x += nv3;
      for (auto &y : HT.vrows) {
        Lrow.push_back(x);
        x += y.T->K;
      }
    }
    { // Space 3 is vec3. Spaces 1, 2 and 4 stay empty: `walk_tail_f` expects the virtual rows from
      // space 5 on.
      Space &z = W.sp[3];
      z.lam = L3;
      z.len = nv3;
    }
    for (int r = 0; r < nrows; r++) {
      Space &z = W.sp[5 + r];
      z.lam = Lrow[r];
      z.len = HT.vrows[r].T->K;
    }
    std::vector<HE> p256v(4);
    p256v[0] = hE(1);
    for (int i = 1; i < 4; i++) p256v[i] = hmul(p256v[i - 1], hE(256));
    // Offsets of the tables in U3's vector vec3, in this order: as_g, ah_g, ao_g, qa_g, bA_g.
    const size_t oAS = 0, oAH = oAS + A.nAS * A.kAS, oAO = oAH + A.nAH * A.eAH, oQA = oAO + A.nAO * A.eAO,
                 oBA = oQA + A.nQA * A.eQA;
    const int nKR = (int)A.psKR.size(), nKF = (int)A.psKF.size(), qE = 2 * nKF;
    auto insOf = [&](const AFront::PS &x) {
      std::vector<int> in;
      for (auto &y : A.P.C[x.ci].pc)
        if (y.seg == x.seg) in.push_back(y.lam * HA + y.a);
      std::sort(in.begin(), in.end());
      return in;
    };
    // as_g's table a (0 TA, 1 TA_VM, 2 TA_VD): plane pl of piece `in`; dg: of the diagonal tiles.
    auto planeA = [&](int a, bool dg, int pl, int in, const T *eh, Sy w) {
      const int n0 = dg ? p.nD : p.nM;
      const size_t b0 = (dg ? (((size_t)(5 * NP)) << p.nM) >> p.lmS : 0) + (((size_t)(pl * NP + in)) << (n0 - p.lmS));
      return rng(3, oAS + (size_t)a * A.kAS + b0, 1, (size_t)1 << (n0 - p.lmS), eh, 1, w);
    };
    auto HvalG = [&](int fi, int X, int in, const T *eg, Sy w) {
      AF o;
      for (int l = 0; l < 4; l++)
        o = o + rng(3, oAH + (size_t)fi * A.eAH + (size_t)X * 4 * Lam * H + (size_t)(l * Lam + in / HA) * H + in % HA,
                    HA, G, eg, 1, w * Sy(p256v[l]));
      return o;
    };
    auto ibG = [&](int fi, int in, int kp, const T *eg, Sy w) {
      return rng(3,
                 oAH + (size_t)fi * A.eAH + (size_t)8 * Lam * H + (size_t)(in / HA) * lt * H + (size_t)kp * H + in % HA,
                 HA, G, eg, 1, w);
    };
    auto XLqG = [&](int fi, int X, int lam, int a, const T *eg, const T *hi, Sy w) {
      const bool qh = X == 0 || X == 3;
      const size_t JQ = qh ? dq : dkv, OF = X == 0   ? 0
                                            : X == 1 ? 2 * Lam * dq
                                            : X == 2 ? 2 * Lam * (dq + dkv)
                                                     : 2 * Lam * (dq + 2 * dkv);
      AF o;
      for (int l = 0; l < 2; l++)
        for (int ch = 0; ch < dh / LS; ch++)
          o = o +
              rng(3,
                  oQA + (size_t)fi * A.eQA + (OF + (size_t)(l * Lam + lam) * JQ + (qh ? (size_t)a * dh : 0)) / LS + ch,
                  (long long)(qh ? HA : 1) * dh / LS, G, eg, 1, w * Sy(p256v[l]) * Sy(hi + ch));
      return o;
    };
    auto XLoG = [&](int fi, int X, int in, const T *eg, const T *hi, Sy w) {
      AF o;
      for (int ch = 0; ch < dh / LS; ch++)
        o = o + rng(3, oAO + (size_t)fi * A.eAO + ((size_t)(X * Lam + in / HA) * dq + (size_t)(in % HA) * dh) / LS + ch,
                    (long long)HA * dh / LS, G, eg, 1, w * Sy(hi + ch));
      return o;
    };
    auto rounds_of = [](const T *r, int n) { // the address of each round's challenge, in order
      std::vector<const T *> v;
      for (int j = 0; j < n; j++) v.push_back(r + j);
      return v;
    };
    // SC3: the masked sumcheck (PSC) of sum_b W_b A_b B_b over nb pairs. Registers one (first slot
    // s0, challenges r, claimed sum `start`) and adds the claim it leaves. The W_b enter no claim:
    // the verifier applies them. End values A*_b, B*_b: W.shown(R, 2 b), W.shown(R, 2 b + 1).
    auto sc3obj = [&](size_t s0, const T *r, int nv, int nb, const AF &start) {
      const Rec R = W.psc(2, rounds_of(r, nv), s0, 2, 2 * (size_t)nb, nullptr);
      W.row(R, start);
      return R;
    };
    // The claims of the sub-proof's lookup forest, A.FO.
    const ForestForms forest = walk_forest(W, A.FO, slPT);
    // KF[c]: lookup call c's key vector at its point A.zkp(c), a form in the slots less public
    // constants. All calls enter one claim, the join. Lookup segment: a call's named list of keys.
    std::vector<AF> KF(NC);
    std::vector<char> joined(NC, 0);
    auto join = [&](int c, const AF &right_side) {
      KF[c] = right_side;
      joined[c] = 1;
    };
    size_t sl = slCl;
    // Lookup segments RM, RD (lookup tables KR): each has 3 slots per call, as_g's planes 0 to 2.
    for (int a = 0; a < nKR;) {
      const int ci = A.psKR[a].ci;
      AF rhs;
      for (; a < nKR && A.psKR[a].ci == ci; a++) {
        AFront::PS &x = A.psKR[a];
        const bool dg = x.seg == A_RD;
        auto in = insOf(x);
        for (int pl = 0; pl < 3; pl++) {
          AF f = slot(sl + pl);
          for (int b = 0; b < x.cnt; b++) f = f - planeA(0, dg, pl, in[b], A.ehAS[a], Sy(x.wp + b));
          W.lin(f);
        }
        rhs = rhs + slot(sl) + slot(sl + 1, Sy(h2(20))) + slot(sl + 2, Sy(h2(40)));
        sl += 3;
      }
      join(ci, rhs);
    }
    // Lookup segments FM and FD: five planes, m, the scores' sum, then the SC3 on q k for that sum.
    for (int f = 0; f < nKF;) {
      const int ci = A.psKF[f].ci;
      AF rhs;
      for (; f < nKF && A.psKF[f].ci == ci; f++) {
        AFront::PS &x = A.psKF[f];
        const bool dg = x.seg == A_FD;
        const int a = nKR + f, nv = dg ? A.nvFD : A.nvFM;
        auto in = insOf(x);
        const T *rr = A.kx[f].rr;
        for (int pl = 0; pl < 5; pl++) {
          AF g = slot(sl + pl);
          for (int b = 0; b < x.cnt; b++) g = g - planeA(0, dg, pl, in[b], A.ehAS[a], Sy(x.wp + b));
          W.lin(g);
        }
        {
          AF g = slot(sl + 5);
          for (int b = 0; b < x.cnt; b++) g = g - HvalG(f, 0, in[b], A.kx[f].eg, Sy(x.wp + b));
          W.lin(g);
        }
        const Rec R = sc3obj(sl + 7, rr, nv, x.cnt, slot(sl + 6));
        const T *erg = W.eq(rr, lg), *hi = W.eq(rr + nv - ldh, ldh - ls);
        for (int b = 0; b < x.cnt; b++) {
          W.lin(W.shown(R, 2 * b) - XLqG(2 * f, 0, in[b] / HA, in[b] % HA, erg, hi, Sy()));
          W.lin(W.shown(R, 2 * b + 1) - XLqG(2 * f + 1, 1, in[b] / HA, 0, erg, hi, Sy()));
        }
        rhs = rhs + slot(sl + 5) - slot(sl + 6) - slot(sl) - slot(sl + 1, Sy(hE(256))) + slot(sl + 2, Sy(h2(32))) +
              slot(sl + 3, Sy(h2(33))) + slot(sl + 4, Sy(h2(41)));
        sl += 7 + AFront::sc3n(nv, x.cnt);
      }
      join(ci, rhs);
    }
    // Lookup segments WL, IX, WH. Slots s5 on: acc = sum e v, Z o, Z, RH, RHP. sIB on: ix's bits.
    T *zsl = A.zkp(A.cR) + A.psW.n - A.psW.b, *zgW = zsl + 1, *ziW = zgW + lg, *zcW = ziW + lt,
      *zgX = A.zkp(A.cX) + A.psX.n - A.psX.b, *zsH = A.zkp(A.cH) + A.psH.n - A.psH.b, *zgH = zsH + (p.wd == 4 ? 1 : 0);
    const bool w4 = p.wd == 4;
    // RH, RHP: third bytes of the remainder rem and of 2 Z - 1 - rem (wd = 4: fourth too). WH
    // (sH2 on) shares WL's channel coordinates and table TO, not its weights or head coordinates.
    const T *egW = W.eq(zgW, lg), *egX = W.eq(zgX, lg), *egH = W.eq(zgH, lg), *hiW = W.eq(zcW, ldh - ls);
    auto inW = insOf(A.psW), inX = insOf(A.psX), inH = insOf(A.psH);
    const size_t s5 = sl, sIB = s5 + (w4 ? 7 : 5), sH2 = sIB + lt;
    {
      AF zc = slot(s5 + 2), rh = slot(s5 + 3), rhp = slot(s5 + 4);
      for (int b = 0; b < A.psW.cnt; b++) {
        const Sy w(A.psW.wp + b);
        zc = zc - HvalG(nKF, 1, inW[b], egW, w);
        rh = rh - XLoG(0, 0, inW[b], egW, hiW, w);
        rhp = rhp - XLoG(0, 1, inW[b], egW, hiW, w);
      }
      W.lin(zc);
      W.lin(rh);
      W.lin(rhp);
      if (w4) {
        AF rg_ = slot(s5 + 5), rgp_ = slot(s5 + 6);
        for (int b = 0; b < A.psW.cnt; b++) {
          const Sy w(A.psW.wp + b);
          rg_ = rg_ - XLoG(0, 2, inW[b], egW, hiW, w);
          rgp_ = rgp_ - XLoG(0, 3, inW[b], egW, hiW, w);
        }
        W.lin(rg_);
        W.lin(rgp_);
      }
      for (int kp = 0; kp < lt; kp++) {
        AF a = slot(sIB + kp);
        for (int b = 0; b < A.psX.cnt; b++) a = a - ibG(nKF + 1, inX[b], kp, egX, Sy(A.psX.wp + b));
        W.lin(a);
      }
      AF a = slot(sH2), b2 = slot(sH2 + 1);
      for (int b = 0; b < A.psH.cnt; b++) {
        const Sy w(A.psH.wp + b);
        a = a - XLoG(0, 0, inH[b], egH, hiW, w);
        b2 = b2 - XLoG(0, 1, inH[b], egH, hiW, w);
      }
      W.lin(a);
      W.lin(b2);
      if (w4) {
        AF c2 = slot(sH2 + 2), d2 = slot(sH2 + 3);
        for (int b = 0; b < A.psH.cnt; b++) {
          const Sy w(A.psH.wp + b);
          c2 = c2 - XLoG(0, 2, inH[b], egH, hiW, w);
          d2 = d2 - XLoG(0, 3, inH[b], egH, hiW, w);
        }
        W.lin(c2);
        W.lin(d2);
      }
      AF REM = slot(s5, Sy(hE(2))) + slot(s5 + 2) - slot(s5 + 1, Sy(hE(2))), t0 = REM - slot(s5 + 3, Sy(h2(16))),
         t1 = slot(s5 + 2, Sy(hE(2))) - REM - slot(s5 + 4, Sy(h2(16)));
      if (w4) {
        t0 = t0 - slot(s5 + 5, Sy(h2(24)));
        t1 = t1 - slot(s5 + 6, Sy(h2(24)));
      }
      AF kW = t0 + (t1 - t0) * Sy(zsl), kX;
      for (int kp = 0; kp < lt; kp++) kX = kX - slot(sIB + kp, Sy(h2(kp + 16 - lt)));
      if (A.cR == A.cX)
        join(A.cR, kW + kX);
      else {
        join(A.cR, kW);
        join(A.cX, kX);
      }
      {
        AF j0 = slot(sH2) + slot(sH2 + 1, Sy(h2(20)));
        if (w4) {
          AF j1 = slot(sH2 + 2) + slot(sH2 + 3, Sy(h2(20)));
          j0 = j0 + (j1 - j0) * Sy(zsH);
        }
        join(A.cH, j0);
      }
    }
    sl = sH2 + (w4 ? 4 : 2);
    {
      const Rec R = sc3obj(sl, A.rrE, A.nvE, NP, slot(s5 + 1));
      const T *erg = W.eq(A.rrE, lg);
      for (int b = 0; b < NP; b++) {
        W.lin(W.shown(R, 2 * b) - HvalG(nKF + 2, 1, inW[b], erg, Sy()));
        W.lin(W.shown(R, 2 * b + 1) - XLqG(qE, 3, inW[b] / HA, inW[b] % HA, erg, hiW, Sy()));
      }
      sl += AFront::sc3n(A.nvE, NP);
    }
    // The join (forest_join(), prover/hiding/walk_forest.cuh): sum_c e_c KF[c] + `cend` is public.
    for (int c = 0; c < NC; c++)
      if (!joined[c]) {
        printf("attention: internal error: af-unclosed-call: lookup call %s has not entered the join\n",
               A.P.C[c].name.c_str());
        name_the_end("af-unclosed-call"); // serving: p258 end 1 af-unclosed-call (kept as an exit(4))
        exit(4);
      }
    W.lin(forest_join(A.FO, forest, KF));
    // The value product: its two sums (s6, s6 + 1: off and on the diagonal) add up to acc + gam Z.
    const size_t s6 = sl;
    W.lin(slot(s6) + slot(s6 + 1) - slot(s5) - slot(s5 + 2, Sy(A.gam)));
    sl += 2;
    for (int dg = 0; dg < 2; dg++) {
      const int nv = dg ? A.nvD : A.nvM, n0 = dg ? p.nD : p.nM;
      const T *rr = dg ? A.rrD : A.rrM, *pt = dg ? A.ptD : A.ptM;
      const Rec R = sc3obj(sl, rr, nv, NP, slot(s6 + dg));
      const T *erg = W.eq(rr, lg), *eh = W.eq(pt, n0 - p.lmS);
      for (int b = 0; b < NP; b++) {
        W.lin(W.shown(R, 2 * b) - planeA(1 + dg, dg, 3, inW[b], eh, Sy()) -
              planeA(1 + dg, dg, 4, inW[b], eh, Sy(hE(256))));
        W.lin(W.shown(R, 2 * b + 1) - XLqG(qE + 1 + dg, 2, inW[b] / HA, 0, erg, hiW, Sy()));
      }
      sl += AFront::sc3n(nv, NP);
    }
    // m is the score at index ix, and ix's bits are bits.
    {
      T *tg = A.tpts;
      const T *etg = W.eq(tg, lg);
      const size_t s7 = sl;
      AF st = rng(0, s7, 1, NP, A.om);
      for (int in = 0; in < NP; in++) W.lin(slot(s7 + in) - HvalG(nKF + 3, 0, in, etg, Sy()));
      sl += NP;
      const T *rg = A.rF + lt, *rc = rg + lg;
      const Rec RF = sc3obj(sl, A.rF, A.nvF, NP, st);
      const T *erg = W.eq(rg, lg), *hiF = W.eq(rc, ldh - ls);
      for (int in = 0; in < NP; in++) W.lin(W.shown(RF, 2 * in) - XLqG(qE + 3, 0, in / HA, in % HA, erg, hiF, Sy()));
      sl += AFront::sc3n(A.nvF, NP);
      // SCD starts from sum_b om4_b B*_b, the B*_b of the SC3 before it: these vectors are not
      // committed. End values: KC_lam (number lam), ix's bits IBP_(b,k) (number Lam + b lt + k).
      // The verifier multiplies out the summand, so the product of lt + 1 factors needs no claim.
      std::vector<int> degrees;
      for (int q = 0; q < 2 * lt + lg; q++) degrees.push_back(ga::scd_degree(lt, lg, q));
      const Rec RD =
          W.psc_degrees(degrees, rounds_of(A.rps, 2 * lt + lg), sl, lt + 1, (size_t)Lam + (size_t)NP * lt, nullptr);
      AF startD;
      for (int in = 0; in < NP; in++) startD = startD + W.shown(RF, 2 * in + 1) * Sy(A.om4 + in);
      W.row(RD, startD);
      const T *epg = W.eq(A.rps + lt, lg);
      for (int lam = 0; lam < Lam; lam++) W.lin(W.shown(RD, lam) - XLqG(qE + 4, 1, lam, 0, epg, hiF, Sy()));
      for (int in = 0; in < NP; in++)
        for (int kp = 0; kp < lt; kp++) W.lin(W.shown(RD, Lam + in * lt + kp) - ibG(nKF + 4, in, kp, epg, Sy()));
      sl += a_scd_nout(lt, lg, Lam, NP);
      const T *rg2 = A.rrH + lt, *rk = rg2 + lg;
      const Rec RH = sc3obj(sl, A.rrH, A.nvH, NP, AF());
      const T *erg2 = W.eq(rg2, lg), *erk = W.eq(rk, 4);
      for (int in = 0; in < NP; in++) {
        W.lin(W.shown(RH, 2 * in + 1) - W.shown(RH, 2 * in));
        AF a = W.shown(RH, 2 * in);
        for (int kp = 0; kp < lt; kp++) a = a - ibG(nKF + 5, in, kp, erg2, Sy(erk + kp));
        W.lin(a);
      }
      sl += AFront::sc3n(A.nvH, NP);
    }
    if (sl != slCl + A.nSend) {
      printf("attention: internal error: the walk reads %zu slots for the claims A-3 to A-8, but steps 6 to 22 send "
             "%zu\n",
             sl - slCl, A.nSend);
      exit(4);
    }
    // The forest's three claims on its multiplicity planes, read from bA_g's tables TB_0, TB_1, TP.
    walk_forest_planes(W, A.FO, forest, 3, oBA, oBA + A.kB, oBA + 2 * A.kB, p.lmb);
    // The tables for the range check's reads (hw::TableRead), in vec3's order.
    std::vector<TableRead> HFUN;
    {
      const size_t kp = kM_[0] / NAs;
      for (int a = 0; a < A.nAS; a++)
        for (int x = 0; x < NAs; x++) HFUN.push_back({x, mM[0], A.phiAS[a], oAS + (size_t)a * A.kAS + (size_t)x * kp});
      for (int i = 0; i < A.nAH; i++)
        HFUN.push_back({NAs, (size_t)Tn,
                        i == nKF       ? A.eiW
                        : i == nKF + 1 ? A.eiX
                                       : A.phiAH[i],
                        oAH + (size_t)i * A.eAH});
      for (int i = 0; i < A.nAO; i++) HFUN.push_back({NAs + 1, (size_t)1 << p.la, A.phiAO[i], oAO + (size_t)i * A.eAO});
      for (int i = 0; i < A.nQA; i++) HFUN.push_back({NAs + 3, (size_t)1 << p.la, A.phiQA[i], oQA + (size_t)i * A.eQA});
      for (int i = 0; i < 3; i++) HFUN.push_back({NAs + 2, mM[3], A.phiBA[i], oBA + (size_t)i * A.kB});
    }
    // The range check's claims, on its virtual rows.
    walk_tail_f(W, HT, spP0, slPZ, HFUN);
    W.seal();
  }
  // Adds each committed array's coefficient vectors to its entry of EQ.list: for U3 the pair
  // (1, L3); for ZZ and BZ a pair (cv, Lrow[r]) for each virtual row r, with the columns that its
  // claims read. Then lists the terms of the sub-proof's one equation for hd::close().
  void feedEqs() {
    using namespace hw;
    const int nrows = (int)HT.vrows.size();
    {
      std::vector<HE> o1(1, hE(1));
      one1 = (T *)W.cvec(o1);
    }
    for (auto &e : EQ.list) {
      if (e.Tn == &U3) {
        e.cv.push_back(one1);
        e.L.push_back(L3);
        e.c0.push_back(0);
        e.c1.push_back(nv3);
        e.vecE = vec3;
      }
      for (int r = 0; r < nrows; r++)
        if (HT.vrows[r].T == e.Tn) {
          e.cv.push_back(HT.vrows[r].cv);
          e.L.push_back(Lrow[r]);
          const Space &z = W.sp[5 + r];
          e.c0.push_back(z.lo < z.hi ? z.lo : 0);
          e.c1.push_back(z.lo < z.hi ? z.hi : 0);
          for (auto &q_ : z.segs()) {
            e.sv.push_back((int)e.L.size() - 1);
            e.s0.push_back(q_.first);
            e.s1.push_back(q_.second);
          }
        }
    }
    door.add_eqs(EQ);
    HT.say_forms(door);  // marks the range check's row vectors as product forms
    door.plain_rows("ZZ", HT.zzPlain, 2 * HT.mz);
    door.plain_rows("BZ", HT.cBZ, HT.kb);
    for (int mu = 0; mu < HT.NM; mu++) door.add_matrix(HT.TH[mu], HT.M[mu].v, HT.cls[mu] != 0);
  }
public:
  // Adds to the plan p the hash chain's rounds after commit point `moment` and its samples, in
  // prove()'s order: 0, the cells; 1, the tables; 2, the range check's digits. The same for every
  // group of this object. Call after setup(). Verifier: AttentionSubProof::rounds (verifier/src/vf_attention_check.h).
  void plan_rounds(blk::Plan &p, unsigned moment) const {
    const std::string where = "attention";
    blk::AttentionRounds x;
    x.lt = (unsigned)A.P.p.lt, x.lgk = (unsigned)A.P.p.lgk, x.pairs = (unsigned)A.NI, x.Lam = (unsigned)A.P.p.Lam;
    x.plane_messages = (unsigned)A.psKR.size();
    for (const auto &s : A.psKF) x.score.push_back({(unsigned)(s.seg == A_FD ? A.nvFD : A.nvFM), (unsigned)s.cnt});
    x.e = (unsigned)A.nvE, x.value_m = (unsigned)A.nvM, x.value_d = (unsigned)A.nvD, x.attained = (unsigned)A.nvF;
    x.bits = (unsigned)A.nvH;
    // The powers of the last round's challenge batch W's claims, then the range check's equations.
    // HT.leads: one entry per matrix (a claim ties Z to its cells), one per table and run of rows.
    x.rows = W.nclaim + HT.equations(HT.leads.size() - (size_t)HT.NM);
    if (moment == 0) {
      blk::LookupSize lookup;
      lookup.v = (unsigned)(A.FO.plan.dT + A.FO.plan.rK), lookup.d = (unsigned)A.FO.plan.d(), lookup.mu = 5;
      lookup.shifted = A.FO.plan.NC > 1;
      blk::plan_lookup(p, where + ", look-up", lookup);
      blk::plan_attention_own(p, where, x);
    }
    if (moment >= 1) HT.plan_rounds(p, where + ", smallness test", moment - 1); // its matrix; then the rest
    if (moment == 2) blk::plan_attention_end(p, where, x);
  }

private:
  // Lists the parts of the sub-proof's section in file order.
  void layoutProof() {
    AH_LOCALS hz::Late late; // the assembled section
    SC.clear();
    SC.push_back({"AROOTS", nullptr, (size_t)0}); // a section of no words
    SC.push_back({"FOREST", MSG + slPT, EWORDS * A.FO.plan.nmsg()});
    SC.push_back({"closes", MSG + slCl, EWORDS * A.nSend});
    SC.push_back({"ROOT-U3", roots + 2 * DW, DW});
    SC.push_back({"ROOT-ZZ-BZ", zsb, DW});
    SC.push_back({"FOREST-9", MSG + slPZ, EWORDS * HT.F.plan.nmsg()});
    SC.push_back({"XR", nullptr, (size_t)0});
    SC.push_back({"HCOLS", nullptr, (size_t)0});
    nw = 0;
    for (auto &x : SC) nw += x.words;
    out = dalA<u32>(nw);
    CK(cudaMemset(out, 0, nw * sizeof(u32)));
  }

public:
  // The caller makes treehash()'s scratch fit inputs of this many words. Nothing here hashes them.
  size_t hash_words() const { return ENJ * HT.mz; }
  // Registers a further sub-proof with the number gid_, from which begin_fork() derives its
  // randomness streams, and returns its handle for begin_fork(). MIDqa is not used. Exits (code 2)
  // at a 65th sub-proof or if gid_ is above 61.
  int add_fork(u32 gid_, u32 MIDqa) {
    AH_LOCALS const int i = (int)gids.size();
    if (i == 64) {
      printf("attention: add_fork: one object supports at most 64 sub-proofs\n");
      exit(2);
    }
    // Numbers above 61 are not given out.
    if (gid_ >= 62) {
      printf("attention: add_fork: sub-proof numbers go up to 61; this one is %u\n", gid_);
      exit(2);
    }
    gids.push_back(gid_);
    return i;
  }
  // Starts the sub-proof with handle i: sets its randomness streams from gid. Start each attempt
  // with hz::set_attempt() and a larger counter: two proofs under one pad would reveal the pad.
  void begin_fork(int i) {
    AH_LOCALS cur = i;
    gid = gids[i];
    base = 4u * (4096u + 64u * gid);
    auto tn = [&](hz::TreeN &t, u32 s_) {
      t.stream = s_;
      t.I.stream = s_;
    };
    S.stream = base;
    tn(U3, base + 12);
    const u32 sid = base + 64;
    HT.sid0 = sid;
    // The same stream offsets as in HiddenTail::init() and stream_ids() (prover/hiding/tail_f.cuh).
    tn(HT.ZZ, sid + 12);
    tn(HT.BZ, sid + 24);
  }
  // Runs only A.phase1(): the forward pass with all cells and lookup keys, A.FO's multiplicities
  // and planes. Uses no private randomness, needs no begin_fork(), waits for the device. Returns 0
  // or: bit 0, a value did not fit its cells; bit 2, A.FO's error word (always 0).
  u32 forward_and_counts(const i16 *q, const i16 *k, const i16 *v) {
    A.phase1(q, k, v);
    CK(cudaDeviceSynchronize());
    u32 value = 0, counts = 0;
    CK(cudaMemcpy(&value, A.err, 4, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(&counts, &A.FO.table.et.st->err, 4, cudaMemcpyDeviceToHost));
    return (value ? 1u : 0u) | (counts ? 4u : 0u);
  }
  // Runs the forward pass (A.phase1()) and draws the slots' pads. q is [lam][t][H dh], and k and
  // v are [lam][t][G dh]: int16, after RoPE and clamp, valid until prove() returns. Output o: A.o.
  void phase1(const i16 *q, const i16 *k, const i16 *v) {
    AH_LOCALS A.phase1(q, k, v);
    S.slot_pads(door.ns);
  }
  // After A.phase1(): sets bit 3 of errors() if A.cell[A_QA], which the messages use, differs from
  // owners_cells: the owner's qa_g copied after its commitment (device, same size, 16-byte aligned;
  // both unmasked). A verifier rejects unless A.cell[A_QA] is what was committed (not read here).
  void compare_qa_with(const i8 *owners_cells) {
    kA_differ<<<nblk(A.ncell[A_QA] / 16), 256>>>((const uint4 *)A.cell[A_QA], (const uint4 *)owners_cells,
                                                 A.ncell[A_QA] / 16, errK + 3);
  }
  // Updates the transcript tr (a hash chain); ignores s0. The caller sets ch::host and door.link.
  void prove(Tr *tr, const u32 *s0) {
    using namespace hw;
    AH_LOCALS;
    (void)s0;
    if (!lost_copies_door::left_out("slot_pads_copy")) // fault injection; not in a production build
      kA_pl2e<<<nblk(door.ns), 256>>>(S.h_plane(0, 0), (long long)S.plane_stride(), door.ns, MSG + NSM);
    // The launch status is read at once: after a rejected launch these cells would contain no mask
    // entries, so the proof file would contain unmasked values.
    CK(cudaGetLastError());
    lost_copies_door::strike("slot_pads_copy", MSG + NSM, EDEG * door.ns, true);
    // Zeroes the unused mask entries from slot ns on, so that no entries of a sub-proof with more
    // slots stay there. Besides set-up's memset of MSG, nothing else in this file touches them.
    CK(cudaMemsetAsync(MSG + NSM + door.ns, 0, (NSM - door.ns) * sizeof(T), cudaStreamPerThread));
    slot_counts_begin(MSG);
    S.slot_pads_held(door.ns);
    ch::host.first(tr, nullptr);
    ch::kept_pads_agree(S, door.ns); // derives the mask entries again from the key and compares
    lost_copies_door::strike("slot_pads_later", MSG + NSM, EDEG * door.ns, true); // fault injection
    A.endW = [&] {
      commit_unit(U3, vec3, roots + 2 * DW, tr, 1);
    };
    A.run(tr, MSG + slPT, MSG + slCl, vec3, [&] {
      hz::late_hook(); // after the lookup: `hz::Late` buffers may overwrite its scratch and keys
    });
    k_forest_err<<<1, 1>>>(A.FO.claims, errc);
    // The range check's coefficients Cf[c][u] in {-1, 0, 1}: column c of M[mu], combination u.
    HT.draw_matrix(); // a round that the host runs, after the tables' samples
    for (int mu = 0; mu < NM; mu++) {
      zfold_prep<ETAU>(M[mu].Cf, (int)M[mu].k, M[mu].CfT);
      zfold_turned_i8<ETAU>(M[mu].v, M[mu].m, (int)M[mu].m, (int)M[mu].k, M[mu].CfT, zfold_kp((int)M[mu].k), M[mu].Z,
                            zfold_work);
    }
    HT.fold_to_group2(tr, err, zsb);
    HT.group2(tr, MSG + slPZ);
    // The range check's multiplicities depend on random coefficients, so HT.F reports to err; errc
    // is for failures that depend only on the inputs. HT.F's word is always 0: see errors().
    k_forest_err<<<1, 1>>>(HT.F.claims, err);
    HT.make_cv();
    // W.finish() draws the challenge whose powers batch all claims and fills the coefficient
    // vectors Lall, door.LS0 and door.LwM. It sends nothing.
    CK(cudaMemsetAsync(Lall, 0, nLall * sizeof(T), cudaStreamPerThread));
    door.clear_async();
    W.finish(tr, door.LS0);
    // The last fields for hd::close(), two more checks of the mask entries, then the closing.
    door.hS = MSG + NSM, door.has_pad = false;
    slot_counts_end("attention sub-proof", (int)gid, MSG, door.ns);
    door.host_words = ch::host.blob();
    S.slot_pads_agree(MSG + NSM, door.ns); // the same comparison after their last use, and then
    ch::masking_pads_hold(MSG + NSM, door.ns); // by a second implementation [own-test-slot-pads]
    if (hd::close(door, tr, "attention sub-proof", (int)gid)) {
      name_the_end("guard"); // the layer sub-proof's close reaches the same name through road_is_broken
      exit(4);
    }
    assemble();
    kA_keep<<<1, 1>>>(errK, A.err, err, errc);
  }
  // Copies the parts of the sub-proof's section into 'out' in file order.
  const u32 *assemble() {
    size_t o = 0;
    for (auto &x : SC) {
      if (x.words) CK(cudaMemcpyAsync(out + o, x.p, x.words * 4, cudaMemcpyDeviceToDevice, cudaStreamPerThread));
      o += x.words;
    }
    return out;
  }
  // The slots' pads as field elements, where prove() writes them at its start.
  const T *slot_pads() const { return MSG + NSM; }
  // Returns the flags raised since the caller zeroed errK. Bit 0: a forward-pass value did not fit
  // its cells. Bit 1: the range check's error word. Bit 2: A.FO's. A forest's word is always 0
  // (ForestTable::planes() clears it): no overflow is reported. Bit 3: see compare_qa_with().
  u32 errors() {
    u32 e[4] = {0, 0, 0, 0};
    CK(cudaMemcpy(e, errK, 16, cudaMemcpyDeviceToHost));
    return (e[0] ? 1u : 0u) | (e[1] ? 2u : 0u) | (e[2] ? 4u : 0u) | (e[3] ? 8u : 0u);
  }
};
