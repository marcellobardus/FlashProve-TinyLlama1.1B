// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// The first part of the attention sub-proof (`fork` in identifiers): plan, forward pass, cells, one
// lookup forest for the plan's lookup calls, the later messages, evaluation tables. fork.cuh adds
// commitments, masks and the approximate range check. The side stream never uses the transcript
// and reads challenge-dependent values only after a default-stream event. The forest commits to
// nothing more per key; source for that aim only, methods not compared: Qu, Guo, Xuan, Liu, Zhang,
// "Celer: A Lookup Argument for Large-Scale Queries", CRYPTO 2026, ePrint 2026/1453, abstract.
#pragma once
#include "common/side_streams.h"
#include "attn/engines.cuh"
#include "attn/forward.cuh"
#include "attn/close.cuh"
#include "hiding/late.cuh"
#include "hiding/forest.cuh"
#include "hiding/walk_calls.cuh"
#include <deque>
#include <functional>
#include <memory>
template <class X> static X *dalA(size_t n) {
  X *p_ = nullptr;
  hz::alloc(p_, n * sizeof(X) + 256, hz::FORK);
  return p_;
}
// A verifier message: n challenges from the transcript (`Tr`, a hash chain) to o[0, n). One thread.
__global__ void k_drawA(Tr *tr, Round round, T *o, int n) {
  Tr t = *tr;
  tr_point(t, round, o, n);
  *tr = t;
}
struct AFront {
  // Call plan() and allocate() once, in this order; then phase1() and run() for each sub-proof.
  APlan P;
  int NC = 0; // number of calls
  // NI = Lam * HA pieces (APiece) per lookup segment kind (ASeg). Piece number `in`: lam * HA + a.
  int NI = 0;
  static constexpr int kapA = 5; // a multiplicity has 2^kapA = 32 bits (unused)
  // How many of the lowest levels of the forest's key trees are rebuilt from the keys, not stored
  // (2 to 5). The proof does not depend on it. More levels take less scratch memory and more time.
  // 2 fits; a timing that was not made on this code found it the faster.
  int rebuilt = 2;
  // One region from hz::region_hook() (prover/hiding/late.cuh): the forest's scratch memory (the
  // first region.bytes bytes), then the keys. After hz::late_hook(), hz::Late buffers may reuse it.
  PtRegion region;
  // [earlyAt, earlyEnd) is the free rest of the region's second part. Only buffers that nothing
  // uses after hz::late_hook() may be placed there. earlyTake() returns a buffer from it, or
  // nullptr if there is no region (the hook may return none) or too little space.
  size_t earlyAt = 0, earlyEnd = 0;
  char *earlyTake(size_t bytes) {
    bytes = PtRegion::rounded(bytes + 256);
    if (!region.base || earlyAt + bytes > earlyEnd) return nullptr;
    char *p_ = region.base + earlyAt;
    earlyAt += bytes;
    return p_;
  }
  // The five request matrices `as`, ah, ao, bA, qa, indexed by AMat (plan.cuh). cell[i] has
  // ncell[i] cells (committed small integers); the last zero_fill[i] are zero fill. An evaluation
  // table: the inner product of a weight vector phi with each run of Ltab[i] cells.
  i8 *cell[5] = {nullptr, nullptr, nullptr, nullptr, nullptr};
  size_t ncell[5] = {0, 0, 0, 0, 0}, Ltab[5] = {0, 0, 0, 0, 0};
  size_t zero_fill[5] = {0, 0, 0, 0, 0}; // cells added by the caller (elsewhere: filler cells)
  // Tn below: the number of tokens p.T (`T` is the field element type). q, k, v: phase1()'s inputs.
  // [lam][h][t]: largest score mo, its index io (ix), softmax sum Zo (Z). [lam][h][t][c]: o, rem.
  const i16 *q = nullptr, *k = nullptr, *v = nullptr;
  i16 *o = nullptr;
  int *mo = nullptr, *io = nullptr, *Zo = nullptr;
  u32 *rem = nullptr;
  u32 *err = nullptr;                // set if a division remainder may not fit in wd bytes
  int *Tt = nullptr, *shl = nullptr; // device copies of P.Tt, P.sh
  void *fws = nullptr;               // the forward kernels' scratch

  // Per lookup call, in P.C's order: keys K, number of padding keys per tree pubd, table rows taud
  // (nullptr for a range table). The keys must not change from phase1() until FO has finished.
  std::vector<u16 *> K;
  std::vector<u32 *> pubd;
  std::vector<std::vector<size_t>> padded; // host: each call's `forest_padded_from` numbers
  std::vector<u64 *> taud;
  Forest FO;
  PtGraph graphs; // FO.prove() as a CUDA graph
  // By segment kind: the offset of each piece's first key in its call's key array, by piece number.
  size_t *oRM = nullptr, *oRD = nullptr, *oFM = nullptr, *oFD = nullptr, *oWL = nullptr, *oWH = nullptr, *oIX = nullptr;
  AOff kM, kD;

  // A unit: one call's pieces of one segment kind. Its inner point: zkp(ci)'s last b coordinates.
  struct PS {
    int ci = 0, seg = 0, cnt = 0; // call, segment kind, number of pieces
    int b = 0, n = 0;             // log2 of a piece's size, log2 of the call's size
    size_t *off = nullptr;        // device: the pieces' offsets in the call
    int *in = nullptr;            // device: the pieces' numbers
    T *wp = nullptr;              // device: the pieces' weights (`gc::k_wps`), then their sum
  };
  std::vector<PS> psKR, psKF;    // units of RM, RD and of FM, FD, in the order of sending
  std::vector<PS *> asp;         // psKR, then psKF
  PS psW, psX, psH;              // the units of WL, IX and WH
  int cR = -1, cX = -1, cH = -1; // their calls

  // The numbers of evaluation tables of `as`, ah, ao and qa (bA has 3), and the number of elements
  // of one table of each (kB: of bA). W is all the tables, in the order `as`, ah, ao, qa, bA.
  int nAS = 0, nAH = 0, nAO = 0, nQA = 0;
  size_t kAS = 0, eAH = 0, eAO = 0, eQA = 0, kB = 0;
  size_t nWe = 0;   // elements of W
  size_t nSend = 0; // elements that `run` writes to `sends`
  // Variables per SC3 (degree-3 sumcheck): scores (2), Z o, value product (2), q k[ix], ix's bits.
  int nvFM = 0, nvFD = 0, nvE = 0, nvM = 0, nvD = 0, nvF = 0, nvH = 0;
  // Elements that an SC3 (engines_s3.cuh) sends: the hiding polynomial's sum, 3 per variable, 2 for
  // the further variable Y (degree 2 in the maps whose end values are sent), 2 per piece.
  static size_t sc3n(int nv, int nb) { return sc_door::slots(sc_door::Form::SUM3, nv, 2 * (size_t)nb, 2); }

  // Must be set: run() calls it, once W is complete in default-stream order, to commit to W.
  std::function<void()> endW;
  // a_s3_run()'s scratch, NI elements: W_b(r) = wb[b] * W0(r); only that run's round in Y reads it.
  T *wsObj = nullptr;

  // Call ci's key point zk and the forest's two points zb on the multiplicity bits (ForestClaims).
  T *zkp(int ci) { return FO.claims->zk[ci]; }
  T *zbp(int e) { return FO.claims->zb[e]; }

  // run()'s challenges (om, om4, om5: powers of one each); buildWalk() (fork.cuh) uses all but om5.
  T *gam = nullptr, *tpts = nullptr, *om = nullptr, *om4 = nullptr, *om5 = nullptr;
  T *rrE = nullptr, *rrM = nullptr, *rrD = nullptr, *rF = nullptr, *rps = nullptr, *rrH = nullptr;
  // Per unit of psKF: eq tables eg, ei, ej of its inner point, tile-slot weights wt, WI, WJ, phiM
  // (unused), q, k contracted with ei, ej (qf, kf), a_s3_run()'s W0, h4, r (rr), 6 sent values v7.
  struct KFx {
    T *eg, *ei, *ej, *wt, *WI, *WJ, *phiM, *qf, *kf, *W0, *h4, *v7, *rr;
  };
  std::vector<KFx> kx;

  // Weight vectors phi (phiAS: tables TA, TA_VM, TA_VD; phiAO: TO; phiBA: TB_0, TB_1, TP). Tables
  // psKF.size() and psKF.size() + 1 of ah use eiW and eiX, not phiAH. ehAS: per unit of asp, the eq
  // table of its inner point's coordinates before the last lmS. v3: 3 values to send per psKR unit.
  std::vector<T *> phiAS, ehAS, phiAH, phiAO, phiQA, phiBA, v3;
  // Eq tables of the inner points of WL (*W, eih, eil), IX (*X), WH (egH) and of later points. WH's
  // token and channel coordinates are WL's (dT >= lt + ldh). ptM, ptD: from rrM, rrD and ziW.
  T *egW = nullptr, *eiW = nullptr, *ecW = nullptr, *hiW = nullptr, *loW = nullptr, *eih = nullptr, *eil = nullptr,
    *egX = nullptr, *eiX = nullptr, *egH = nullptr, *etg = nullptr,
    *etk = nullptr, *etbg = nullptr, *etbi = nullptr, *eri = nullptr, *erg = nullptr, *erc = nullptr, *loF = nullptr,
    *epj = nullptr, *epg = nullptr, *ptM = nullptr, *ptD = nullptr;
  // The sources, weights and round-0 sums of the sumchecks, and values to send.
  T *BE = nullptr, *AC = nullptr, *W0E = nullptr, *n2b = nullptr, *AVM = nullptr, *AVD = nullptr, *W0VM = nullptr,
    *W0VD = nullptr, *wvM = nullptr, *wvD = nullptr, *W0F = nullptr, *W0H = nullptr, *KC = nullptr, *h4E = nullptr,
    *h4M = nullptr, *h4D = nullptr, *h4F = nullptr, *h4H = nullptr, *v5 = nullptr, *vX = nullptr, *vH = nullptr,
    *vbuf = nullptr, *KCs = nullptr, *ibs = nullptr, *cyD = nullptr;
  // The work arrays of SC3 and its end values.
  T *AA = nullptr, *BA = nullptr, *sW1 = nullptr, *sW2 = nullptr, *As = nullptr, *Bs = nullptr, *BsF = nullptr;
  struct Tmp {
    T *lo, *wt, *WI, *WJ, *ej;
  };
  std::vector<Tmp> tq;
  T *nws() { return wsObj; }
  cudaStream_t sxs = nullptr; // the side stream
  ga::Eng eng0, engX;         // scratch memory of engines.cuh for the default and the side stream
  std::vector<cudaEvent_t> evp;
  size_t evn = 0;

  // Before the call, set P.p's values from the parameter file (`policy` in messages), P.sh, P.form
  // and optionally `rebuilt`. Afterwards ncell[] has the lengths without zero fill; the caller may
  // round them up to whole columns before allocate(). Also allocates what depends only on the plan.
  void plan() {
    AP &p = P.p;
    P.build();
    if (!a_forward_ok(P)) {
      printf("attention: the forward pass supports 32, 64 or 128 channels per head, 64 to 32,768 tokens, tiles of at "
             "least 16 tokens and at most 16 layers per sub-proof; the parameter file's shape is outside these "
             "limits\n");
      exit(2);
    }
    if (P.dT < std::max(p.lmS, p.lt + p.ldh)) { // lets units share table TA of `as` and TO of ao
      printf(
          "policy: forest-depth-A: a tree of attention's look-up forest has 2^%d keys; it must have at least 2^lmS and "
          "2^(lt + ldh), that is 2^%d\n",
          P.dT, std::max(p.lmS, p.lt + p.ldh));
      exit(2);
    }
    if (p.iota != 32 + P.dT) {
      printf("policy: policy-iotaA: the line iotaA_0 must be 32 + %d, the log2 of the keys of a tree of attention's "
             "look-up forest; it is %d\n",
             P.dT, p.iota);
      exit(2);
    }
    NC = (int)P.C.size();
    NI = p.Lam * p.HA;
    const int Tn = p.T, H = p.H, Lam = p.Lam, HA = p.HA, dh = p.dh, lg = p.lgk, lt = p.lt, lB = p.lB, lnT = p.lnT,
              ldh = p.ldh;
    const size_t nq = (size_t)Lam * Tn * p.dq, nr = (size_t)Lam * H * Tn;
    Ltab[A_AS] = (size_t)1 << p.lmS;
    Ltab[A_AH] = (size_t)Tn;
    Ltab[A_AO] = Ltab[A_QA] = (size_t)1 << p.la;
    Ltab[A_BA] = (size_t)1 << p.lmb;
    ncell[A_AS] = ((size_t)(5 * NI) << p.nM) + ((size_t)(5 * NI) << p.nD);
    ncell[A_AH] = (size_t)(8 + lt) * Lam * H * Tn;
    ncell[A_AO] = (size_t)(p.wd == 4 ? 4 : 2) * Lam * p.dq * Tn;
    ncell[A_BA] = 0; // set below, once the forest's numbers are known
    ncell[A_QA] = (size_t)2 * Lam * (2 * p.dq + 2 * p.dkv) * Tn;
    o = dalA<i16>(nq);
    mo = dalA<int>(nr);
    io = dalA<int>(nr);
    Zo = dalA<int>(nr);
    Tt = dalA<int>(P.Tt.size());
    shl = dalA<int>(Lam);
    rem = dalA<u32>(nq);
    err = dalA<u32>(1);
    CK(cudaMemset(err, 0, 4));
    CK(cudaMemcpy(Tt, P.Tt.data(), P.Tt.size() * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(shl, P.sh.data(), Lam * 4, cudaMemcpyHostToDevice));
    K.resize(NC);
    pubd.resize(NC);
    padded.resize(NC);
    taud.assign(NC, nullptr);
    {
      std::vector<ForestCall> calls; // in P.C's order
      for (int i = 0; i < NC; i++) {
        ACall &c = P.C[i];
        const size_t I = (size_t)1 << c.iota;
        pubd[i] = dalA<u32>(I);
        CK(cudaMemcpy(pubd[i], c.pub.data(), I * 4, cudaMemcpyHostToDevice));
        ForestRuns runs;
        runs.add(APlan::realKeys(c), (size_t)1 << c.n);
        padded[i] = forest_padded_from(c.n - c.iota, c.iota, runs, c.pub);
        if (c.fn) {
          taud[i] = dalA<u64>(c.tau.size());
          CK(cudaMemcpy(taud[i], c.tau.data(), c.tau.size() * 8, cudaMemcpyHostToDevice));
        }
        calls.push_back(ForestCall{c.n, c.k, c.fn, taud[i], c.pk0});
      }
      const int cap = identity_cap();
      {
        region.bytes = PtRegion::rounded(forest_scratch_bytes(calls, cap, 18, rebuilt));
        // The keys go into the region's second part: nothing reads them after the forest has run.
        size_t early = 0;
        for (int i = 0; i < NC; i++) early += PtRegion::rounded(((size_t)2 << P.C[i].n) + 256);
        region.base = (char *)hz::region_hook(region.bytes + early);
        earlyAt = region.bytes;
        earlyEnd = region.bytes + early;
      }
      for (int i = 0; i < NC; i++) {
        K[i] = (u16 *)earlyTake((size_t)2 << P.C[i].n);
        if (!K[i]) K[i] = dalA<u16>((size_t)1 << P.C[i].n);
      }
      {
        // Scope guard: FO.init()'s scratch allocation (kind 2 of pt_alloc) comes from the region.
        std::unique_ptr<PtTurn> in_the_region;
        if (region.base) in_the_region = std::make_unique<PtTurn>(region, 0, region.bytes);
        FO.init(calls, cap, 18, rebuilt);
      }
      if (FO.plan.dT != P.dT) {
        printf("attention: internal error: the plan has trees of 2^%d keys and the forest has trees of 2^%d\n", P.dT,
               FO.plan.dT);
        exit(4);
      }
    }
    // The cells of bA: bit j of the multiplicity of table row t of lookup call c < NC is at
    // (c * 2^kT + t) * 32 + j.
    ncell[A_BA] = FO.plan.plane_bits();
    auto offs = [&](int seg) {
      std::vector<size_t> ov(NI);
      for (int lam = 0; lam < Lam; lam++)
        for (int a = 0; a < HA; a++) ov[lam * HA + a] = P.piece(P.callOf(seg, lam), seg, lam, a)->off;
      size_t *d = dalA<size_t>(NI);
      CK(cudaMemcpy(d, ov.data(), NI * 8, cudaMemcpyHostToDevice));
      return d;
    };
    oRM = offs(A_RM);
    oRD = offs(A_RD);
    oFM = offs(A_FM);
    oFD = offs(A_FD);
    oWL = offs(A_WL);
    oWH = offs(A_WH);
    oIX = offs(A_IX);
    for (int lam = 0; lam < Lam; lam++) {
      kM.kr[lam] = K[P.callOf(A_RM, lam)];
      kM.kf[lam] = K[P.callOf(A_FM, lam)];
      kD.kr[lam] = K[P.callOf(A_RD, lam)];
      kD.kf[lam] = K[P.callOf(A_FD, lam)];
    }
    fws = dalA<char>(std::max(a_forward_scratch(p), a_forward_all_scratch(p)));
    nAO = 1; // TO
    nvFM = lg + 2 * lnT - 1 + ldh;
    nvFD = lg + lnT + ldh;
    nvE = lg + lt;
    nvM = lB + 2 * lnT - 1 + lg;
    nvD = lB + lnT + lg;
    nvF = ldh + lg + lt;
    nvH = 4 + lg + lt;
    // The unit of call ci and segment kind seg. Its pieces are sorted by piece number.
    auto mkps = [&](int ci, int seg) {
      PS s;
      s.ci = ci;
      s.seg = seg;
      s.b = P.segB(seg);
      s.n = P.C[ci].n;
      std::vector<std::pair<int, size_t>> v;
      for (auto &x : P.C[ci].pc)
        if (x.seg == seg) v.push_back({x.lam * HA + x.a, x.off});
      std::sort(v.begin(), v.end());
      s.cnt = (int)v.size();
      if (s.cnt > gc::WP_PIECES) {
        printf("attention: a lookup call has %d pieces of one segment kind; at most %d are supported\n", s.cnt,
               gc::WP_PIECES);
        exit(2);
      }
      std::vector<size_t> of;
      std::vector<int> in;
      for (auto &x : v) {
        in.push_back(x.first);
        of.push_back(x.second);
      }
      s.off = dalA<size_t>(s.cnt);
      s.in = dalA<int>(s.cnt);
      s.wp = dalA<T>(s.cnt + 1);
      CK(cudaMemcpy(s.off, of.data(), s.cnt * 8, cudaMemcpyHostToDevice));
      CK(cudaMemcpy(s.in, in.data(), s.cnt * 4, cudaMemcpyHostToDevice));
      return s;
    };
    for (int i = 0; i < NC; i++) {
      if (P.C[i].kind == A_KR)
        for (int s_ : {A_RM, A_RD})
          if (std::find(P.C[i].segs.begin(), P.C[i].segs.end(), s_) != P.C[i].segs.end()) psKR.push_back(mkps(i, s_));
      if (P.C[i].kind == A_KFM || P.C[i].kind == A_KFD)
        for (int s_ : {A_FM, A_FD})
          if (std::find(P.C[i].segs.begin(), P.C[i].segs.end(), s_) != P.C[i].segs.end()) psKF.push_back(mkps(i, s_));
      for (int s_ : P.C[i].segs) {
        if (s_ == A_WL) cR = i;
        if (s_ == A_IX) cX = i;
        if (s_ == A_WH) cH = i;
      }
    }
    psW = mkps(cR, A_WL);
    psX = mkps(cX, A_IX);
    psH = mkps(cH, A_WH);
    nAS = 3; // TA, TA_VM, TA_VD
    nAH = (int)psKF.size() + 6;
    nQA = 2 * (int)psKF.size() + 5;
    nSend = 3 * psKR.size();
    for (auto &s : psKF) nSend += 7 + sc3n(s.seg == A_FD ? nvFD : nvFM, s.cnt);
    nSend += 5 + lt + 2 + (p.wd == 4 ? 4 : 0) + sc3n(nvE, NI) + 2 + sc3n(nvM, NI) + sc3n(nvD, NI) + NI + sc3n(nvF, NI) +
             a_scd_nout(lt, lg, Lam, NI) + sc3n(nvH, NI);
    const size_t nBig = (size_t)NI << std::max(std::max(nvF, nvFM), std::max(nvM, nvH));
    {
      hz::Late late; // AA and BA are first used after `hz::late_hook`
      AA = dalA<T>(nBig);
      BA = dalA<T>(nBig);
    }
    const size_t mxL = std::max(std::max(Ltab[A_AS], Ltab[A_BA]), std::max((size_t)1 << p.la, (size_t)Tn));
    vbuf = dalA<T>(64 + Lam + NI * (lt + 1));
    As = dalA<T>(NI);
    Bs = dalA<T>(NI);
    egW = dalA<T>(p.G);
    eiW = dalA<T>(Tn);
    ecW = dalA<T>(dh);
    hiW = dalA<T>(dh);
    gam = dalA<T>(1);
    eih = dalA<T>(p.nT);
    eil = dalA<T>(p.B);
    tpts = dalA<T>(64);
    om = dalA<T>(NI);
    etg = dalA<T>(p.G);
    eri = dalA<T>(Tn);
    erg = dalA<T>(p.G);
    erc = dalA<T>(dh);
    KC = dalA<T>((size_t)Lam * p.G * Tn);
    rps = dalA<T>(64);
    KCs = dalA<T>(Lam);
    ibs = dalA<T>((size_t)NI * lt);
    cyD = dalA<T>(2 * lt + 1);
    etk = dalA<T>(16);
    rF = dalA<T>(64);
    BsF = dalA<T>(NI);
  }
  void allocate() {
    layoutTables();
    allocCells();
    allocPointTables();
    allocEngines();
  }
  void layoutTables() {
    AP &p = P.p;
    kAS = ncell[A_AS] / Ltab[A_AS];
    eAH = ncell[A_AH] / Ltab[A_AH];
    eAO = ncell[A_AO] / Ltab[A_AO];
    eQA = ncell[A_QA] / Ltab[A_QA];
    kB = ncell[A_BA] / Ltab[A_BA];
    nWe = nAS * kAS + nAH * eAH + nAO * eAO + nQA * eQA + 3 * kB;
  }
  void allocCells() {
    for (int i = 0; i < 5; i++) {
      cell[i] = dalA<i8>(ncell[i]);
      CK(cudaMemset(cell[i], 0, ncell[i]));
    }
  }
  void allocPointTables() {
    AP &p = P.p;
    const int Tn = p.T, Lam = p.Lam, dh = p.dh;
    const size_t LA = (size_t)1 << p.la;
    for (auto &s : psKR) asp.push_back(&s);
    for (auto &s : psKF) asp.push_back(&s);
    for (int a = 0; a < nAS; a++) phiAS.push_back(dalA<T>(Ltab[A_AS]));
    for (auto *s : asp) {
      ehAS.push_back(dalA<T>((size_t)1 << ((s->seg == A_RD || s->seg == A_FD ? p.nD : p.nM) - p.lmS)));
      v3.push_back(dalA<T>(8));
    }
    for (auto &s : psKF) {
      (void)s;
      KFx x;
      x.eg = dalA<T>(p.G);
      x.ei = dalA<T>(p.B);
      x.ej = dalA<T>(p.B);
      x.wt = dalA<T>(p.nT * p.nT);
      x.WI = dalA<T>(p.nT);
      x.WJ = dalA<T>(p.nT);
      x.phiM = dalA<T>(Tn);
      x.qf = dalA<T>((size_t)Lam * p.H * p.nT * dh);
      x.kf = dalA<T>((size_t)Lam * p.G * p.nT * dh);
      x.W0 = dalA<T>((size_t)p.G * p.nT * p.nT);
      x.h4 = dalA<T>(4);
      x.v7 = dalA<T>(8);
      x.rr = dalA<T>(64);
      kx.push_back(x);
    }
    for (int i = 0; i < nAH; i++) phiAH.push_back(dalA<T>(Tn));
    for (int i = 0; i < nAO; i++) phiAO.push_back(dalA<T>(LA));
    for (int i = 0; i < nQA; i++) phiQA.push_back(dalA<T>(LA));
    for (int i = 0; i < 3; i++) phiBA.push_back(dalA<T>(Ltab[A_BA]));
    loW = dalA<T>(dh);
    egX = dalA<T>(p.G);
    eiX = dalA<T>(Tn);
    egH = dalA<T>(p.G);
    BE = dalA<T>((size_t)NI * p.G * Tn);
    AC = dalA<T>((size_t)NI * p.G * Tn);
    W0E = dalA<T>((size_t)p.G * Tn);
    n2b = dalA<T>((size_t)Lam * p.G * Tn);
    {
      hz::Late late; // AVM and AVD are first used after `hz::late_hook`
      AVM = dalA<T>((size_t)NI << nvM);
      AVD = dalA<T>((size_t)NI << nvD);
    }
    W0VM = dalA<T>((size_t)p.G * p.nT * p.nT);
    W0VD = dalA<T>((size_t)p.G * p.nT);
    wvM = dalA<T>(p.nT * p.nT);
    wvD = dalA<T>(p.nT);
    h4E = dalA<T>(4);
    h4M = dalA<T>(4);
    h4D = dalA<T>(4);
    h4F = dalA<T>(4);
    h4H = dalA<T>(4);
    v5 = dalA<T>(8);
    vX = dalA<T>(p.lt + 1);
    vH = dalA<T>(4);
    W0F = dalA<T>((size_t)p.G * Tn);
    W0H = dalA<T>((size_t)p.G * Tn * 16);
    rrE = dalA<T>(64);
    rrM = dalA<T>(64);
    rrD = dalA<T>(64);
    rrH = dalA<T>(64);
    ptM = dalA<T>(64);
    ptD = dalA<T>(64);
    epg = dalA<T>(p.G);
    etbg = dalA<T>(p.G);
    etbi = dalA<T>(Tn);
    epj = dalA<T>(Tn);
    om4 = dalA<T>(NI);
    om5 = dalA<T>(NI);
    loF = dalA<T>(dh);
    {
      int mw = std::max(std::max(nvH, nvE), std::max(std::max(nvFM, nvF) - p.ldh, nvM - p.lB));
      sW1 = dalA<T>(((size_t)1 << mw) / 2 + 2);
      sW2 = dalA<T>(((size_t)1 << mw) / 4 + 2);
    }
  }
  void allocEngines() {
    AP &p = P.p;
    wsObj = dalA<T>(NI);
    int lo_, hi_;
    CK(cudaDeviceGetStreamPriorityRange(&lo_, &hi_));
    CK(cudaStreamCreateWithPriority(&sxs, cudaStreamNonBlocking, lo_));
    CK(side_streams::enter(sxs));
    size_t s = 0;
    for (int i = 0; i < 5; i++) s = std::max(s, a_table_scratch(9, ncell[i] / Ltab[i], Ltab[i]));
    engX.reserve(s);
    eng0.reserve(std::max(s, a_scd_scratch(p.lt, p.lgk, p.HA, p.Lam)));
    for (size_t i = 0; i < psKF.size() + 2; i++)
      tq.push_back({dalA<T>(p.dh), dalA<T>(p.nT * p.nT), dalA<T>(p.nT), dalA<T>(p.nT), dalA<T>(p.B)});
  }
  // q is [lam][t][H dh], and k and v are [lam][t][G dh], all int16 on the device. They must stay
  // valid until run()'s kernels have finished. Fills all cells except the zero fill, all keys, and
  // the forest's multiplicities, whose bits are the cells of bA (cell[A_BA]). g_st: as for run().
  void phase1(const i16 *q_, const i16 *k_, const i16 *v_) {
    q = q_;
    k = k_;
    v = v_;
    // Zeroes the cells again. ForkMemory::run_begin() repeats allocCells()'s zeroing in one kernel
    // launch; if the GPU accepts but never runs it, no CUDA call reports it. Nothing below writes
    // the zero fill, and the verifier does not need zeros there: old bytes would be committed.
    for (int i = 0; i < 5; i++) CK(cudaMemsetAsync(cell[i], 0, ncell[i], cudaStreamPerThread));
    for (int i = 0; i < NC; i++) {
      const size_t e_ = APlan::realKeys(P.C[i]);
      if (e_ < ((size_t)1 << P.C[i].n))
        kA_fill16<<<nbk(((size_t)1 << P.C[i].n) - e_), 256>>>(K[i] + e_, ((size_t)1 << P.C[i].n) - e_, (u16)P.C[i].pk0);
    }
    a_forward_all(P, shl, q, k, v, Tt, mo, io, Zo, o, rem, cell[A_AS], kM, kD, oRM, oFM, oRD, oFD, cell[A_AH],
                  cell[A_AO], cell[A_QA], K[P.callOf(A_WL, 0)], K[P.callOf(A_WH, 0)], oWL, oWH, oIX, err, fws, 0);
    for (int i = 0; i < NC; i++) FO.counts(i, K[i], pubd[i], padded[i].data());
    FO.planes();
    CK(cudaMemcpyAsync(cell[A_BA], FO.bits(0), FO.plan.plane_bits(), cudaMemcpyDeviceToDevice,
                       cudaStreamPerThread));
  }
  // Runs FO, calls after_the_lookup (fork.cuh: hz::late_hook()), sends values and runs sumchecks
  // ("steps 6 to 22" in messages), calls endW. Writes msg[0, FO.plan.nmsg()), sends[0, nSend),
  // W[0, nWe). g_st (FO's stream) must be cudaStreamPerThread (the build's default stream), not 0.
  void run(Tr *tr, T *msg, T *sends, T *W, std::function<void()> after_the_lookup) {
    AP &p = P.p;
    const int Tn = p.T, H = p.H, Lam = p.Lam, HA = p.HA, dh = p.dh, lg = p.lgk, lt = p.lt, lB = p.lB, lnT = p.lnT,
              ldh = p.ldh, ls = p.ls;
    const cudaStream_t s0 = 0, sx = sxs;
    ga::Eng *e0 = &eng0;
    evn = 0;
    auto rec = [&](cudaStream_t s) {
      if (evn == evp.size()) {
        cudaEvent_t e;
        CK(cudaEventCreateWithFlags(&e, cudaEventDisableTiming));
        evp.push_back(e);
      }
      cudaEvent_t e = evp[evn++];
      CK(cudaEventRecord(e, s));
      return e;
    };
    auto wait = [&](cudaStream_t s, cudaEvent_t e) {
      CK(cudaStreamWaitEvent(s, e, 0));
    };
    forest_prove_graph(graphs, FO, tr, msg);
    after_the_lookup();
    T *o_ = sends, *WAS = W, *WAH = WAS + nAS * kAS, *WAO = WAH + nAH * eAH, *WQA = WAO + nAO * eAO,
      *WBA = WQA + nQA * eQA;
    const int n1 = (int)asp.size(), nKF = (int)psKF.size(), nKR = (int)psKR.size(), lslM = 2 * lnT - 1, lslD = lnT,
              qE = 2 * nKF, qV = qE + 1, qF = qE + 3, qJ = qE + 4;
    gc::EqList E;
    gc::OuList O;
    const S3Work wk{AA, BA, sW1, sW2};
    auto tabs = [&](int Mi, int n, T *const *phi, T *dst, size_t stride, cudaStream_t st, ga::Eng *e) {
      std::vector<T *> o(n);
      for (int i = 0; i < n; i++) o[i] = dst + (size_t)i * stride;
      a_tables_n(n, cell[Mi], ncell[Mi], phi, Ltab[Mi], o.data(), st, e);
    };
    auto sendl = [&](std::vector<gc::SdJob> v) {
      gc::SdJobs J;
      J.n = (int)v.size();
      size_t n = 0;
      for (int i = 0; i < J.n; i++) {
        J.j[i] = v[i];
        n += v[i].na + v[i].nb;
      }
      gc::k_sends<<<1, 1>>>(tr, J, o_);
      o_ += n;
    };
    // Dynamic shared memory of the launches: smS for k_slots; smB and smC for kernels that stage B
    // or dh elements as limbs (gc::SL); smV for k_arrV (F::SP).
    const size_t smS = (p.hP + p.nT) * sizeof(T), smB = p.B * sizeof(gc::SL), smC = dh * sizeof(gc::SL),
                 smV = p.B * sizeof(F::SP) + 256 * sizeof(T);
    T *zsl = zkp(cR) + psW.n - psW.b, *zgW = zsl + 1, *ziW = zgW + lg, *zcW = ziW + lt, *zgX = zkp(cX) + psX.n - psX.b,
      *ziX = zgX + lg, *zgH = zkp(cH) + psH.n - psH.b + (p.wd == 4 ? 1 : 0);
    {
      gc::WpJobs J;
      int n = 0;
      auto add = [&](PS &s) {
        J.j[n++] = {zkp(s.ci), s.n, s.b, s.cnt, s.off, s.wp};
        if (n == 32) {
          gc::k_wps<<<n, gc::WP_PIECES>>>(J);
          n = 0;
        }
      };
      for (auto *s : asp) add(*s);
      add(psW);
      add(psX);
      add(psH);
      if (n) gc::k_wps<<<n, gc::WP_PIECES>>>(J);
    }
    for (int a = 0; a < n1; a++) {
      PS &s = *asp[a];
      const int n0 = (s.seg == A_RD || s.seg == A_FD) ? p.nD : p.nM;
      E.add(zkp(s.ci) + s.n - s.b, n0 - p.lmS, ehAS[a]);
    }
    // One table TA serves all units of asp: each inner point ends in zx's last lmS coordinates.
    E.add(FO.claims->zx + FO.plan.dT - p.lmS, p.lmS, phiAS[0]);
    for (int f = 0; f < nKF; f++) {
      PS &s = psKF[f];
      const bool dg = s.seg == A_FD;
      T *zg = zkp(s.ci) + s.n - s.b, *zP = zg + lg, *zi = dg ? zP + lnT : zP + 2 * lnT - 1, *zj = zi + lB;
      E.add(zg, lg, kx[f].eg);
      E.add(zi, lB, kx[f].ei);
      E.add(zj, lB, kx[f].ej);
    }
    E.add(zgW, lg, egW);
    E.add(ziW, lt, eiW);
    E.add(zcW, ldh, ecW);
    E.add(zcW, ldh - ls, hiW);
    E.add(zcW + ldh - ls, ls, loW);
    E.add(ziW, lnT, eih);
    E.add(ziW + lnT, lB, eil);
    E.add(zgX, lg, egX);
    E.add(ziX, lt, eiX);
    E.add(zgH, lg, egH);
    E.run();
    for (int f = 0; f < nKF; f++) {
      PS &s = psKF[f];
      gc::k_slots<<<1, 256, smS>>>(p, s.seg == A_FD, zkp(s.ci) + s.n - s.b + lg, kx[f].wt, kx[f].WI, kx[f].WJ);
    }
    gc::k_slotI<<<1, 256>>>(p, 0, eih, wvM);
    gc::k_slotI<<<1, 256>>>(p, 1, eih, wvD);
    for (int f = 0; f < nKF; f++) {
      const bool dg = psKF[f].seg == A_FD;
      O.add(kx[f].WI, lnT, kx[f].ei, lB, phiAH[f]);
      O.add(kx[f].eg, lg, kx[f].wt, dg ? lslD : lslM, kx[f].W0);
    }
    O.add(loW, ls, eiW, lt, phiAO[0]); // TO, for WL and WH
    O.add(egW, lg, eiW, lt, W0E);
    O.add(egW, lg, wvM, lslM, W0VM);
    O.add(egW, lg, wvD, lslD, W0VD);
    O.run();
    const cudaEvent_t ev0 = rec(s0);
    wait(sx, ev0);
    {
      std::vector<T *> ph(phiAH.begin(), phiAH.begin() + nKF);
      ph.push_back(eiW);
      ph.push_back(eiX);
      tabs(A_AH, nKF + 2, ph.data(), WAH, eAH, sx, &engX);
    }
    for (int f = 0; f < nKF; f++) {
      PS &s = psKF[f];
      const bool dg = s.seg == A_FD;
      k_rowval<<<1, 256, 0, sx>>>(p, WAH + (size_t)f * eAH, 0, s.wp, s.in, s.cnt, kx[f].eg, kx[f].v7 + 5);
      gc::k_foldtile<<<nbk((size_t)Lam * H * p.nT * dh, 256, 2048), 256, smB, sx>>>(p, q, H, kx[f].ei, kx[f].qf);
      gc::k_foldtile<<<nbk((size_t)Lam * p.G * p.nT * dh, 256, 2048), 256, smB, sx>>>(p, k, p.G, kx[f].ej, kx[f].kf);
      a_s3_prep(gc::Src4{p, dg, dg ? lslD : lslM, s.in, kx[f].qf, kx[f].kf}, dg ? nvFD : nvFM, ldh, s.cnt, s.wp,
                kx[f].W0, kx[f].h4, sx, &engX);
    }
    const cudaEvent_t evF = rec(sx);
    gc::k_rowsE<<<nbk((size_t)NI * p.G * Tn * 8, 256, 4096), 256, smC, sx>>>(p, psW.in, psW.cnt, ecW, Zo, o, rem, BE,
                                                                             AC);
    gc::k_wsum<<<dim3(16, NI), 256, 0, sx>>>(psW.wp, W0E, AC, lg + lt, engX.part);
    ga::k_sum_parts<<<1, 256, 0, sx>>>(engX.part, 16 * NI, 1, 1, v5);
    a_s3_prep(gc::SrcE{p, psW.in, Zo, BE}, nvE, 0, NI, psW.wp, W0E, h4E, sx, &engX);
    k_cat<<<1, 32, 0, sx>>>(v5 + 1, h4E + 3, 1, nullptr, 0);
    tabs(A_AO, 1, phiAO.data(), WAO, eAO, sx, &engX);
    k_rowval<<<1, 256, 0, sx>>>(p, WAH + (size_t)nKF * eAH, 1, psW.wp, psW.in, psW.cnt, egW, v5 + 2);
    k_rhval<<<1, 256, 0, sx>>>(p, WAO, hiW, psW.wp, psW.in, psW.cnt, egW, v5 + 3);
    k_ibval<<<lt, 256, 0, sx>>>(p, WAH + (size_t)(nKF + 1) * eAH, psX.wp, psX.in, psX.cnt, egX, vX);
    k_rhval<<<1, 256, 0, sx>>>(p, WAO, hiW, psH.wp, psH.in, psH.cnt, egH, vH);
    const cudaEvent_t evE = rec(sx);
    gc::k_dotc<<<nbk((size_t)Lam * p.G * Tn * 8, 256, 4096), 256, smC, sx>>>(p, v, ecW, n2b);
    gc::k_arrV<<<(unsigned)std::min<size_t>((size_t)NI * p.G * p.hP * p.nT, 1 << 16), 256, smV, sx>>>(
        p, 0, psW.in, psW.cnt, eil, cell[A_AS], AVM);
    gc::k_arrV<<<(unsigned)std::min<size_t>((size_t)NI * p.G * p.nT, 1 << 16), 256, smV, sx>>>(p, 1, psW.in, psW.cnt,
                                                                                                eil, cell[A_AS], AVD);
    const cudaEvent_t evV = rec(sx);
    for (int e = 0; e < 2; e++) // TB_0 and TB_1; then TP
      hw::k_forest_phi_bits<<<nbk(Ltab[A_BA]), 256, 0, sx>>>(zbp(e), FO.plan.kT, FO.plan.cK, p.lmb, phiBA[e]);
    hw::k_forest_phi_powers<<<nbk(Ltab[A_BA]), 256, 0, sx>>>(p.lmb, phiBA[2]);
    tabs(A_BA, 3, phiBA.data(), WBA, kB, sx, &engX);
    tabs(A_AS, 1, phiAS.data(), WAS, kAS, s0, e0);
    for (int a = 0; a < n1; a++) {
      PS &s = *asp[a];
      const bool dg = s.seg == A_RD || s.seg == A_FD;
      const int n0 = dg ? p.nD : p.nM;
      T *dst = a < nKR ? v3[a] : kx[a - nKR].v7;
      k_planes<<<(a < nKR ? 3 : 5), 256>>>(WAS, ehAS[a], n0 - p.lmS, s.wp, s.in, s.cnt,
                                           dg ? ((size_t)(5 * NI) << p.nM) >> p.lmS : 0, ((size_t)NI << n0) >> p.lmS,
                                           dst);
    }
    {
      std::vector<gc::SdJob> v;
      for (int a = 0; a < nKR; a++) {
        v.push_back({v3[a], 3, nullptr, 0});
        if (v.size() == 16) {
          sendl(v);
          v.clear();
        }
      }
      if (v.size()) sendl(v);
    }
    wait(s0, evF);
    for (int f = 0; f < nKF; f++) {
      PS &s = psKF[f];
      const bool dg = s.seg == A_FD;
      const int nv = dg ? nvFD : nvFM;
      T *rr_ = kx[f].rr;
      sendl({{kx[f].v7, 6, kx[f].h4 + 3, 1}});
      a_s3_run(gc::Src4{p, dg, dg ? lslD : lslM, s.in, kx[f].qf, kx[f].kf}, nv, ldh, s.cnt, s.wp, kx[f].W0, wk,
               kx[f].h4, tr, o_, rr_, As, Bs, nws(), s0, e0);
      o_ += sc3n(nv, s.cnt);
      wait(sx, rec(s0));
      Tmp &t = tq[f];
      gc::k_slots<<<1, 256, smS, sx>>>(p, dg, rr_ + lg, t.wt, t.WI, t.WJ);
      a_eq(rr_ + nv - ls, ls, t.lo, sx);
      O.add3(t.lo, ls, t.WI, lnT, kx[f].ei, lB, phiQA[2 * f]);
      O.add3(t.lo, ls, t.WJ, lnT, kx[f].ej, lB, phiQA[2 * f + 1]);
      O.run(sx);
    }
    wait(s0, evE);
    sendl({{v5, p.wd == 4 ? 7 : 5, nullptr, 0}, {vX, lt, nullptr, 0}, {vH, p.wd == 4 ? 4 : 2, nullptr, 0}});
    a_s3_run(gc::SrcE{p, psW.in, Zo, BE}, nvE, 0, NI, psW.wp, W0E, wk, h4E, tr, o_, rrE, As, Bs, nws(), s0, e0);
    o_ += sc3n(nvE, NI);
    wait(sx, rec(s0));
    a_eq(rrE + lg, lt, phiAH[nKF + 2], sx);
    O.add(loW, ls, phiAH[nKF + 2], lt, phiQA[qE]);
    O.run(sx);
    k_drawA<<<1, 1>>>(tr, Round::BAT_SIGMA, gam, 1);
    wait(s0, evV);
    const gc::SrcV sM{p, 0, lslM, psW.in, AVM, n2b, gam}, sD{p, 1, lslD, psW.in, AVD, n2b, gam};
    a_s3_prep(sM, nvM, lB, NI, psW.wp, W0VM, h4M, s0, e0);
    a_s3_prep(sD, nvD, lB, NI, psW.wp, W0VD, h4D, s0, e0);
    sendl({{h4M + 3, 1, h4D + 3, 1}});
    a_s3_run(sM, nvM, lB, NI, psW.wp, W0VM, wk, h4M, tr, o_, rrM, As, Bs, nws(), s0, e0);
    o_ += sc3n(nvM, NI);
    a_s3_run(sD, nvD, lB, NI, psW.wp, W0VD, wk, h4D, tr, o_, rrD, As, Bs, nws(), s0, e0);
    o_ += sc3n(nvD, NI);
    wait(sx, rec(s0));
    for (int dg = 0; dg < 2; dg++) {
      const int nv = dg ? nvD : nvM;
      T *rr_ = dg ? rrD : rrM, *pt = dg ? ptD : ptM;
      Tmp &t = tq[nKF + dg];
      k_cat<<<1, 64, 0, sx>>>(pt, rr_, nv - lB, ziW + lnT, lB);
      k_cat<<<1, 64, 0, sx>>>(pt + nv, rr_ + nv - lB, lB, rr_, 0);
      E.add(pt + (dg ? p.nD : p.nM) - p.lmS, p.lmS, phiAS[1 + dg]);
      E.add(rr_ + nv - lB, lB, t.ej);
      gc::k_slots<<<1, 256, smS, sx>>>(p, dg, rr_ + lg, t.wt, t.WI, t.WJ);
      O.add3(loW, ls, t.WJ, lnT, t.ej, lB, phiQA[qV + dg]);
    }
    E.run(sx);
    O.run(sx);
    tabs(A_AS, 2, phiAS.data() + 1, WAS + kAS, kAS, sx, &engX);
    k_drawA<<<1, 1>>>(tr, Round::ZERO, tpts, 2 * lg + 2 * lt + 4);
    {
      T *tg = tpts, *ti = tg + lg, *tk = ti + lt, *tbg = tk + 4, *tbi = tbg + lg, *eti_ = phiAH[nKF + 3];
      E.add(tg, lg, etg);
      E.add(ti, lt, eti_);
      E.add(tk, 4, etk);
      E.add(tbg, lg, etbg);
      E.add(tbi, lt, etbi);
      E.run();
      O.add(eti_, lt, etg, lg, W0F);
      O.add3(etbi, lt, etbg, lg, etk, 4, W0H);
      O.run();
      tabs(A_AH, 1, &phiAH[nKF + 3], WAH + (size_t)(nKF + 3) * eAH, eAH, s0, e0);
      k_m6<<<1, 64>>>(p, WAH + (size_t)(nKF + 3) * eAH, etg, vbuf);
      sendl({{vbuf, NI, nullptr, 0}});
      a_coins(tr, NI, om);
      const gc::SrcF sF{p, q, k, io};
      a_s3_prep(sF, nvF, ldh, NI, om, W0F, h4F, s0, e0);
      a_s3_run(sF, nvF, ldh, NI, om, W0F, wk, h4F, tr, o_, rF, As, BsF, nws(), s0, e0);
      o_ += sc3n(nvF, NI);
      T *ri = rF, *rg = ri + lt, *rc = rg + lg;
      E.add(ri, lt, eri);
      E.add(rg, lg, erg);
      E.add(rc, ldh, erc);
      E.add(rc + ldh - ls, ls, loF);
      E.run();
      wait(sx, rec(s0));
      O.add(loF, ls, eri, lt, phiQA[qF]);
      O.run(sx);
      a_coins(tr, NI, om4);
      gc::k_dotc<<<nbk((size_t)Lam * p.G * Tn * 8, 256, 4096), 256, smC>>>(p, k, erc, KC);
      a_scd(lt, lg, HA, Lam, io, KC, eri, erg, om4, BsF, tr, o_, rps, KCs, s0, e0);
      T *pj = rps, *pg = pj + lt, *pi_ = pg + lg;
      E.add(pj, lt, epj);
      E.add(pi_, lt, phiAH[nKF + 4]);
      E.add(pg, lg, epg);
      E.run();
      wait(sx, rec(s0));
      O.add(loF, ls, epj, lt, phiQA[qJ]);
      O.run(sx);
      tabs(A_AH, 1, &phiAH[nKF + 4], WAH + (size_t)(nKF + 4) * eAH, eAH, s0, e0);
      k_ibs<<<nbk((size_t)NI * lt), 256>>>(p, WAH + (size_t)(nKF + 4) * eAH, epg, ibs);
      a_scd_end(lt, lg, HA, Lam, KCs, ibs, om4, ri, rg, rps, cyD, tr, s0, e0);
      o_ += a_scd_nout(lt, lg, Lam, NI);
      tabs(A_QA, nQA, phiQA.data(), WQA, eQA, sx, &engX);
      a_coins(tr, NI, om5);
      const gc::SrcH sH{p, io};
      a_s3_prep(sH, nvH, 0, NI, om5, W0H, h4H, s0, e0);
      a_s3_run(sH, nvH, 0, NI, om5, W0H, wk, h4H, tr, o_, rrH, As, Bs, nws(), s0, e0);
      o_ += sc3n(nvH, NI);
      a_eq(rrH, lt, phiAH[nKF + 5]);
    }
    wait(s0, rec(sx));
    {
      T *ph[2] = {phiAH[nKF + 2], phiAH[nKF + 5]},
        *od[2] = {WAH + (size_t)(nKF + 2) * eAH, WAH + (size_t)(nKF + 5) * eAH};
      a_tables_n(2, cell[A_AH], ncell[A_AH], ph, Ltab[A_AH], od, s0, e0);
    }
    if ((size_t)(o_ - sends) != nSend) {
      printf("attention: internal error: steps 6 to 22 sent %zu elements, but the layout has %zu\n",
             (size_t)(o_ - sends), nSend);
      name_the_end("guard"); // serving: p258 end 1 guard (kept as an exit(4))
      exit(4); // not 5: `end_on_exit` (prover/prover/ends.h) reads 5 as a failed CUDA call
    }
    endW();
  }
};
