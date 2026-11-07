// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// g_setup_walk() records every claim of a layer sub-proof (all are linear) once, at set-up, in the
// verifier's order, in W, its hw::Walk (prover/hiding/walk.cuh), as jobs that each proof evaluates.
// Names: `fork` is a sub-proof, `slot` a sent scalar's number, `tail` the approximate range check.
#pragma once
#include "common/end_name.h" // name_the_end (M5g)
#include "prover/prover.h"

// Returns the offset in U3, the committed evaluation tables, of table U at point pt; exits if none.
inline size_t Prover::off3(const char *U, const std::string &pt) {
  auto it = OFF.find(std::string(U) + "|" + pt);
  if (it == OFF.end()) {
    printf("internal error: hiding has no table %s at point %s\n", U, pt.c_str());
    name_the_end("walk-plan"); // serving: p258 end 1 walk-plan (M5g)
    exit(2);
  }
  return it->second;
}

// Declares W's spaces (arrays of variables). 0: the hidden value under each slot (what was sent,
// less a multiple of its mask entry). 1, 2, 4: empty. 3: U3. Then one per virtual row of the range
// check (prover/hiding/tail_f.cuh). spCP: empty; in every sub-proof the spaces of HT.M follow it.
inline int Prover::g_walk_spaces(int g) {
  using namespace hw;
  Fork &K = FK[g];
  HFk &Q = *K.h;
  HiddenTail &HT = Q.HT;
  hw::Walk &W = Q.W;
  {
    hz::Late late; // a run uses W's pool only after `late_begin`, which `g_fold` calls
    W.init((size_t)24 << 20);
  }
  W.scratch = MSG + 2 * NSM; // MSG's third plane: scratch for the sumchecks (`psc` in identifiers)
  nrows = (int)HT.vrows.size();
  LS = 1 << ls;

  W.sp.resize(6 + nrows);
  spCP = 5 + nrows;
  // For the sub-proof's closing (hd::close(), prover/hiding/door.cuh): one space per matrix of the
  // range check's list HT.M, with one variable per column. Its Space::lam receives the coefficients
  // of the right side of the matrix's tie (the claim that binds the check's combinations to the
  // matrix), signed and batched. D.LwM holds these vectors end to end; D.LS0 is that of space 0.
  W.sp.resize(6 + nrows + HT.NM);
  {
    hd::Door &D = Q.door;
    D.ns = Q.ns, D.nLwM = 0;
    D.atM.assign(HT.NM, 0);
    for (int mu = 0; mu < HT.NM; mu++) D.atM[mu] = D.nLwM, D.nLwM += HT.M[mu].k;
    CK(fork_alloc(&D.LwM, D.nLwM * sizeof(T), Place::Late));
    CK(fork_alloc(&D.LS0, (D.ns + D.nC) * sizeof(T), Place::Late));
    for (int mu = 0; mu < HT.NM; mu++) {
      Space &z = W.sp[spCP + 1 + mu];
      z.lam = D.LwM + D.atM[mu];
      z.len = HT.M[mu].k;
    }
    W.sp[0].lam = D.LS0;
  }
  {
    Space &z = W.sp[0];
    z.len = Q.ns;
    z.neg = 1;
  }
  {
    size_t n = Q.nv[3];
    for (auto &v : HT.vrows) {
      n += v.T->K;
    }
    Q.nLall = n;
    CK(fork_alloc(&Q.Lall, n * sizeof(T), Place::Late));
    T *p = Q.Lall;
    Q.LU[3] = p;
    p += Q.nv[3];
    for (auto &v : HT.vrows) {
      Q.Lrow.push_back(p);
      p += v.T->K;
    }
  }
  {
    Space &z = W.sp[3];
    z.lam = Q.LU[3];
    z.len = Q.nv[3];
  }
  for (int r = 0; r < nrows; r++) {
    Space &z = W.sp[5 + r];
    z.lam = Q.Lrow[r];
    z.len = HT.vrows[r].T->K;
  }
  return 0;
}

// Records the claims from those of the lookup forest (the product trees of the sub-proof's lookup
// calls) to the range check's, as verifier/src/vf_layers_check.inc lists them. Fills gClaim (see below).
inline int Prover::g_walk_relations(int g) {
  using namespace hw;
  Fork &K = FK[g];
  const LayerLayout &L = K.g.L;
  HFk &Q = *K.h;
  HiddenTail &HT = Q.HT;
  hw::Walk &W = Q.W;
  const HE m1 = hneg(hE(1));
  std::vector<HE> p256v(8);
  p256v[0] = hE(1);
  for (int i = 1; i < 8; i++) {
    p256v[i] = hmul(p256v[i - 1], hE(256));
  }
  const T *P256 = W.cvec(p256v);
  std::vector<TableRead> HFUN;
  auto addFun = [&](const LM &lm, size_t L_, const T *phi, int tree, size_t off) {
    for (int p = 0; p < lm.n; p++) {
      HFUN.push_back({lm.first + p, L_, phi, off, tree});
      off += K.M[lm.first + p].k * (K.M[lm.first + p].m / L_);
    }
  };
  std::map<int, size_t> OFFK, NTAB;
  {
    size_t o = 0;
    for (auto &w : K.wk) {
      OFFK[w.kind] = o;
      NTAB[w.kind] = w.ntab;
      o += w.ntab;
    }
  }
  OFF = std::map<std::string, size_t>();
  // Tables: one element per feature x at a short point (K.SPn), per x / LS at a long one (K.LPn).
  std::vector<size_t> OFFB;
  {
    size_t o = 0;
    auto tb = [&](const char *U, const LM &lm, const std::vector<std::pair<std::string, int>> &pts,
                  const std::vector<std::string> &rd, size_t L_) {
      for (auto &p : pts) {
        if (has(rd, p.first)) {
          OFF[std::string(U) + "|" + p.first] = o;
          addFun(lm, L_, K.phiP[p.first], 3, o);
          o += cells_of(K, lm) / L_;
        }
      }
    };
    tb("UF", K.af, K.LPn, rdAF, LA);
    tb("UQ", K.qa, K.LPn, rdQA, LA);
    tb("UX", K.xg, K.LPn, rdXG, LA);
    tb("UY", K.xn, K.LPn, rdXN, LA);
    tb("UD", K.ad, K.SPn, rdAD, NT);
    tb("UR", K.ar, K.SPn, rdAR, NT);
    tb("Uq", K.qa, K.SPn, {"z"}, NT);
    for (int i = 0; i < 3; i++) { // TB_0, TB_1, TP: the tables of b, the multiplicities' bits
      addFun(K.b, (size_t)1 << lmb, K.phiB[i], 3, o);
      OFFB.push_back(o);
      o += K.M[K.b.first].k;
    }
    if (o != K.nW) {
      printf("internal error: the offsets of the tables do not add up to the planned number of table elements\n");
      name_the_end("walk-plan"); // serving: p258 end 1 walk-plan (M5g)
      return 2;
    }
  }
  // Lookup segment sg: EW(sg)[x] = eq(zj_c, pos(sg, x)); EWH(sg)[xh]: same on pos(sg, LS xh) / LS
  std::map<std::string, const T *> EWc, EWHc;
  std::vector<const T *> eh(NC, nullptr);
  auto EW = [&](const char *sg) {
    auto it = EWc.find(sg);
    if (it != EWc.end()) {
      return it->second;
    }
    const int id = L.segid(sg), c = L.seg[id].call;
    std::vector<u32> ix(L.seg[id].n);
    for (size_t x = 0; x < ix.size(); x++) {
      ix[x] = (u32)L.pos(id, (long long)x);
    }
    return EWc[sg] = W.gather(K.ec[c], ix);
  };
  auto EWH = [&](const char *sg) {
    auto it = EWHc.find(sg);
    if (it != EWHc.end()) {
      return it->second;
    }
    const int id = L.segid(sg), c = L.seg[id].call;
    if (!eh[c]) {
      eh[c] = W.eq((const T *)K.zj(c), K.lJ[c] - ls);
    }
    std::vector<u32> ix(L.seg[id].n / LS);
    for (size_t x = 0; x < ix.size(); x++) {
      ix[x] = (u32)(L.pos(id, (long long)x * LS) / LS);
    }
    return EWHc[sg] = W.gather(eh[c], ix);
  };
  Ten XBT;
  XBT.nl = xl;
  XBT.cnt = 1;
  XBT.J = d;
  XBT.off = 0;
  auto hSx = [&](const char *sg, const char *U, const std::string &pt, const Ten &X, int i) {
    AF a;
    const T *v = EWH(sg);
    for (int l = 0; l < X.nl; l++) {
      a = a + rng(3, off3(U, pt) + (size_t)(X.off + ((long long)l * X.cnt + i) * X.J) / LS, 1, (size_t)X.J / LS, v, 1,
                  Sy(p256v[l]));
    }
    return a;
  };
  auto hXSr = [&](const char *U, const std::string &pt, const Ten &X, int i, size_t x0, size_t n, const T *vec) {
    AF a;
    for (int l = 0; l < X.nl; l++) {
      a = a + rng(3, off3(U, pt) + (size_t)(X.off + ((long long)l * X.cnt + i) * X.J) + x0, 1, n, vec, 1, Sy(p256v[l]));
    }
    return a;
  };
  auto hXLr = [&](const char *U, const std::string &pt, const Ten &X, int i, size_t n, const T *vec) {
    AF a;
    for (int l = 0; l < X.nl; l++) {
      a = a + rng(3, off3(U, pt) + (size_t)(X.off + ((long long)l * X.cnt + i) * X.J) / LS, 1, n, vec, 1, Sy(p256v[l]));
    }
    return a;
  };
  auto hSst = [&](const char *sg, int pos, const std::string &pt) {
    if (pos == 0) {
      return hSx(sg, "UX", pt, XBT, 0);
    }
    if (pos == NW * Lam) {
      return hSx(sg, "UY", pt, XBT, 0);
    }
    return hSx(sg, "UF", pt, L.t[X_xs], pos - 1);
  };

  auto tOf = [](int k) {
    return k < 16 ? 16 - k : 0;
  };
  enum { KG_ = 4, KU_ = 5, KD_ = 6, M2_ = 9, K2_ = 10 };
  const int mu2 = mo ? ll - 1 + lfe + lt : 0;
  /* W numbers the claims as they are recorded, and claim c is batched with theta^c: hence the
     verifier's order. A form is recorded without its constant term; the verifier computes that.
     The code below fills ff (the forms in which the lookup forest ends) and gClaim, the summands of
     c_start, the batched sumcheck's start form: JF (defined below), then gw_EN, which TSC(zt), the
     sumcheck over tokens at the point zt, leaves to the batched one. Without TSC(zt): JF only. */
  const int gw_lNI = l2i((size_t)NI), gw_nqk = dq + dkv, gw_hqk = policy["hq"], gw_hv = policy["hv"],
            gw_ho = policy["ho"], gw_hg = policy["hgD"], gw_hu = policy["huD"], gw_hd = policy["hdD"],
            gw_hr = mo ? (int)policy["hr"] : 0;
  auto gw_isM = [&](int lam) {
    return mo && (lam & 1);
  };
  auto gw_kl = [&](int lam) -> decltype(K.g.kl[0]) & {
    return K.g.kl[g * Lam + lam];
  };
  /* gw_SxR (long point): sum over xh in [b0 / LS, b1 / LS) of `EWH`(sg)[xh] X[pt][i][xh - x0 / LS].
     gw_sumS (short point): sum over x < n of `EW`(sg)[x] X[pt][i][x]. LS divides x0, b0, b1. */
  auto gw_SxR = [&](const char *sg, const char *U, const std::string &pt, const Ten &X, int i, long long x0,
                    long long b0, long long b1) {
    AF a;
    const T *v = EWH(sg);
    for (int l = 0; l < X.nl; l++)
      a = a + rng(3, off3(U, pt) + (size_t)((X.off + ((long long)l * X.cnt + i) * X.J) / LS + (b0 - x0) / LS), 1,
                  (size_t)((b1 - b0) / LS), v + b0 / LS, 1, Sy(p256v[l]));
    return a;
  };
  // gi's limbs as one-limb tensors. With gcl in the parameter file (`policy`): g0, gc = clamp(gh).
  Ten gw_g0 = L.t[X_gi], gw_gc = L.t[X_gi];
  gw_g0.nl = gw_gc.nl = 1;
  gw_gc.off += gw_gc.cnt * gw_gc.J;
  auto gw_sumS = [&](const char *sg, const char *U, const std::string &pt, const Ten &X, int i, size_t n) {
    return hXSr(U, pt, X, i, 0, n, EW(sg));
  };
  const ForestForms ff = walk_forest(W, *K.F, Q.slPT);
  const int NS = norm_sum_count(xl), NR = norm_row_len(xl), NP = norm_product_count(xl);
  const T *gp = W.take(NS);
  {
    T *gpw = (T *)gp;
    const T *gm = K.gam;
    W.prep.push_back([=] { k_pows<<<1, 1>>>(gm, NS, gpw); });
  }
  {
    std::vector<const T *> r(muV);
    for (int i = 0; i < muV; i++) r[i] = &K.WP->chd[0].r[i];
    // The norms' sums of squares; their vector is never sent. The product layer starts from 2^-lPd
    // sum_i eq(rhoI, i) sum_s gam^s slot(Q.slVW + NS i + s). rhoI: K.rhoV's first lNI coordinates.
    Rec R = W.psc(0, r, Q.slWPL, 2, 2, &K.WP->out->tau);
    const T *eI = W.eq(K.rhoV, gw_lNI);
    AF C;
    for (int i = 0; i < NI; i++) C = C + rng(0, Q.slVW + NS * i, 1, NS, gp, 1, Sy(eI + i) * Sy(K.halves));
    W.layer(R, C);
    W.lin(W.shown(R, 0) - W.shown(R, 1));
    const T *e2v = W.eq(K.WP->out->r, gw_lNI + lPd - ls);
    AF v = W.shown(R, 0);
    for (int i = 0; i < NI; i++)
      for (int l = 0; l < xl; l++)
        v = v - rng(3,
                    i == 0
                        ? off3("UX", "Z:V") + ((size_t)l * d) / LS
                        : off3("UF", "Z:V") + (size_t)(L.t[X_xs].off + ((long long)l * L.t[X_xs].cnt + i - 1) * d) / LS,
                    1, d / LS, e2v + (size_t)i * Pd / LS, 1, Sy(gp + l));
    W.lin(v);
  }
  const Ten &XR_ = L.t[X_row];
  auto rowAt = [&](const std::string &pt, int i, int idx) {
    return off3("UR", pt) + (size_t)(XR_.off + (long long)i * XR_.J) + idx;
  };
  auto Dg = [&](const std::string &pt, int i, int idx) {
    return var(3, rowAt(pt, i, idx)) + AF(hE(128));
  };
  auto numA = [&](const std::string &pt, int i, int at, int n) {
    AF a = rng(3, rowAt(pt, i, at), 1, n, P256);
    HE c = hE(0);
    for (int q = 0; q < n; q++) c = hadd(c, hmul(hE(128), p256v[q]));
    a.c = c;
    return a;
  };
  const T *SIGC = W.cvec({hE(2), hE(1), hE(256), hE(65536), hE(1 << 24)});
  auto sigA = [&](const std::string &pt, int i) {
    AF a = rng(3, rowAt(pt, i, 44), 1, 5, SIGC);
    a.c = hmul(hE(128), hE(2 + 1 + 256 + 65536 + (1 << 24)));
    return a;
  };
  {
    std::vector<const T *> r(lt);
    for (int j = 0; j < lt; j++) r[j] = &K.tV.out->r[j];
    // TSC(P_V)'s end values: `NO` row values per norm instance, read from UR at its end point r_tV.
    const int NO = norm_row_value_count(xl);
    Rec R = W.psc(0, r, Q.slTV + 1, 2, (size_t)NO * NI, nullptr);
    const size_t slCl = Q.slTV, slPV = Q.slPV;
    W.lin(slot(slCl) - rng(0, slPV, 1, NP * NI, K.omV));
    W.row(R, slot(slCl));
    if (xl == 5) {
      // Five limbs; sigma = a + 2^24 b. Digits and relations: see k_norm_ledger_wide() (forward.cuh).
      const T *AC = W.cvec({hE(2), hE(1), hE(256), hE(65536)});
      auto aA = [&](const std::string &pt, int i) {
        AF a = rng(3, rowAt(pt, i, 58), 1, 4, AC);
        a.c = hmul(hE(128), hE(2 + 1 + 256 + 65536));
        return a;
      };
      for (int i = 0; i < NI; i++) {
        W.lin(W.shown(R, 4 * i) - aA("r_tV", i));
        W.lin(W.shown(R, 4 * i + 1) - numA("r_tV", i, 62, 2));
        W.lin(W.shown(R, 4 * i + 2) - numA("r_tV", i, 80, 3));
        W.lin(W.shown(R, 4 * i + 3) - numA("r_tV", i, 83, 3));
      }
      for (int i = 0; i < NI; i++) {
        AF cprev;
        for (int s = 0; s < 9; s++) {
          AF cn = s < 8 ? numA("P_V", i, 10 + 6 * s, 6) : Dg("P_V", i, 9);
          W.lin(slot(Q.slVW + 9 * i + s) + cprev - Dg("P_V", i, s) - cn * Sy(hE(256)));
          cprev = cn;
        }
      }
      for (int i = 0; i < NI; i++) {
        AF Vlo = numA("P_V", i, 0, 6), Vhi = numA("P_V", i, 6, 4),
           sg = aA("P_V", i) + numA("P_V", i, 62, 2) * Sy(h2(24)), l1 = numA("P_V", i, 64, 5),
           h1 = numA("P_V", i, 69, 5), Blo = numA("P_V", i, 74, 3), Bhi = numA("P_V", i, 77, 2),
           kap = Dg("P_V", i, 79), q1 = numA("P_V", i, 86, 3), rr = numA("P_V", i, 89, 5), rq = numA("P_V", i, 94, 5);
        auto Pj = [&](int j) { return slot(slPV + 7 * i + j); };
        W.lin(Pj(1) - Blo - Bhi * Sy(h2(23)));
        W.lin(Vlo - Pj(0) - Blo * Sy(h2(25)) - l1 + kap * Sy(h2(48)));
        W.lin(Vhi - Bhi - Pj(2) - kap);
        W.lin(l1 + h1 - sg * Sy(hE(2)));
        W.lin(Pj(4) + Pj(5) - q1);
        W.lin(Pj(6));
        W.lin(sg - Dg("P_V", i, 59) - Pj(3) * Sy(hE(2)) - q1 * Sy(h2(25)) - rr * Sy(hE(2)));
        W.lin(rr + rq - sg);
      }
    } else {
    for (int i = 0; i < NI; i++) {
      W.lin(W.shown(R, 3 * i) - sigA("r_tV", i));
      W.lin(W.shown(R, 3 * i + 1) - numA("r_tV", i, 57, 3));
      W.lin(W.shown(R, 3 * i + 2) - numA("r_tV", i, 60, 3));
    }
    for (int i = 0; i < NI; i++) {
      AF cprev;
      for (int s7 = 0; s7 < 7; s7++) {
        AF cn = s7 < 6 ? numA("P_V", i, 8 + 6 * s7, 6) : Dg("P_V", i, 7);
        W.lin(slot(Q.slVW + 7 * i + s7) + cprev - Dg("P_V", i, s7) - cn * Sy(hE(256)));
        cprev = cn;
      }
    }
    for (int i = 0; i < NI; i++) {
      AF Vv = numA("P_V", i, 0, 8), sg = sigA("P_V", i), l1 = numA("P_V", i, 49, 4), h1 = numA("P_V", i, 53, 4),
         p1 = numA("P_V", i, 63, 3), rr = numA("P_V", i, 66, 4), rq = numA("P_V", i, 70, 4), pv0 = slot(slPV + 3 * i),
         pv1 = slot(slPV + 3 * i + 1), pv2 = slot(slPV + 3 * i + 2);
      W.lin(Vv - pv0 - l1);
      W.lin(l1 + h1 - sg * Sy(hE(2)));
      W.lin(pv2 - p1);
      W.lin(sg - Dg("P_V", i, 45) - pv1 * Sy(hE(2)) - p1 * Sy(h2(25)) - rr * Sy(hE(2)));
      W.lin(rr + rq - sg);
    }
    }
  }
  // TSC(zt); Cl, its claimed sum, is slot Q.slTS. Of each product in its terms one factor, unless
  // public, is an end value: R of every norm instance, then sc and the selector's entries of every
  // layer with routed experts, each read from UR at the end point rt ("r"). TSC(zt)'s claim is not
  // recorded: gw_EN, the value of its terms with the factors that are not sent, goes into c_start.
  AF gw_EN;
  if (!K.ts.empty()) {
    std::vector<const T *> r(lt);
    for (int j = 0; j < lt; j++) r[j] = &K.ts[0].eng.out->r[j];
    Rec R = W.psc(0, r, Q.slTS + 1, 1, Q.nEndT, nullptr);
    gw_EN = W.kept_value(R, slot(Q.slTS));
    size_t j = 0;
    for (int i = 0; i < NN * Lam; i++) W.lin(W.shown(R, j++) - numA("r", i, xl == 5 ? 80 : 57, 6));
    auto cell = [&](const Ten &X, int i, int x) {
      AF a;
      for (int l = 0; l < X.nl; l++)
        a = a + var(3, off3("UR", "r") + (size_t)(X.off + ((long long)l * X.cnt + i) * X.J) + x, Sy(p256v[l]));
      return a;
    };
    if (mo) {
      for (int lm = 0; lm < LamM; lm++) W.lin(W.shown(R, j++) - cell(L.t[X_sc], lm, 0));
      for (int lm = 0; lm < LamM; lm++)
        for (int e = 0; e < NE; e++) W.lin(W.shown(R, j++) - cell(L.t[X_sel], lm, e));
    }
  }
  // The batched sumcheck (`stack`) is registered here: the product layers below start from its end
  // values for the gated products, 1 (H1) and 2 .. 1 + LamM (X_lamM). Its claims: g_walk_stack().
  {
    std::vector<const T *> r(Q.lamS);
    for (int j = 0; j < Q.lamS; j++) r[j] = Q.rrS + j;
    gw_RS = W.psc(1, r, Q.slSCH, 2, Q.nEndS, nullptr);
  }
  {
    std::vector<const T *> r(muH);
    for (int i = 0; i < muH; i++) r[i] = i < 9 ? &K.HL->chd[0].r[i] : &K.HL->chd[1].r[i - 9];
    Rec R = W.psc(0, r, Q.slPL, 2, 2, &K.HL->out->tau);
    W.layer(R, W.shown(gw_RS, 1));
    const T *e3 = W.eq(K.HL->out->r, ll + lPf - ls);
    AF vl = W.shown(R, 0), vr = W.shown(R, 1);
    for (int lam = 0; lam < Lam; lam++) {
      vl = vl + hXLr("UF", "Z:a", L.t[X_s], lam, f / LS, e3 + (size_t)lam * Pf / LS);
      vr = vr + hXLr("UF", "Z:a", L.t[X_u], lam, f / LS, e3 + (size_t)lam * Pf / LS);
    }
    W.lin(vl);
    W.lin(vr);
  }
  if (mo) {
    const int lh2 = ll - 1 + lfe;
    std::vector<const T *> r(mu2);
    for (int i = 0; i < mu2; i++) r[i] = i < 9 ? &K.HL2->chd[0].r[i] : &K.HL2->chd[1].r[i - 9];
    Rec R = W.psc(0, r, Q.slPL2, 2, 2, &K.HL2->out->tau);
    // The X_lamM, one per MoE layer, are joined at rhoM, the first ll - 1 coordinates of K.rhoH2.
    const T *eM = W.eq(K.rhoH2, ll - 1);
    AF C;
    for (int lm = 0; lm < LamM; lm++) C = C + W.shown(gw_RS, 2 + lm) * Sy(eM + lm);
    W.layer(R, C);
    const T *e3 = W.eq(K.HL2->out->r, lh2 - ls);
    AF vl = W.shown(R, 0), vr = W.shown(R, 1);
    for (int lm = 0; lm < LamM; lm++)
      for (int q = 0; q < 2; q++) {
        const Ten &X = L.t[q ? X_u : X_s];
        AF a;
        for (int l = 0; l < X.nl; l++)
          a = a + rng(3,
                      off3("UF", "Z:b") + (size_t)((X.off + ((long long)l * X.cnt + 2 * lm + 1) * X.J) / LS + fe / LS),
                      1, (size_t)(fe / LS), e3 + (size_t)lm * fe / LS, 1, Sy(p256v[l]));
        if (q)
          vr = vr + a;
        else
          vl = vl + a;
      }
    W.lin(vl);
    W.lin(vr);
  }
  // hsite(), hhigh(): a rounding site's terms in its low and high lookup segment, without constants
  // and el_c[lam] (see JF). SQ: quotient; k: shift; Srh: the remainder's `form` high bytes.
  auto hsite = [&](const AF &SQ, int k, int form, const AF &Srh) {
    AF ln = SQ * Sy(hneg(h2(k + tOf(k))));
    if (form == 0) return ln;
    return ln - Srh * Sy(h2(16));
  };
  auto hhigh = [&](const AF &Srh, int k, int form) {
    return Srh * Sy(h2(8 * form + 16 - (k < 16 ? 16 : k)));
  };
  const T *EWMU = nullptr;
  {
    std::vector<HE> mv(NR * NN);
    for (int x = 0; x < NR * NN; x++) mv[x] = hE(norm_digit_factor(xl, x % NR));
    const T *mc = W.cvec(mv), *ewr = EW("ROW");
    T *o = W.take(mv.size());
    const size_t nn = mv.size();
    W.prep.push_back([=] { k_vmul<<<1, 256>>>(ewr, mc, nn, o); });
    EWMU = o;
  }
  // LINE_n(lam): the terms, linear in table entries, of lookup segment n's keys at layer lam.
  auto LINl = [&](const std::string &n, int c, int lam) -> AF {
    const std::string Z = "Z", z = "z"; // the calls' common points: Z (long) and zt (short)
    const char *sg = n.c_str();
    const int lm = lam / 2;
    auto &kl = gw_kl(lam);
    AF ln;
    if (n == "G" || n == "U" || n == "GH" || n == "UH") {
      const bool gt = n[0] == 'G', hi = n.size() == 2;
      const int form = gt ? gw_hg : gw_hu, nb = gw_isM(lam) ? 2 : 1;
      const Ten &XQ = L.t[gt ? X_gi : X_u], &XH = L.t[gt ? X_rg : X_ru];
      // The quotient on [b0, b1): u, gi or, with the parameter file's line gcl, g0 + 256 gh.
      auto quotient = [&](long long b0, long long b1) {
        if (!gt || !L.s.gcl) return gw_SxR(sg, "UF", Z, XQ, lam, 0, b0, b1);
        return gw_SxR(sg, "UF", Z, gw_g0, lam, 0, b0, b1) +
               gw_SxR(sg, "UF", Z, L.t[X_gh], lam, 0, b0, b1) * Sy(hE(256));
      };
      for (int b = 0; b < nb; b++) {
        const long long b0 = (long long)b * fe, b1 = gw_isM(lam) ? b0 + fe : f;
        const int k = b ? (gt ? kl.kgR : kl.kuR) : (gt ? kl.kg : kl.ku);
        const AF Srh = form ? gw_SxR(sg, "UF", Z, XH, lam, 0, b0, b1) : AF();
        ln = ln + (hi ? hhigh(Srh, k, form) : hsite(quotient(b0, b1), k, form, Srh));
      }
      return ln;
    }
    if (n == "UT") return hSx(sg, "UF", Z, L.t[X_u], lam);
    if (n == "XR") return gw_sumS(sg, "UD", z, L.t[X_xr], lm, d) * Sy(hneg(h2(16)));
    if (n == "LR")
      return hsite(L.s.rcl
                       ? gw_sumS(sg, "UR", z, L.t[X_ly], lm, NE) * Sy(hE(8)) + gw_sumS(sg, "UR", z, L.t[X_l3], lm, NE)
                       : gw_sumS(sg, "UR", z, L.t[X_lr], lm, NE),
                   kl.kr, gw_hr, gw_hr ? gw_sumS(sg, "UR", z, L.t[X_rl], lm, NE) : AF());
    if (n == "L3") return gw_sumS(sg, "UR", z, L.t[X_l3], lm, NE) * Sy(hE(32));
    if (n == "CLR")
      return gw_sumS(sg, "UR", z, L.t[X_ly], lm, NE) +
             (gw_sumS(sg, "UR", z, L.t[X_lr], lm, NE) - gw_sumS(sg, "UR", z, L.t[X_l3], lm, NE)) * Sy(h2(25));
    if (n == "LT") return gw_sumS(sg, "UR", z, L.t[X_lr], lm, NE) * Sy(hE(2));
    if (n == "DD") { // - sum_x ew(x) (lr[x] + sum_(e > x) sel[e])
      ln = gw_sumS(sg, "UR", z, L.t[X_lr], lm, NE) * Sy(hneg(hE(1)));
      const Ten &XS = L.t[X_sel];
      const size_t s0 = off3("UR", z) + (size_t)(XS.off + (long long)lm * XS.J);
      const T *ewd = EW(sg);
      for (int x = 0; x + 1 < NE; x++)
        ln = ln + rng(3, s0 + x + 1, 1, (size_t)(NE - 1 - x), nullptr, 1, Sy(hneg(hE(1))) * Sy(ewd + x));
      return ln;
    }
    if (n == "LH") return hhigh(gw_sumS(sg, "UR", z, L.t[X_rl], lm, NE), kl.kr, gw_hr);
    if (n == "SEL") return gw_sumS(sg, "UR", z, L.t[X_sel], lm, NE) * Sy(hE(128));
    if (n == "MR") return gw_sumS(sg, "UR", z, L.t[X_sc], lm, 1) * Sy(h2(17));
    if (n == "N1" || n == "N2") {
      const bool one = n == "N1";
      return AF() - hSx(sg, "UF", Z, L.t[one ? X_aw : X_mw], lam) * Sy(h2(one ? kl.K1 : kl.K2)) -
             hSx(sg, "UF", Z, L.t[one ? X_ra : X_rm], lam) * Sy(h2(16));
    }
    if (n == "N1H" || n == "N2H") {
      const bool one = n == "N1H";
      return hSx(sg, "UF", Z, L.t[one ? X_ra : X_rm], lam) * Sy(h2(32 - (one ? kl.K1 : kl.K2)));
    }
    if (n == "O" || n == "D") {
      const bool isO = n == "O";
      const int form = isO ? gw_ho : gw_hd, hi = isO ? NW * lam + 1 : NW * lam + NW;
      return hsite(hSst(sg, hi, Z) - hSst(sg, hi - 1, Z), isO ? kl.ko : kl.kd, form,
                   form ? hSx(sg, "UF", Z, L.t[isO ? X_ro : X_rd], lam) : AF());
    }
    if (n == "V")
      return hsite(hSx(sg, "UF", Z, L.t[X_vw], lam), kl.kv, gw_hv, gw_hv ? hSx(sg, "UF", Z, L.t[X_rv], lam) : AF());
    if (n == "QK" || n == "QKH") {
      for (int b = 0; b < 2; b++) {
        const int k = b ? kl.kk : kl.kq;
        const long long b0 = b ? dq : 0, b1 = b ? gw_nqk : dq;
        const AF Srh = gw_hqk ? gw_SxR(sg, "UF", Z, L.t[X_rqk], lam, 0, b0, b1) : AF();
        ln = ln + (n == "QK" ? hsite(gw_SxR(sg, "UF", Z, L.t[X_qkw], lam, 0, b0, b1), k, gw_hqk, Srh)
                             : hhigh(Srh, k, gw_hqk));
      }
      return ln;
    }
    if (n == "OH" || n == "DH" || n == "VH") {
      const int k = n == "OH" ? kl.ko : n == "DH" ? kl.kd : kl.kv, form = n == "OH" ? gw_ho : n == "DH" ? gw_hd : gw_hv;
      return hhigh(hSx(sg, "UF", Z, L.t[n == "OH" ? X_ro : n == "DH" ? X_rd : X_rv], lam), k, form);
    }
    if (n == "ROW") return rng(3, rowAt(z, NN * lam, 0), 1, (size_t)NR * NN, EWMU);
    if (n == "CN1" || n == "CN2") {
      const bool one = n == "CN1";
      return hSx(sg, "UF", Z, L.t[one ? X_aw : X_mw], lam) +
             gw_sumS(sg, "UD", z, L.t[one ? X_a : X_m], lam, d) * Sy(h2(28));
    }
    if (n == "CQK")
      return hSx(sg, "UF", Z, L.t[X_qkw], lam) +
             (gw_SxR(sg, "UQ", Z, L.qa[0], lam, 0, 0, dq) + gw_SxR(sg, "UQ", Z, L.qa[1], lam, dq, dq, gw_nqk)) *
                 Sy(h2(28));
    if (n == "CV") return hSx(sg, "UF", Z, L.t[X_vw], lam) + hSx(sg, "UQ", Z, L.qa[2], lam) * Sy(h2(28));
    if (n == "F") return hSx(sg, "UF", Z, L.t[X_gi], lam) + hSx(sg, "UF", Z, L.t[X_s], lam) * Sy(h2(28));
    if (n == "G0") return hSx(sg, "UF", Z, gw_g0, lam);
    if (n == "CG") return hSx(sg, "UF", Z, L.t[X_gh], lam) + hSx(sg, "UF", Z, gw_gc, lam) * Sy(h2(28));
    printf("internal error: hiding has no line formula for segment %s\n", sg);
    name_the_end("walk-plan"); // serving: p258 end 1 walk-plan (M5g)
    exit(2);
  };
  auto gw_moeSeg = [](const std::string &n) {
    return n == "XR" || n == "LR" || n == "LT" || n == "DD" || n == "LH" || n == "SEL" || n == "MR" || n == "L3" ||
           n == "CLR";
  };
  // JF, the forest's join without matrix products: - C_end (its end value) - Cl - sum_c e_c sum_lam
  // el_c[lam] sum_s LINE_s(lam). e_c: call c's weight; el_c: eq at K.zk(c)'s first ll coordinates.
  AF gw_JF = AF() - ff.cend;
  if (!K.ts.empty()) gw_JF = gw_JF - slot(Q.slTS);
  for (int c = 0; c < NC; c++)
    for (int lam = 0; lam < Lam; lam++)
      for (auto &sn : policy.cl[c].seg)
        if (!gw_moeSeg(sn) || gw_isM(lam)) gw_JF = gw_JF - LINl(sn, c, lam) * Sy(K.ele[c] + lam);
  // The join has no claim of its own: it is in JF. One claim per MoE layer: the selector sums to 1.
  if (mo) {
    const std::string z = "z";
    const Ten &XS = L.t[X_sel];
    for (int lm = 0; lm < LamM; lm++)
      W.lin(rng(3, off3("UR", z) + (size_t)(XS.off + (long long)lm * XS.J), 1, (size_t)NE, nullptr));
  }
  // The lookup forest's three claims that read the multiplicities' bits, on TB_0, TB_1 and TP.
  walk_forest_planes(W, *K.F, ff, 3, OFFB[0], OFFB[1], OFFB[2], lmb);
  // The embedding: sum_kh eq(tkh, kh) sum_l 256^l U3[off3(UX, Z:E) + (l d) / LS + kh] = v_E.
  // v_E, the value of the last weight claim, is sent outside the sumchecks, in the slot Q.slVE.
  if (g == 0) {
    const T *ek = W.eq(K.tkh, lPd - ls);
    AF a;
    for (int l = 0; l < xl; l++)
      a = a + rng(3, off3("UX", "Z:E") + ((size_t)l * d) / LS, 1, (size_t)(d / LS), ek, 1, Sy(p256v[l]));
    W.lin(a - slot(Q.slVE));
  }
  // The summands of c_start, in the order in which gs::Fork::walk10() weights them.
  gClaim = std::vector<AF>();
  gClaim.push_back(gw_JF);
  if (!K.ts.empty()) gClaim.push_back(gw_EN);

  walk_tail_f(W, HT, spCP + 1, Q.slPZ, HFUN);
  return 0;
}
// Sets up gs::Fork (prover/head/stack.cuh): batched sumcheck, weight claims, their merge into one
// pending claim on the weight table. Records the last claims: the sumcheck's three, the merge's.
inline int Prover::g_walk_stack(int g) {
  using namespace hw;
  Fork &K = FK[g];
  const LayerLayout &L = K.g.L;
  HFk &Q = *K.h;
  hw::Walk &W = Q.W;
  gs::In gin;
  gin.g = g;
  gin.NC = NC;
  gin.ls = ls;
  gin.lt = lt;
  for (int c = 0; c < NC; c++) {
    gin.cname.push_back(policy.cl[c].name);
    gin.cseg.push_back(policy.cl[c].seg);
    gin.lJ.push_back(K.lJ[c]);
    gin.zj.push_back((const T *)K.zj(c));
    gin.el.push_back(K.ele[c]); // e_c el_c (see JF): gs::In::el includes e_c
    gin.ec.push_back(K.ec[c]);
    gin.rt.push_back(nullptr);
  }
  for (int c : tscCalls) {
    gin.TS.push_back(c);
    gin.rt[c] = K.ts[0].eng.out->r; // the one point rt
  }
  gin.pos = [&L, this](const std::string &sg, long long x) {
    return (long long)L.pos(L.segid(sg.c_str()), x);
  };
  gin.off3 = [=](const char *U, const std::string &pt) {
    return off3(U, pt);
  };
  gin.ten = [&L, this](const char *n) {
    static const std::map<std::string, int> id = {{"m", X_m},   {"a", X_a},   {"xr", X_xr},   {"sel", X_sel},
                                                  {"sc", X_sc}, {"lr", X_lr}, {"row", X_row}, {"xs", X_xs}};
    const Ten &t = std::string(n) == "o" ? L.qa[3] : L.t[id.at(n)];
    gs::TenG o;
    o.nl = t.nl;
    o.cnt = t.cnt;
    o.J = t.J;
    o.off = t.off;
    return o;
  };
  gin.kw = [&K, g, this](int lam, const char *w) {
    const auto &k = K.g.kl[g * Lam + lam];
    const std::string n = w;
    return n == "kg"    ? k.kg
           : n == "ku"  ? k.ku
           : n == "kv"  ? k.kv
           : n == "ko"  ? k.ko
           : n == "kd"  ? k.kd
           : n == "kq"  ? k.kq
           : n == "kr"  ? k.kr
           : n == "kgR" ? k.kgR
                        : k.kuR;
  };
  gin.Hh = K.sHH;
  gin.U3 = Q.vec[3];
  gin.MSG = MSG;
  gin.slSCH = Q.slSCH;
  gin.slVE = Q.slVE;
  gin.slMSC = Q.slMSC;
  gin.slEndT = K.ts.empty() ? 0 : Q.slTS + K.ts[0].eng.nmsg() - Q.nEndT;
  gin.NN = NN;
  gin.rr = Q.rrS;
  gin.c = merge_mask;
  gin.kept = [&K](const std::string &sg, int i) {
    return K.kept.at(sg + std::to_string(i));
  };
  for (auto &w : K.wk) {
    gin.tab[w.kind] = w.tab;
    gin.tabw[w.kind] = w.tab;
    gin.ntab[w.kind] = w.ntab;
  }
  gin.AH = K.AH;
  gin.lfc = K.lfc;
  gin.ropeC = K.g.b.Cr;
  gin.ropeS = K.g.b.Sr;
  for (int lam = 0; lam < Lam; lam++) {
    gin.rp.push_back(K.g.kl[g * Lam + lam].rp);
  }

  {
    Fork *Kp = &K;
    auto *gf = &make_weight_cells;
    i8 *sl = slab;
    // The group's cells of a weight kind: in place if `weights` has them, else built in `slab`.
    gin.cells = [Kp, gf, sl, g, this](int kind) -> const i8 * {
      for (auto &w : Kp->wk) {
        if (w.kind == kind) {
          if (w.lm.n != 1) {
            printf("internal error: weight kind %d is split into %d parts, but the direct pass needs it in one part\n",
                   kind, w.lm.n);
            name_the_end("walk-plan"); // serving: p258 end 1 walk-plan (M5g)
            exit(2);
          }
          if (const i8 *cells = weights->cells_in_place(kind, g)) return cells;
          (*gf)(g, w, 0);
          return (const i8 *)sl;
        }
      }
      printf("internal error: no weight of kind %d was found for the direct pass\n", kind);
      name_the_end("walk-plan"); // serving: p258 end 1 walk-plan (M5g)
      exit(2);
    };
  }
  if (g == 0) {
    gin.tok = tokd;
    gin.ztE = K.ZE + ls;
    gin.pk = K.pk;
  }
  if (g == 0) {
    gs::Shape s8;
    s8.Lam = Lam;
    s8.LamM = LamM;
    s8.mo = mo;
    s8.d = d;
    s8.dq = dq;
    s8.dkv = dkv;
    s8.dh = dh;
    s8.H = H;
    s8.Gk = Gk;
    s8.f = f;
    s8.fe = fe;
    s8.NE = NE;
    s8.lkW = lkW;
    s8.L = Lr;
    s8.V = Vv;
    gs::Fork P;
    gs::Shared t;
    t.s = s8;
    P.plan(t, gin);
    SHg.setup(s8, P.lam_s, P.NV + 1);
    // SHg.Z.fv: the merged weight claim's value without its mask, a request's data in a Place::Own
    // buffer. Each run writes it before reading it; the restore in memory_as_set_up.cuh covers it.
    fork_memory.of_a_request.push_back({SHg.Z.fv, sizeof(T)});
  }
  Q.GS.setup(SHg, gin);
  // fvp = fv + beta c: that value under its mask c; the proof sends it. With fv it would reveal c.
  fork_memory.of_a_request.push_back({Q.GS.fvp, sizeof(T)});
  if (Q.GS.lam_s != Q.lamS || Q.GS.nshown() != Q.nEndS || Q.GS.NV != Q.NV) {
    printf("internal error: the stack plan does not match the hiding plan: lam_s %d vs %d, shown values %zu vs %zu, NV %d vs "
           "%d\n",
           Q.GS.lam_s, Q.lamS, Q.GS.nshown(), Q.nEndS, Q.GS.NV, Q.NV);
    name_the_end("walk-plan"); // serving: p258 end 1 walk-plan (M5g)
    return 4;
  }
  Q.GS.walk10(W, gw_RS, gClaim);
  Q.GS.walk11(W);
  return 0;
}
// Prepares the row jobs for gs::Fork's run (`RowJob`: each row's inner product with each phi): in
// group 0 over the embedding; with routed experts over their two input projections (kind 7 and 8).
inline int Prover::g_walk_end(int g) {
  using namespace hw;
  Fork &K = FK[g];
  HFk &Q = *K.h;
  hw::Walk &W = Q.W;
  if (g == 0) {
    for (size_t p = 0; p < npE; p++) {
      RowJob j;
      j.N = 1 << SHg.tE;
      j.src = embC;
      j.L = d;
      j.rows = (size_t)4 * Vv / npE;
      for (int q = 0; q < j.N; q++) {
        j.phi[q] = Q.GS.phiE(q);
        j.out[q] = Q.GS.beRaw(q) + p * j.rows;
      }
      j.prep();
      rscB = std::max(rscB, j.scratch());
      K.jobE.push_back(j);
    }
  }
  if (mo) {
    for (auto &w : K.wk) {
      if (w.kind == 7 || w.kind == 8) {
        for (int p = 0; p < w.lm.n; p++) {
          RowJob j;
          j.N = 1;
          j.src = slab;
          j.L = (size_t)fe >> SHg.tG;
          j.rows = (w.ntab / w.lm.n) << SHg.tG;
          j.phi[0] = Q.GS.phiLow(w.kind);
          j.out[0] = Q.GS.tab2(w.kind) + (size_t)p * j.rows;
          j.prep();
          rscB = std::max(rscB, j.scratch());
          Q.job2[w.kind - 7].push_back(j);
        }
      }
    }
  }
  return 0;
}

// Once per layer group, after g_setup_token_ends() fills K.kept; g_setup_equations() then seals W.
inline int Prover::g_setup_walk(int g) {

  if (int rc = g_walk_spaces(g)) return rc;
  if (int rc = g_walk_relations(g)) return rc;
  if (int rc = g_walk_stack(g)) return rc;
  if (int rc = g_walk_end(g)) return rc;
  return 0;
}
