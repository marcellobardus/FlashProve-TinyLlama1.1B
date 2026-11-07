// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// Set-up of the layer sub-proofs (`fork` in identifiers), one for each layer group g of Lam layers.
// setup_g_forks(), once per process, fills FK[g] (`K` below) with what every run (one sub-proof for
// one request) reuses. Group 0 comes first: it allocates buffers and engines that all groups share.
#pragma once
#include "common/blas.h"
#include "prover/prover.h"
#include "common/zfold_turned.cuh"

// Returns a buffer that the layer sub-proofs use in turn. Sub-proof 0 allocates each buffer; later
// ones receive the same buffers in the same order, so all must request the same sequence of sizes.
inline void *Prover::shared_alloc(int g, size_t bytes, bool middle) {
  void *p;
  if (g == 0) {
    if (middle)
      // A `middle` buffer is in fork_memory's slab (scratch that sub-proofs share), for the arena's
      // third user (g_setup_engines()). Members `arena` and `slab` (weight cells) are unrelated.
      CK(fork_alloc(&p, bytes, Place::Fork));
    else
      CK(cudaMalloc(&p, bytes));
    CK(cudaMemset(p, 0, bytes));
    arena.push_back(p);
    arenaB.push_back(bytes);
  } else {
    if (ai >= arena.size() || arenaB[ai] != bytes) {
      printf("internal error: shared memory: sub-proof %d requests %zu bytes for buffer %zu, but sub-proof 0 requested "
             "%zu there\n", g,
             bytes, ai, ai < arena.size() ? arenaB[ai] : (size_t)0);
      exit(2);
    }
    p = arena[ai++];
  }
  return p;
}

// Builds K.g, group g's forward-pass object (ForwardPass, forward.cuh). Its layout K.g.L (matrices; lookup
// calls with named segments of keys) is g's own. For g > 0 the rest copies group 0's, buffers too.
inline int Prover::g_setup_layout(int g) {
  FK[g].h = new HFk;
  Fork &K = FK[g];
  const LayerLayout &L = K.g.L;
  fork_memory.setup_layer_fork(g);
  ai = 0;
  for (auto &c : policy.cl) {
    K.lists += c.name + ":";
    for (size_t i = 0; i < c.seg.size(); i++) {
      K.lists += (i ? "," : "") + c.seg[i];
    }
    K.lists += ";";
  }
  K.lists.pop_back();
  S = LayerShape{lt,
           ll,
           d,
           H,
           Gk,
           dh,
           f,
           (int)policy["fe"],
           mo ? NE : 0,
           {(int)KL[7], (int)KL[9]}, // M1, M2 and K1, K2 of the first layer's record
           {(int)KL[8], (int)KL[10]},
           {(long long)(policy["eps1_lo"] | ((u64)policy["eps1_hi"] << 32)),
            (long long)(policy["eps2_lo"] | ((u64)policy["eps2_hi"] << 32))},
           g,
           mo ? 3 : 2,
           nmin,
           la};

  digit_forms(S);
  S.xl = xl;
  S.att = 1;
  S.lists = K.lists.c_str();
  K.g.L = make_layer_layout(S);
  K.g.kl.resize(Lr);
  for (int l = 0; l < Lr; l++) K.g.kl[l] = layer_constants(l);
  K.g.seed = seed;
  if (g == 0) {
    weights->bind(K.g);
    if (weight_stream.experts_by_layer) {
      weight_stream.last_turn = &last_of_wave; // false while a group's weights have a later reader
      K.g.routed_wanted = [this](int l) {
        weight_stream.layer_ready(l);
      };
      weight_stream.down_is_asked_for_apart = true; // `layer_ready` may skip the down tensor
      K.g.routed_down_wanted = [this](int l) {
        weight_stream.layer_down_ready(l);
      };
      // If stretches_of() is 1 (at most one MoE layer), one use of its experts spans the forward
      // pass and the row passes in g_weight_tables() (layer_prove.cuh), which reports them as used.
      K.g.routed_read = [this](int l) {
        if (weight_stream.stretches_of(l / Lam) == 2) weight_stream.layer_used(l);
      };
    }
    K.g.alloc(false); // its scratch memory is handled in g_setup_engines()
    // fork_memory repeats set-up's writes into the slab for every run. But each run's forward pass,
    // forward(1), calls ForwardPass::pad() before it stores a lookup key (here forward_pass_buffers_last is
    // false), and pad() rewrites each key array in full. So these arrays' zeroing is not repeated,
    // except in the 256 spare bytes after each, which pad() does not write. forward(0) skips pad().
    for (size_t c = 0; c < K.g.keyN.size(); c++)
      fork_memory.every_run_writes(K.g.b.key[c], K.g.keyN[c] * sizeof(*K.g.b.key[c]));
    // From the slab ForwardPass::alloc() takes the key arrays first and xst, the residual stream, next.
    fork_memory.late_on_keys = args.late_on_keys;
    if (fork_memory.in_slab(K.g.b.xst) && (char *)K.g.b.key[0] == fork_memory.slab + fork_memory.begin_of[0])
      fork_memory.keys_end = (size_t)((char *)K.g.b.xst - fork_memory.slab);
    else if (args.late_on_keys)
      printf("# -memory late-on-keys: the forward pass's keys are not the first buffers of the slab (a mode of the "
             "memory test?), so the setting has no effect\n");
    K.g.b.Ts = Tsd;
    K.g.b.Sig = Sigd;

    K.g.fa = &attF;

    K.g.rope_tables((double)(policy["theta0_lo"] | ((u64)policy["theta0_hi"] << 32)));

  } else {
    const LayerLayout Lg = K.g.L;
    K.g = FK[0].g;
    K.g.L = Lg;
  }
  {
    ForwardPass::WP probe;
    if (weights->planes(0, 0, probe)) {
      K.g.wres = [this](int matrix_class, int layer) {
        ForwardPass::WP w;
        weights->planes(matrix_class, layer, w);
        return w;
      };
    }
  }
  if ((int)L.call.size() != NC) {
    printf("internal error: the layout has a different number of calls from the parameter file\n");
    return refused("layout-calls"); // serving: p258 end 1 layout-calls
  }
  return 0;
}

// Lists in K.M sub-proof g's committed byte matrices: activations, multiplicity (`plane`) bits, the
// residual stream's limbs; then xg, committed by g - 1 if g > 0. K.wk: each weight kind it reads.
inline int Prover::g_setup_matrices(int g) {
  Fork &K = FK[g];
  const LayerLayout &L = K.g.L;
  auto put = [&](i8 *buf, size_t m, size_t cells, int n) {
    LM lm{(int)K.M.size(), n};
    const size_t per = cells / n;
    for (int p = 0; p < n; p++) {
      MatD M{};
      M.m = m;
      M.k = per / m;
      M.v = buf + p * per;
      K.M.push_back(M);
    }
    return lm;
  };
  auto big = [&](i8 *&p, size_t feats, size_t m, int n) {
    const size_t unit = m * n, cells = (feats * NT + unit - 1) / unit * unit;
    if (g == 0) {
      CK(cudaFree(p));
      // The new buffer, rounded up to whole units of m * n cells, is in the slab. All layer
      // sub-proofs share it.
      CK(fork_alloc(&p, cells, Place::Fork));
      CK(cudaMemset(p, 0, cells));
    }
    K.filler.push_back({p + feats * NT, cells - feats * NT});
    return put(p, m, cells, n);
  };
  K.af = big(K.g.b.af, L.G[0], (size_t)1 << policy["lmaf"], nAF);
  K.ad = big(K.g.b.ad, L.G[1], (size_t)1 << policy["lmad"], nAD);
  K.ar = big(K.g.b.ar, L.G[2], (size_t)1 << policy["lmar"], 1);
  for (int c = 0; c < NC; c++) {
    K.lJ[c] = L.call[c].lJ;
    K.n[c] = ll + K.lJ[c] + lt;
  }
  if (g == 0) {
    // The calls run as one lookup, the lookup forest, with each call's keys in trees of 2^dT keys.
    layer_calls.clear();
    for (int c = 0; c < NC; c++)
      layer_calls.push_back(ForestCall{K.n[c], kk_[c], knd[c] >= 2 ? 1 : 0,
                                       knd[c] == 2   ? tFCd
                                       : knd[c] == 3 ? tFSd
                                       : knd[c] == 4 ? tFLd
                                                     : nullptr,
                                       (u32)t0_[c]});
    layer_forest = forest_plan(layer_calls, identity_cap());
    // If a tree has at least 2^la keys, all calls' points end in the same la coordinates: the long
    // point Z of g_setup_points().
    if (layer_forest.dT < la) {
      printf("policy: a tree of a layer sub-proof's lookups must have at least 2^la keys; raise nmin "
             "[forest-depth-G]\n");
      return refused("forest-depth-G"); // serving: p258 end 1 forest-depth-G
    }
    if (lmb < 5 || lmb > layer_forest.kT + 5) {
      printf("policy: the line lmb must be from 5 to %d [forest-planes]\n", layer_forest.kT + 5);
      return refused("forest-planes"); // serving: p258 end 1 forest-planes
    }
    for (int c = 0; c < NC; c++)
      if (io[c] != layer_forest.iota[c]) {
        printf("policy: the line iota_%d must be %d: log2 of the call's keys, %d, minus log2 of a tree's keys, %d "
               "[shape-call]\n",
               c, layer_forest.iota[c], K.n[c], layer_forest.dT);
        return refused("shape-call"); // serving: p258 end 1 shape-call
      }
  }
  // Matrix b: bit j of the multiplicity of row t of call c's table is byte (c * 2^kT + t) * 32 + j,
  // as in Forest::bits(0). In the verifier: forestPlaneCols() (verifier/src/vf_lookup_argument.h).
  nb = layer_forest.plane_bits();
  {
    i8 *bb = (i8 *)shared_alloc(g, nb);
    K.b = put(bb, (size_t)1 << lmb, nb, 1);
  }
  K.qa = big(K.g.b.qa, L.Gqa, (size_t)1 << policy["lmq"], 1);
  if (g == 0) {
    CK(cudaFree(K.g.b.xb0));
    CK(cudaFree(K.g.b.xb1));
  }
  K.g.b.xb0 = XBv[g];
  K.g.b.xb1 = XBv[g + 1];
  K.xn = put(XBv[g + 1], mx, kx * mx, 1);
  K.xg = put(XBv[g], mx, kx * mx, 1);
  K.NR = (int)K.M.size();
  if (K.xg.first != iXG || K.xn.first != iXN) {
    printf("internal error: the list numbers of xn and xg do not match the expected indices iXN and iXG\n");
    return refused("border-index"); // serving: p258 end 1 border-index
  }
  K.nWT = 0;
  for (int kd : kinds) {
    const std::string kn = KN[kd];
    WeightKind w{};
    w.kind = kd;
    w.L = kind_row_length(kd);
    w.bytes = 2 * kind_cells(kd);
    if (!weights->cells_number(kd, g, w.bytes)) {
      printf("internal error: the weight source holds a different number of cells of kind %s from what the prover "
             "expects\n",
             kn.c_str());
      return refused("cells-number"); // serving: p258 end 1 cells-number
    }

    if (policy["wb_" + kn] != 13) {
      printf("policy: the line wb_%s must be 13\n", kn.c_str());
      return refused("policy-weight-kind"); // serving: p258 end 1 policy-weight-kind
    }
    w.buf = slab;
    w.lm = LM{-1, (int)parts_of_kind(kd)};

    w.ntab = w.bytes / w.L;
    K.nWT += w.ntab;
    K.wk.push_back(w);
  }
  NM = (int)K.M.size();

  for (int i = 0; i < NM; i++) {
    MatD &M = K.M[i];
    M.support = (uint4 *)shared_alloc(g, M.k * (ETAU / 8));
    M.sign = (uint4 *)shared_alloc(g, M.k * (ETAU / 8));
    M.Cf = (i8 *)shared_alloc(g, M.k * ETAU);
    M.CfT = (i8 *)shared_alloc(g, (size_t)zfold_kp((int)M.k) * ETAU);
    M.Z = (int *)shared_alloc(g, M.m * ETAU * 4);
  }

  return 0;
}

// Creates the engines: lookup forest, product layers, sumchecks over tokens. Allocates the arena.
inline int Prover::g_setup_engines(int g) {
  Fork &K = FK[g];
  const LayerLayout &L = K.g.L;

  if (g == 0) {
    // The arena is memory that three steps of a run use in turn. First: the forward pass's scratch.
    fork_memory.arena_begin();
    K.g.alloc_scratch();
    attF.init(lt, policy["lB"], H, l2i(Gk), ldh, policy["S"], policy["rx"], policy["kFM"], policy["kFD"]);
    fork_memory.arena_next();
    // The arena's second user: the lookup forest, whose scratch memory is one allocation.
    K.F = new Forest;
    K.FG = new PtGraph;
    K.F->init(layer_calls, identity_cap(), 18, args.layer_rebuilt);
    fork_memory.arena_next();
  } else {
    K.F = FK[0].F;
    K.FG = FK[0].FG;
  }
  // From here to arena_end(): the arena's third user, a run's steps from middle_begin() to
  // late_begin(). middle_begin() repeats what set-up writes there.
  if (g == 0) fork_memory.arena_middle = true;
  {
    // Counts each tree's padding keys. Column q = lam * J + j of call c has a key for each token t,
    // key q * NT + t, all in the call's tree number floor(q / 2^(dT - lt)), since dT >= la > lt.
    std::vector<u32> pb((size_t)layer_forest.NT, 0);
    K.padded.resize(NC);
    for (int c = 0; c < NC; c++) {
      const size_t J = (size_t)1 << K.lJ[c];
      std::vector<char> real(J, 0);
      for (auto &p : L.call[c].pc) {
        for (long long x = 0; x < p.size; x++) {
          // 2: a column of an MoE segment, which contains only padding keys in a dense layer.
          real[p.off + x] = L.seg[p.seg].moe ? 2 : 1;
        }
      }
      ForestRuns runs;
      for (int lam = 0; lam < Lam; lam++)
        for (size_t j = 0; j < J; j++)
          if (real[j] == 0 || (real[j] == 2 && !(is_moe_layer(g * Lam + lam)))) {
            const size_t column = ((size_t)lam << K.lJ[c]) + j;
            pb[layer_forest.first[c] + (column >> (layer_forest.dT - lt))] += (u32)NT;
            runs.add(column << lt, (column + 1) << lt);
          }
      const auto first = pb.begin() + layer_forest.first[c];
      K.padded[c] = forest_padded_from(layer_forest.dT, layer_forest.iota[c], runs,
                                       std::vector<u32>(first, first + ((size_t)1 << layer_forest.iota[c])));
    }
    CK(cudaMalloc(&K.pub, pb.size() * 4));
    CK(cudaMemcpy(K.pub, pb.data(), pb.size() * 4, cudaMemcpyHostToDevice));
    K.sPT = (T *)shared_alloc(g, sizeof(T) * layer_forest.nmsg());
  }

  muH = ll + lPf + lt;
  muV = lNI + lPd + lt;

  if (g == 0) {
    K.HL = new PtPL;
    K.WP = new PtWPL;
    K.HL->init(muH);
    if (mo) {
      K.HL2 = new PtPL;
      K.HL2->init(ll - 1 + lfe + lt);
    }
    K.WP->init(muV);
    K.tV.init(lt, norm_product_count(xl) * NI, TscShape{(size_t)norm_row_value_count(xl) * NI, 2});
  } else {
    K.HL = FK[0].HL;
    K.HL2 = FK[0].HL2;
    K.WP = FK[0].WP;
    K.tV = FK[0].tV;
  }
  // tscCalls: calls with terms in K.ts, the sumcheck over tokens at zt; `terms`: how many terms.
  // All calls share the token point zt, so one such sumcheck serves them all.
  tscCalls = std::vector<int>();
  int terms = 0;
  for (int c = 0; c < NC; c++) {
    int nt = 0;
    for (auto &s : policy.cl[c].seg) {
      if (s == "N1" || s == "N2") {
        nt += Lam;
      }
      if (s == "QK") {
        nt += Lam * dh;
      }
      if (mo) {
        if (s == "G" || s == "U" || s == "D" || s == "DD" || s == "MR") {
          nt += LamM * NE;
        }
        if (s == "XR") {
          nt += LamM;
        }
      }
    }
    if (nt) tscCalls.push_back(c);
    terms += nt;
  }
  // nEndT counts the end values (factors' values at the last point) that this sumcheck sends (see
  // g_setup_token_ends()): the norms' R, the MoE layers' sc, then NE selector entries for each.
  K.h->nEndT = terms ? (size_t)NN * Lam + (mo ? (size_t)LamM * (1 + NE) : 0) : 0;
  if (terms) {
    TokenSumcheck q{};
    q.NTm = terms;
    if (g == 0) {
      q.eng.init(lt, terms, TscShape{K.h->nEndT, 1});
    } else {
      q.eng = FK[0].ts[0].eng;
    }
    alloc_elements(g, q.Fv, (size_t)terms * NT, true);
    alloc_elements(g, q.Gv, (size_t)terms * NT, true);
    alloc_elements(g, q.co, terms);
    alloc_elements(g, q.msg, q.eng.nmsg());
    alloc_elements(g, q.yb, 32);
    K.ts.push_back(q);
  }
  // The other large arrays of the arena's third user, which also has Fv and Gv above and the
  // product layers' scratch memory.
  alloc_elements(g, K.A, (size_t)1 << muV, true);
  // k_bg() (base.cuh) writes rows 0 to NI * Pd - 1 of K.A, of NT = 2^lt elements each, in full
  // before anything reads K.A. So sub-proof 0's set-up registers them as written by every run, and
  // only the rows after them get set-up's zeros again.
  if (g == 0 && NT == 1 << lt)
    fork_memory.every_run_writes(K.A, std::min((size_t)NI * Pd * NT, (size_t)1 << muV) * sizeof(T));
  // K.FV, K.GV: each term's two factors in K.tV's sumcheck, norm_product_count(xl) terms for each of NI norms.
  alloc_elements(g, K.FV, (size_t)norm_product_count(xl) * NI * NT, true);
  alloc_elements(g, K.GV, (size_t)norm_product_count(xl) * NI * NT, true);
  alloc_elements(g, K.gnp, (size_t)32 * NT, true);
  K.Ws = (long long *)shared_alloc(g, (size_t)NI * NT * norm_sum_count(xl) * 8, true);
  if (mo) K.KH2 = (u16 *)shared_alloc(g, (size_t)2 * LamM * fe * NT * 2, true);
  for (int i = 0; i < (mo ? 4 : 2); i++) {
    K.aT[i] = (i8 *)shared_alloc(g, (size_t)NT * std::max(d, mo ? fe : 0), true);
    K.Cg[i] = (int *)shared_alloc(g, (size_t)std::max(dh, mo ? NE : 0) * 8 * EDEG * NT * 4, true);
  }
  // A separate cuBLAS workspace: the forward pass's (ForwardPass::blas_room) belongs to the arena's first
  // user, so these steps' buffers may overlap it.
  K.blas_room = (char *)shared_alloc(g, blas::BLAS_WORKSPACE, true);
  if (g == 0) {
    fork_memory.arena_end();
    printf(
        "# scratch memory that three steps of a layer sub-proof's run use one after the other, GB: the forward pass "
        "%.3f; the forest of the lookups %.3f (%d trees of 2^%d keys; %d levels made again from the keys); the steps "
        "from the norms to the gated products %.3f\n",
        fork_memory.arena_users[0] / 1e9, fork_memory.arena_users[1] / 1e9, layer_forest.NT, layer_forest.dT,
        args.layer_rebuilt, fork_memory.arena_users[2] / 1e9);
  }
  return 0;
}

// Names the evaluation points of sub-proof g and records which matrices are read at which point.
inline int Prover::g_setup_points(int g) {
  Fork &K = FK[g];
  // Long points (K.LPn) have la coordinates, short ones (K.SPn) lt. All calls share Z and zt ("z");
  // those in tscCalls also Y and rt ("r"). Z:V, Z:a, Z:b: from K.WP, K.HL, K.HL2; r_tV: from K.tV.
  K.LPn.push_back({"Z", -1});
  K.LPn.push_back({"Z:V", -1});
  if (!tscCalls.empty()) K.LPn.push_back({"Y", -1});
  K.LPn.push_back({"Z:a", -1});
  if (mo) {
    K.LPn.push_back({"Z:b", -1});
  }
  if (g == 0) {
    K.LPn.push_back({"Z:E", -1}); // Must be the last long point; only xg is read there.
  }
  K.SPn.push_back({"z", -1});
  K.SPn.push_back({"P_V", -1});
  K.SPn.push_back({"r_tV", -1});
  if (!tscCalls.empty()) K.SPn.push_back({"r", -1});
  rdAF = std::vector<std::string>();
  rdQA = std::vector<std::string>();
  rdXG = std::vector<std::string>();
  rdXN = std::vector<std::string>();
  rdAD = std::vector<std::string>();
  rdAR = std::vector<std::string>();

  // z..: some segment reads the matrix at Z or zt; r..: at Y or rt.
  bool zAF = 0, zQA = 0, zXG = 0, zXN = 0, zAD = 0, zAR = 0, rAF = 0, rXG = 0, rAD = 0, rAR = 0;
  for (int c = 0; c < NC; c++) {
    for (auto &n : policy.cl[c].seg) {
      const bool sm = n == "XR" || n == "LR" || n == "LT" || n == "DD" || n == "LH" || n == "SEL" || n == "MR" ||
                      n == "L3" || n == "CLR"; // Small MoE segments read nothing at a long point.
      if (n == "ROW" || (sm && n != "XR")) {
        zAR = 1;
      } else if (!sm) {
        zAF = 1;
      }
      if (n == "XR" || n == "LR") {
        zAD = 1;
      }
      if (mo && (n == "G" || n == "U" || n == "XR")) {
        rAR = rAD = 1;
      }
      if (mo && (n == "D" || n == "DD" || n == "MR")) {
        rAR = 1;
      }
      if (n == "CQK" || n == "CV") {
        zQA = 1;
      }
      if (n == "O") {
        zXG = 1;
      }
      if (n == "D") {
        zXN = 1;
      }
      if (n == "G" || n == "U" || n == "V" || n == "CN1" || n == "CN2") {
        zAD = 1;
      }
      if (n == "N1" || n == "N2") {
        rAF = rAR = 1;
        if (n == "N1") {
          rXG = 1;
        }
      }
      if (n == "QK") {
        rAD = 1;
      }
    }
  }
  if (zAF) rdAF.push_back("Z");
  if (zQA) rdQA.push_back("Z");
  if (zXG) rdXG.push_back("Z");
  if (zXN) rdXN.push_back("Z");
  if (zAD) rdAD.push_back("z");
  if (zAR) rdAR.push_back("z");
  if (rAF) rdAF.push_back("Y");
  if (rXG) rdXG.push_back("Y");
  if (rAD) rdAD.push_back("r");
  if (rAR) rdAR.push_back("r");
  rdAF.push_back("Z:V");
  rdAF.push_back("Z:a");
  if (mo) {
    rdAF.push_back("Z:b");
  }
  rdXG.push_back("Z:V");
  if (g == 0) {
    rdXG.push_back("Z:E");
  }
  rdAR.push_back("P_V");
  rdAR.push_back("r_tV");
  for (auto &p : K.LPn) {
    if (p.first == "Z:E") {
      CK(cudaMalloc(&K.phiP[p.first], LA * sizeof(T)));
      continue;
    }
    alloc_elements(g, K.phiP[p.first], LA);
  }
  for (auto &p : K.SPn) {
    alloc_elements(g, K.phiP[p.first], NT);
  }
  auto cnt = [&](const std::vector<std::pair<std::string, int>> &pts, const std::vector<std::string> &rd) {
    size_t n = 0;
    for (auto &p : pts) {
      if (has(rd, p.first)) {
        n++;
      }
    }
    return n;
  };
  K.nW = cnt(K.LPn, rdAF) * (cells_of(K, K.af) / LA) + ((cnt(K.LPn, rdQA) * (cells_of(K, K.qa) / LA))) +
         (cnt(K.LPn, rdXG) + cnt(K.LPn, rdXN)) * (kx * mx / LA) + cnt(K.SPn, rdAD) * (cells_of(K, K.ad) / NT) +
         cnt(K.SPn, rdAR) * (cells_of(K, K.ar) / NT) + ((cells_of(K, K.qa) / NT)) + (size_t)3 * K.M[K.b.first].k;
  return 0;
}

// Allocates message arrays and evaluation tables: for each point where a matrix is read, in K.LPn's
// or K.SPn's order, its rows' inner products with eq(point, .). RowJob (base.cuh) computes them.
inline int Prover::g_setup_tables(int g) {
  Fork &K = FK[g];
  {
    const size_t n1 = K.nWT + (size_t)NI * Pd;
    CK(fork_alloc_in_slab(&K.sWT, n1 * sizeof(T)));
    CK(cudaMemset(K.sWT, 0, n1 * sizeof(T)));
    K.sHV = K.sWT + K.nWT;
  }
  alloc_elements(g, K.sVW, norm_sum_count(xl) * NI);
  alloc_elements(g, K.sWPL, K.WP->nmsg());
  alloc_elements(g, K.sPV, norm_product_count(xl) * NI);
  alloc_elements(g, K.sTV, K.tV.nmsg());
  alloc_elements(g, K.sHH, (size_t)Lam * f + (size_t)LamM * fe);
  alloc_elements(g, K.sPL, K.HL->nmsg());
  if (mo) {
    K.sHH2 = K.sHH + (size_t)Lam * f; // sHH holds the token sums Hh1 then Hh2; sHH2 points at Hh2.
    alloc_elements(g, K.sPL2, K.HL2->nmsg());
    alloc_elements(g, K.rhoH2, 64);
    alloc_elements(g, K.etD2, NT);
    alloc_elements(g, K.ewX, d);
  }
  // Only sub-proof 0 has xg's table at Z:E. Others add its size: shared_alloc() needs equal sizes.
  alloc_elements(g, K.sW, K.nW + (g ? kx * mx / LA : 0));

  w = K.sW;

  // Appends row jobs over src to dst, at most 9 of phis each; outputs follow one another from w.
  auto mkjobs = [&](std::vector<RowJob> &dst, const i8 *src, size_t L_, size_t rows, const std::vector<T *> &phis) {
    for (size_t a0 = 0; a0 < phis.size(); a0 += 9) {
      RowJob j;
      j.N = (int)std::min<size_t>(9, phis.size() - a0);
      j.src = src;
      j.L = L_;
      j.rows = rows;
      for (int q = 0; q < j.N; q++) {
        j.phi[q] = phis[a0 + q];
        j.out[q] = w;
        w += rows;
      }
      j.prep();
      rscB = std::max(rscB, j.scratch());
      dst.push_back(j);
    }
  };
  auto tabs = [&](const LM &lm, const std::vector<std::pair<std::string, int>> &pts, const std::vector<std::string> &rd,
                  size_t L_) {
    std::vector<T *> ph;
    for (auto &p : pts) {
      if (has(rd, p.first)) {
        ph.push_back(K.phiP[p.first]);
      }
    }
    mkjobs(K.jobs, K.M[lm.first].v, L_, cells_of(K, lm) / L_, ph);
  };
  tabs(K.af, K.LPn, rdAF, LA);
  tabs(K.qa, K.LPn, rdQA, LA);
  tabs(K.xg, K.LPn, rdXG, LA);
  tabs(K.xn, K.LPn, rdXN, LA);
  K.short_jobs_from = K.jobs.size();
  tabs(K.ad, K.SPn, rdAD, NT);
  tabs(K.ar, K.SPn, rdAR, NT);
  tabs(K.qa, K.SPn, {"z"}, NT);
  {
    // b's tables: TB_0, TB_1, for its value at two lookup points; TP, for the multiplicities' sum.
    const size_t mb = (size_t)1 << lmb;
    std::vector<T *> ph;
    for (int i = 0; i < 3; i++) {
      alloc_elements(g, K.phiB[i], mb);
      ph.push_back(K.phiB[i]);
    }
    mkjobs(K.jobsB, K.M[K.b.first].v, mb, K.M[K.b.first].k, ph);
  }
  if (w != K.sW + K.nW) {
    printf("internal error: the tables as placed do not fill the planned number of table elements\n");
    return 2;
  }
  w = K.sWT;
  for (auto &q : K.wk) {
    alloc_elements(g, q.phi, q.L);
    q.tab = w;

    for (int p = 0; p < q.lm.n; p++) {
      mkjobs(q.jobs, slab, q.L, q.ntab / q.lm.n, {q.phi});
    }
  }
  for (int c = 0; c < NC; c++) {
    alloc_elements(g, K.ec[c], (size_t)1 << K.lJ[c]);
    alloc_elements(g, K.el[c], Lam);
    alloc_elements(g, K.ele[c], Lam);
  }
  alloc_elements(g, K.eV, NT);
  alloc_elements(g, K.etD, NT);
  alloc_elements(g, K.PV, 32);
  alloc_elements(g, K.gam, 1);
  alloc_elements(g, K.rhoV, 64);
  alloc_elements(g, K.halves, 1);
  alloc_elements(g, K.start, 1);
  alloc_elements(g, K.rhoH, 64);
  alloc_elements(g, K.omV, norm_product_count(xl) * NI);
  alloc_elements(g, K.ewN[0], d);
  alloc_elements(g, K.ewN[1], d);
  {
    alloc_elements(g, K.AH, 64);
    alloc_elements(g, K.lfc, dh);
    alloc_elements(g, K.ehq, (size_t)1 << 20);
    if (K.lJ[K.g.callOf("QK")] - ldh > 20) { // `g_weight_tables` writes 2^(lJ - ldh) elements there
      printf("internal error: the call QK needs a table of 2^%d elements where the set-up has room for 2^20\n",
             K.lJ[K.g.callOf("QK")] - ldh);
      return 2;
    }
    const size_t VV = std::max(dh, mo ? NE : 0), KK = std::max(d, mo ? fe : 0);
    alloc_elements(g, K.Cm, VV * KK);
    K.Bl = (i8 *)shared_alloc(g, VV * 8 * EDEG * KK); // eight byte limbs per coordinate
  }
  return 0;
}

// Numbers the sub-proof's slots, one for each masked value that it sends. Sets up its approximate
// range check (HT; `tail` in identifiers), U3 (the tables in K.sW) and its mask entries (S).
inline int Prover::g_setup_slots(int g) {
  using namespace hw;
  Fork &K = FK[g];
  HFk &Q = *K.h;
  HiddenTail &HT = Q.HT;

  {
    // Public source: one shared value per key: Boost.Flyweight's tutorial (key-value flyweights).
    // Here the key of `cls` is whether the group is the first, then its layers' public scales:
    // their records in KL, each without its last word, the rotary flag. The shared value is the
    // list of Shape buffers (fork_memory.h) that a class's first group allocated; from here to
    // setup_working_set_done() a plain cudaMalloc asks for one. A later group gets them in the
    // order of its allocations and writes the same constants into them, where the source builds its
    // value only once. Of the list's buffers that runs write, the source says nothing.
    std::vector<unsigned> ck = {(unsigned)(g == 0 ? 0 : 1)};
    for (int lam = 0; lam < Lam; lam++) {
      for (int w = 0; w < KLW; w++) {
        if (w != KLW - 1) {
          ck.push_back(KL[(size_t)KLW * (g * Lam + lam) + w]);
        }
      }
    }
    // KL has a record of KLW words for each of the Lr layers, then three words for each MoE layer.
    // The key also takes the three words of each MoE layer of this group.
    for (int l = 0, experts_before = 0; l < (g + 1) * Lam; l++) {
      if (!is_moe_layer(l)) continue;
      if (l >= g * Lam)
        for (int w = 0; w < 3; w++) ck.push_back(KL[(size_t)Lr * KLW + 3 * (size_t)experts_before + w]);
      experts_before++;
    }
    const int cls = fork_memory.shape_class(ck);
    // The key of the Lasting buffers' class has no scales: groups whose scales differ share these
    // buffers, so the sizes must not depend on the scales (fork_alloc() exits if the sizes differ).
    const int lasting = fork_memory.shape_class({(unsigned)(g == 0 ? 0 : 1)});
    fork_memory.setup_working_set(cls, lasting, g & 1, g);
  }
  const u32 sbase = 4u * (4096u + 64u * (u32)(g + 1));
  NM_ = (int)K.M.size();

  {
    std::vector<int> cls(NM_, 0);

    for (int i = 0; i < NM_; i++) {
      if (i == iXG && g > 0) {
        cls[i] = 1;
      }
    }
    std::vector<int> shr(NM_, 0);
    shr[iXN] = 1; // read by another sub-proof: group g + 1 (as its xg) or the output head

    if (K.qa.first != iQA) {
      printf("internal error: the list number of qa does not match its expected position\n");
      return 2;
    }
    shr[iQA] = 1;

    HT.chainMoment = 2; // the range check commits at commit point (`moment`) 2, after U3's (1)
    HT.rebuilt = args.tail_rebuilt;
    HT.init(K.M.data(), NM_, cls.data(), shr.data(), s8::hb_r(), sbase + 64, g > 0 ? &FK[g - 1].h->HT : nullptr, iXG, iXN);
  }
  // Slots must be numbered in the order of sending, as in verifier/src/vf_layers_check.inc.
  s = 0;
  Q.slPT = s;
  s += layer_forest.nmsg();
  Q.slVW = s;
  s += norm_sum_count(xl) * NI;
  Q.slWPL = s;
  s += K.WP->nmsg();
  Q.slPV = s;
  s += norm_product_count(xl) * NI;
  Q.slTV = s;
  s += K.tV.nmsg();
  if (!K.ts.empty()) {
    Q.slTS = s;
    s += K.ts[0].eng.nmsg();
  }
  { // batched sumcheck (`stack` in identifiers), lamS rounds; `g_walk_stack` checks lamS and NV
    auto Wd = [](size_t ni, size_t ny) {
      return (size_t)1 << (l2i(ni) + l2i(ny));
    };
    size_t tot = 4 * Wd(Lam, d) + Wd(Lam, dq) + Wd(Lam, f) + 2 * Wd(Lam, 1);
    if (mo) {
      tot += Wd(LamM, d) + Wd(LamM, 1) + 2 * Wd(LamM, NE) + (size_t)LamM * (2 * Wd(NE, d) + Wd(NE, fe));
    }
    Q.lamS = l2i(tot);
    Q.NCL = 6 * Lam + (mo ? 4 * LamM : 0);
    Q.NV = Q.NCL + (g == 0 ? 1 : 0);
  }
  if (g == 0) {
    for (T **p : {&K.tkh, &K.ZE, &K.pk}) {
      CK(cudaMalloc(p, 64 * sizeof(T)));
    }
  }
  // The batched sumcheck's end values: A0, H1, the LamM values X_lamM, B0 and the values of the
  // panels' NCL weight claims, all defined in prover/head/stack.cuh.
  Q.nEndS = 2 + 1 + (size_t)(mo ? LamM : 0) + Q.NCL;
  Q.slSCH = s;
  s += CSumcheck::slots(Q.lamS, Q.nEndS, 2);
  Q.slPL = s;
  s += K.HL->nmsg();
  if (mo) {
    Q.slPL2 = s;
    s += K.HL2->nmsg();
  }
  Q.slVE = s; // the embedding's value, sent alone
  if (g == 0) s++;
  Q.slMSC = s;
  s += CSumcheck::slots(lkW, 1, 1); // the merge of the weight claims sends one end value
  Q.slPZ = s;
  s += HT.F.plan.nmsg();
  Q.nmsg = s;
  Q.oLate = s;
  Q.where = "G" + std::to_string(g);
  Q.look.v = (unsigned)layer_forest.v(), Q.look.d = (unsigned)layer_forest.d(), Q.look.mu = 5;
  Q.look.shifted = layer_forest.NC > 1;
  {
    blk::LayerRounds &x = Q.own;
    x.lt = (unsigned)lt, x.ls = (unsigned)ls, x.lNI = (unsigned)lNI, x.lPd = (unsigned)lPd, x.ll = (unsigned)ll;
    x.lPf = (unsigned)lPf, x.lfe = (unsigned)lfe, x.lamS = (unsigned)Q.lamS, x.lkW = (unsigned)lkW;
    x.NI = (u64)NI, x.NS = (u64)norm_sum_count(xl), x.NP = (u64)norm_product_count(xl), x.NO = (u64)norm_row_value_count(xl);
    x.with_terms = !K.ts.empty(), x.shown_by_the_terms = Q.nEndT;
    x.never_sent = 1 + (u64)(mo ? LamM : 0), x.weight_claims = (u64)Q.NCL;
    x.first_group = g == 0, x.routed = mo != 0;
  }
  // One hidden vector is committed at commit point 1 (the first is 0): U3. The norms' sums of
  // squares (K.sHV) and the gated products' token sums (K.sHH) are never sent.
  Q.nv[3] = K.nW;
  Q.vec[3] = K.sW;
  Q.U[3].setup("U3g" + std::to_string(g), 1, Q.nv[3], EDEG, 1, sbase + 12, s8::hb_r());
  chain_plan_later_moments(chain.plan[chain.layer(g)], {{"U3", Q.nv[3]}}, Q.HT);
  {
    he::Ent e;
    e.Tn = &Q.U[3];
    e.num = 2;
    e.tag = 15;
    Q.EQ.list.push_back(e);
  }
  for (auto &l : HT.list) {
    he::Ent e;
    e.Tn = l.T;
    e.num = l.num;
    e.tag = l.tag;
    Q.EQ.list.push_back(e);
  }
  Q.ns = Q.oLate;
  Q.S.setup("S" + std::to_string(g), 2, 0, Q.ns, 1, 2, sbase, s8::hb_r());
  // g_fold()'s workspace; not K.blas_room, whose memory has another user from late_begin() on.
  CK(fork_alloc(&K.zfold_work, ZFOLD_TURNED_WORKSPACE, Place::Late));
  CK(cudaMalloc(&Q.rrS, 64 * sizeof(T)));
  CK(cudaMalloc(&Q.sc, 8 * sizeof(T)));
  // `roots` (zeros, not U3's root) goes unmasked into the message sections (plan_sections()). No
  // cudaMemset: cudaMalloc (fork_alloc_default()) zeroes and records each buffer, and what is zero
  // when set-up ends is zeroed again before every request and, with requests in step, before every
  // turn (a request's work on a group; memory_as_set_up.cuh). If the GPU did not run the copy into
  // `roots` (commit_unit()), it has zeros or, with one request at a time, this request's words from
  // an earlier group of the same shape class; another request's only if that zeroing was lost too.
  CK(cudaMalloc(&Q.roots, 8 * DW * 4));
  return 0;
}

// Sets which end values the two sumchecks over tokens send, masked ("shown" below), and which
// terms read them (TscEnd, prover/hiding/token_sumcheck.cuh). Verifier: verifier/src/vf_layers_check.inc.
inline int Prover::g_setup_token_ends(int g) {
  Fork &K = FK[g];
  HFk &Q = *K.h;
  const size_t T_ = (size_t)NT;
  {
    // K.tV, the sumcheck at P_V: every factor is shown. k_rowterms() (base.cuh) writes them.
    std::vector<const T *> value;
    std::vector<TscTerm> terms;
    const int NO = norm_row_value_count(xl);
    for (int i = 0; i < NI; i++) {
      if (xl == 5) { // shown: a, b, Rlo, Rhi
        static const int left[7] = {0, 0, 1, 2, 2, 3, 3}, right[7] = {0, 1, 1, 0, 1, 0, 1}, first[4] = {0, 2, 3, 5};
        for (int k = 0; k < 4; k++) value.push_back(K.FV + ((size_t)7 * i + first[k]) * T_);
        for (int c = 0; c < 7; c++) terms.push_back({NO * i + left[c], NO * i + right[c], K.omV + 7 * i + c, nullptr});
      } else { // shown: sigma, Rlo, Rhi
        for (int k = 0; k < 3; k++) value.push_back(K.FV + ((size_t)3 * i + k) * T_);
        for (int c = 0; c < 3; c++) terms.push_back({NO * i + c, NO * i, K.omV + 3 * i + c, nullptr});
      }
    }
    K.tVend.setup(K.tV.shape, value, terms);
  }
  if (K.ts.empty()) return 0;
  // K.ts[0], at zt: the terms in the order of g_token_sumchecks(). A term's first factor (Fv) is
  // shown unless it is public (QK's rotary tables), once if terms share it; the second (Gv) is not.
  TokenSumcheck &q = K.ts[0];
  const int iSc = NN * Lam, iSel = iSc + LamM;
  std::vector<const T *> value(Q.nEndT, nullptr);
  std::vector<TscTerm> terms;
  K.kept.clear();
  int ti = 0;
  auto term = [&](int a) {
    if (a >= 0 && !value[a]) value[a] = q.Fv + (size_t)ti * T_;
    terms.push_back({a, -1, q.co + ti, q.Gv + (size_t)ti * T_});
    ti++;
  };
  for (int c : tscCalls)
    for (int lam = 0; lam < Lam; lam++)
      for (auto &s : policy.cl[c].seg) {
        if (s == "N1" || s == "N2") {
          K.kept[s + std::to_string(lam)] = q.Gv + (size_t)ti * T_;
          term(NN * lam + (s == "N1" ? 0 : NN - 1));
        }
        if (s == "QK")
          for (int i = 0; i < dh; i++) term(-1);
        if (!is_moe_layer(g * Lam + lam)) continue;
        const int lm = lam / 2;
        if (s == "G" || s == "U" || s == "D" || s == "DD" || s == "MR")
          for (int e = 0; e < NE; e++) term(iSel + lm * NE + e);
        if (s == "XR") {
          K.kept[s + std::to_string(lm)] = q.Gv + (size_t)ti * T_;
          term(iSc + lm);
        }
      }
  if (ti != q.NTm) {
    printf("internal error: the sum-check over tokens has %d terms, but %d were planned\n", ti, q.NTm);
    return 4;
  }
  if (std::count(value.begin(), value.end(), nullptr)) {
    printf("internal error: a value that the sum-check over tokens is to show stands in none of its terms\n");
    return 4;
  }
  q.end.setup(q.eng.shape, value, terms);
  return 0;
}

// Completes the walk (Q.W, the list of the sub-proof's linear claims) and passes the terms of its
// one linear equation, on the arrays in Q.EQ and K.M, to `Q.door` (prover/hiding/door.cuh).
inline int Prover::g_setup_equations(int g) {
  using namespace hw;
  Fork &K = FK[g];
  HFk &Q = *K.h;
  HiddenTail &HT = Q.HT;
  hw::Walk &W = Q.W;
  {
    hz::Late late; // the walk runs after the fold
    W.seal();
  }
  {
    std::vector<HE> o1(1, hE(1));
    Q.one = (T *)W.cvec(o1);
  }
  for (auto &e : Q.EQ.list) {
    if (e.Tn == &Q.U[3]) {
      e.cv.push_back(Q.one);
      e.L.push_back(Q.LU[3]);
      e.c0.push_back(0);
      e.c1.push_back(Q.nv[3]);
      e.vecE = Q.vec[3];
    }
    for (int r = 0; r < nrows; r++) {
      if (HT.vrows[r].T == e.Tn) {
        e.cv.push_back(HT.vrows[r].cv);
        e.L.push_back(Q.Lrow[r]);
        const Space &z = W.sp[5 + r];
        e.c0.push_back(z.lo < z.hi ? z.lo : 0);
        e.c1.push_back(z.lo < z.hi ? z.hi : 0);
      }
    }
  }
  Q.door.add_eqs(Q.EQ);
  HT.say_forms(Q.door); // after `add_eqs`: marks the range check's row vectors as product forms
  Q.door.plain_rows("ZZ", HT.zzPlain, 2 * HT.mz);
  Q.door.plain_rows("BZ", HT.cBZ, HT.kb);
  for (int mu = 0; mu < HT.NM; mu++) Q.door.add_matrix(HT.TH[mu], HT.M[mu].v, HT.cls[mu] != 0);
  // HT.fold_to_group2() writes Q.zs's first DW words: zeros, not the root of its arrays ZZ and BZ.
  CK(fork_alloc(&Q.zs, 4 * DW * 4, Place::Late));
  CK(cudaMemset(Q.zs, 0, 4 * DW * 4)); // its first DW words are sent
  fork_memory.setup_working_set_done();
  return 0;
}

// Checks that sub-proof g > 0 took all the shared buffers that sub-proof 0 allocated, and sizes the
// hashing scratch memory.
inline int Prover::g_setup_end(int g) {
  Fork &K = FK[g];

  if (g > 0 && ai != arena.size()) {
    printf("internal error: sub-proof %d used %zu of %zu shared buffers, but it should use all of them\n", g, ai,
           arena.size());
    return 2;
  }
  thW = std::max(thW, std::max<size_t>(EWORDS * K.nW, EWORDS * K.nWT));
  return 0;
}

// Sets up the layer sub-proofs in the order of g. g_setup_walk() is in layer_walk.cuh.
inline int Prover::setup_g_forks() {
  for (int g = 0; g < NGr; g++) {
    if (int rc = g_setup_layout(g)) return rc;
    if (int rc = g_setup_matrices(g)) return rc;
    if (int rc = g_setup_engines(g)) return rc;
    if (int rc = g_setup_points(g)) return rc;
    if (int rc = g_setup_tables(g)) return rc;
    if (int rc = g_setup_slots(g)) return rc;
    if (int rc = g_setup_token_ends(g)) return rc;
    if (int rc = g_setup_walk(g)) return rc;
    if (int rc = g_setup_equations(g)) return rc;
    if (int rc = g_setup_end(g)) return rc;
  }
  return 0;
}
