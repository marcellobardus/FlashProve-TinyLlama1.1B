// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// The layout (LayerLayout) of the committed cells and lookup keys of a layer sub-proof, the argument for
// one group of layers. make_layer_layout() builds it from a model shape (LayerShape). Tensors of cells (small
// integers) fill four matrices (af, ad, ar, qa). A lookup call looks up segments, named lists of
// keys, in a public table (Kind).
#pragma once
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <string>
#include <algorithm>
struct LayerShape {
  // 2^lt tokens; 2^ll layers in the layer group; widths: d of the residual stream, f of the
  // feed-forward part, fe of an expert; H query and Gk key/value heads of size dh; NE routed
  // experts.
  int lt = 0, ll = 0, d = 0, H = 0, Gk = 0, dh = 0, f = 0, fe = 0, NE = 0;
  // M, K, eps: index 0 for a layer's first norm, 1 for its second. A norm multiplies by a
  // reciprocal scaled by 2^M and shifts right by K bits; eps is added to the sum of squares.
  // ForwardPass::norm() (prover/prover/forward.cuh) reads eps, but takes M and K from the layer's LayerScaleSet.
  int M[2] = {0, 0}, K[2] = {0, 0};
  long long eps[2] = {1, 1};
  // g: the layer group's number. prof: 1 feed-forward part only, 2 also attention's projections,
  // 3 also experts in odd layers; not the prof of the parameter file (`policy.txt`; `policy` in
  // messages). la: features (see Ten) are evaluated in runs of 2^(la - lt). A call has at least
  // 2^nmin keys and needs at least 2^(la - lt) columns.
  int g = 0, prof = 3, nmin = 10, la = 6;
  // Digit forms: how many bytes (0, 1 or 2) store a rounding's remainder above its low 16 bits, by
  // tensor (segment): rqk (QKH), rv (VH), ro (OH), rg (GH), ru (UH), rd (DH), rl (LH). At form 0
  // the shape has neither the tensor nor the segment.
  int hqk = 2, hv = 1, ho = 1, hg = 1, hu = 1, hdn = 1, hr = 1;
  int att = 0; // 1: attention is proved too (the parameter file's prof is 20); o is then the fourth tensor of qa
  // No program sets cw: the parameter file has no line cw.
  int cw = 12; // width in bits of the clamped cells a, m, v: 12 or 14
  // 1 with the parameter file's line "rcl 1": the router's wide logit is 8 ly + l3 with l3 in 0 .. 7, and the
  // routing reads the logit lr = 8 clamp(ly, +-2047) + l3. This adds the tensors ly, l3 and the segments L3, CLR.
  int rcl = 0;
  // 1 with the line "gcl 1" of the parameter file: the gate's quotient is g0 + 256 gh, where
  // g0 is its balanced low byte, and the cell gi holds the key 256 clamp(gh, +-127) + g0,
  // at which the gate's table is read. This adds the tensor gh and the segments G0 and CG.
  // G0 looks up g0 + 128, to show that g0 is a byte. gh needs profile 3.
  int gcl = 0;
  int xl = 4;  // limbs of a residual-stream value: 4, or 5 with the parameter file's line "xl 5"
  // The call lists as text, for example "R16a:G,U;R16b:N2,N2H,D". Null selects make_layer_layout()'s
  // built-in lists, from which a segment that the shape lacks is dropped without a message.
  const char *lists = nullptr;
};
// Ceiling of log2(n)
static inline int p2c(long long n) {
  int l = 0;
  while ((1ll << l) < n) l++;
  return l;
}
// How many instances a tensor has: I_ALL one per layer; I_MOE one per expert (odd) layer; I_STR one
// per addition to the residual stream, less one (NW * Lam - 1); I_ROW one per norm (NN * Lam).
enum Inst { I_ALL, I_MOE, I_STR, I_ROW };
enum TenId {
  X_gi,
  X_s,
  X_u,
  X_rg,
  X_ru,
  X_xs,
  X_ro,
  X_rd,
  X_aw,
  X_ra,
  X_mw,
  X_rm,
  X_qkw,
  X_rqk,
  X_vw,
  X_rv,
  X_a,
  X_m,
  X_xr,
  X_lr,
  X_rl,
  X_sel,
  X_sc,
  X_row,
  X_ly,
  X_l3,
  X_gh,
  NTEN
};
// A tensor has cnt instances of J elements of nl limbs. Each limb is a feature of its matrix (one
// cell per token). off: the tensor's first feature.
struct Ten {
  const char *name = "";
  int nl = 0;
  Inst inst = I_ALL;
  long long J = 0;
  int mat = 0 /* 0 af, 1 ad, 2 ar */, minprof = 0;
  long long cnt = 0, off = 0;
  bool on = false;
};
// A residual-stream value as one integer; as cells, xl balanced byte limbs (base-256 digits in
// -128 .. 127).
typedef long long ResidualInt;
// A norm's row of digits has norm_row_len() bytes per token; the sum of squares is the sum over s of
// 256^s W_s, where each of the norm_sum_count() sums W_s adds, over the token's d values, the products of
// limbs i and j with i + j = s; per norm, norm_product_count() products of two of norm_row_value_count() row values are proved.
static inline int norm_row_len(int xl) { return xl == 5 ? 99 : 74; }
static inline int norm_sum_count(int xl) { return 2 * xl - 1; }
static inline int norm_product_count(int xl) { return xl == 5 ? 7 : 3; }
static inline int norm_row_value_count(int xl) { return xl == 5 ? 4 : 3; }
enum { NORM_ROW_LEN_MOST = 99, NORM_SUM_COUNT_MOST = 9, NORM_PRODUCT_COUNT_MOST = 7 };
// Segment ROW looks up norm_digit_factor() times digit i of a norm's row, to show that digit < 256 / norm_digit_factor().
static inline int norm_digit_factor(int xl, int i) {
  if (xl == 5) return i == 9 ? 4 : i == 58 ? 2 : i == 59 ? 128 : i == 63 ? 2 : i == 76 ? 2 : i == 79 ? 32 : 1;
  return i == 7 ? 32 : i == 44 ? 2 : i == 45 ? 128 : i == 48 ? 4 : 1;
}
// A layer sub-proof's largest number of lookup calls; make_layer_layout() does not check it.
constexpr int MAX_CALLS = 8;
// A call's table. R16, R8: all 16-bit, 8-bit values. FC: clamp to +-2047. FS: router's sigmoid,
// gh's clamp. FL: SiLU.
enum Kind { K_R16, K_R8, K_FC, K_FS, K_FL, K_FW /* clamp at cw 14 */, K_FE /* output head only */ };
// The segment's indices [s, s + size) are in the call's columns [off, off + size). Sizes are powers
// of two, placed largest first, so off is a multiple of size.
struct Piece {
  int seg = 0;
  long long s = 0, size = 0, off = 0;
};
// n indices. call: index into LayerLayout::call. pc: indices into that call's pc.
struct Seg {
  std::string name;
  Kind kind = K_R16;
  long long n = 0;
  bool moe = false;
  int call = 0;
  std::vector<int> pc;
};
// 2^lJ columns (the first `total` in use), each with a key per layer and token.
struct Call {
  std::string name;
  Kind kind = K_R16;
  std::vector<int> seg;
  std::vector<Piece> pc;
  long long total = 0;
  int lJ = 0;
};
struct LayerLayout {
  LayerShape s;
  // NW additions to the residual stream and NN norms per layer. Pd, Pf: d, f rounded up to a
  // power of two.
  int T = 0, Lam = 0, LamM = 0, NW = 0, NN = 0;
  long long dq = 0, dkv = 0, nqk = 0, Pd = 0, Pf = 0;
  Ten t[NTEN];
  // G[0], G[1], G[2], Gqa: the number of features of af, ad, ar, qa, over the tensors that are on.
  long long G[3] = {0, 0, 0};
  Ten qa[4];
  long long Gqa = 0;
  std::vector<Seg> seg;
  std::vector<Call> call;
  int segid(const char *n) const {
    for (size_t i = 0; i < seg.size(); i++)
      if (seg[i].name == n) return (int)i;
    return -1;
  }
  long long pos(int sg, long long x) const {
    const Call &c = call[seg[sg].call];
    for (int p : seg[sg].pc)
      if (x >= c.pc[p].s && x < c.pc[p].s + c.pc[p].size) return c.pc[p].off + x - c.pc[p].s;
    printf("internal error: a position does not fall inside any piece of its segment\n");
    exit(1);
  }
};
// A layer is a mixture-of-experts layer iff the model has routed experts (NE > 0) and the layer is
// odd. NE > 0 must be part of the test: with NE 0 every layer is dense (docs/model.md; PROVING_PLAN
// M2), and the rest of the prover already treats dense that way (mo = NE > 0 in
// prover/prover/inputs.cuh, prover/weights_commitment/kernels.cuh), as does the verifier (MOE = NE
// > 0 in verifier/src/vf_weights_layout.h). Without the NE > 0 clause the forward pass alone ran
// ffn_moe() on odd layers at NE 0, against an all-dense layout, and a forward-pass value left its
// range (UNFIT).
static inline bool moe_at_layer(int prof, int l, int NE) {
  return prof >= 3 && (l & 1) && NE > 0;
}
// Exits with code 2 if it rejects the shape.
static inline LayerLayout make_layer_layout(const LayerShape &s) {
  LayerLayout L;
  L.s = s;
  L.T = 1 << s.lt;
  L.Lam = 1 << s.ll;
  L.LamM = L.Lam / 2;
  L.dq = (long long)s.H * s.dh;
  L.dkv = (long long)s.Gk * s.dh;
  L.nqk = L.dq + L.dkv;
  L.Pd = 1ll << p2c(s.d);
  L.Pf = 1ll << p2c(s.f);
  if (L.dq != s.d || s.H + s.Gk > 64 || s.ll < 1 || (s.prof >= 3 && s.f != 2 * s.fe)) {
    printf("policy: model shape rejected: H x dh must equal d, H + Gk must be at most 64, ll at least 1, and in a "
           "model with expert layers f must equal 2 fe\n");
    exit(2);
  }
  if (s.cw != 12 && s.cw != 14) {
    printf("policy: cw must be 12 or 14\n");
    exit(2);
  }
  const long long d = s.d, f = s.f, NE = s.NE;
  const int pf = s.prof;
  L.NW = pf == 1 ? 1 : 2;
  L.NN = L.NW;
  Ten tt[NTEN] = {{"gi", 2, I_ALL, f, 0, 1},        {"s", 2, I_ALL, f, 0, 1},           {"u", 2, I_ALL, f, 0, 1},
                  {"rg", s.hg, I_ALL, f, 0, 1},     {"ru", s.hu, I_ALL, f, 0, 1},       {"xs", s.xl, I_STR, d, 0, 1},
                  {"ro", s.ho, I_ALL, d, 0, 2},     {"rd", s.hdn, I_ALL, d, 0, 1},      {"aw", 2, I_ALL, d, 0, 2},
                  {"ra", 2, I_ALL, d, 0, 2},        {"mw", 2, I_ALL, d, 0, 1},          {"rm", 2, I_ALL, d, 0, 1},
                  {"qkw", 2, I_ALL, L.nqk, 0, 2},   {"rqk", s.hqk, I_ALL, L.nqk, 0, 2}, {"vw", 2, I_ALL, L.dkv, 0, 2},
                  {"rv", s.hv, I_ALL, L.dkv, 0, 2}, {"a", 2, I_ALL, d, 1, 2},           {"m", 2, I_ALL, d, 1, 1},
                  {"xr", 2, I_MOE, d, 1, 3},        {"lr", 2, I_MOE, NE, 2, 3},         {"rl", s.hr, I_MOE, NE, 2, 3},
                  {"sel", 1, I_MOE, NE, 2, 3},      {"sc", 3, I_MOE, 1, 2, 3},          {"row", 1, I_ROW, norm_row_len(s.xl), 2, 1},
                  {"ly", 2 * s.rcl, I_MOE, NE, 2, 3}, {"l3", s.rcl, I_MOE, NE, 2, 3},
                  {"gh", 2 * s.gcl, I_ALL, f, 0, 3}};
  L.G[0] = L.G[1] = L.G[2] = 0;
  for (int i = 0; i < NTEN; i++) {
    Ten &X = L.t[i];
    X = tt[i];
    X.on = pf >= X.minprof && X.nl > 0;
    X.cnt = X.inst == I_ALL ? L.Lam : X.inst == I_MOE ? L.LamM : X.inst == I_STR ? L.NW * L.Lam - 1 : L.NN * L.Lam;
    X.off = L.G[X.mat];
    if (X.on) L.G[X.mat] += X.nl * X.cnt * X.J;
  }
  Ten q3[4] = {{"q", 2, I_ALL, L.dq}, {"kx", 2, I_ALL, L.dkv}, {"v", 2, I_ALL, L.dkv}, {"o", 2, I_ALL, L.dq}};
  long long o = 0;
  for (int i = 0; i < 4; i++) {
    L.qa[i] = q3[i];
    L.qa[i].cnt = L.Lam;
    L.qa[i].off = o;
    L.qa[i].on = i < 3 ? pf >= 2 : s.att;
    if (L.qa[i].on) o += 2 * L.Lam * q3[i].J;
  }
  L.Gqa = o;
  struct SD {
    const char *n;
    Kind k;
    long long len;
    bool moe;
    int form = 1;
  };
  auto HK = [](int fm) {
    return fm == 2 ? K_R16 : K_R8;
  };
  const Kind KW = s.cw == 14 ? K_FW : K_FC;
  const SD all[] = {{"G", K_R16, f, 0},
                    {"U", K_R16, f, 0},
                    {"UT", K_R16, f, 0},
                    {"QK", K_R16, L.nqk, 0},
                    {"QKH", HK(s.hqk), L.nqk, 0, s.hqk},
                    {"N1", K_R16, d, 0},
                    {"N1H", K_R16, d, 0},
                    {"N2", K_R16, d, 0},
                    {"N2H", K_R16, d, 0},
                    {"O", K_R16, d, 0},
                    {"D", K_R16, d, 0},
                    {"V", K_R16, L.dkv, 0},
                    {"XR", K_R16, d, 1},
                    {"LR", K_R16, NE, 1},
                    {"LT", K_R16, NE, 1},
                    {"DD", K_R16, NE, 1},
                    {"GH", HK(s.hg), f, 0, s.hg},
                    {"UH", HK(s.hu), f, 0, s.hu},
                    {"OH", HK(s.ho), d, 0, s.ho},
                    {"DH", HK(s.hdn), d, 0, s.hdn},
                    {"VH", HK(s.hv), L.dkv, 0, s.hv},
                    {"ROW", K_R8, (long long)norm_row_len(s.xl) * L.NN, 0},
                    {"LH", HK(s.hr), NE, 1, s.hr},
                    {"SEL", K_R8, NE, 1},
                    {"L3", K_R8, NE, 1, s.rcl},
                    {"CLR", K_FC, NE, 1, s.rcl},
                    {"G0", K_R8, f, 0, s.gcl},
                    {"CG", K_FS, f, 0, s.gcl},
                    {"CN1", KW, d, 0},
                    {"CN2", KW, d, 0},
                    {"CQK", K_FC, L.nqk, 0}, // q, kx stay at 12 bits whatever cw is
                    {"CV", KW, L.dkv, 0},
                    {"MR", K_FS, 1, 1},
                    {"F", K_FL, f, 0}};
  struct CD {
    std::string n;
    Kind k;
    std::vector<std::string> sg;
  };
  std::vector<CD> cl;
  // The built-in call lists. The prover program `cp_prove` never uses them; at cw 12 they lack
  // segment UT.
  if (s.cw == 14 && pf == 1)
    cl = {{"R16a", K_R16, {"G", "U"}}, {"R16b", K_R16, {"UT", "N2", "N2H", "D"}},
          {"R8a", K_R8, {"GH", "UH"}}, {"R8b", K_R8, {"DH", "ROW"}},
          {"FW", K_FW, {"CN2"}},       {"FL", K_FL, {"F"}}};
  else if (s.cw == 14 && pf == 2)
    cl = {{"R16a", K_R16, {"G", "U"}},
          {"R16b", K_R16, {"N1", "N1H", "N2", "N2H", "O", "D", "V"}},
          {"R16c", K_R16, {"QK", "QKH", "UT"}},
          {"R8a", K_R8, {"GH", "UH"}},
          {"R8b", K_R8, {"OH", "DH", "VH", "ROW"}},
          {"FC", K_FC, {"CQK"}},
          {"FW", K_FW, {"CN1", "CN2", "CV"}},
          {"FL", K_FL, {"F"}}};
  else if (s.cw == 14)
    cl = {{"R16a", K_R16, {"G", "U"}},
          {"R16b", K_R16, {"N1", "N1H", "N2", "N2H", "O", "D", "V"}},
          {"R16c", K_R16, {"QK", "QKH", "XR", "LR", "LT", "DD", "UT"}},
          {"R8", K_R8, {"GH", "UH", "OH", "DH", "VH", "ROW", "LH", "SEL"}},
          {"FC", K_FC, {"CQK"}},
          {"FW", K_FW, {"CN1", "CN2", "CV"}},
          {"FS", K_FS, {"MR"}},
          {"FL", K_FL, {"F"}}};
  else if (pf == 1)
    cl = {{"R16a", K_R16, {"G", "U"}}, {"R16b", K_R16, {"N2", "N2H", "D"}},
          {"R8a", K_R8, {"GH", "UH"}}, {"R8b", K_R8, {"DH", "ROW"}},
          {"FC", K_FC, {"CN2"}},       {"FL", K_FL, {"F"}}};
  else if (pf == 2)
    cl = {{"R16a", K_R16, {"G", "U"}},
          {"R16b", K_R16, {"N1", "N1H", "N2", "N2H", "O", "D", "V"}},
          {"R16c", K_R16, {"QK", "QKH"}},
          {"R8a", K_R8, {"GH", "UH"}},
          {"R8b", K_R8, {"OH", "DH", "VH", "ROW"}},
          {"FC", K_FC, {"CN1", "CN2", "CQK", "CV"}},
          {"FL", K_FL, {"F"}}};
  else
    cl = {{"R16a", K_R16, {"G", "U"}},
          {"R16b", K_R16, {"N1", "N1H", "N2", "N2H", "O", "D", "V"}},
          {"R16c", K_R16, {"QK", "QKH", "XR", "LR", "LT", "DD"}},
          {"R8", K_R8, {"GH", "UH", "OH", "DH", "VH", "ROW", "LH", "SEL"}},
          {"FC", K_FC, {"CN1", "CN2", "CQK", "CV"}},
          {"FS", K_FS, {"MR"}},
          {"FL", K_FL, {"F"}}};
  if (s.lists) {
    cl.clear();
    std::string t = s.lists;
    size_t p = 0;
    while (p < t.size()) {
      size_t e = t.find(';', p);
      if (e == std::string::npos) e = t.size();
      std::string one = t.substr(p, e - p);
      size_t c = one.find(':');
      CD cd;
      cd.n = one.substr(0, c);
      cd.k = cd.n.rfind("R16", 0) == 0  ? K_R16
             : cd.n.rfind("R8", 0) == 0 ? K_R8
             : cd.n.rfind("FC", 0) == 0 ? K_FC
             : cd.n.rfind("FS", 0) == 0 ? K_FS
             : cd.n.rfind("FW", 0) == 0 ? K_FW
                                        : K_FL;
      if (cd.k == K_FL && cd.n.rfind("FL", 0) != 0) {
        printf("policy: the name of call %s must begin with its kind: R16, R8, FC, FS, FL or FW\n", cd.n.c_str());
        exit(2);
      }
      for (auto &o : cl)
        if (o.n == cd.n) {
          printf("policy: two call lines have the name %s; every call needs a unique name\n", cd.n.c_str());
          exit(2);
        }
      if (cd.k == K_FW && s.cw != 14) {
        printf("policy: call %s is of kind FW, which is allowed only with the line \"cw 14\"\n", cd.n.c_str());
        exit(2);
      }
      size_t q = c + 1;
      while (q < one.size()) {
        size_t r = one.find(',', q);
        if (r == std::string::npos) r = one.size();
        cd.sg.push_back(one.substr(q, r - q));
        q = r + 1;
      }
      cl.push_back(cd);
      p = e + 1;
    }
  }
  const int fl = std::max(0, s.nmin - s.ll - s.lt);
  for (auto &c : cl) {
    Call C;
    C.name = c.n;
    C.kind = c.k;
    for (auto &nm : c.sg) {
      const SD *sd = nullptr;
      for (auto &a : all)
        if (nm == a.n) sd = &a;
      if (!sd) {
        printf("policy: unknown segment name %s\n", nm.c_str());
        exit(2);
      }
      if (s.lists && sd->moe && pf < 3) {
        printf("policy: segment %s belongs to expert layers, and this model has none, so it cannot be listed\n",
               nm.c_str());
        exit(2);
      }
      if (!sd->form) {
        if (s.lists) {
          printf("policy: segment %s does not exist under the chosen digit forms (L3, CLR: without the line rcl; G0, CG: "
                 "without gcl), so it cannot be listed\n",
                 nm.c_str());
          exit(2);
        }
        continue;
      }
      if (sd->k != c.k) {
        printf("policy: segment %s is of a different kind than call %s, so it cannot be placed there\n", nm.c_str(),
               c.n.c_str());
        exit(2);
      }
      Seg S{sd->n, sd->k, sd->len, sd->moe, (int)L.call.size(), {}};
      int id = (int)L.seg.size();
      L.seg.push_back(S);
      C.seg.push_back(id);
      long long st = 0;
      for (int b = 40; b >= 0; b--)
        if ((sd->len >> b) & 1) {
          C.pc.push_back(Piece{id, st, 1ll << b, 0});
          st += 1ll << b;
        }
    }
    std::stable_sort(C.pc.begin(), C.pc.end(), [](const Piece &a, const Piece &b) { return a.size > b.size; });
    long long off = 0;
    for (size_t p = 0; p < C.pc.size(); p++) {
      C.pc[p].off = off;
      off += C.pc[p].size;
      L.seg[C.pc[p].seg].pc.push_back((int)p);
    }
    C.total = off;
    C.lJ = std::max(p2c(off), fl);
    if (C.seg.empty()) {
      for (auto &S : L.seg)
        if (S.call > (int)L.call.size()) S.call--;
      continue;
    }
    // The verifier has the same rule: layout(), verifier/src/cp_verify.cpp. The layer sub-proof
    // splits the call's point zj (lJ coordinates) before the last la - lt
    // (prover/prover/layer_walk.cuh). With fewer, the split would fall before the point's start.
    if (s.ll + C.lJ + s.lt < s.la || C.lJ < s.la - s.lt) {
      printf("policy: call %s has 2^%d columns, and a call needs at least 2^(la - lt) = 2^%d: raise nmin to la + ll = "
             "%d or more (the verifier rejects such a file: shape-call-%s)\n",
             C.name.c_str(), C.lJ, s.la - s.lt, s.la + s.ll, C.name.c_str());
      exit(2);
    }
    L.call.push_back(C);
  }
  // Every segment of this shape must be listed exactly once: the forward pass
  // (prover/prover/forward.cuh) would write an unlisted segment's keys into call 0 or through a
  // null pointer. Profile 1 and the built-in lists are not checked.
  if (s.lists && pf >= 2)
    for (auto &a : all) {
      if (!a.form || (a.moe && pf < 3)) continue;
      int times = 0;
      for (auto &S : L.seg) times += S.name == a.n;
      if (times != 1) {
        printf("policy: segment %s is listed in %d call lines; it must be listed in exactly one\n", a.n, times);
        exit(2);
      }
    }
  return L;
}
// The feature of X's matrix that stores limb l of element j of instance i.
static inline long long limb_feature(const Ten &X, int l, long long i, long long j) {
  return X.off + ((long long)l * X.cnt + i) * X.J + j;
}
