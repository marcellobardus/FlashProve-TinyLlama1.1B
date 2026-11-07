// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

/* The GPU prover's sub-proof (`fork` in identifiers) of the output head: part of a request's proof.
   What it proves, in short. A request has NT tokens and a range of positions, r <= t < e. The
   sub-proof covers (1) the final normalization at all NT positions and (2) a statement about the
   logits at the positions of the range. Those below NT - 1 are the "counted" positions, NB of them.
   - lam_t[v] is the logit of vocabulary entry v at position t. Wh is the head's matrix.
   - w_t is the checked token of position t: the request's token t + 1 or, at position NT - 1, the
     next token y, which the prover announces (0 unless e = NT).
   - The margin is D[t][v] = lam_t[w_t] + (c_t + 2^37 b_t) - lam_t[v] - (1 if v < w_t), with the
     term in c and b only at the counted positions. D = K_2 + 2^sh K_1 + 2^(sh + 15) K_0, and range
     lookups on the digits are meant to show D >= 0. The parameter file (policy.txt) sets sh and F.
   - ex[t][v] = EX[K_1], where EX is a public table of a decaying exponential with EX[0] = 2^F.
   - Six hidden numbers per position tie the margins to the acceptance policy's public bounds: the
     clip C, the bound theta on the mean clipped cost, and cap, the largest number of positions
     above the clip. c is the position's honest cost (`k_head_price`, prover/head/forward.cuh) if that
     is at most C, and 0 otherwise. b is 1 if the honest cost is above C. k = C - c. u, z and zs are
     the slack in the relations below. At a position that is not counted all six are 0. Their digits
     follow the normalization's digits in each position's "digit row" (`ledger` in identifiers).
       the sum over the counted positions of (c + C b + u) = theta NB
       the sum over the counted positions of (b + z) = cap (or 0 if NB = 0)
       at each counted position: zs + the sum over v of ex[t][v] = 2^F
   The core of how the head treats the softmax of the logits (a committed shift, here lam_t[w_t] +
   c_t + 2^37 b_t, of which c_t and b_t are committed; looked-up digits; a table of the exponential;
   a lowest digit that is only range-checked; a condition on the sum) is that of zkAttn, which is
   for the softmax of attention: Sun, Li, Zhang, "zkLLM", arXiv:2404.16109, sections 5.1, 5.1.1 and
   5.2 with Protocol 2. The head differs. zkAttn multiplies the table values of several digits
   (their number is a parameter), requires the sum of the products to be within a tolerance on
   either side of the scale (here 2^F), and uses the products as the softmax. Here only K_1 has a
   table of the exponential, every rounding goes against the prover, and the sum is bounded only
   from above. The shift itself is used: c_t + 2^37 b_t is then an upper bound on -2^sT ln p, where
   p is w_t's softmax probability under the logits divided by 2^sT. sT is a policy.txt line: nothing
   shows 2^sT to be the model's own scale. The first two relations and k = C - c bound c and b.

   How to use it. Call `setup` once, and `weights_from` unless the head's matrix is to be synthetic.
   In each request, after the last layer's sub-proof, call `weights` (the GPU memory that the
   sub-proofs share, `Place::Fork` in prover/prover/fork_memory.h, does not keep Wh's cells),
   `phase1` and `prove`. `Prover::prove_head_fork` (prover/prover/request.cuh) shows what else a
   caller sets before `prove`. `sections`, `member` and `T_H` give the results.
   Terms. kk < d is a feature, x or v < V a vocabulary entry. T is the type of a field element.
   - eq(z, i): the multilinear extension of equality, the Lagrange basis polynomial of index i at z.
   - A "cell" is one signed byte of a committed matrix. A wider value takes several cells ("limbs"),
     low limb first. The request's committed matrices are hf (the normalization's quotient and the
     high part of its remainder, the margins' digits K_1 and K_2, and the values ex), hd (the
     normalized activations), hr (the digit rows) and b (the lookups' multiplicity bits). xl is the
     last residual stream, which the last layer's sub-proof committed.
   - A "call" is a lookup call: every key of an array is to occur in a public table. A `callH` line
     of the parameter file defines one. Its keys form 2^lJ columns, a row per position.
     Nothing more is committed for each key; source for that aim alone: Qu, Guo, Xuan, Liu, Zhang,
     "Celer: A Lookup Argument for Large-Scale Queries", CRYPTO 2026, ePrint 2026/1453, abstract.
     How Celer reaches that aim has not been compared with the lookup argument used here.
   - A "segment" is a named set of a call's columns (not a segment of the request's proof):
       NF, NFH   the low and high part of the normalization's remainder
       CNF       the normalization's quotient together with its clamped output
       ROW       each position's digit row
       HD0, HD1, HD2   the margin's digits K_0, K_1, K_2, one column per vocabulary entry
       HE0, HE1  K_1 + 2^16 ex; HE0: vocabulary entries below JVa, a power of two; HE1: from JVa on
   - All calls together are one lookup "forest" (struct Forest, prover/hiding/forest.cuh): one
     lookup argument runs on the keys of all calls. Call q's keys are cut into 2^io[q] consecutive
     product "trees" of 2^dT keys. A column that belongs to no segment holds a public "padding key".
   - A "row pass" cuts a matrix's cells, in memory order, into "rows" of L cells: out[j] = the sum
     over i < L of phi[i] cell[j L + i]. Its result is a "table": long if L = 2^la, short if L = NT.
   - A "challenge" (coin in `sch_coins`) is a field element derived from the Fiat-Shamir transcript.
   - A "slot" is the index of one scalar in the sub-proof's message buffer MSG, not a request's slot
     in its batch. prover/hiding/sumcheck_door.cuh decides what is sent in it, and under which mask.
     Entry i of the sub-proof's committed mask row (random values) belongs to slot i.
   - A "plane": one coordinate of an array over the cubic extension, or bit j of every multiplicity.
   - The "walk" (hw::Walk, prover/hiding/walk.cuh) batches all linear claims on hidden values and
     tables into one equation; each sumcheck (`psc`) leaves one. Its variables form "spaces".
   - The "tail" (`HiddenTail`, prover/hiding/tail_f.cuh; "smallness test" in round names) is the
     approximate range check, which is to bound the size of the cells: below, "the range check".
   - `hd::close` passes the walk's equation, as coefficient vectors on the committed arrays, to the
     frame of the request's proof (blocks/iop/req.h), which is to prove it (the "closing").
   - fv is the value of the merged claim on the weights. MERGE (`CL.merge`) sends it masked, as its
     one end value fv' = fv + beta c. Here c is an entry of the request's index-mask message, not of
     this sub-proof's mask row: the claim "fv' - beta c is the weights' value" is left for the test
     on the weight table (`CEnd::send_merge`, prover/head/claims_lib.cuh). No claim here is on fv.
   Public source for the fills and copies on the GPU: NVIDIA, CUDA Runtime API, "Memory Management"
   (cudaMemsetAsync(), cudaMemcpyAsync() from device to device, in the calling thread's stream). */
#pragma once
#include "common/end_name.h" // name_the_end (M5g)
#include "common/index_guard.cuh"
#include "head/forward.cuh"
#include "head/rows.cuh"
#include "common/zfold_turned.cuh"
template <class X> static X *dalH(size_t n, Place place) {
  X *p_ = nullptr;
  CK(fork_alloc((void **)&p_, n * sizeof(X) + 256, place));
  return p_;
}
// Writes the head's committed cells, in the order [limb][kk][x], from an int16 array wh[V][d] or,
// if wh is null, from synthetic weights. GRID_LOOP (prover/prover/forward.cuh) is a grid-stride loop.
__global__ void kH_wkH(const int16_t *wh, u64 seed, int d, int V, i8 *c) {
  GRID_LOOP(i, (size_t)d * V) {
    const size_t kk = i / V, x = i % V;
    const int v = wh ? (int)wh[x * d + kk] : synth_weight(seed, 14, i, 13);
    write_limbs(v, 2, c + i, (size_t)d * V);
  }
}
/* Writes the same cells as `kH_wkH`, from a wh that is not null. The two limbs of a value v are its
   low byte and the byte (v + 128) >> 8, as `write_limbs` produces them. Requires, and does not check: d
   even, V a multiple of 4, wh and c at addresses that are multiples of 4 (`weights` checks these).
   Launch with (V + 127) / 128 by (d + 63) / 64 blocks of 256 threads. */
// The idea is public: a padded tile: M. Harris, "An Efficient Matrix Transpose in CUDA C/C++",
// NVIDIA Technical Blog, https://developer.nvidia.com/blog/efficient-matrix-transpose-cuda-cc/.
// Here the transpose goes through a tile in shared memory; a row of it has 130 entries for 128
// values, to reduce bank conflicts.
// The idea is public: a digit s_i >= L/2 becomes s_i - L, carrying 1: drouyang, "Signed Bucket
// Indexes for Multi-Scalar Multiplication (MSM)", https://hackmd.io/@drouyang/signed-bucket-index.
// The page finds the carry by a comparison, digit by digit. Here, for the high byte, L = 256, and
// adding 128 before the shift by 8 bits adds the 1.
// The compiler hint is public: __launch_bounds__: NVIDIA, "CUDA C++ Best Practices Guide",
// "Occupancy", https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/.
// The compiler hint is public: #pragma unroll: NVIDIA, "CUDA Programming Guide", page "C/C++
// Language Extensions", "#pragma unroll",
// https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-extensions.html.
__global__ void __launch_bounds__(256) kH_wkH_tiles(const int16_t *wh, int d, int V, i8 *c) {
  __shared__ int16_t s[64][130];
  const int x0 = blockIdx.x * 128, k0 = blockIdx.y * 64, lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  for (int r = warp; r < 128; r += 8) {
    const int x = x0 + r, kk = k0 + 2 * lane;
    const u32 v = x < V && kk < d ? *(const u32 *)(wh + (size_t)x * d + kk) : 0u;
    s[2 * lane][r] = (int16_t)(v & 65535u);
    s[2 * lane + 1][r] = (int16_t)(v >> 16);
  }
  __syncthreads();
  for (int q = warp; q < 64; q += 8) {
    const int kk = k0 + q, x = x0 + 4 * lane;
    if (kk >= d || x >= V) continue;
    u32 lo = 0, hi = 0;
#pragma unroll
    for (int j = 0; j < 4; j++) {
      const int v = s[q][4 * lane + j];
      lo |= (u32)(v & 255) << (8 * j);
      hi |= (u32)(((v + 128) >> 8) & 255) << (8 * j);
    }
    *(u32 *)(c + (size_t)kk * V + x) = lo;
    *(u32 *)(c + (size_t)d * V + (size_t)kk * V + x) = hi;
  }
}
__global__ void kH_ones(T *o, int n) {
  GRID_LOOP(i, n) o[i] = F::one();
}
__global__ void kH_addch(const T *part, int nch, size_t rows, T *o) {
  GRID_LOOP(r, rows) {
    T a = part[r];
    for (int q = 1; q < nch; q++) a = F::add(a, part[(size_t)q * rows + r]);
    o[r] = a;
  }
}
#include "head/claims.cuh"
#include "head/plan_rounds.h"
#include "hiding/walk_calls.cuh"
__global__ void kHH_gat(const T *const *src, size_t n, T *o) {
  GRID_LOOP(i, n) o[i] = *src[i];
}
// HV[hi] = eq(z, hi) for hi < o (z[0]: top bit of hi). Public sources: see plan_member_record().
__global__ void kHH_at(const T *z, int nb, int o, T *HV) {
  const int hi = threadIdx.x;
  if (hi >= o) return;
  T v = F::one();
  for (int q = 0; q < nb; q++) v = F::mul(v, ((hi >> (nb - 1 - q)) & 1) ? z[q] : F::sub(F::one(), z[q]));
  HV[hi] = v;
}
// For the product layer of the normalization's sum of squares, which gives all 2^lf features one
// weight: rho = (1/2, .., 1/2 | PV), so that the weight is 2^-lf, and claim[0] = 2^-lf sum_k
// HV[k], the true value of the layer's sum. Launch with one thread.
__global__ void kHH_norm_layer(const T *PV, int lf, int lt, const T *HV, T *rho, T *claim) {
  const T half = F::inv(F::small(2));
  T s = F::zero();
  for (int k = 0; k < (1 << lf); k++) s = F::add(s, HV[k]);
  for (int i = 0; i < lf; i++) {
    rho[i] = half;
    s = F::mul(s, half);
  }
  for (int i = 0; i < lt; i++) rho[lf + i] = PV[i];
  claim[0] = s;
}
// Absorbs the n values already stored in `slots` into the transcript, as one message. One thread.
__global__ void kHH_absorb(Tr *tr, const T *slots, int n) {
  Tr t = *tr;
  sc_door::absorb(t, slots, n);
  *tr = t;
}
// o[i] = 2^-lf for i < n.
__global__ void kHH_plain_weight(T *o, int n, int lf) {
  const T half = F::inv(F::small(2));
  T w = F::one();
  for (int i = 0; i < lf; i++) w = F::mul(w, half);
  GRID_LOOP(i, n) o[i] = w;
}
constexpr int kRowPassMax = 65536; // the longest row that `row_core_i8` accepts, in entries
struct HHidCfg {
  int lt = 0, la = 0, nmin = 0, d = 0, V = 0, D = 2048, NGr = 0, lmx = 0, hb = 100, lkW = 0, L = 0, mo = 0,
      xl = 4 /* limbs per residual-stream value; 5 if the parameter file has the line "xl 5" */;
  // The parameter file's line nminH: a call of the head has at least 2^nminH keys. 0: the file has
  // no such line, and nmin is the limit.
  int nminH = 0;
  // How many of the lowest levels of the forest's key trees are recomputed from the keys instead of
  // stored (2 to 5). More levels take less memory and more time. The proof does not depend on it.
  // Why 4: at a full-size configuration not in this repository, Forest::measure()'s buffers with 3
  // levels were more than the memory that the sub-proofs share.
  int rebuilt = 4;
  u64 seed = 0;
  std::function<u32(const std::string &)> word;
  std::string hcalls;
};
// The sub-proof's state. NT = 2^lt positions, LA = 2^la, ls = la - lt, Pd = 2^lPd >= d, NC calls.
struct HForkHid {
  HHidCfg c;
  HeadForward h;
  std::string lists;
  int NC = 0, NR = 0, NHf = 0, iB = 0, iXL = 0, gid = 0, lPd = 0, Pd = 0, ls = 0, NT = 0, cN = -1, cH = -1, fw = 0;
  size_t LA = 0, nW = 0, rscBytes = 0, cellsHF = 0, cellsHD = 0, cellsHR = 0, cellsB = 0, cellsX = 0,
         nH = 0, NSM = 0;
  // The filler cells that extend hf, hd and hr to their committed sizes: for each, first and count.
  std::vector<std::pair<const i8 *, size_t>> filler;
  std::vector<std::string> cname;
  std::vector<std::vector<std::string>> cseg;
  std::vector<int> knd, io /* log2 of a call's number of trees */, kk_, t0_, lJ, n;
  struct KK {
    std::vector<MatD> M;
  } K;
  LM hf, hd, hr, b, xl;
  std::vector<ForestCall> calls; // the calls, in the order of the callH lines
  ForestPlan plan;               // the forest's sizes; plan.NT counts its trees, NT the tokens
  Forest forest;
  PtGraph forest_graph;
  PtWPL WP;
  Tsc tV, tN, tH;
  TscEnd endV, endN; // the ends of tV's and tN's sumchecks; tH's continues in CL
  std::vector<u32 *> pub; // per call: each tree's number of padding keys (GPU memory)
  std::vector<std::vector<size_t>> padded; // per call (host): what `forest_padded_from` returns
  std::vector<T *> ec, phiB;
  const int16_t *wh16 = nullptr;
  i8 *bH = nullptr, *bB = nullptr;
  char *zfold_work = nullptr; // zfold_turned_i8()'s work buffer (common/zfold_turned.cuh)
  T *MSG = nullptr;
  hz::Tree *CP = nullptr;
  HiddenTail *owner = nullptr;
  int muOwner = 0;
  std::vector<std::string> LPn, SPn; // the names of the long and short points and weight vectors
  std::map<std::string, T *> phiP;
  std::map<std::string, size_t> OFF;
  std::vector<size_t> OFFB;
  std::vector<RowJob> jobs, jobsB;
  RowJob jobH;
  std::vector<size_t> chO, chK;
  T *bhPart = nullptr, *sBH = nullptr, *bhd = nullptr, *AV = nullptr, *wx = nullptr, *PV = nullptr,
    *ewH = nullptr, *erH = nullptr, *sR = nullptr, *eV = nullptr, *gam = nullptr, *rhoV = nullptr, *omV = nullptr,
    *A = nullptr, *FV = nullptr, *GV = nullptr, *FN = nullptr, *GN = nullptr, *FH = nullptr, *GH = nullptr,
    *co1 = nullptr, *coH = nullptr, *gnp = nullptr, *ewN = nullptr, *yb = nullptr, *HV = nullptr, *clW = nullptr;
  long long *Ws = nullptr;
  const ResidualInt *xLp = nullptr;
  const u32 *rows = nullptr; // the request's two words r, e (GPU memory)
  struct QQ {
    size_t ns = 0, nmsg = 0, slPT = 0, slVW = 0, slWPL = 0, slPV = 0, slTV = 0, slTN = 0, slTH = 0, slPZ = 0,
           slMSC = 0, nv[5] = {}, nLall = 0;
    T *vec[5] = {}, *LU[5] = {}, *Lall = nullptr, *one = nullptr;
    u32 *roots = nullptr, *zs = nullptr;
    std::vector<T *> Lrow;
  } Q;
  HiddenTail HT;
  hz::Tree S;
  hz::TreeN U[5];
  hw::Walk W;
  he::Eqs EQ;
  // EQ lists U3 (the tables; of U only U[3] is used), then ZZ and BZ, the range check's digits and
  // multiplicity bits. HT's matrices of packed cells, HT.M, are not in EQ: see seal_equations().
  hd::Door door; // the coefficient vectors of the walk's equation, for hd::close()
  HeadClaim CL;

  i8 *xbL = nullptr;
  const u64 *tFCd = nullptr;
  u64 tFC0 = 0;
  // For the calls of kind FE (lookups into the table whose row u holds u + 2^16 EX[u]): the table,
  // its value at the padding row 0, the length of segment HE0, and whether these calls read their
  // row numbers directly from the key array of HD1's call.
  const u64 *tFEd = nullptr;
  u64 tFE0 = 0;
  long long JVa = 0;
  bool keysInPlace = false;
  u32 sbase = 0;
  std::vector<hw::TableRead> HFUN;
  T *fvp = nullptr, *fv_mask = nullptr, *recD = nullptr, *HVk = nullptr;
  const T **dSrc = nullptr;
  size_t nrec = 0;
  const T *T_H() const { return CL.end.TH_; } // 2^lkW elements
  int callOf(const char *sg) const { return h.L.seg[h.L.segid(sg)].call; }
  static void stop(const char *s) {
    printf("%s\n", s);
    name_the_end("head-setup"); // serving: p258 end 1 head-setup (M5g: the head set-up's own stops)
    exit(2);
  }
  LM put(i8 *buf, size_t m, size_t cells, int np) {
    LM lm{(int)K.M.size(), np};
    const size_t per = cells / np;
    for (int p = 0; p < np; p++) {
      MatD X{};
      X.m = m;
      X.k = per / m;
      X.v = buf + p * per;
      K.M.push_back(X);
    }
    return lm;
  }
  size_t cellsOf(const LM &lm) const { return K.M[lm.first].m * K.M[lm.first].k * lm.n; }
  // Reads and checks the parameter file's numbers and callH lines. Not checked: each ends in '\n'.
  void read_policy() {
    using namespace hw;
    const int lt = c.lt, d = c.d, V = c.V, lkW = c.lkW;
    auto Wd = [&](const char *nme) {
      return c.word(nme);
    };
    NT = 1 << lt;
    ls = c.la - lt;
    LA = (size_t)1 << c.la;
    lPd = l2i(d);
    Pd = 1 << lPd;
    gid = 1 + c.NGr;
    fw = c.NGr;
    NHf = Wd("NHf");
    const int LS = 1 << ls;
    if (V < 2 || V > (1 << 20) || d % LS)
      stop("policy: V must be from 2 to 2^20, and d must be a multiple of 2 to the power (la minus lt)");
    if (l2i(V) > lkW) stop("policy: lkW must be at least ceil(log2 V); raise lkW [hid-vocab-cols]");
    {
      size_t p = 0;
      const std::string &s = c.hcalls;
      while (p < s.size()) {
        size_t q = s.find('\n', p);
        std::string ln = s.substr(p, q - p);
        p = q + 1;
        std::vector<std::string> tk;
        size_t a = 0;
        while (a < ln.size()) {
          size_t e = ln.find(' ', a);
          if (e == std::string::npos) e = ln.size();
          if (e > a) tk.push_back(ln.substr(a, e - a));
          a = e + 1;
        }
        if (tk.size() < 4) stop("policy: each callH line needs a name, a kind and at least one segment");
        const int kd = tk[2] == "R16" ? 0 : tk[2] == "R8" ? 1 : tk[2] == "FC" ? 2 : tk[2] == "FE" ? 3 : -1;
        if (kd < 0) stop("policy: callH kind must be R16, R8, FC (the clamp's call) or FE (the exponential's calls)");
        if (tk[1].compare(0, tk[2].size(), tk[2])) stop("policy: a callH line's name must begin with its kind");
        cname.push_back(tk[1]);
        knd.push_back(kd);
        cseg.push_back(std::vector<std::string>(tk.begin() + 3, tk.end()));
      }
    }
    NC = (int)cname.size();
    if (NC < 1 || NC > 6) stop("policy: the head's sub-proof needs 1 to 6 callH lines");
    for (int q = 0; q < NC; q++) {
      lists += cname[q] + ":";
      for (size_t i = 0; i < cseg[q].size(); i++) lists += (i ? "," : "") + cseg[q][i];
      lists += ";";
    }
    lists.pop_back();
  }
  // Allocates the cells, the lookup keys and the range check's buffers; lists the matrices in K.M.
  void lay_out_matrices() {
    using namespace hw;
    const int lt = c.lt, d = c.d, V = c.V;
    // prove() launches kHH_at with 32 threads; it needs one per output, d / 2^z of them (2^z: the
    // largest power of 2 dividing d). Same limit: prover/common/parameter_file.h, `prover-only-d`.
    if (d / (d & -d) > 32) stop("policy: d divided by the largest power of two that divides d must be at most 32");
    auto Wd = [&](const char *nme) {
      return c.word(nme);
    };
    HeadShape s;
    s.lt = lt;
    s.d = d;
    s.V = V;
    s.la = c.la;
    s.nmin = c.nminH ? c.nminH : c.nmin;
    s.MF = Wd("MF");
    s.KF = Wd("KF");
    s.epsF = (long long)(Wd("epsF_lo") | ((u64)Wd("epsF_hi") << 32));
    s.C = Wd("C");
    s.theta = Wd("theta");
    s.cap = Wd("cap");
    if (s.theta > s.C || s.cap > (u32)NT || s.cap >= 65536)
      stop("policy: theta must be at most C, and cap at most 2^lt and below 2^16 [policy-gap-words]");
    if (Wd("sh") > 8 || Wd("s") > 63 || Wd("sT") > 63 || Wd("F") != 24)
      stop("policy: sh must be at most 8, s and sT at most 63, and F must be 24");
    s.sh = (int)Wd("sh");
    s.s = (int)Wd("s");
    s.sT = (int)Wd("sT");
    s.F = (int)Wd("F");
    const int q = s.sT - s.sh; // selects EX: -4 is the step table, 4 to 10 is exp(-u / 2^q)
    if (s.sh < 7 || (q != -4 && (q < 4 || q > 10)) || 2ll * V > 1ll << s.F)
      stop("policy: sh must be 7 or 8, sT - sh either -4 or from 4 to 10, and 2 V at most 2^24");
    s.xl = c.xl;
    s.lists = lists.c_str();
    h.L = make_head_layout(s);
    h.seed = c.seed;
    const HeadLayout &L = h.L;
    for (const char *nme : {"NF", "NFH", "HD0", "HD1", "ROW", "HD2", "CNF", "HE0", "HE1"}) {
      int times = 0;
      for (auto &sg : L.seg) times += sg.name == nme;
      if (times != 1)
        stop("policy: callH lines must name each of NF, NFH, HD0, HD1, ROW, HD2, CNF, HE0, HE1 exactly once");
    }
    cN = callOf("NF");
    cH = callOf("HD0");
    JVa = L.seg[L.segid("HE0")].n;
    for (const char *nme : {"HD1", "HE0", "HE1"})
      if (cseg[callOf(nme)].size() != 1 || JVa % (1 << ls))
        stop("policy: HD1, HE0 and HE1 must each be the only segment of its call, and HE0's "
             "length must be a multiple of 2 to the power (la minus lt)");
    auto up = [](size_t x, size_t u) {
      return (x + u - 1) / u * u;
    };
    const int lmb = Wd("lmbH");
    const size_t mHF = (size_t)1 << Wd("lmhf"), mHD = (size_t)1 << Wd("lmhd"), mHR = (size_t)1 << Wd("lmhr"),
                 mB = (size_t)1 << lmb, mx = (size_t)1 << c.lmx;
    cellsHF = up((size_t)L.G[0] * NT, mHF * NHf);
    cellsHD = up((size_t)L.G[1] * NT, mHD);
    cellsHR = up((size_t)L.G[2] * NT, mHR);
    cellsX = up((size_t)c.xl * d * NT, mx);
    h.hf = dalH<i8>(cellsHF, Place::Fork);
    h.hd = dalH<i8>(cellsHD, Place::Fork);
    h.hr = dalH<i8>(cellsHR, Place::Fork);
    filler = {{h.hf + (size_t)L.G[0] * NT, cellsHF - (size_t)L.G[0] * NT},
              {h.hd + (size_t)L.G[1] * NT, cellsHD - (size_t)L.G[1] * NT},
              {h.hr + (size_t)L.G[2] * NT, cellsHR - (size_t)L.G[2] * NT}};
    h.keyN.clear();
    for (int q = 0; q < NC; q++) {
      const size_t nk = ((size_t)1 << L.call[q].lJ) * NT;
      h.keyN.push_back(nk);
      h.key[q] = knd[q] == 3 ? nullptr : dalH<u16>(nk, Place::Fork);
    }
    {
      /* For a call of kind FE the GPU stores, instead of a key u + 2^16 EX[u], only the row number
         u. These row numbers are HD1's keys, entry by entry: HE0 has those of the first JVa columns
         of HD1's call, and HE1 has those of the columns from JVa on. If each of the two calls has
         JVa columns and HD1's call has twice as many, they read the two halves of HD1's array in
         place (`keysInPlace`). Otherwise they get separate arrays, which `forward_and_counts`
         fills. */
      // Public source: parts of an array as views, not copies: PyTorch's torch.chunk.
      // Public source: parts of an array as views, not copies: NumPy's numpy.split.
      const int c1 = callOf("HD1"), ca = callOf("HE0"), cb = callOf("HE1");
      keysInPlace = h.keyN[ca] == (size_t)JVa * NT && h.keyN[cb] == h.keyN[ca] && h.keyN[c1] == 2 * h.keyN[ca];
      h.key[ca] = keysInPlace ? h.key[c1] : dalH<u16>(h.keyN[ca], Place::Fork);
      h.key[cb] = keysInPlace ? h.key[c1] + h.keyN[ca] : dalH<u16>(h.keyN[cb], Place::Fork);
      // Uploads the table of exponentials EX for the forward pass, and the calls' table, whose row
      // u holds u + 2^16 EX[u].
      std::vector<u64> tau(L.EX.size());
      for (size_t u = 0; u < tau.size(); u++) tau[u] = u + ((u64)L.EX[u] << 16);
      u32 *ex = dalH<u32>(L.EX.size(), Place::Own);
      u64 *fe = dalH<u64>(tau.size(), Place::Own);
      CK(cudaMemcpy(ex, L.EX.data(), L.EX.size() * 4, cudaMemcpyHostToDevice));
      CK(cudaMemcpy(fe, tau.data(), tau.size() * 8, cudaMemcpyHostToDevice));
      h.EX = ex;
      tFEd = fe;
      tFE0 = tau[0];
    }
    h.Rv = dalH<u64>(NT, Place::Fork);
    h.sigv = dalH<u64>(NT, Place::Fork);
    h.err = dalH<u32>(1, Place::Result);
    h.Wg = dalH<unsigned long long>((size_t)NORM_SUM_COUNT_MOST * NT, Place::Fork);
    h.y = dalH<int>(1, Place::Result);
    h.scratch = dalH<char>(h.scratch_bytes(), Place::Fork);
    h.posRow = dalH<int>((size_t)L.t[Y_row].J, Place::Own);
    {
      std::vector<int> pr((size_t)L.t[Y_row].J);
      const int id = L.segid("ROW");
      for (size_t x = 0; x < pr.size(); x++) pr[x] = (int)L.pos(id, x);
      CK(cudaMemcpy(h.posRow, pr.data(), pr.size() * 4, cudaMemcpyHostToDevice));
    }
    hf = put(h.hf, mHF, cellsHF, NHf);
    hd = put(h.hd, mHD, cellsHD, 1);
    hr = put(h.hr, mHR, cellsHR, 1);
    for (int q = 0; q < NC; q++) {
      kk_.push_back(knd[q] == 1 ? 8 : 16);
      t0_.push_back(knd[q] == 2 ? 32768 : 0);
      lJ.push_back(L.call[q].lJ);
      n.push_back(lJ[q] + lt);
      ForestCall call;
      call.n = n[q];
      call.k = kk_[q];
      call.fn = knd[q] >= 2 ? 1 : 0;
      call.tau = knd[q] == 2 ? tFCd : knd[q] == 3 ? tFEd : nullptr;
      call.t0 = (u32)t0_[q];
      calls.push_back(call);
    }
    plan = forest_plan(calls, identity_cap());
    // With trees of at least 2^la keys, the last la coordinates of every call's point are the
    // same, so that one table of a matrix can serve all calls (setup_tables()).
    if (plan.dT < c.la)
      stop("policy: a tree of the head's lookups must have at least 2^la keys; raise nminH (or nmin) [forest-depth-H]");
    for (int q = 0; q < NC; q++) {
      io.push_back(plan.iota[q]);
      if ((int)Wd(("iotaH_" + std::to_string(q)).c_str()) != io[q]) {
        printf("policy: the line iotaH_%d must be %d: log2 of the call's keys, %d, minus log2 of a tree's keys, %d "
               "[shape-call-H]\n", q, io[q], n[q], plan.dT);
        name_the_end("head-setup"); // serving: p258 end 1 head-setup (M5g)
        exit(2);
      }
    }
    if (lmb < 5 || lmb > plan.kT + 5)
      stop("policy: lmbH must be from 5 to 5 + log2 of the largest table's rows [forest-planes]");
    // A call has 2^kT rows of 32 multiplicity bits, even with a shorter table: whole columns of b.
    cellsB = plan.plane_bits(); // one vector of multiplicities a call
    bB = dalH<i8>(cellsB, Place::Fork);
    b = put(bB, mB, cellsB, 1);
    xl = put(xbL, mx, cellsX, 1);
    NR = (int)K.M.size();
    iB = b.first;
    iXL = xl.first;
    nH = (size_t)2 * d * V;
    bH = dalH<i8>(nH, Place::Fork);
    for (int i = 0; i < NR; i++) {
      MatD &X = K.M[i];
      X.support = (uint4 *)dalH<u32>(X.k * (ETAU / 32), Place::Fork); // ETAU bits per column
      X.sign = (uint4 *)dalH<u32>(X.k * (ETAU / 32), Place::Fork);
      X.Cf = dalH<i8>(X.k * ETAU, Place::Fork);
      X.CfT = dalH<i8>((size_t)zfold_kp((int)X.k) * ETAU, Place::Fork);
    }
    zfold_work = dalH<char>(ZFOLD_TURNED_WORKSPACE, Place::Fork);
    sbase = 4u * (4096u + 64u * (u32)gid);
  }
  void setup_tail() {
    std::vector<int> cls(NR, 0), shr(NR, 0);
    cls[iXL] = 1;
    HT.init(K.M.data(), NR, cls.data(), shr.data(), c.hb, sbase + 64, owner, iXL, muOwner);
  }
  /* Sets up, in this order, the lookup forest, the range check, each tree's number of padding keys
     and the sumchecks WP, tV, tN and tH. The order matters for memory: see "early region". */
  void setup_product_tests() {
    using namespace hw;
    const int lt = c.lt, V = c.V;
    const HeadLayout &L = h.L;
    /* The "early region" holds buffers that nothing reads after the forest's proof: the forest's
       scratch memory, its multiplicities and their bit masks. The range check's buffers marked
       hz::Late (prover/hiding/tail_f.cuh) reuse it where they fit, from `fork_memory.late_begin`
       in `prove` on. The calls' keys are not in it; the forest reads them until its proof ends. */
    fork_memory.early_begin();
    forest.init(calls, identity_cap(), 18, c.rebuilt);
    fork_memory.early_end();
    setup_tail();
    for (int q = 0; q < NC; q++) {
      // Counts the padding keys of each tree. The key at column j and position t has the index
      // (j << lt) + t, so a column is within one tree (dT >= la >= lt). A column in no segment, or
      // of a vocabulary index of V or more, holds the padding key at every position. A zero at a
      // position outside the request's range is an ordinary key, not a padding key.
      const size_t I = (size_t)1 << io[q], J = (size_t)1 << lJ[q];
      std::vector<char> used(J, 0);
      for (auto &pc : L.call[q].pc) {
        const std::string &nme = L.seg[pc.seg].name;
        const long long first = nme == "HE1" ? JVa : 0; // the segment's first vocabulary index
        for (long long x = 0; x < pc.size; x++) used[pc.off + x] = nme[0] != 'H' || first + pc.s + x < V;
      }
      std::vector<u32> pb(I, 0);
      ForestRuns runs;
      for (size_t j = 0; j < J; j++)
        if (!used[j]) {
          pb[j >> (plan.dT - lt)] += (u32)NT;
          runs.add(j << lt, (j + 1) << lt);
        }
      padded.push_back(forest_padded_from(plan.dT, io[q], runs, pb));
      pub.push_back(dalH<u32>(I, Place::Own));
      CK(cudaMemcpy(pub[q], pb.data(), I * 4, cudaMemcpyHostToDevice));
      ec.push_back(dalH<T>(J, Place::Fork));
    }
    WP.init(lPd + lt);
    // End values sent: by tV the normalization's per-position values that enter its products, by
    // tN its two factors. tH's rounds over the positions are followed by lPd rounds over the
    // features (prover/head/claims.cuh), and its two factors are sent at the end of those.
    tV.init(lt, norm_product_count(c.xl), TscShape{(size_t)norm_row_value_count(c.xl), 2, 0});
    tN.init(lt, 1, TscShape{2, 2, 0});
    tH.init(lt, c.d, TscShape{2, 2, lPd});
  }
  // Allocates the vectors named in LPn and SPn and the committed matrices' tables under them, and
  // plans the tables' row passes. OFF["hf@Z"] is the offset in Q.vec[3] of hf's table under Z.
  void setup_tables() {
    using namespace hw;
    const int d = c.d, V = c.V;
    auto Wd = [&](const char *nme) {
      return c.word(nme);
    };
    const size_t mB = (size_t)1 << Wd("lmbH");
    // Each name is a point p, with the vector eq(p, .), or a weight vector (phiZ, phiC, rhoC).
    // prove() fills them all. Z and z are the last la and the last lt coordinates of zx, the lookup
    // forest's point inside a tree. Every call's point ends in zx.
    LPn = {"Z", "phiZ", "Z:V", "Y"};
    SPn = {"z", "P_V", "r_tV", "r", "r_H", "phiC", "rhoC"};
    const std::vector<std::string> rdHF = {"Z", "phiZ"}, rdHD = {"z", "r_H"},
                                   rdHR = {"z", "P_V", "r_tV", "r", "phiC", "rhoC"};
    for (auto &p : LPn) phiP[p] = dalH<T>(LA, Place::Fork);
    for (auto &p : SPn) phiP[p] = dalH<T>(NT, Place::Fork);
    auto has = [](const std::vector<std::string> &v, const std::string &x) {
      return std::find(v.begin(), v.end(), x) != v.end();
    };
    auto cnt = [&](const std::vector<std::string> &pts, const std::vector<std::string> &rd) {
      size_t k = 0;
      for (auto &p : pts)
        if (has(rd, p)) k++;
      return k;
    };
    nW = cnt(LPn, rdHF) * (cellsHF / LA) + 2 * (cellsX / LA) + cnt(SPn, rdHD) * (cellsHD / NT) +
         cnt(SPn, rdHR) * (cellsHR / NT) + (size_t)3 * K.M[iB].k;
    T *sW = dalH<T>(nW, Place::Fork);
    Q.vec[3] = sW;
    Q.nv[3] = nW;
    auto addFun = [&](const LM &lm, size_t L_, const T *phi, int tree, size_t off) {
      for (int p = 0; p < lm.n; p++) {
        HFUN.push_back({lm.first + p, L_, phi, off, tree});
        off += K.M[lm.first + p].k * (K.M[lm.first + p].m / L_);
      }
    };
    T *w = sW;
    // Groups the vectors phis into row passes (RowJob) of at most 9 vectors over the matrix src.
    // Each output takes the next `rows` elements at the captured cursor w, which runs through the
    // tables in sW and is set to sBH for jobH.
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
        rscBytes = std::max(rscBytes, j.scratch());
        dst.push_back(j);
      }
    };
    auto tabs = [&](const char *mn, const LM &lm, const std::vector<std::string> &pts,
                    const std::vector<std::string> &rd, size_t L_) {
      std::vector<T *> ph;
      size_t o = w - sW;
      for (auto &p : pts)
        if (has(rd, p)) {
          ph.push_back(phiP[p]);
          OFF[std::string(mn) + "@" + p] = o;
          addFun(lm, L_, phiP[p], 3, o);
          o += cellsOf(lm) / L_;
        }
      mkjobs(jobs, K.M[lm.first].v, L_, cellsOf(lm) / L_, ph);
    };
    tabs("hf", hf, LPn, rdHF, LA);
    tabs("xl", xl, LPn, {"Z:V", "Y"}, LA);
    tabs("hd", hd, SPn, rdHD, NT);
    tabs("hr", hr, SPn, rdHR, NT);
    {
      // On b: tables at the two points ForestClaims::zb, then each column's sum of multiplicities.
      std::vector<T *> ph;
      size_t o = w - sW;
      for (int i = 0; i < 3; i++) {
        phiB.push_back(dalH<T>(mB, Place::Fork));
        ph.push_back(phiB[i]);
        addFun(b, mB, phiB[i], 3, o);
        OFFB.push_back(o);
        o += K.M[iB].k;
      }
      mkjobs(jobsB, bB, mB, K.M[iB].k, ph);
    }
    if (w != sW + nW)
      stop("internal error: head sub-proof: the number of short-table elements placed "
           "differs from the planned number");
    ewH = dalH<T>(V, Place::Fork);
    sBH = dalH<T>((size_t)2 * d, Place::Fork);
    bhd = dalH<T>(d, Place::Fork);
    AV = dalH<T>(V, Place::Fork);
    wx = dalH<T>(1, Place::Fork);
    w = sBH;
    {
      std::vector<RowJob> one;
      mkjobs(one, bH, V, (size_t)2 * d, {ewH});
      jobH = one[0];
    }
    if (V > kRowPassMax && V % 16 == 0) {
      const size_t un = V / 16, nch = (V + kRowPassMax - 1) / kRowPassMax + ((size_t)kRowPassMax % 16 ? 1 : 0);
      size_t at = 0;
      for (size_t q = 0; q < nch; q++) {
        const size_t u1 = un * (q + 1) / nch;
        chO.push_back(at * 16);
        chK.push_back((u1 - at) * 16);
        at = u1;
        rscBytes = std::max(rscBytes, row_core_scratch(1, (int)chK[q], 2 * d));
        if (chK[q] > (size_t)kRowPassMax || !chK[q])
          stop("internal error: head sub-proof: the head's rows cannot be cut into pieces of at "
               "most 65536 entries");
      }
      bhPart = dalH<T>(nch * 2 * d, Place::Fork);
    }
  }
  // Allocates scratch buffers in the memory that the sub-proofs share, and sets up endV and endN.
  void allocate_scratch() {
    using namespace hw;
    const int lt = c.lt, d = c.d;
    PV = dalH<T>(32, Place::Fork);
    erH = dalH<T>(NT, Place::Fork);
    sR = dalH<T>(1, Place::Fork);
    eV = dalH<T>(NT, Place::Fork);
    gam = dalH<T>(1, Place::Fork);
    rhoV = dalH<T>(64, Place::Fork);
    omV = dalH<T>(norm_product_count(c.xl), Place::Fork);
    A = dalH<T>((size_t)1 << (lPd + lt), Place::Fork);
    FV = dalH<T>((size_t)norm_product_count(c.xl) * NT, Place::Fork);
    GV = dalH<T>((size_t)norm_product_count(c.xl) * NT, Place::Fork);
    FN = dalH<T>(NT, Place::Fork);
    GN = dalH<T>(NT, Place::Fork);
    FH = dalH<T>((size_t)d * NT, Place::Fork);
    GH = dalH<T>((size_t)d * NT, Place::Fork);
    co1 = dalH<T>(1, Place::Fork);
    coH = dalH<T>(d, Place::Fork);
    gnp = dalH<T>((size_t)32 * NT, Place::Fork);
    ewN = dalH<T>(d, Place::Fork);
    yb = dalH<T>(64, Place::Fork);
    Ws = dalH<long long>((size_t)NT * norm_sum_count(c.xl), Place::Fork);
    HV = dalH<T>((size_t)Pd, Place::Fork);
    clW = dalH<T>(1, Place::Fork);
    // After the rounds, a row's first entry is its end value. k_rowterms(), k_gn2() fill the rows.
    const size_t n = (size_t)NT;
    if (c.xl == 5) // a, b, Rlo, Rhi; the products a a, a b, b b, Rlo a, Rlo b, Rhi a, Rhi b
      endV.setup(tV.shape, {FV, FV + 2 * n, FV + 3 * n, FV + 5 * n},
                 {TscTerm{0, 0, omV, nullptr}, TscTerm{0, 1, omV + 1, nullptr}, TscTerm{1, 1, omV + 2, nullptr},
                  TscTerm{2, 0, omV + 3, nullptr}, TscTerm{2, 1, omV + 4, nullptr}, TscTerm{3, 0, omV + 5, nullptr},
                  TscTerm{3, 1, omV + 6, nullptr}});
    else // sigma, Rlo, Rhi; the products sigma sigma, Rlo sigma, Rhi sigma
      endV.setup(tV.shape, {FV, FV + n, FV + 2 * n},
                 {TscTerm{0, 0, omV, nullptr}, TscTerm{1, 0, omV + 1, nullptr}, TscTerm{2, 0, omV + 2, nullptr}});
    endN.setup(tN.shape, {FN, GN}, {TscTerm{0, 1, nullptr, nullptr}}); // R, and the weighted stream
  }
  // Assigns the slots. Their order must be that of the message sections (see sections()).
  void lay_out_slots() {
    using namespace hw;
    const int d = c.d, V = c.V, hb_ = c.hb, lkW = c.lkW;
    size_t sl_ = 0;
    Q.slPT = sl_;
    sl_ += plan.nmsg();
    Q.slVW = sl_;
    sl_ += norm_sum_count(c.xl);
    Q.slWPL = sl_;
    sl_ += WP.nmsg();
    Q.slPV = sl_;
    sl_ += norm_product_count(c.xl);
    Q.slTV = sl_;
    sl_ += tV.nmsg();
    Q.slTN = sl_;
    sl_ += tN.nmsg();
    Q.slTH = sl_;
    sl_ += tH.nmsg();
    Q.slPZ = sl_;
    sl_ += HT.F.plan.nmsg();
    Q.slMSC = sl_;
    sl_ += sc_door::slots(sc_door::Form::SUM2, lkW, 1, 1); // MERGE; its last slot takes fv'
    Q.nmsg = sl_;
    U[3].setup("U3H", 1, Q.nv[3], EDEG, 1, sbase + 12, hb_);
    {
      he::Ent e;
      e.Tn = &U[3];
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
    Q.ns = Q.nmsg;
    S.setup("SH", 2, 0, Q.ns, 1, 2, sbase, hb_);
    if (S.sh.Kc > NSM)
      stop("internal error: head fork pad tree needs more coefficients than the shared message buffer holds");
    Q.roots = dalH<u32>(8 * DW, Place::Result);
    fvp = dalH<T>(1, Place::Result);
    fv_mask = dalH<T>(1, Place::Fork);
    CL.setup(lkW, d, V, MSG + Q.slMSC);
  }
  // Builds the walk W; returns 0. Spaces: 0 the slots; 3 the tables Q.vec[3]; 1, 2, 4 unused; from
  // 5 the virtual rows HT.vrows; spCP the K entries of the tree CP ("crossing pads"); then HT.M.
  int build_walk() {
    using namespace hw;
    const int lt = c.lt, d = c.d, lkW = c.lkW;
    const HeadLayout &L = h.L;
    const int LS = 1 << ls;
    W.init((size_t)24 << 20);
    W.scratch = MSG + 2 * NSM; // the sumchecks' scratch: one element for each slot
    const int nrows = (int)HT.vrows.size();
    W.sp.resize(6 + nrows);
    const int spCP = 5 + nrows;
    /* For `hd::close`: one more space for each matrix of the range check (HT.M), with one variable
       per column. LwM receives these columns' coefficients in the walk's equation, and LS0 those of
       the slots, then those of the crossing pads, which stay 0: no claim has a term on them. */
    W.sp.resize(6 + nrows + HT.NM);
    {
      door.ns = Q.ns, door.nC = CP->K, door.iC = (size_t)fw, door.nLwM = 0;
      door.atM.assign(HT.NM, 0);
      for (int mu = 0; mu < HT.NM; mu++) door.atM[mu] = door.nLwM, door.nLwM += HT.M[mu].k;
      door.LwM = dalH<T>(door.nLwM, Place::Fork);
      door.LS0 = dalH<T>(door.ns + door.nC, Place::Fork);
      for (int mu = 0; mu < HT.NM; mu++) {
        Space &z = W.sp[spCP + 1 + mu];
        z.lam = door.LwM + door.atM[mu];
        z.len = HT.M[mu].k;
      }
      W.sp[spCP].lam = door.LS0 + door.ns;
      W.sp[0].lam = door.LS0;
    }
    {
      Space &z = W.sp[spCP];
      z.len = CP->K;
    }
    {
      Space &z = W.sp[0];
      z.len = Q.ns;
      z.neg = 1;
    }
    {
      size_t nn = Q.nv[3];
      for (auto &v : HT.vrows) nn += v.T->K;
      Q.nLall = nn;
      Q.Lall = dalH<T>(nn, Place::Fork);
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
    std::vector<HE> p256v(8);
    p256v[0] = hE(1);
    for (int i = 1; i < 8; i++) p256v[i] = hmul(p256v[i - 1], hE(256));
    const T *P256 = W.cvec(p256v);
    auto zkp = [&](int q) {
      return (const T *)forest.claims->zk[q];
    };
    std::map<std::string, const T *> EWc, EWHc;
    std::vector<const T *> ehc(NC, nullptr);
    auto EW = [&](const char *sg) {
      auto it = EWc.find(sg);
      if (it != EWc.end()) return it->second;
      const int id = L.segid(sg), q = L.seg[id].call;
      std::vector<u32> ix(L.seg[id].n);
      for (size_t x = 0; x < ix.size(); x++) ix[x] = (u32)L.pos(id, (long long)x);
      return EWc[sg] = W.gather(ec[q], ix);
    };
    auto EWH = [&](const char *sg) {
      auto it = EWHc.find(sg);
      if (it != EWHc.end()) return it->second;
      const int id = L.segid(sg), q = L.seg[id].call;
      if (!ehc[q]) ehc[q] = W.eq(zkp(q), lJ[q] - ls);
      std::vector<u32> ix(L.seg[id].n / LS);
      for (size_t x = 0; x < ix.size(); x++) ix[x] = (u32)(L.pos(id, (long long)x * LS) / LS);
      return EWHc[sg] = W.gather(ehc[q], ix);
    };
    Ten XBT;
    XBT.nl = c.xl;
    XBT.cnt = 1;
    XBT.J = d;
    XBT.off = 0;
    // Returns the linear form evaluating tensor X in the long table tb. Its coefficients EWH(sg)
    // have one entry per block of LS columns. Segment sg's entries are X's from index `first` on.
    auto hSx = [&](const char *sg, const std::string &tb, const Ten &X, long long first = 0) {
      AF a;
      const T *v = EWH(sg);
      const size_t blocks = (size_t)L.seg[L.segid(sg)].n / LS;
      for (int l = 0; l < X.nl; l++)
        a = a + rng(3, OFF.at(tb) + (size_t)(X.off + (long long)l * X.J + first) / LS, 1, blocks, v, 1, Sy(p256v[l]));
      return a;
    };
    auto hXSr = [&](const std::string &tb, const Ten &X, size_t nn, const T *vec) {
      AF a;
      for (int l = 0; l < X.nl; l++)
        a = a + rng(3, OFF.at(tb) + (size_t)(X.off + (long long)l * X.J), 1, nn, vec, 1, Sy(p256v[l]));
      return a;
    };
    const std::string hfZ = "hf@Z";
    const ForestForms forms = walk_forest(W, forest, Q.slPT);
    const int limbs = c.xl, NS = norm_sum_count(limbs), NR = norm_row_len(limbs); // the members xl (a matrix) and NR mean other things
    const T *gp = W.take(NS);
    {
      T *gpw = (T *)gp;
      const T *gm = gam;
      W.prep.push_back([=] { k_pows<<<1, 1>>>(gm, NS, gpw); });
    }
    const int muV = lPd + lt;
    {
      // The product layer `WP` is a sumcheck on A times A, each feature weighted by 2^-lPd (see
      // prove()). It starts from 2^-lPd sum_s gam^s W_s (slot slVW + s). Its end values are equal.
      std::vector<const T *> r(muV);
      for (int i = 0; i < muV; i++) r[i] = &WP.chd[0].r[i];
      const Rec R = W.psc(0, r, Q.slWPL, 2, 2, &WP.out->tau);
      W.row(R, rng(0, Q.slVW, 1, NS, gp) * Sy(F::inv(h2(lPd))));
      W.lin(W.shown(R, 0) - W.shown(R, 1));
      const T *e2v = W.eq(WP.out->r, lPd - ls);
      AF v = W.shown(R, 0);
      for (int l = 0; l < limbs; l++)
        v = v - rng(3, OFF.at("xl@Z:V") + ((size_t)l * d) / LS, 1, d / LS, e2v, 1, Sy(gp + l));
      W.lin(v);
    }
    const Ten &XR_ = L.t[Y_row];
    auto rowAt = [&](const std::string &pt, int idx) {
      return OFF.at("hr@" + pt) + (size_t)XR_.off + idx;
    };
    auto Dg = [&](const std::string &pt, int idx) {
      return var(3, rowAt(pt, idx)) + AF(hE(128));
    };
    auto numA = [&](const std::string &pt, int at, int nn) {
      AF a = rng(3, rowAt(pt, at), 1, nn, P256);
      HE k = hE(0);
      for (int q = 0; q < nn; q++) k = hadd(k, hmul(hE(128), p256v[q]));
      a.c = k;
      return a;
    };
    const T *SIGC = W.cvec({hE(2), hE(1), hE(256), hE(65536), hE(1 << 24)});
    auto sigA = [&](const std::string &pt) {
      AF a = rng(3, rowAt(pt, 44), 1, 5, SIGC);
      a.c = hmul(hE(128), hE(2 + 1 + 256 + 65536 + (1 << 24)));
      return a;
    };
    {
      std::vector<const T *> r(lt);
      for (int j = 0; j < lt; j++) r[j] = &tV.out->r[j];
      const Rec R = W.psc(0, r, Q.slTV + 1, 2, (size_t)norm_row_value_count(limbs), nullptr);
      const size_t slCl = Q.slTV, slPV = Q.slPV;
      W.lin(slot(slCl) - rng(0, slPV, 1, norm_product_count(limbs), omV));
      W.row(R, slot(slCl)); // the end values are sent masked; the claims below say what each one is
      if (limbs == 5) {
        // Digits and relations: see k_norm_ledger_wide() (prover/prover/forward.cuh). sigma = a + 2^24 b.
        const T *AC = W.cvec({hE(2), hE(1), hE(256), hE(65536)});
        auto aA = [&](const std::string &pt) {
          AF a = rng(3, rowAt(pt, 58), 1, 4, AC);
          a.c = hmul(hE(128), hE(2 + 1 + 256 + 65536));
          return a;
        };
        W.lin(W.shown(R, 0) - aA("r_tV"));
        W.lin(W.shown(R, 1) - numA("r_tV", 62, 2));
        W.lin(W.shown(R, 2) - numA("r_tV", 80, 3));
        W.lin(W.shown(R, 3) - numA("r_tV", 83, 3));
        AF cprev;
        for (int s = 0; s < 9; s++) {
          AF cn = s < 8 ? numA("P_V", 10 + 6 * s, 6) : Dg("P_V", 9);
          W.lin(slot(Q.slVW + s) + cprev - Dg("P_V", s) - cn * Sy(hE(256)));
          cprev = cn;
        }
        AF Vlo = numA("P_V", 0, 6), Vhi = numA("P_V", 6, 4), sg = aA("P_V") + numA("P_V", 62, 2) * Sy(h2(24)),
           l1 = numA("P_V", 64, 5), h1 = numA("P_V", 69, 5), Blo = numA("P_V", 74, 3), Bhi = numA("P_V", 77, 2),
           kap = Dg("P_V", 79), q1 = numA("P_V", 86, 3), rr = numA("P_V", 89, 5), rq = numA("P_V", 94, 5);
        auto Pj = [&](int j) { return slot(slPV + j); };
        W.lin(Pj(1) - Blo - Bhi * Sy(h2(23)));
        W.lin(Vlo - Pj(0) - Blo * Sy(h2(25)) - l1 + kap * Sy(h2(48)));
        W.lin(Vhi - Bhi - Pj(2) - kap);
        W.lin(l1 + h1 - sg * Sy(hE(2)));
        W.lin(Pj(4) + Pj(5) - q1);
        W.lin(Pj(6));
        W.lin(sg - Dg("P_V", 59) - Pj(3) * Sy(hE(2)) - q1 * Sy(h2(25)) - rr * Sy(hE(2)));
        W.lin(rr + rq - sg);
      } else {
      W.lin(W.shown(R, 0) - sigA("r_tV"));
      W.lin(W.shown(R, 1) - numA("r_tV", 57, 3));
      W.lin(W.shown(R, 2) - numA("r_tV", 60, 3));
      AF cprev;
      for (int s7 = 0; s7 < 7; s7++) {
        AF cn = s7 < 6 ? numA("P_V", 8 + 6 * s7, 6) : Dg("P_V", 7);
        W.lin(slot(Q.slVW + s7) + cprev - Dg("P_V", s7) - cn * Sy(hE(256)));
        cprev = cn;
      }
      AF Vv = numA("P_V", 0, 8), sg = sigA("P_V"), l1 = numA("P_V", 49, 4), h1 = numA("P_V", 53, 4),
         p1 = numA("P_V", 63, 3), rr = numA("P_V", 66, 4), rq = numA("P_V", 70, 4), pv0 = slot(slPV),
         pv1 = slot(slPV + 1), pv2 = slot(slPV + 2);
      W.lin(Vv - pv0 - l1);
      W.lin(l1 + h1 - sg * Sy(hE(2)));
      W.lin(pv2 - p1);
      W.lin(sg - Dg("P_V", 45) - pv1 * Sy(hE(2)) - p1 * Sy(h2(25)) - rr * Sy(hE(2)));
      W.lin(rr + rq - sg);
      }
    }
    {
      std::vector<const T *> r(lt);
      for (int j = 0; j < lt; j++) r[j] = &tN.out->r[j];
      const Rec R = W.psc(0, r, Q.slTN + 1, 2, 2, nullptr);
      W.row(R, slot(Q.slTN));
      W.lin(W.shown(R, 0) - numA("r", limbs == 5 ? 80 : 57, 6));
      W.lin(W.shown(R, 1) - hSx("NF", "xl@Y", XBT));
    }
    const T *EWMU = nullptr;
    {
      std::vector<HE> mv((size_t)XR_.J, hE(1)); // multipliers: the normalization's NR digits, then 1 for the head's
      for (int x = 0; x < NR; x++) mv[x] = hE(norm_digit_factor(limbs, x));
      const T *mc = W.cvec(mv), *ewr = EW("ROW");
      T *o = W.take(mv.size());
      const size_t nn = mv.size();
      W.prep.push_back([=] { k_vmul<<<1, 256>>>(ewr, mc, nn, o); });
      EWMU = o;
    }
    const int KF = L.s.KF;
    // Returns the form of one of a position's six numbers: the nn digits from digit `at` after the
    // normalization's NR digits, in hr's table under the vector pt. The numbers start at digit 0
    // (c, 5 digits), 5 (b, 1), 6 (u, 5), 11 (k, 5), 16 (z, 2) and 18 (zs, 3).
    auto ledger = [&](const char *pt, int at, int nn) {
      return rng(3, rowAt(pt, NR + at), 1, nn, P256);
    };
    auto LINl = [&](const std::string &nme) -> AF {
      const char *sg = nme.c_str();
      if (nme == "NF") return AF() - hSx(sg, hfZ, L.t[Y_aw]) * Sy(h2(KF)) - hSx(sg, hfZ, L.t[Y_ra]) * Sy(h2(16));
      if (nme == "NFH") return hSx(sg, hfZ, L.t[Y_ra]) * Sy(h2(32 - KF));
      // HD0's key is the margin's top digit, K_0 = (margin - 2^sh K_1 - K_2) / 2^(sh + 15). The
      // margin's part in the logits is 2^lPd times the sum in slot slTH, which the sumcheck over
      // positions and features sends first: there each feature has the weight 2^-lPd. The part
      // c + 2^37 b is multiplied by wx, the sum of the coefficients ewH[x] (see prove()).
      if (nme == "HD0")
        return (slot(Q.slTH, Sy(h2(lPd))) + (ledger("phiC", 0, 5) + ledger("phiC", 5, 1) * Sy(h2(37))) * Sy(wx) -
                hSx(sg, hfZ, L.t[Y_d1]) * Sy(h2(L.s.sh)) - hSx(sg, hfZ, L.t[Y_d2])) *
               Sy(F::inv(h2(L.s.sh + 15)));
      // A key of HE0 or HE1 is K_1 + 2^16 ex: the margin's middle digit and its exponential.
      if (nme == "HE0" || nme == "HE1") {
        const long long first = nme == "HE1" ? JVa : 0;
        return hSx(sg, hfZ, L.t[Y_d1], first) + hSx(sg, hfZ, L.t[Y_e], first) * Sy(h2(16));
      }
      if (nme == "HD1") return hSx(sg, hfZ, L.t[Y_d1]);
      if (nme == "HD2") return hSx(sg, hfZ, L.t[Y_d2]);
      if (nme == "ROW") return rng(3, rowAt("z", 0), 1, (size_t)XR_.J, EWMU);
      if (nme == "CNF") return hSx(sg, hfZ, L.t[Y_aw]) + hXSr("hd@z", L.t[Y_aF], d, EW(sg)) * Sy(h2(28));
      stop("internal error: head sub-proof: the walk requested the linear relation of a "
           "segment that has none");
      return AF();
    };
    auto rightOf = [&](int q) {
      AF a;
      if (q == cN) a = a + slot(Q.slTN);
      for (auto &sn : cseg[q]) a = a + LINl(sn);
      return a;
    };
    {
      // The join of all calls: forms.cend = - sum_q e_q keys[q] + a public constant. keys[q]: call
      // q's keys at its point. e_q: the weight of its trees at the forest's point, ForestClaims::e.
      std::vector<AF> keys(NC);
      for (int q = 0; q < NC; q++) keys[q] = rightOf(q);
      W.lin(forest_join(forest, forms, keys));
    }
    // The first two relations at this file's top, under 0/1 weights rhoC; c + k = C under phiC.
    // The prover forms no constants (prover/hiding/walk.cuh), so the right sides are left out.
    W.lin(ledger("rhoC", 0, 5) + ledger("rhoC", 5, 1) * Sy(hE(L.s.C)) + ledger("rhoC", 6, 5));
    W.lin(ledger("rhoC", 5, 1) + ledger("rhoC", 16, 2));
    W.lin(ledger("phiC", 0, 5) + ledger("phiC", 11, 5));
    {
      // The third relation, under phiC, over all JV columns of ex: those from V on hold 2^F.
      const Ten &E = L.t[Y_e];
      AF z = ledger("phiC", 18, 3);
      for (int l = 0; l < E.nl; l++)
        z = z + rng(3, OFF.at("hf@phiZ") + (size_t)(E.off + (long long)l * E.J) / LS, 1, (size_t)E.J / LS, nullptr, 1,
                    Sy(p256v[l]));
      W.lin(z);
    }
    walk_forest_planes(W, forest, forms, 3, OFFB[0], OFFB[1], OFFB[2], l2i(K.M[iB].m));
    walk_tail_f(W, HT, spCP + 1, Q.slPZ, HFUN);
    {
      // The head's matrix product: one sumcheck over the positions (point r_H), then the features,
      // from the sum in slot slTH. End values: the activations' and v_HD, that of U (see prove()).
      std::vector<const T *> r(lt + lPd);
      for (int j = 0; j < lt; j++) r[j] = &tH.out->r[j];
      for (int j = 0; j < lPd; j++) r[lt + j] = CL.sch_coins + j;
      const Rec R = W.psc(0, r, Q.slTH + 1, 2, 2, nullptr);
      W.row(R, slot(Q.slTH));
      W.lin(W.shown(R, 0) - hXSr("hd@r_H", L.t[Y_aF], d, W.eq(CL.sch_coins, lPd)));
      // MERGE starts from rho v_HD and sends fv'. Its claim is on this sub-proof's mask entries
      // only: what fv' is, is left for the test on the weight table (see fv at this file's top).
      std::vector<const T *> rm(lkW);
      for (int j = 0; j < lkW; j++) rm[j] = CL.end.rm + j;
      const Rec M = W.psc(1, rm, Q.slMSC, 1, 1, nullptr);
      W.row(M, W.shown(R, 1) * Sy(CL.end.rho));
    }
    W.seal();
    return 0;
  }
  // Fills in EQ's coefficient vectors and builds door.t, the list of terms. Always returns 0.
  int seal_equations() {
    using namespace hw;
    const int nrows = (int)HT.vrows.size();
    {
      std::vector<HE> o1(1, hE(1));
      Q.one = (T *)W.cvec(o1);
    }
    for (auto &e : EQ.list) {
      if (e.Tn == &U[3]) {
        e.cv.push_back(Q.one);
        e.L.push_back(Q.LU[3]);
        e.c0.push_back(0);
        e.c1.push_back(Q.nv[3]);
        e.vecE = Q.vec[3];
      }
      for (int r = 0; r < nrows; r++)
        if (HT.vrows[r].T == e.Tn) {
          e.cv.push_back(HT.vrows[r].cv);
          e.L.push_back(Q.Lrow[r]);
          const Space &z = W.sp[5 + r];
          e.c0.push_back(z.lo < z.hi ? z.lo : 0);
          e.c1.push_back(z.lo < z.hi ? z.hi : 0);
        }
    }
    door.add_eqs(EQ);
    HT.say_forms(door);  // marks the range check's row vectors as product forms
    door.plain_rows("ZZ", HT.zzPlain, 2 * HT.mz);
    door.plain_rows("BZ", HT.cBZ, HT.kb);
    for (int mu = 0; mu < HT.NM; mu++) door.add_matrix(HT.TH[mu], HT.M[mu].v, HT.cls[mu] != 0);
    Q.zs = dalH<u32>(4 * DW, Place::Fork); // receives ZZ and BZ's root in its first DW words
    CK(cudaMemset(Q.zs, 0, 4 * DW * sizeof(u32))); // under `Prover::chain_bind` these words are 0
    return 0;
  }
  void plan_member_record() {
    using namespace hw;
    const int d = c.d, lkW = c.lkW;
    /* Plans recD, the pending claim on the weights: `prove` fills it, and `member` returns it.
       recD's elements, in order: fv'; beta, the challenge in fv' = fv + beta c; rm, the lkW
       challenges of MERGE; ka, the claim's coefficient; and the row factor of the claim about Wh:
       HVk, with d / 2^z entries, then the challenges of SCH, the rounds over kk (`CL.rounds`), for
       the bits z - 1 down to CL.tb of kk. 2^z is the largest power of two that divides d. */
    // Public source: a vector as tensor factors: J. Bootle, A. Chiesa, J. Groth, "Linear-Time
    // Arguments with Sublinear Verification from Tensor Codes", TCC 2020, ePrint 2020/1426, section
    // 2.1 "IOPs with tensor queries", https://eprint.iacr.org/2020/1426. They state it for a
    // verifier's queries to proof messages; here it is a claim's row factor.
    // Public source: that eq factors in this way: Q. Dao, J. Thaler, "More Optimizations to
    // Sum-Check Proving", ePrint 2024/1210, section 1.1 and equation (4),
    // https://eprint.iacr.org/2024/1210.
    // The row factor is a tensor product. HVk, of free length, holds eq of the high coordinates at
    // the d / 2^z values that occur. Each other factor has length 2 and is passed as its challenge.
    {
      int z = 0;
      while (!((d >> z) & 1)) z++;
      const int o = d >> z;
      HVk = dalH<T>(o, Place::Own);
      std::vector<const T *> src;
      src.push_back(fvp);
      src.push_back(CL.end.beta);
      for (int j = 0; j < lkW; j++) src.push_back(CL.end.rm + j);
      src.push_back(CL.end.ka);
      for (int x = 0; x < o; x++) src.push_back(HVk + x);
      for (int i = 0; i < z - CL.tb; i++) src.push_back(CL.sch_coins + (lPd - z) + i);
      nrec = src.size();
      dSrc = Walk::up(src);
      recD = dalH<T>(nrec, Place::Result);
    }
  }
  // xbL_: the last residual stream's cells. owner_, muOwner_: the range check of the sub-proof that
  // committed them, and their matrix number there. tFCd_, tFC0_: the clamp's lookup table and its
  // value at the padding row. MSG_: the shared message buffer, (3 + SLOT_COUNT) NSM_ elements (not
  // checked): slots, mask entries, scratch, counters. CP_: only its K is read. Returns 0 or exits.
  int setup(const HHidCfg &cfg, i8 *xbL_, const u64 *tFCd_, u64 tFC0_, HiddenTail *owner_, int muOwner_, T *MSG_,
            size_t NSM_, hz::Tree *CP_) {
    c = cfg;
    xbL = xbL_;
    tFCd = tFCd_;
    tFC0 = tFC0_;
    owner = owner_;
    muOwner = muOwner_;
    MSG = MSG_;
    NSM = NSM_;
    CP = CP_;
    read_policy();
    lay_out_matrices();
    setup_product_tests();
    setup_tables();
    allocate_scratch();
    lay_out_slots();
    if (int rc = build_walk()) return rc;
    if (int rc = seal_equations()) return rc;
    plan_member_record();
    CK(cudaDeviceSynchronize());
    return 0;
  }
  void weights_from(const int16_t *wh16_) {
    wh16 = wh16_;
    h.whsrc = wh16_;
  }
  // Runs only the forward pass. tok points to the request's NT tokens and rows_ to its two words
  // r, e, both in GPU memory.
  void forward(const ResidualInt *xL, const int *tok, const u32 *rows_) { h.forward(xL, tok, rows_); }
  void weights() {
    if (wh16 && c.d % 2 == 0 && c.V % 4 == 0 && !(((size_t)wh16 | (size_t)bH) & 3))
      kH_wkH_tiles<<<dim3((unsigned)((c.V + 127) / 128), (unsigned)((c.d + 63) / 64)), 256, 0, cudaStreamPerThread>>>(
          wh16, c.d, c.V, bH);
    else
      kH_wkH<<<grid_blocks(nH / 2), 256, 0, cudaStreamPerThread>>>(wh16, c.seed, c.d, c.V, bH);
  }
  // Runs the forward pass, which fills the cells and all keys but HE0's and HE1's: copies of HD1's
  // keys, made here unless `keysInPlace`. Then counts each call's keys and copies the multiplicity
  // bits into b. xL and rows_ are stored, not copied: they must stay valid until prove() returns.
  void forward_and_counts(const ResidualInt *xL, const int *tok, const u32 *rows_) {
    xLp = xL;
    rows = rows_;
    CK(cudaMemsetAsync(h.hf, 0, cellsHF, cudaStreamPerThread));
    CK(cudaMemsetAsync(h.hd, 0, cellsHD, cudaStreamPerThread));
    CK(cudaMemsetAsync(h.hr, 0, cellsHR, cudaStreamPerThread));
    for (int q = 0; q < NC; q++) {
      if (t0_[q])
        k_fill_u16<<<grid_blocks(h.keyN[q]), 256, 0, cudaStreamPerThread>>>(h.key[q], h.keyN[q], (u16)t0_[q]);
      else
        CK(cudaMemsetAsync(h.key[q], 0, h.keyN[q] * 2, cudaStreamPerThread));
    }
    // Zeroes hf, hd and hr again. If the GPU accepted a fill but never ran it, no call would report
    // it, and an earlier sub-proof's bytes would be committed in the filler cells: the forward pass
    // does not write them, and the verifier does not require zeros. The request would get no proof.
    CK(cudaMemsetAsync(h.hf, 0, cellsHF, cudaStreamPerThread));
    CK(cudaMemsetAsync(h.hd, 0, cellsHD, cudaStreamPerThread));
    CK(cudaMemsetAsync(h.hr, 0, cellsHR, cudaStreamPerThread));
    // The forward pass writes no exponential at vocabulary indexes of V or more. There the cells must encode EX[0] =
    // 2^F. The three cells hold the digits of ex - 1, each minus 128, so each cell is 127.
    CK(cudaMemsetAsync(h.cell(Y_e, 0), 127, (size_t)3 * h.L.JV * NT, cudaStreamPerThread));
    forward(xL, tok, rows);
    if (!keysInPlace) {
      const u16 *k1 = h.key[callOf("HD1")];
      CK(cudaMemcpyAsync(h.key[callOf("HE0")], k1, (size_t)JVa * NT * 2, cudaMemcpyDeviceToDevice,
                         cudaStreamPerThread));
      CK(cudaMemcpyAsync(h.key[callOf("HE1")], k1 + (size_t)JVa * NT, (size_t)(h.L.JV - JVa) * NT * 2,
                         cudaMemcpyDeviceToDevice, cudaStreamPerThread));
    }
    for (int q = 0; q < NC; q++) forest.counts(q, h.key[q], pub[q], padded[q].data());
    forest.planes();
    // forest.bits(0) orders the bits by call, then by table row: the order of b's cells.
    CK(cudaMemcpyAsync(bB, forest.bits(0), cellsB, cudaMemcpyDeviceToDevice, cudaStreamPerThread));
  }
  // Runs the steps that need no challenge: forward_and_counts(), whose condition on xL and rows_
  // applies, and the drawing of the slots' mask entries (`pads` in identifiers).
  void phase1(const ResidualInt *xL, const int *tok, const u32 *rows_) {
    forward_and_counts(xL, tok, rows_);
    S.slot_pads(door.ns);
  }
  // sd is unused. rsc: row-pass scratch, >= rscBytes bytes. Returns 0, or 4 on an internal error.
  int prove(Tr *tr, u32 *sd, void *rsc, u32 *err, u32 *errc) {
    const int lt = c.lt, d = c.d, V = c.V, la = c.la, NM = NR;
    if (!lost_copies_door::left_out("slot_pads_copy")) // fault-injection hook (test builds only)
      k_pl2e<<<gsl_blocks_of(door.ns), 256>>>(S.h_plane(0, 0), (long long)S.plane_stride(), door.ns, MSG + NSM);
    // The launch above copies slot j's mask entry from S to MSG[NSM + j]. The kernels read it at
    // slot + PADOFF; that PADOFF is NSM is not checked. After a rejected launch the entries would
    // be missing, and unmasked values would go into the proof file. This check exits before that.
    CK(cudaGetLastError());
    lost_copies_door::strike("slot_pads_copy", MSG + NSM, EDEG * door.ns, true);
    // This sub-proof uses only the first door.ns mask entries of the message buffer. The others are
    // zeroed, because a layer's sub-proof may have left secret entries there.
    CK(cudaMemsetAsync(MSG + NSM + door.ns, 0, (NSM - door.ns) * sizeof(T), cudaStreamPerThread));
    slot_counts_begin(MSG);
    S.slot_pads_held(door.ns);
    ch::host.first(tr, (const u32 *)h.y);
    // The mask c of fv' is the next entry of the request's index-mask message, copied to fv_mask.
    // The mask entry of the slot of fv' stays as it was drawn and committed, and is not used.
    ch::host.index_mask_entry(fv_mask);
    ch::kept_pads_agree(S, door.ns); // self-check by value of the array that `ch::host.first` read
    lost_copies_door::strike("slot_pads_later", MSG + NSM, EDEG * door.ns, true); // fault injection
    forest_prove_graph(forest_graph, forest, tr, MSG + Q.slPT);
    k_forest_err<<<1, 1>>>(forest.claims, errc);
    k_point<<<1, 1>>>(tr, Round::ZERO, PV, lt);
    auto zk = [&](int q) {
      return (const T *)forest.claims->zk[q];
    };
    // zt, Z: the last lt and the last la coordinates of every call's point ("z", "Z" in SPn, LPn).
    const T *zt = forest.claims->zx + plan.dT - lt, *Z = forest.claims->zx + plan.dT - la;
    for (int q = 0; q < NC; q++) k_eq<<<gsl_blocks_of((size_t)1 << lJ[q]), 256>>>(zk(q), lJ[q], ec[q]);
    // ewH[x], vocabulary entry x's coefficient, is ec[cH] at x's column of HD0; wx is their sum.
    // sBH becomes g[kk], the sum over x of ewH[x] Wh[kk][x], as two limbs in the order [limb][kk].
    k_gather2<<<gsl_blocks_of(V), 256>>>(ec[cH], h.pos2("HD0"), V, ewH);
    if (chO.empty())
      jobH.run(rsc);
    else {
      for (size_t q = 0; q < chO.size(); q++) {
        const T *ph[1] = {ewH + chO[q]};
        T *ou[1] = {bhPart + q * 2 * d};
        row_core_i8<1>(bH + chO[q], (size_t)V, 2 * d, (int)chK[q], ph, ou, rsc);
      }
      kH_addch<<<grid_blocks(2 * d), 256>>>(bhPart, (int)chO.size(), (size_t)2 * d, sBH);
    }
    kG_sum1<<<1, 1024>>>(ewH, V, wx);
    T *sVW = MSG + Q.slVW, *sPV = MSG + Q.slPV;
    k_eq<<<gsl_blocks_of(NT), 256>>>(PV, lt, eV);
    const int NS = norm_sum_count(c.xl), NP = norm_product_count(c.xl);
    ws_of(c.xl, xLp, NT, d, 1, Ws);
    {
      const long long *Wp = Ws;
      if (!lost_copies_door::left_out("norm_sums")) // fault-injection hook (test builds only)
        k_tsum<<<NS, 256>>>(NT, eV, sVW, [=] __device__(int q, int t) { return ext_from_signed(Wp[(size_t)t * NS + q]); });
    }
    CK(cudaGetLastError()); // the kernel stores each sum with its mask entry added
    kHH_absorb<<<1, 1>>>(tr, sVW, NS);
    k_point<<<1, 1>>>(tr, Round::BAT_SIGMA, gam, 1);
    // The features' sums of squares, HV, are not sent: the product layer sums over the features.
    k_bg<<<16384, 256>>>(c.xl, xLp, NT, lt, d, Pd, 1, gam, eV, A, HV);
    kHH_norm_layer<<<1, 1>>>(PV, lPd, lt, HV, rhoV, clW);
    WP.prove(tr, A, A, rhoV, clW, MSG + Q.slWPL);
    k_rowterms<<<gsl_blocks_of(NT), 256>>>(c.xl, h.Rv, h.sigv, NT, NT, FV, GV);
    if (!lost_copies_door::left_out("norm_sums")) k_pvsum<<<NP, 256>>>(NT, eV, FV, GV, sPV);
    CK(cudaGetLastError());
    kHH_absorb<<<1, 1>>>(tr, sPV, NP);
    k_batching_powers<<<1, 1>>>(tr, omV, NP);
    tV.prove(tr, PV, omV, FV, GV, NP, MSG + Q.slTV);
    endV.run(tr);
    k_gather2<<<gsl_blocks_of(d), 256>>>(ec[cN], h.pos2("NF"), d, ewN);
    k_gn1<<<dim3((NT + 127) / 128, 1, 32), 128>>>(xLp, NT, d, ewN, gnp);
    k_gn2<<<gsl_blocks_of(NT), 256>>>(gnp, 32, h.Rv, NT, FN, GN);
    kH_ones<<<1, 32>>>(co1, 1);
    tN.prove(tr, zt, co1, FN, GN, 1, MSG + Q.slTN);
    endN.run(tr);
    // The sumcheck over positions t and features kk for the head's matrix product first sends
    // 2^-lPd times the sum over t of eq(zt, t) times the sum over kk of a[t][kk] U[kk][t]. FH is
    // a[t][kk], the normalized activations. GH is U[kk][t] = wx Wh[kk][w_t] - g[kk] at the
    // positions of the request's range, 0 elsewhere. The rounds over t end at r_H and leave a and U
    // at r_H in the first entry of each kk's row. CL.rounds() runs the rounds over kk and the end.
    const int *stated = h.part<int>(HeadForward::P_WIN); // the checked tokens w_t, as the forward pass stored them
    k_cellv<<<gsl_blocks_of((size_t)d * NT), 256>>>(h.cell(Y_aF, 0), (size_t)d * NT, 2, (size_t)d * NT, FH);
    kH_rows_of_U<<<grid_blocks((size_t)d * NT), 256>>>(bH, sBH, wx, stated, rows, d, V, NT, GH);
    kHH_plain_weight<<<grid_blocks(d), 256>>>(coH, d, lPd);
    tH.prove(tr, zt, coH, FH, GH, d, MSG + Q.slTH);
    kH_every<<<grid_blocks(d), 256>>>(GH, (size_t)NT, d, bhd);
    CL.rounds(tr, lt, FH, (size_t)NT, bhd);
    for (auto &p : LPn) {
      const T *pt;
      if (p == "phiZ") continue; // computed below, after phiC
      if (p == "Z:V")
        pt = WP.out->r + (lPd + lt - la);
      else if (p == "Y") {
        k_cat<<<1, 64>>>(zk(cN) + lJ[cN] - ls, ls, tN.out->r, lt, yb);
        pt = yb;
      } else
        pt = Z;
      k_eq<<<gsl_blocks_of(LA), 256>>>(pt, la, phiP[p]);
    }
    for (auto &p : SPn) {
      // rhoC is 1 at the counted positions (those of the request's range below NT - 1) and 0
      // elsewhere. phiC[t] = rhoC[t] eq(zt, t).
      if (p == "rhoC") {
        kH_callers_rows<<<grid_blocks(NT), 256>>>(nullptr, rows, NT, phiP[p]);
        continue;
      }
      const T *pt = p == "P_V"    ? PV
                    : p == "r_tV" ? tV.out->r
                    : p == "r"    ? tN.out->r
                    : p == "r_H"  ? tH.out->r
                                        : zt; // "z" and "phiC"
      k_eq<<<gsl_blocks_of(NT), 256>>>(pt, lt, phiP[p]);
      if (p == "phiC") kH_callers_rows<<<grid_blocks(NT), 256>>>(phiP[p], rows, NT, phiP[p]);
    }
    // phiZ repeats phiC LA / NT times: once per run of NT cells under a long coefficient vector.
    kH_on_every_feature<<<grid_blocks(LA), 256>>>(phiP["phiC"], NT, LA, phiP["phiZ"]);
    // The public vector in the claim about Wh: AV[x] = wx (the sum of eq(r_H, t) over the range's t
    // with w_t = x) - sR ewH[x]; sR sums eq(r_H, t) over the range. The first sum is a sparse
    // matrix's multilinear extension. Setty, "Spartan", ePrint 2019/550, treats committed ones.
    kH_stated_rows<<<grid_blocks(NT), 256>>>(phiP["r_H"], rows, NT, erH);
    kG_sum1<<<1, 1024>>>(erH, NT, sR);
    kH_AV<<<grid_blocks(V), 256>>>(ewH, sR, V, AV);
    kH_AV_tokens<<<1, 1>>>(stated, erH, wx, NT, V, AV);
    {
      const int lmb = l2i(K.M[iB].m);
      const size_t mb = (size_t)1 << lmb;
      for (int e = 0; e < 2; e++)
        hw::k_forest_phi_bits<<<gsl_blocks_of(mb), 256>>>(forest.claims->zb[e], plan.kT, plan.cK, lmb, phiB[e]);
      hw::k_forest_phi_powers<<<gsl_blocks_of(mb), 256>>>(lmb, phiB[2]);
    }
    for (auto &j : jobs) j.run(rsc);
    for (auto &j : jobsB) j.run(rsc);
    commit_unit(U[3], Q.vec[3], Q.roots + 3 * DW, tr,
                1); // commit point (`moment`) 1: the tables (0: cells; 2: the range check's ZZ, BZ)
    // The forest's proof is done. From here on the late buffers may use the early region.
    fork_memory.late_begin();
    // The range check's matrix Cf[c][u] in {-1, 0, 1}: column c of matrix i, combination u < ETAU.
    HT.draw_matrix(); // a round that the host runs, after the tables' sample
    for (int i = 0; i < NM; i++) zfold_prep<ETAU>(K.M[i].Cf, (int)K.M[i].k, K.M[i].CfT);
    for (int i = 0; i < NM; i++)
      zfold_turned_i8<ETAU>(K.M[i].v, K.M[i].m, (int)K.M[i].m, (int)K.M[i].k, K.M[i].CfT, zfold_kp((int)K.M[i].k),
                            K.M[i].Z, zfold_work);
    HT.fold_to_group2(tr, err, Q.zs);
    HT.group2(tr, MSG + Q.slPZ);
    // The range check's multiplicities depend on challenges, so their error word (for an overflow;
    // always 0: ForestTable::planes()) goes to err. errc: failures that depend only on the inputs.
    k_forest_err<<<1, 1>>>(HT.F.claims, err);
    HT.make_cv();
    // A test hook (prover/common/lie.cuh), empty in this repository: it leaves fv_mask as it is.
    lie::zeroed_in_memory<T>(0, lie::HEAD, lie::MASK, cudaStreamPerThread, [&](long long row, long long token, long long second) -> T * { return row || token || second ? nullptr : fv_mask; });
    CL.merge(tr, AV, bH, fv_mask);
    CK(cudaGetLastError()); // after a rejected launch the slot of fv' would still hold its old contents
    CK(cudaMemcpyAsync(fvp, MSG + Q.slMSC + CL.end.merge_slots() - 1, sizeof(T), cudaMemcpyDeviceToDevice,
                       cudaStreamPerThread)); // a copy of fv' for the record
    {
      int z = 0;
      while (!((d >> z) & 1)) z++;
      kHH_at<<<1, 32>>>(CL.sch_coins, lPd - z, d >> z, HVk);
    }
    kHH_gat<<<grid_blocks(nrec), 64>>>(dSrc, nrec, recD);
    // W.finish() derives the challenge that batches all linear claims and computes the coefficient
    // vectors that hd::close() reads. Nothing more is sent.
    CK(cudaMemsetAsync(Q.Lall, 0, Q.nLall * sizeof(T), cudaStreamPerThread));
    door.clear_async();
    W.finish(tr, door.LS0);
    // The closing: hd::close() passes the walk's equation to the frame of the request's proof.
    door.hS = MSG + NSM;
    slot_counts_end("head sub-proof", gid, MSG, door.ns);
    door.host_words = ch::host.blob();
    S.slot_pads_agree(MSG + NSM, door.ns); // self-check by value again, after the entries' last use
    ch::masking_pads_hold(MSG + NSM, door.ns); // by a second implementation [own-test-slot-pads]
    ch::host.index_mask_entry_holds(fv_mask);  // the mask of fv' is still the entry that the index-mask message holds
    if (int rc = hd::close(door, tr, "head sub-proof", gid)) return rc;
    return 0;
  }
  // Adds to p this sub-proof's rounds on the hash chain (the transcript), in prove()'s order:
  // those after the SEND (oracle message and sample) of commit point `moment`: 0 cells, 1 tables,
  // 2 ZZ, BZ. Call after build_walk(). Counterpart: HPlanH2::rounds (verifier/src/vf_output_head_plan.inc).
  void plan_rounds(blk::Plan &p, unsigned moment) const {
    const std::string where = "head";
    blk::HeadRounds x;
    x.lt = (unsigned)c.lt, x.lPd = (unsigned)lPd, x.lkW = (unsigned)c.lkW;
    x.sums = (unsigned)norm_sum_count(c.xl), x.products = (unsigned)norm_product_count(c.xl), x.row_values = (unsigned)norm_row_value_count(c.xl);
    // The walk's claims and the range check's equations: powers of one challenge batch them all.
    // HT.leads has one entry per matrix and, per entry of HFUN, one per run of L cells in a column.
    x.rows = W.nclaim + HT.equations(HT.leads.size() - (size_t)HT.NM);
    if (moment == 0) {
      blk::LookupSize lookup;
      lookup.v = (unsigned)(plan.dT + plan.rK), lookup.d = (unsigned)plan.d(), lookup.mu = 5,
      lookup.shifted = plan.NC > 1;
      blk::plan_lookup(p, where + ", look-up", lookup);
      blk::plan_head_own(p, where, x);
    }
    if (moment >= 1) HT.plan_rounds(p, where + ", smallness test", moment - 1); // its matrix; then the rest
    if (moment == 2) blk::plan_head_end(p, where, x);
  }
  // Appends this sub-proof's message sections to SC. A section of 0 words only keeps its place.
  void sections(std::vector<SecD> &SC) const {
    SC.push_back({"FORKH", nullptr, (size_t)0});
    SC.push_back({"H.FOREST", MSG + Q.slPT, EWORDS * plan.nmsg()});
    SC.push_back({"H.VW", MSG + Q.slVW, (size_t)EWORDS * norm_sum_count(c.xl)});
    SC.push_back({"H.WPL", MSG + Q.slWPL, EWORDS * WP.nmsg()});
    SC.push_back({"H.PV", MSG + Q.slPV, (size_t)EWORDS * norm_product_count(c.xl)});
    SC.push_back({"H.TSCV", MSG + Q.slTV, EWORDS * tV.nmsg()});
    SC.push_back({"H.TSC", MSG + Q.slTN, EWORDS * tN.nmsg()});
    SC.push_back({"H.TSCH", MSG + Q.slTH, EWORDS * tH.nmsg()});
    SC.push_back({"H.U3", Q.roots + 3 * DW, DW});
    SC.push_back({"H.ROOT-ZZ-BZ", Q.zs, DW});
    SC.push_back({"H.FOREST-9", MSG + Q.slPZ, EWORDS * HT.F.plan.nmsg()});
    SC.push_back(
        {"H.MSC", MSG + Q.slMSC, EWORDS * sc_door::slots(sc_door::Form::SUM2, c.lkW, 1, 1)}); // fv' is its last
    SC.push_back({"H.XR", nullptr, (size_t)0});
    SC.push_back({"H.COLS", nullptr, (size_t)0});
  }
  // Returns this sub-proof's pending claim on the weights as 32-bit words: recD's elements (see
  // plan_member_record()), EWORDS words each, and after rm two words: 1, the number of claims, and
  // ui, the index of Wh among the weight table's units (tensors). Call it after prove().
  std::vector<u32> member(int ui) const {
    std::vector<T> hh(nrec);
    CK(cudaMemcpy(hh.data(), recD, sizeof(T) * nrec, cudaMemcpyDeviceToHost));
    std::vector<u32> w;
    auto put = [&](const T &x) {
      for (int k = 0; k < EDEG; k++) {
        w.push_back((u32)coord(x, k));
        w.push_back((u32)(coord(x, k) >> 32));
      }
    };
    size_t at = 0;
    for (; at < 2 + (size_t)c.lkW; at++) put(hh[at]); // fv' | beta | rm
    w.push_back(1u);
    w.push_back((u32)ui);
    for (; at < nrec; at++) put(hh[at]);
    return w;
  }
};
