// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md. The words and short names of these sources: verifier/docs/source-map.md.

// cp_verify, the verifier program, one translation unit. A verification call checks one request's
// proof. It takes the configuration (policy.txt, the parameter file; KL.bin; layout.txt), the
// public key directory `ivk`, a request, the next token y (`reply` in identifiers and options), a
// proof file and the commitment ID (-id; without it the call ends at best NOT-A-VERIFICATION).
// Request (T token ids, then r and e) and y are files of 32-bit little-endian words. Only the
// build's version word (vf_common.h) is accepted.
// cpStagesRun() runs three stages, each in a child process on the same bytes, each a mode of
// runStage(), which runs only once in a process: it sets file-scope variables, and refuse() exits.
//   key      ivk/key.bin: the committer's proofs about the anchor, the first weight table
//   leaf     ivk/leaf/<k>.bin, the leaf file that the proof names: the LINK proof, which is meant
//            to tie leaf table k, a further table of the same weights, to the anchor
//   request  the request's proof, with its test on leaf table k
// Last line and exit status (cpUsage() prints all calls and options):
//   YES 0                      every stage passed every test
//   WITHIN LIMITS: <policy> 0  the same, but a token of the request was checked under an acceptance
//                              policy other than exact-match: it says only what C, theta, cap allow
//   NOT-A-VERIFICATION 14      no test that ran failed, but the call is not a verification
//   NO: <name> 1               a test failed
//   NOT CHECKED: <name> 5      the program or the machine failed: nothing has been verified
//   input-unreadable-<role>: .. 2  a file could not be read; 2 also after a bad command line
//   PART PASSED 3 (-key, -leaf); PLAN ONLY 4 (-plan); no last line 15 (standard output failed)
// "proof of work: .." precedes the first three and PART PASSED. It says whether a round meets the
// target (line sec: each round's error must be at most 2^-sec) only with its work. A verification
// call then prints "result-v1 ..": what the answer is about, to compare with what the caller asked.
// In identifiers and output, `fork` is a sub-proof (the argument for one part of the forward pass),
// `link` a segment (a sub-proof's part of a request's proof), `policy` the parameter file (but
// -policy and <policy> name the acceptance policy: Hdr), `header` the statement (the words that a
// request's proof is about), `entry` a leaf file, `refuse` reject, `covered` a position t with
// r <= t < e. "result:" has no plain function of the request or y (see statedW); r and e are on
// "covered:" and "result-v1". No environment variable is read.
#ifndef VF_SOURCE_DIGEST
#error "VF_SOURCE_DIGEST, the source digest, is not defined: the build script computes it"
#endif
static_assert(sizeof(VF_SOURCE_DIGEST) == 65, "VF_SOURCE_DIGEST must have 64 digits: the build script computes it");
// verifier/build.sh sets BLK_COUNT_HASH_CALLS to 1: the hash of blocks/ (tree leaves, vertices,
// chain rounds) counts its compressions. The printed q_V, the verifier's hash calls, is that count
// plus vf_earlier_hash_f.h's (acctCounted(), vf_common.h); without the macro the first is left out and no line
// says so. Not set here: all units of a program (cp_configured.h) must compile blocks/ alike.
#include "vf_smallness_matrices.h"
#include "vf_input_files.h"
#include "frame/heads.h"      // blocks/: the digest of a parameter file
#include "frame/parameters.h" // blocks/: the parameter file's first lines and their one parser
#include "vf_attention_shape.h"
#ifdef __linux__
#include <sys/prctl.h>
#endif
#include <sys/resource.h>
#include <sys/wait.h>
#include <fcntl.h>
#include <csignal>
#include "vf_subproof_engine.h"
#include "vf_smallness_test.h"
#include "vf_norm_digit_rows.h"
#include "vf_weights_layout.h"
#include "vf_attention_check.h"
#include <algorithm>
#include <map>
#include <functional>
#include <thread>
#include <atomic>
#include <future>
#include <memory>
#include <mutex>
#include <condition_variable>
#include <array>
#include <set>
#include <tuple>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include "frame/check.h"
#include "frame/check_entry.h"
#include "frame/check_key.h"
#include "cp_configured.h"
#include "frame/prints.h"
#include "frame/req_frame.h"
#include "frame/wf_units.h"
#include "vf_weights_pending_claim.h"
// The layout of one tensor, a named family of values in a committed matrix. mat, the request
// matrix: 0 af, 1 ad, 2 ar, 3 qa. Other fields: TenAt (vf_attention_shape.h).
struct Ten {
  string name;
  int mat = 0, nl = 0;
  u64 cnt = 0, J = 0, OFF = 0;
  TenAt at() const { return {OFF, (u64)nl, cnt, J}; }
};
typedef vector<std::pair<string, TenAt>> TenList;
// Rejects with "plan-tensor-layout" unless a matrix's owner and a reader (two sub-proofs) list the
// same tensors in the same order.
static void tensorsAgree(const TenList &owner, const TenList &reader) {
  if (owner.size() != reader.size()) refuse("plan-tensor-layout");
  for (size_t i = 0; i < owner.size(); i++) {
    const TenAt &o = owner[i].second, &r = reader[i].second;
    if (owner[i].first != reader[i].first || o.off != r.off || o.nl != r.nl || o.cnt != r.cnt || o.J != r.J)
      refuse("plan-tensor-layout");
  }
}
// Rejects with "plan-tensor-layout-fill-<arr>" unless the tensors cover each feature below feat
// exactly once and S2 divides each J: some reads take S2 features at once, from a multiple of S2.
static void tensorsFill(const TenList &ten, u64 feat, u64 S2, const char *arr) {
  const string name = string("plan-tensor-layout-fill-") + arr;
  vector<std::pair<u64, u64>> run; // one limb of one layer: its first feature and how many it has
  for (auto &t : ten) {
    const TenAt &X = t.second;
    if (!X.J || X.J % S2) refuse(name);
    for (u64 l = 0; l < X.nl; l++)
      for (u64 i = 0; i < X.cnt; i++) run.push_back({tenCell(X, l, i, 0), X.J});
  }
  std::sort(run.begin(), run.end());
  u64 at = 0;
  for (auto &r : run) {
    if (r.first != at) refuse(name);
    at += r.second;
  }
  if (at != feat) refuse(name);
}
// A lookup segment: a named part of a lookup call's keys (CallD): n columns of T keys per layer.
// The format pages say "a call's segment".
// pos[x]: its column x in the call. moe: of mixture-of-experts (MoE) layers; padding on dense ones.
struct Seg {
  string name;
  u64 n = 0;
  vector<u64> pos;
  bool moe = false;
};
struct CallD {
  Call c;
  string kind;
  vector<Seg> segs;
  u64 J = 0;
  int lJ = 0;
  vector<char> used;
  // True if column j holds padding keys on layer lam of its group. layout() counts them with this.
  // vf_layers_check.inc compares the counts with `joinPart`'s: no independent check, both use Seg::pos.
  bool padding(u64 lam, u64 j) const { return !used[j] || (used[j] == 2 && lam % 2 == 0); }
  Seg &seg(const string &n) {
    for (auto &s : segs)
      if (s.name == n) return s;
    refuse("internal-segment-" + n);
  }
};
// Places cd's segments in J = 2^lJ columns (lJ >= floorLg) as aligned power-of-two chunks.
// cd.used[j]: 0 padding, 1 a real key on every layer, 2 a real key on MoE layers (odd lam) only.
static void place(CallD &cd, int floorLg) {
  struct Pc {
    int s = 0;
    u64 first = 0, size = 0;
  };
  vector<Pc> pcs;
  for (size_t s = 0; s < cd.segs.size(); s++) {
    u64 first = 0;
    for (int b = 40; b >= 0; b--)
      if ((cd.segs[s].n >> b) & 1) {
        pcs.push_back({(int)s, first, 1ull << b});
        first += 1ull << b;
      }
    cd.segs[s].pos.assign(cd.segs[s].n, 0);
  }
  std::stable_sort(pcs.begin(), pcs.end(), [](const Pc &a, const Pc &b) { return a.size > b.size; });
  u64 off = 0;
  for (auto &p : pcs) {
    for (u64 x = 0; x < p.size; x++) cd.segs[p.s].pos[p.first + x] = off + x;
    off += p.size;
  }
  cd.lJ = clog2(off);
  if (cd.lJ < floorLg) cd.lJ = floorLg;
  cd.J = 1ull << cd.lJ;
  cd.used.assign(cd.J, 0);
  for (auto &s : cd.segs)
    for (u64 p : s.pos) cd.used[p] = s.moe ? 2 : 1;
}
// Public lookup table SIG: docs/format/FORMAT.md, "The digests", sig(t).
static int sigmoidTableEntry(int t) {
  double z = (t - 32768) / 256.0;
  return 16 * (int)round(4096.0 / (1.0 + exp(-z)));
}
// The parameter file's optional lines eg and smax: docs/format/FORMAT.md, "The digests", SiLU.
static int gateTableInputFractionBits = 8, SMAX = SMAXTOP;
// The optional line "rcl 1" clamps the router logits. How: verifier/docs/parameter-file.md, "The
// optional lines on the model's arithmetic".
static int RCL = 0;
// The optional line "gcl 1" clamps the gating value, the key at which table SILU is read. The wide
// key is gw = g0 + 256 gh, g0 its balanced low byte; the table is read at
// 256 clamp(gh, +-127) + g0. SILU is constant beyond +-16384, so the result is SILU(clamp(gw)).
static int GCL = 0;
// Table FS: docs/format/FORMAT.md, "The digests", Sigmoid.
static int sigmoidClampTableEntry(int t);
// Table SILU: docs/format/FORMAT.md, "The digests", SiLU; the callers add the 4096.
static int gateTableEntry(int t) {
  const int top = 64 << gateTableInputFractionBits;
  int c = t - 32768;
  if (c < -top) c = -top;
  if (c > top - 1) c = top - 1;
  double z = c / (double)(1 << gateTableInputFractionBits), v = round(64.0 * (z / (1.0 + exp(-z))));
  return v < -SMAX ? -SMAX : v > SMAX ? SMAX : (int)v;
}
static int sigmoidClampTableEntry(int t) {
  if (!GCL) return sigmoidTableEntry(t);
  if (t < 32768) return sigmoidTableEntry(t + 16384);
  const int c = t - 49152;
  return (c < -127 ? -127 : c > 127 ? 127 : c) + 128;
}
struct Hdr {
  u32 ver = 1; // the statement's version word
  // Hdr: the statement as parsed. atW, offA, offEH: where blocks W, A, EH begin.
  size_t atW = 0;
  size_t offA = 0;
  u32 prof = 0, lt = 0, L = 0, ll = 0;
  u64 d = 0, H = 0, Gk = 0, dh = 0, f = 0, fe = 0, NE = 0, V = 0;
  u64 eps[2] = {0};
  u32 cs = 0;
  u64 theta0 = 0;
  // hf: the digit form of 13 classes of rounded matrix products (enum below): the bytes, 0 to 2,
  // committed above a remainder's low 16 bits. Seven have lines: hq hv ho hgD huD hdD hr.
  u32 hf[13] = {0}, ut = 0;
  const u32 *hCalls = nullptr;
  u32 la = 0, nmin = 0;
  u64 D = 0;
  u32 tau = 0, lmaf = 0, NAf = 0, lmad = 0, NAd = 0, lmar = 0, lmb = 0, lmq = 0, lmx = 0;
  u32 wb[12] = {0}, iota[8] = {0}; // a weight kind's width; a lookup call has 2^iota trees (Call)
  u32 lB = 0, Sx = 0, rx = 0, kFM = 0, kFD = 0, lmS = 0, NAs = 0, lmh = 0, lmo = 0, lmbA = 0, iotaA0 = 0;
  const u32 *hTexp = nullptr, *hCallsA = nullptr;
  size_t offEH = 0;
  // eh = 1: embedding and output head are proved (bit 4 of the line prof). Nothing reads eh.
  static constexpr u32 eh = 1;
  u32 MF = 0, KF = 0, toksrc = 0, lmhf = 0, NHf = 0, lmhd = 0, lmhr = 0, lmbH = 0, wbH[2] = {0}, iotaH[6] = {0},
      clipC = 0, theta = 0, cap = 0, logS = 0, digSh = 0, tabS = 0, famF = 0, yTok = 0;
  // clipC, logS, digSh, tabS, famF: the lines C, s, sh, sT, F. Position t of the range r <= t < e
  // is about token t + 1; y is token T. A position's gap is the largest logit minus its token's
  // logit, plus 1 if a token with the largest logit has a lower index: it is 0 exactly for the
  // greedy choice (among equal logits, the lowest index). Its cost is the larger of gap + 2^sh and,
  // if sT >= sh + 4, the token's negative log-probability at the exponent sT. The statement: over
  // the R positions of the range below T - 1, the sum of min(cost, C) is at most theta R, and at
  // most cap costs exceed C; if e = T, y is the argmax of position T - 1. hTok: the tree hash TH()
  // of tokens, r, e.
  u64 epsF = 0;
  const u32 *hCallsH = nullptr, *hTok = nullptr;
  const u32 *hTs = nullptr, *hSig = nullptr, *hRope = nullptr, *hKV = nullptr;
};
enum { kq_, kk_, kv_, ko_, kgD, kuD, kdD, kr_, kgS, kuS, kgR, kuR, kdM };
// True if the token index (V values) can be taken: become column bits of W (WsGrp, vf_weights_layout.h).
static bool tokenFactorTaken(u64 V, u32 lkW) {
  const int c = clog2(V);
  return c >= 1 && (u32)c <= lkW;
}
// Each unit of kind 12 or 13 (embedding, head) needs an index group of class A (`WsGrp`), each one
// taken; else WsForkOut::rec, a sub-proof's claims on W, would hold its whole factor unmasked.
static void tokenFactorUnits(const WsLay &lay) {
  for (auto &u : lay.U)
    if (u.X == 12 || u.X == 13) {
      bool any = false;
      for (auto &g : u.g)
        if (g.cls == 'A') {
          any = true;
          const int c = clog2(g.R);
          if (c < 1 || c > lay.lk || !g.aTaken()) refuse("w-list-token-factor");
        }
      if (!any) refuse("w-list-token-factor");
    }
}
// The statement's length in words. In order: base block, A, EH, hb, W; word by word:
// docs/format/FORMAT.md, "12.3 The statement", R15.5. Block A (attention sub-proofs): 11 words, two
// hashes.
static constexpr size_t statementLengthWords() { return 29u + DW + 34u + 4 * DW + 11u + 2 * DW + 25u + 2 * DW + 1u + 1u + 4u; }
static_assert(statementLengthWords() == 177, "the statement's length");
// Finds the first line of the parameter file that begins with "name " and has at least one more
// character. If the rest of that line is at most ten digits and nothing else, it sets v to their
// value and returns true. Otherwise, or if no line begins so, it returns false; it looks at no
// later line, and v may have changed. polPrescan() reads prof, lt and toksrc in this way before the
// strict scan (polScan()), which writes their words again ("internal-statement-twice" if one
// differs). whole: an entire parameter file, or only the lines after those that
// blk::read_parameters() reads.
static bool polWord(const string &whole, const char *name, u64 &v) {
  blk::Parameters frame;
  std::string fault;
  const bool framed = blk::read_parameters(whole, frame, fault);
  if (!framed && !whole.compare(0, 8, "version ")) return false; // an entire file that was rejected
  const string &pol = framed ? frame.rest : whole;
  const size_t L = strlen(name);
  for (size_t a = 0; a < pol.size();) {
    size_t e = pol.find('\n', a);
    if (e == string::npos) e = pol.size();
    if (e - a > L + 1 && !pol.compare(a, L, name) && pol[a + L] == ' ') {
      v = 0;
      if (e - a - L - 1 > 10) return false;
      for (size_t i = a + L + 1; i < e; i++) {
        if (pol[i] < '0' || pol[i] > '9') return false;
        v = 10 * v + (u64)(pol[i] - '0');
      }
      return true;
    }
    a = e + 1;
  }
  return false;
}
// The caller's input: the request (req), its n = T tokens (tok), r, e, y. The lines version, prof,
// lt, toksrc: found (hv ..), value (pv ..). `laid[i]`: word i of the statement has been written.
struct StOwn {
  bool hv = false, hp = false, hl = false, ht = false;
  u64 pv = 0, pp = 0, plt = 0, pts = 0;
  u32 r = 0, e = 0, y = 0;
  vector<u32> req, tok;
  vector<unsigned char> laid;
};
// Fills so from the request P (T tokens, then r and e) and from Q, the one word y.
// "statement-range" requires T + 2 words in P, one in Q, r < e <= T, and y = 0 if e < T: no
// position of the range is then about y. T comes only from the parameter file's line lt.
static void stFromCall(StOwn &so, const vector<u32> &P, const vector<u32> &Q, u64 T) {
  if (P.empty() || Q.empty()) refuse("statement-required");
  if (P.size() != T + 2 || Q.size() != 1) refuse("statement-range");
  so.req = P;
  so.tok.assign(P.begin(), P.begin() + (ptrdiff_t)T);
  so.r = P[T];
  so.e = P[T + 1];
  so.y = Q[0];
  if (so.r >= so.e || so.e > T || (so.e < T && so.y != 0)) refuse("statement-range");
}
// The verifier builds the statement itself and never reads one from a file. parseHeader() parses
// it, complete or partly written, and sets the global statedTarget. Hdr's pointers point into st.
static Hdr parseHeader(const vector<u32> &st) {
  Hdr h;
  if (st.size() < 3) refuse("statement-length");
  size_t p = 0;
  auto nx = [&]() {
    return st[p++];
  };
  auto nx2 = [&]() {
    u64 v = (u64)st[p] | ((u64)st[p + 1] << 32);
    p += 2;
    return v;
  };
  if (nx() != 0x4D363100u) refuse("header-magic");
  {
    const u32 ver = nx();
    // An earlier call of admitWord() (polSkeleton()) has fixed
    // the run's version word runVersionWord; this call accepts only that value.
    if (!admitWord(ver)) refuse("header-version");
    h.ver = ver;
  }
  h.prof = nx();
  if (h.prof >> 4 != 1) refuse("policy-profile");
  h.prof &= 15; // prof is 16 + the profile: verifier/docs/parameter-file.md, "1. Profile".
  if (st.size() != statementLengthWords()) refuse("statement-length");
  h.lt = nx();
  h.L = nx();
  h.ll = nx();
  h.d = nx();
  h.H = nx();
  h.Gk = nx();
  h.dh = nx();
  h.f = nx();
  h.fe = nx();
  h.NE = nx();
  h.V = nx();
  for (auto &x : h.eps) x = nx2();
  h.cs = nx();
  h.theta0 = nx2();
  // Five classes of the MoE layers share a digit tensor, and the form, with a dense class; k with
  // q. Without MoE layers the five get the form 0 here, and layout() requires 0 of the sixth (hr).
  {
    const bool experts = h.NE > 0;
    for (int i : {kq_, kv_, ko_, kgD, kuD, kdD, kr_}) h.hf[i] = nx();
    h.hf[kk_] = h.hf[kq_];
    h.hf[kgS] = h.hf[kgR] = experts ? h.hf[kgD] : 0;
    h.hf[kuS] = h.hf[kuR] = experts ? h.hf[kuD] : 0;
    h.hf[kdM] = experts ? h.hf[kdD] : 0;
  }
  h.ut = nx();
  h.hCalls = &st[p];
  p += DW;
  h.la = nx();
  h.nmin = nx();
  h.D = nx();
  h.tau = nx();
  h.lmaf = nx();
  h.NAf = nx();
  h.lmad = nx();
  h.NAd = nx();
  h.lmar = nx();
  h.lmb = nx();
  h.lmq = nx();
  h.lmx = nx();
  for (auto &x : h.wb) x = nx();
  for (auto &x : h.iota) x = nx();
  (void)nx2(); // seed_lo, seed_hi: no check reads them
  h.hTs = &st[p];
  h.hSig = h.hTs + DW;
  h.hRope = h.hTs + 2 * DW;
  h.hKV = h.hTs + 3 * DW;
  p += 4 * DW;
  h.offA = p;
  h.lB = nx();
  h.Sx = nx();
  h.rx = nx();
  h.kFM = nx(); // the line kFM: hashed like every line; no test and no formula reads it
  h.kFD = nx();
  h.lmS = nx();
  h.NAs = nx();
  h.lmh = nx();
  h.lmo = nx();
  h.lmbA = nx();
  h.iotaA0 = nx();
  h.hTexp = &st[p];
  p += DW;
  h.hCallsA = &st[p];
  p += DW;
  h.offEH = p;
  h.MF = nx();
  h.KF = nx();
  h.epsF = nx2();
  h.toksrc = nx();
  h.lmhf = nx();
  h.NHf = nx();
  h.lmhd = nx();
  h.lmhr = nx();
  h.lmbH = nx();
  for (auto &x : h.wbH) x = nx();
  for (auto &x : h.iotaH) x = nx();
  h.clipC = nx();
  h.theta = nx();
  h.cap = nx();
  h.logS = nx();
  h.digSh = nx();
  h.tabS = nx();
  h.famF = nx();
  h.hCallsH = &st[p];
  p += DW;
  h.hTok = &st[p];
  p += DW;
  h.yTok = nx();
  // Block HD: the one word hb, the value of the parameter file's line sec.
  statedTarget.at = p;
  statedTarget.hb = nx();
  // Block W: w, b, n, m of the parameter file's line `table`. It is the statement's last block.
  h.atW = p;
  p += 4;
  if (p != st.size()) refuse("internal-header-length");
  return h;
}
// A weight kind (a family of weight matrices: qk, v, o ..) of the model. idx: its number,
// 0 to 11 (Hdr::wb's index). L, segs: length and number (with limbs) of its runs. Only idx is read.
struct KindD {
  int idx = 0;
  string name;
  u64 L = 0, segs = 0;
};
// The layout of a layer sub-proof, as layout() derives it. T tokens; NGr groups of Lam layers each.
struct Lay {
  Hdr h;
  u64 T = 0, Lam = 0, NGr = 0, Pd = 0, Pf = 0, S2 = 0, NW = 0, NN = 0, NI = 0, dq = 0, dkv = 0, nqk = 0;
  int lPf = 0, lfe = 0, lPd = 0, ls = 0, ldh = 0;
  vector<Ten> ten;
  u64 feat[4] = {0}, filled[4] = {0}, kcol[4] = {0}, mrow[4] = {0}, parts[4] = {0};
  u64 xbFilled = 0, kxb = 0, mxb = 0;
  TenAt xb; // the residual stream's tensor in each matrix xb
  vector<KindD> kinds;
  vector<CallD> calls;
  vector<u64> tauFC, tauFL, tauFS;
  u64 LamM = 0;
  int lNE = 0;
  u64 kbm = 0;
  // The layer group's lookup forest (its calls' product trees): shape and pub; `cl` is empty.
  HForest fo;
  const Ten &T_(const string &n) const {
    for (auto &t : ten)
      if (t.name == n) return t;
    refuse("internal-tensor-" + n);
  }
};
typedef vector<vector<string>> CallLines; // From the parameter file: name, kind, segments.
// Rejects with "plan-groups-base" above 28 layer groups (2 + 2 NGr > 59), a limit of the format:
// beyond it the prover's random streams for a sub-proof and for the opening would share numbers.
static void groupsBaseHeld(u64 NGr) {
  if (2 + 2 * NGr > 59) refuse("plan-groups-base");
}
// Tests the statement's parameters and derives the layout; reads globals that polScan() sets.
static Lay layout(const Hdr &h, const CallLines &pc) {
  Lay y;
  y.h = h;
  const u64 d = h.d, f = h.f, fe = h.fe;
  if (h.prof != 4) refuse("policy-profile");
  const bool p3 = h.NE > 0; // the layout has MoE (expert) layers
  y.NW = y.NN = 2;
  if (h.lt < 4 || h.lt > 16 || h.ll < 1 || h.ll > 6 || !d || !f || d > (1u << 20) || f > (1u << 20) ||
      (p3 && ((fe & (fe - 1)) || f != 2 * fe || (h.ll == 1 && fe == 1))))
    refuse("shape-words"); // f need not be a power of two. MoE layers (p3) require f = 2 fe, and
                           // a second product layer of ll - 1 + lfe + lt variables, more than lt.
  y.T = 1ull << h.lt;
  y.Lam = 1ull << h.ll;
  if (!h.L || h.L % y.Lam) refuse("shape-L");
  y.NGr = h.L / y.Lam;
  // A request matrix's id (TMat::aId, vf_smallness_matrices.h) is 256 gid + its index in sub-proof gid's list, and
  // gid is at most 1 + 2 NGr: with this bound 256 gid is below 2^24. No hash takes an id: the
  // verifier only compares the ids in its own plans ("internal-nmlab"). The next test is stricter.
  if (2 + 2 * y.NGr >= (1ull << 16)) refuse("shape-groups");
  groupsBaseHeld(y.NGr);
  y.Pd = 1ull << clog2(d);
  y.lPd = clog2(y.Pd);
  y.Pf = 1ull << clog2(f);
  y.lPf = clog2(y.Pf);
  if (p3) y.lfe = clog2(fe);
  // la = lt (runs of one feature) is rejected too: that case has not been analysed.
  if (h.la <= h.lt || h.la > 16) refuse("shape-la");
  y.ls = (int)(h.la - h.lt);
  y.S2 = 1ull << y.ls;
  if (d % y.S2 || f % y.S2 || (p3 && fe % y.S2)) refuse("shape-la");
  // eps < 2^59: with four limbs a norm's sum of squares is below 2^59 (vf_norm_digit_rows.h), with eps below
  // 2^60 < p = 2^64 - 2^32 + 1. With five, runStage() requires eps < 2^48 ("no-wrap-ledger-eps").
  // From 2^63 on, the map to the field would make eps negative. runStage() tests M and K, from KL.
  for (int n = 0; n < 2; n++)
    if (h.eps[n] < 1 || h.eps[n] >= (1ull << 59)) refuse("policy-norm");
  if (!h.H || !h.Gk || !h.dh || (h.dh & (h.dh - 1)) || h.dh < 2 || h.H * h.dh != d || h.H + h.Gk > 64)
    refuse("shape-heads");
  y.dq = h.H * h.dh;
  y.dkv = h.Gk * h.dh;
  y.nqk = y.dq + y.dkv;
  y.ldh = clog2(h.dh);
  if (y.dkv % y.S2 || y.nqk % y.S2) refuse("shape-la");
  if (h.cs != 12) refuse("policy-cs");
  // Rejects a digit form above 2, and any form but 0 for a class that is absent.
  {
    vector<int> present = {kgD, kuD, kdD, kq_, kk_, kv_, ko_};
    if (p3)
      for (int i : {kr_, kgS, kuS, kgR, kuR, kdM}) present.push_back(i);
    // The shifts, whose ranges depend on the form, are per layer, from KL; runStage() tests them.
    for (int i = 0; i < 13; i++) {
      bool pr = false;
      for (int j : present)
        if (i == j) pr = true;
      if (!pr) {
        if (h.hf[i]) refuse("header-absent-class");
        continue;
      }
      if (h.hf[i] > 2) refuse("policy-forms");
    }
    if (h.ut != 1) refuse("policy-ut"); // u has a range segment of its own, UT
  }
  if (h.D < 2 || h.D > 4096 || (h.D & (h.D - 1))) refuse("shape-D");
  const u64 Lam = y.Lam;
  y.NI = y.NN * Lam;
  u64 off[4] = {0, 0, 0, 0};
  const u64 NE = h.NE, LamM = Lam / 2;
  y.LamM = LamM;
  if (p3) {
    if (NE < 2 || NE > 4096 || (NE & (NE - 1))) refuse("shape-NE");
    y.lNE = clog2(NE);
  }
  // nl = 0: a digit tensor of form 0 is absent and leaves no gap in the list.
  auto add = [&](const char *n, int mat, int nl, u64 cnt, u64 J) {
    if (!nl) return;
    y.ten.push_back({n, mat, nl, cnt, J, off[mat]});
    off[mat] += nl * cnt * J;
  };
  if (GCL && !p3) refuse("policy-gcl");
  if (GCL) {
    add("g0", 0, 1, Lam, f); // g0 and gc take the place of the two limbs of gi = 256 gc + g0
    add("gc", 0, 1, Lam, f);
  } else
    add("gi", 0, 2, Lam, f);
  add("s", 0, 2, Lam, f);
  add("u", 0, 2, Lam, f);
  add("rg", 0, (int)h.hf[kgD], Lam, f);
  add("ru", 0, (int)h.hf[kuD], Lam, f);
  add("xs", 0, residualStreamLimbs, y.NW * Lam - 1, d);
  add("ro", 0, (int)h.hf[ko_], Lam, d);
  add("rd", 0, (int)h.hf[kdD], Lam, d);
  add("aw", 0, 2, Lam, d);
  add("ra", 0, 2, Lam, d);
  add("mw", 0, 2, Lam, d);
  add("rm", 0, 2, Lam, d);
  add("qkw", 0, 2, Lam, y.nqk);
  add("rqk", 0, (int)h.hf[kq_], Lam, y.nqk);
  add("vw", 0, 2, Lam, y.dkv);
  add("rv", 0, (int)h.hf[kv_], Lam, y.dkv);
  if (GCL) add("gh", 0, 2, Lam, f);
  add("a", 1, 2, Lam, d);
  add("m", 1, 2, Lam, d);
  if (p3) {
    add("xr", 1, 2, LamM, d);
    add("lr", 2, 2, LamM, NE);
    add("rl", 2, (int)h.hf[kr_], LamM, NE);
    add("sel", 2, 1, LamM, NE);
    add("sc", 2, 3, LamM, 1);
  }
  add("row", 2, 1, y.NN * Lam, (u64)ledNR());
  if (RCL && !p3) refuse("policy-rcl");
  if (RCL) {
    add("ly", 2, 2, LamM, NE);
    add("l3", 2, 1, LamM, NE);
  }
  add("q", 3, 2, Lam, y.dq);
  add("kx", 3, 2, Lam, y.dkv);
  add("v", 3, 2, Lam, y.dkv);
  add("o", 3, 2, Lam, y.dq);
  const u32 lm[4] = {h.lmaf, h.lmad, h.lmar, h.lmq};
  const u64 na[4] = {h.NAf, h.NAd, 1, 1};
  if (h.lmq < h.la) refuse("shape-la");
  for (int m = 0; m < 4; m++) {
    if (lm[m] < h.lt || lm[m] > 24 || !na[m] || na[m] > 64) refuse("shape-request-matrix");
    y.feat[m] = off[m];
    y.mrow[m] = 1ull << lm[m];
    y.parts[m] = na[m];
    u64 unit = na[m] * (y.mrow[m] / y.T);
    y.filled[m] = (off[m] + unit - 1) / unit * unit;
    y.kcol[m] = y.filled[m] / unit;
  }
  if (h.lmaf < h.la) refuse("shape-la");
  if (h.lmx < h.la || h.lmx > 24) refuse("shape-request-matrix");
  y.mxb = 1ull << h.lmx;
  {
    u64 unit = y.mxb / y.T;
    // xb: docs/format/11-2-feed-forward-segment.md, "The arrays that the segment commits to".
    const u64 NLx = (u64)residualStreamLimbs;
    y.xbFilled = (NLx * d + unit - 1) / unit * unit;
    y.kxb = y.xbFilled / unit;
    y.xb = {0, NLx, 1, d};
    tensorsFill({{"x", y.xb}}, NLx * d, y.S2, "stream");
    TenList qa;
    for (auto &t : y.ten)
      if (t.mat == 3) qa.push_back({t.name, t.at()});
    tensorsFill(qa, y.feat[3], y.S2, "qa");
  }
  y.tauFC.resize(65536);
  y.tauFL.resize(65536);
  for (int t = 0; t < 65536; t++) {
    int c = t - 32768;
    c = c < -CLAMP ? -CLAMP : c > CLAMP ? CLAMP : c;
    y.tauFC[t] = (u64)t + ((u64)(c + CLAMP + 1) << 28);
    y.tauFL[t] = (u64)t + ((u64)(gateTableEntry(t) + 4096) << 28);
  }
  if (p3) {
    y.tauFS.resize(65536);
    // With gcl the clamp's entries (t >= 32768) add a flag, 2^40, that no SIG entry reaches. That a
    // key of CG keeps the flag follows from F and G0 (the condition no-wrap-guard-flag-gate-clamp).
    for (int t = 0; t < 65536; t++)
      y.tauFS[t] = GCL && t >= 32768 ? (u64)t + ((u64)sigmoidClampTableEntry(t) << 28) + (1ull << 40) : (u64)t + ((u64)sigmoidClampTableEntry(t) << 17);
  }
  // Lookup segments present: docs/format/11-2-feed-forward-segment.md, "The calls' segments".
  struct SD {
    string name, kind;
    u64 n = 0;
    bool moe = false;
  };
  vector<SD> sd;
  auto sg = [&](const char *n, const char *k, u64 len, bool moe = false) {
    sd.push_back({n, k, len, moe});
  };
  auto hi = [&](const char *n, u32 form, u64 len, bool moe = false) {
    if (form) sd.push_back({n, form == 1 ? "R8" : "R16", len, moe});
  };
  sg("G", "R16", f);
  sg("U", "R16", f);
  sg("UT", "R16", f);
  sg("QK", "R16", y.nqk);
  hi("QKH", h.hf[kq_], y.nqk);
  sg("N1", "R16", d);
  sg("N1H", "R16", d);
  sg("N2", "R16", d);
  sg("N2H", "R16", d);
  sg("O", "R16", d);
  sg("D", "R16", d);
  sg("V", "R16", y.dkv);
  if (p3) {
    sg("XR", "R16", d, true);
    sg("LR", "R16", NE, true);
    sg("LT", "R16", NE, true);
    sg("DD", "R16", NE, true);
  }
  hi("GH", h.hf[kgD], f);
  hi("UH", h.hf[kuD], f);
  hi("OH", h.hf[ko_], d);
  hi("DH", h.hf[kdD], d);
  hi("VH", h.hf[kv_], y.dkv);
  sg("ROW", "R8", (u64)ledNR() * y.NN);
  if (p3) {
    hi("LH", h.hf[kr_], NE, true);
    sg("SEL", "R8", NE, true);
    if (RCL) sg("L3", "R8", NE, true);
  }
  if (GCL) sg("G0", "R8", f);
  sg("CN1", "FC", d);
  sg("CN2", "FC", d);
  sg("CQK", "FC", y.nqk);
  sg("CV", "FC", y.dkv);
  if (RCL) sg("CLR", "FC", NE, true);
  if (p3) sg("MR", "FS", 1, true);
  if (GCL) sg("CG", "FS", f);
  sg("F", "FL", f);
  // The call list is the parameter file's. (Both built-in lists lack UT and are rejected below.)
  CallLines cl = pc;
  if (cl.empty()) {
    if (!p3)
      cl = {{"R16a", "R16", "G", "U"},
            {"R16b", "R16", "N1", "N1H", "N2", "N2H", "O", "D", "V"},
            {"R16c", "R16", "QK", "QKH"},
            {"R8a", "R8", "GH", "UH"},
            {"R8b", "R8", "OH", "DH", "VH", "ROW"},
            {"FC", "FC", "CN1", "CN2", "CQK", "CV"},
            {"FL", "FL", "F"}};
    else
      cl = {{"R16a", "R16", "G", "U"},
            {"R16b", "R16", "N1", "N1H", "N2", "N2H", "O", "D", "V"},
            {"R16c", "R16", "QK", "QKH", "XR", "LR", "LT", "DD"},
            {"R8", "R8", "GH", "UH", "OH", "DH", "VH", "ROW", "LH", "SEL"},
            {"FC", "FC", "CN1", "CN2", "CQK", "CV"},
            {"FS", "FS", "MR"},
            {"FL", "FL", "F"}};
  }
  /* The structural test of a layer sub-proof's call list, made before any word of the proof is
   * used: at most 8 calls (the statement has eight words iota_q), and each lookup segment that is
   * present appears in exactly one call, of its kind. In full: docs/format/FORMAT.md, "13.8 Every
   * rejection by name", row policy-call-list. So each lookup segment's keys are tied to the
   * committed cells in exactly one call: the parameter file can neither leave out nor repeat a
   * lookup. */
  if (cl.size() > 8u) refuse("policy-call-list");
  /* seen[j]: a call line has named the lookup segment sd[j]. */
  vector<char> seen(sd.size(), 0);
  for (auto &ln : cl) {
    if (ln.size() < 3) refuse("policy-call-list");
    CallD cd;
    cd.c.name = ln[0];
    cd.kind = ln[1];
    for (auto &o : y.calls)
      if (o.c.name == ln[0]) refuse("policy-call-list");
    // The layer sub-proof stores four of its points as Z_V, Z_a, Z_b and Z_E (vf_layers_check.inc), and
    // no call may be named V, a, b or E. This program forms no point's name from a call's name.
    for (const char *own : {"V", "a", "b", "E"})
      if (ln[0] == own) refuse("policy-call-list");
    for (size_t i = 2; i < ln.size(); i++) {
      size_t j = 0;
      while (j < sd.size() && sd[j].name != ln[i]) j++;
      if (j == sd.size() || seen[j] || sd[j].kind != ln[1]) refuse("policy-call-list");
      seen[j] = 1;
      Seg s;
      s.name = sd[j].name;
      s.n = sd[j].n;
      s.moe = sd[j].moe;
      cd.segs.push_back(s);
    }
    y.calls.push_back(cd);
  }
  for (char c : seen)
    if (!c) refuse("policy-call-list");
  y.kinds.push_back({0, "qk", 64, 2 * Lam * d * h.dh});
  y.kinds.push_back({1, "v", y.dkv, 2 * Lam * d});
  y.kinds.push_back({2, "o", d, 2 * Lam * y.dq});
  // segs: 2 limbs * the group's nd dense layers * inner width d.
  {
    const u64 nd = p3 ? LamM : Lam;
    y.kinds.push_back({3, "g", f, 2 * nd * d});
    y.kinds.push_back({4, "u", f, 2 * nd * d});
  }
  if (p3) {
    y.kinds.push_back({5, "gs", fe, 2 * LamM * d});
    y.kinds.push_back({6, "us", fe, 2 * LamM * d});
    y.kinds.push_back({7, "gr", fe, 2 * LamM * NE * d});
    y.kinds.push_back({8, "ur", fe, 2 * LamM * NE * d});
  }
  y.kinds.push_back({9, "dn", d, p3 ? 2 * (LamM * f + LamM * fe) : 2 * Lam * f});
  if (p3) {
    y.kinds.push_back({10, "dnr", d, 2 * LamM * NE * fe});
    y.kinds.push_back({11, "r", NE, 2 * LamM * d});
  }
  /* Request matrix b (the bits of the lookups'
   * multiplicities) has 2^lmb rows. */
  if (h.lmb > 24) refuse("shape-lmb");
  /* This test protects the program itself: the minimum nmin sizes vectors of
   * 2^(nmin - ll - lt) slots before any test of n_c, so a value of 40 or 2^31
   * would cause an allocation failure or a hang instead of a rejection. */
  if (h.nmin > 28) refuse("shape-words");
  for (size_t q = 0; q < y.calls.size(); q++) {
    CallD &cd = y.calls[q];
    int fl = (int)h.nmin - (int)h.ll - (int)h.lt;
    place(cd, fl < 0 ? 0 : fl);
    Call &c = cd.c;
    c.n = (int)h.ll + cd.lJ + (int)h.lt;
    c.iota = (int)h.iota[q];
    c.kappa = 5;
    c.k = cd.kind == "R8" ? 8 : 16;
    if (cd.kind == "FC" || cd.kind == "FL" || cd.kind == "FS") {
      c.fn = 1;
      c.t0 = 32768;
    }
    /* A parameter from 2^31 on becomes a negative iota here. It must never be used as a shift
     * count, which would be undefined behaviour. */
    if (c.iota < 0) refuse("shape-call-" + c.name);
    if (c.n < 10 || c.n < (int)h.nmin || c.n > 44 || c.kappa != 5) refuse("shape-call-" + c.name);
    if (c.n < 10) refuse("hid-call-floor");
    /* Call::zk has n_c coordinates: ll (layer), lJ_c (column), lt (token). vf_layers_check.inc reads the
     * last la of them and the last ls of the lJ_c, which must therefore exist. */
    if (c.n < (int)h.la || cd.lJ < y.ls) refuse("shape-call-" + c.name);
  }
  /* The calls' product trees form one lookup forest (HForest, vf_lookup_argument.h), 2^dT keys each. The
   * tests, in order: `forestShape`; dT >= la, because the calls share the last la coordinates of
   * their points; matrix b's shape (`forestPlaneCols`); each line iota_q must equal n_q - dT. */
  {
    vector<Call *> cl;
    for (auto &cd : y.calls) cl.push_back(&cd.c);
    y.fo = forestShape(cl, LIDB);
    y.fo.cl.clear();
    const HForest &fo = y.fo;
    if (fo.dT < (int)h.la) refuse("forest-depth-G");
    y.kbm = forestPlaneCols(fo, (int)h.lmb);
    for (size_t q = 0; q < y.calls.size(); q++)
      if (y.calls[q].c.iota != fo.iota[q]) refuse("shape-call-" + y.calls[q].c.name);
    /* pub[i]: tree i's padding keys. Key (lam, j, t) has the index (lam J_c + j) T + t,
     * and dT >= la > lt, so the T keys of a column (lam, j) are in one tree of their call, number
     * floor((lam J_c + j) / 2^(dT - lt)). */
    y.fo.pub.assign(fo.NT, 0);
    for (size_t q = 0; q < y.calls.size(); q++) {
      const CallD &cd = y.calls[q];
      for (u64 lam = 0; lam < Lam; lam++)
        for (u64 j = 0; j < cd.J; j++)
          if (cd.padding(lam, j)) {
            const u64 b = (lam * cd.J + j) >> (fo.dT - (int)h.lt);
            if (b >> fo.iota[q]) refuse("internal-forest-pub");
            y.fo.pub[(u64)fo.first[q] + b] += y.T;
          }
    }
    forestPubAsk(fo, cl, y.fo.pub); // here too, on public numbers, not only when a proof is read
  }
  for (size_t q = y.calls.size(); q < 8; q++)
    if (h.iota[q]) refuse("header-absent-call");
  if (y.NW != 2u || y.NN != y.NW || y.NI != y.NN * y.Lam) refuse("internal-layout-words");
  return y;
}
static void setTau(Lay &dst, const Lay &src) {
  for (auto &c : dst.calls)
    c.c.tau = c.kind == "FC" ? &src.tauFC : c.kind == "FL" ? &src.tauFL : c.kind == "FS" ? &src.tauFS : nullptr;
}
/* Forms h_par: the configuration digest of the three configuration files (parameter file, layout
   file, KL.bin). It is printed (h_par=) and compared with nothing. Printed word by word, as isk.txt's
   line ranges has it, its hex digits differ from those of the line configuration=, which prints the
   same digest byte by byte. */
static blk::Digest hParOf(const string &polB, const vector<u32> &KL, const string *layoutB) {
  rejectLongTextFile(polB.size());
  // A KL.bin of fewer than DW words is the caller's fault: TH, which a stage calls on it, takes no
  // shorter list ("internal-th").
  if (KL.size() < DW) refuse("KL-length");
  if (layoutB) rejectLongTextFile(layoutB->size());
  return blk::configuration_digest({polB, layoutB ? *layoutB : string(), string((const char *)KL.data(), 4 * KL.size())});
}
/* The results of the parameter file's strict scan. st: the verifier's statement; h: st as parsed.
   The scan also sets globals (LIDB, NMINH, gateTableInputFractionBits, SMAX, RCL, GCL,
   residualStreamLimbs, POW, POWS, POWC) and reads RCL and GCL to see whether their lines have
   come. So a process may scan only once: a second scan rejects a valid file with rcl or gcl. */
struct PolScan {
  /* The leading lines, version to table: verifier/docs/parameter-file.md, "Syntax". frame.rest: the
     lines after them. */
  blk::Parameters frame;
  vector<u32> st;
  StOwn so;
  Hdr h;
  CallLines polCalls, polCallsH, polCallsA;
  bool lidbSeen = false, nminHSeen = false, egSeen = false, smaxSeen = false, xlSeen = false;
  /* The optional lines of the parameter file, tsilu0 to texp<DW - 1>, hold the digests of the public
     tables, DW words each, in the order SILU, FS/SIG, the RoPE table, the softmax table. A model
     with experts (NE > 0) gives all four (TBLW lines); a dense model (NE 0) gives only the three it
     uses, SILU, RoPE and softmax, and leaves out the FS/SIG line (3 * DW lines). tblCount[k] counts
     the lines read for table k: after the scan each is 0 or DW. tblAt is their sum. */
  u32 tblPol[4][DW] = {};
  size_t tblCount[4] = {};
  size_t tblAt = 0;
  vector<u32> polWordsH;
};
// Writes word i of the statement.
static void polLayW(PolScan &PS, size_t i, u32 v) {
  vector<u32> &st = PS.st;
  StOwn &so = PS.so;
  if (i >= st.size() || so.laid.size() != st.size()) refuse("internal-statement-cover");
  // No word of the verifier's statement may have two sources that disagree.
  if (so.laid[i] && st[i] != v) refuse("internal-statement-twice");
  st[i] = v;
  so.laid[i] = 1;
}
// Reads version, prof, lt and toksrc before the scan and tests all but lt, which the caller tests.
static void polPrescan(PolScan &PS, const string &polB) {
  StOwn &so = PS.so;
  {
    // The parser's fault text has the form "<name>[: <which line>] (line <number>)"; <name> becomes
    // the rejection name.
    std::string fault;
    if (!blk::read_parameters(polB, PS.frame, fault)) {
      printf("parameter file: %s\n", fault.c_str());
      const string name = fault.substr(0, std::min(fault.find(": "), fault.find(" (line ")));
      if (!name.rfind("version-", 0)) checkVersionLine(true, PS.frame.version);
      if (name.rfind("parameter-", 0)) refuse("internal-parameter-name");
      refuse(name + " (policy)");
    }
    so.hv = true, so.pv = PS.frame.version;
    so.hp = polWord(PS.frame.rest, "prof", so.pp);
    so.hl = polWord(PS.frame.rest, "lt", so.plt);
    so.ht = polWord(PS.frame.rest, "toksrc", so.pts);
    checkVersionLine(so.hv, so.pv); // redundant: the parser accepts only this program's version word
    // A parameter file whose tokens do not come from the caller (toksrc other than 1), or whose
    // line prof is not in 16 to 31, is rejected here: the verifier checks only a request that the
    // caller provides, and it forms the statement itself.
    if (!so.hp || (so.pp >> 4) != 1 || !so.ht || so.pts != 1) refuse("statement-required");
  }
}
// Sizes the statement; writes words 0-3, toksrc and (if withY) so.y.
static void polSkeleton(PolScan &PS, bool withY) {
  vector<u32> &st = PS.st;
  StOwn &so = PS.so;
  Hdr &h = PS.h;
  auto layW = [&](size_t i, u32 v) {
    polLayW(PS, i, v);
  };
  {
    if (!admitWord((u32)so.pv)) refuse("header-version");
    st.assign(statementLengthWords(), 0);
    so.laid.assign(st.size(), 0);
    // The strict scan later writes prof and lt again, from their lines of the parameter file; layW
    // ends the run without a verdict if a value differs from the one written here.
    layW(0, 0x4D363100u);
    layW(1, (u32)so.pv);
    layW(2, (u32)so.pp);
    layW(3, (u32)so.plt);
    h = parseHeader(st);
    {
      const size_t iT = (size_t)(h.hTok - st.data());
      layW(h.offEH + 4, 1 /* toksrc */);
      if (withY) layW(iT + DW, so.y);
    }
    h = parseHeader(st);
  }
}
/* The strict scan of the lines after the leading ones: it accepts one order and one spelling. It
   fills PS. */
static void polScan(PolScan &PS, const string &polB) {
  Hdr &h = PS.h;
  CallLines &polCalls = PS.polCalls, &polCallsH = PS.polCallsH, &polCallsA = PS.polCallsA;
  bool &lidbSeen = PS.lidbSeen, &nminHSeen = PS.nminHSeen, &egSeen = PS.egSeen, &smaxSeen = PS.smaxSeen,
       &xlSeen = PS.xlSeen;
  u32(&tblPol)[4][DW] = PS.tblPol;
  size_t(&tblCount)[4] = PS.tblCount;
  size_t &tblAt = PS.tblAt;
  vector<u32> &polWordsH = PS.polWordsH;
  auto layW = [&](size_t i, u32 v) {
    polLayW(PS, i, v);
  };
  // After the leading lines the file has a value line ("name value") for each word of the statement
  // from prof to F that is not a hash, in the statement's order; then the optional lines; then the
  // call lines "call <name> <kind> <SEG> ...". runStage() computes the statement's hashes.
  {
    vector<string> nm = {"magic",   "version", "prof",    "lt",      "L",  "ll",        "d",
                         "H",       "G",       "dh",      "f",       "fe", "NE",        "V",
                         "eps1_lo", "eps1_hi", "eps2_lo", "eps2_hi", "cs", "theta0_lo", "theta0_hi"};
    size_t gapAt = ~(size_t)0;
    for (const char *s : {"hq", "hv", "ho", "hgD", "huD", "hdD", "hr", "ut"}) nm.push_back(s);
    gapAt = nm.size(); // here the statement has the DW words of the call lines' hash
    for (const char *s : {"la", "nmin", "D", "tau", "lmaf", "NAf", "lmad", "NAd", "lmar", "lmb", "lmq", "lmx"})
      nm.push_back(s);
    for (const char *k : {"qk", "v", "o", "g", "u", "gs", "us", "gr", "ur", "dn", "dnr", "r"})
      nm.push_back(string("wb_") + k);
    for (int c = 0; c < 8; c++) nm.push_back("iota_" + std::to_string(c));
    for (const char *s : {"seed_lo", "seed_hi"}) nm.push_back(s);
    const size_t nBase = nm.size(); // the base block's 63 words that are not hashes
    for (const char *s : {"lB", "S", "rx", "kFM", "kFD", "lmS", "NAs", "lmh", "lmo", "lmbA", "iotaA_0"}) nm.push_back(s);
    const size_t nA = nm.size(); // the 11 value lines of block A
    for (const char *s : {"MF",   "KF",     "epsF_lo", "epsF_hi", "toksrc",  "lmhf",    "NHf",     "lmhd",    "lmhr",
                          "lmbH", "wb_emb", "wb_head", "iotaH_0", "iotaH_1", "iotaH_2", "iotaH_3", "iotaH_4", "iotaH_5",
                          "C",    "theta",  "cap",     "s",       "sh",      "sT",      "F"})
      nm.push_back(s);
    const size_t nE = nm.size(); // the 25 value lines of block EH
    nm.push_back("hb");          // the one word of block HD
    for (const char *s : {"lkW", "fb", "fln", "fnm", "lNmax"})
      nm.push_back(s); // The four words of block W, and lNmax, which has no header word.
    /* Of these names only prof to F are lines here. `polSkeleton` has written the words magic and
       version. The words of hb and of block W follow from the leading lines: frameWords, below,
       writes them once F has been read. lNmax has neither a line nor a word. */
    LineIterator pls(PS.frame.rest);
    char name[64], extra[8];
    unsigned long long val;
    char line[1024];
    size_t i = 2;
    bool tblPrev = false; // the line before this one was one of the tables' lines
    bool llSeen = false;  // the value lines up to F have been read, and `frameWords` has run
    auto frameWords = [&]() {
      const blk::Parameters &f = PS.frame;
      for (u64 wide : {f.table_data, f.table_random})
        if (wide >> 32) refuse("parameter-table-shape (policy)"); // stored in 32 bits here
      // One cap for every line "work": a stage's record and the parent that reads it hold no more.
      for (const auto &kind : f.work)
        for (const auto bits : kind)
          if (bits > WORK_BITS_MOST) refuse("policy-work-most");
      const u32 depth = blk::ceil_log2(f.slots);
      if (depth > 8) refuse("policy-leafd");
      layW(statedTarget.at, f.sec);
      layW(h.atW, f.block_log), layW(h.atW + 1, f.table_rows_log), layW(h.atW + 2, f.table_log_n), layW(h.atW + 3, (u32)f.table_random);
      POW = (int)f.work[3][0], POWS = (int)f.work[3][1], POWC = (int)f.work[3][2];
      i = nm.size();
      llSeen = true;
    };
    // sscanf also accepts tabs, a sign and leading zeros, but a value line has one valid spelling
    // (wordLineForm(), vf_common.h): equal values give equal bytes, which the commitment ID covers.
    // oneSpelling() follows each call of scanned(); the call lines' parsing is strict by itself.
    auto oneSpelling = [&]() {
      if (!wordLineForm(line)) refuse("policy-line-form");
    };
    // Scans a value line: true if it read a name, a number and nothing after them. The C standard
    // leaves open what %llu makes of a number that unsigned long long cannot hold, so a line with
    // over 19 digits in a row, which no valid spelling has, is rejected before sscanf sees it.
    auto scanned = [&]() {
      if (digitsBeyondScan(line)) refuse("policy-line-form");
      return sscanf(line, "%63s %llu %7s", name, &val, extra) == 2;
    };
    for (int lk; (lk = pls.next(line, sizeof line)) != 0;) {
      if (lk == 2) refuse("policy-line-long");
      if (lk == 3) refuse("policy-zero-byte");
      const bool afterTbl = tblPrev;
      tblPrev = false;
      if (i < nm.size()) {
        if (!scanned() || val > 0xFFFFFFFFull) refuse("policy-line-" + std::to_string(i));
        if (nm[i] != name) refuse("policy-name-" + nm[i]);
        oneSpelling();
        layW(i < nBase ? i + (i >= gapAt ? DW : 0)
             : i < nA  ? h.offA + (i - nBase)
                       : h.offEH + (i - nA),
             (u32)val); // into the statement
        if (++i == nE) frameWords();
        continue;
      }
      // Not lines of the format: verifier/docs/parameter-file.md, "Syntax".
      if (!strncmp(line, "clt ", 4) || !strncmp(line, "clns ", 5) || !strncmp(line, "cms ", 4))
        refuse("policy-cert-words");
      /* Optional line "lidb": verifier/docs/parameter-file.md, "9. Optional lines"; on defaults,
       * "Syntax". */
      if (!strncmp(line, "lidb ", 5)) {
        if (!llSeen || lidbSeen || nminHSeen || egSeen || smaxSeen || RCL || GCL || xlSeen || tblAt ||
            !polCalls.empty() || !polCallsA.empty() || !polCallsH.empty() || !scanned())
          refuse("policy-lidb");
        oneSpelling();
        if (val <= (u64)LIDB_SEALED) refuse("policy-lidb");
        if (val > (u64)LIDB_MAX) refuse("plan-identity-cap");
        LIDB = (int)val;
        lidbSeen = true;
        continue;
      }
      /* Optional line "nminH", its position and value: verifier/docs/parameter-file.md,
       * "9. Optional lines". That it is above nmin is tested after the scan (`polLayout`), where
       * the statement's words are read, and before anything is sized by it. */
      if (!strncmp(line, "nminH ", 6)) {
        if (!llSeen || !lidbSeen || nminHSeen || egSeen || smaxSeen || RCL || GCL || xlSeen || tblAt ||
            !polCalls.empty() || !polCallsA.empty() || !polCallsH.empty() || !scanned())
          refuse("policy-nminH");
        oneSpelling();
        if (val < 2 || val > (u64)NMINH_MAX) refuse("policy-nminH");
        NMINH = (int)val;
        nminHSeen = true;
        continue;
      }
      /* Optional lines "eg" and "smax", which set what gateTableEntry() reads:
       * verifier/docs/parameter-file.md, "9. Optional lines"; on defaults, "Syntax". */
      if (!strncmp(line, "eg ", 3) || !strncmp(line, "smax ", 5)) {
        const bool isEg = line[0] == 'e';
        if (!llSeen || smaxSeen || (isEg && egSeen) || RCL || GCL || xlSeen || tblAt ||
            !polCalls.empty() || !polCallsA.empty() || !polCallsH.empty() || !scanned())
          refuse("policy-gate-table");
        oneSpelling();
        if (isEg ? val < 4 || val > 7 : val != 2047 && val != 1023 && val != 511 && val != 255)
          refuse("policy-gate-table");
        (isEg ? gateTableInputFractionBits : SMAX) = (int)val;
        (isEg ? egSeen : smaxSeen) = true;
        continue;
      }
      // Optional line "rcl 1": verifier/docs/parameter-file.md, "9. Optional lines".
      if (!strncmp(line, "rcl ", 4)) {
        if (!llSeen || RCL || GCL || xlSeen || tblAt || !polCalls.empty() || !polCallsA.empty() ||
            !polCallsH.empty() || !scanned() || val != 1)
          refuse("policy-rcl");
        oneSpelling();
        RCL = 1;
        continue;
      }
      // Optional line "gcl 1": verifier/docs/parameter-file.md, "9. Optional lines".
      if (!strncmp(line, "gcl ", 4)) {
        if (!llSeen || GCL || xlSeen || tblAt || !polCalls.empty() || !polCallsA.empty() ||
            !polCallsH.empty() || !scanned() || val != 1)
          refuse("policy-gcl");
        oneSpelling();
        GCL = 1;
        continue;
      }
      /* Optional line "xl 5": verifier/docs/parameter-file.md, "9. Optional lines" and
       * "The optional lines on the model's arithmetic". */
      if (!strncmp(line, "xl ", 3)) {
        if (!llSeen || xlSeen || tblAt || !polCalls.empty() || !polCallsA.empty() ||
            !polCallsH.empty() || !scanned())
          refuse("policy-xl");
        oneSpelling();
        if (val != 5) refuse("policy-xl");
        residualStreamLimbs = 5;
        xlSeen = true;
        continue;
      }
      /* Optional lines with the digests of the public tables, DW words each:
       * verifier/docs/parameter-file.md, "9. Optional lines" and "Digests of the public tables".
       * tblHeld() compares them. Which set of tables the model uses (four with experts, three for a
       * dense model) is tested after the scan (`polLayout`), once the word NE is parsed from the
       * statement: here h is still the parse of the skeleton (`polSkeleton`), in which NE is not
       * written. The scan only checks that the lines are well formed and well ordered. */
      if (!strncmp(line, "tsilu", 5) || !strncmp(line, "tsig", 4) || !strncmp(line, "trope", 5) ||
          !strncmp(line, "texp", 4)) {
        static const char *const TB[4] = {"tsilu", "tsig", "trope", "texp"};
        if (!llSeen || tblAt >= TBLW || (tblAt && !afterTbl) || !polCalls.empty() ||
            !polCallsA.empty() || !polCallsH.empty() || !scanned() || val > 0xFFFFFFFFull)
          refuse("policy-tables");
        oneSpelling();
        // The table index k (0 tsilu, 1 tsig, 2 trope, 3 texp) and the word index w, e.g. "trope3"
        // is table 2, word 3. A table's words arrive in order 0 .. DW - 1, the tables in ascending
        // index order, and the FS/SIG table (index 1) may be absent (a dense model). So the lines
        // cannot be reordered, interleaved or repeated; polLayout() checks the set against NE.
        int k = -1;
        size_t pfx = 0;
        for (int t = 0; t < 4; t++) {
          const size_t len = strlen(TB[t]);
          if (!strncmp(name, TB[t], len) && name[len] >= '0' && name[len] <= '9') {
            k = t;
            pfx = len;
            break;
          }
        }
        if (k < 0) refuse("policy-tables");
        size_t w = 0;
        const char *p = name + pfx;
        if (p[0] == '0' && p[1] != 0) refuse("policy-tables"); // one spelling: no leading zero
        for (; *p; p++) {
          if (*p < '0' || *p > '9') refuse("policy-tables");
          w = 10 * w + (size_t)(*p - '0');
          if (w >= DW) refuse("policy-tables");
        }
        if (w != tblCount[k]) refuse("policy-tables"); // contiguous within the table
        if (w == 0)
          for (int t = k + 1; t < 4; t++)
            if (tblCount[t]) refuse("policy-tables"); // a later table has not started yet
        tblPol[k][w] = (u32)val;
        tblCount[k]++;
        tblAt++;
        tblPrev = true;
        continue;
      }
      string s = line;
      if (s.empty() || s.back() != '\n') refuse("policy-call-line");
      s.pop_back();
      vector<string> tok;
      size_t a = 0;
      // Form of a call line: verifier/docs/parameter-file.md, "Syntax".
      while (true) {
        size_t b = s.find(' ', a);
        string t = s.substr(a, b == string::npos ? b : b - a);
        if (t.empty()) refuse("policy-call-line");
        // A call line's tokens, the call's name among them, are printed at the end of rejection
        // names. So each byte must be one of [A-Za-z0-9._-]: no control byte or blank gets printed.
        for (unsigned char ch_ : t)
          if (!((ch_ >= 'A' && ch_ <= 'Z') || (ch_ >= 'a' && ch_ <= 'z') || (ch_ >= '0' && ch_ <= '9') || ch_ == '.' ||
                ch_ == '_' || ch_ == '-'))
            refuse("policy-call-line");
        tok.push_back(t);
        if (b == string::npos) break;
        a = b + 1;
      }
      // The file's writer chose this printed name: 8 characters are too few for an ID or sentence.
      if (tok.size() > 1 && tok[1].size() > 8) refuse("policy-call-line");
      const int kindL = tok[0] == "call" ? 0 : tok[0] == "callA" ? 1 : tok[0] == "callH" ? 2 : -1;
      if (tok.size() < 4 || kindL < 0 || (kindL == 0 && (!polCallsA.empty() || !polCallsH.empty())) ||
          (kindL == 1 && !polCallsH.empty()))
        refuse("policy-call-line"); // order: call, callA, callH
      (kindL == 0   ? polCalls
       : kindL == 1 ? polCallsA
                    : polCallsH)
          .push_back(vector<string>(tok.begin() + 1, tok.end()));
    }
    /* polWordsH: the file's digest, then the values of the value lines in frame.rest, in order.
     * Nothing reads polWordsH. */
    {
      for (u32 w : blk::parameter_file_digest(polB).w) polWordsH.push_back(w);
      LineIterator p2_(PS.frame.rest);
      char l2[1024];
      for (int lk; (lk = p2_.next(l2, sizeof l2)) != 0;) {
        if (!strncmp(l2, "call", 4)) continue;
        char n2[64];
        unsigned long long v2;
        if (sscanf(l2, "%63s %llu", n2, &v2) != 2) refuse("policy-line-form");
        polWordsH.push_back((u32)v2);
      }
    }
    if (i != nm.size()) refuse("policy-length");
    // A table's digest is DW words, all or none; a part of one is never valid. Which tables must be
    // present is checked in polLayout(), where NE is known.
    for (int t = 0; t < 4; t++)
      if (tblCount[t] != 0 && tblCount[t] != DW) refuse("policy-tables");
  }
}
/* Tests after the scan, on the statement's words together and on the lines "call"; returns the
   layout. The caller runs `setTau` on the layout it keeps. */
static Lay polLayout(PolScan &PS) {
  vector<u32> &st = PS.st;
  StOwn &so = PS.so;
  Hdr &h = PS.h;
  CallLines &polCalls = PS.polCalls;
  // Reads back from the statement the words that polSkeleton() wrote (version, prof, lt, toksrc)
  // and y (still 0 in runStage()'s call). No run can fail this test as the code is written: each
  // word is written from the value it is compared with, and a differing line of the parameter file
  // has ended the run as "internal-statement-twice". It guards against a later edit of that code.
  h = parseHeader(st); // strict parse of the verifier's own statement, with every parameter written
  if (h.ver != (u32)so.pv || h.prof + 16 != (u32)so.pp || h.lt != (u32)so.plt || h.toksrc != 1 || h.yTok != so.y)
    refuse("internal-statement-prescan");
  if (PS.nminHSeen && (u32)NMINH <= h.nmin)
    refuse("policy-nminH"); // nmin itself is the default, which no line may state
  if (!tauServed(h.tau)) refuse("hid-tau");
  TAU = (int)h.tau; // combinations in a sub-proof's approximate range check (`tail` in names)
  Lay y = layout(h, polCalls);
  if (statedTarget.hb < 1 || statedTarget.hb > 4096) refuse("hid-hb");
  // hb (the target in bits, line sec) must not exceed TAU, a range check's count of combinations.
  if ((u32)TAU < statedTarget.hb) refuse("policy-tau-hb");
  for (auto &c : y.calls)
    if (c.c.kappa != 5) refuse("hid-planes-" + c.c.name);
  // The tables' digest lines, if any are given, must be exactly the set the model uses, so that
  // every table it uses is compared: all four (SILU, FS/SIG, RoPE, softmax) with experts, and the
  // three without the FS/SIG table (index 1) for a dense model. A part of this set is already
  // refused in the scan (each table is 0 or DW words there); here NE decides which set is owed.
  if (PS.tblAt) {
    const bool ok = h.NE ? (PS.tblCount[0] == DW && PS.tblCount[1] == DW && PS.tblCount[2] == DW && PS.tblCount[3] == DW)
                         : (PS.tblCount[0] == DW && PS.tblCount[1] == 0 && PS.tblCount[2] == DW && PS.tblCount[3] == DW);
    if (!ok) refuse("policy-tables");
  }
  return y;
}
// Requested bits: an integer from 0 to 1,000,000. A program that links this unit passes a double.
static void askedWhole(double asked) {
  if (!(asked >= 0) || asked > 1e6 || asked != floor(asked)) refuse("internal-asked-bits");
}
static bool askedBelowBar(double asked) {
  return !(asked >= BAR);
}
/* How `runStage` passes the sub-proofs' shapes to `blk::configure` (blocks/frame/configure.h),
   which takes them from a plain function of the public files. buildLinksH (vf_segments.inc) sets
   stageShapesMade directly before that call and clears it directly after: nothing else may come
   between. Setting it twice or reading it unset ends the run ("internal-shapes-order"). The shapes
   must follow from the three files only: see SealedCallInputs. The pointer has file scope because
   `runStage` goes on working on the plans from which the shapes are made: a function that built
   them from the files would have to leave them in such an object, or they would be built twice and
   one public quantity would have two descriptions. `runStage` is not reentrant, and it runs once in
   a process (see the top of this file): to check many proofs at once, start as many processes. */
static const blk::Shapes *stageShapesMade = nullptr;
static blk::Shapes stageShapesOf(const std::vector<std::string> &, const blk::Parameters &) {
  if (!stageShapesMade) refuse("internal-shapes-order");
  return *stageShapesMade;
}
// What a call is about.
struct CallInputs {
  vector<u32> request, y;          // the words of the caller's request file and y file
  bool oddStatement = false;       // one of the two has a length that is no multiple of four bytes
  const string *proofB = nullptr;  // the proof file's bytes; nullptr exactly if StageInputs::planOnly
  u32 entry = 0;                   // the leaf table's number; the proof file's header must name it
  bool nextTokenAsked = false;     // -next-token: no verification unless r = T - 1 and e = T
  PolicyNamed policy;            // -policy: the condition policy-named replaces exact-match
  const string *keyB = nullptr, *entryB = nullptr; // bytes of key.bin, leaf file; null if not given
  string idAsked;                  // the caller's commitment ID (-id), lower-case hex; empty: none
};
/* The envelope of a CallInputs: `runStage` cannot read what a call is about before its plans are
   fixed, since the numbers, types and lengths of all that a proof holds must follow from the three
   configuration files. `open` needs a blk::Configured, which only `blk::configure` constructs: an
   earlier read would not compile. No type sees a global set from these values or an earlier call of
   `configure`: a test that is not in this repository compares differing requests' plans. */
class SealedCallInputs {
public:
  explicit SealedCallInputs(CallInputs ask) : ask_(std::move(ask)) {}
  SealedCallInputs(const SealedCallInputs &) = delete;
  SealedCallInputs &operator=(const SealedCallInputs &) = delete;
  const CallInputs &open(const blk::Configured &c) const {
    if (!c.refused().empty()) refuse("internal-sealed-request"); // rejected files open nothing
    return ask_;
  }

private:
  CallInputs ask_;
};
/* The inputs of `runStage`. Its callers have read every file: it opens none and gets no file's
   name. `then`, if set, replaces the stage (cp_configured.h). */
struct StageInputs {
  const CpConf *conf = nullptr;          // the three configuration files' bytes
  const SealedCallInputs *sealed = nullptr;   // what the call is about
  bool planOnly = false;                 // no proof file: true except in the stage request
  const std::function<int(const blk::Configured &)> *then = nullptr; // if set: no `sealed`
  int oneTime = 0;                       // 0: stage request; 1: key; 2: leaf; 3: planning call "id"
  double asked = 0;
  int nthreads = 1;
  bool quiet = false, shapesAsked = false, fieldsAsked = false;
  bool arraysAsked = false;        // only lists segments' arrays, for prover/tools/pack_layout.py
};
/* Runs one stage of a call: makes the sub-proofs' plans, which `blk::configure` needs, then runs
   the stage key (`blk::check_key`), leaf (`blk::check_entry_if_tested`) or request
   (`blk::check_proof_if_tested`). Last line, status:
   "STAGE PASSED: <stage>" 3 (WITH WORK: 12), NOT-A-VERIFICATION 14, "NO: <name>" 1, PLAN ONLY 4. */
static int runStage(const StageInputs &in) {
  // The bytes of the parameter file and the layout file: every parse and hash of them reads these.
  const string &polB = in.conf->pol;
  const string *const layoutP = in.conf->layout();
  const vector<u32> &KL = in.conf->KL;
  const int nthreads = in.nthreads;
  const bool quiet = in.quiet, planOnly = in.planOnly, shapesAsked = in.shapesAsked, fieldsAsked = in.fieldsAsked;
  if (!in.sealed && !in.then) refuse("internal-cp-cut");
  /* Start-up check: the source digest VF_SOURCE_DIGEST is 64 lower-case hex digits. The length test can
     fail only for a literal with a zero byte; a static_assert covers the rest. That the value is
     the SHA-256 of the source list rests on verifier/build.sh: no test here can show it. */
  if (strlen(VF_SOURCE_DIGEST) != 64) refuse("internal-build-word");
  for (const char *c = VF_SOURCE_DIGEST; *c; c++)
    if (!((*c >= '0' && *c <= '9') || (*c >= 'a' && *c <= 'f'))) refuse("internal-build-word");
  // The requested bits, an integer. acctOfH(), further down, rejects ("security-bits") unless the
  // parameter file's target is at least that and every round of the plan meets it, work counted.
  const double minBits = in.asked;
  // A run that requests fewer bits than BAR makes every test and ends NOT-A-VERIFICATION at best.
  const bool belowBar = askedBelowBar(minBits);
  double t00 = now(), tl = t00;
  auto lap = [&]() {
    double n = now(), r = n - tl;
    tl = n;
    return r;
  };
  // The references below are short names for PS's fields.
  PolScan PS;
  vector<u32> &st = PS.st;
  StOwn &so = PS.so;
  Hdr &h = PS.h;
  CallLines &polCallsH = PS.polCallsH, &polCallsA = PS.polCallsA;
  bool &egSeen = PS.egSeen, &smaxSeen = PS.smaxSeen;
  u32(&tblPol)[4][DW] = PS.tblPol;
  size_t(&tblCount)[4] = PS.tblCount;
  size_t &tblAt = PS.tblAt;
  /* The public tables (see tblPol) are computed with the platform's floating-point library. If the
     parameter file gives a table's digest, a table with another digest ends the run before a word
     of a proof is used: what is proved must not depend on the last bit of a library's result. Each
     table whose DW digest words were given is compared; a dense model gives three (not the FS/SIG
     table), an experts model four. polLayout() has already refused any other set, so tblCount[k] is
     DW exactly when table k is owed and present. */
  auto tblHeld = [&](int k, const u32 *mine, const char *refusal) {
    if (tblCount[k] == DW && memcmp(mine, tblPol[k], 4 * DW)) refuse(refusal);
  };
  /* Set after the tests that identify format and commitment: version word, commitment ID (if the
     caller named one), leaf file's table (form). Unset: "internal-naming-first" (`runLinksH`). */
  bool namingDone = false;
  std::function<string()> statedW;
  auto layW = [&](size_t i, u32 v) {
    polLayW(PS, i, v);
  };
  // Writes a hash that the verifier computed itself into the statement.
  auto ownW = [&](const u32 *at, const u32 *mine) {
    const size_t i = (size_t)(at - st.data());
    for (size_t j = 0; j < DW; j++) layW(i + j, mine[j]);
  };
  auto zeroW = [&](const u32 *at, size_t k) {
    const size_t i = (size_t)(at - st.data());
    for (size_t j = 0; j < k; j++) layW(i + j, 0);
  };
  polPrescan(PS, polB);
  {
    if (!so.hl) refuse("policy-name-lt");
    // lt outside 4 .. 16 is rejected here, before anything of 2^lt words is built.
    if (so.plt < 4 || so.plt > 16) refuse("shape-words");
    polSkeleton(PS, false); // y's word is written where the request is opened
  }
  // A redundant test: the run continues only if admitWord(), called by
  // polSkeleton(), has set the run's version word to an accepted value.
  if (!runVersionWordFixed || !versionWordAdmitted(runVersionWord)) refuse("internal-version-word");
  polScan(PS, polB);
  // polLayout() builds the layout of a layer sub-proof and tests the list of the lines `call`.
  // setTau() stores addresses of members of y in y.calls, so it is called here, on the object
  // that stays.
  Lay y = polLayout(PS);
  setTau(y, y);

  // Rebuilds the lines `call` (name, kind, segment names) from the layout and hashes them.
  // Format: docs/format/FORMAT.md, "The digests".
  if (h.hCalls) {
    string s;
    for (auto &c : y.calls) {
      s += "call " + c.c.name + " " + c.kind;
      for (auto &g : c.segs) s += " " + g.name;
      s += "\n";
    }
    vector<u32> w((s.size() + 3) / 4, 0);
    for (size_t i = 0; i < s.size(); i++) w[i / 4] |= (u32)(unsigned char)s[i] << (8 * (i % 4));
    u32 hh[DW];
    earlier_hash_h_digest(hh, 73, w.data(), w.size());
    ownW(h.hCalls, hh);
  }
  const bool p3 = h.NE > 0;
  const u64 NE = h.NE, LamM = y.LamM;
  const u64 d = h.d, f = h.f, fe = h.fe, T = y.T, Lam = y.Lam, NGr = y.NGr, Pd = y.Pd, S2 = y.S2, D = h.D, NW = y.NW,
            NN = y.NN, NI = y.NI, dq = y.dq, nqk = y.nqk, dh = h.dh;
  const u64 Pf = y.Pf;
  const int lt = (int)h.lt, ll = (int)h.ll, lPf = y.lPf, lfe = y.lfe, lPd = y.lPd, ls = y.ls, la = (int)h.la,
            ldh = y.ldh;
  // KL: the words of KL.bin, the scale file. Format: docs/format/FORMAT.md,
  // "13.4 What every stage does first: the public files", step 2.
  const u64 KW_ = 13; // words per layer record
  const u64 nMoe = p3 ? h.L / 2 : 0;
  {
    if (KL.size() != KW_ * (u64)h.L + 3 * nMoe) refuse("KL-length");
    u32 hh[DW];
    TH(hh, KL.data(), KL.size());
    ownW(h.hKV, hh);
  }
  enum { Lkq, Lkk, Lkv, Lko, Lkg, Lku, Lkd, LM1, LK1, LM2, LK2 };
  auto kl = [&](u64 l, int i) {
    return KL[KW_ * l + i];
  };
  // klM returns word i (0: kr, 1: kgR, 2: kuR) of the MoE layer l (l is odd).
  auto klM = [&](u64 l, int i) {
    return KL[KW_ * (u64)h.L + 3 * (l / 2) + i];
  };
  auto klRp = [&](u64 l) {
    return KL[KW_ * l + KW_ - 1];
  };
  auto klSh = [&](u64 l) {
    return KL[KW_ * l + 11];
  };
  const int fQK = (int)h.hf[kq_], fV = (int)h.hf[kv_], fO = (int)h.hf[ko_], fG = (int)h.hf[kgD], fU = (int)h.hf[kuD],
            fD = (int)h.hf[kdD], fR = (int)h.hf[kr_]; // one digit form per class
  auto isMoeL = [&](u64 l) {
    return p3 && l % 2 == 1;
  };
  // Words k..: right shifts; M, K: a norm's exponents; rp: the RoPE flag.
  for (u64 l = 0; l < h.L; l++) {
    auto rng = [&](u32 k, int form) {
      if (k < 1 || k > (form == 0 ? 16u : form == 1 ? 24u : 32u)) refuse("KL-shift-range");
    };
    rng(kl(l, Lkg), fG);
    rng(kl(l, Lku), fU);
    rng(kl(l, Lkd), fD);
    rng(kl(l, Lkq), fQK);
    rng(kl(l, Lkk), fQK);
    rng(kl(l, Lkv), fV);
    rng(kl(l, Lko), fO);
    if (kl(l, Lkq) != kl(l, Lkk) && (kl(l, Lkq) < 16 || kl(l, Lkk) < 16)) refuse("KL-kq-kk");
    for (int n = 0; n < 2; n++) {
      const u32 M = kl(l, n ? LM2 : LM1), K = kl(l, n ? LK2 : LK1);
      if (M < 24 || M > 47 || K + 14 < M || K < 17 || K > 32) refuse("KL-norm");
    }
    if (klRp(l) > 1) refuse("KL-rp");
    if (isMoeL(l)) {
      rng(klM(l, 0), fR);
      rng(klM(l, 1), fG);
      rng(klM(l, 2), fU);
    }
  }
  {
    vector<u32> w(65536);
    for (int t = 0; t < 65536; t++) w[t] = (u32)(gateTableEntry(t) + 4096);
    u32 hh[DW];
    earlier_hash_h_digest(hh, 72, w.data(), w.size());
    ownW(h.hTs, hh);
    tblHeld(0, hh, "table-digest-silu");
    for (int i = 0; i < (int)DW; i++)
      if (!p3 && h.hSig[i]) refuse("header-absent-hash");
    if (!p3) zeroW(h.hSig, DW);
    if (p3) {
      for (int t = 0; t < 65536; t++) w[t] = (u32)sigmoidClampTableEntry(t);
      earlier_hash_h_digest(hh, 72, w.data(), w.size());
      ownW(h.hSig, hh);
      tblHeld(1, hh, "table-digest-sigmoid");
    }
  }
  // q9 evaluates, in u128, the layer sub-proofs' no-overflow conditions, under which equations
  // modulo p are read as equations of integers. Ty: the bound on a committed cell that they assume.
  auto q9 = [&](const u128 Ty) {
    // Capitals (G, U, LR ..) name lookup segments, small letters values of the integer model.
    // BWw = 2^(WEIGHT_WIDTH_BITS - 1) = 2^12: the bound on a weight of kind 0 to 11 that these
    // conditions assume; the key file's range proof is meant to show it. BC = 2^11 bounds the
    // clamped cells a, m and v. o has no clamp, and no lookup of a layer sub-proof constrains it:
    // the attention sub-proof is meant to show that o is a rounded mean of values of v, which keeps
    // v's bound; the site "o" relies on that. BWw is a constant: "policy-weight-kind" requires
    // wb_X = WEIGHT_WIDTH_BITS = 13 of every present kind. Ty2, Ty3 bound values of two, three
    // base-256 limbs.
    const u128 P2 = (u128)1 << 60, Ty2 = Ty * 257, Ty3 = Ty * 65793, BWw = WEIGHT_BOUND, BC = CLAMPED_CELL_BOUND, acc = (u128)d * BC * BWw;
    auto need = [&](bool ok, const string &n) {
      if (!ok) refuse("no-wrap-" + n);
    };
    auto pw = [](u32 k) {
      return (u128)1 << k;
    };
    /* A norm's digit row (vf_norm_digit_rows.h) is meant to show |x| <= sigma for a stream value x, and
       sigma < 2^lsg. A norm's product: |x R| <= R sigma <= 2^M + sigma / 2. The quotient at O and
       at D is the difference of two stream values: below 2^(lsg + 1), in a span of 2^(lsg + 2). */
    const u32 lsg = residualStreamLimbs == 5 ? 39 : 30;
    auto site = [&](const string &n, u32 k, int form, u128 A, u128 SP, u128 Qb) {
      const u32 t = k < 16 ? 16 - k : 0; // below, kh = max(k, 16)
      // A site: a rounding right shift by k. W- assumes bounds: A on the value, Qb on the quotient.
      // U-: quotients in the span SP differ modulo p (SP 2^kh <= P2 < p): the relation fixes one.
      need((SP << (k > 16 ? k : 16)) <= P2, "U-" + n);
      need(((A + pw(k - 1) + (Qb << k)) << t) + (form == 1   ? pw(16) * (Ty + 128)
                                                 : form == 2 ? pw(16) * (Ty2 + 32896)
                                                             : 0) <
               P2,
           "W-" + n);
    };
    for (u64 l = 0; l < h.L; l++) {
      const bool mo = isMoeL(l);
      // With gcl the gating value's quotient is g0 + 256 gh. G0 confines g0 to a byte and CG
      // confines gh to 15 bits, CG only together with F, which reads the same cell gc (see the
      // flag 2^40, below). The no-overflow conditions of these lookups do not depend on this site.
      const u128 gQ = GCL ? 128 + pw(22) : pw(15);
      site("gate", kl(l, Lkg), fG, acc, GCL ? 2 * gQ + 1 : pw(16), gQ);
      if (mo) site("gate", klM(l, 1), fG, acc, GCL ? 2 * gQ + 1 : pw(16), gQ);
      for (u32 k : mo ? vector<u32>{kl(l, Lku), klM(l, 2)} : vector<u32>{kl(l, Lku)})
        site("up", k, fU, acc, pw(16), pw(15));
      // The bound on u that the lookup segment UT gives, used by the layer's "down" site.
      const u128 Bu = pw(LBU);
      for (int n = 0; n < 2; n++)
        site("norm", kl(l, n ? LK2 : LK1), 2, pw(kl(l, n ? LM2 : LM1)) + pw(lsg - 1), pw(16), pw(15));
      site("down", kl(l, Lkd), fD, (u128)f * (SMAXTOP + 1) * Bu * BWw, pw(lsg + 2), pw(lsg + 1));
      for (u32 k : {kl(l, Lkq), kl(l, Lkk)}) site("qk", k, fQK, 2 * pw(h.cs) * acc, pw(16), pw(15));
      site("o", kl(l, Lko), fO, (u128)dq * BC * BWw, pw(lsg + 2), pw(lsg + 1));
      site("v", kl(l, Lkv), fV, acc, pw(16), pw(15));
      // With rcl the quotient of LR is 8 ly + l3, and only Ty2 bounds ly.
      if (mo) site("lr", klM(l, 0), fR, acc, RCL ? 16 * Ty2 + 15 : pw(15), RCL ? 8 * Ty2 + 7 : pw(14));
      // CLR's key packs lr - l3 at radix 2^25, below Ty2, the bound on ly. LR fixes ly to the true
      // quotient's high part: that, plus the key's 2^15 and a row's 2^16, must be below 2^25.
      if (mo && RCL)
        need(((acc + pw(klM(l, 0) - 1)) >> (klM(l, 0) + 3)) + 1 + pw(15) + pw(16) < pw(25), "guard-radix-clamp");
      // Keys 32 l3 and (ly + 2^15) + 2^25 (lr - l3) + 2^39: no overflow by the cell bounds alone.
      if (mo && RCL) need(32 * Ty < P2 && (Ty2 + pw(15)) + pw(25) * (Ty2 + Ty) + pw(39) < P2, "clamp-keys");
    }
    if (p3) {
      need(((2 * Ty2 + 1) << 16) <= P2, "U-xr");
      need(pw(16) * BC + pw(15) + pw(16) * Ty2 < P2, "W-xr");
      need(pw(15) + pw(14) + pw(17) * Ty3 < P2 / 2, "guard-radix-FS");
      // CG's key is (gh + 49152) + 2^28 (gc + 128) + 2^40. The whole key must not overflow, and
      // gh + 49152 minus the number t < 2^16 of a table row must stay below 2^28 in magnitude,
      // so that key = row forces gh + 49152 = t for a row of the clamp's half of table FS.
      if (GCL)
        need((Ty2 + 49152) + ((Ty + 128) << 28) + pw(40) < P2 / 2 && Ty2 + 49152 + 65536 < pw(28),
             "guard-radix-gate-clamp");
      // The rows t < 32768 of table FS are the sigmoid's, t + 2^17 SIG with SIG <= 2^16,
      // without the flag 2^40. CG by itself does not exclude that its key is one of them,
      // with gc not the clamp of gh. F's key reads the same cell gc at every feature of
      // every layer, and with g0 a byte (G0) and the condition guard-radix it confines gc
      // to -128 .. 128. So gc + 128 >= 0 cannot cancel the flag, and CG's smallest key,
      // 2^40 + 49152 - Ty2, must be above the largest row of the sigmoid's half.
      if (GCL) need(pw(40) + 49152 > Ty2 + 32767 + (pw(16) << 17), "guard-flag-gate-clamp");
      // Nor is MR's key, (mr + 2^14) + 2^17 sc with gcl, a row of the clamp's half, t + a multiple
      // of 2^28 with 2^15 <= t < 2^16: by LT, SEL and the one-hot claim mr is one lr_e, in
      // -2^14 .. 2^14 - 1, so the key is below 2^15 modulo 2^17 (both as integers: guard-radix-FS).
      static_assert((1 << 14) + (1 << 14) <= 32768 && 65536 <= (1 << 17) && 28 >= 17,
                    "modulo 2^17 a key of MR is below 2^15, and a row of the clamp's half of table FS is not");
      // Only cell bounds limit mr = sum_e sel_e lr_e (keys of DD, MR) and the one-hot sum_e sel_e.
      need((u128)NE * Ty * Ty2 + Ty2 + (u128)NE * Ty < P2, "MR");
      need((u128)NE * Ty + 1 < P2, "one-hot");
    }
    need(pw(16) * (Ty2 + 32896) < P2, "linear");
    need((Ty2 + 32768) + ((Ty2 + 4096) << 28) < P2 / 2 && Ty2 + 32768 + 65536 < pw(28),
         "guard-radix"); // 4096 is the offset 2^12 of FL, the largest offset of a function table
    // A convolution sum has at most residualStreamLimbs products of two limbs per feature.
    need((u128)residualStreamLimbs * d * Ty * Ty < pw(53), "ledger");
    // With five limbs, eps is in the equation for the square root's low 48 bits (vf_norm_digit_rows.h, E1).
    if (residualStreamLimbs == 5) {
      for (int n = 0; n < 2; n++) need(h.eps[n] < (1ull << 48), "ledger-eps");
      need(h.epsF < (1ull << 48), "ledger-eps");
    }
  };
  // Ty = 2 beta_z = 2^18 (LBZ, vf_common.h): the bound on |cell| that these conditions assume.
  const u128 TyReq = (u128)1 << (1 + LBZ);
  // Every matrix with a cell of such a condition is in its sub-proof's approximate range check,
  // which is meant to show that bound (lookups R10z, R8d: vf_smallness_test_members.inc). No parameter changes that.
  q9(TyReq);
  // The field ms= of the line "result:": the commitment ID, in hex digits.
  char commitmentIdHex[HEXW + 1] = "";
  // *layoutP holds the bytes that buildLinksH() parses further down.
  const blk::Digest hParDigest = hParOf(polB, KL, layoutP);
  {
    // Two computations of one hash: TH(KL) here and in the statement.
    u32 kl[DW];
    TH(kl, KL.data(), KL.size());
    if (memcmp(kl, h.hKV, 4 * DW)) refuse("internal-par-digests");
  }
  for (int k = 0; k < 12; k++) {
    const KindD *kp = nullptr;
    for (auto &x : y.kinds)
      if (x.idx == k) kp = &x;
    if (!kp) {
      if (h.wb[k]) refuse("header-absent-kind");
      continue;
    }
    if (h.wb[k] != WEIGHT_WIDTH_BITS) refuse("policy-weight-kind"); // wb is 13 for every kind.
  }
  // Kinds 12 (emb, the embedding) and 13 (head): docs/format/11-5-weights-in-the-table.md, "Units".
  const u64 Vv = h.V;
  {
    if (Vv < 2 || Vv > (1u << 24)) refuse("shape-V");
    if (h.wbH[0] < 2 || h.wbH[0] > 31 || h.wbH[1] != WEIGHT_WIDTH_BITS) refuse("policy-weight-kind");
  }
  // The width is a constant of the format, which the verifier enforces ("policy-weight-kind"
  // above: kind 0 to 11 and the head): wb_X = 13 for every kind X except emb. A width fitted to
  // the weights would be public and would reveal the smallest and the largest weight of a class.
  if (h.wbH[0] != 21) refuse("policy-weight-width-emb"); // The width parameter of emb is 21.

  askedWhole(minBits);
  // W's layout (vf_weights_layout.h), from the shape and lkW: 2^lkW columns (last number of line `table`).
  const u32 lkW = PS.frame.block_log;
  WsLay wsLay;
  {
    WsShape s;
    s.D = D;
    s.L = h.L;
    s.d = d;
    s.dh = dh;
    s.dkv = y.dkv;
    s.dq = dq;
    s.f = f;
    s.fe = fe;
    s.NE = NE; // build() tests NE if and only if some layer is MoE.
    s.V = h.V;
    s.eh = 1;
    wsSetMoe(s, p3 ? 1 : 0);
    // The vocabulary index must fit W's column bits: units 12, 13 (embedding, head) put it there.
    if ((u32)clog2(h.V) > lkW) refuse("hid-vocab-cols");
    if (!tokenFactorTaken(h.V, lkW)) refuse("w-list-token-factor");
    wsLay.build(s, lkW); // it tests lkW and the shape's numbers itself
    // The token-factor test runs unit by unit on the layout as built; if the
    // layout has no unit of kind 12 or 13, the test has nothing to check.
    tokenFactorUnits(wsLay);
  }
  // A theta0 of zero is rejected as "shape-theta0" before any RoPE table
  // entry is computed: pow(0, -x) is infinite, 0 times infinity is not a
  // number, and the cast of that value to an integer is undefined.
  if (!h.theta0) refuse("shape-theta0");
  // A RoPE (rotary position embedding) table entry for layer l, token position t, pair i.
  // Format: docs/format/11-1-machinery.md, "11.1 The machinery", public tables.
  auto ropeC = [&](u64 l, u64 t, u64 i, bool sine) -> i64 {
    // l = ~0 is used for the hash below.
    if (l != ~0ull && !klRp(l)) return sine ? 0 : (i64)1 << h.cs;
    double ang = (double)t * pow((double)h.theta0, -2.0 * (double)i / (double)dh);
    return (i64)round((double)(1u << h.cs) * (sine ? sin(ang) : cos(ang)));
  };
  // The statement's RoPE hash. Format: docs/format/FORMAT.md, "The digests".
  {
    vector<u32> w;
    for (int sine = 0; sine < 2; sine++)
      for (u64 t = 0; t < T; t++)
        for (u64 i = 0; i < dh / 2; i++) w.push_back((u32)(ropeC(~0ull, t, i, sine) + 4096));
    u32 hh[DW];
    earlier_hash_h_digest(hh, 72, w.data(), w.size());
    ownW(h.hRope, hh);
    tblHeld(2, hh, "table-digest-rope");
  }
  // tails[g]: the matrices in layer sub-proof g's range check, in the order that the format fixes.
  vector<Tail> tails(NGr);
  for (u64 g = 0; g < NGr; g++) {
    Tail &t = tails[g];
    const u64 gid = 1 + g;
    auto req = [&](const string &n, u64 m, u64 k) {
      TMat M;
      M.name = n;
      M.m = m;
      M.k = k;
      M.aId = gid * 256 + t.mats.size();
      t.mats.push_back(M);
    };
    for (u64 p = 0; p < y.parts[0]; p++) req("af" + std::to_string(p), y.mrow[0], y.kcol[0]);
    for (u64 p = 0; p < y.parts[1]; p++) req("ad" + std::to_string(p), y.mrow[1], y.kcol[1]);
    req("ar", y.mrow[2], y.kcol[2]);
    req("b", 1ull << h.lmb, y.kbm);
    req("qa", y.mrow[3], y.kcol[3]);
    // The list's order, and where xbI comes from: docs/format/11-2-feed-forward-segment.md,
    // "The arrays that the segment commits to".
    req("xbO", y.mxb, y.kxb);
    req("xbI", y.mxb, y.kxb);
  }
  const size_t iAF = 0, iAD = y.parts[0], iAR = iAD + y.parts[1], iB = iAR + 1, iQA = iB + 1, iXO = iQA + 1,
               iXI = iXO + 1;
  for (u64 g = 1; g < NGr; g++) {
    Tail &t = tails[g];
    t.mats[iXI].kind = 1;
    t.mats[iXI].foreign = 0;
    t.mats[iXI].aId = tails[g - 1].mats[iXO].aId;
  }
  // The layout of the head's sub-proof.
  struct HL {
    u64 JV = 0, JVa = 0, oE = 0, feat = 0, filled = 0, k = 0, m = 0, oAw = 0, oRa = 0, oD1 = 0, oD2 = 0, fillD = 0,
        kD = 0, mD = 0, fillR = 0, kR = 0, mR = 0, kb = 0;
    vector<CallD> calls;
    vector<u64> tauFE;
    // The head's lookup forest. It points into HLy.calls.
    HForest fs;
  } HLy;
  vector<u32> tokv;
  /* hOn[t] = 1 in the request's range, r <= t < e; hWin[t] is the checked token
   * (printed: "covered token"): the request's token t + 1, or y at T - 1. hOnC[t] = 1 at the
   * range's hRC positions other than T - 1. Only they may have a logit gap (see Hdr), a hidden
   * number; y must be the argmax of its position. */
  vector<u32> hWin;
  vector<char> hOn, hOnC;
  u64 hRC = 0;
  {
    HL &L = HLy;
    if (h.MF < 24 || h.MF > 47 || h.KF + 14 < h.MF) refuse("policy-norm-F");
    if (h.KF < 17 || h.KF > 32 || h.epsF < 1 || h.epsF >= (1ull << 59)) refuse("policy-norm-F");
    if (h.toksrc != 1) refuse("policy-toksrc");
    // The clip C, the bound theta on the average cost, and cap, the limit on the number of
    // positions above C. A clipped cost is at most C, and there are at most T positions: larger
    // values say no more. cap minus that number has two byte digits, so cap is also below 2^16.
    if (h.theta > h.clipC || h.cap > T || h.cap >= (1u << 16)) refuse("policy-gap-words");
    /* The middle digit of each margin (a logit difference: see h9) has the weight 2^sh and is
     * looked up in table FE, exponentials at the scale 2^F; a position's values add up to at most
     * 2^F. FE exists for q = sT - sh from 4 to 10, where the bound is on negative log-probabilities
     * at the exponent sT, and for q = -4, a step table (2^F, 2, 1, 1 ..): there the sum's equation
     * says only that no middle digit is 0, and the bound is on the logit gaps, each plus 2^sh. s:
     * an integer logit is a real logit times 2^s; s is printed and is in no equation. */
    if (h.famF != 24 || h.logS > 63) refuse("policy-gap-words");
    if (h.digSh < 7 || h.digSh > 8 || (h.tabS + 4 != h.digSh && (h.tabS < h.digSh + 4 || h.tabS > h.digSh + 10)))
      refuse("policy-gap-words");
    /* A position has V logit differences, one per vocabulary entry v. Tensor d1 has two features
     * per entry and d2 one; the lookup segments HD0, HD1, HD2 one column. */
    L.JV = (Vv + S2 - 1) / S2 * S2;
    L.oAw = 0;
    L.oRa = 2 * d;
    L.oD1 = 4 * d;
    L.oD2 = 4 * d + 2 * L.JV;
    // Table values ex: docs/format/11-4-output-head.md, "Where a cell is", "The call list". Their
    // lookup segments: HE0 has the entries below JVa, HE1 the rest.
    L.oE = 4 * d + 3 * L.JV;
    L.feat = 4 * d + (3 + EXC) * L.JV;
    L.JVa = 1ull << (clog2(L.JV) - 1);
    // 2 V <= 2^F: under every table, a position with no middle digit 0 can have a sum <= 2^F.
    if (L.JVa % S2 || 2 * Vv > (1ull << h.famF)) refuse("policy-gap-words");
    if (h.lmhf < h.la || h.lmhf > 24 || !h.NHf || h.NHf > 64 || h.lmhd < h.lt || h.lmhd > 24 || h.lmhr < h.lt ||
        h.lmhr > 24 || h.lmbH > 24)
      refuse("shape-request-matrix-H");
    L.m = 1ull << h.lmhf;
    {
      u64 unit = h.NHf * (L.m / T);
      L.filled = (L.feat + unit - 1) / unit * unit;
      L.k = L.filled / unit;
    }
    L.mD = 1ull << h.lmhd;
    {
      u64 unit = L.mD / T;
      L.fillD = (2 * d + unit - 1) / unit * unit;
      L.kD = L.fillD / unit;
    }
    L.mR = 1ull << h.lmhr;
    {
      u64 unit = L.mR / T;
      L.fillR = ((u64)ledNRH() + unit - 1) / unit * unit;
      L.kR = L.fillR / unit;
    }
    struct SD {
      string name, kind;
      u64 n = 0;
    };
    vector<SD> sd = {{"NF", "R16", d},
                     {"NFH", "R16", d},
                     {"HD0", "R16", L.JV},
                     {"HD1", "R16", L.JV},
                     {"ROW", "R8", (u64)ledNRH()},
                     {"HD2", "R8", L.JV},
                     {"CNF", "FC", d},
                     {"HE0", "FE", L.JVa},
                     {"HE1", "FE", L.JV - L.JVa}};
    CallLines cl = polCallsH;
    if (cl.empty())
      cl = {{"R16a", "R16", "NF", "NFH", "HD0"},
            {"R16b", "R16", "HD1"},
            {"R8", "R8", "ROW", "HD2"},
            {"FEa", "FE", "HE0"},
            {"FEb", "FE", "HE1"},
            {"FC", "FC", "CNF"}};
    if (cl.size() > 6) refuse("policy-call-list-H");
    /* The table FE: row u is u + 2^16 EX[u]. EX[u] is 2^F exp(- u 2^-q), q = sT - sh, rounded up by
     * a recurrence on integers: the loop below, with N = ceil(2^32 exp(- 2^-q)), a stored constant.
     * Soundness requires only EX[u] >= 2^F exp(- u 2^-q) at every row, which follows by induction
     * from N / 2^32 >= exp(- 2^-q). EX[0] = 2^F, every EX[u] is at least 1, and EX[u] = 1 for
     * u >= 2^15. */
    {
      static const u64 NQ[7] = {4034748383ull, 4162825045ull, 4228380000ull, 4261543596ull,
                                4278222806ull, 4286586875ull, 4290775040ull};
      const u64 N = h.tabS < h.digSh ? 484 : NQ[h.tabS - h.digSh - 4];
      L.tauFE.resize(65536);
      u128 G = (u128)1 << (h.famF + 17);
      for (u64 u = 0; u < 65536; u++) {
        L.tauFE[u] = u + ((u64)((G + 131071) >> 17) << 16);
        G = (G * N + 0xFFFFFFFFull) >> 32;
      }
    }
    vector<char> seen(sd.size(), 0);
    for (auto &ln : cl) {
      if (ln.size() < 3) refuse("policy-call-list-H");
      CallD cd;
      cd.c.name = ln[0];
      cd.kind = ln[1];
      for (auto &o : L.calls)
        if (o.c.name == ln[0]) refuse("policy-call-list-H");
      for (size_t i = 2; i < ln.size(); i++) {
        size_t j = 0;
        while (j < sd.size() && sd[j].name != ln[i]) j++;
        if (j == sd.size() || seen[j] || sd[j].kind != ln[1]) refuse("policy-call-list-H");
        seen[j] = 1;
        Seg s;
        s.name = sd[j].name;
        s.n = sd[j].n;
        cd.segs.push_back(s);
      }
      L.calls.push_back(cd);
    }
    for (char c : seen)
      if (!c) refuse("policy-call-list-H");
    {
      string s;
      for (auto &c : L.calls) {
        s += "callH " + c.c.name + " " + c.kind;
        for (auto &g : c.segs) s += " " + g.name;
        s += "\n";
      }
      vector<u32> w((s.size() + 3) / 4, 0);
      for (size_t i = 0; i < s.size(); i++) w[i / 4] |= (u32)(unsigned char)s[i] << (8 * (i % 4));
      u32 hh[DW];
      earlier_hash_h_digest(hh, 73, w.data(), w.size());
      ownW(h.hCallsH, hh);
    }
    for (size_t q = 0; q < L.calls.size(); q++) {
      CallD &cd = L.calls[q];
      // Least number of key bits of a call: docs/format/11-4-output-head.md, "The call list".
      const int nminH = NMINH ? NMINH : (int)h.nmin;
      int fl = nminH - lt;
      place(cd, fl < 0 ? 0 : fl);
      Call &c = cd.c;
      c.n = cd.lJ + lt;
      c.kappa = 5;
      c.k = cd.kind == "R8" ? 8 : 16;
      if (cd.kind == "FC") {
        c.fn = 1;
        c.t0 = 32768;
        c.tau = &y.tauFC;
      }
      if (cd.kind == "FE") {
        c.fn = 1;
        c.t0 = 0;
        c.tau = &L.tauFE;
        // Without a stored table the call would use the closed form tau(t) = t, whatever fn is.
        if (L.tauFE.size() != 65536) refuse("internal-table-FE");
      }
      if (c.n < 10 || c.n < nminH || c.n > 44) refuse("shape-call-H-" + c.name);
      if (c.n < la || cd.lJ < ls) refuse("shape-call-H-" + c.name);
    }
    /* The head's lookup calls run as one lookup forest; `forestShape` derives and tests its shape.
     * Also tested: the lines iotaH_q, which carry no freedom: each must be n_q - dT. */
    {
      vector<Call *> clH;
      for (auto &cd : L.calls) clH.push_back(&cd.c);
      HForest &fs = L.fs;
      fs = forestShape(clH, LIDB);
      // With dT >= la all calls share the token coordinates zt and
      // the long point Z, at which the merged tables are read.
      if (fs.dT < la) refuse("forest-depth-H");
      L.kb = forestPlaneCols(fs, (int)h.lmbH); // rejects unless 5 <= lmbH <= kT + 5 ("forest-planes")
      for (size_t q = 0; q < L.calls.size(); q++) {
        if ((u64)h.iotaH[q] != (u64)fs.iota[q]) refuse("shape-call-H-" + L.calls[q].c.name);
        L.calls[q].c.iota = fs.iota[q];
      }
      // fs.pub[i]: the number of padding keys in tree i. Which keys count, and which tree has
      // column j: docs/format/11-4-output-head.md, "The padding keys of a tree".
      fs.pub.assign((size_t)fs.NT, 0);
      const int cs = fs.dT - lt;
      for (size_t q = 0; q < L.calls.size(); q++) {
        const CallD &cd = L.calls[q];
        u64 *pb = fs.pub.data() + fs.first[q];
        if ((cd.J >> cs) != (1ull << fs.iota[q])) refuse("internal-forest-place");
        for (u64 j = 0; j < cd.J; j++)
          if (!cd.used[j]) pb[j >> cs] += T;
        for (auto &s : cd.segs)
          if (s.name[0] == 'H') { // Entry x of HE1 is vocabulary entry JVa + x; in the other segments it is entry x.
            const u64 v0 = s.name == "HE1" ? L.JVa : 0;
            for (u64 x = Vv > v0 ? Vv - v0 : 0; x < s.n; x++) pb[s.pos[x] >> cs] += T;
          }
      }
      forestPubAsk(fs, clH, fs.pub); // here, on the public numbers, and not only when a proof is read
    }
    for (size_t q = L.calls.size(); q < 6; q++)
      if (h.iotaH[q]) refuse("header-absent-call-H");
    // The head sub-proof's no-overflow conditions, in u128. BW: the bound assumed on a head weight.
    auto h9 = [&](const u128 Ty) {
      // TyX bounds a residual-stream value: Ty (256^residualStreamLimbs - 1) / 255, residualStreamLimbs = 4 or 5 limbs.
      const u128 P2 = (u128)1 << 60, Ty2 = Ty * 257, TyX = Ty * (residualStreamLimbs == 5 ? 4311810305ull : 16843009ull), BW = WEIGHT_BOUND,
                 BCh = CLAMPED_CELL_BOUND;
      auto pw = [](u32 k) {
        return (u128)1 << k;
      };
      auto need = [&](bool ok, const char *n) {
        if (!ok) refuse(string("no-wrap-H-") + n);
      };
      need((pw(16) << h.KF) <= P2, "U-NF");
      // |x R| <= 2^M + sigma / 2, sigma < 2^30 (five limbs: 2^39), as at the layers' site "norm".
      need(pw(h.MF) + pw(residualStreamLimbs == 5 ? 38 : 29) + pw(h.KF - 1) + (pw(15) << h.KF) + pw(16) * (Ty2 + 32896) < P2, "W-NF");
      /* A margin is K_2 + 2^sh K_1 + 2^(sh + 15) K_0, with K_2 a byte and K_1, K_0 < 2^16 by their lookups, so it is
       * below 2^40. It is a difference of two logits minus the tie-break term, at most SH in absolute value, plus c +
       * b BIGH, with c < 2^40 and b < 2^8 by their digits. K_0 has no cell: its key is the margin minus the two
       * lower digits, times the inverse of 2^(sh + 15) in the field. This equates 2^(sh + 15) K_0 with that difference
       * in the field only. Both sides are integers within the second condition's bound, so they are equal. */
      const u128 SH = 2 * (u128)d * BCh * BW + 1;
      need(pw(8) + pw(h.digSh + 16) + pw(h.digSh + 31) <= pw(40), "W-HD0");
      need(SH + pw(40) + pw(8) * BIGH + pw(h.digSh) * (Ty2 + 32896) + (Ty + 128) < P2, "W-HD0");
      /* An honest margin has c <= C < 2^32 and b = 0, or c = 0 and b = 1. With SH <= BIGH it is
       * at least 0 for every logit gap and fits the three digits (the top two overlap by a bit). */
      need(SH <= BIGH && SH + BIGH + pw(32) <= (pw(16) + 1) * pw(h.digSh + 15), "three-digits");
      // At a position above the clip, every margin's middle digit is in the rows where EX is 1.
      need(BIGH >= SH + pw(h.digSh + 15), "three-digits");
      /* HE's key is K_1 + 2^16 ex. 2^16 is a unit of the field, so the lookup alone fixes nothing.
       * That the key equals a row u + 2^16 EX[u] as integers, so K_1 = u and ex = EX[u], follows
       * from bounds on ex (its EXC cells' approximate range check) and K_1 (its own lookup). */
      const u128 r256 = (pw(8 * EXC) - 1) / 255, Be = Ty * r256 + 128 * r256 + 1;
      need(pw(16) * (Be + pw(h.famF) + 1) + pw(16) < P2, "HE");
      // The equation on the table values ex of one position is a sum of JV of them and of zs.
      need((u128)L.JV * pw(h.famF) + pw(8 * EXC) + (u128)(L.JV - Vv + 1) * pw(h.famF) < P2, "Z");
      // The three equations on the hidden numbers sum range-checked digits of at most T positions.
      need((u128)T * (pw(40) + pw(8) * h.clipC + pw(40)) + (u128)h.theta * T < P2, "E2");
      need((u128)T * (pw(8) + pw(16)) + h.cap < P2, "E3");
      need(pw(40) + pw(40) + h.clipC < P2, "E4");
      // Second term: a weight of emb, a limb's assumed bound times the sum of 256^i, i < 4. The key
      // file's range proof is meant to show signed bytes; a request's proof shows nothing of it.
      need(TyX + pw(LBZW + 1) * 16843009ull < P2, "J-E");
      // CNF's key is packed at radix 2^28 like the layers' keys: their "guard-radix", same bounds.
      need((Ty2 + 32768) + ((Ty2 + (BCh > 4096 ? BCh : (u128)4096)) << 28) < P2 / 2 && Ty2 + 32768 + 65536 < pw(28),
           "guard-radix");
    };
    h9(TyReq);
    for (auto &c : L.calls)
      if (c.c.kappa != 5) refuse("hid-planes-" + c.c.name);
  }

  // The attention sub-proofs A_g.
  vector<AFork> AF(NGr);
  vector<AttentionSubProof> AH2(NGr);
  {
    vector<ACfg::Line> lines;
    const std::map<string, int> sn = {{"RM", sRM}, {"RD", sRD}, {"FM", sFM}, {"FD", sFD},
                                      {"WL", sWL}, {"WH", sWH}, {"IX", sIX}};
    // Every line gives at least one expanded call, and at most twelve are accepted below. The
    // count is taken here, since the loop compares each name with all names before it.
    if (polCallsA.size() > 12) refuse("policy-call-list-A");
    for (auto &ln : polCallsA) {
      ACfg::Line L;
      L.name = ln[0];
      L.kind = ln[1];
      if (L.kind != "KR" && L.kind != "KFD" && L.kind != "R16" && L.kind != "WH") refuse("policy-call-list-A");
      for (auto &o : lines)
        if (o.name == L.name) refuse("policy-call-list-A");
      for (size_t i = 2; i < ln.size(); i++) {
        auto it = sn.find(ln[i]);
        if (it == sn.end()) refuse("policy-call-list-A");
        L.segs.push_back(it->second);
      }
      lines.push_back(L);
    }
    if (lines.empty()) refuse("policy-call-list-A"); // the list has no default
    {
      const char *nmS[7] = {"RM", "RD", "FM", "FD", "WL", "WH", "IX"};
      string s;
      for (auto &L : lines) {
        s += "callA " + L.name + " " + L.kind;
        for (int g : L.segs) s += string(" ") + nmS[g];
        s += "\n";
      }
      vector<u32> w((s.size() + 3) / 4, 0);
      for (size_t i = 0; i < s.size(); i++) w[i / 4] |= (u32)(unsigned char)s[i] << (8 * (i % 4));
      u32 hh[DW];
      earlier_hash_h_digest(hh, 73, w.data(), w.size());
      ownW(h.hCallsA, hh);
    }
    if (h.Gk & (h.Gk - 1)) refuse("af-shape-heads");
    if (h.lmh < h.lt || h.lmh > 24 || h.lmo < h.lt || h.lmo > 24 || h.lmbA > 24 || h.lmS > 24 || !h.NAs || h.NAs > 64)
      refuse("shape-request-matrix-A");
    for (u64 g = 0; g < NGr; g++) {
      AFork &A = AF[g];
      ACfg &c = A.cf;
      c.lt = lt;
      c.lB = (int)h.lB;
      c.H = h.H;
      c.lgk = clog2(h.Gk);
      c.ldh = ldh;
      c.ll = ll;
      // Tested before the cast. shape() requires S <= 14 and lt + S + 1 <= 24 (3-byte remainder).
      if (h.Sx >= 32) refuse("af-policy-wd");
      c.S = (int)h.Sx;
      c.rx = (int)h.rx;
      c.kFD = (int)h.kFD;
      c.lmS = (int)h.lmS;
      c.la = la;
      c.lmb = (int)h.lmbA;
      c.nmin = (int)h.nmin;
      c.NAs = h.NAs;
      c.wcH = (1ull << h.lmh) / T;
      c.wcO = (1ull << h.lmo) / T;
      c.wcQ = y.mrow[3] / T;
      c.Ty = TyReq;
      c.lines = lines;
      for (u64 lam = 0; lam < Lam; lam++) c.sh.push_back(klSh(g * Lam + lam));
      {
        vector<u32> s(c.sh);
        std::sort(s.begin(), s.end());
        s.erase(std::unique(s.begin(), s.end()), s.end());
        size_t n = 0;
        for (auto &L : lines) n += L.kind[0] == 'K' ? s.size() : 1;
        if (n > 12) refuse("policy-call-list-A"); // At most twelve expanded calls.
      }
      // Layer sub-proof g owns qa_g, in a tree that others read (TX): k_qa columns (ktO), a = 2
      // (aO: one reader), qa_g at column 0 (col0Q). midQ is qa_g's matrix id, taken from tails[g].
      {
        AttentionSubProof &X = AH2[g];
        X.A.cf = c;
        X.in.hb = statedTarget.hb - (u32)POW; // for PlaceF::hb, as in the other sub-proofs' plans
        X.in.gid = (u32)(2 + NGr + g);
        X.in.ktO = y.kcol[3];
        X.in.aO = 2;
        X.in.col0Q = 0;
        X.in.midQ = tails[g].mats[iQA].aId;
        X.plan();
        // The line iotaA_0: verifier/docs/parameter-file.md, "7. Attention".
        if ((u64)h.iotaA0 != 32 + (u64)X.fs.dT) refuse("policy-iotaA");
        trimL(X.seam.Fp);
        if (X.A.fillQ != y.filled[3] || X.A.featQ != y.feat[3]) refuse("internal-qa-layout");
        // Layer and attention sub-proof (the reader) each record where q, kx, v and o are in qa.
        // The records and the sizes Lam, dq, dkv and ls must agree, and S2 must divide dh
        // (the reader adds a multiple of dh, then divides by S2).
        {
          TenList own_, read_ = {{"q", X.A.atQ}, {"kx", X.A.atK}, {"v", X.A.atV}, {"o", X.A.atO}};
          for (auto &t_ : y.ten)
            if (t_.mat == 3) own_.push_back({t_.name, t_.at()});
          tensorsAgree(own_, read_);
          if (X.A.Lam != Lam || X.A.dq != y.dq || X.A.dkv != y.dkv || X.A.ls != y.ls || X.A.dh % y.S2)
            refuse("plan-tensor-layout");
          // (head, channel) numbers features one to one: H heads of dh in q and o, G in kx and v.
          if (X.A.atQ.J != X.A.cf.H * X.A.dh || X.A.atO.J != X.A.atQ.J || X.A.atK.J != X.A.G * X.A.dh ||
              X.A.atV.J != X.A.atK.J)
            refuse("plan-tensor-layout");
        }
        A = X.A;
      }
    }
    {
      vector<u32> w;
      for (u64 u = 0; u < AF[0].u0; u++) w.push_back((u32)AF[0].Tt[u]);
      u32 hh[DW];
      earlier_hash_h_digest(hh, 72, w.data(), w.size());
      ownW(h.hTexp, hh);
      tblHeld(3, hh, "table-digest-exp");
    }
  }
  // The evaluation tables a layer sub-proof commits to (its vector U3: vf_smallness_test_members.inc), in order: for
  // each matrix, the points where it is read. readsZ(), readsR(): a lookup segment's reads at z, r.
  // Letters: f = af, d = ad, r = ar, q = qa, Q = a short table of qa_g, i = xb_g, o = xb_(g+1).
  auto readsZ = [&](const string &s) -> string {
    if (s == "G" || s == "U" || s == "V" || s == "CN1" || s == "CN2") return "df";
    if (s == "O") return "fiQ"; // Q: a short table of qa_g (L = T).
    if (s == "D") return "fo";
    if (s == "XR") return "d";
    if (s == "LR") return "dr";
    if (s == "LT" || s == "DD" || s == "ROW" || s == "LH" || s == "SEL" || s == "MR" || s == "L3" || s == "CLR")
      return "r";
    if (s == "CQK" || s == "CV") return "fq";
    return "f";
  };
  auto readsR = [&](const string &s) -> string {
    if (s == "N1") return "rfi";
    if (s == "N2") return "rf";
    if (s == "QK") return "d";
    if (p3) {
      if (s == "G" || s == "U" || s == "XR") return "rd";
      if (s == "D" || s == "DD" || s == "MR") return "r";
    }
    return "";
  };
  vector<string> tscCalls, rdAF, rdQA, rdXI, rdXO, rdAD, rdAR, rdQS;
  {
    auto has = [](const string &s, char c) {
      return s.find(c) != string::npos;
    };
    vector<string> zs(y.calls.size()), rs(y.calls.size());
    for (size_t q = 0; q < y.calls.size(); q++)
      for (auto &s : y.calls[q].segs) {
        zs[q] += readsZ(s.name);
        rs[q] += readsR(s.name);
      }
    for (size_t q = 0; q < y.calls.size(); q++)
      if (!rs[q].empty()) tscCalls.push_back(y.calls[q].c.name);
    struct LM {
      char c = 0;
      vector<string> *v = nullptr;
      bool zv = false, zab = false;
    } lmx[4] = {
        {'f', &rdAF, true, true}, {'q', &rdQA, false, false}, {'i', &rdXI, true, false}, {'o', &rdXO, false, false}};
    // All calls share the points Z, zt, Y and rt (`pt`, vf_layers_check.inc), so a matrix has at most one
    // table at each: at Z (short table: zt) if a lookup segment reads it at z, at Y (rt) if at r.
    auto any = [&](const vector<string> &v, char c) {
      for (auto &s : v)
        if (has(s, c)) return true;
      return false;
    };
    for (auto &m : lmx) {
      if (any(zs, m.c)) m.v->push_back("Z");
      if (m.zv) m.v->push_back("Z_V");
      if (any(rs, m.c)) m.v->push_back("Y");
      if (m.zab) {
        m.v->push_back("Z_a");
        if (p3) m.v->push_back("Z_b");
      }
    }
    // The short tables, in the order ad_g, ar_g, qa_g.
    struct SM {
      char c = 0;
      vector<string> *v = nullptr;
      bool pv = false;
    } smx[3] = {{'d', &rdAD, false}, {'r', &rdAR, true}, {'Q', &rdQS, false}};
    for (auto &m : smx) {
      if (any(zs, m.c)) m.v->push_back("zt");
      if (m.pv) {
        m.v->push_back("P_V");
        m.v->push_back("r_tV");
      }
      if (any(rs, m.c)) m.v->push_back("rt");
    }
  }
  const int lNI = clog2(NI), muV = lNI + lPd + lt, muH = ll + lPf + lt, muH2 = ll - 1 + lfe + lt;
  if (muV < 9 || muH < 9 || (p3 && muH2 < 9)) refuse("shape-product-layer");
  const u64 GSaf = y.filled[0] * T >> la, GSxb = y.xbFilled * T >> la, GSqa = y.filled[3] * T >> la;
  const size_t wW =
      WPE * (rdAF.size() * GSaf + rdQA.size() * GSqa + (rdXI.size() + rdXO.size()) * GSxb + rdAD.size() * y.filled[1] +
             rdAR.size() * y.filled[2] + rdQS.size() * y.filled[3] + 3 * y.kbm);

#include "vf_layers_plan.inc"
#include "vf_output_head_plan.inc"
  // Layer sub-proof G_g commits to an array PX_i (a "tree") for each border matrix, one that
  // another sub-proof reads. Each has one reader, whose placement record (plFor) is compared once
  // with the owner's (plOwn): the three counts must be 2 NGr, also under -plan.
  {
    size_t own = 0, fts = 0;
    for (auto &P : hidPlans2) {
      own += P.tl.plOwn.size();
      fts += P.tl.plFor.size();
    }
    fts += PH2.tl.plFor.size();
    for (auto &X : AH2) fts += X.seam.tl.plFor.size();
    if (borderCmpF != own || own != fts || fts != 2 * (size_t)NGr) refuse("internal-border-count");
    // a_X, the number of sub-proofs that use symbol tree X (a matrix's cells, three to an element),
    // must be 1 + the number of plFor records that name X. In these sources a_X sizes nothing:
    // it is recorded, compared and tested here; no number of masks or random coordinates follows
    // from it.
    auto readersOf = [&](int owner, int idx) {
      u32 n = 0;
      auto look = [&](const HTailF2 &t) {
        for (auto &pl : t.plFor) n += pl.owner == owner && pl.rootIndex == idx;
      };
      for (auto &P : hidPlans2) look(P.tl);
      look(PH2.tl);
      for (auto &X : AH2) look(X.seam.tl);
      return n;
    };
    for (size_t g = 0; g < hidPlans2.size(); g++) {
      const HPlan2 &P = hidPlans2[g];
      for (size_t i = 0; i < P.tl.ix.PK.size(); i++)
        if (P.Tp.nt[P.tl.ix.PK[i]].a != 1 + readersOf((int)g, (int)i)) refuse("policy-sym-readers");
      for (size_t i = 0; i < P.tl.ix.TX.size(); i++)
        if (P.Tp.nt[P.tl.ix.TX[i]].a != 1 + readersOf((int)g, (int)(P.tl.ix.PK.size() + i)))
          refuse("policy-sym-readers");
    }
    for (int t : PH2.tl.ix.PK)
      if (PH2.Tp.nt[t].a != 1) refuse("policy-sym-readers");
    if (!PH2.tl.ix.TX.empty()) refuse("policy-sym-readers");
    for (auto &X : AH2) {
      for (int t : X.seam.tl.ix.PK)
        if (X.seam.Tp.nt[t].a != 1) refuse("policy-sym-readers");
      if (!X.seam.tl.ix.TX.empty()) refuse("policy-sym-readers");
    }
  }
  trimL(PH2.Fp);
  {
    // tails[g] and the plan hidPlans2[g] each list the matrices of layer sub-proof g. The two
    // lists' total numbers of columns (foldColumns()) must be equal.
    for (u64 g = 0; g < NGr; g++) {
      u64 k = 0;
      for (auto &M : hidPlans2[g].tl.mats) k += M.k;
      if (k != tails[g].foldColumns()) refuse("internal-fold-columns");
    }
    u64 NMlab = 0;
    std::map<u64, int> labs;
    auto look = [&](const vector<TMat> &m) {
      for (auto &M : m) {
        if (M.kind == 0) {
          NMlab++;
          if (labs[M.aId] & 1) refuse("internal-nmlab");
          labs[M.aId] |= 1;
        } else if (M.kind == 1)
          labs[M.aId] |= 2;
        else
          refuse("internal-nmlab");
      }
    };
    for (auto &P : hidPlans2) look(P.tl.mats);
    look(PH2.tl.mats);
    for (auto &X : AH2) look(X.seam.tl.mats);
    for (auto &q : labs)
      if (!(q.second & 1)) refuse("internal-nmlab");
    // NMlab counts the matrices that some sub-proof owns (TMat::kind 0); no bound uses it.
    // Rejected above: a label (aId) owned twice, a foreign matrix (kind 1) whose label no sub-proof
    // owns, a weight matrix (kind 2).
    if (labs.size() != NMlab || !NMlab) refuse("internal-nmlab");
  }
  const int muVH = lPd + lt;
  if (muVH < 9) refuse("shape-product-layer-H");
  if (!quiet)
    printf(
        "ok public inputs: parameter file, shape, no-overflow conditions; profile %u; %llu blocks of layers  %.3f s\n",
        h.prof, (unsigned long long)NGr, lap());

  vector<string> forkLine(2 * NGr + 1);
#include "vf_layers_check.inc"
#include "vf_output_head_check.inc"
  // KL.bin's bytes, as the file has them: its words, the low byte first.
  string klBytesH;
  for (u32 x : KL)
    for (int i = 0; i < 4; i++) klBytesH.push_back((char)(u8)(x >> (8 * i)));
#include "vf_segments.inc"
  // buildLinksH() (vf_segments.inc) builds the request's segments, one per sub-proof. It returns
  // false if the call only asked for the list of arrays.
  std::function<int()> planEndH; // the last output of a -plan call and its exit status
  if (!buildLinksH()) {
    printf("PLAN ONLY: no proof was checked\n");
    return 4;
  }
  const blk::Parameters &parH = confH->parameters(); // as parsed for the plans (not PS.frame)
  // acctOfH(), below, compares one file plan's accounting with the requested bits. oneTime: a key
  // or leaf file.
  auto acctFormH = [&](const blk::FilePlan &plan) { return blk::account(blk::merged(confH->sized().d.set, {plan})); };
  /* A key or leaf file's proximity terms (`closeness` in names and output) rest on the bound within
     a third of a code's distance (closeness[0]) or on the theorem on codes, taken as given
     (closeness[1]). oneTimeLineH prints the counts. */
  auto oneTimeLineH = [&](const char *what, const blk::Accounting &acct) {
    printf("%s\n", oneTimeLine(what, acct.closeness[0], acct.closeness[1]).c_str());
  };
  /* True if the weight table's test has its split step, under the shapes that `blk::check_proof`
     and `blk::check_key` use; blocks/iop/lin.h branches on this call. Why: SERVED_TWO_CODES. */
  const bool twoCodesH = confH->sized().batch.S.shared_split_needed() && confH->sized().once.S.shared_split_needed();
  auto acctOfH = [&](const blk::FilePlan &plan, const char *what, const string &object, bool oneTime) {
    const blk::Accounting acct = acctFormH(plan);
    if (!acct.formed || !acct.holds || !((double)acct.target >= minBits)) {  // the plan's own `sec`
      blk::print_accounting(acct, what);
      // The rounds meet the file's own target: without this line nothing printed says what failed.
      if (acct.formed && acct.holds) printf("accounting: asked of this call: %g bits, above the parameter file's target of %u bits\n", minBits, (unsigned)acct.target);
      refuse("security-bits");
    }
    // Some round meets the target only with its work: a passing stage ends STAGE PASSED WITH WORK
    // (WorkNote). needs_work is exact; the parent compares bits and work with the call's record.
    if (acct.needs_work) noteWorkNeeded(acct.lists.bits, acct.lists.work, object + ": " + acct.lists.name);
    if (oneTime && !in.then) oneTimeLineH(what, acct);  // not into another program's output (cp_configured.h)
    acctNote = acct; // this stage's record for the parent process (vf_accounting_record.h)
  };
  if (in.then) {
    // For cp_with_configured() (cp_configured.h): another program gets *confH. First the three
    // stages' checks at the requested bits, so that it makes nothing that they reject.
    acctOfH(confH->key(), "a key", "key", true);
    acctOfH(confH->entry(), "an entry", "leaf", true);
    acctOfH(confH->proof(), "a proof", "request", false);
    return (*in.then)(*confH);
  }
  /* ---- The envelope (`SealedCallInputs`) is opened: key and leaf file, request, y, proof. So far only
     the configuration files and the options outside it were read, and all plans are made. ---- */
  const CallInputs &ask = in.sealed->open(*confH);
  // The anchor and the leaf table, each read from its file's first bytes without the proofs that
  // follow. The stages leaf and request work on such tables, so their checkers say at best "yes if
  // the table is tested" (blk::IfTested, blocks/iop/foldeq.h): a YES needs all three to pass.
  blk::UnprovedTable first, further;
  string idHexH;
  /* The commitment ID (blocks/frame/heads.h). Every stage computes it from the key file's first
     bytes, checked for form only, and compares it with -id before it uses more of the envelope. */
  if (ask.keyB) {
    const char *why = nullptr;
    blk::Reader head((const blk::u8 *)ask.keyB->data(), ask.keyB->size());
    if (!blk::read_key_table_unproved(*confH, head, first, why)) {
      printf("key: %s\n", why ? why : "refused");
      refuse("key-refused");
    }
    blk::u8 by[blk::DIGEST_BYTES];
    blk::digest_to_bytes(blk::published_id(*confH, first.reader()), by);
    char c[3];
    for (blk::u8 x : by) snprintf(c, sizeof c, "%02x", x), idHexH += c;
    // An ID other than -id is not printed: a caller who pasted it into -id would accept another
    // commitment's files (see cpStagesRun()).
    if (!ask.idAsked.empty() && ask.idAsked != idHexH) {
      refuse("model-statement");
    }
    if (in.oneTime == 3) printf("h_par=%s\n", hex64(hParDigest.w).c_str());
    printf("%s%s\n", in.oneTime == 3 ? IDLINE : STAGEIDLINE, idHexH.c_str());
  }
  // The checker's verdict and stageWorkNote come from the same plan: they must agree on proof of work.
  auto workHeldH = [&](bool withWork) {
    if (withWork != stageWorkNote.needed) refuse("internal-work-two-ways");
  };
  // The stages key and leaf are modes of this function: both need *confH, which needs the plans
  // above.
  if (in.oneTime == 3) { // the planning call "id": the ID of these files, and nothing checked
    if (!planOnly || !ask.keyB || ask.entryB) refuse("internal-cp-cut");
    printf("PLAN ONLY: no proof was checked\n");
    return 4;
  }
  if (in.oneTime) {
    const bool isKey = in.oneTime == 1;
    if (!planOnly || !ask.keyB || (isKey ? ask.entryB != nullptr : !ask.entryB)) refuse("internal-cp-cut");
    acctOfH(isKey ? confH->key() : confH->entry(), isKey ? "a key" : "an entry", isKey ? "key" : "leaf", true);
    const char *why = nullptr, *place = nullptr; // place: which chain (reasons recur); only printed
    bool passed = false, withWork = false;
    if (isKey) {
      blk::Reader file((const blk::u8 *)ask.keyB->data(), ask.keyB->size());
      blk::TestedTable tested; // not used further: this process ends with the stage
      const blk::ProofVerdict verdict = blk::check_key(*confH, file, tested, why, nullptr, &place);
      passed = verdict != blk::ProofVerdict::NO, withWork = verdict == blk::ProofVerdict::YES_WITH_WORK;
    } else {
      // `first` was checked for form only: the stage key checks its proofs, and a YES needs both.
      blk::Reader file((const blk::u8 *)ask.entryB->data(), ask.entryB->size());
      const blk::IfTested verdict = blk::check_entry_if_tested(*confH, first, file, why, nullptr, &place);
      passed = verdict != blk::IfTested::NO, withWork = verdict == blk::IfTested::YES_WITH_WORK_IF_THE_TABLE_IS_TESTED;
    }
    if (!passed) {
      printf("%s: %s%s%s%s\n", isKey ? "key" : "entry", why ? why : "refused", place ? " (" : "", place ? place : "", place ? ")" : "");
      refuse(isKey ? "key-refused" : "entry-refused");
    }
    workHeldH(withWork);
    if (!quiet) printf("ok the %s: %zu bytes, read to the end\n", isKey ? "key" : "entry", isKey ? ask.keyB->size() : ask.entryB->size());
    // A condition of the standard parameter form: its name if it is unmet.
    const string form = twoCodesH ? "" : SERVED_TWO_CODES;
    const string fields = " word=" + std::to_string(runVersionWord) + " sec=" + std::to_string(parH.sec) + " asked=" + std::to_string((long long)minBits);
    return lastLine(belowBar, form, LIDB_SEALED, 0, isKey ? "STAGE PASSED: key" : "STAGE PASSED: leaf", 3, fields.c_str(),
                      isKey ? "research code" : "research code; the key's table was taken on its form alone: what a key says of it is the stage key's to check, on the same bytes");
  }
  if (planOnly != !ask.proofB) refuse("internal-cp-cut");
  {
    // A request or y file whose length is not a multiple of 4 bytes is rejected here, an empty one
    // in stFromCall().
    if (ask.oddStatement) refuse("statement-required");
    stFromCall(so, ask.request, ask.y, T);
    // The word of y follows the tokens' digest. The statement is parsed again so that h has y.
    layW((size_t)(h.hTok - st.data()) + DW, so.y);
    h = parseHeader(st);
    if (h.yTok != so.y) refuse("internal-statement-prescan");
    // y is not part of the token row, so its range is checked here.
    if (so.y >= Vv) refuse("token-range");
    if (h.yTok >= Vv) refuse("header-y");
    {
      // The request is T tokens, then r and e. The statement contains the tree hash of all T + 2
      // words; the tokens and positions checked below are taken from the words that are hashed.
      if (so.req.size() != T + 2) refuse("token-length");
      tokv.assign(so.req.begin(), so.req.begin() + (ptrdiff_t)T);
      u32 hh[DW];
      TH(hh, so.req.data(), so.req.size());
      ownW(h.hTok, hh);
    }
    const u64 rowR = so.req[T], rowE = so.req[T + 1];
    for (u32 v : tokv)
      if (v >= Vv) refuse("token-range");
    // The request's range is positions r to e - 1. stFromCall() has already rejected all but
    // r < e <= T; otherwise the range would be empty, or the loop would run past the vectors' end.
    if (rowR >= rowE || rowE > T) refuse("statement-range");
    hWin.assign(T, 0);
    hOn.assign(T, 0);
    hOnC.assign(T, 0);
    for (u64 t = rowR; t < rowE; t++) {
      hOn[t] = 1;
      hWin[t] = t + 1 < T ? tokv[t + 1] : h.yTok;
      hOnC[t] = t + 1 < T;
      hRC += t + 1 < T;
    }
  }
  {
    /* The leaf table on which the proof was made is read from the leaf file, for form only. The
       stages leaf and key check that file and the key file: a YES needs all three to pass. */
    if (!planOnly) {
      if (!ask.keyB || !ask.entryB) refuse("internal-cp-cut");
      const char *why = nullptr;
      blk::Reader head((const blk::u8 *)ask.entryB->data(), ask.entryB->size());
      if (!blk::read_entry_table_unproved(*confH, head, further, why)) {
        printf("entry: %s\n", why ? why : "refused");
        refuse("entry-refused");
      }
      static_assert(HEXW == 2 * blk::DIGEST_BYTES, "the field ms= takes the ID's digits");
      memcpy(commitmentIdHex, idHexH.c_str(), HEXW + 1);
    }
  }
  // Every word of the statement must have been written through layW(), which marks it in so.laid.
  // The test comes after the statement's last write and before any word of the proof is used.
  for (size_t i = 0; i < st.size(); i++)
    if (!so.laid[i]) refuse("internal-statement-cover");
  if (!quiet) printf("ok the request's form; key's and entry's tables read for their form alone\n");
  /* The accounting (blocks/numbers/accounting.h) is that of the plan which the proof's checker
     follows. The deciding comparison is exact: every round's error, proof of work counted, is at
     most 2^-sec (the parameter file's line `sec`), and the requested bits must not exceed sec. */
  acctOfH(confH->proof(), "a proof", "request", false);
  // The call above compared the proof file's plan. The record for the parent is the whole call's:
  // all three files' plans together, as in a planning call. Stages key and leaf compare and record
  // their own.
  acctNote = blk::account(blk::merged(confH->sized().d.set, {confH->key(), confH->entry(), confH->proof()}));

  // The conditions of the standard parameter form; `Served` (vf_common.h) explains each. Three
  // cannot fail here, since such inputs are rejected earlier; they stay to keep the list complete:
  // toksrc not 1 ("statement-required"), no layout file ("plan-layout"), S >= 32 ("af-policy-wd").
  // The softmax and rotary tables are computed with C library functions; their conditions are meant
  // to accept only parameters with every entry far from a rounding tie.
  bool oneShift = true;
  for (u64 l = 0; l < h.L; l++) oneShift = oneShift && klSh(l) == klSh(l - l % Lam);
  // norm-shifts. A norm multiplies by a reciprocal scaled by 2^M, then shifts right by K bits.
  // "KL-norm" and "policy-norm-F" allow K > M. |output| is at most about 2^(M + 1 - K): 0 for every
  // input from K = M + 3 on. K <= M rules that out; a small M - K still leaves little.
  bool normShifts = h.KF <= h.MF;
  for (u64 l = 0; l < h.L; l++) normShifts = normShifts && kl(l, LK1) <= kl(l, LM1) && kl(l, LK2) <= kl(l, LM2);
  // one-placement: every layer group places its border matrices as the first does.
  auto placedAlike = [](const blk::ReqArray *a, const blk::ReqArray *b) {
    if (!a || !b || a->message != b->message || a->laid.size_i != b->laid.size_i || a->laid.size_c != b->laid.size_c ||
        a->laid.blocks.size() != b->laid.blocks.size())
      return false;
    for (size_t i = 0; i < a->laid.blocks.size(); i++) {
      const blk::LaidBlock &x = a->laid.blocks[i], &y = b->laid.blocks[i];
      if (x.first_i != y.first_i || x.first_c != y.first_c || x.first_row != y.first_row || x.first_place != y.first_place ||
          x.bits_i != y.bits_i || x.bits_c != y.bits_c || x.row_bits != y.row_bits || x.place_bits != y.place_bits || x.i_is_low != y.i_is_low)
        return false;
    }
    return true;
  };
  bool onePlacement = true;
  for (size_t l = 0; l < linksH.size(); l++) // linksH[0] is the first layer group's segment
    if (linksH[l].kind == 0) {
      onePlacement = onePlacement && linksH[l].tl->pxMat.size() == linksH[0].tl->pxMat.size();
      for (size_t i = 0; onePlacement && i < linksH[l].tl->pxMat.size(); i++)
        onePlacement = placedAlike(layoutH.find(l, "M" + std::to_string(linksH[l].tl->pxMat[i])),
                                   layoutH.find(0, "M" + std::to_string(linksH[0].tl->pxMat[i])));
    }
  // A dense model (NE 0) with neither clamp line meets routed-experts and both-clamps: it is the
  // standard dense parameter form (verifier/docs/parameter-file.md). With experts the two
  // conditions are unchanged. A clamp line on a dense model is refused earlier (policy-gcl,
  // policy-rcl), so denseStandard never hides a clamp.
  const bool denseStandard = h.NE == 0 && !RCL && !GCL;
  // table-digests is met when the digest set the model uses is complete: four tables with experts,
  // the three it uses (SILU, RoPE, softmax; not FS/SIG) for a dense model.
  const bool tablesComplete =
      h.NE ? tblAt == TBLW
           : (tblCount[0] == DW && tblCount[1] == 0 && tblCount[2] == DW && tblCount[3] == DW);
  string notServed =
      servedForm({h.NE > 0 || denseStandard, residualStreamLimbs == 5, (RCL && GCL) || denseStandard,
                    h.toksrc == 1, layoutP != nullptr,
                    tablesComplete, !egSeen && !smaxSeen, h.Sx < 32, h.D == 2048, oneShift, onePlacement,
                    h.Sx == 12 && h.rx == 8, rotaryServed(h.theta0, dh, h.lt),
                    normShifts, twoCodesH,
                    !ask.nextTokenAsked || ((u64)so.r + 1 == T && so.e == T),
                    // A planning call has neither -next-token nor -policy, which meets next-token
                    // and policy-named; planOnly meets exact-match. So it names none of the three.
                    planOnly || ask.policy.given || !hRC ||
                        acceptancePolicyIsExactMatch(h.clipC, h.theta, h.cap, h.digSh, h.tabS),
                    !ask.policy.given || ask.policy.is(h.clipC, h.theta, h.cap, h.logS, h.digSh, h.tabS)});

  if (!quiet) printf("layout: the layout file; it is lawful (blocks/frame/req_frame.h, blocks/iop/req_rows.h)\n");
  if (!quiet) printf("%s", planLineH().c_str());
  auto hexOfH = [](const blk::Digest &d) {
    blk::u8 by[blk::DIGEST_BYTES];
    blk::digest_to_bytes(d, by);
    string o;
    char c[3];
    for (blk::u8 x : by) snprintf(c, sizeof c, "%02x", x), o += c;
    return o;
  };
  auto bytesOfH = [](const vector<u32> &w) {
    string o;
    for (u32 x : w)
      for (int i = 0; i < 4; i++) o.push_back((char)(u8)(x >> (8 * i)));
    return o;
  };
  // The configuration digest: docs/format/FORMAT.md, "4.1 The first query".
  const blk::Digest configurationH = confH->fixed().configuration;
  /* The statement as the first chain's first query has it: every word, but 0 in place of y
     (statedTarget.at indexes the word hb, which follows y's word). y is not part of the statement: the
     prover sends it unmasked, and `check_proof_if_tested` compares it with the caller's. */
  vector<u32> stNoReplyH = st;
  stNoReplyH.at(statedTarget.at - 1) = 0;
  planEndH = [&]() {
    // For all three files together.
    if (!blk::print_plan("request", confH->sized(), subH, shapesAsked, fieldsAsked)) refuse("security-bits");
    printf("plan: profile %u: the profile rules out no result\n", h.prof);
    // The lines that the stages key and leaf print.
    oneTimeLineH("a key", acctFormH(confH->key()));
    oneTimeLineH("an entry", acctFormH(confH->entry()));
    if (!notServed.empty())
      printf(
          "plan: not the standard parameter form (%s): no run with these public inputs can be a verification (at best status %d)\n",
          notServed.c_str(), NOT_A_VERIFICATION_STATUS);
    if (h.tabS + 4 != h.digSh)
      printf("plan: the policy's kind is %s: every whole call under this parameter file ends NO: policy\n", policyKind(h.clipC, h.theta, h.cap, h.logS, h.digSh, h.tabS));
    if (LIDB != LIDB_SEALED) printf("plan: lidb %d\n", LIDB);
    if (NMINH) printf("plan: nminH %d\n", NMINH);
    if (gateTableInputFractionBits != 8 || SMAX != 4095)
      printf("plan: the SiLU table: key step 2^-%d, values clamped to +-%d (lines eg, smax)\n", gateTableInputFractionBits, SMAX);
    if (RCL) printf("plan: the router's logits are clamped (line rcl)\n");
    if (GCL) printf("plan: the input of the SiLU table is clamped (line gcl)\n");
    if (residualStreamLimbs == 5) printf("plan: a value of the residual stream has five limbs (line xl)\n");
    if (tablesComplete)
      printf("plan: this program's %s public tables match the parameter file's digests (lines %s)\n",
             h.NE ? "four" : "three", h.NE ? "tsilu0 .. texp" : "tsilu0, trope and texp");
    // The statement's length in words, and the place of y in it; words are counted from 0.
    printf("statement: %zu words, y at word %zu\n", st.size(), (size_t)(h.hTok - st.data()) + DW);
    printf("h_par=%s\nconfiguration=%s\n", hex64(hParDigest.w).c_str(), hexOfH(configurationH).c_str());
    printf("dw=%u\nPLAN ONLY: no proof was checked\n", DW);
    return 4;
  };
  if (planOnly) return planEndH();

  /* The proof file. `check_proof_if_tested` (blocks/frame/check.h) reads it once. Its parts in
     order: docs/format/FORMAT.md, "7.8 A proof file". The first chain has the segments, checked
     through runLinksH, then their pending claims on the weight table. The tests that identify
     format and commitment are done (see namingDone). */
  namingDone = true;
  const blk::UnprovedTable &tableH = further; // named by its root; blocks/ reads its opened columns
  const string requestH = bytesOfH(stNoReplyH);
  const string fileWordsH = bytesOfH(so.req); // the request file's words
  blk::Reader fileH((const blk::u8 *)ask.proofB->data(), ask.proofB->size());
  blk::ProofFacts factsH;
  {
    const char *why = nullptr;
    // The caller's y.
    const blk::E givenH = blk::E::from(blk::from_u64(so.y));
    const blk::LinksCheck linksH = runLinksH;
    const double tProof = now();
    if (!quiet) printf("seconds: before the proof is read (the scan, the plans, blk::configure(), the request's form, the accounting) %.3f\n", tProof - t00);
    const blk::IfTested verdict = blk::check_proof_if_tested(*confH, tableH, ask.entry, std::vector<blk::u8>(requestH.begin(), requestH.end()), std::vector<blk::u8>(fileWordsH.begin(), fileWordsH.end()), givenH, fileH, linksH, nullptr, why, &factsH);
    if (verdict == blk::IfTested::NO) {
      printf("proof: %s\n", why ? why : "refused");
      refuse(why && !strcmp(why, "the proof is about another reply") ? "reply-differs" : "proof-refused");
    }
    workHeldH(verdict == blk::IfTested::YES_WITH_WORK_IF_THE_TABLE_IS_TESTED);
    if (!quiet)
      printf("seconds: the links with their last test %.3f | behind them (the rows on the table, the place, the shared chain, the second chain) %.3f\n", linksEndH - tProof, now() - linksEndH);
  }
  // The flags above index NGr are the attention sub-proofs'.
  for (u64 g = 0; g < 2 * NGr + 1; g++)
    if (!forkRan[g]) refuse("internal-fork-not-finished");
  if (!quiet)
    printf("ok the proof file: %zu bytes, read to the end: %zu links with what they leave on the table of weights; the chain that the whole batch shares; the chain that ends the request\n",
           ask.proofB->size(), linksH.size());
  // Returns the fields T= to F= of the line "result:".
  statedW = [&]() {
    // T is the parameter file's number of token positions. The constant lim=- says that no length
    // limit or stop rule is checked: the statement is about the positions of its range only.
    //
    // C=, theta=, cap=, s=, sh=, sT= and F= are the parameters of the acceptance policy. An
    // acceptance means only as much as they allow: Hdr has the inequalities. (Under the exact
    // match, C = theta = 2^sh, cap = 0 and sT = sh - 4, every checked token is the argmax of its
    // position.) So they are in the line "result:".
    // R, the number of the range's positions below T - 1, is a function of the caller's r and e,
    // so it is not in the line (a line that lists next-token or exact-match as not met reveals
    // that much of r and e).
    //
    // The line must hold no plain function of the caller's request or y. So it has no y (a token)
    // and no unsalted digest of the tokens or the statement: a guess at the tokens could be tested.
    //
    // dj= ties a stored result to its request and proof: the seal of the request's first chain (the
    // digest that ends it), which depends on the statement and on the chain's 32 random bytes.
    return " T=" + std::to_string(T) + " lim=- C=" + std::to_string(h.clipC) + " theta=" + std::to_string(h.theta) +
           " cap=" + std::to_string(h.cap) + " s=" + std::to_string(h.logS) + " sh=" + std::to_string(h.digSh) +
           " sT=" + std::to_string(h.tabS) + " F=" + std::to_string(h.famF);
  };
  // The line "seal" has the same digest as the field dj=.
  printf("seal %s\nweight hiding: NOT CHECKED BY THIS RUN (it rests on how the commitment was made)\n", hexOfH(factsH.seal).c_str());
  {
    // Repeats the check on the version word that follows polSkeleton().
    if (!runVersionWordFixed || !versionWordAdmitted(runVersionWord)) refuse("internal-version-word");
    // Each field: docs/format/FORMAT.md, "14.7 The output of a whole call".
    char fields[1024];
    if ((size_t)snprintf(fields, sizeof fields,
                         " word=%u ms=%s sec=%u t=%llu asked=%.0f by=" REQUESTED_BITS_RULE_TEXT
                         "%s leaves=%llu place=%llu card=0 judged=1 of 1 dj=%s build=%s dw=%u",
                         runVersionWord, commitmentIdHex, parH.sec, (unsigned long long)confH->sized().d.link.t,
                         minBits, // the requested bits
                         statedW().c_str(), (unsigned long long)parH.slots, (unsigned long long)factsH.place, hexOfH(factsH.seal).c_str(),
                         VF_SOURCE_DIGEST, DW) >= sizeof fields)
      refuse("internal-last-line"); // never truncated; the fields stay far under 1,024 characters
    return lastLine(belowBar, notServed, LIDB, NMINH, "STAGE PASSED: request", 3, fields,
                      "research code; the test on the weight table was checked");
  }
}

// ---- The stages of a call, each in a process of its own ----
/* `cpMain` reads every file once; `cpStagesRun` forks one child process for each stage. So all
   stages work on the same bytes, each has its own copy of the file-scope variables that `runStage`
   sets, and a stage can end through the exit in `refuse`. The parent reads a stage's result from
   its printed text and its exit status. So a stage returns 3 or 12 only through `lastLine`, and no
   other printed line may begin like one that `cpLineOf` seeks. A call name could spell the first
   word of one (`result`, `work`, `computed`): by design a file's bytes appear only inside a line,
   as hex digests, numbers or call names. */
struct CpStage {
  const char *name;
  int threads;
  std::function<int()> run;
};
// Ends a call that verified nothing because the program or the machine failed: prints the last two
// lines, returns exit status 5. how: what happened. name: a stage's rejection name that satisfies
// internalName(), or one of this file's. Only under the two names below can a repeated call end
// differently: the cause may be the machine.
// Or the program: a failed BLK_REQUIRE (blocks/base.h), C++ library or hardening check, an uncaught
// exception and, in places, lack of memory give the same signal 6; the parent cannot tell which.
static constexpr int NOT_CHECKED_STATUS = 5;
// Exit status if standard output cannot be written: docs/format/FORMAT.md, "14.3 The results".
// Without it a caller that reads only the status sees 0.
static constexpr int NOTWRITTENSTATUS = 15;
static int cpNotChecked(const string &how, const string &name = "internal-stage") {
  const string bare = name.substr(name.rfind(':') + 1); // drops a sub-proof's prefix, if any
  const bool machine = bare == "internal-stage" || bare == "internal-exception-parFor";
  printf("%s: %s\nNOT CHECKED: %s\n", machine ? "the verifier itself failed: nothing has been verified. The cause may be the machine (a want of memory, for one) or the program (a guard that ended a stage has left its line on standard error); what the program causes comes again on the same files" : "a guard of the verifier's own spoke: nothing has been verified, and on the same files the call ends so again",
         how.c_str(), name.c_str());
  return NOT_CHECKED_STATUS;
}
// Returns the rest of the last line of `out` that begins with `key`, or "".
static string cpLineOf(const string &out, const string &key) {
  size_t at = out.rfind("\n" + key);
  if (at == string::npos) {
    if (out.compare(0, key.size(), key)) return "";
    at = 0;
  } else
    at++;
  const size_t e = out.find('\n', at);
  return out.substr(at + key.size(), (e == string::npos ? out.size() : e) - at - key.size());
}
// Returns ": <why>" if the line before the last line "NO: .." of `out` is "key: <why>",
// "entry: <why>" or "proof: <why>". Returns "" without such a line, or if <why> has anything but
// letters, blanks and , . : ; ' ( ) - in it: a rejected call's last lines must show nothing that
// was read from a file.
static string cpReasonOf(const string &out) {
  const size_t no = out.rfind("\nNO: ");
  if (no == string::npos || !no) return "";
  const size_t from = out.rfind('\n', no - 1) + 1; // 0 if the line is the first
  const string line = out.substr(from, no - from);
  for (const char *head : {"key: ", "entry: ", "proof: "}) {
    if (line.compare(0, strlen(head), head)) continue;
    const string why = line.substr(strlen(head));
    for (char c : why)
      if (!c || (!isalpha((unsigned char)c) && !strchr(" ,.:;'()-", c))) return "";
    return why.empty() ? "" : ": " + why;
  }
  return "";
}
/* What `cpMain` passes to `cpStagesRun`. hPar: the digest of the three configuration files, only
   printed. named: the call had -id. needed: no pass without -id. */
struct CpId {
  string hPar, asked; // asked: the caller's ID, in lower-case hex digits; empty: none
  bool named = false, needed = true;
  bool policyGiven = false; // the call had -policy
  bool showStages = false;  // -show-stages: prints the stages' output also if the call cannot pass
  bool noWork = false;      // -no-proof-of-work: no pass if a round meets the target only with its work
};
// What cpCovered() returns. line: "covered: ..". policy: the parameter file's acceptance policy,
// as printed after "WITHIN LIMITS: ". otherKind: a request token was checked under one other than
// exact-match (r = T - 1 covers only y). kindTaken: sT = sh - 4, an accepted kind. fields (r= to
// leaf=) and claim: for the line RESULTWORD. claim: "next-token", y alone; "exact-match", request
// tokens, each the model's first choice; "policy", tokens within another policy's limits.
struct Covered {
  string line, policy, fields;
  const char *claim = "";
  bool otherKind = false, kindTaken = false;
};
// clang-format off
/* Runs the stages and prints the call's last lines. good, goodStatus: the last line and exit status
   if every stage passed. Last line (exit status); the first case that holds:
     NO: <name> (1)               a stage rejected
     NOT CHECKED: <name> (5)      the program or the machine failed
     NOT-A-VERIFICATION (14)      a stage ended so (the lines before repeat its causes), a
                                  verification call (printed: whole call) had no -id (id-named), or
                                  the call had -no-proof-of-work and a round needs its work
     WITHIN LIMITS: <policy> (0)  a request token was checked under a policy other than exact-match
     good (goodStatus)
   NO: the first rejecting stage's rejection name.
   NOT CHECKED: a stage ended under a name that begins with "internal-", or has it right after a
   sub-proof's prefix ("G0:internal-.."): that whole name (`internalName`, vf_common.h).
   internal-stage: a stage ended in no valid way (a signal, another status, a status and a last line
   that disagree), could not be started, or printed more than memory could store.
   internal-covered-line: `covered` returned no line. internal-sums-line: a stage's accounting
   record is missing or has a target below `asked`, or statuses, lines and records disagree on work.
   internal-result-line: a stage's output contains RESULTWORD, the line for programs' first word.
   internal-policy-not-given: a call without -policy came as far as WITHIN LIMITS;
   internal-policy-kind: a verification call came as far under a kind other than exact-match and
   clipped-gap. Some of these tests come after NOT-A-VERIFICATION: docs/format/FORMAT.md,
   "14.3 The results".
   The lines before NO, and before each of the last three endings (WORKLINE, COMMITMENTLINE,
   RESULTWORD: vf_common.h): docs/format/FORMAT.md, "14.3 The results" and "14.7 The output of a
   whole call". Some round may need its work to meet the parameter file's target. A stage then ends
   STAGE PASSED WITH WORK (status 12), and the stages' lines "work:" are repeated.
   A verification call prints COMMITMENTLINE right before WORKLINE, and the line for programs
   (RESULTWORD, vf_common.h) right after it, at these three endings only, not with NO or NOT CHECKED.
   id: each stage has computed the commitment ID. Before the same three endings a verification call
   prints it (without -id: IDHIDDEN in its place), and every call h_par and the line on weight
   hiding. internal-id-two-ways: an ID is missing, or differs from another or from -id. id-named:
   without -id a YES would be about a key file that no one named.
   covered: builds the line "covered:" (`cpCovered`) from the last stage's line "result:", once no
   stage has rejected or failed; printed before the same three endings. Null under -key, -leaf.
   asked: the requested bits. Each stage passes its accounting record here as text
   (vf_accounting_record.h). The last stage's record is the call's: the request stage's covers all
   three files' plans. Proof of work counts toward the target. COMMITMENTLINE: in how many rounds of
   key's and leaf's records the proximity term does not rest on the bound within a third of the distance. */
// clang-format on
static int cpStagesRun(const vector<CpStage> &stages, const char *good, int goodStatus, bool belowBar, double asked, const CpId &id,
                         const std::function<Covered(const string &)> &covered = nullptr) {
  struct Child {
    pid_t pid = -1;
    int fd = -1, status = -1;
    double secs = 0; // from the first fork to the end of the child's output
    string out;
    bool lost = false; // there was no memory to keep all of the child's output
    std::thread reader;
  };
  vector<Child> ch(stages.size());
  // Joins the readers on every return: destroying a joinable std::thread would end the program.
  struct Joined {
    vector<Child> &c;
    ~Joined() {
      for (auto &x : c)
        if (x.reader.joinable()) x.reader.join();
    }
  } joined{ch};
  signal(SIGCHLD, SIG_DFL); // under a caller that ignores the signal, waitpid would find no child
  fflush(stdout);
  fflush(stderr);
  const double t0 = now();
  string broken; // why a stage could not be started
  for (size_t i = 0; i < stages.size() && broken.empty(); i++) {
    int p[2];
    if (pipe(p)) {
      broken = string("pipe: ") + strerror(errno);
      break;
    }
    const pid_t pid = fork();
    if (pid < 0) {
      broken = string("fork: ") + strerror(errno);
      close(p[0]);
      close(p[1]);
      break;
    }
    if (!pid) {
      for (size_t j = 0; j < i; j++) close(ch[j].fd);
      close(p[0]);
      if (dup2(p[1], STDOUT_FILENO) < 0) _exit(2);
      close(p[1]);
      innerThreads = stages[i].threads;
      f2b::calls_begin(stages[i].name); // sets the hash-call count to 0: the parent reports its own
#if BLK_COUNT_HASH_CALLS
      blk::hash_calls_made.store(0, std::memory_order_relaxed); // likewise for the blocks' hash
#endif
      const int s = stages[i].run();
      fflush(stdout);
      exit(s);
    }
    close(p[1]);
    ch[i].pid = pid;
    ch[i].fd = p[0];
  }
  // Reads a child's output to its end; if memory runs out, drops the rest so the child can finish.
  auto readAll = [t0](Child &x) {
    char b[65536];
    ssize_t n;
    while ((n = read(x.fd, b, sizeof b)) > 0 || (n < 0 && errno == EINTR))
      if (n > 0 && !x.lost) {
        try {
          x.out.append(b, (size_t)n);
        } catch (std::bad_alloc &) {
          x.lost = true;
        }
      }
    x.secs = now() - t0;
  };
  // The reader threads start only after the last fork(). A child for which none can be started (a
  // limit on memory or threads) is read by the loop below, on this thread; until then it may block.
  for (auto &x : ch)
    if (x.pid > 0) {
      try {
        x.reader = std::thread(readAll, std::ref(x));
      } catch (std::system_error &) {
      }
    }
  string noWhy; // the rejecting stage's reason, for the line that names the stage
  string no, noStage, failed = broken, failedName = "internal-stage", causes;
  // Each stage's output under its heading. It holds values read from the files (roots, parameters,
  // the proof's leaf number). A caller could copy or trust one from rejected files, or take a
  // stage's "STAGE PASSED" for the call's answer: it is printed only if the call can still pass
  // (mayPass) or under -show-stages, rejecting stages last: no "STAGE PASSED" follows a rejection.
  vector<string> stagesText(ch.size());
  // A verification call without -id prints IDHIDDEN in place of the files' ID: a caller who copied
  // the printed ID into -id would compare the files only with themselves.
  const bool hideId = id.needed && !id.named;
  blk::Accounting call; // the call's record of its accounting: the last stage's
  u64 callsAll = 0;     // the stages' own counts of hash calls, added up
  u64 oneTimeAssumed = 0; // for COMMITMENTLINE and the line for programs
  bool countedAll = true;
  size_t acctsRead = 0;
  string workLines, workPair; // the stages' lines "work:"; the pair of the one with the fewest bits
  double workLeast = 0;
  bool workStage = false; // a stage has ended with STAGE PASSED WITH WORK
  for (size_t i = 0; i < ch.size(); i++) {
    Child &x = ch[i];
    if (x.pid <= 0) continue;
    if (x.reader.joinable())
      x.reader.join();
    else
      readAll(x);
    close(x.fd);
    int w = 0;
    pid_t r;
    while ((r = waitpid(x.pid, &w, 0)) < 0 && errno == EINTR) {
    }
    x.status = r == x.pid && WIFEXITED(w) ? WEXITSTATUS(w) : -1;
    if (x.lost) { // nothing is concluded from a part of a stage's output
      if (failed.empty()) failed = string("no memory for the output of the stage ") + stages[i].name;
      continue;
    }
    // RESULTWORD begins this function's own line. A stage's output that has it is not printed (a
    // rejection name after "NO: " is), and the call ends NO or NOT CHECKED: at status 0 or 14 only
    // that line has the word, and at no ending does another begin with it. No stage's constant text
    // has it; a call name (8 bytes at most, one too few) could, with digits that the program adds.
    const bool holdsWord = x.out.find(RESULTWORD) != string::npos;
    if (holdsWord && failed.empty())
      failed = string("the output of the stage ") + stages[i].name + " holds the first word of the line for programs, which only the call's closing lines may hold",
      failedName = "internal-result-line";
    string last = x.out;
    if (!last.empty() && last.back() == '\n') last.pop_back();
    const size_t nl = last.rfind('\n');
    if (nl != string::npos) last = last.substr(nl + 1);
    blk::Accounting own;
    const bool sumsRead = acctReadH(x.out, own);
    {
      char b[48];
      snprintf(b, sizeof b, ", %.3f s\n", x.secs);
      string &text = stagesText[i];
      text = string("---- stage ") + stages[i].name + " ---- " + std::to_string(stages[i].threads) + " thread(s)" + b;
      string shown = holdsWord ? string("not printed: this output holds the first word of the line for programs\n") : x.out;
      const string ownId = cpLineOf(x.out, STAGEIDLINE).substr(0, HEXW); // the ID that this stage has computed
      if (hideId && ownId.size() == HEXW)
        for (size_t at; (at = shown.find(ownId)) != string::npos;) shown.replace(at, ownId.size(), IDHIDDEN);
      text += shown.c_str();
      if (!shown.empty() && shown.back() != '\n') text += "\n";
    }
    if (sumsRead) {
      callsAll += own.calls, countedAll = countedAll && own.counted;
      // A stage before the last accounts for its own file only. The stages' lines "accounting:
      // one-time files:" have the counts for each file. The counts are printed as read: nothing
      // here can check them.
      if (i + 1 < stages.size()) oneTimeAssumed += own.closeness[1];
      call = own;
      acctsRead++;
      // The stage's record is compared with the requested bits and with its status on work; the
      // latter not for a verification call's last stage, whose record covers all three files.
      if (!((double)own.target >= asked) && failed.empty()) failed = string("the target in the record of the stage ") + stages[i].name + " is below the bits asked of the call", failedName = "internal-sums-line";
      if ((stages.size() == 1 || i + 1 < stages.size()) && (x.status == 3 || x.status == STAGE_PASSED_WITH_WORK_STATUS) && own.needs_work != (x.status == STAGE_PASSED_WITH_WORK_STATUS) && failed.empty())
        failed = string("the status of the stage ") + stages[i].name + " and its record disagree on work", failedName = "internal-sums-line";
    }
    // A stage's line "work: <bits> + <work> (one round's error without its work, in bits, + its bits of work; <round>)". A stage that ends WITH WORK must have one of this form.
    const string wl = cpLineOf(x.out, "work: ");
    bool workRead = false;
    {
      double b = 0;
      unsigned g = 0;
      const size_t par = wl.find(" (");
      if (par != string::npos && sscanf(wl.c_str(), "%lf + %u", &b, &g) == 2 && g >= 1 && g <= WORK_BITS_MOST) {
        workRead = true;
        workLines += string("stage ") + stages[i].name + ": work: " + wl + "\n";
        if (workPair.empty() || b < workLeast) {
          workLeast = b;
          workPair = wl.substr(0, par);
        }
      }
    }
    if (x.status == STAGE_PASSED_WITH_WORK_STATUS) workStage = true;
    if (x.status == 1 && last.rfind("NO: ", 0) == 0 && internalName(last.substr(4))) {
      if (failed.empty()) failed = string("the stage ") + stages[i].name + " met a fault of its own", failedName = last.substr(4);
    } else if (x.status == 1 && last.rfind("NO: ", 0) == 0 && last.size() > 4) {
      if (no.empty()) no = last.substr(4), noStage = stages[i].name, noWhy = cpReasonOf(x.out);
    } else if (x.status == NOT_A_VERIFICATION_STATUS && last == NOT_A_VERIFICATION_TEXT) {
      const string res = cpLineOf(x.out, "result:");
      const size_t c = res.find("(not a verification: ");
      string why = c == string::npos ? "no cause printed" : res.substr(c + 1);
      if (!why.empty() && why.back() == ')') why.pop_back();
      causes += string("stage ") + stages[i].name + ": " + why + "\n";
    } else if (!(x.status == 3 && !workRead && last == string("STAGE PASSED: ") + stages[i].name) &&
               !(x.status == STAGE_PASSED_WITH_WORK_STATUS && workRead &&
                 last == string("STAGE PASSED WITH WORK: ") + stages[i].name) &&
               failed.empty()) {
      failed = string("the stage ") + stages[i].name +
               (r != x.pid        ? string(" could not be waited for: ") + strerror(errno)
                : WIFSIGNALED(w)  ? " was ended by signal " + std::to_string(WTERMSIG(w))
                : x.status == NOT_CHECKED_STATUS ? string(" had no memory or could not start a thread") // main's catches
                                  : " ended with status " + std::to_string(x.status) + " and a last line that does not fit it");
    } else if (!sumsRead && failed.empty() && x.status != 1) {
      failed = string("the stage ") + stages[i].name + " printed no record of its accounting", failedName = "internal-sums-line";
    }
  }
  if (no == "model-statement") {
    printf("id: these are not the files of the commitment that you named; nothing in them was checked\n"
           "id: take the commitment ID from your own record of where it was published, never from these files\n"
           "NO: model-statement\n");
    return 1;
  }
  // False if the call must end NO, NOT CHECKED or NOT-A-VERIFICATION. True: checks still follow.
  const bool mayPass = no.empty() && failed.empty() && causes.empty() && (id.named || !id.needed) && !belowBar && !TEST_BUILD && !(id.noWork && (workStage || call.needs_work));
  if (mayPass || id.showStages)
    for (int refused = 0; refused < 2; refused++)
      for (size_t i = 0; i < ch.size(); i++)
        if (no.empty() ? !refused : (ch[i].status == 1) == (refused == 1)) printf("%s", stagesText[i].c_str());
  printf("---- the call ----\n");
  if (!no.empty()) {
    printf("refused by the stage %s%s\nNO: %s\n", noStage.c_str(), noWhy.c_str(), no.c_str());
    return 1;
  }
  if (!failed.empty()) return cpNotChecked(failed, failedName);
  const Covered about = covered ? covered(cpLineOf(ch.back().out, "result:")) : Covered();
  const string &coveredLine = about.line;
  if (covered && coveredLine.empty())
    return cpNotChecked("the line on the covered positions could not be formed", "internal-covered-line");
  // Each stage has compared its own file's rounds with the target, so no error is compared here.
  if (acctsRead != stages.size()) return cpNotChecked("a stage's record of its accounting is missing", "internal-sums-line");
  // Adds the parent's own hash calls.
  call.counted = countedAll, call.calls = callsAll + f2b::calls_total();
#if BLK_COUNT_HASH_CALLS
  call.calls += blk::hash_calls_counted();
#endif
  // The ID is the first stage's, and every stage must have formed the same.
  const string idHex = cpLineOf(ch[0].out, STAGEIDLINE).substr(0, HEXW);
  if (idHex.size() != HEXW) return cpNotChecked("the first stage printed no ID", "internal-id-two-ways");
  for (auto &x : ch)
    if (cpLineOf(x.out, STAGEIDLINE).substr(0, HEXW) != idHex) return cpNotChecked("two stages formed two IDs", "internal-id-two-ways");
  // With the option a call in which these two disagree would end NOT-A-VERIFICATION below, before the same test there.
  if (id.noWork && causes.empty() && workStage != call.needs_work) return cpNotChecked("the stages' lines on work and the call's record disagree", "internal-sums-line");
  // Each stage has compared it with -id. This function prints the sentence below, so it does too.
  if (id.named != !id.asked.empty() || (id.named && id.asked != idHex)) return cpNotChecked("the stages passed under another ID than the caller's", "internal-id-two-ways");
  if (id.needed && !id.named)
    causes += "the call: not the standard parameter form: id-named (the call named no ID: add -id with the commitment "
              "ID)\n";
  // -no-proof-of-work. The record is asked too: a stage that needs work and ends NOT-A-VERIFICATION has status 14, not 12.
  if (id.noWork && (workStage || call.needs_work))
    causes += "the call: not the standard parameter form: no-proof-of-work (the call had -no-proof-of-work, and " + std::to_string((unsigned long long)call.by_work) +
              (call.by_work == 1 ? " round meets" : " rounds meet") + " the parameter file's target only with proof of work counted)\n";
  // No run of a verifier can know how the masks were made.
  // Names the files as a planning call does (blocks/frame/prints.h), so the outputs are comparable.
  blk::print_accounting(call, stages.size() > 1 ? "key, entry and proof" : !strcmp(stages[0].name, "key") ? "the key" : "an entry");
  printf("accounting: asked of this call: %g bits\n", asked);
  // -key and -leaf print no IDLINE: a committer that copies their output still prints at most one.
  const string idLine = covered ? string(IDLINE) + (hideId ? IDHIDDEN : idHex.c_str()) + "\n" : "";
  printf("%s%sid: %s\nh_par=%s\n%s"
         "weight hiding: NOT CHECKED BY THIS CALL (it rests on how the commitment was made)\n%s%s",
         causes.c_str(), idLine.c_str(), id.named ? "named by the caller and equal" : "not compared: the call named none", id.hPar.c_str(),
         stages.size() == 1 && !strcmp(stages[0].name, "leaf") ? "key: NOT CHECKED BY THIS CALL (its table was taken on its form alone; the part-call -key or a whole call checks it)\n" : "",
         coveredLine.c_str(), workLines.c_str());
  unsigned mostWork = 0;
  for (auto &kind : call.work)
    for (unsigned g : kind) mostWork = std::max(mostWork, g);
  char workLine[200];
  if (call.needs_work)
    snprintf(workLine, sizeof workLine, WORKLINE "up to %u bits counted in %llu round%s; with it the least round has %.2f bits, without it %.2f bits", mostWork, (unsigned long long)call.by_work,
             call.by_work == 1 ? "" : "s", blk::cut2(call.groups[0].all()), blk::cut2(call.lists.bits));
  else
    snprintf(workLine, sizeof workLine, WORKLINE "none counted");
  string commitmentLine; // the line on the commitment's own files; a part-call has none
  if (covered)
    commitmentLine = COMMITMENTLINE + std::to_string(oneTimeAssumed) + (oneTimeAssumed == 1 ? " closeness round of the one-time files rests" : " closeness rounds of the one-time files rest") +
                     " on another source than the bound within a third of the distance" RECORDCLAUSE "\n";
  // The line for programs. Each value is a number computed or parsed here, hex digits or a constant
  // word: no blank or line feed. id=: the stages' ID, only if equal to -id, which cpMain() tested
  // to be hex digits. Of the request: what "covered:" shows, no token or digest of tokens.
  auto resultLine = [&](int status) {
    if (!covered) return;
    printf(RESULTWORD " status=%d claim=%s id=%s bits=%.0f work=%u work-rounds=%llu least-round=%.2f least-round-without-work=%.2f commitment-assumed-rounds=%llu record=1 %s "
                      "stages=key,leaf,request dw=%u build=%s\n",
           status, about.claim, hideId ? IDHIDDEN : idHex.c_str(), asked, call.needs_work ? mostWork : 0u, (unsigned long long)(call.needs_work ? call.by_work : 0),
           blk::cut2(call.groups[0].all()), blk::cut2(call.lists.bits), (unsigned long long)oneTimeAssumed, about.fields.c_str(), DW, VF_SOURCE_DIGEST);
  };
  // Each stage has checked belowBar and TEST_BUILD for itself; both are checked again here.
  if (!causes.empty() || belowBar || TEST_BUILD) {
    printf("%s%s\n", commitmentLine.c_str(), workLine);
    resultLine(NOT_A_VERIFICATION_STATUS);
    printf("%s\n", NOT_A_VERIFICATION_TEXT);
    return NOT_A_VERIFICATION_STATUS;
  }
  // Status 12 (WITH WORK) and a valid line "work:" must come together.
  if (workStage != !workPair.empty()) return cpNotChecked("a stage's status and its line on work disagree", "internal-sums-line");
  // The call's record must agree, on `workPair` too. Both bit figures were truncated: within 0.01.
  const size_t plus = workPair.find(" + ");  // sscanf above takes "96+32" too; substr(npos) throws
  if (workStage != call.needs_work || (workStage && (plus == string::npos || !(fabs(workLeast - call.lists.bits) < 0.0101) || workPair.substr(plus) != " + " + std::to_string(call.lists.work))))
    return cpNotChecked("the stages' lines on work and the call's record disagree", "internal-sums-line");
  // WITHIN LIMITS is only for a caller that passed -policy. Without it cpPolicyHeld() rejects the
  // call, and the request stage would end with status 14. This third check guards the return.
  if (about.otherKind && !id.policyGiven) return cpNotChecked("a call without -policy came as far as WITHIN LIMITS", "internal-policy-not-given");
  // Likewise the kind: cpPolicyHeld() tested it on policy.txt, this on the request stage's values.
  if (covered && !about.kindTaken) return cpNotChecked("the parameter file's policy is neither exact-match nor clipped-gap, and no earlier check refused it", "internal-policy-kind");
  if (about.otherKind) {
    printf("%s%s\n", commitmentLine.c_str(), workLine);
    resultLine(WITHINSTATUS);
    printf("%s%s\n", WITHIN, about.policy.c_str());
    return WITHINSTATUS;
  }
  printf("%s%s\n", commitmentLine.c_str(), workLine);
  resultLine(goodStatus);
  printf("%s\n", good);
  return goodStatus;
}

/* Builds the line "covered:", which states what the call's result is about: positions r to e - 1 of
   the caller's request, under the acceptance policy of the caller's parameter file, and nothing
   else. Its fields, in order, and what each means: docs/format/FORMAT.md, "14.7 The output of a
   whole call". above-clip=: an upper bound on how many of these tokens can cost more than C
   (`aboveClip`, vf_common.h); each such token can be any token. It is no bound on the tokens that
   can be any token.
   result: the text after "result:" in the request stage's output. C, theta, cap, s, sh, sT, F are
   read from it: the values that the stage used. T is the request's length less 2; the stage has
   checked that length and r < e <= T ("statement-range"). Returns no line unless r < e <= T and all
   seven are numbers.
   leaf: the proof's leaf number, the last of `fields`, which is built here from the same values. */
static Covered cpCovered(const vector<u32> &request, u32 leaf, const string &result) {
  if (request.size() < 3) return {};
  const u64 T = request.size() - 2, r = request[T], e = request[T + 1];
  if (r >= e || e > T) return {};
  // v: the number after " <name>="; false unless 1 to 18 digits, then a space.
  auto field = [&result](const char *name, u64 &v) {
    const string key = string(" ") + name + "=";
    const size_t at = result.find(key);
    if (at == string::npos) return false;
    size_t i = at + key.size();
    const size_t first = i;
    for (v = 0; i < result.size() && result[i] >= '0' && result[i] <= '9' && i - first < 18; i++)
      v = 10 * v + (u64)(result[i] - '0');
    return i > first && i < result.size() && result[i] == ' ';
  };
  u64 C = 0, theta = 0, cap = 0, s = 0, sh = 0, sT = 0, F = 0;
  if (!field("C", C) || !field("theta", theta) || !field("cap", cap) || !field("s", s) || !field("sh", sh) ||
      !field("sT", sT) || !field("F", F))
    return {};
  auto n = [](u64 v) { return std::to_string(v); };
  const u64 last = std::min(e, T - 1); // the request's last checked token, if r < T - 1
  Covered o;
  o.policy = string(policyKind(C, theta, cap, s, sh, sT)) + " C=" + n(C) + " theta=" + n(theta) + " cap=" + n(cap) + " s=" + n(s) + " sh=" + n(sh);
  o.otherKind = r + 1 <= last && !acceptancePolicyIsExactMatch(C, theta, cap, sh, sT);
  o.claim = r + 1 > last ? "next-token" : o.otherKind ? "policy" : "exact-match";
  o.fields = "r=" + n(r) + " e=" + n(e) + " T=" + n(T) + " tokens=" + (r + 1 <= last ? n(r + 1) + ".." + n(last) : string("none")) + " y=" + (e == T ? "yes" : "no") +
             " policy=" + policyKind(C, theta, cap, s, sh, sT) + " C=" + n(C) + " theta=" + n(theta) + " cap=" + n(cap) + " s=" + n(s) + " sh=" + n(sh) + " sT=" + n(sT) + " F=" + n(F) +
             " above-clip=" + n(aboveClip(last - r, C, theta, cap, sh)) + " leaf=" + n(leaf);
  o.kindTaken = sT + 4 == sh;
  o.line = "covered: r=" + n(r) + " e=" + n(e) + " T=" + n(T) + " C=" + n(C) + " theta=" + n(theta) + " cap=" + n(cap) +
         " s=" + n(s) + " sh=" + n(sh) + " positions=" + n(r) + ".." + n(e - 1) +
         " tokens=" + (r + 1 <= last ? n(r + 1) + ".." + n(last) : string("none")) + " y=" + (e == T ? "yes" : "no") +
         " next-token=" + (r + 1 == T ? "yes" : "no") + " policy=" + policyKind(C, theta, cap, s, sh, sT) +
         " above-clip=" + n(aboveClip(last - r, C, theta, cap, sh)) +
         " (places in the request, counted from 0; y=yes: the answer is about y too; it is about nothing else; above-clip: "
         "so many of the tokens named by tokens= may cost more than C, and such a token may be any token. It is no limit on "
         "the tokens that may be any token: one that costs C or less may be any token too where the committed model's "
         "logits lie close together, and no verifier sees that. All costs, each counted as at most C, add up to at most "
         "theta times the number of these tokens)\n";
  return o;
}

// For cpUsage()'s last line. The parsers: parseRequestedBits(), decimal() (vf_common.h).
#define ASKED_FORM "<asked>, the requested bits, is a whole number from 0 to 1000000, in digits alone, without a leading zero"
#define THREADS_FORM "after -j comes a whole number from 1 to 256, in digits alone, without a leading zero"
// fault: what is wrong with the command line; it is printed as the last line, below the text. It
// names an argument by its number, never by its text: cp_verify prints no text from argv.
static int cpUsage(const string &fault) {
  fprintf(
      stderr,
      "usage: cp_verify <config folder> <ivk folder> <request file> <y file> <proof file> -id <%u hex> [-bits <asked>]\n"
      "    [-no-proof-of-work] [-j <threads>] [-q] [-show-stages] [-next-token]\n"
      "    [-policy <kind>,<C>,<theta>,<cap>,<s>,<sh>]\n"
      "  one call checks ivk's key.bin, the entry ivk/leaf/<k>.bin that the proof names, and the proof\n"
      "  -id: the commitment's ID, which names the key's table and the three files of the config folder together. Take it\n"
      "    from where the commitment is published, or from whoever vouches for the model, before you send your request,\n"
      "    and never from the files that you were sent. Each stage forms the ID of the files, once it has planned from the\n"
      "    config folder, and compares it before it looks at anything else of key, entry, request or proof; files with\n"
      "    another ID end NO: model-statement, and their own ID is not printed.\n"
      "    Without -id every stage runs its other checks and the call ends at best with NOT-A-VERIFICATION: a YES about a\n"
      "    key that nobody has named would say nothing. Such a call prints not-shown-without-id in place of the files' ID\n"
      "  <config folder>: your copy of policy.txt, KL.bin and layout.txt, which in practice the provider writes and -id\n"
      "    names; without layout.txt every call ends NO: plan-layout, except one that lists the arrays (-plan arrays,\n"
      "    -plan request -arrays)\n"
      "  <request file>: exactly T = 2^lt token ids, then r and e (r < e <= T); <y file>: one token id; 32-bit\n"
      "    little-endian words. The proof covers positions r to e - 1, and position t is about token t + 1. It states\n"
      "    that tokens r + 1 to min(e, T - 1) meet the acceptance policy set by C, theta and cap in the parameter file\n"
      "    and, if e = T, that y is the argmax of the last position. If e < T, y must be 0. The answer is about these\n"
      "    positions alone: it says nothing of any other token and, if e < T, nothing of y. The line \"covered:\" near the\n"
      "    end repeats r and e: see that they are the ones you mean\n"
      "  -next-token: the request must have r = T - 1 and e = T (the proof is about y alone); with other r, e the call\n"
      "    ends at best with NOT-A-VERIFICATION\n"
      "  -policy: the acceptance policy that you expect. <kind> is exact-match (C = theta = 2^sh, cap = 0, sT = sh - 4:\n"
      "    every covered token is the model's first choice) or clipped-gap (sT = sh - 4, other values); the five\n"
      "    numbers are the lines C, theta, cap, s and sh, in decimal. The last line is NO: policy\n"
      "    if your parameter file sets another policy than the named one or a kind other than these two (sT is not\n"
      "    sh - 4), if none is named and a token of the request is covered under a policy other than exact-match, or if\n"
      "    arithmetic on the named numbers and the request's T, r and e shows that a result would mean nothing; the line\n"
      "    before the last says why. Under clipped-gap an answer can say less, down to nothing, and the last line of a\n"
      "    passing call that covers a token of the request is \"WITHIN LIMITS: ..\", not YES. A covered token has a cost:\n"
      "    its gap plus 2^sh. The gap is the largest, over all token ids v, of v's logit minus the token's, plus 1 if v is\n"
      "    the lower id: a token that ties for the largest logit with a lower id has the gap 1, not 0. The answer says that\n"
      "    the costs, each counted as at most C, have a mean of at most theta, and that at most cap of them lie above C.\n"
      "    The line \"covered:\" repeats the five numbers, names the kind that the file sets (policy=) and says how many\n"
      "    of the covered tokens may cost more than C (above-clip=). Such a token may be any token, and so may others,\n"
      "    where the model's logits lie close together: the number is no limit on them\n"
      "  -bits: the requested bits, 0 to 1000000; without it %.0f. " REQUESTED_BITS_RULE_USAGE_TEXT "\n"
      "    Below %.0f every stage runs its checks and the call ends at best with NOT-A-VERIFICATION\n"
      "  -no-proof-of-work: if some round meets the parameter file's target only with its proof of work counted, every stage\n"
      "    runs its checks and the call ends at best with NOT-A-VERIFICATION. Work that the file declares and no round needs\n"
      "    changes nothing. The option changes no number of that file: it holds under whatever parameter file\n"
      "    is given\n"
      "  -j: 1 to 256 threads for the request's stage; the two shorter stages take an eighth of that each beside it\n"
      "  -q: leave out most progress lines\n"
      "  -show-stages: print the stages' output also if the call does not pass. After a rejection it holds values read from\n"
      "    the rejected files: use it to find out which check failed, and trust or copy nothing in it\n"
      "  last line and status:\n"
      "    YES, 0\n"
      "    WITHIN LIMITS: <the policy>, %d: as YES, with a token of the request checked under the policy that you named,\n"
      "      other than exact-match. It is no YES; the status is the same, so tell the two apart by the last line\n"
      "    NO: <name>, 1\n"
      "    NOT-A-VERIFICATION, %d\n"
      "    a bad command line (this text, whose last line says what was not accepted, or a sentence on -id or -policy), or a\n"
      "      line that begins input-unreadable- (a file could not be read), 2\n"
      "    no last line, %d: the answer could not be written to standard output\n"
      "    NOT CHECKED: internal-<..> or <sub-proof>:internal-<..>, %d: a call that ends so has verified nothing. Under\n"
      "      internal-stage and internal-exception-parFor the machine may be the cause (a want of memory, for one): run it\n"
      "      again. If it ends so again, the cause may be the program: a guard of the library's ends a stage by a signal,\n"
      "      which goes under internal-stage, and leaves its line on standard error.\n"
      "      Under any other name a guard of the verifier's own spoke, and on the same files the call ends so again\n"
      "    Before YES, WITHIN LIMITS, PART PASSED and NOT-A-VERIFICATION stands the line \"" WORKLINE "..\", which says\n"
      "    whether some round holds at the parameter file's target (sec) only with its proof of work: in how many rounds, the\n"
      "    most bits of work that the file declares, and the bits of the round with the largest error, once with the work\n"
      "    counted and once with it left out (not always the same round). Each is a single round's figure, and neither is a\n"
      "    security level. If you want the checks alone to carry the target, pass -no-proof-of-work\n"
      "    In a whole call the line \"" COMMITMENTLINE "..\" stands right before the line on proof of work: how many closeness rounds\n"
      "    of the commitment's own files, key and entry, rest on another source than the bound within a third of the distance\n"
      "    In a whole call one line for programs, which begins \"" RESULTWORD " \", stands between the line on proof of work and the\n"
      "    last line. It says what the answer is about, as named fields in a fixed order. Exit status 0 alone is no reason to\n"
      "    accept an answer: compare the fields with your own values (the verifier's README lists them and what to compare)\n"
      "    If a stage rejects, the call prints only a heading, that stage's name with the check that failed, and\n"
      "    NO: <name>, not the stages' output (-show-stages prints it); under another ID, two lines \"id:\" and the NO\n"
      "parts, for the committer's own test; they never say YES (PART PASSED: key or leaf, status 3):\n"
      "  cp_verify -key <config folder> <key.bin> [-id <%u hex>] [-bits <asked>] [-no-proof-of-work] [-j <threads>]\n"
      "    [-q] [-show-stages]\n"
      "  cp_verify -leaf <config folder> <key.bin> <entry file> [-id <%u hex>] [-bits <asked>] [-no-proof-of-work]\n"
      "    [-j <threads>] [-q] [-show-stages]\n"
      "    -leaf takes the key's table from key.bin's first bytes, on its form alone, and looks at none of the key's\n"
      "    proofs: its pass counts only beside a -key that passed on the same key.bin, config folder, bits and build\n"
      "plans, which check nothing (PLAN ONLY, status 4):\n"
      "  cp_verify -plan id <config folder> <key.bin>\n"
      "    prints, for the publisher, the line \"" IDLINE "..\" for these files, and h_par=, the digest of the\n"
      "    three files of the config folder alone (-plan request prints h_par= too). If the files were sent to you, do\n"
      "    not pass that ID as -id\n"
      "  cp_verify -plan request <policy.txt> <KL.bin> <asked> [-layout <layout.txt>]\n"
      "    -prompt <request file> -reply <y file> [-shapes] [-fields] [-arrays] [-j <threads>] [-q]\n"
      "  cp_verify -plan accounting <config folder> <asked>\n"
      "    prints the accounting of a whole call under these files (lines \"accounting:\") and the greatest lengths of key,\n"
      "    entry and proof; the provers print it through this call\n"
      "  a plan of request or accounting prints a line \"work: <bits> + <work> (..)\" if all rounds hold at the parameter\n"
      "    file's target (sec) and some round only with its proof of work, whatever <asked> is: a checking call under\n"
      "    these files will count proof of work\n"
      "  cp_verify -plan arrays <config folder> <asked>\n"
      "    lists the arrays that a request's sub-proofs commit to (lines \"array:\"), from policy.txt and KL.bin alone, for\n"
      "    the tool that makes layout.txt (prover/tools/pack_layout.py --arrays)\n"
      "research code: nothing is proved about it and no security level is claimed\n",
      (unsigned)HEXW, BAR, BAR, WITHINSTATUS, NOT_A_VERIFICATION_STATUS, NOTWRITTENSTATUS, NOT_CHECKED_STATUS, (unsigned)HEXW, (unsigned)HEXW);
  fprintf(stderr, "cp_verify: this command line was not accepted: %s\n", fault.c_str());
  return 2;
}

// The calls "cp_verify -plan ..". They check nothing.
static int cpPlanMain(int argc, char **argv) {
  if (argc < 3) return cpUsage("-plan wants id, request, accounting or arrays");
  const string what = argv[2];
  double asked = 0;
  if (what == "id") {
    if (argc != 5) return cpUsage("-plan id wants exactly <config folder> <key.bin>");
    const CpConf conf = cpConfRead(argv[3]);
    string keyB;
    needBytes(argv[4], keyB, "key");
    // oneTime = 3: after its plans runStage() prints h_par= and IDLINE and ends.
    {
      CallInputs a;
      a.keyB = &keyB;
      const SealedCallInputs sealed(std::move(a));
      StageInputs in;
      in.conf = &conf;
      in.sealed = &sealed;
      in.planOnly = in.quiet = true;
      in.oneTime = 3;
      return runStage(in);
    }
  }
  if (what == "accounting" || what == "arrays") {
    // The request is made up, T tokens 0
    // with r = 0 and e = T: the accounting depends only on the configuration files. If the line lt
    // is missing or above 20 the request stays empty; runStage() rejects such a parameter file.
    if (argc != 5 || !parseRequestedBits(argv[4], asked)) return cpUsage("-plan accounting and -plan arrays want exactly <config folder> <asked>; " ASKED_FORM);
    const CpConf conf = cpConfRead(argv[3]);
    CallInputs about;
    u64 lt = 0;
    if (polWord(conf.pol, "lt", lt) && lt <= 20) {
      about.request.assign(((size_t)1 << lt) + 2, 0);
      about.request.back() = (u32)1 << lt;
    }
    about.y.assign(1, 0);
    const SealedCallInputs sealed(std::move(about));
    StageInputs made;
    made.conf = &conf;
    made.sealed = &sealed;
    made.asked = asked;
    made.planOnly = made.quiet = true;
    made.arraysAsked = what == "arrays";
    return runStage(made);
  }
  if (what != "request") return cpUsage("-plan wants id, request, accounting or arrays");
  if (argc < 6 || !parseRequestedBits(argv[5], asked)) return cpUsage("-plan request wants <policy.txt> <KL.bin> <asked>; " ASKED_FORM);
  const char *layoutPath = nullptr, *promptPath = nullptr, *replyPath = nullptr, *jArg = nullptr;
  StageInputs in;
  CallInputs about;
  in.asked = asked;
  in.planOnly = true;
  for (int i = 6; i < argc; i++) {
    auto value = [&](const char *&into) {
      if (into || i + 1 >= argc) return false;
      into = argv[++i];
      return true;
    };
    const string a = argv[i];
    bool ok = true;
    if (a == "-layout")
      ok = value(layoutPath);
    else if (a == "-prompt")
      ok = value(promptPath);
    else if (a == "-reply")
      ok = value(replyPath);
    else if (a == "-j")
      ok = value(jArg);
    else if (a == "-shapes")
      in.shapesAsked = true;
    else if (a == "-fields")
      in.fieldsAsked = true;
    else if (a == "-arrays")
      in.arraysAsked = true;
    else if (a == "-q")
      in.quiet = true;
    else
      ok = false;
    if (!ok) return cpUsage("argument " + std::to_string(i) + " is not an option of -plan request, comes a second time, or lacks its value");
  }
  u32 threads = 1;
  if (jArg && !decimal(jArg, 256, threads)) return cpUsage(THREADS_FORM);
  in.nthreads = (int)threads;
  if (in.nthreads < 1 || !promptPath || !replyPath) return cpUsage("-plan request wants -prompt and -reply; " THREADS_FORM);
  innerThreads = in.nthreads;
  CpConf conf;
  needBytes(argv[3], conf.pol, "policy");
  bool odd = false, oP = false, oQ = false;
  mustReadWords(argv[4], conf.KL, odd, "KL");
  if (odd) refuse("KL-length");
  conf.hasLayout = layoutPath != nullptr;
  if (layoutPath) needBytes(layoutPath, conf.layoutB, "layout");
  mustReadWords(promptPath, about.request, oP, "request");
  mustReadWords(replyPath, about.y, oQ, "y");
  about.oddStatement = oP || oQ;
  const SealedCallInputs sealed(std::move(about));
  in.conf = &conf;
  in.sealed = &sealed;
  return runStage(in);
}

// Returns the version word of the parameter file's first line, `version`. Rejects unless
// checkVersionLine() (vf_common.h) admits it.
static u32 cpPolicyWord(const string &pol) {
  const bool has = !pol.compare(0, 8, "version ") && pol.size() > 12 && pol[12] == '\n';
  const u32 v = has ? blk::version_from_text(pol.substr(8, 4)) : 0;
  checkVersionLine(has, v);
  return v;
}
// Rejects, without reading it, a regular file that is not 4 * words bytes long. Other
// paths are left to the read.
static void cpLengthHeld(const char *path, u64 words) {
  struct stat sb;
  if (stat(path, &sb) || !S_ISREG(sb.st_mode) || (u64)sb.st_size == 4 * words) return;
  refuse(!sb.st_size || sb.st_size % 4 ? "statement-required" : "statement-range");
}
[[noreturn]] static void cpPolicyRefused(const string &why) {
  printf("policy: %s\n", why.c_str());
  refuse("policy");
}
// Rejects ("policy") if policyEmpty() (vf_common.h) is not 0 for the policy that -policy names.
// n of the request's T tokens are checked.
static void cpPolicySays(const PolicyNamed &p, u64 n, u64 T) {
  static const char *const why[] = {"", "C is below 2^sh, the least cost of a token, so no cost is read",
                                    "theta is below 2^sh, the least cost of a token, so no request can meet it",
                                    "theta is above C, which no parameter file may set",
                                    "theta = C and cap is not below the number of the request's tokens that are checked, so none of them is barred"};
  const int kind = p.given ? policyEmpty(n, p.C, p.theta, p.cap, p.sh) : 0;
  if (!kind) return;
  cpPolicyRefused(string("a result under the named policy would mean nothing: ") + why[kind] +
                  (kind == 4 ? "; for the output token alone send a request with r = " + std::to_string(T - 1) : ""));
}
// Compares the caller's policy (-policy; else exact-match) with parameter file and request.
// No stage has checked the two. If a line or word is missing or
// malformed, a stage rejects. The request stage repeats two tests: exact-match, policy-named.
static void cpPolicyHeld(const PolicyNamed &p, const string &pol, const vector<u32> &request) {
  u64 C = 0, theta = 0, cap = 0, s = 0, sh = 0, sT = 0, lt = 0;
  if (!polWord(pol, "C", C) || !polWord(pol, "theta", theta) || !polWord(pol, "cap", cap) || !polWord(pol, "s", s) || !polWord(pol, "sh", sh) || !polWord(pol, "sT", sT))
    return;
  // A verification call accepts only exact-match and clipped-gap. The stages' code can still check
  // under a table of exponentials (sT other than sh - 4), but no such call checks a token that way.
  if (sT + 4 != sh)
    cpPolicyRefused(string("the parameter file sets the kind of policy ") + policyKind(C, theta, cap, s, sh, sT) + "; this program takes exact-match and clipped-gap");
  if (p.given && !p.is(C, theta, cap, s, sh, sT))
    cpPolicyRefused("the option -policy names another kind of policy or another C, theta, cap, s or sh than the parameter file sets");
  if (!polWord(pol, "lt", lt) || lt > 20 || request.size() != (1ull << lt) + 2) return;
  const u64 T = 1ull << lt, r = request[T], e = request[T + 1], last = std::min(e, T - 1);
  if (r >= e || e > T) return;
  const u64 n = r < last ? last - r : 0;
  if (!p.given && n && !acceptancePolicyIsExactMatch(C, theta, cap, sh, sT))
    cpPolicyRefused("the request covers more than the output token, the parameter file sets another policy than exact-match, and the call has no -policy");
  cpPolicySays(p, n, T);
}
/* Parses the command line, reads every input file once and runs the stages.
   It stores nothing and takes no record of an earlier call from its caller. */
static int cpMain(int argc, char **argv) {
  testBuildLine(); // prints only in a test build (vf_common.h, TEST_BUILD)
  if (argc > 1 && !strcmp(argv[1], "-plan")) return cpPlanMain(argc, argv);
  const bool keyPart = argc > 1 && !strcmp(argv[1], "-key"), leafPart = argc > 1 && !strcmp(argv[1], "-leaf");
  const int first = keyPart || leafPart ? 2 : 1, places = keyPart ? 2 : leafPart ? 3 : 5;
  if (argc < first + places) return cpUsage("too few arguments: this call wants " + std::to_string(places) + " file and folder names");
  char **const at = argv + first;
  const char *bitsArg = nullptr, *jArg = nullptr, *idArg = nullptr;
  bool quiet = false, nextTokenAsked = false, showStages = false, noWork = false;
  PolicyNamed policy;
  for (int i = first + places; i < argc; i++) {
    if (!strcmp(argv[i], "-bits") && !bitsArg && i + 1 < argc)
      bitsArg = argv[++i];
    else if (!strcmp(argv[i], "-id") && !idArg && i + 1 < argc)
      idArg = argv[++i];
    else if (!strcmp(argv[i], "-j") && !jArg && i + 1 < argc)
      jArg = argv[++i];
    else if (!strcmp(argv[i], "-q"))
      quiet = true;
    else if (!strcmp(argv[i], "-show-stages"))
      showStages = true;
    else if (!strcmp(argv[i], "-no-proof-of-work"))
      noWork = true;
    else if (!strcmp(argv[i], "-next-token") && !keyPart && !leafPart)
      nextTokenAsked = true;
    else if (!strcmp(argv[i], "-policy") && !policy.given && !keyPart && !leafPart && i + 1 < argc) {
      if (!policyNamedRead(argv[++i], policy)) {
        fprintf(stderr, "%s\n", POLICY_OPTION_FORM_TEXT);
        return 2;
      }
    } else
      return cpUsage("argument " + std::to_string(i) + " is not an option of this call, comes a second time, or lacks its value");
  }
  double asked = BAR;
  u32 threads = 1;
  if (bitsArg && !parseRequestedBits(bitsArg, asked)) return cpUsage("after -bits comes <asked>; " ASKED_FORM);
  if (jArg && (!decimal(jArg, 256, threads) || !threads)) return cpUsage(THREADS_FORM);
  string idAsked = idArg ? idArg : "";
  for (char &c : idAsked)
    if (c >= 'A' && c <= 'F') c = (char)(c - 'A' + 'a');
  if (idArg && (idAsked.size() != HEXW || idAsked.find_first_not_of("0123456789abcdef") != string::npos)) {
    fprintf(stderr, "-id requires %u hex digits\n", (unsigned)HEXW);
    return 2;
  }
  const int nthreads = (int)threads;
  const bool belowBar = askedBelowBar(asked);
  // No stage can reach more than half a digest's bits (vf_common.h, DIGEST_HALF_BITS).
  if (asked > DIGEST_HALF_BITS) refuse("cp-bits-above-digest");
  cpPolicySays(policy, 0, 0); // checks -policy by its numbers only, before any file is opened
  innerThreads = nthreads; // readWords reads a large file on several threads

  string keyB, entryB, proofB;
  CallInputs about; // what the call is about; the stage request seals it
  const CpConf conf = cpConfRead(at[0]);
  needBytes(keyPart || leafPart ? at[1] : (string(at[1]) + "/key.bin").c_str(), keyB, "key");
  /* hPar alone is computed here; each stage computes the commitment ID:
     docs/format/FORMAT.md, "13.5 The commitment ID". */
  CpId id;
  id.hPar = hex64(hParOf(conf.pol, conf.KL, conf.layout()).w);
  id.named = idArg != nullptr, id.asked = idAsked;
  id.needed = !keyPart && !leafPart;
  id.policyGiven = policy.given, id.showStages = showStages, id.noWork = noWork;
  const u32 polVer = cpPolicyWord(conf.pol);
  cpKeyHead(keyB, polVer);
  if (leafPart) {
    needBytes(at[2], entryB, "entry");
  } else if (!keyPart) {
    bool oP = false, oQ = false;
    u64 lt = 0;
    if (polWord(conf.pol, "lt", lt) && lt <= 20) cpLengthHeld(at[2], (1ull << lt) + 2); // without a line lt that gives T every stage rejects the call
    cpLengthHeld(at[3], 1);
    mustReadWords(at[2], about.request, oP, "request");
    mustReadWords(at[3], about.y, oQ, "y");
    about.oddStatement = oP || oQ;
    if (!about.oddStatement) cpPolicyHeld(policy, conf.pol, about.request);
    needBytes(at[4], proofB, "proof");
    const u32 k = cpProofHead(proofB, polVer);
    about.entry = k;
    if (!readWholeFile((string(at[1]) + "/leaf/" + std::to_string(k) + ".bin").c_str(), entryB))
      refuse("cp-entry-missing"); // k is the proof header's: a rejection, not an unreadable input
  }

  /* The stages, each `runStage` in its own process: key checks key.bin; leaf checks
     the leaf file, trusting key.bin's table; request checks the proof, trusting the leaf file's. A
     YES needs all three. */
  {
    auto stageOn = [&](const char *name, int oneTime, int threads) {
      return CpStage{name, threads, [&, oneTime]() {
                         CallInputs a;
                         if (!oneTime) printf("leaf number %u\n", (unsigned)about.entry); // in the stage's output; `stagesText` says when that is printed
                         if (!oneTime) a = about, a.proofB = &proofB, a.nextTokenAsked = nextTokenAsked, a.policy = policy;
                         a.keyB = &keyB;
                         if (oneTime != 1) a.entryB = &entryB;
                         a.idAsked = idAsked;
                         const SealedCallInputs sealed(std::move(a));
                         StageInputs in;
                         in.conf = &conf;
                         in.sealed = &sealed;
                         in.planOnly = oneTime != 0; // the stages key and leaf have no proof file
                         in.oneTime = oneTime;
                         in.asked = asked;
                         in.quiet = quiet;
                         in.nthreads = innerThreads;
                         return runStage(in);
                       }};
    };
    if (keyPart) return cpStagesRun({stageOn("key", 1, nthreads)}, "PART PASSED: key", 3, belowBar, asked, id);
    if (leafPart) return cpStagesRun({stageOn("leaf", 2, nthreads)}, "PART PASSED: leaf", 3, belowBar, asked, id);
    const int few = std::max(1, nthreads / 8);
    return cpStagesRun({stageOn("key", 1, few), stageOn("leaf", 2, few), stageOn("request", 0, nthreads)}, "YES", 0, belowBar, asked, id,
                         [&](const string &result) { return cpCovered(about.request, about.entry, result); });
  }
}
#ifdef CP_VERIFY_AS_A_UNIT
// Gives another program the blk::Configured that a stage builds from these files: cp_configured.h.
int cp_with_configured(const std::string &policy_txt, const std::string &layout_txt, const std::vector<uint32_t> &KL_words, double asked_bits, bool quiet,
                       const std::function<int(const blk::Configured &)> &then) {
  (void)cpMain; // this build offers no verifier calls; avoids an unused-function warning
  static bool called = false; // `runStage` sets statics: one call per process, none inside `then`
  if (called) refuse("internal-configured-twice");
  called = true;
  CpConf conf;
  conf.pol = policy_txt, conf.layoutB = layout_txt, conf.hasLayout = true, conf.KL = KL_words;
  StageInputs in;
  in.conf = &conf;
  in.then = &then;
  in.planOnly = true;
  in.asked = asked_bits;
  in.quiet = quiet;
  return runStage(in);
}
#else
static int cpNotWritten(int why) {
  fprintf(stderr, "cp_verify: the answer could not be written to standard output%s%s: take this call as not made\n", why ? ": " : "", why ? strerror(why) : "");
  return NOTWRITTENSTATUS;
}
// Returns status if everything printed so far has reached standard output, else NOTWRITTENSTATUS.
// main() passes every status that cpMain() returns through it. A rejection that exits where it is
// found (refuse(), vf_common.h) keeps status 1, an unreadable input 2: nobody takes these for a pass.
static int cpWritten(int status) {
  errno = 0;
  return !ferror(stdout) && !fflush(stdout) ? status : cpNotWritten(errno);
}
int main(int argc, char **argv) {
  // With descriptor 1 closed, the first file or pipe that the program opens would get that number.
  if (fcntl(STDOUT_FILENO, F_GETFD) < 0) return cpNotWritten(errno);
#ifdef __linux__
  // Disables core dumps of this process and its stages: the committer runs this program on files
  // not yet published, and its RLIMIT_CORE has no effect where core dumps are piped to a program.
  if (prctl(PR_SET_DUMPABLE, 0)) return cpNotChecked("core dumps of this process could not be switched off (prctl)");
#endif
  try {
    return cpWritten(cpMain(argc, argv));
  } catch (std::bad_alloc &) { // in a stage's process too: the parent then sees exit status 5
    return cpNotChecked("no memory");
  } catch (std::system_error &) {
    return cpNotChecked("a thread could not be started (a limit on threads or on memory)");
  }
}
#endif
