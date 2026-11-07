// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// Sizes of a lookup forest, the product trees of a group of NC lookup calls (ForestPlan), and a
// lookup table as GPU kernels read it (fk::Table). Include after prover/hiding/product_test.cuh.
// blocks/iop/lkp.h defines the argument; verifier/src/vf_lookup_argument.h checks it (its NT is NR here).
// Key side: tree b of a call has the call's lookup keys b * 2^dT .. (b + 1) * 2^dT - 1. The calls'
// trees are numbered y < NR, larger calls first; trees NR .. NT - 1, NT = 2^rK, have only call 0's
// padding key. Flat index (x | y): x inside a tree, high dT bits; y, low rK bits; v = dT + rK.
// Table side: row index (t | c): row t of call c's table, high kT bits (a shorter table repeats);
// c, low cK bits; d = kT + cK. Numbers c = NC .. 2^cK - 1 repeat the last call's table and count no
// key. Plane j < 32, index (t | c | j): bit j of a row's multiplicity (count of non-padding keys).
// Forest (prover/hiding/forest.cuh), the engine, sends one element per message slot. In order: the
// tie of the two sides' roots | v key layers | d table layers | table leaves | bits are 0 or 1.
#pragma once
#include "common/end_name.h" // name_the_end (M5g)
#include <algorithm>
#include <numeric>
#include <utility>
#include <vector>
// One lookup call: 2^n keys looked up in one table. Calls are numbered in the caller's order.
struct ForestCall {
  int n;                    // log2 of its number of keys; n >= dT
  int k;                    // log2 of its table's rows, 5 to 16
  int fn;                   // 0: range table; 1, 2: explicit table (2: byte limbs); only validated
  const u64 *tau = nullptr; // GPU: 2^k row values below 2^61 (`table_of` checks); nullptr at fn = 0
  u32 t0 = 0;               // the row of the padding key
};
// NC calls, 2^kT rows in the largest table. *_slots(), nmsg(): slot counts; at_*(): first slots.
struct ForestPlan {
  int NC = 0, dT = 0, kT = 0, NR = 0, NT = 0, rK = 0, cK = 0;
  std::vector<int> iota;    // n_c - dT: call c has 2^iota_c trees
  std::vector<int> first;   // the number of call c's first tree, a multiple of 2^iota_c
  std::vector<int> call_of; // [NT] the call of tree y; 0 for trees NR .. NT - 1
  int v() const { return dT + rK; }
  int d() const { return kT + cK; }
  static size_t layers(int from, int to) { // the slots of the product layers mu = from .. to
    size_t e = 0;
    for (int mu = from; mu <= to; mu++) e += sc_door::layer_slots(mu);
    return e;
  }
  // The tie: 31 sumchecks of 0 rounds, degree 3 in the end values; the first sends 3, the rest 2.
  static size_t tie_slots() { return sc_door::slots(sc_door::Form::EQ2, 0, 3, 3) + 30 * sc_door::slots(sc_door::Form::EQ2, 0, 2, 3); }
  size_t key_slots() const { return layers(0, v() - 1); }
  size_t tree_slots() const { return layers(5, 4 + d()); }
  // The table leaves: degree 2 in d row variables, 1 in 5 plane variables; no end value sent.
  size_t leaves_slots() const { return 1 + 2 * (size_t)d() + 5; }
  size_t bits_slots() const { return sc_door::slots(sc_door::Form::EQ2, 5 + d(), 1, 2); }
  size_t table_slots() const { return tree_slots() + leaves_slots() + bits_slots(); }
  size_t nmsg() const { return tie_slots() + key_slots() + table_slots(); }
  size_t at_tie() const { return 0; }
  size_t at_key_layer(int l) const { return tie_slots() + (l ? layers(0, l - 1) : 0); }
  size_t at_table_layer(int l) const { return tie_slots() + key_slots() + (l ? layers(5, 4 + l) : 0); }
  size_t at_leaves() const { return at_table_layer(d()); }
  size_t at_bits() const { return at_leaves() + leaves_slots(); }
  size_t plane_bits() const { return (size_t)NC << (kT + 5); } // the NC calls' multiplicity bits
  // The plan for ForestTable::init() (forest_table.cuh), whose "trees" are the 2^cK call numbers.
  ForestPlan of_the_table_side() const {
    ForestPlan t = *this;
    t.NR = t.NT = 1 << cK, t.rK = cK;
    t.call_of.clear();
    for (int c = 0; c < t.NT; c++) t.call_of.push_back(std::min(c, NC - 1));
    return t;
  }
};
// lidb (parameter file) caps dT. Exits with code 2 if the format or the engine rejects the group.
inline ForestPlan forest_plan(const std::vector<ForestCall> &calls, int lidb) {
  ForestPlan p;
  auto refuse = [](const char *what, int number) {
    printf("look-up forest: rejected: %s (%d)\n", what, number);
    fflush(stdout);
    name_the_end("forest-plan"); // serving: p258 end 1 forest-plan (M5g)
    exit(2);
  };
  p.NC = (int)calls.size();
  if (p.NC < 1 || p.NC > 16) refuse("a group has 1 to 16 calls", p.NC);
  p.dT = lidb;
  for (const ForestCall &c : calls) {
    if (c.k < 5 || c.k > 16) refuse("a table has 2^5 to 2^16 rows; log2 of the rows", c.k);
    if (c.fn < 0 || c.fn > 2 || (c.fn != 0) != (c.tau != nullptr)) refuse("fn is 0 without a table, 1 or 2 with one", c.fn);
    if (c.t0 >> c.k) refuse("the padding key's row is not in the table", (int)c.t0);
    p.dT = std::min(p.dT, c.n), p.kT = std::max(p.kT, c.k);
  }
  std::vector<int> order(p.NC);
  std::iota(order.begin(), order.end(), 0);
  std::stable_sort(order.begin(), order.end(), [&](int a, int b) { return calls[a].n > calls[b].n; });
  p.iota.resize(p.NC), p.first.resize(p.NC);
  for (int c : order) {
    p.iota[c] = calls[c].n - p.dT, p.first[c] = p.NR;
    if (p.iota[c] > 8) refuse("a call has more than 256 trees; log2 of its trees", p.iota[c]);
    p.NR += 1 << p.iota[c];
    p.call_of.insert(p.call_of.end(), (size_t)1 << p.iota[c], c);
  }
  while ((1 << p.rK) < p.NR) p.rK++;
  while ((1 << p.cK) < p.NC) p.cK++;
  if (p.NR > 256) refuse("a group has at most 256 trees", p.NR);
  p.NT = 1 << p.rK;
  p.call_of.resize(p.NT, 0); // trees NR .. NT - 1: padding only
  // A limit of this engine, not of the format: Forest::last() needs at least 2^17 keys in a tree.
  if (p.dT < 17) refuse("this engine wants trees of 2^17 keys or more; log2 of a tree's keys", p.dT);
  // 32 bits hold any multiplicity, and the multiplicities' sum cannot wrap around in the field.
  if (p.v() > 31) refuse("a group has fewer than 2^32 keys, the trees of padding counted; log2 of them", p.v());
  if (((unsigned __int128)1 << (p.d() + 32)) >= FIELD_P) refuse("the tables of a group have too many rows; log2 of them", p.d()); // FIELD_P: the field's prime, 2^64 - 2^32 + 1
  // A limit of this engine: a point has at most MAXV = 32 coordinates (pt_transcript.h).
  if (p.d() + 5 > 31) refuse("this engine takes kT + cK + 5 up to 31", p.d() + 5);
  return p;
}
// Where a call's padding keys are: ascending runs [first, second) of key indices, no two adjacent.
// They must be public, from the configuration only (not checked): Forest's launches depend on them.
struct ForestRuns {
  std::vector<std::pair<size_t, size_t>> run;
  void add(size_t from, size_t to) { // from is at or after the end of the last run (not checked)
    if (from >= to) return;
    if (!run.empty() && run.back().second == from)
      run.back().second = to;
    else
      run.push_back({from, to});
  }
};
// For each of a call's 2^iota trees: the smallest m <= 2^(dT - 1) such that keys m ..
// 2^(dT - 1) - 1 of both halves of the tree are padding (m = 0: the tree has only padding). pub:
// the caller's 2^iota counts of padding keys; exits with code 1 unless the runs match them.
inline std::vector<size_t> forest_padded_from(int dT, int iota, const ForestRuns &runs, const std::vector<u32> &pub) {
  const size_t half = (size_t)1 << (dT - 1);
  std::vector<size_t> from((size_t)1 << iota, 0), places((size_t)1 << iota, 0);
  for (size_t h = 0; h < 2 * from.size(); h++) {
    const size_t begin = h * half, end = begin + half;
    size_t padded = end; // the half's keys from this index on are padding
    for (const auto &r : runs.run) {
      if (r.first < end && r.second > begin) places[h / 2] += std::min(r.second, end) - std::max(r.first, begin);
      if (r.first < end && r.second >= end) padded = std::max(r.first, begin);
    }
    from[h / 2] = std::max(from[h / 2], padded - begin);
  }
  for (size_t b = 0; b < from.size(); b++)
    if (pub.size() != from.size() || places[b] != pub[b]) {
      printf("internal error: look-up forest: a call's runs of padding places do not add up to its numbers of padding keys\n");
      fflush(stdout);
      name_the_end("forest-plan"); // serving: p258 end 1 forest-plan (M5g)
      exit(1);
    }
  return from;
}
namespace fk {
// The form of a table's row values tau(t). RANGE: tau(t) = t. BYTE_LIMBS (2^16 rows): (t & 255) +
// 2^20 * (t >> 8) + 2^40 * tab8[t]. PACKED: t + 2^28 * ctab[t]. GENERAL: none of these forms.
enum Kind : int { RANGE = 0, BYTE_LIMBS = 1, PACKED = 2, GENERAL = 3 };
// table_of() sets all but lut's contents; Forest::prove() writes them after the challenge alpha.
struct Table {
  Kind kind = RANGE;
  int k = 0;                      // 2^k rows
  int bits = 0;                   // every row value is below 2^bits
  const u64 *tau = nullptr;       // 2^k row values; nullptr for RANGE
  const u8 *tab8 = nullptr;       // BYTE_LIMBS: 2^16 bytes
  const uint16_t *ctab = nullptr; // PACKED: 2^16 entries, 0 from 2^k on; 16-byte aligned
  T *lut = nullptr;               // lut[t] = (its call's alpha) - tau(t), 2^k entries, every kind
};
// Copies an explicit table's row values to the host (synchronously) to find the Kind. Allocates via
// pt_alloc (product_test.cuh): lut as kind 2, scratch memory; tab8 or ctab as kind 0, long-lived.
inline Table table_of(const ForestCall &c) {
  Table t;
  t.k = c.k, t.tau = c.tau, t.bits = c.k;
  PA(t.lut, 16ull << c.k, 2);
  if (!c.tau) return t;
  const size_t rows = (size_t)1 << c.k;
  std::vector<u64> row(rows);
  CK(cudaMemcpy(row.data(), c.tau, 8 * rows, cudaMemcpyDeviceToHost));
  bool limbs = c.k == 16, packed = true;
  std::vector<u8> b8(65536, 0);
  std::vector<uint16_t> c16(65536, 0);
  u64 all = 0;
  for (size_t j = 0; j < rows; j++) {
    const u64 base = (u64)(j & 255) + ((u64)(j >> 8) << 20), d8 = row[j] - base, d16 = row[j] - j;
    limbs = limbs && row[j] >= base && !(d8 & ((1ull << 40) - 1)) && (d8 >> 40) < 256;
    packed = packed && row[j] >= j && !(d16 & ((1ull << 28) - 1)) && (d16 >> 28) < 65536;
    b8[j] = (u8)(d8 >> 40), c16[j] = (uint16_t)(d16 >> 28), all |= row[j];
  }
  if (all >> 61) { // else forest_keys.cuh's sums of row values can overflow: an honest proof fails
    printf("look-up forest: rejected: a table has a row value of 2^61 or more\n");
    fflush(stdout);
    name_the_end("forest-plan"); // serving: p258 end 1 forest-plan (M5g)
    exit(2);
  }
  t.bits = 1;
  while (t.bits < 61 && (all >> t.bits)) t.bits++;
  t.kind = limbs ? BYTE_LIMBS : packed ? PACKED : GENERAL;
  if (limbs) {
    u8 *p;
    PAN(p, 65536, 0);
    CK(cudaMemcpy(p, b8.data(), 65536, cudaMemcpyHostToDevice));
    t.tab8 = p;
  } else if (packed) {
    uint16_t *p;
    PAN(p, 131072, 0);
    CK(cudaMemcpy(p, c16.data(), 131072, cudaMemcpyHostToDevice));
    t.ctab = p;
  }
  return t;
}
// One tree: 2^dT consecutive keys (16-bit row numbers; nullptr for y >= NR), its call's table
// and number, and the GPU address of alpha - call * theta (challenges of lookup_tie::k_draws()).
struct Tree {
  const uint16_t *keys = nullptr;
  Table tab;
  int call = 0;
  const T *alpha = nullptr;
};
} // namespace fk
