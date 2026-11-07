// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// A set-up check that no two trees (hz::Tree, hz::TreeN) reserve the same stream of private
// randomness (printed: "coin streams"). It compares only the request's pad tree `CP` and these
// trees of each layer, attention and output-head sub-proof (`fork` in identifiers): S, U3, and ZZ
// and BZ of its approximate range check (`HT`). On an overlap it returns 2 (otherwise 0) and
// Prover::run() proves no request.
#pragma once
#include <algorithm>
#include <cstdlib>
#include <map>
#include <string>
#include <utility>
#include <vector>
#include "prover/prover.h"
// A tree numbered id reserves streams id to id + 3: random tails; random rows B and h_i; salts;
// unused. Call plan_stream_ids() after the trees' set-up: a sub-proof added later is not seen.
inline int Prover::plan_stream_ids() {
  typedef std::vector<std::pair<std::string, u32>> Ids;
  Ids all;
  auto named = [&](const std::string &who, const Ids &ids, long long shift) {
    for (auto &x : ids) all.push_back({who + "." + x.first, (u32)((long long)x.second + shift)});
  };
  auto base_of = [](u32 gid) {
    return 4u * (4096u + 64u * gid);
  };
  all.push_back({"CP", CP.stream});
  for (int g = 0; g < NGr; g++) {
    HFk &Q = *FK[g].h; // ZZ and BZ: computed from HT.sid0
    Ids ids = {{"S", Q.S.stream}, {"U3", Q.U[3].stream}}; // layer_setup.cuh sets up no other U[k]
    FK[g].h->HT.stream_ids(ids);
    named("G" + std::to_string(g), ids, 0);
  }
  // An object in AO can serve several attention sub-proofs (`gids`). AForkHid::begin_fork()
  // (prover/attn/fork.cuh) sets its trees' numbers for each; base_of(), the offsets and the shift
  // of `tail` (ZZ, BZ) repeat its rule, so a change made only in begin_fork() is not seen.
  for (auto *Hd : AO)
    for (u32 gid : Hd->gids) {
      const u32 b = base_of(gid);
      Ids ids = {{"S", b}, {"U3", b + 12}}, tail;
      Hd->HT.stream_ids(tail);
      named("A" + std::to_string(gid), ids, 0);
      named("A" + std::to_string(gid), tail, (long long)(b + 64) - (long long)Hd->HT.sid0);
    }
  {
    Ids ids = {{"S", HH.S.stream}, {"U3", HH.U[3].stream}}; // as head/hidden_fork.cuh sets them up
    HH.HT.stream_ids(ids);
    named("H", ids, 0);
  }
  std::sort(all.begin(), all.end(), [](const auto &a, const auto &b) { return a.second < b.second; });
  for (size_t i = 1; i < all.size(); i++)
    if (all[i].second - all[i - 1].second < 4) {
      printf("plan: the coin streams of %s (id %u) and %s (id %u) overlap; nothing was proved\n",
             all[i - 1].first.c_str(), all[i - 1].second, all[i].first.c_str(), all[i].second);
      return refused("stream-plan"); // serving: p258 end 1 stream-plan
    }
  return 0;
}
