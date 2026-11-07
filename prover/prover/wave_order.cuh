// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// Decides whose turn it is in a batch pass, and tells the weight stream how far the batch pass has
// come. A batch pass (`Wave` in identifiers) is the part of a batch, at most -in-step requests,
// that the program cp_prove proves in one pass over the weights. A turn is one request's work on
// one layer group or on the output head: each request has its turn in a group before any goes on.
// Public source for this order: Sheng et al., "FlexGen", ICML 2023, arXiv:2303.06865,
// section 4.2, "Compute schedule".
// With NGr layer groups, a batch pass of n requests has NGr + 1 rounds of n turns. Round g < NGr is
// layer group g; the last round is the output head's sub-proof and each request's end. In every
// round the turns come in ascending order of `place`, a request's index within its batch pass. So
// requests start and end in the order of their slots; seat_begins() (prover/prover/in_step.cuh)
// ends the process on any other order of starts. The order of the turns depends only on n and NGr.
//
// With -overlap an earlier batch (a "rider"; with -overlap-two, up to two) may have an unfinished
// opening of the weight table. Each rider's opening then makes one of its passes over the weights
// (`walk` in identifiers) within this batch pass: it uses group g's loaded weights after turn
// folds_behind(n) of round g, the rest (embedding, output head) before the last round. If n = 1,
// that turn keeps the group's routed experts loaded, and the_rider_folds() releases their buffers.
#pragma once
#include "prover/in_step.cuh"

// The `place` of the turn after which riders use a round's weights: the last but one of n turns
// (the only turn if n = 1). Any turn but the last would do: the group is loaded, and no buffer is
// released before the last. A late one is chosen because lending memory to a rider's opening and
// taking it back wait for the whole GPU (prover/weights_commitment/lent_memory.h), and so for all
// queued loads. After a round's first turn, loads into buffers released by the previous round, some
// for the next group, can still be running (seen in a trace at full size). The choice should change
// no byte of a response; a test not part of this repository compares only with runs without riders.
static inline int folds_behind(int n) { return n > 1 ? n - 2 : 0; }

// Proves the slots of `w`, each to its end, whatever its failure word records. An empty slot gets
// no turn. That a slot is empty is not a secret of any request. Returns 0, or at once the nonzero
// result of a call for the riders. Other faults of the machine or program end the process (ends.h).
inline int prove_wave(Prover &P, const Wave &w) {
  if (!w.batch || w.first < 0 || w.n < 1 || w.n > P.args.in_step || w.first + w.n > w.batch->B) {
    printf("internal error: invalid pass: %d slots starting at slot %d\n", w.n, w.first);
    name_the_end("wave-pass"); // serving: p258 end 1 wave-pass (kept as an exit(4))
    exit(4);
  }
  std::vector<Slot *> at_place;
  for (int j = 0; j < w.n; j++)
    if (w.batch->slot[w.first + j].state != SlotState::Empty) at_place.push_back(&w.batch->slot[w.first + j]);
  const int n = (int)at_place.size(), groups = P.NGr, turns = (groups + 1) * n;
  if (!n) return 0;
  for (int turn = 0; turn < turns; turn++) {
    const int g = turn / n, place = turn % n;
    const bool first_of_round = place == 0, last_of_round = place == n - 1;
    if (g < groups && first_of_round && P.weights_ready) P.weights_ready(g);
    // For the weight stream: false while a later turn or a rider still needs g's routed experts.
    P.last_of_wave = g == groups || (last_of_round && !(P.wave_carries && n == 1));
    Slot &slot = *at_place[place];
    // Each rider's pass over the weights ends here. service.cuh defines the calls for the riders.
    if (first_of_round && g == groups && P.wave_carries)
      if (int rc = P.the_riders_walk_ends()) return rc;
    if (g == 0) P.seat_begins(slot, place);
    if (g < groups)
      P.seat_turn(slot, place, g);
    else
      P.seat_ends(slot, place);
    if (g < groups && place == folds_behind(n))
      if (int rc = P.the_rider_folds(g, last_of_round)) return rc;
    if (g < groups && last_of_round && P.weights_done) P.weights_done(g);
  }
  P.wave_has_ended();
  return 0;
}
