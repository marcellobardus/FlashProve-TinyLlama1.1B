// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// A CPU test of moe_at_layer() (prover/prover/layout.h), which the forward pass
// (prover/prover/forward.cuh) uses to decide whether a layer runs the mixture-of-experts path
// (ffn_moe) or the dense path (ffn_dense). A layer is MoE only if the model has routed experts
// (NE > 0). The rest of the prover already treats a model with NE 0 as all dense (mo = NE > 0 in
// prover/prover/inputs.cuh and prover/weights_commitment/kernels.cuh), and so does the verifier
// (MOE = NE > 0 in verifier/src/vf_weights_layout.h). This test pins that agreement: without the
// NE > 0 clause the forward pass alone ran ffn_moe() on odd layers of a dense (NE 0) model, against
// an all-dense layout, and a forward-pass value left its range (UNFIT), so cp_prove wrote no proof.
// Build, from prover/: ./build.sh moe_at_layer_test. Prints one line per check. Exit code 0 if all
// pass, 1 if not.
#include "prover/layout.h"
#include <cstdio>

static int failed = 0;
static void check(bool good, const char *what) {
  printf("%s %s\n", good ? "as wanted" : "DIFFERS  ", what);
  failed += !good;
}

// The rule that the verifier and the rest of the prover use: a layer is MoE iff the model has
// routed experts and the layer is odd. moe_at_layer() adds the prover's whole-model gate prof >= 3.
static bool reference(int prof, int l, int NE) {
  return prof >= 3 && (l & 1) && NE > 0;
}

int main() {
  // A dense model (NE 0): no layer is MoE, whatever the profile or the layer's parity. This is the
  // case that failed before the fix.
  bool any_moe_at_ne0 = false;
  for (int prof : {0, 1, 2, 3, 20})
    for (int l = 0; l < 24; l++)
      any_moe_at_ne0 = any_moe_at_ne0 || moe_at_layer(prof, l, 0);
  check(!any_moe_at_ne0, "a model with NE 0 has no MoE layer (every layer is dense)");

  // A model with routed experts (NE > 0) at the whole-model profile: odd layers are MoE, even are
  // dense. This is the unchanged behaviour for the kits that already prove.
  bool ok_ne4 = true;
  for (int l = 0; l < 24; l++) ok_ne4 = ok_ne4 && (moe_at_layer(20, l, 4) == ((l & 1) != 0));
  check(ok_ne4, "with NE 4 and prof 20 the odd layers are MoE and the even layers are dense");

  // moe_at_layer() agrees with the reference rule over a range of profiles, layers and NE.
  bool agrees = true;
  for (int prof = 0; prof <= 21; prof++)
    for (int l = 0; l < 48; l++)
      for (int NE : {0, 1, 4, 8, 64})
        agrees = agrees && (moe_at_layer(prof, l, NE) == reference(prof, l, NE));
  check(agrees, "moe_at_layer matches prof >= 3 and odd layer and NE > 0 over a wide range");

  // The profile gate: below prof 3 no layer is MoE even with experts (the prover proves no MoE
  // layer outside the whole-model profile).
  check(!moe_at_layer(2, 1, 4), "prof 2 has no MoE layer even with NE 4");
  check(moe_at_layer(3, 1, 4), "prof 3, odd layer, NE 4 is a MoE layer");

  printf("moe_at_layer_test: %s\n", failed ? "some checks DIFFER" : "all checks as wanted");
  return failed ? 1 : 0;
}
