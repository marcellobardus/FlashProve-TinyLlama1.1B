// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// Sets up the attention sub-proofs (`fork` in identifiers), the sub-proof of the output head and
// further buffers. Call the three functions after setup_g_forks(), setup_buffers() last; a non-zero
// result ends Prover::run(). cudaMalloc() is a macro for ForkMemory's allocator (fork_memory.h).
#pragma once
#include "prover/prover.h"

// One AForkHid per distinct list of shifts (KL's `sh`); registers each layer group with its object.
inline int Prover::setup_attention_forks() {
  AG = std::vector<AGr>(NGr);
  Aq = nullptr;
  Ak = nullptr;
  Av = nullptr;
  Aqa = nullptr;
  Art = nullptr;
  Aef = nullptr;
  trA = nullptr;
  {
    // Only list 3 of lookup calls is supported (APlan::makeCallList(), prover/attn/plan.cuh).
    // limit_missed() (prover/common/parameter_file.h) rejects other lists: prover-only-callA-lines.
    std::map<std::vector<int>, AForkHid *> bySh;
    for (int g = 0; g < NGr; g++) {
      std::vector<int> sh;
      for (int lam = 0; lam < Lam; lam++) {
        sh.push_back((int)KL[(size_t)KLW * (g * Lam + lam) + 11]);
      }
      const u32 gid = (u32)(2 + NGr + g), mq = (u32)((1 + g) * 256 + iQA);
      auto it = bySh.find(sh);
      if (it == bySh.end()) {
        AForkHid *Hd = new AForkHid;
        fork_memory.setup_single(NGr + 1 + (int)AO.size());
        AP &p = Hd->A.P.p;
        p.lt = lt;
        p.lB = policy["lB"];
        p.H = H;
        p.lgk = l2i(Gk);
        p.ldh = ldh;
        p.ll = ll;
        p.S = policy["S"];
        p.rx = policy["rx"];
        p.kFM = policy["kFM"];
        p.kFD = policy["kFD"];
        p.D = D;
        p.lmS = policy["lmS"];
        p.lwH = (int)policy["lmh"] - lt;
        p.lwO = (int)policy["lmo"] - lt;
        p.lwQ = (int)policy["lmq"] - lt;
        p.la = la;
        p.lmb = policy["lmbA"];
        p.nmin = nmin;
        p.iota = policy["iotaA_0"]; // 32 + log2(keys per lookup tree), or `AFront::plan` exits
        Hd->A.P.sh = sh;
        Hd->A.P.form = 3;
        Hd->rank = rkR;
        Hd->A.rebuilt = args.attention_rebuilt;
        Hd->HT.rebuilt = args.tail_rebuilt;
        Hd->HT.chainMoment = 2; // `HT` (`tail`): approximate range check; commit point (`moment`) 2
        Hd->setup((int)policy["NAs"], s8::hb_r(), gid, mq);
        fork_memory.setup_single_done();
        bySh[sh] = Hd;
        AO.push_back(Hd);
        AG[g] = AGr{Hd, 0};
        thW = std::max(thW, Hd->hash_words());
      } else {
        AG[g] = AGr{it->second, it->second->add_fork(gid, mq)};
      }
      chain_plan_later_moments(chain.plan[chain.attention(g)], {{"U3", AG[g].o->U3.K}}, AG[g].o->HT);
    }
    CK(cudaMalloc(&Aq, (size_t)Lam * NT * dq * 2));
    CK(cudaMalloc(&Ak, (size_t)Lam * NT * dkv * 2));
    CK(cudaMalloc(&Av, (size_t)Lam * NT * dkv * 2));
    for (auto *Hd : AO)
      if (Hd->A.ncell[A_QA] != cells_of(FK[0], FK[0].qa)) {
        printf("internal error: an attention sub-proof and the layer sub-proofs give the matrix qa different numbers "
               "of cells\n");
        return refused("attention-qa"); // serving: p258 end 1 attention-qa
      }
    CK(cudaMalloc(&Aqa, cells_of(FK[0], FK[0].qa)));
    CK(cudaMalloc(&Art, (size_t)NGr * ART_WORDS * 4));
    CK(cudaMalloc(&Aef, (size_t)NGr * SW * 4));
    CK(cudaMalloc(&trA, sizeof(Tr)));
  }
  return 0;
}

// Sets up the sub-proof of the output head: final norm, logits, and the test of each checked token.
inline int Prover::setup_head_fork() {
  {
    fork_memory.setup_single(NGr);
    HHidCfg hc;
    hc.lt = lt;
    hc.la = la;
    hc.nmin = nmin;
    hc.nminH = policy.m.count("nminH") ? (int)policy["nminH"] : 0;
    hc.rebuilt = args.head_rebuilt;
    hc.d = d;
    hc.V = Vv;
    hc.D = D;
    hc.NGr = NGr;
    hc.lmx = policy["lmx"];
    hc.hb = s8::hb_r();
    hc.lkW = lkW;
    hc.seed = seed;
    hc.word = [&](const std::string &n) {
      return (u32)policy[n];
    };
    hc.hcalls = policy.hcalls;
    hc.xl = xl;
    HH.h.cm = 2047;
    HH.HT.rebuilt = args.tail_rebuilt;
    if (int rc = HH.setup(hc, XBv[NGr], tFCd, tFC[32768] % FIELD_P, &FK[NGr - 1].h->HT, iXN, MSG, NSM, &CP)) {
      // name_the_end keeps the first name, so a specific name HH.setup already set stays; this only
      // names an otherwise-bare end from a head-setup refusal that dropped the name. serving: p258 end 1 head-setup
      name_the_end("head-setup");
      return rc;
    }
    // Lists the head's unmasked weight claim as a request's data, as in layer_walk.cuh. The masked
    // claim is a Place::Result buffer. ForkMemory records both: restoring GPU memory covers them.
    fork_memory.of_a_request.push_back({HH.CL.end.fv, sizeof(T)});
    HH.HT.chainMoment = 2; // commit points, here as for attention: 0 first, 1 U3, 2 the range check
    chain_plan_later_moments(chain.plan[chain.head()], {{"U3", HH.U[3].K}}, HH.HT);
    rscB = std::max(rscB, HH.rscBytes);
    if (weights->head()) {
      HH.weights_from(weights->head());
    }
    HH.weights();
    fork_memory.setup_single_done();
  }
  return 0;
}

// Allocates further buffers and sets the layer sub-proofs' addresses in the message area MSG.
inline int Prover::setup_buffers() {
  thW = std::max(thW, (size_t)2 * NT * d);
  for (int i = 0; i < 2; i++) {
    CK(cudaMalloc(&g_th[i], (thW * DW / THPW + 4096) * 4));
  }
  CK(cudaMalloc(&rsc, rscB));
  if (weights->row_pass_scratch()) CK(fork_alloc(&row_pass_scratch, weights->row_pass_scratch(), Place::Own));
  CK(cudaMalloc(&errc, 4));
  CK(cudaMemset(errc, 0, 4));
  CK(cudaMalloc(&errf, 4));
  CK(cudaMemset(errf, 0, 4));
  CK(cudaMalloc(&slot_pads_agree, 8));
  CK(cudaMemset(slot_pads_agree, 0, 8));
  hz::slot_pads_agreeing = slot_pads_agree;
  CK(cudaMalloc(&filler_is_zero, 8));
  CK(cudaMemset(filler_is_zero, 0, 8));

  CK(cudaMalloc(&err, 4));
  CK(cudaMemset(err, 0, 4));
  CK(cudaMalloc(&sd, 2 * 4 * SW));
  CK(cudaMalloc(&hdr, 4096));
  CK(cudaMalloc(&tr, sizeof(Tr)));
  nPh1 = DW; // ph1 (allocated below) has DW four-byte words; `plan_sections` lists it with length 0
  CK(cudaMalloc(&efs, SW * 4 * 128));

  for (auto &K : FK) {
    if (K.h->S.sh.Kc > NSM) {
      printf("internal error: S has %zu coefficients, but the message area was sized for %zu\n", K.h->S.sh.Kc, NSM);
      return refused("message-area"); // serving: p258 end 1 message-area
    }
  }
  for (auto &K : FK) {
    HFk &Q = *K.h;
    K.sPT = MSG + Q.slPT;
    K.sVW = MSG + Q.slVW;
    K.sWPL = MSG + Q.slWPL;
    K.sPV = MSG + Q.slPV;
    K.sTV = MSG + Q.slTV;
    if (!K.ts.empty()) K.ts[0].msg = MSG + Q.slTS;
    K.sPL = MSG + Q.slPL;
    if (mo) {
      K.sPL2 = MSG + Q.slPL2;
    }
  }

  CK(cudaMalloc(&ph1, nPh1 * 4));
  make_weight_cells = [&](int g, const WeightKind &w, int p) {
    weights->cells(w.kind, g, w.bytes, p, w.lm.n, slab);
  };
  CK(cudaDeviceSynchronize());
  // [own-test-slot-pads] Pinned host memory for the largest sub-proof's slot pads (message masks).
  size_t most_slots = HH.door.ns;
  for (int g = 0; g < NGr; g++) most_slots = std::max({most_slots, (size_t)FK[g].h->door.ns, (size_t)AG[g].o->door.ns});
  if (!slot_pads_second.set_up(most_slots)) {
    printf("no pinned host memory for the test of the slot pads by value (own-test-slot-pads)\n");
    return refused("no-host-memory");
  }
  return 0;
}
