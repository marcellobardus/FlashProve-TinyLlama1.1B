// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// Two steps of `Prover`'s set-up: they build a request's statement (`header` in identifiers) and
// list the message sections of the sub-proofs (`fork` in identifiers). Call them after the set-up
// of all sub-proofs and setup_buffers() (prover/prover/other_setup.cuh).
#pragma once
#include "prover/prover.h"
#include "hash_commitment/hf_format.h" // HF_DW
static_assert(DW == HF_DW, "common/setting.h and hf_format.h must agree");

// Returns the tree hash of v: DW words. treehash() (prover/prover/base.cuh) exits if v has fewer,
// and does not check the size of its scratch (setup_buffers()).
inline std::vector<u32> Prover::hash_words(const std::vector<u32> &v) {
  u32 *dv;
  CK(cudaMalloc(&dv, v.size() * 4));
  CK(cudaMemcpy(dv, v.data(), v.size() * 4, cudaMemcpyHostToDevice));
  std::vector<u32> o(DW);
  CK(cudaMemcpy(o.data(), treehash(dv, v.size()), 4 * DW, cudaMemcpyDeviceToHost));
  CK(cudaFree(dv));
  return o;
}

// Builds the statement (`hd` on the host, `hdr` on the GPU): values of the parameter file
// (`policy.txt`; `policy` in identifiers) and, between them, digests of its call lines, the
// public tables and the quantization scales (KL), with zeros where each request writes the tree
// hash of its tokens, r and e (atTokens) and its next token (atY).
// The statement does not name the proof's leaf table (a stored copy of the weights):
// bind_request_statement() (blocks/frame/check.h) puts that table's root, record and number, the
// statement (atY still 0) and then the request file's own words (the tokens, r and e) into the
// first query (hash input) of the request's first chain.
// NH, the statement's length in words (105 + 9 * DW), is not checked against `hdr`'s 1,024 words
// (setup_buffers()). Returns 0, or 2 after printing the reason.
inline int Prover::build_header() {
  hd = policy.w;
  // The parameter file may give the four public tables' digests (4 * DW lines, all or none). They
  // must match the tables that this program computes with the machine's floating-point library;
  // otherwise it would prove a different function.
  const bool tables_given = policy.has("tsilu0");
  // A model with experts (`mo`) uses all four tables and needs the digest lines. A dense model
  // (NE 0, !mo) uses three (SILU, RoPE, softmax; not the FS/SIG table): the sigmoid block below is
  // skipped and its digest slot is written as zeros, and table_fits() checks each of the three when
  // the lines are given. The configuration digest covers the digest lines, and every batch's first
  // key depends on it (batch_first_key(), prover/common/coin_keys.h). So two machines whose
  // floating-point libraries differ cannot prove with different tables under the same private
  // randomness.
  if (!tables_given && mo) {
    printf("refused (policy-tables-absent): the commitment's parameter file lacks the lines tsilu0 .. texp%d, the digests of the four "
           "public tables; cp_prove serves no such commitment. Write the lines with scales/policy_tables.py and commit anew\n",
           DW - 1);
    return refused("policy-tables-absent");
  }
  auto table_fits = [&](const char *lines, const char *name, const u32 *digest) {
    if (!tables_given) return true;
    bool same = true;
    for (int i = 0; i < DW; i++) same = same && policy[(lines + std::to_string(i)).c_str()] == digest[i];
    if (!same)
      printf("policy: table-digest-%s: the table computed on this machine does not have the digest given in the lines "
             "%s0 .. %s%d of the parameter file\n",
             name, lines, lines, DW - 1);
    return same;
  };
  // Hash number 73: call lines; 72: a table.
  {
    std::vector<u32> cw((policy.calls.size() + 3) / 4, 0);
    u32 o[DW];
    for (size_t i = 0; i < policy.calls.size(); i++) {
      cw[i / 4] |= (u32)(unsigned char)policy.calls[i] << (8 * (i % 4));
    }
    earlier_hash_h_digest(o, 73, cw.data(), cw.size());
    size_t at = std::find(policy.names.begin(), policy.names.end(), "ut") - policy.names.begin() + 1;
    hd.insert(hd.begin() + at, o, o + DW);
  }
  {
    std::vector<u32> tw(65536);
    u32 o[DW];
    for (int j = 0; j < 65536; j++) {
      tw[j] = Ts[j] + 4096;
    }
    earlier_hash_h_digest(o, 72, tw.data(), 65536);
    if (!table_fits("tsilu", "silu", o)) return refused("table-digest"); // serving: p258 end 1 table-digest
    hd.insert(hd.end(), o, o + DW);
  }
  if (mo) {
    std::vector<u32> tw(65536);
    u32 o[DW];
    for (int j = 0; j < 65536; j++) {
      tw[j] = FSw[j];
    }
    earlier_hash_h_digest(o, 72, tw.data(), 65536);
    if (!table_fits("tsig", "sigmoid", o)) return refused("table-digest"); // serving: p258 end 1 table-digest
    hd.insert(hd.end(), o, o + DW);
  } else {
    hd.insert(hd.end(), DW, 0u);
  }
  {
    std::vector<short> c((size_t)NT * dh / 2), s((size_t)NT * dh / 2);
    CK(cudaMemcpy(c.data(), FK[0].g.b.Cr, c.size() * 2, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(s.data(), FK[0].g.b.Sr, s.size() * 2, cudaMemcpyDeviceToHost));
    std::vector<u32> tw;
    for (auto v : c) {
      tw.push_back((u32)(v + 4096));
    }
    for (auto v : s) {
      tw.push_back((u32)(v + 4096));
    }
    u32 o[DW];
    earlier_hash_h_digest(o, 72, tw.data(), tw.size());
    if (!table_fits("trope", "rope", o)) return refused("table-digest"); // serving: p258 end 1 table-digest
    hd.insert(hd.end(), o, o + DW);
  }
  {
    auto th = hash_words(KL);
    hd.insert(hd.end(), th.begin(), th.end());
  }
  NHb = (int)hd.size();
  {
    hd.insert(hd.end(), policy.wA.begin(), policy.wA.end());
    std::vector<u32> tw(Te.begin(), Te.end());
    u32 o[DW];
    earlier_hash_h_digest(o, 72, tw.data(), tw.size());
    if (!table_fits("texp", "exp", o)) return refused("table-digest"); // serving: p258 end 1 table-digest
    hd.insert(hd.end(), o, o + DW);
    std::vector<u32> cw((policy.callsA.size() + 3) / 4, 0);
    for (size_t i = 0; i < policy.callsA.size(); i++) {
      cw[i / 4] |= (u32)(unsigned char)policy.callsA[i] << (8 * (i % 4));
    }
    earlier_hash_h_digest(o, 73, cw.data(), cw.size());
    hd.insert(hd.end(), o, o + DW);
  }
  {
    hd.insert(hd.end(), policy.wEH.begin(), policy.wEH.end());
    std::vector<u32> cw((policy.hcalls.size() + 3) / 4, 0);
    u32 o[DW];
    for (size_t i = 0; i < policy.hcalls.size(); i++) {
      cw[i / 4] |= (u32)(unsigned char)policy.hcalls[i] << (8 * (i % 4));
    }
    earlier_hash_h_digest(o, 73, cw.data(), cw.size());
    hd.insert(hd.end(), o, o + DW);
    atTokens = (int)hd.size();
    hd.insert(hd.end(), DW, 0u);
    atY = (int)hd.size();
    hd.push_back(0u);
  }
  hd.push_back((u32)hb_);
  for (const u32 n : {(u32)policy.frame.block_log, (u32)policy.frame.table_rows_log, (u32)policy.frame.table_log_n, (u32)policy.frame.table_random}) { // four numbers of the line `table`
    hd.push_back(n);
  }
  NH = (int)hd.size();
  CK(cudaMemcpy(hdr, hd.data(), NH * 4, cudaMemcpyHostToDevice));
  return 0;
}

// Fills SC with the sections of the sub-proofs' messages (lengths in 32-bit words; no address for
// attention's); one of 0 words only keeps its place. Allocates `out`, where a layer sub-proof's or
// the head's sections are concatenated; `cp_prove` never reads the result (`Door::host_words`,
// prover/hiding/door.cuh).
inline int Prover::plan_sections() {
  SC.push_back({"CP", ph1, (size_t)0});
  for (int g = 0; g < NGr; g++) {
    Fork &K = FK[g];

    HFk &Q = *K.h;
    HiddenTail &HT = Q.HT;
    SC.push_back({"FORK" + std::to_string(g), nullptr, (size_t)0});
    SC.push_back({"FOREST", K.sPT, EWORDS * layer_forest.nmsg()});
    SC.push_back({"WT", MSG, 0});
    SC.push_back({"VW", K.sVW, (size_t)norm_sum_count(xl) * EWORDS * NI});
    SC.push_back({"WPL", K.sWPL, EWORDS * K.WP->nmsg()});
    SC.push_back({"PV", K.sPV, (size_t)norm_product_count(xl) * EWORDS * NI});
    SC.push_back({"TSCV", K.sTV, EWORDS * K.tV.nmsg()});
    for (auto &q : K.ts) { // none or one
      SC.push_back({"TSC", q.msg, EWORDS * q.eng.nmsg()});
    }
    SC.push_back({"SCH", MSG + Q.slSCH, EWORDS * CSumcheck::slots(Q.lamS, Q.nEndS, 2)});
    SC.push_back({"PL1", K.sPL, EWORDS * K.HL->nmsg()});
    if (mo) SC.push_back({"PL2", K.sPL2, EWORDS * K.HL2->nmsg()});
    SC.push_back({"VE", MSG + Q.slVE, (size_t)(g == 0 ? EWORDS : 0)});
    SC.push_back({"MSC", MSG + Q.slMSC, EWORDS * (CSumcheck::slots(lkW, 1, 1) - 1)});
    SC.push_back({"FVP", Q.GS.fvp, EWORDS}); // a copy of the merge's last slot, which MSC omits
    SC.push_back({"W", Q.roots + 3 * DW, DW});
    SC.push_back({"ROOT-ZZ-BZ", Q.zs, DW});
    SC.push_back({"FOREST-9", MSG + Q.slPZ, EWORDS * HT.F.plan.nmsg()});
    SC.push_back({"+", MSG + Q.oLate, (size_t)0});
    SC.push_back({"+", Q.sc + 2, 0});
    SC.push_back({"XR", nullptr, (size_t)0});
    SC.push_back({"HCOLS", nullptr, (size_t)0});
  }
  secPerFork = (SC.size() - 1) / NGr;
  iSecH = SC.size();
  HH.sections(SC);
  iSecHe = SC.size();
  {
    size_t o = 0;
    for (auto &q : SC) {
      o += q.words;
    }
    for (int g = 0; g < NGr; g++) {
      oA.push_back(o);
      for (auto &q : AG[g].o->SC) {
        SC.push_back({q.name == "+" ? std::string("+") : "A" + std::to_string(g) + ":" + q.name, nullptr, q.words});
      }
      o += AG[g].o->nw;
    }
  }
  nw = 0;
  for (auto &s : SC) {
    nw += s.words;
  }
  {
    size_t largest = 0, section = 1, head = 0;
    for (int g = 0; g < NGr; g++) {
      size_t n = 0;
      for (size_t q = 0; q < secPerFork; q++, section++) n += SC[section].words;
      largest = std::max(largest, n);
    }
    for (size_t q = iSecH; q < iSecHe; q++) head += SC[q].words;
    largest = std::max(largest, head);
    fork_memory.purpose = "the proof's buffer";
    CK(cudaMalloc(&out, largest * 4));
    fork_memory.purpose = "other buffers of the prover";
  }
  {
    size_t fr, tot;
    CK(cudaMemGetInfo(&fr, &tot));
    size_t keys = 0;
    for (int c = 0; c < NC; c++) {
      keys += (size_t)1 << FK[0].n[c];
    }
    size_t cells = 0;
    for (int i = 0; i < FK[0].NR; i++) {
      cells += FK[0].M[i].m * FK[0].M[i].k;
    }
    printf("# %d layer sub-proofs of %d layers each, T %d, d %d, f %d; per sub-proof: %d lookup calls, %zu key slots, "
           "%zu committed cells of the request in %d matrices; kt %zu; GPU memory used %.1f of %.1f GB; proof %.4f GB; "
           "header %d words\n",
           NGr, Lam, NT, d, f, NC, keys, cells, FK[0].NR, FK[0].h->HT.mz, (tot - fr) / 1e9,
           tot / 1e9, nw * 4 / 1e9, NH);
    fflush(stdout);
  }
  return 0;
}
