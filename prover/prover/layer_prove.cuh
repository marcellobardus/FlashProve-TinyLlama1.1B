// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// The steps proving layer group g (Lam layers): a layer sub-proof (`fork` in identifiers), then an
// attention sub-proof. prove_layer_group() (request.cuh) calls them in transcript order. `link`: a
// segment of the proof; `slot` j: the masked value MSG[j]; its `pad`: the mask entry MSG[NSM + j].
#pragma once
#include "common/blas.h"
#include "prover/prover.h"
#include "common/zfold_turned.cuh"

// Adds to *count the number of the n bytes at p that are zero.
// Public source for the loop: see k_fork_memory_count_not() (fork_memory.h); here 32 blocks of 256.
static __global__ void k_count_zero_bytes(const i8 *p, size_t n, unsigned long long *count) {
  unsigned long long found = 0;
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x)
    found += p[i] == 0;
  atomicAdd(count, found);
}

// Empty test hooks, with four calls below. A malicious-prover test build that is not part of this
// repository has its own forms, which alter one entry of layer group g that no sumcheck sends:
//   cell   of ad_0's committed cells, after the forward pass, before the commitment
//   gated  of the gated products' token sums K.sHH | K.sHH2, before the batched sumcheck
//   read   of the reads U3 (row-pass results), after their sumchecks, before commitment
static inline void layer_lie_words(int) {}
static inline void layer_lie_element(const char *, int, T *, size_t) {}
static inline void layer_lie_cell(const char *, int, i8 *, size_t) {}

// Runs the forward pass of layer group g and commits to its matrices and mask entries.
inline int Prover::g_commit(int g) {
  Fork &K = FK[g];
  HFk &Q = *K.h;
  HiddenTail &HT = Q.HT;

  stage_clock.at(StageClock::Forward);
  fork_memory.run_begin(g);
  // Repeats set-up's writes to the layer sub-proofs' shared buffers. run_begin() did so in one
  // launch, but no call reports a launch that is accepted and never run: old bytes would stay, also
  // in filler cells (padding), which the forward pass never writes and the verifier does not pin.
  for (const ForkMemory::Write &w : fork_memory.shared_writes)
    CK(w.copy_of ? cudaMemcpyAsync(fork_memory.slab + w.offset, w.copy_of, w.bytes, cudaMemcpyDeviceToDevice,
                                   cudaStreamPerThread)
                 : cudaMemsetAsync(fork_memory.slab + w.offset, w.value, w.bytes, cudaStreamPerThread));
  chain_bind(chain.layer(g));
  Q.door.link = chain.layer(g);
  ch::host.blob = [this, g] { // the fork's sections in file order
    size_t o = 0;
    for (size_t q = 0; q < secPerFork; q++) {
      const SecD &s = SC[1 + (size_t)g * secPerFork + q];
      if (s.words) CK(cudaMemcpyAsync(out + o, s.p, s.words * 4, cudaMemcpyDeviceToDevice));
      o += s.words;
    }
    return (const u32 *)out;
  };
  if (g == 0) {
    if (weight_stream.ends_by_turns) weight_stream.experts_ready(weight_stream.end_unit[0]);
    lie::here(0, lie::EMBEDDING, cudaStreamPerThread);
    weights->embed(tokd, NT, xl, K.g.xs(0), XBv[0]);
    if (weight_stream.ends_by_turns) weight_stream.experts_used(weight_stream.end_unit[0]);
  }
  forward_and_counts(g);
  stage_clock.at(StageClock::Commit);
  if (g == 0) layer_lie_words(NGr);
  layer_lie_cell("cell", g, K.M[K.ad.first].v, K.M[K.ad.first].m * K.M[K.ad.first].k); // test hook
  // Counts the filler cells that are zero, as an honest prover's are, before the commitment.
  // finish_request() (request.cuh) compares the count with their number. Zeros are counted, not
  // other bytes, so that a request also gets no proof if one of these launches never ran.
  for (auto &range : K.filler)
    if (range.second) k_count_zero_bytes<<<32, 256>>>(range.first, range.second, filler_is_zero);
  CK(cudaGetLastError());
  CK(cudaMemcpyAsync(xin[g + 1], K.g.xs(NW * Lam), (size_t)d * NT * sizeof(ResidualInt), cudaMemcpyDeviceToDevice));
  Q.S.slot_pads(Q.door.ns);
  // Copies the ns mask entries that slot_pads() has drawn to MSG + NSM. The cells after them are
  // zeroed below, so that no entries of a sub-proof with more slots stay there.
  if (!lost_copies_door::left_out("slot_pads_copy")) // test hook: false in a production build
    k_pl2e<<<nblk(Q.door.ns), 256>>>(Q.S.h_plane(0, 0), (long long)Q.S.plane_stride(), Q.door.ns, MSG + NSM);
  // After a rejected launch MSG + NSM would not contain this sub-proof's mask entries, and unmasked
  // or wrongly masked values would go into the proof file. So the status is checked at once: CK()
  // exits before that file is written.
  CK(cudaGetLastError());
  lost_copies_door::strike("slot_pads_copy", MSG + NSM, EDEG * Q.door.ns, true);
  CK(cudaMemsetAsync(MSG + NSM + Q.door.ns, 0, (NSM - Q.door.ns) * sizeof(T), cudaStreamPerThread));
  slot_counts_begin(MSG);
  Q.S.slot_pads_held(Q.door.ns);
  ch::host.first(tr, nullptr);
  ch::kept_pads_agree(Q.S, Q.door.ns); // re-derives the mask entries, counts matches at MSG + NSM
  lost_copies_door::strike("slot_pads_later", MSG + NSM, EDEG * Q.door.ns, true); // test hook
  // The value that the merge of the weight claims sends is masked by an entry of the request's
  // index-mask message, not of this segment's mask row, because the claim on that value leaves the
  // sub-proof as a pending claim on the weight table. The entry goes to merge_mask, where
  // kG_merge_end() reads it. The mask row's entry for that value's `slot` stays unused.
  ch::host.index_mask_entry(merge_mask);

  return 0;
}

// Runs the lookup forest, the product trees of all the sub-proof's lookup calls, then draws points.
inline int Prover::g_product_tests(int g) {
  Fork &K = FK[g];
  HFk &Q = *K.h;
  fork_memory.arena_idle();
  forest_prove_graph(*K.FG, *K.F, tr, MSG + Q.slPT);
  k_forest_err<<<1, 1>>>(K.F->claims, errc);
  fork_memory.arena_idle();
  k_point<<<1, 1>>>(tr, Round::ZERO, K.PV, lt);
  if (g == 0) {
    k_point<<<1, 1>>>(tr, Round::ZERO, K.ZE, ls + lt);
    k_point<<<1, 1>>>(tr, Round::ZERO, K.tkh, lPd - ls);
    k_cat<<<1, 64>>>(K.tkh, lPd - ls, K.ZE, ls, K.pk);
    Q.GS.preE();
    if (weight_stream.ends_by_turns) weight_stream.experts_ready(weight_stream.end_unit[0]);
    for (size_t p = 0; p < npE; p++) {
      const size_t per = (size_t)4 * Vv * d / npE;
      if (!weights->embedding_in_place()) {
        weights->embedding_cells(p * per, per / 4, (u32 *)embC);
      }
      K.jobE[p].run(rsc);
    }
    if (weight_stream.ends_by_turns) weight_stream.experts_used(weight_stream.end_unit[0]);
  }
  return 0;
}

// Builds the weight-side tables by row passes, out[r] = sum_y phi[y] cell[r L + y], over the
// weights at the drawn points. They feed the batched sumcheck (Q.GS, `stack`) and are not sent.
inline int Prover::g_weight_tables(int g) {
  Fork &K = FK[g];
  HFk &Q = *K.h;
  for (int c = 0; c < NC; c++) {
    k_eq<<<nblk((size_t)1 << K.lJ[c]), 256>>>(K.zj(c), K.lJ[c], K.ec[c]);
    k_eq<<<1, 32>>>(K.zk(c), ll, K.el[c]);
    k_times<<<1, 32>>>(K.el[c], &K.F->claims->e[c], Lam, K.ele[c]);
  }
  cD = K.g.callOf("D");
  cQK = K.g.callOf("QK");

  k_eq<<<nblk((size_t)1 << (K.lJ[cQK] - ldh)), 256>>>(K.zj(cQK), K.lJ[cQK] - ldh, K.ehq);
  k_gatherh<<<1, 64>>>(K.ehq, K.g.pos2("QK"), dh, H + Gk, K.AH);
  k_eq<<<1, 256>>>(K.zj(cQK) + K.lJ[cQK] - ldh, ldh, K.lfc);

  Q.GS.pre_rows();
  // `kind`: 0 q and k, 1 v, 2 o, 3/4 dense gating/up, 5/6 shared, 7/8 routed, 9 dense and shared
  // down, 10 routed down, 11 router. With -memory experts-by-layer or tensor-rooms the row passes
  // of 7, 8 and 10 run last, in the order in which the weight stream has the experts ready
  // (WeightStream::use_of()). Each writes only its part of a table, so the order changes no table.
  // Public sources: see weight_stream.cuh, use_of().
  std::vector<WeightKind *> routed;
  for (auto &q : K.wk) {
    if (q.kind == 0) {
      CK(cudaMemcpyAsync(q.phi, K.AH, 64 * sizeof(T), cudaMemcpyDeviceToDevice));
    } else {
      const int kd = q.kind;
      const char *sg = kd == 1                         ? "V"
                       : kd == 2                       ? "O"
                       : kd == 3 || kd == 5 || kd == 7 ? "G"
                       : kd == 4 || kd == 6 || kd == 8 ? "U"
                       : kd == 11                      ? "LR"
                                                       : "D";
      k_gather2o<<<gsl_blocks_of(q.L), 256>>>(K.ec[K.g.callOf(sg)], K.g.pos2(sg), kd == 7 || kd == 8 ? fe : 0, (int)q.L,
                                       q.phi);
    }
    if (weights->has_row_pass(q.kind, g)) {
      if (weight_stream.experts_by_layer && (q.kind == 7 || q.kind == 8 || q.kind == 10)) {
        routed.push_back(&q);
        continue;
      }
      const bool cut = (q.kind == 7 || q.kind == 8) && !Q.job2[q.kind - 7].empty();
      weights->row_pass(q.kind, g, cut ? SHg.tG : 0, cut ? Q.GS.phiLow(q.kind) : q.phi, cut ? Q.GS.tab2(q.kind) : q.tab,
                        row_pass_scratch, cudaStreamPerThread);
      continue;
    }
    const i8 *cells = weights->cells_in_place(q.kind, g);
    for (int p = 0; p < q.lm.n; p++) {
      if (cells) {
        RowJob job = q.jobs[p];
        job.src = cells + (size_t)p * (q.bytes / q.lm.n);
        job.run(rsc);
        continue;
      }
      make_weight_cells(g, q, p);
      if ((q.kind == 7 || q.kind == 8) && !(Q.job2[q.kind - 7].empty())) {
        Q.job2[q.kind - 7][p].run(rsc);
      } else {
        q.jobs[p].run(rsc);
      }
    }
  }
  for (int i = 0; !routed.empty() && i < weight_stream.experts_of(g).second - weight_stream.experts_of(g).first; i++) {
    const int e = weight_stream.use_of(g, i); // in the stream's own order
    weight_stream.experts_ready(e);
    for (WeightKind *q : routed) {
      if (!weight_stream.holds_kind(e, q->kind)) continue;
      const bool cut = (q->kind == 7 || q->kind == 8) && !Q.job2[q->kind - 7].empty();
      weights->row_pass(q->kind, g, cut ? SHg.tG : 0, cut ? Q.GS.phiLow(q->kind) : q->phi,
                        cut ? Q.GS.tab2(q->kind) : q->tab, row_pass_scratch, cudaStreamPerThread,
                        weight_stream.layer_of_experts(e));
    }
    weight_stream.experts_used(e);
  }
  Q.GS.post_rows();
  // Nothing after this line reads the routed experts' weights. If nothing later in the batch pass
  // does (`last_of_wave`), experts_done() reports that, but not during stream capture: the graph's
  // launches would skip it; weights_done() would report later. No capture outlasts pt_graphed().
  cudaStreamCaptureStatus capture = cudaStreamCaptureStatusNone;
  CK(cudaStreamIsCapturing(cudaStreamPerThread, &capture));
  if (experts_done && last_of_wave && capture == cudaStreamCaptureStatusNone) experts_done(g);
  return 0;
}

// Proves the norms' sums of squares: the sums, a product layer (K.WP), the row products' sumcheck.
inline int Prover::g_norm_sums(int g) {
  Fork &K = FK[g];
  HFk &Q = *K.h;
  fork_memory.middle_begin();
  k_eq<<<nblk(NT), 256>>>(K.PV, lt, K.eV);
  const int NS = norm_sum_count(xl), NP = norm_product_count(xl);
  // Prints two numbers that show if the fifth limb is in use. Both depend on weights and prompt, so
  // only on synthetic weights under a test seed.
  // Outside a production build, read_isk() rejects any other run that sets the variable.
  // This code does not rely on that rejection.
  if (made_up_under_a_tests_seed() && TESTENV("FORMF_STREAMS_REACH") && xl == 5) {
    unsigned long long *most = (unsigned long long *)K.Ws, host[2];
    CK(cudaMemset(most, 0, 16));
    k_absmost<<<1024, 256>>>(K.g.b.xst, (size_t)NI * d * NT, most);
    k_absmost<<<64, 256>>>((const long long *)K.g.b.sigv, (size_t)NI * NT, most + 1);
    CK(cudaMemcpy(host, most, 16, cudaMemcpyDeviceToHost));
    printf("# layer group %d: the largest stream value 2^%.2f, the largest norm's sigma 2^%.2f\n", g,
           log2((double)host[0] + 1), log2((double)host[1] + 1));
  }
  ws_of(xl, K.g.b.xst, NT, d, NI, K.Ws);
  {
    const long long *Ws = K.Ws;
    const int T_ = NT;
    if (!lost_copies_door::left_out("norm_sums")) // test hook: false in a production build
      k_tsum<<<NS * NI, 256>>>(NT, K.eV, K.sVW, [=] __device__(int q, int t) {
        return ext_from_signed(Ws[((size_t)(q / NS) * T_ + t) * NS + q % NS]);
      });
  }
  // The kernel adds each sum's mask entry, so the status is checked at once, as in g_commit().
  CK(cudaGetLastError());
  k_absorb_slots<<<1, 1>>>(tr, K.sVW, NS * NI);
  k_point<<<1, 1>>>(tr, Round::BAT_SIGMA, K.gam, 1); // its powers batch the limbs' convolution sums
  // K.sHV, the vector of the sums of squares, is never sent. The product layer starts from its sum,
  // with a fresh point on the instance bits and equal weights on the feature bits (k_rho_norms()).
  k_bg<<<16384, 256>>>(xl, K.g.b.xst, NT, lt, d, Pd, NI, K.gam, K.eV, K.A, K.sHV);
  k_rho_norms<<<1, 1>>>(tr, lNI, lPd, K.PV, lt, K.rhoV, K.halves);
  k_norms_start<<<1, 256>>>(K.rhoV, lNI, NI, Pd, K.sHV, K.halves, K.start);
  K.WP->prove(tr, K.A, K.A, K.rhoV, K.start, K.sWPL);
  k_rowterms<<<nblk((size_t)NI * NT), 256>>>(xl, K.g.b.Rv, K.g.b.sigv, NI * NT, NT, K.FV, K.GV);
  if (!lost_copies_door::left_out("norm_sums")) k_pvsum<<<NP * NI, 256>>>(NT, K.eV, K.FV, K.GV, K.sPV);
  CK(cudaGetLastError()); // as for the sums above
  k_absorb_slots<<<1, 1>>>(tr, K.sPV, NP * NI);
  k_batching_powers<<<1, 1>>>(tr, K.omV, NP * NI);
  K.tV.prove(tr, K.PV, K.omV, K.FV, K.GV, NP * NI, K.sTV);
  K.tVend.run(tr);
  return 0;
}

// One sumcheck over tokens, sum_t eq(zt, t) sum_m co[m] Fv[m][t] Gv[m][t], with the terms m of
// all lookup calls that have any (norms, rotary embedding, routing). zt: the last lt coordinates of
// the forest's point. co[m] includes e_c, call c's coefficient in the forest's join.
inline int Prover::g_token_sumchecks(int g) {
  Fork &K = FK[g];
  if (K.ts.empty()) return 0;
  TokenSumcheck &q = K.ts[0];
  int ti = 0;
  for (int c : tscCalls) {
    bool g1 = false, g2 = false;
    for (auto &s : policy.cl[c].seg) {
      if (s == "N1") {
        g1 = true;
      }
      if (s == "N2") {
        g2 = true;
      }
    }
    if (g1) {
      k_gather2<<<nblk(d), 256>>>(K.ec[c], K.g.pos2("N1"), d, K.ewN[0]);
    }
    if (g2) {
      k_gather2<<<nblk(d), 256>>>(K.ec[c], K.g.pos2("N2"), d, K.ewN[1]);
    }
    for (int lam = 0; lam < Lam; lam++) {
      for (auto &s : policy.cl[c].seg) {
        const int l = g * Lam + lam;
        if (s == "N1" || s == "N2") {
          const int one = s == "N1", sg = NW * lam + (one ? 0 : NW - 1), inst = NN * lam + (one ? 0 : NN - 1);
          k_gn1<<<dim3((NT + 127) / 128, 1, 32), dim3(128, 4)>>>(K.g.xs(sg), NT, d, K.ewN[one ? 0 : 1], K.gnp);
          k_gn2<<<nblk(NT), 256>>>(K.gnp, 32, K.g.b.Rv + (size_t)inst * NT, NT, q.Fv + (size_t)ti * NT,
                                   q.Gv + (size_t)ti * NT);
          k_scale<<<1, 1>>>(K.ele[c], lam, 0, q.co + ti, 1);
          ti++;
        }
        if (s == "QK") {
          const WeightKind &wq = K.wk[0];
          k_bcs<<<nblk((size_t)d * dh / 2), 256>>>(wq.tab, wq.ntab / 2, lam, d, dh, K.lfc, K.Cm);
          k_limb9<<<limb9_blocks((size_t)dh * d), 256>>>(K.Cm, (size_t)dh * d, d, K.Bl);
          for (int i = 0; i < 2; i++) {
            transpose_i8(K.g.cell(X_a, i, lam), K.aT[i], d, NT, cudaStreamPerThread);
            prover_gemm(blas::blas(cudaStreamPerThread, K.blas_room), K.aT[i], K.Bl, K.Cg[i], NT, dh * 8 * EDEG, d, 0,
                   cudaStreamPerThread);
          }
          k_recomb<<<(unsigned)(((size_t)dh * NT + 255) / 256), 256>>>(K.Cg[0], K.Cg[1], dh, NT,
                                                                       q.Gv + (size_t)ti * NT);
          k_ropeF<<<nblk((size_t)dh * NT), 256>>>(K.g.b.Cr, K.g.b.Sr, K.g.kl[l].rp, NT, dh / 2, q.Fv + (size_t)ti * NT);
          const int kq = (int)KL[(size_t)KLW * l];
          k_scale<<<(dh + 127) / 128, 128>>>(K.ele[c], lam, kq < 16 ? 16 - kq : 0, q.co + ti, dh);
          ti += dh;
        }
        if (is_moe_layer(l)) {
          const int lm = lam / 2;
          const int *es = K.g.b.es + (size_t)lm * NT;
          auto wkOf = [&](int kd) -> const WeightKind & {
            for (auto &w : K.wk) {
              if (w.kind == kd) {
                return w;
              }
            }
            printf("internal error: no weight table found for weight kind %d\n", kd);
            name_the_end("guard"); // serving: p258 end 1 guard (kept as an exit(2)); only a routed-experts model reaches this
            exit(2);
          };
          auto tOf = [](int k) {
            return k < 16 ? 16 - k : 0;
          };
          T *Fq = q.Fv + (size_t)ti * NT, *Gq = q.Gv + (size_t)ti * NT;
          auto vecs = [&](const WeightKind &w, int Kd, int np) {
            // G_e[t] = sum_kk b(lm, e, kk) A[t][kk], computed by limb GEMMs.
            // A's np limbs must already be in aT.
            k_bjoin<<<gsl_blocks_of((size_t)NE * Kd), 256>>>(w.tab, w.ntab / 2, (size_t)lm * NE * Kd, (size_t)NE * Kd, K.Cm);
            k_limb9<<<limb9_blocks((size_t)NE * Kd), 256>>>(K.Cm, (size_t)NE * Kd, Kd, K.Bl);
            for (int i = 0; i < np; i++) {
              prover_gemm(blas::blas(cudaStreamPerThread, K.blas_room), K.aT[i], K.Bl, K.Cg[i], NT, NE * 8 * EDEG, Kd, 0,
                     cudaStreamPerThread);
            }
            k_recombN<<<(unsigned)(((size_t)NE * NT + 255) / 256), 256>>>(K.Cg[0], K.Cg[1], K.Cg[2], K.Cg[3], np, NE,
                                                                          NT, Gq);
          };
          if (s == "G" || s == "U") {
            for (int i = 0; i < 2; i++) {
              transpose_i8(K.g.cell(X_xr, i, lm), K.aT[i], d, NT, cudaStreamPerThread);
            }
            vecs(wkOf(s == "G" ? 7 : 8), d, 2);
            k_selF<<<gsl_blocks_of((size_t)NE * NT), 256>>>(es, NT, NE, Fq);
            k_scale<<<(NE + 127) / 128, 128>>>(K.ele[c], lam, tOf(s == "G" ? K.g.kl[l].kgR : K.g.kl[l].kuR), q.co + ti,
                                               NE);
            ti += NE;
          }
          if (s == "D") {
            k_hl4<<<dim3(NT / 32, fe / 32), dim3(32, 8)>>>(K.g.b.KH + ((size_t)lam * Pf + fe) * NT,
                                                           K.g.b.KH + (((size_t)Lam + lam) * Pf + fe) * NT, NT, fe,
                                                           K.aT[0], K.aT[1], K.aT[2], K.aT[3]);
            vecs(wkOf(10), fe, 4);
            k_selF<<<gsl_blocks_of((size_t)NE * NT), 256>>>(es, NT, NE, Fq);
            k_scale<<<(NE + 127) / 128, 128>>>(K.ele[c], lam, tOf(K.g.kl[l].kd), q.co + ti, NE);
            ti += NE;
          }
          if (s == "XR") {
            k_gather2<<<gsl_blocks_of(d), 256>>>(K.ec[c], K.g.pos2("XR"), d, K.ewX);
            k_gn1c<<<dim3((NT + 127) / 128, 1, 32), 128>>>(K.g.cell(X_m, 0, lam), K.g.lst(X_m), NT, d, K.ewX, K.gnp);
            k_gsum<<<gsl_blocks_of(NT), 256>>>(K.gnp, 32, NT, Gq);
            k_cellv<<<gsl_blocks_of(NT), 256>>>(K.g.cell(X_sc, 0, lm), K.g.lst(X_sc), 3, NT, Fq);
            k_scale<<<1, 1>>>(K.ele[c], lam, 0, q.co + ti, 1);
            ti++;
          }
          if (s == "DD" || s == "MR") {
            k_selF<<<gsl_blocks_of((size_t)NE * NT), 256>>>(es, NT, NE, Fq);
            k_cellv<<<gsl_blocks_of((size_t)NE * NT), 256>>>(K.g.cell(X_lr, 0, lm), K.g.lst(X_lr), 2, (size_t)NE * NT, Gq);
            k_scalew<<<1, 256>>>(K.ele[c], lam, K.ec[c], K.g.pos2(s.c_str()), s == "DD" ? NE : 1, q.co + ti, NE);
            ti += NE;
          }
        }
      }
    }
  }
  if (ti != q.NTm) {
    printf("internal error: the sum-check over tokens has %d terms, but %d were planned\n", ti, q.NTm);
    name_the_end("token-terms"); // the set-up counterpart's name; serving: p258 end 1 token-terms (kept as a return 4)
    return 4;
  }
  q.eng.prove(tr, K.zx_last(lt), q.co, q.Fv, q.Gv, q.NTm, q.msg);
  q.end.run(tr);
  return 0;
}

// Runs the batched inner-product sumcheck. Before it: the row passes at the token points, whose
// results its vectors contain, and the gated products' token sums (K.sHH, K.sHH2), never sent.
inline int Prover::g_stack(int g) {
  Fork &K = FK[g];
  HFk &Q = *K.h;
  for (auto &p : K.SPn) {
    const T *pt;
    if (p.first == "P_V") {
      pt = K.PV;
    } else if (p.first == "r_tV") {
      pt = K.tV.out->r;
    } else if (p.first == "z") {
      pt = K.zx_last(lt);
    } else {
      pt = K.ts[0].eng.out->r;
    }
    k_eq<<<nblk(NT), 256>>>(pt, lt, K.phiP[p.first]);
  }
  for (size_t j = K.short_jobs_from; j < K.jobs.size(); j++) K.jobs[j].run(rsc);
  k_eq<<<nblk(NT), 256>>>(K.zt(cD), lt, K.etD);
  k_hh3<<<8192, 256>>>(K.g.b.KH, NT, Lam, f, Pf, K.etD, K.sHH);
  const T *rD = nullptr;
  if (mo) {
    rD = K.ts[0].eng.out->r;
    k_kh2<<<gsl_blocks_of((size_t)LamM * fe * NT), 256>>>(K.g.b.KH, NT, Lam, Pf, LamM, fe, K.KH2);
    k_eq<<<gsl_blocks_of(NT), 256>>>(rD, lt, K.etD2);
    k_hh3<<<8192, 256>>>(K.KH2, NT, LamM, fe, fe, K.etD2, K.sHH2);
  }
  layer_lie_element("gated", g, K.sHH, (size_t)Lam * f + (size_t)(mo ? LamM : 0) * fe); // test hook
  Q.GS.run_stack(tr);
  return 0;
}

// Proves the feed-forward blocks' gated products, one product layer each, from the values of their
// token sums at the batched sumcheck's end point; then merges the weight claims.
inline int Prover::g_gated_products(int g) {
  Fork &K = FK[g];
  HFk &Q = *K.h;
  // Q.GS.uv[0] (H1): the token sums K.sHH at the point of ll + lPf index bits, the last coordinates
  // of the sumcheck's point Q.rrS, top bit first. 32768: KH stores the factors plus 2^15.
  k_cat<<<1, 64>>>(Q.rrS + (Q.lamS - (ll + lPf)), ll + lPf, K.zt(cD), lt, K.rhoH);
  K.HL->prove(tr, K.g.b.KH, K.rhoH, F::small(32768), Q.GS.uv, K.sPL);
  if (mo) {
    // uv + 1 (the X_lamM, one per MoE layer): K.sHH2 at the point of lfe column bits. They are
    // joined at a fresh point of ll - 1 layer bits.
    if (ll > 1) k_point<<<1, 1>>>(tr, Round::JOIN, K.rhoH2, ll - 1);
    k_join_at<<<1, 1>>>(K.rhoH2, ll - 1, Q.GS.uv + 1, LamM, K.start);
    k_cat<<<1, 64>>>(Q.rrS + (Q.lamS - lfe), lfe, K.ts[0].eng.out->r, lt, K.rhoH2 + (ll - 1));
    K.HL2->prove(tr, K.KH2, K.rhoH2, F::small(32768), K.start, K.sPL2);
  }
  Q.GS.run_merge(tr);
  return 0;
}

// Builds the equality tables at the long points (la = ls + lt coordinates), runs the remaining row
// passes over the request's cells and multiplicity bits, and commits to the reads U3 (Q.vec[3]).
inline int Prover::g_row_passes(int g) {
  Fork &K = FK[g];
  HFk &Q = *K.h;
  for (auto &p : K.LPn) {
    const T *pt;
    if (p.first == "Z:E") {
      pt = K.ZE;
    } else if (p.first == "Z:V") {
      pt = K.WP->out->r + (lNI + lPd + lt - la);
    } else if (p.first == "Z:a") {
      pt = K.HL->out->r + (ll + lPf + lt - la);
    } else if (p.first == "Z:b") {
      pt = K.HL2->out->r + (ll - 1 + lfe + lt - la);
    } else if (p.first == "Z") {
      pt = K.zx_last(la);
    } else { // Y = (zx's coordinates dT - la .. dT - lt - 1 | the token sumcheck's r)
      k_cat<<<1, 64>>>(K.zx_last(la), ls, K.ts[0].eng.out->r, lt, K.ts[0].yb);
      pt = K.ts[0].yb;
    }
    k_eq<<<nblk(LA), 256>>>(pt, la, K.phiP[p.first]);
  }
  {
    const size_t mb = (size_t)1 << lmb;
    for (int e = 0; e < 2; e++)
      hw::k_forest_phi_bits<<<nblk(mb), 256>>>(K.F->claims->zb[e], layer_forest.kT, layer_forest.cK, lmb, K.phiB[e]);
    hw::k_forest_phi_powers<<<nblk(mb), 256>>>(lmb, K.phiB[2]);
  }
  for (size_t j = 0; j < K.short_jobs_from; j++) K.jobs[j].run(rsc);
  for (auto &j : K.jobsB) {
    j.run(rsc);
  }
  layer_lie_element("read", g, Q.vec[3], K.nW); // test hook
  commit_unit(Q.U[3], Q.vec[3], Q.roots + 3 * DW, tr, 1);
  return 0;
}

// The approximate range check (`HiddenTail`): combines the committed matrices' columns with the
// verifier's coefficients, commits to the results' digits and runs a separate lookup forest.
inline int Prover::g_fold(int g) {
  Fork &K = FK[g];
  HFk &Q = *K.h;
  HiddenTail &HT = Q.HT;
  const int NM = (int)K.M.size();
  if (int rc = test_fault("attempt")) return rc;
  // late_begin() requires that nothing reads the shared engines' scratch memory from here on.
  fork_memory.late_begin();
  // The range check's matrix: Cf[c][u] in {-1, 0, 1}, column c of matrix i, combination u < ETAU.
  HT.draw_matrix(); // a round that the host runs, after the reads' out-of-domain samples
  for (int i = 0; i < NM; i++) zfold_prep<ETAU>(K.M[i].Cf, (int)K.M[i].k, K.M[i].CfT);
  for (int i = 0; i < K.NR; i++) {
    zfold_turned_i8<ETAU>(K.M[i].v, K.M[i].m, (int)K.M[i].m, (int)K.M[i].k, K.M[i].CfT, zfold_kp((int)K.M[i].k),
                          K.M[i].Z, K.zfold_work);
  }

  HT.fold_to_group2(tr, err, Q.zs);
  HT.group2(tr, MSG + Q.slPZ);
  // This forest's counts depend on the random matrix, so its error word goes to err (errc: failures
  // that depend only on inputs). No overflow is reported: the word is 0 (ForestTable::planes()).
  k_forest_err<<<1, 1>>>(HT.F.claims, err);
  return 0;
}

// After the sub-proof's last message: draws the challenge that batches all its linear claims
// (hw::Walk) and adds them up into one equation's coefficient vectors. Nothing more is sent.
inline int Prover::g_walk(int g) {
  Fork &K = FK[g];
  HFk &Q = *K.h;
  HiddenTail &HT = Q.HT;
  HT.make_cv();
  CK(cudaMemsetAsync(Q.Lall, 0, Q.nLall * sizeof(T)));
  Q.door.clear_async();
  Q.W.finish(tr, Q.door.LS0);
  return 0;
}

// An injected fault: adds 1 to a cell.
__global__ void k_change_a_cell(i8 *cell) {
  *cell += 1;
}

// Saves for the attention sub-proof: q, k, v as 16-bit numbers (Aq, Ak, Av) and qa's cells (Aqa).
inline void Prover::copy_qkv_for_attention(int g) {
  Fork &K = FK[g];
  CK(cudaMemcpyAsync(Aqa, K.g.b.qa, cells_of(K, K.qa), cudaMemcpyDeviceToDevice, cudaStreamPerThread));
  for (int lam = 0; lam < Lam; lam++) {
    k_c2t<<<grid_blocks((size_t)NT * dq), 256>>>(K.g.qcell(0, 0, lam), K.g.qlst(0), NT, dq, Aq + (size_t)lam * NT * dq);
    k_c2t<<<grid_blocks((size_t)NT * dkv), 256>>>(K.g.qcell(1, 0, lam), K.g.qlst(1), NT, dkv, Ak + (size_t)lam * NT * dkv);
    k_c2t<<<grid_blocks((size_t)NT * dkv), 256>>>(K.g.qcell(2, 0, lam), K.g.qlst(2), NT, dkv, Av + (size_t)lam * NT * dkv);
  }
}

// Passes the sub-proof's one linear equation over its committed arrays to the segment's closing.
// Before it, three self-checks on mask entries. A difference is only recorded in a count or a flag:
// finish_request() reads them, and the request gets no proof.
// Limits: hz::Tree::slot_pads_agree() (prover/hiding/trees.cuh).
inline int Prover::g_equations_and_columns(int g) {
  Fork &K = FK[g];
  HFk &Q = *K.h;
  HiddenTail &HT = Q.HT;
  // The equation is a list of terms with their coefficient vectors. hd::close(), below, runs the
  // closing on tr: it reduces the segment's linear claims to one, carried into the next segment.
  Q.door.hS = MSG + NSM;
  slot_counts_end("layer sub-proof", g, MSG, Q.door.ns);
  Q.door.host_words = ch::host.blob();
  // Before the closing, which may overwrite the forward pass's lookup keys and cells.
  copy_qkv_for_attention(g);
  Q.S.slot_pads_agree(MSG + NSM, Q.door.ns); // re-derives the mask entries after their last use
  ch::masking_pads_hold(MSG + NSM, Q.door.ns); // by a second implementation [own-test-slot-pads]
  ch::host.index_mask_entry_holds(merge_mask); // also compares the index-mask entry in merge_mask
  if (int rc = hd::close(Q.door, tr, "layer sub-proof", g)) return rc;
  return 0;
}

// Call after attention's forward pass: sets bit 3 of attention.errors() if its qa differs from Aqa.
inline void Prover::compare_attention_cells(AForkHid &attention) {
  if (g_slot_entered && fork_memory.test.changed_cell_seat == g_slot_entered->j)
    k_change_a_cell<<<1, 1>>>(attention.A.cell[A_QA] + attention.A.ncell[A_QA] - 1);
  attention.compare_qa_with(Aqa);
}

// Proves group g's attention sub-proof. AG[g].o serves all layer groups of one attention shape.
inline int Prover::prove_attention_fork(int g) {
  {
    AForkHid &Hd = *AG[g].o;
    stage_clock.at(StageClock::Forward);
    fork_memory.run_begin(NGr + 1 + (int)(std::find(AO.begin(), AO.end(), AG[g].o) - AO.begin()));
    Hd.begin_fork(AG[g].h);
    chain_bind(chain.attention(g));
    Hd.door.link = chain.attention(g);
    ch::host.blob = [&Hd] {
      return Hd.assemble();
    };
    lie::here(g * Lam, lie::ATTENTION, cudaStreamPerThread); // test hook: the group's first layer
    Hd.phase1(Aq, Ak, Av);
    compare_attention_cells(Hd);
    // As in g_commit(): counts the zero filler cells (`zero_fill`) before the first commitment.
    for (int i = 0; i < 5; i++)
      if (const size_t n = Hd.A.zero_fill[i])
        k_count_zero_bytes<<<32, 256>>>(Hd.A.cell[i] + Hd.A.ncell[i] - n, n, filler_is_zero);
    CK(cudaGetLastError());
    stage_clock.at(StageClock::Attention);
    Hd.prove(tr, nullptr); // continues the request's transcript tr
    fork_memory.run_end();
  }
  return 0;
}
