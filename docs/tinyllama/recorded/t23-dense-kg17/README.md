# Recorded T2.3 dense synthetic proof (kg=17)

Verifier-checked dense synthetic proof from 2026-11-06, outside the Lean
theorem. A dense YES needs the later verifier policy change (standard form),
which is not in this branch.

## Recorded run

- Branch: `feature/tinyllama-proof`
- Code commit that produced the proof: `d34836bb7e1bc2f18d1193f61023040fd2007099`
- Host: NVIDIA H100 80GB HBM3, nvcc 13.2.86, driver 595.91.07, test build
- Kit: NE 0, L 8, d 512, T 64, no gcl, kg=17 on every layer
- Commitment ID: `1885474735213c0bbb40d79806ff11f78876708010c78953950c4ee545a83b2d`
- `out/example.proof`: 4113324 bytes
- `cp_verify` on that host exited 14 (`NOT-A-VERIFICATION`) with
  `not the standard parameter form: routed-experts, both-clamps, table-digests`

See `SHA256SUMS` for file hashes.

`in/example.req` was rebuilt after the run from `kit/policy.txt` with
`prover/tools/make_request.py`. It was not copied off the pod.

## Verify without running the prover (this file)

`cp_verify` needs:

```
cp_verify CONFIG IVK REQUEST Y PROOF -id ID
```

This directory has CONFIG (`kit/`), REQUEST (`in/example.req`), Y, PROOF, and
the ID. It does **not** include the `ivk/` directory written by that
`cp_commit`. The test seed was 32 bytes from `/dev/urandom` and was not saved,
and the pod disk was wiped on stop, so that commitment cannot be reconstructed.
Without that `ivk`, you cannot finish verify of this exact proof.

Build the test verifier from the repository root (CPU is enough):

```
./verifier/build.sh
cd prover && ./build.sh tests && cd -
BUILD=prover/build/hc/test_build
```

The command you would run, once `ivk/` is present, is:

```
ID=$(tr -d ' \n' < docs/tinyllama/recorded/t23-dense-kg17/commitment-id.txt)
REC=docs/tinyllama/recorded/t23-dense-kg17
"$BUILD/cp_verify" "$REC/kit" "$REC/ivk" "$REC/in/example.req" \
  "$REC/out/example.y" "$REC/out/example.proof" -id "$ID"
```

Expected if `ivk/` were present (this is what the original host printed):
exit 14, `NOT-A-VERIFICATION`, and
`not the standard parameter form: routed-experts, both-clamps, table-digests`.
No YES (a dense YES needs the later policy change).

## Run the prover (optional, needs an H100)

To produce a new proof you can verify without relying on these files, follow
`docs/tinyllama/PROOF_PREP.md` section "Confirming T2.3 on the next H100".
You will get a new commitment ID and a new proof. Copy `ivk/`,
`in/example.req`, `out/example.y`, `out/example.proof`, the seed, and the
commitment ID off the machine before stopping it.

Do not pre-create `ivk/`, `isk/`, or `out/`. Do not use an all-zero test seed
(`cp_prove` refuses it with `hc-key-file`); use `head -c 32 /dev/urandom`.
