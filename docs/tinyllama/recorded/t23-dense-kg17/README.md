# Recorded T2.3 dense synthetic proof (kg=17)

Verifier-checked dense synthetic proof from 2026-11-06, outside the Lean
theorem. A dense YES needs the later verifier policy change (standard form),
which is not in this branch.

This package includes `ivk/`, so you can run `cp_verify` without running
`cp_commit` or `cp_prove`.

## Recorded run

- Branch: `feature/tinyllama-proof`
- Code commit archived onto the pod: `7596dd8c`
- Host: NVIDIA H100 80GB HBM3, nvcc 13.2.86, driver 595.91.07, test build
- Kit: NE 0, L 8, d 512, T 64, no gcl, kg=17 on every layer
- Commitment ID: `75b0835c557021f727a964a7a3431fd5bfc4d300d03f37036dbc8ffd622cbae4`
- `out/example.proof`: 4121052 bytes
- `ivk/`: `key.bin` (1059440 bytes) and `leaf/0.bin` (537248 bytes)
- `seed`: 32 bytes (the urandom `-test-seed` used for this run)
- `cp_verify` on that host exited 14 (`NOT-A-VERIFICATION`) with
  `not the standard parameter form: routed-experts, both-clamps, table-digests`

See `SHA256SUMS` for file hashes. `recorded-verify.log` is the verifier
output from that host.

An earlier proof from the same day (ID
`1885474735213c0bbb40d79806ff11f78876708010c78953950c4ee545a83b2d`) is not
in this directory. Its `ivk/` was not saved, so it cannot be verified.

## Verify without running the prover

From the repository root, build the test verifier (CPU is enough; no GPU):

```
./verifier/build.sh
cd prover && ./build.sh tests && cd -
BUILD=prover/build/hc/test_build
```

Then:

```
REC=docs/tinyllama/recorded/t23-dense-kg17
ID=$(tr -d ' \n' < "$REC/commitment-id.txt")
"$BUILD/cp_verify" "$REC/kit" "$REC/ivk" "$REC/in/example.req" \
  "$REC/out/example.y" "$REC/out/example.proof" -id "$ID"
```

Expected: exit 14, `NOT-A-VERIFICATION`, and
`not the standard parameter form: routed-experts, both-clamps, table-digests`.
No YES (a dense YES needs the later policy change).

## Run the prover (optional, needs an H100)

To produce a new proof, follow `docs/tinyllama/PROOF_PREP.md` section
"Confirming T2.3 on the next H100". You will get a new commitment ID and a
new proof. Do not pre-create `ivk/`, `isk/`, or `out/`. Do not use an
all-zero test seed; use `head -c 32 /dev/urandom`. Copy `ivk/`,
`in/example.req`, `out/`, the seed, and the commitment ID off the machine
before stopping it.
