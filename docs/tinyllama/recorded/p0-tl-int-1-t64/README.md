# Recorded TL-INT-1 P0 T=64 verify package (M5m YES)

This is a C++ `cp_verify` YES of TL-INT-1 prompt P0 at T 64: exact-match
of request token 24, which is 1576 (P0's greedy next token). It is not a
Lean claim or theorem. This package includes the verification key and needs
neither the prover nor a GPU to verify.

## Provenance and scope

- Proof-producing code: `1c5c2aa0` (`derive kFM like kFD`).
- Kit policy: `kFM 14`, `kFD 15`, `S 12`, `rx 8`, `NE 0`.
- Artifact id: `1d8e5cf46c89d951c1c449dc3c03d963f32176360a016d78e4d2537d39b088a9`.
- Commitment ID: `d319bc487793a0682821d7e47f1a0cf0cb4e5ee674b34749c65e2d804812c227`.
- Run date: 2026-11-07, Asia/Almaty; pod `velpk1d1gd2s5z`, H100 SXM, US-MO-1.
- `cp_prove`: RC 0 in 8.17 s; last line `p258 end 0`.
- `cp_verify`: RC 0 in 4.03 s; last line `YES`.
- Recorded result fields: status=0, claim=exact-match, tokens=24..24, y=no,
  bits=128, least-round=128.00 (96.23 without work), r=23, e=24.
  These are quoted verifier fields, not an independent security-level claim.

`out/p0.y` is four zero bytes. The token lives in the request (index 24 =
1576); the H100 verifier call still passed `p0.y` as shown below. The verifier
printed `y=no` and `next-token=no`: neither was a separate YES.

Public-weights: `isk.txt` says `seed public`; weights are not hidden.
The worst-case check was left out (decision (f), option 2). The verifier said
`weight hiding: NOT CHECKED BY THIS CALL (it rests on how the commitment was made)`.
No stronger hiding or worst-case claim is made.

## Verify without the prover (CPU only)

From the repository root:

```sh
./verifier/build.sh
REC=docs/tinyllama/recorded/p0-tl-int-1-t64
ID=$(tr -d ' \n' < "$REC/commitment-id.txt")
bin/cp_verify "$REC/kit" "$REC/ivk" "$REC/in/p0.req" \
  "$REC/out/p0.y" "$REC/out/p0.proof" -id "$ID"
```

Expected: exit 0, last line `YES`. The saved H100 verifier output is
`recorded-verify.log`; its actual result line is:

```text
result-v1 status=0 claim=exact-match id=d319bc487793a0682821d7e47f1a0cf0cb4e5ee674b34749c65e2d804812c227 bits=128 work=32 work-rounds=56 least-round=128.00 least-round-without-work=96.23 commitment-assumed-rounds=60 record=1 r=23 e=24 T=64 tokens=24..24 y=no policy=exact-match C=256 theta=256 cap=0 s=17 sh=8 sT=4 F=24 above-clip=0 leaf=0 stages=key,leaf,request dw=8 build=04252d966c4310f06fedefffc6b23bfb6dd7014235052ca68214a4f7c9f30aae
```

The archived H100 `tl/bin/cp_verify` was also run on the Linux box against
these new package paths with `-id`: exit 0, last line `YES`. All copied files
were SHA-256 matched against the M5m verify bundle; none were regenerated.

## Copied file sizes

| File | Bytes |
| --- | ---: |
| `commitment-id.txt` | 64 |
| `in/p0.req` | 264 |
| `isk.txt` | 68 |
| `ivk/key.bin` | 2043400 |
| `ivk/leaf/0.bin` | 2217912 |
| `kit/KL.bin` | 1144 |
| `kit/layout.txt` | 124606 |
| `kit/policy.txt` | 2283 |
| `out/p0.proof` | 15608900 |
| `out/p0.y` | 4 |
| `recorded-verify.log` | 26434 |
| `seed` | 32 |

`SHA256SUMS` covers every packaged file except itself (including this README).
Check from the repository root:

```sh
(cd docs/tinyllama/recorded/p0-tl-int-1-t64 && sha256sum -c SHA256SUMS)
```
