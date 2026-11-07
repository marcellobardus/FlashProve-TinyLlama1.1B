# The parameter file (`policy.txt`)

The parameter file is a text file of named numbers. With `KL.bin` (the quantization scales) and `layout.txt` (where each committed array lies in the rows that a proof commits to) it is one of the three *configuration files*. Together they fix the shape of the model, its integer arithmetic, every parameter of the proof and the acceptance policy. You need this page if you write or check such a file, or if a run ended with a name that begins `version-`, `parameter-`, `policy-`, `shape-`, `forest-`, `af-`, `no-wrap-` or `table-digest-`.

Paths are relative to the tree's top. "The program" is `cp_verify`, "you" are the client who runs it. Complete files: `prover/examples/target_small_f2/policy.txt` and `target_full_f2/policy.txt` (`prover/examples/TARGET-KITS.md`). The integer model is defined in `docs/model.md`, the sub-proofs in `docs/format/11-sub-proofs.md` and the five parts beside it, and the file's first lines in `docs/format/08-parameters.md`; for everything outside the sub-proofs the code is `blocks/`.

## Who writes this file, and who holds it

**The provider writes it.** The party that commits to the weights also writes the three configuration files. The scales are chosen for these weights; the shape, the lines of the proof system and the acceptance policy are the provider's choices. There is no trusted set-up.

**You hold your own copy, and you name the ID.** The ID (printed in the line `commitment ID: `) is one hash of the digest of the three files, of the name of the first committed table in `key.bin` (its root and shape) and of that table's record's seeds and answers (`blocks/frame/heads.h`, `published_id`; to the byte: docs/format/FORMAT.md, R4.11). The programs, their files and the format say "the record" for a table's published out-of-domain (OOD) answers (masked sums of the weights). They say "the record's seeds" for its OOD seeds ([glossary](../../docs/glossary.md#record)). The seeds stand in no file: anyone forms them from the name by hashing (R4.8). Take it from where the commitment is published, before you send the prompt, and never from the files that you were sent: an ID that is read off the very files that are judged shows nothing. Pass it with `-id`. Each stage forms the ID of your files for itself, once it has made its plans from the three configuration files and has read the first bytes of `key.bin` for their form, and compares it before it looks at anything else of key, entry, request or proof; another ID ends the call with `NO: model-statement`. A whole call without `-id` runs all three stages and ends at best with `NOT-A-VERIFICATION` (`id-named`); before that ending it skips four late checks that the program makes on its own lines on work and policy ([output.md](output.md)). The ID does not cover the entries `leaf/<k>.bin`, further stored copies of the table, which may come from anyone: the stage leaf ties each to the key's table. If a prover supplies its own configuration files, key and ID, a call accepts for the model and the policy that those files describe.

**What the program does about an author whom it does not trust.**

- *Form.* A file has one spelling and one order of its lines, and every byte enters the digest that all hash chains bind ("Digests of the files"). So the ID stands for exactly these values.
- *Strength.* No line states a number of opened positions, and one alone a number of random coordinates: the fourth number of `table`, the random entries of a stored copy's rows, which is held against the columns that are opened of a copy (`parameters-not-admitted: hiding: ..`). All others are derived from the target `sec` and the shapes (`blocks/frame/derive.h`), so no file can ask for a list that is too short for its own target. Before a byte of a proof is used, the program forms the error of every round of every hash chain from the plan that the run will walk (`blocks/numbers/accounting.h`) and asks that each, work counted, is at most 2^-sec, and that the bits of YOUR call (`-bits`, 128 without it), which stand in no file, do not exceed `sec` (`security-bits`). Below 128 asked bits a call ends at best with `NOT-A-VERIFICATION`.
- *Proof of work.* A call that is not refused, and in which some round holds only with its bits of work, says so in a line near its end (a part-call: the line before the last; a whole call: the third from the end): `proof of work: up to <n> bits counted in <m> rounds; with it the least round has <a> bits, without it <b> bits` (`1 round` where it is one). `<a>` and `<b>` are the bits of the round with the largest error, once with the work counted and once with it left out: not always the same round. Each is a single round's figure, and neither is a security level. Unless the call has `-no-proof-of-work` (it then ends `NOT-A-VERIFICATION`, 14), counted work changes neither the last line nor the status: they are those of any other call that is not refused (`YES` or `WITHIN LIMITS: ..`, 0; a part-call's `PART PASSED: ..`, 3; `NOT-A-VERIFICATION`, 14). Where no round needs work the line reads `proof of work: none counted`.
- *Laxer choices.* Wherever a line lets the writer choose something laxer than what a plain `YES` means, a condition of the standard parameter form is owed (below); where every other choice is stricter, none. One kind of line is an exception, at which no condition of that form looks: the five lines `work` (the program only caps them: more than 32 bits of work in a round are refused, `policy-work-most`). Through its numbers the file also decides on which source some closeness rounds rest, and no condition tests that; the committer's files rest on the theorem on codes (Theorem 4.6 of the full version, ePrint 2025/2055, of the paper by Ben-Sasson, Carmon, Haböck, Kopparty and Saraf published at STOC 2026) (below, the row on `record`); the lines `accounting: one-time files: ..` and `commitment: ..` show it.
- *The acceptance policy.* With theta = C and cap = T any tokens of the request pass. So a call that covers a token of the request is judged only under the exact match, unless you name the file's own policy with `-policy`; otherwise it ends `NO: policy` before any stage. So does a call that names one kind of policy which bars no token (theta = C, and cap not below the number of covered tokens), and every whole call under a file whose `sT` is not `sh` - 4. That kind is not the only one: a policy that cp_verify does not reject can still let every token through, and such a call ends `WITHIN LIMITS` ([../../docs/limits.md](../../docs/limits.md#within-limits), "Worst case"). A pass in which a token of the request was judged under a named policy other than the exact match ends `WITHIN LIMITS: ..`, not `YES`; a request that covers the output token alone ends `YES`. The lines `covered:` and `result:` repeat its numbers.

**What it leaves to you.**

- *Which model.* The program cannot know whether shape and scales describe the model that you have in mind, or a faithful integer version of it. It tests ranges, its no-overflow conditions at the file's own values, and `norm-shifts` (K <= M). That shuts out every setting at which a norm's output is zero for every input, and not every setting that turns the layers off: settings that meet it can still round every ordinary input to zero ("Scales, and lines without a range check", below: "What norm-shifts does not cover").
- *Time and memory.* The program sets itself no limit on either, and reads every file whole, but for a request file or a token file of a wrong length, which a whole call rejects before it opens it. The ID needs the table's shape, so it is compared only after the plans have been made from the three files. Run a planning call, a call without `-id`, and any call on configuration files from a source that you do not know under limits of your own.

## Terms

- A *call* checks one proof file in three *stages*, each a process of its own: key (`key.bin`), leaf (the entry that the proof names) and request. All three read the parameter file out of your config folder. A `-plan` call works out shapes and errors from the public inputs, checks nothing and ends with status 4. `verifier/README.md` describes the checking calls and their endings; the planning calls with all their options stand in the program's usage text.
- A *sub-proof* is one part of a request's own proof: a *layer sub-proof* covers the norms, projections, router and feed-forward networks of one block of layers, an *attention sub-proof* its attention, the *sub-proof of the output head* the final norm and the logits. A sub-proof reduces the equations of its part of the model to linear claims on committed values; each is one *link* of the request's chain, which proves these claims (`blocks/iop/req.h`).
- A *lookup call* shows that every *key* of a list occurs among the rows of a public table. No key is committed: each is computed from committed values. The keys come in named lists, the call's *segments*.
- The *statement* is the list of 32-bit words that the program builds from the public inputs, your request and y.

Names with more than one meaning. "Call": a lookup call and its call line, or a run of the program. "Policy": in file, option and refusal names the whole parameter file; the *acceptance policy* is only the lines `C` to `F`. `sh`: the line is the policy's offset exponent; in `KL.bin` sh is a layer's attention shift. `hr`: a digit form (table 3), and a matrix of the output head (table 8). `kFD` is a line, `KFD` a kind of table. `theta` belongs to the policy, theta0 is the rotary base. "Ledger": the sources' word for a norm's row of digits, which has nothing to do with the accounting of errors.

## Syntax

A file has two parts. Two files that differ in a byte are two parameter sets.

**1. The frame's lines**, from `version` to `table`, are read by one strict parser for provers and verifier (`blocks/frame/parameters.h`; its head lists every line with its meaning). A line is a name of one or two words and its values, parted by single blanks and ended by a line feed. A value is a decimal number of at most 15 digits without sign or leading zero. No blank line, comment or carriage return. A fault is printed as `parameter file: <name>[: <which line>] (line <number>)`, and the call ends `NO: <name> (policy)`; blank and parentheses are part of the name.

**2. The model's lines**, from `prof` on, are read by the strict scan `polScan` in `verifier/src/cp_verify.cpp`. Every line ends with one line feed and has at most 1,023 bytes.

- A *value line*, such as `prof 20`, is a name, one blank, and a value below 2^32 in decimal without sign or leading zeros. A 64-bit value takes two lines, `<name>_lo` and `<name>_hi`.
- A *call line* describes one lookup call: `call` (layer sub-proofs), `callA` (attention) or `callH` (output head), then the call's name, the kind of its table and one or more segment names, parted by single blanks. The items consist of letters, digits, `.`, `_` and `-`: they are printed at the end of refusal names, and no control byte or blank may reach printed output. A call's name has at most 8 characters (`policy-call-line`): too few to spell an ID or a sentence in a refusal name.

The value lines come first, in the order of the tables below, then all `call`, all `callA`, all `callH` lines. An optional line with a default must be left out when its value would be the default, so that given values have one spelling. Within a kind the call lines may stand in any order (for `callA` this follows from the source and is covered by no test), but the order counts: `iota_<i>` belongs to the i-th `call` line and `iotaH_<i>` to the i-th `callH` line, counted from 0; the layout of the proof follows it; another order is another file, with another ID.

| name | cause |
|---|---|
| `version-unknown (policy)` | the word on the first line is not this program's |
| `parameter-line-missing`, `-line-order`, `-line-blanks`, `-line-values`, `-number-spelling` | the form of a frame line |
| `parameter-range`, `-group-shape`, `-stages-count`, `-stages-shape`, `-own-code`, `-one-mask-code`, `-table-shape` | a frame line's values |
| `statement-required` | `prof` is absent or not 16 to 31, or `toksrc` is absent or not 1 (both are read by name before the scan, as is `lt`) |
| `policy-line-form` | a value line is not in the one spelling, or has more than 19 digits in a row |
| `policy-line-long`, `policy-zero-byte` | a line of more than 1,023 bytes; a zero byte |
| `policy-line-<i>`, `policy-name-<name>` | from `prof` to `F`: the line is no name and value below 2^32 (i counts `prof` as 2); the line `<name>` was due and another came |
| `policy-length` | the file ends before `F` |
| `policy-cert-words` | behind `F`: a line `clt`, `clns` or `cms`, which no format has |
| `policy-call-line` | a call line is malformed or out of order, or a line behind `F` has a name that the scan does not know |

On a line from `prof` to `F` the scan tests in this order: more than 19 digits in a row; a name and a value below 2^32; the name that is due; the spelling. The range of a value is tested where the part of the proof that uses it is set up. So a value can pass the scan and be refused later, and of several values out of range the test that runs first gives the name. Many lines allow exactly one value: such a line is a constant of the format and leaves nothing to choose.

## All lines in their order

### 0. The frame

`version`, `sec`, `reading`, `rowcost`, `slots`, `answers`, `record`, five lines `work <kind> A B C`, `salt`, `listed`, eight pairs `group <name> ..` / `stages <name> ..` (`link`, `folded`, `omega`, `own`, `once-folded`, `once-omega`, `once-own`, `once-ranges`), `ranges-part`, `table`. Meaning and ranges: the head of `blocks/frame/parameters.h` and `read_parameters` there. (Where cp_commit rejects a file with `the key file: the ranges' parts: ..`, the message may speak of `lwr`, which is no line of this format: it is the number of the line `ranges-part` for the lookup on bytes, and one less for the lookup on weights. Some of these messages also speak of `lkW`, which is no line of this format either: it is the last number of the line `table`. With `policy-range-calls` cp_commit refuses a `ranges-part` that gives fewer than 2 lookup calls or more than 4096.) What follows is what a reader of this page needs, and what `cp_verify` asks beyond the parser.

| line | note |
|---|---|
| `version` | four characters: `H`, the format's number, the field (2: p = 2^64 - 2^32 + 1, the programs' one field), the hash: `H821` (`blocks/frame/version.h`). Any other word is answered with `NO: version-unknown (policy)`, and no more is printed |
| `sec` | the target: every round's error times 2^-work is at most 2^-sec. 1 to 256 by the parser; at most `tau` (`policy-tau-hb`); at least the asked bits (`security-bits`) |
| `reading` | 1: the theorem's integer is read as the theorem is printed; any other value ends `parameter-range`: the other reading would give shorter lists |
| `group folded`, `group omega`, `group once-folded`, `group once-omega` | a pair (folded and omega, or once-folded and once-omega) with the same code length and message length misses the condition `two-codes` |
| `answers` | answers to each sample outside the domain: 1; any other value ends `parameter-range`: this project's proofs cover a table only where it has one answer |
| `slots` | the most requests in a batch, a power of two; this program: at most 256 (`policy-leafd`) |
| `work ..` | bits of proof of work before a verifier's message, by the kind of its round's error, in a request's own rounds, a batch's shared rounds, the committer's rounds. Below `sec` and at most 58 by the parser; this program: at most 32 (`policy-work-most`) |
| `record`, the first number of a line `group`, the fifth number of `table` | constants of the format: `record 1`; rule 1, up to the Johnson radius; 0, since the table's distances are derived. A call can end YES with closeness rounds of the key and of the entries on the theorem on codes. This program holds the table's three large numbers in 32 bits (`parameter-table-shape (policy)`). The paragraphs under this table say on what the table rests and what a call prints about it |
| `stages <name>` | `cp_commit` and `cp_prove`: the 1st, 5th, 9th, .. numbers of the line add up to less than the third number of the line `group <name>` above it (`prover-only-stages`): these programs cannot run a proximity test whose last message has one place |
| `table`, its last number | the column variables of the committed matrix W: 1 to 20 in this program (`w-shape-lk`), and at least the bits of the vocabulary's index (`hid-vocab-cols`). `cp_commit` and `cp_prove` take 4 to 18 (`prover/common/setting.h`) |

**On what it rests.** The committed table has a record (line `record 1`; [../../prover/examples/TARGET-KITS.md](../../prover/examples/TARGET-KITS.md)). The truth of an accepted request and the binding of the weights both rest on the theorem on codes, and at the default setting both lean on counted proof of work in the rounds that draw positions ([docs/limits.md](../../docs/limits.md#what-the-theorem-assumes)). A plain `YES` shows the counted work in its `proof of work:` line; that a proof's own closeness rounds rest on the theorem on codes it shows only in the `accounting:` lines (../../verifier/README.md).

The call's own lines say that the table has a record (the stages' accounting lines, the closing `commitment:` line and the `record=1` field of `result-v1`) and count its rounds as resting on the theorem on codes (the field `commitment-assumed-rounds=` of `result-v1` counts the closeness rounds of the key file and the leaf file that do not rest on the bound within a third of the distance: [output.md](output.md)). The stages key and leaf and the planning calls `-plan request` and `-plan accounting` print on what these rounds rest, one line for each of the two plans: `accounting: one-time files: the plan of a key: <n> closeness rounds rest on the theorem on codes (ePrint 2025/2055, Thm 4.6), taken as given, <m> on the bound within a third of the distance; the table has a record`.

A whole call says the two plans' sum in its line `commitment: ..`. That line ends `; the table has a record: the binding of the weights rests on the theorem on codes (ePrint 2025/2055, Theorem 4.6), taken as given`, and the line `result-v1` has the field `record=1` directly behind `commitment-assumed-rounds=`; the last line stays `YES`.

No condition of the standard parameter form looks at `salt` or `ranges-part`. The parser takes `rowcost 1` alone: combining the rows of an interleaved word costs nothing, by our own short argument, unrefereed; it is the laxest reading. It takes `listed 0` alone. Every group has rule 1: up to the Johnson radius. Through its numbers the file decides on which source some closeness rounds rest, the bound within a third of the distance or the theorem on codes; no condition tests that, and a standard yes does not say which. The other lines named here move sizes, the room for hiding or the number of rounds, and every round is held at `sec` whatever they are.

`sec` and four numbers of `table` are also the last five words of the statement, which the program builds itself.

### 1. Profile

| line | meaning | allowed values |
|---|---|---|
| `prof` | 16 plus the *profile*, 4: the whole model. The 16 says that embedding and output head are proved | 20 (`statement-required` outside 16 to 31, else `policy-profile`, unless a test that runs before it fails, such as `policy-name-lB` where a line from `lB` to `iotaA_0` is missing) |

### 2. Shape of the model

| line | meaning | allowed values |
|---|---|---|
| `lt` | log2 of T, the tokens in the window of a request | 4 to 16 (`shape-words`) |
| `L` | number of layers | a multiple of 2^ll (`shape-L`); at most 28 blocks of layers (`plan-groups-base`: beyond it the prover's random streams would share numbers) |
| `ll` | log2 of the layers in a *block of layers*, which one layer sub-proof and one attention sub-proof cover | 1 to 6 (`shape-words`) |
| `d` | width of the residual stream | 1 to 2^20 (`shape-words`); `no-wrap-H-three-digits` narrows it, to 8,191 at `sh 8` and 7,936 at `sh 7` |
| `H`, `G`, `dh` | query heads; key and value heads; dimension of a head | H * dh = d; H + G at most 64; dh a power of two, at least 2 (`shape-heads`); G a power of two that divides H (`af-shape-heads`) |
| `f`, `fe` | hidden width of a dense layer's feed-forward network; of one expert | f 1 to 2^20; with experts fe a power of two and f = 2 fe (`shape-words`). cp_commit and cp_prove: f must have at most eight one-bits in binary and be below 2^16 (`prover-only-f`) |
| `NE` | routed experts | 0, or a power of two from 2 to 4,096 (`shape-NE`) |
| `V` | size of the vocabulary | 2 to 2^20 by these tests: below 2 and above 2^24 `shape-V`; from 2^20 + 1 to 2^24 `hid-vocab-cols` where the bits of V's index exceed the last number of `table`, otherwise `w-shape-lk` (that number is then above 20). Not every V in the range passes: among the later tests is `policy-gap-words` (2 V <= 2^F). cp_commit and cp_prove: V, rounded up to a multiple of 2^(la - lt), must have at most eight one-bits in binary (`prover-only-V`) |

With `NE` above 0 every layer with an odd number, counted from 0, is a mixture-of-experts layer, in which one routed expert is chosen for each token.

Where 2^(la - lt) does not divide `V`, the output head has padding columns behind column V - 1, with committed cells, held by one lookup key (segment HD0, at a place that is not real: `docs/format/11-4-output-head.md` describes the segment). No configuration under `prover/examples/` has such columns, so no honest run on them meets these cells.

### 3. Constants of the integer arithmetic

| line | meaning | allowed values |
|---|---|---|
| `eps1_lo`, `eps1_hi`; `eps2_lo`, `eps2_hi` | the constants added to the sum of squares in the first and second norm of every layer | 1 to 2^59 - 1 (`policy-norm`); with `xl 5` below 2^48 (`no-wrap-ledger-eps`) |
| `cs` | the rotary table's entries are cosines and sines times 2^cs | 12 (`policy-cs`) |
| `theta0_lo`, `theta0_hi` | the base of the rotary position embedding | not 0 (`shape-theta0`); no upper limit |
| `hq`, `hv`, `ho`, `hgD`, `huD`, `hdD`, `hr` | the *digit form* of seven classes of matrix products: queries and keys; values; attention output; gate; up; down; router | 0, 1 or 2 (`policy-forms`); 0 for a class that the model lacks (`header-absent-class`: `hr` without experts) |
| `ut` | the output of the up projection is range-checked to 16 bits | 1 (`policy-ut`) |

After each matrix product the integer model divides by 2^k and rounds; k is that product's shift in `KL.bin`. A lookup checks the low 16 bits of the remainder; the bits above are committed as bytes, and the digit form is the number of these bytes: 0, 1 or 2 allow shifts up to 16, 24 or 32.

### 4. Committed matrices of a layer sub-proof

A sub-proof commits to its intermediate values as *cells* (in an honest proof signed bytes), arranged in matrices; a *feature* is a run of T cells, one for each token. The *smallness test*, the last part of every sub-proof, forms random combinations of the committed columns and shows that every cell is below 2^18 in absolute value (`docs/format/11-1-machinery.md`, "The smallness test: values, equations and claims"). What each matrix holds: `docs/format/11-2-feed-forward-segment.md`.

| line | meaning | allowed values |
|---|---|---|
| `la` | features are evaluated in runs of 2^(la - lt), that is 2^la cells at a time | lt + 1 to 16, and 2^(la - lt) divides d, f, fe, G dh and (H + G) dh (`shape-la`). la = lt is refused too: no kit has it, and it has not been analysed |
| `nmin` | a lookup call with fewer keys is filled with padding keys up to 2^nmin | at most 28 (`shape-words`: it sizes vectors before any other test); at least 10 (`af-policy-nmin`) |
| `D` | the number of rows of the committed weight matrix, which is the sum of all units' rows, is rounded up to a multiple of D, once; a unit's own rows are not rounded, and a unit begins where the one before it ends | 2,048, or 32 for toy models (`shape-D`, given at two places: at once to a value that is no power of two from 2 to 4,096, later, in `verifier/src/vf_weights_layout.h`, to the others). With 32 no call can end `YES` (`ring-degree`) |
| `tau` | random combinations in the smallness test | 32, 64, 96, 128, 160, 192 or 256 (`hid-tau`: a multiple of 32 whose rows fall into one or two blocks of a power of two). In effect `sec` itself: at least `sec` (`policy-tau-hb`), and equal to the number of combinations that `sec` less a request's work on smallness gives (`plan-tau`), so that work is 0 |
| `lmaf`, `NAf` | matrix `af`, most intermediate values of the layers: log2 of its rows; parts into which its columns are cut | la to 24; 1 to 64 (`shape-request-matrix`, `shape-la`) |
| `lmad`, `NAd` | matrix `ad`, outputs of the two norms and input of the routed expert | lt to 24; 1 to 64 |
| `lmar` | matrix `ar`, the router's values and the norms' rows of digits | lt to 24 |
| `lmb` | matrix `b`, the multiplicities of the lookup calls, bit by bit | at most 24 (`shape-lmb`); 5 to kT + 5 (`forest-planes`, under table 6) |
| `lmq` | matrix `qa`, queries, keys, values and attention outputs, which the attention sub-proof reads again | la to 24 |
| `lmx` | the residual stream at the two ends of a block of layers | la to 24 |

### 5. Width of the weights

| line | meaning | allowed values |
|---|---|---|
| `wb_qk`, `wb_v`, `wb_o`, `wb_g`, `wb_u`, `wb_gs`, `wb_us`, `wb_gr`, `wb_ur`, `wb_dn`, `wb_dnr`, `wb_r` | declared bit width of a weight, by kind: queries and keys; values; attention output; gate and up of a dense layer, of the shared expert, of the routed experts; down of dense layers and shared expert, of the routed experts; router | 13 (`policy-weight-kind`); 0 for the six kinds that exist only with experts, `gs`, `us`, `gr`, `ur`, `dnr`, `r`, in a model without them (`header-absent-kind`) |

Each line must equal the constant, and no other test reads it: a width fitted to the weights would be public and would show the smallest and largest weight of a kind. The no-overflow conditions assume that a weight outside the embedding is at most 4,096 in absolute value. A request's proof does not show that; the key's proof of the ranges does (`blocks/frame/wf_units.h`).

### 6. Lookup calls of a layer sub-proof, and the seed

| line | meaning | allowed values |
|---|---|---|
| `iota_0` to `iota_7` | the i-th `call` line's lookup call has 2^iota_i trees | 0 if there is no such line (`header-absent-call`); else n - dT (`shape-call-<name>`) |
| `seed_lo`, `seed_hi` | a public number, which has nothing to do with a commitment's secret seed (`seed.bin`, which no verifier ever needs): no check reads it; `cp_commit` and `cp_prove` take it as the seed of their generators of synthetic weights, so anybody who has the parameter file can make those weights again; "Lines without a range check" | none |

All lookup calls of a sub-proof run as ONE lookup argument by product trees (`verifier/src/vf_lookup_argument.h`, whose head states it; `blocks/iop/lkp.h`). The keys are laid out as a *forest* of trees of 2^dT keys, dT the smaller of `lidb` (table 9) and the log2 of the keys of the smallest call; a call with 2^n keys has 2^(n - dT) trees. The lines `iota_<i>`, `iotaH_<i>`, `iotaA_0` carry no freedom: each must state the number that the program derives. The two lookup calls of a sub-proof's smallness test are a second forest by the same rule, without lines. Each call has one vector of multiplicities of 32 bits a table row, so with NC calls, 2^kT rows in the largest table and 2^lm rows the matrix of multiplicities has NC * 2^(kT + 5 - lm) columns. Every committed matrix of a request has at most 8,191 columns (`hid-k-range`).

| name | what must hold, before a word of the proof is read |
|---|---|
| `forest-calls`, `forest-trees`, `forest-depth` | 1 to 16 calls; at most 256 trees; dT at least 10 |
| `forest-keys`, `no-wrap-pin-forest` | fewer than 2^32 keys, trees of padding counted, so that a multiplicity of 32 bits has room for all; the sum of all multiplicities cannot wrap in the field |
| `forest-depth-G`, `forest-depth-H` | layer sub-proof and output head: dT at least `la` |
| `forest-depth-A` | attention: dT at least `lmS` and at least lt + log2 of `dh` |
| `forest-planes` | `lmb`, `lmbA`, `lmbH`: 5 to kT + 5 |
| `forest-only-padding-<name>` | no call has nothing but padding keys: a yes would say nothing of it. Asked where each host's plan has made its numbers of padding keys (`forestPubAsk` in `vf_lookup_argument.h`: the layers', attention's, the head's and the smallness test's plans), so by every call that makes the plan, a planning call among them; and once more at the head of the routine that reads a sub-proof's lookup (`forestPad`) |

### 7. Attention

The *score matrix* of a head is handled in square tiles. The *softmax table* is the public table from which the integer model reads the weight of a score. The exact conditions, and what each matrix holds: `docs/format/11-3-attention-segment.md`.

| line | meaning | allowed values |
|---|---|---|
| `lB` | log2 of the side of a tile | 0 to lt - 2 (`af-shape-tiles`) |
| `S` | the largest softmax weight is 2^S | 1 to 14 with lt + S + 1 at most 24 (`af-policy-softmax`; 32 or more: `af-policy-wd`, which is tested first); the standard form allows 12 alone |
| `rx` | the table holds round(2^S exp(-u / 2^rx)) for u = 0, 1, .. until the value is 0 | 1 to 12 (`af-policy-softmax`) |
| `kFM` | cp_verify does not read it | none in cp_verify. cp_commit and cp_prove: at most 30 (`prover-only-kFM`); cp_prove's plan of attention has a further test on it (`build()` in `prover/attn/plan.cuh`) |
| `kFD` | log2 of the rows of the lookup table that contains the softmax table | 8 to 16, with a further bound from the table's length (`af-policy-kF`) |
| `lmS`, `NAs` | matrix `as`, the softmax weight of each score and two auxiliary numbers: log2 of the rows of a part; parts | 1 to 24 with a further bound from the shape (`af-shape-lmS`); 1 to 64 |
| `lmh` | matrix `ah`: for each head and token the largest score, its position, the sum of the weights | lt to 24 |
| `lmo` | matrix `ao`: high digits of the remainders of the division in attention | la to 24 (`af-shape-la`) |
| `lmbA` | matrix `bA`: the multiplicities | at most 24; 5 to kT + 5 |
| `iotaA_0` | 32 + dT of the attention sub-proofs' forest. Unlike `iota_<i>` it gives the size of a tree, for all `callA` lines together; it is tested in every attention sub-proof, so all have one dT | 32 + dT (`policy-iotaA`) |

The ranges of `NAs`, `lmh` and `lmo` and the upper limits 24 share the name `shape-request-matrix-A`.

### 8. Output head and acceptance policy

| line | meaning | allowed values |
|---|---|---|
| `MF`, `KF` | the final norm multiplies by a reciprocal scaled by 2^MF and shifts right by KF bits, as M and K of `KL.bin` do in the layers | MF 24 to 47 and at most KF + 14; KF 17 to 32 (`policy-norm-F`) |
| `epsF_lo`, `epsF_hi` | the constant added to the sum of squares in the final norm | as eps1 (`policy-norm-F`) |
| `toksrc` | the input of the model is the embedding of the tokens of your request file | 1 (`statement-required`) |
| `lmhf`, `NHf` | matrix `hf`: quotients of the final norm and, for each position and token of the vocabulary, the digits of a difference of logits | la to 24; 1 to 64 (`shape-request-matrix-H`, also for the next three) |
| `lmhd`, `lmhr` | matrices `hd` (output of the final norm) and `hr` (its row of digits, and the digits of the numbers with which the proof bounds the costs) | lt to 24 |
| `lmbH` | matrix `bH`: the multiplicities | at most 24; 5 to kT + 5 |
| `wb_emb` | declared bit width of an entry of the embedding | 21 (`policy-weight-width-emb`; below 2 or above 31 `policy-weight-kind`, which is tested first) |
| `wb_head` | the same for the output matrix | 13 (`policy-weight-kind`) |
| `iotaH_0` to `iotaH_5` | as `iota_<i>`, for the i-th `callH` line | 0 if there is none (`header-absent-call-H`); else n - dT (`shape-call-H-<name>`) |

No test but the equality, and the range test before it, uses `wb_emb`. The one condition on the embedding (`no-wrap-H-J-E`) assumes that each of the four base-256 limbs of an entry is below 2^21 in absolute value; the key's proof of the ranges shows more, that each is a signed byte, and a request's proof shows nothing of it.

The last seven lines are the *acceptance policy*: what an accepted proof says about the checked tokens (`docs/verifying.md`, "Acceptance policies").

- A request is a window of T tokens with two numbers r < e <= T; the proof *covers* positions r to e - 1. At a covered position t below T - 1 the *checked token* is token t + 1 of the window. Position T - 1 is about y, the next token, which follows the window: it is not counted among the checked tokens and has no cost: if e = T it must be the greedy choice of position T - 1, whatever the policy.
- The *greedy choice* is the token with the largest logit, of several the one with the lowest id. The *logit gap* is the largest logit minus the checked token's, plus 1 if a token with the largest logit has a lower id. It is 0 exactly for the greedy choice.
- nll is the negative log-probability of the checked token under the softmax of the logits divided by 2^sT, times 2^sT.
- The *cost* of a covered position below T - 1 is the larger of (logit gap + 2^sh) and nll. At T - 1, y must be the greedy choice.

With R covered positions below T - 1 the proof shows that the R costs, each clipped at C, sum to at most theta * R, and that at most cap of them exceed C.

The statement says nothing about how the tokens were generated (sampling, temperature, top-k, top-p). With a smaller C the sum says less about the positions above it, and with a larger C the count says less; and a bound in unknown units says nothing. So a caller holds `C` and `s` equal to values of their own (`-policy`), not merely below a bound.

| line | meaning | allowed values (all `policy-gap-words`) |
|---|---|---|
| `C`, `theta`, `cap` | the clip; the bound on the mean; the most positions above the clip | theta at most C; cap at most T and below 2^16 (the GPU programs: C and theta at least 2^sh, the least cost of a token; cap 0 where theta = C: `prover-only-policy-says-nothing`) |
| `s` | a logit of the model is an integer logit divided by 2^s. It is the provider's statement: no equation ties it to the model. It is compared in two places: sT = s makes the kind of policy that the program calls `log-probability`, and `-policy` names it | at most 63 |
| `sh` | 2^sh is added to every logit gap | 7 or 8 |
| `sT` | the scale of nll; it selects the public table of a decaying exponential that the head looks up | sh - 4, or sh + 4 to sh + 10 (the GPU programs: sh - 4 alone, `prover-only-policy-kind`) |
| `F` | the table's values are scaled by 2^F | 24 |

With sT = sh - 4 the table is the *step table* (2^F, 2, 1, 1, ..): the lookups enforce only that the cost is at least the logit gap plus 2^sh, and the cost is exactly that. With sT from sh + 4 to sh + 10 it is an exponential table at the scale 2^sT; with sT = s that is the scale of the model's logits, if the provider's s is right. A whole call takes the step table alone: under any other `sT` (the program calls such a policy `log-probability` or `another-table`) it ends `NO: policy` before any stage. The part-calls and the planning calls do not ask.

### 9. Optional lines

Behind `F`, in this order; each may be absent.

| line | meaning | allowed values, default |
|---|---|---|
| `lidb` | log2 of the most keys in one tree of a request's lookups | 23 to 30; absent means 22 (`policy-lidb`, `plan-identity-cap`). cp_prove: at most its build's EIDB (`prover/build.sh`; 28 in the default build), under the same name `plan-identity-cap`; cp_commit makes no such test, so it can commit under a file that cp_prove then rejects |
| `nminH` | takes the place of `nmin` in the output head | only directly behind `lidb`; above `nmin`, 2 to 28; absent means `nmin` (`policy-nminH`) |
| `eg`, `smax` | the gate table's input resolution and bound | 4 to 7, absent 8; 2047, 1023, 511 or 255, absent 4095 (`policy-gate-table`) |
| `rcl`, `gcl` | clamp the router's logits; the key of the gate table | 1 (`policy-rcl`, `policy-gcl`); only with experts |
| `xl` | five limbs for a value of the residual stream | 5; absent means 4 (`policy-xl`) |
| `tsilu0`..`7`, `tsig0`..`7`, `trope0`..`7`, `texp0`..`7` | the digests of the public tables, one word a line | with experts all 32 or none; a dense model (`NE 0`) uses three tables and gives `tsilu`, `trope` and `texp` (24 lines) or none, and never the `tsig` lines; the lines come one directly behind the other, in this order (`policy-tables`) |

### 10. Call lines

| line | limits |
|---|---|
| `call <name> <kind> <segment> ..` | 1 to 8 lines (`policy-call-list`). They cannot be left out: the built-in lists lack the segment UT and fail the test below |
| `callA ..` | 1 to 12 lines, and at most 12 calls once a line of kind K.. has been taken once for each attention shift of a block of layers (`policy-call-list-A`, `af-policy-call-list`) |
| `callH ..` | at most 6 lines (`policy-call-list-H`). All may be absent: the program then uses a built-in list of six calls. A file without them and a file that spells that list out describe the same parameters and are two files with two IDs |

| kind | table | on |
|---|---|---|
| `R16`, `R8` | range tables of 2^16 and 2^8 rows | `R16` on all three; `R8` on `call`, `callH` |
| `FC` | clamp to -2047 .. 2047 | `call`, `callH` |
| `FL`, `FS` | gate table; router sigmoid and gate clamp | `call` |
| `FE` | the output head's table of exponentials | `callH` |
| `KR` | 2^16 rows: the two bytes of t and a flag that says whether t is at least 2^sh, sh the attention shift | `callA` |
| `KFD` | 2^kFD rows, which contain the softmax table | `callA` |
| `WH` | 2^16 rows: the two bytes of t | `callA` |

Every line names at least one segment. Each segment that the file's other lines require must be listed exactly once, in a call of the kind prescribed for it, so that its keys are tied to the committed cells in exactly one call; no two calls of a sub-proof share a name. A `call` line may not be named `V`, `a`, `b` or `E`, the names of four evaluation points of the layer sub-proof. Segments, tables and keys: `docs/format/11-2-feed-forward-segment.md`, `docs/format/11-3-attention-segment.md` and `docs/format/11-4-output-head.md`.

## The optional lines on the model's arithmetic

**For a user these are not options.** A call can end with `YES` only with `xl 5`, without `eg` and `smax` (conditions `five-limbs`, `no-range-words`), and meeting `both-clamps`: with experts that means `rcl 1` and `gcl 1`; a dense model (`NE 0`) meets it with neither clamp line (it has no router and no gate clamp), and a clamp line on a dense model is refused (`policy-rcl`, `policy-gcl`). The other settings exist so that small configurations can serve as tests. Before its long work, `cp_commit` makes a planning call of the `cp_verify` that lies beside it, and at 128 bits it rejects a file of which that call says that it misses a condition, `five-limbs` among them (`cp-commit-not-standard`); its [production build](../../docs/glossary.md#production-build) has no option for fewer bits. Only the planning calls `-plan request` and `-plan accounting` print the settings (lines `plan: ..`).

The *proved function* is the function from the tokens to the logits that the integer model computes. `eg`, `smax` and `rcl` change it. `gcl` and `xl` leave it unchanged and widen the range of intermediate values for which a proof exists: without them a prompt on which a value leaves the narrower range has no proof.

**The gate's table.** The activation is SiLU, z * sigmoid(z), read from a public table of 2^16 rows. A key c, a signed 16-bit integer, stands for z = c / 2^eg. The key is limited to -64 * 2^eg .. 64 * 2^eg - 1; the entry is 64 * SiLU(z), rounded, then limited to -smax .. smax (`docs/model.md`, section 4, item 3; on `eg` and `smax` section 10).

**The router's clamp** (`rcl 1`) keeps a very large router logit from leaving the prompt without a proof. A logit lw is a number of steps of 2^-8; write lw = 8 h + w with w its low three bits. The routing uses 8 clamp(h) + w, clamp limiting h to -2047 .. 2047. From -16376 to 16383 (logits from -63.97 to just under 64) that is lw itself. Beyond, only w tells logits apart: a logit of 64 or more can rank below one between 63.97 and 64, and likewise at the lower end. The call lines must then list the segments `L3` (kind `R8`) and `CLR` (kind `FC`).

**The gate's clamp** (`gcl 1`) lets the gate table's key exceed 16 bits. The key gw is written g0 + 256 gh, g0 its low byte from -128 to 127, and the table is read at 256 clamp(gh) + g0, clamp limiting gh to -127 .. 127. The table is constant below key -64 * 2^eg and from 64 * 2^eg - 1 on, and whenever clamp(gh) differs from gh both gw and the clamped key lie in the same constant part. So the value read is the table's at gw wherever gw fits 16 bits and its end value beyond, and wherever a proof exists with and without the line the proved function is the same. Keys up to about 2^22 in absolute value are allowed (gh has 15 bits). The call lines must list `G0` (kind `R8`) and `CG` (kind `FS`). The pairs (gh, clamp(gh)) stand in the upper half of the router sigmoid's table, whose digest therefore differs with the line.

**The limbs of the residual stream.** A value of the stream is stored as four *limbs*, signed bytes that are its digits in base 256; `xl 5` gives it five. For each norm and token the proof holds a row of byte digits: of Vx, the sum of the squares of the token's stream values, of its square root, of the reciprocal of the root, and of the carries and remainders that show these to be right. With four limbs the row has 74 digits and shows Vx below 2^59; with five it has 99, Vx may reach 2^78 and its root 2^39. Where four limbs suffice the numbers are the same. `verifier/src/vf_norm_digit_rows.h` says why nothing in the row (the sources call it the norm's "ledger") overflows; the row itself, digit by digit: `docs/format/11-2-feed-forward-segment.md`, "A norm's row of digits".

A *no-overflow condition* is a test on the public parameters which is meant to ensure that an equation that the proof checks modulo p also holds over the integers. That it does is not proved, and for the programs' output head no proof is known that its claims imply the statement over the integers (row SO-9.3 of [docs/limits-all-rows.tsv](../../docs/limits-all-rows.tsv)). A failed condition is refused with a name that begins `no-wrap-`, or `af-no-wrap-` for attention's. With `xl 5` the conditions `no-wrap-ledger`, `-W-norm`, `-U-o`, `-W-o`, `-U-down`, `-W-down`, `-H-W-NF`, `-H-J-E` use the wider bounds. Two consequences: the shifts ko and kd of `KL.bin` (behind the attention output and the down projection) are at most 19, and eps1, eps2, epsF are below 2^48.

## Digests of the public tables

The integer model reads up to four public tables: the gate table, the router's sigmoid (2^16 rows each), the cosines and sines of the rotary positions, the exponentials of the softmax. A model with experts uses all four; a dense model (`NE 0`) uses three, all but the router's sigmoid. The verifier builds the tables it uses itself, with the floating-point library of its platform and a final rounding, and puts their digests into the statement (a dense model's sigmoid digest slot is zero). Two platforms whose libraries round differently could build tables that differ in the last bit, and so disagree on the proved function.

The digest lines state the digests, H(72; table) in the words of `verifier/HASHES.md`, word i of a digest on line i: 32 lines with experts (`tsilu`, `tsig`, `trope`, `texp`), 24 for a dense model (`tsilu`, `trope`, `texp`; no `tsig`). With them a verifier whose own table hashes differently refuses before it uses a word of a proof, also in a `-plan` call (`table-digest-silu`, `-sigmoid`, `-rope`, `-exp`; a dense model does not check `-sigmoid`). Without them it has nothing to compare its tables with.

| table | depends on |
|---|---|
| gate table | `eg`, `smax` |
| sigmoid | `gcl` |
| rotary positions | `lt`, `dh`, `cs`, `theta0_lo`, `theta0_hi` |
| softmax exponentials | `S`, `rx` |

`scales/policy_tables.py` makes the digest lines from these values with code of its own, which imports nothing of the verifier or the provers; its text of BLAKE2b's compression function has the same outside source as theirs (the BLAKE2 team's reference implementation). Run it again whenever one of them changes.

## Scales, and lines without a range check

Most quantization scales are in `KL.bin`, layer by layer, and their ranges are tested there: M and K of the two norms, the right shift behind each matrix product, the attention shift, the flag for rotary positions. Eight lines of the parameter file hold the rest: `MF`, `KF` and eps1, eps2, epsF. Scales that the program accepts can still be badly chosen (`scales/README.md`).

*What norm-shifts does not cover.* The condition shuts out every setting at which a norm's output is zero for every input (from K = M + 3 on), and also K = M + 1 and K = M + 2. Settings that meet it can still round every ordinary input to zero, so that the layers do nothing. **A YES is about the model as the parameter files under the ID define it.**

Three lines have no range check in cp_verify, and like every line are covered by the ID. `kFM`: cp_verify does not use it (cp_commit and cp_prove reject a value above 30, and cp_prove's plan of attention has a further test on it). cp_prove also asks more than cp_verify of the lines `lmS`, `lmh`, `lmo`, `lmbA` and `lmq`: each column length (2 to the power of the line) must be a multiple of 2,048, so each of these lines must be at least 11; its refusal speaks of "the ring degree D", gives no number and does not say which line to change. `seed_lo`, `seed_hi`: no check reads them (`cp_commit` and `cp_prove` take them as the seed of their generators of synthetic weights).

## The standard parameter form

A call can end with `YES` or `WITHIN LIMITS: ..` only under the *standard parameter form*, a list of conditions on the public inputs that the program tests by name (`Served`, `servedForm` in `verifier/src/vf_common.h`). Under any other form a whole call ends at best with status 14 and `NOT-A-VERIFICATION`, whatever bits you ask; the line `result:` says `not the standard parameter form:` and lists what is missed, and a `-plan request` call prints `plan: not the standard parameter form (<conditions>): no run with these public inputs can be a verification (at best status 14)`. Other forms are admitted as input so that small configurations can serve as tests.

The stage request tests these eighteen. Read the names as labels only.

| condition | what it requires | decided by |
|---|---|---|
| `routed-experts` | the model has routed experts, or is dense with neither clamp line | `NE` above 0; or `NE 0` with no `rcl` and no `gcl` |
| `five-limbs`, `both-clamps`, `no-range-words` | see above | `xl`; with experts `rcl` and `gcl`, or a dense model (`NE 0`) with neither; neither `eg` nor `smax` |
| `prompt-source` | the input is the embedding of your tokens | `toksrc 1` |
| `placement-table` | the placement of the arrays comes from a layout file | the config folder holds the name `layout.txt` (`verifier/src/vf_input_files.h`); a name that is there and cannot be read ends the call with the line `input-unreadable-layout: layout.txt cannot be read: ..` and status 2: that is no verdict |
| `table-digests` | the file states the digests of the tables the model uses | with experts the 32 lines of all four tables; a dense model (`NE 0`) the 24 lines of the three it uses (`tsilu`, `trope`, `texp`) |
| `three-bytes` | the remainder of the division in attention has three bytes | `S` below 32 |
| `ring-degree` | rows of the committed weight matrix are counted in blocks of 2,048 (there is no ring; the name is a label) | `D 2048` |
| `one-shift` | all layers of a block of layers have the same attention shift, which turns a score's distance from the largest score of its row into an index of the softmax table | sh of each layer in `KL.bin` |
| `one-placement` | `qa` and the residual stream are laid in the same way in all blocks of layers: the same oracle message, sizes and blocks, block by block in every number | the layout file |
| `softmax-table` | scale 2^12, step 2^-8 | `S 12`, `rx 8` |
| `rotary-table` | base 10000 or 500000, a head of at most 128 dimensions, a window of at most 2^10 tokens | `theta0_lo`, `theta0_hi` 0, `dh`, `lt` |
| `norm-shifts` | K <= M for both norms of every layer and KF <= MF. A norm's output is at most about 2^(M + 1 - K) in size, and from K = M + 3 on it is 0 for every input, so that the layer adds nothing whatever its weights are | `KL.bin`; `MF`, `KF` |
| `two-codes` | the folded row and Omega lie under two codes, in a batch's test on the committed table and in the key's one-time test. Under one code the programs run that test without the step that splits a row between two groups. No proof of this project covers that form. No attack on it is known, and none has been searched for | `group folded` and `group omega` differ in code length or message length, and so do `group once-folded` and `group once-omega` |
| `next-token` | asked only under `-next-token`: the proof is about y and about no token of the request | r = T - 1, e = T |
| `exact-match` | if the request covers a token of its own, every covered token is the first choice of its position. Counts as met if r = T - 1, or under `-policy`. A whole call without `-policy` that covers a token looks at these lines before any stage and ends `NO: policy` if they set another policy, so the stage's condition is a second check | C = theta = 2^sh, cap = 0, sT = sh - 4 |
| `policy-named` | asked only under `-policy <kind>,<C>,<theta>,<cap>,<s>,<sh>`: the file sets that kind, `exact-match` or `clipped-gap` (sT = sh - 4, not the exact match), and these five numbers, each compared by equality. Together they fix every line of the policy. The option knows no other kind: another word, as `log-probability`, is a bad command line (status 2). A whole call compares before any stage and ends `NO: policy` if the two differ, so the stage's condition is a second check | `C`, `theta`, `cap`, `s`, `sh`, `sT` |

Several names are printed with words in parentheses behind them, which say the same.

The stages key and leaf ask `two-codes`. Two more names can be printed as missed. `id-named` is the whole call's own: it must name the ID. `no-proof-of-work` is the call's own too: it had `-no-proof-of-work`, and some round meets the file's target only with its work counted.

- No run can fail `prompt-source` or `three-bytes`: `toksrc 0` is refused with `statement-required` and a wider remainder with `af-policy-wd` before the conditions are tested. They stay in the list all the same. Nor can a checking call fail `placement-table`: without `layout.txt` every checking call ends `NO: plan-layout`.
- A planning call has neither `-next-token` nor `-policy` and never names the last three.
- `softmax-table` and `rotary-table` guard against differences between floating-point libraries: the program builds the two tables with `exp`, `pow`, `cos`, `sin` of the C library. The conditions are meant to allow only values at which every entry is far from a rounding tie, so that every faithfully rounding library gives the same table. **This is computed, not proved:** by a test in this repository (`scales/tests/table_ties.py`; the row `scales.table_ties` in SUITE.md), for the values that the two conditions allow, no entry changes if all results of the platform's library are moved by 1, 16 or 1,024 units in the last place, given binary64 arithmetic in the stated order. That no smaller error changes an entry is argued in that test and not run; it does not cover 80-bit intermediate results, under which some entries differ.
- `one-shift` limits acceptance to shapes in which each `callA` line gives one lookup call in a block of layers.

**The form does not fix the shape of the model.** It asks only `NE` above 0, `lt` at most 10 and `dh` at most 128. That the program supports a shape says nothing about what has been run or evaluated at it, and `cp_verify` compares your configuration files with nothing but the ID. A `YES` under other shape parameters says that the program's tests passed, and nothing about the numerical conditions of any theorem at that shape.

## Digests of the files

- `configuration=` is what binds: one hash of the three files, each with its length before its bytes: the parameter file first, then the layout file, then the scale file (`blocks/frame/heads.h`, `configuration_digest`). It is an item of the parameters in the first query of every hash chain of key, entry and proof, and it enters the ID. So the files are bound byte for byte, not just the values derived from them: files checked under another parameter file, scales or layout than they were made under give other challenges, and the run ends in a refusal. This also holds for a file whose text differs and whose meaning does not. No word of it is sent. Other files on the same weights (another acceptance policy, say) need a new commitment.
- `h_par=` is the same digest, printed word by word, so its digits differ. **It is printed, and no verifier compares it with anything; cp_commit writes it into isk.txt's line `ranges` when its worst-case check has passed, and cp_prove compares that line with its own digest of the three files that it has read, and refuses to start if they differ or the line is absent (`ranges-not-guarded`; under `seed test`, in test builds, it does not ask for the line).**

Both are printed by `cp_verify -plan request` and `-plan accounting`; `h_par=` also by `-plan id` (not where that call refuses the files), by a part-call and among the last lines of a whole call. If honest files are refused at the first test that depends on a challenge, compare the verifier's and the prover's digests of their configuration files first. If these agree, see that `cp_prove` and `cp_verify` were built from one copy of the repository: the version word does not tell every way of making a proof apart (docs/format/10-proof-file.md, R14.3).

