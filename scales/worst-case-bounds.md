<a id="worst-case-bounds"></a>
# The bounds of the worst-case check

This page gives reasons why the bounds that `cp_commit`'s worst-case check computes hold for every
prompt, and so why, together with the second fact below, a passed check should be enough for no
request to fail because a value leaves its range. These reasons are arguments only: no proof of the
bounds has been verified, none in writing comes with this repository, and that the code computes the
bounds as stated here rests on a reading of the code of other programs
([../docs/glossary.md#other-programs](../docs/glossary.md#other-programs)). Two more facts belong
beside that:

- The bounds on matrix norms are computed in 64-bit floating point on the GPU by a vendor's library:
  they hold if that library computes the classical matrix product in IEEE binary64, which the check
  asks for by a mode of the library and spot-checks in every run, but does not prove. The rest of
  its arithmetic is exact in integers, rounded upwards, or rounded to the nearest with a margin
  added ("What is exact and what is rounded up"); all of that assumes IEEE rounding. The build sets
  no flag against fused multiply-adds.
- Each range test that the forward pass makes needs its bounding line in the check's table. That
  each has one rests on a reading of each kernel of other programs, and not all of it is
  established.

Before `cp_commit` commits, it computes for every site an upper bound on the value there, from the
committed integers and the public scales alone. It samples no prompt: each bound is meant to hold
for every prompt. It commits only if every bound is within its site's limit, except where it leaves
the check out: a test's call under `-test-seed`, or a production call under `-public-weights`, which
hides nothing (the check is only the hiding safeguard below, so a published-weights commit may skip
it); either leaves it out unless `-worst-case` or `-worst-case-table` asks for it. This page states each bound as the code
computes it, with the argument for it. README.md, "Terms", defines site, limit, bits short, shift
and lookup key; [README.md](README.md) is the guide to the tools around the check.

The check is about whether a range test can stop an honest prover on some prompt. That bears on
weight hiding: a request that fails because a value leaves its range tells its sender something
about the weights, at a prompt of its choice, and what
[../docs/what-is-not-hidden.md](../docs/what-is-not-hidden.md) says of such failures rests on this
check. Soundness does not rest on it: the verifier never sees it, and every proof shows again that
its own values lie in the ranges in which the model is defined
([../docs/model.md](../docs/model.md), section 9), which are nowhere narrower than the check's
limits (a little wider, except at the gates, where the two are the same).

**The code** is the struct `Check` of `prover/weights_commitment/worst_case.cuh`; functions named
without a file are there. `cp_commit` (`prover/hash_commitment/hf_commit.cu`) calls `read`, which
takes the parameters, before it loads the weights. It calls `take` while it goes through the weights
once, testing every integer's range as well: one call for each kind of matrix in each layer (Wq with
Wk, all experts' matrices of one kind together); `take` records what the bounds need. After that
pass `lines` forms the bounds and compares them with the limits, and `table` writes the result, a
text file that this page calls the table. All this happens before the seed is drawn.

**The symbols** are those of [../docs/model.md](../docs/model.md): d, L, n_h, n_g, dh, isqrt, the
matrices' names, and rq(a, k), which is a / 2^k rounded to the nearest integer. Four more: abs(x) is
the absolute value; len(v), the *length* of a vector, is the square root of the sum of its squares;
sv(W) is the largest singular value of a matrix, the largest factor by which it stretches a vector's
length (the table's comment lines write ||W||); the *row* of an output is the vector of the integer
weights that feed it (a column W[.][j] in that document, a contiguous row in the model's files).
Other letters are local to a section.

**The argument in five steps.** A norm's output lies in a public ball and box, whatever the prompt
("The ball and the box after a norm"). A sum of products over that ball and box is bounded, row by
row ("The bound on one row's sum", "Query and key rows"). The residual stream grows by a sum of
terms that do not depend on it ("The length of the residual stream"). The largest singular values
that these terms need are bounded from above in floating point ("The largest singular value of a
matrix"). Public sizes alone bound the output head's margins ("The head margin").

<a id="bounds-sites"></a>
## The sites of the check and their limits

`lines` writes these sites, in the order of the forward pass. A line of the table holds the largest
bound over the rows, heads and experts of its site and layer. "Condition" is the range condition's
number in [../docs/model.md](../docs/model.md), section 9; the limits follow from its last column,
the intervals that the provers accept. That the programs accept just these intervals rests on a
reading of the sources of other programs, and that reading does not answer every question. Among
other things it is not known whether each test is applied to the model's own value. These are not
the fit's eight sites (README.md, "The sites"): `q`, `k` and `v` are the same points of the forward
pass, and the fit's `u` and `uR` are the up sites here.

| site | the value that is bounded | condition | limit on the value | shift |
|---|---|---|---|---|
| `stream at norm 1`, `stream at norm 2`, `stream at the last norm` | the length of the stream row that the norm reads | 1, 8 | 547,599,908,735 | |
| `norm 1 quotient`, `norm 2 quotient`, `last norm quotient` | w[k], a norm's output before the clamp | 2 | 32,639 | |
| `q`, `k` | qw, kw | 3 | 32,639 | `kq`, `kk` |
| `v` | vw | 4 | 32,639 | `kv` |
| `router` | lw | 5 | 261,119 = 8 * 32,639 + 7 | `kr` |
| `gate` (dense layer); `gate, shared`; `gate, routed` | gw | 6 | 4,194,175 = 256 * 16,383 + 127 | `kg`; `kg`; `kgR` |
| `up` (dense layer); `up, shared`; `up, routed` | u | 7 | 32,639 | `ku`; `ku`; `kuR` |
| `last stream, a cell` | one entry of the stream row after the last layer; its bound is the row's length | 8 | 547,599,908,735 | |
| `head margin` | a margin of the output head | | 65,537 * 2^22 - 1 | |

- **The conditions.** Each of section 9's eight conditions has a site, at the provers' interval. For
  condition 2 the 32,639 is `model.py`'s; the GPU prover tests no interval there. The head margin is
  no condition of the model.
- **Other parameter files.** These limits are for a parameter file with the lines `xl 5`, `rcl 1`
  and `gcl 1`, which the standard parameter form requires. Without `rcl 1` a router's limit is
  16,383; without `gcl 1` a gate's is 32,639. Without `xl 5` a stream value has four limbs (signed
  bytes, in base 256), not five: the stream's length may reach isqrt(2^59 - 1) = 759,250,124, which
  keeps the sum of squares below 2^59 and every entry within four limbs, and the last stream's entry
  2^30 - 1.
- **The upper end.** Every range reaches at least as far below zero as above it, so a bound on the
  absolute value is compared with the upper end. The margin is the exception: its lower end is 0
  ("The head margin").
- **The stream.** No entry of a row exceeds the row's length. So a row not longer than
  547,599,908,735, the largest number that five limbs hold, meets condition 8. It also meets
  condition 1, because `read` admits only eps < 2^48 under `xl 5`.
- **A site with a shift.** The value is rq(A, k) for a sum of products A and the shift k, and the
  bound and the limit in the table are those of A. With top the limit on the value in the list
  above, rq(A, k) <= top holds exactly if `A <= (2 top + 1) * 2^(k-1) - 1`, which is the line's
  limit (`sum` inside `lines`). At a site with a clamp, top is that of the wide value, before the
  clamp. If abs(A) <= b, then abs(rq(A, k)) <= rq(b, k), because rq does not fall as A grows and
  rq(-b, k) >= -rq(b, k). The stream's bounds use this (`quotient`).
- **Bits short** is computed by `shortfall_of`. A bound that is not a number never fits. 99 means
  that the check has no bound.
- `lines` also has a site `down quotient`, under a line `dcl` that `read` refuses, so no table of
  `cp_commit` has it.

<a id="bounds-ball"></a>
## The ball and the box after a norm

Every matrix except the embedding, Wo and the down matrices reads a norm's output nr = N(x; M, K,
eps) ([../docs/model.md](../docs/model.md), section 5; Vx, sigma, Rc and w[k] are that section's).
For every stream row x that meets condition 1, nr lies in a public box and a public ball.

| what | bound | function |
|---|---|---|
| the box | `abs(nr[k]) <= 2,047` | the norm's clamp |
| the ball | `len(nr) <= B(M, K, eps) = 2^(M-K) * (1 + 1 / isqrt(eps)) + (r + 2) / 2^(K+1) + sqrt(d) / 2` | `ball` |
| r | `2^39 + isqrt(eps)` under `xl 5`; `isqrt(2^59 + eps)` with four limbs | `ball` |
| B1, B2 | `B(M1, K1, eps1)`, `B(M2, K2, eps2)`: the radii of a layer's two norms | `take` |
| the three sites `... quotient` | `abs(w[k]) <= 2^(M-K) * (1 + 1 / isqrt(eps)) + (r + 2) / 2^(K+1) + 1/2`; at the last norm with MF, KF, epsF | `ball` with `one_cell` |

*Argument.* Rc is 2^M / sigma rounded to an integer, so Rc <= 2^M / sigma + 1/2. By the definition
of isqrt, len(x) = sqrt(Vx) <= sqrt(Vx + eps) < sigma + 1. So len(x) Rc / 2^K < 2^(M-K) (1 + 1 /
sigma) + (sigma + 1) / 2^(K+1). The first term is largest at the smallest sigma, isqrt(eps). The
second is largest at the largest sigma, which is at most r while condition 1 holds. Each w[k]
differs from x[k] Rc / 2^K by at most 1/2, which adds at most sqrt(d) / 2 to the length, and the
clamp makes no entry larger. For one entry the same steps hold with abs(x[k]) <= len(x) and a single
1/2. B has r + 2 where this argument needs r + 1; the larger radius is safe.

A routed expert reads xr[k] = rq(sc m[k], 16) in place of the norm's output m. sc is at most 2^16,
so no entry of xr is larger in absolute value than m's, its own rounding included: the same ball and
box.

No weight enters a quotient's bound. So a parameter file that the verifier admits can fail there
whatever the weights are: with `xl 5`, M = 38, K = 24 and eps = 53,687 the bound is about 32,839,
though the provers accept that pair and no row within the stream's limit has a quotient above 32,639
there. Under `xl 5` the term (r + 2) / 2^(K+1) alone is above 32,768 for every K <= 23, so no file
with such a K passes, whatever M and the weights are, though the provers' own test of M and K
accepts K = 17 to 23 with M <= K + 13. So the check can call a quotient short where no row can leave
the range. That errs on the safe side.

<a id="bounds-rows"></a>
## The bound on one row's sum

Let u be a row of weights, a any vector in the ball of radius B and in the box, and Q = ceil(B^2).
`k_row_bounds` (through `row_bounds`) takes as the row's bound b the least of these numbers, rounded
down, because the sum u . a is an integer (the kernel adds up the abs(u_k) in 32 bits, which is
enough for d <= 2^20; the check does not test that, and the verifier's limit on d gives it):

| from | upper bound on abs(u . a) |
|---|---|
| the ball and the box, for each tau of a grid | `[ tau^2 * Q / 2,047 + 2,047 * ( sum over abs(u_k) <= tau of u_k^2 + sum over abs(u_k) > tau of (2 tau abs(u_k) - tau^2) ) ] / (2 tau)` |
| the box alone | `2,047 * sum of abs(u_k)` |
| the ball alone | `sqrt(Q * sum of u_k^2)` |

*Argument.* For any mu >= 0 and any such a, u . a <= u . a + mu (B^2 - len(a)^2) <= mu B^2 + the sum
over k of the largest value of abs(u_k) t - mu t^2 for 0 <= t <= 2,047. That largest value is u_k^2
/ (4 mu) if abs(u_k) <= 2 mu * 2,047, and 2,047 abs(u_k) - 2,047^2 mu otherwise. With mu = tau /
(2 * 2,047) the threshold is tau, and the sum is the first row of the table with B^2 in the place of
Q, which is no smaller. The right side depends on u only through abs(u_k), so it bounds -u . a as
well, and so abs(u . a). The second row is the triangle inequality with abs(a_k) <= 2,047. The third
is the Cauchy-Schwarz inequality with len(a)^2 <= B^2 <= Q. This is weak Lagrangian duality, so
every tau > 0 gives a valid bound. The problem is convex with a strictly feasible point, so the best
tau gives the true maximum over the ball and the box.

**The grid** (`tau_grid`) is every integer from 1 to 67, then each value 1.03 times the one before,
rounded down, up to 4,095, the largest absolute value of a weight that `cp_commit` admits. It
decides only how tight b is.

**Which sites.** A site's bound is the largest b over its rows: `v` over Wv with B1 (`take`, case
1); with B2, the gates over Wg or all Wgr_e (cases 3, 5, 7), the up sites over Wu or all Wur_e
(cases 4, 6, 8), and `router` over Wr (case 11).

**Published results.** The bound is a routine use of duality, not a new result. Close to it are:

- the box alone: I. Colbert et al., "A2Q: Accumulator-Aware Quantization with Guaranteed Overflow
  Avoidance", arXiv:2308.13504, section 3.2, equation 11;
- the ball alone, in an approximate form without the rounding terms: M. Salmani et al., "SLaNC:
  Static LayerNorm Calibration", arXiv:2410.10553, section 3.2 (equations 5 and 8);
- that any feasible dual point gives a guaranteed bound: E. Wong and J. Z. Kolter, "Provable
  Defenses against Adversarial Examples via the Convex Outer Adversarial Polytope", ICML 2018 (PMLR
  80), arXiv:1711.00851, section 3.2;
- the two cases of the sum have the shape of the Huber function.

<a id="bounds-rotary"></a>
## Query and key rows

The rotary step turns the sums A0 and A1 of the rows 2 i and 2 i + 1 into c A0 - s A1 and s A0 + c
A1 ([../docs/model.md](../docs/model.md), section 6, step 2, which writes C and S). `take`, case 0,
bounds a pair of rows so:

| layer | the pair's bound |
|---|---|
| `rp` 1 | `floor( sqrt(4,096^2 + 5,794) * min( B1 * sv2, sqrt(b0^2 + b1^2) ) )`, with `sv2^2 = ( n0 + n1 + sqrt((n0 - n1)^2 + 4 p^2) ) / 2` |
| `rp` 0 (c = 4,096, s = 0) | `4,096 * max(b0, b1)` |

*Argument.* By the Cauchy-Schwarz inequality each of the two rotated sums is at most sqrt(c^2 +
s^2) * sqrt(A0^2 + A1^2).

- c and s are 4,096 times a cosine and a sine of one angle, each rounded to an integer. Rounding
  moves the point (c, s) by at most 1 / sqrt(2), so c^2 + s^2 <= (4,096 + 1 / sqrt(2))^2 < 4,096^2 +
  5,794.
- sqrt(A0^2 + A1^2) is the length of the image of a, the first norm's output, under the two rows as
  a 2 x d matrix, so it is at most B1 times that matrix's largest singular value sv2. sv2^2 is the
  larger eigenvalue of the rows' 2 x 2 Gram matrix: n0 and n1 are their sums of squares and p their
  inner product (`k_pair_dots`), exact integers.
- It is also at most sqrt(b0^2 + b1^2), where b0 and b1 are the two rows' own bounds, with B1.

The rotated sums are integers, hence the floor. The site's bound is the largest over the pairs of
Wq, or of Wk. `kq` and `kk` also remove the factor 4,096. With the rotary step the bound is not
tight: it can be up to half a bit above what the ball and the box allow.

<a id="bounds-stream"></a>
## The length of the residual stream

The stream starts as a row of the embedding and receives 2 L additions. Each is the output of a
block that reads a norm's output, which lies in the ball and the box whatever the stream is, as long
as it meets condition 1. So each addition's length has a bound that does not depend on the stream,
and the stream's bound X grows by a sum, not by a product.

| what | bound | function |
|---|---|---|
| X at the first norm | X_0, the length of the embedding's longest row | `k_wide_row_squares`; `take`, kind 12 |
| X after an addition | X before it + the addition's bound | `lines` |
| V_g: a value row of key-value head g | `min( sqrt( sum over the head's dh rows j of min(2,047, rq(b_j, kv))^2 ), B1 * sv(Wv_g) / 2^kv + sqrt(dh) / 2 )` | `take`, case 1 |
| O: all heads' outputs together | `sqrt( (n_h / n_g) * sum over g of (V_g + sqrt(dh) / 2)^2 )` | `take`, case 1 |
| the addition after attention | `sv(Wo) * O / 2^ko + sqrt(d) / 2` | `take`, case 2; `lines` |
| S_j: the SiLU table's output at hidden unit j | `min( smax, 1 + the largest abs(Ts[c]) over the keys c = -y .. y )`, with `y = min(32,768, rq(b_j of the gate row, kg))` | `lay_the_gate_table`, `gate_outputs` |
| U_j: the up projection at unit j | `min(32,768, rq(b_j of the up row, ku))` | `hidden_norm` |
| H: the vector of the products s[j] u[j] | `min( B2 * min(sv(diag(S) Wu), smax * sv(Wu)) / 2^ku + len(S) / 2, sqrt( sum over j of (S_j U_j)^2 ) )` | `hidden_norm` |
| the addition after a feed-forward block | `( sv(Wd) * H + the largest sv(Wdr_e) * H_e over the experts e ) / 2^kd + sqrt(d) / 2` | `take`, cases 9 and 10; `lines` |

*Argument, attention.* b_j are the row bounds of the site `v`, and Wv_g is the head's dh rows of Wv.
A value row of head g is rq(a Wv_g, kv), clamped, where a is the first norm's output. Entry by entry
it is at most min(2,047, rq(b_j, kv)). As a whole it is at most B1 sv(Wv_g) / 2^kv long before
rounding, and rounding dh entries adds at most sqrt(dh) / 2; the clamp makes no entry larger. A
head's output entry is floor((2 acc + Z) / (2 Z)), which is acc / Z rounded to the nearest. acc / Z
is a mean of value rows with non-negative weights e_ij / Z that add up to 1, so it is not longer than
the longest of them, and the rounding adds the second sqrt(dh) / 2. n_h / n_g query heads read each
key-value head, which gives O. The product with Wo stretches O by at most sv(Wo), and sqrt(d) / 2 is
for the rounding of rq( . , ko).

*Argument, feed-forward.* S_j bounds abs(s[j]). The wide key gw is at most rq(b_j, kg) in absolute
value. The key that Ts reads is gw with its high part clamped ([../docs/model.md](../docs/model.md),
section 7): it is no larger than gw in absolute value, and it is always one of Ts's keys, so it is
at most y, cap included. (Without `gcl 1` the key is gw itself, and the cap is the lower end of
condition 6.) The 1 covers a table made with another library's exp. smax is the bound to which the
table clamps every entry, so the cap at smax holds for any such table. U_j bounds abs(u[j]); its cap
of 32,768 is the lower end of condition 7. So the products are at most S_j U_j entry by entry. As a
whole: z is the vector that the block reads (m, or xr for a routed expert), and u[j] is (z Wu)[j] /
2^ku plus a rounding of at most 1/2, so the vector of the products is at most len(diag(S) Wu applied
to z) / 2^ku + len(S) / 2 long, where diag(S) Wu is Wu with row j multiplied by S_j, and z lies in
the ball of radius B2. sv(diag(S) Wu) is at most smax sv(Wu); both are computed upper bounds, and
either can be the smaller. The down product stretches each vector by at most its matrix's sv, and
sqrt(d) / 2 is for the rounding of rq( . , kd).

*Experts.* A dense layer has no expert term. In a layer with experts, Wd, H, `kg` and `ku` are the
shared expert's, and H_e comes from expert e's matrices with `kgR` and `kuR` and the same B2. A
token goes to exactly one routed expert, so the experts contribute their maximum, not their sum.

*Argument, the stream.* By induction over the 2 L additions, in the order of the forward pass, for
the rows of all positions at once: attention at one position reads value rows of earlier positions,
and each of these comes from a norm's output at its own position. Before the first addition the row
is a row of the embedding, at most X_0 long. Suppose that the bound X holds up to some addition and
is within the stream's limit. Then the row meets condition 1, so the norm that the next block reads
puts its output into the ball and the box. The block's output is then at most as long as the
addition's bound, which depends on the weights and the ball only, and the triangle inequality gives
X after the addition. This argument is not given in full detail.

**The lines support each other in the order of the forward pass.** A line is justified only if every
stream line before it fits. By this argument a verdict that every bounded site fits leans on no
range condition that the check has not itself bounded. In a table with a short stream site the lines
after that site are not established, and after a short up site (without `gcl 1`: or gate site) the
stream's lines hold only for requests that meet condition 7 (or 6) there.

<a id="bounds-singular"></a>
## The largest singular value of a matrix

`operator_norm` returns an upper bound on sv(W), under the hypothesis below, by Gram iteration.
(Power iteration approaches sv(W) from below, so it proves no bound.) With G = W^T W, or W W^T if
that is smaller, and fro(A) the square root of the sum of all entries' squares (the Frobenius norm),
for every k >= 0: `sv(W)^(2^(k+1)) <= fro( G^(2^k) )`.

Source: B. Delattre, Q. Barthélemy, A. Araujo, A. Allauzen, "Efficient Bound of Lipschitz Constant
for Convolutional Layers by Gram Iteration", ICML 2023 (PMLR 202), arXiv:2305.16173, Theorem 3,
first item (its G is W here, and its t is k + 2). The treatment of rounding below is not from that
paper.

- **Levels.** A_0 is W in binary64, which holds it exactly. A_1 is G, and each further level is A^T
  A of the one before: at most five products (`SQUARINGS` = 4), made by cuBLAS's `cublasDgemm`. Each
  product is multiplied by 4^-e_j, as if A_j had been divided by 2^e_j, the power of two that brings
  f_j (next point) into 1/2 .. 1.
- **f_j, an upper bound on fro(A_j).** The squares are summed with rounding to the nearest, the sum
  is multiplied by 1 + 2^-20 for the additions' errors (all terms are non-negative, and no chain has
  2^27 additions, each off by at most 2^-53 of its result), and product and root are rounded up
  (`frobenius`).
- **Back to W.** At the last level sv(A) <= fro(A) <= f. From there back to W:
  `sv(A_j)^2 <= 4^e_j * ( sv(A_(j+1)) + 2 * (m_j + 64) * 2^-53 * (f_j / 2^e_j)^2 )`, with m_j the
  product's inner length. sv(A)^2 = sv(A^T A) holds for every matrix, so the computed matrices need
  not be symmetric. A_(j+1) is 4^-e_j A_j^T A_j plus a rounding error E, so sv(4^-e_j A_j^T A_j) <=
  sv(A_(j+1)) + sv(E) <= sv(A_(j+1)) + fro(E). The second term bounds fro(E): a classical product of
  inner length m in binary64, in any order of summation, is off by at most gamma * fro(A)^2 in the
  Frobenius norm, with gamma = m 2^-53 / (1 - m 2^-53) (N. J. Higham, "Accuracy and Stability of
  Numerical Algorithms"). (m + 64) 2^-53 is at least gamma at these sizes, and twice that is added.
- **The result** is the least bound over the levels reached; each of them is valid. The function
  returns early once a level improves it by under 1 part in 256, once it is at most the caller's
  `enough`, or when a guard fails. Only `take`, case 10, passes `enough`, so that sv(Wdr_e) is
  refined only while its product with H_e could still be the largest.

**The hypothesis.** The error term assumes that the library computes a classical product in true
IEEE binary64. `begin` sets `CUBLAS_PEDANTIC_MATH`, and two guards test the assumption in every run:

- `product_holds`, on every product C = alpha P^T P (alpha is the 4^-e_j above). For one fixed
  vector r of entries +1 and -1 it compares C r with alpha P^T (P r), both computed by this file's
  own kernels. The difference's length may be at most (m' + 64) 2^-53 times an upper bound on alpha
  fro(P)^2, with m' the larger dimension of P: half of what the error term allows. One fixed vector
  does not show that a product is right.
- `gram_is_exact`, on the first product of a plain weight matrix (not of diag(S) Wu), which true
  binary64 gives exactly (its entries are sums of at most 2^29 products below 2^24). Every entry
  must be an integer below 2^53; and, modulo 2^64, the trace must be the sum of the weights' squares
  and the sum of all entries the squared length of W's row sums (or column sums). An error that
  changes none of the three is not noticed.

After a failed guard the check has no bound at all, so a false alarm is safe. By the code,
`product_holds` also fails where the matrix or slice whose product it tests holds only zeros,
because alpha is then infinite; this has not been run. The inequality is standard; the error
recursion is argued, not machine-checked.

<a id="bounds-margin"></a>
## The head margin

A margin is the difference of two logits of one position, less 1 at some entries, plus what the
acceptance policy adds for that position. `lines` bounds it from public sizes alone:
`2 * d * 2,047 * 4,095 + 1 + 2^37 + 2^32`.

*Argument.* 2,047 bounds an entry of the final norm's output and 4,095 a weight of the output
matrix, so a logit is at most d * 2,047 * 4,095 in absolute value. A position adds the constant 2^37
or a cost of 32 bits (`prover/head/forward.cuh`); the bound adds both, and its + 1 is slack. The
limit is the largest number that a margin's three digits hold at 7, the smaller of the two admitted
values of the line `sh` of `policy.txt` (not a layer's `sh` in `KL.bin`), so the line covers both.
The check does not read the output matrix (`take`, kind 13): that its integers are at most 4,095 is
shown by `cp_commit`'s test of the integers' ranges, which comes first.

*The lower end.* The forward pass also tests that a margin is not negative, and no line of the table
stands for that test. By the argument that follows it does not fail for an honest prover: a margin
is at least what the position adds, less the position's logit gap
([../docs/model.md](../docs/model.md), section 8). A cost within the policy's clip (the line `C` of
`policy.txt`) is the gap plus a positive multiple of 2^sh. Above the clip the position adds 2^37,
and the gap is at most 2 * d * 2,047 * 4,095 + 1, which is below 2^37 wherever this line fits (it
fits for d <= 7,942). At the last position both are 0, because the output token is the first argmax.

<a id="bounds-rounding"></a>
## What is exact and what is rounded up

| kind of step | how it is computed |
|---|---|
| sums over weights: a row's sum of absolute values and of squares, the sums over each step of the grid, a pair's inner product | exact, in integers |
| every other floating-point step of a bound, outside the matrix products and the SiLU table | rounded up: on the host the next binary64 number above the result (`up`, `add_up`, `mul_up`, `div_up`, `sqrt_up`), on the GPU the instructions that round up (`__dadd_ru` and the like) |
| matrix products, the sums for Frobenius norms, the guards' own sums | rounded to the nearest; the first two are covered by an error term |
| what the table prints | log2 of a bound, rounded up to three decimal places; log2 of (limit + 1). Bits short comes from the numbers themselves |

Where a bound lies above anything that a prompt can reach:

- The ball and the box hold many vectors that no prompt produces. This is the largest slack, and it
  depends on the weights. `worst_case.py against` shows the distance to the calibration set's maxima
  at the sites `q`, `k`, `v` and the three up sites.
- The stream's bound adds the worst case of all 2 L additions, as if all pointed the same way.

**Which error is safe.** A bound that is too large is safe: at worst the check refuses scales that
would have served every prompt, and the model is made coarser than it had to be. A bound that is too
small is the error that matters: the check would pass although some prompt can drive a value beyond
its limit, and that request would fail, which tells its sender whether a hidden value crosses a
bound (README.md, "What the tools do not check", "Other prompts"). So every choice in the code is
meant to err upwards.

<a id="bounds-table"></a>
## The lines of the table

`cp_commit -worst-case-table FILE` writes the table to a new file that only its owner can read; no
build prints it. Its numbers are computed from the weights: keep it as secret as the weights
(README.md, "What the files reveal"). A line that begins with `#` names the fields of the lines
after it.

| line | fields | what `worst_case.py` reads of it |
|---|---|---|
| `worst-case 1` | the first line | the line |
| `d` | the width | the line |
| `site` | site; layer (L after the last layer); the parameter that moves the site's range (the shift; `K1`, `K2` or `KF` for a quotient; `e_res` for the stream; `-` for the head margin); its value (0 for the last two); bound; limit + 1; bits short | site, layer, bits short; `against` also the value and the bound |
| `layer`, `embedding`, `split`, `largest`, `needs` | the parts of the stream's bound, named by the `#` line before each (its "heads' outputs" is O, its h is H). A layer's two additions are given without sqrt(d) / 2; the largest S_j are plain integers; the experts' largest sv(Wdr_e) is coarse, because of `enough` | - |
| `what if` | a value of `smax`; the stream's last bound under it; bits short. First the parameter file's own value, then each smaller admitted one (2,047, 1,023, 511, 255) | the value, bits short |
| `verdict` | `fits` or `short`; the last line | the line |

The lines `what if` are computed from what `take` recorded: each S_j is capped at the value, and
sv(diag(S) Wu) is not computed again; the lesser of the recorded one and the value times sv(Wu)
stands for it. So such a line can be above what a run under that value would give. These lines never
count for the verdict, and a parameter file with a line `smax` is outside the standard parameter
form.

<a id="bounds-short"></a>
## When a site does not fit

| case | what `cp_commit` does |
|---|---|
| `read` cannot handle the parameters, for example another profile or cell width, a shift outside 1 to 32, a layer's `sh` outside 4 to 12, a norm's M, K or eps out of range, a line `dcl` | prints the reason; status 2, before a weight is read |
| every bounded site fits | prints `cp_commit: the worst-case check: every bounded site fits: ...`, writes the line `ranges` of `isk.txt`, goes on |
| a site is short | prints `cp_commit: the worst-case check: NOT every bounded site fits: ...`; status 7, before the seed is drawn; nothing is committed; the table stays |
| no bound: a matrix product failed a guard, as the product of a matrix or slice that holds only zeros also does | prints `cp_commit: the worst-case check: <reason>: no bound can be given`; every line of the table has 99 bits short; then as for a short site |
| the check cannot go on (`stop`), among other causes: cuBLAS refuses the mode, a product fails or overflows, a matrix comes out of order | prints the reason; status 2 |

The line `... every bounded site fits: ...` ends "at none of the sites that the check bounds can a
prompt drive the value out of its range". That is worded more firmly than what is shown: the bounds
are argued, not proved.

`Check` also gives no bound if a matrix has a weight beyond 4,095. `cp_commit` does not get that
far: its own test of the integers' ranges ends such a run first, without a table.

After status 7, `worst_case.py` turns the bits short into coarser scales. The new scales are public
and follow the table, and so the weights: README.md, "What the files reveal", "What a passed check
tells". Its docstring has the rule for each site; README.md, "A real model", has the call;
README.md, "Where a step stops", says what to do after its other endings; README.md, "Making a cell
coarser", says which sums of scales stay the same. It refuses a table with a 99 (status 2), and it
computes no bound itself.

<a id="bounds-not-covered"></a>
## What the bounds do not cover

- **Two values are not bounded,** because, by a reading of the code of other programs and an
  argument that is not written out here, public sizes exclude an overflow: the multiplicities of the
  lookup arguments and the remainder of attention's division. The prover refuses a parameter file
  that would allow either. The test that a margin is not negative has no line of the table either
  ("The head margin").
- **The limits are written twice in the programs,** in `worst_case.cuh` and as literal numbers in
  the prover's kernels, and the compiler does not link the two. No test in this repository compares
  them, and nothing shows that the forward pass has no other range test. The prover also fails a
  request if an index that it computes from the request's values falls outside its table (the flag
  `INDEX`); that this does not happen while every value is in its range is assumed here, not argued.
  Nothing compares this page's numbers with either.
- **What is tested.** No test in this repository compares the check's bounds with true values, or
  makes either guard (`product_holds`, `gram_is_exact`) fail.
- **The verdict is about the integers that the check read, under the three configuration files that
  it read.** It says nothing of other weights or other files. The line `ranges` of `isk.txt` holds
  the digest of the three public configuration files and nothing that depends on the weights, and no
  key protects it. So it shows only that a run of `cp_commit` under these files has passed the
  check, and only if nothing else has written `isk.txt`. That the prover decodes the same integers
  that the check bounded has not been read in the sources.
- **Another machine may give other bounds.** Except for the first product of a plain weight matrix,
  which is exact, `operator_norm` works with rounded products, and the order in which `cublasDgemm`
  adds is the library's choice: another release of cuBLAS or another kind of GPU may add in another
  order. Each outcome is a bound by the argument of "The largest singular value of a matrix", which
  holds for any order of summation. But the bounds can differ: in their last digits, or by more
  where the function returns early on one machine and makes a further product on the other. So a
  site of the stream (no other site's bound uses `operator_norm`) whose bound lies very close to its
  limit can fit on one machine and be short on another, where `cp_commit` then makes no commitment
  under these scales (status 7). Whether such a site gets the same verdict under every release of
  cuBLAS is not known, and the tables of two models of GPU have not been compared.
- **If a value did leave its range.** At a site where the forward pass raises a range flag the
  request fails ("Which error is safe"). Where the prover has no test of its own, as for condition
  2, the request may not be stopped, and what its files would show of hidden values is not known.
- **The times that `cp_commit` prints.** `cp_commit` prints times of the check on its own output,
  which is the provider's: at every eighth layer the line
  `cp_commit: the worst-case check: at layer <l> of <L>, <s> s in the bounds so far`; after the pass
  over the weights `cp_commit: the check of the weights: <s> s`; and after a verdict that every
  bounded site fits `cp_commit: phase check: <s> s`. These times follow the weights: how many matrix
  products `operator_norm` makes depends on the matrix. What they tell of the weights has not been
  examined or measured. They count as timing, which nothing on
  [../docs/what-is-not-hidden.md](../docs/what-is-not-hidden.md) bounds.
- **One more printed line depends on the weights.** Before the check's verdict, with the check or
  without it, `cp_commit` prints the outcome of its test of the integers' ranges,
  `cp_commit: every committed integer lies in its declared range`, and it ends that line with
  `, except for the embedding, which is wider than the policy's wb_emb` where the greatest absolute
  value among the embedding's integers is greater than 2^20 - 1, the greatest that the width line
  `wb_emb` declares. That is one bit about the weights. What it tells of them beyond that has not
  been examined. This is read in the source, without a run. This line and the times above stay
  outside what a client learns only while `cp_commit`'s output stays with the provider: README.md,
  "What the files reveal".
- **Scales that are fitted again to the table.** `worst_case.py` chooses new scales from the table,
  so scales that are published afterwards depend on the weights' norms: a few bits for each site, by
  an estimate that no run and no calculation backs. Nothing bounds what they tell. The settings files
  are on the list of what a client learns by design
  ([../docs/what-is-not-hidden.md](../docs/what-is-not-hidden.md)). README.md, "What the files
  reveal", says what the published scales tell of every weight matrix, however they were fitted, and
  that they are a test of a guessed model.
