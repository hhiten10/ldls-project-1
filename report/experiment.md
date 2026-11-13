# Own experiment: quantization type of one tensor (`ssm_in`) — results vs pre-registration

Pre-registration: `docs/experiment_preregistration.md` (written before running). mamba2-1.3b, only the type of
the `ssm_in` tensors changes (55.7 % of bytes/token); all variants requantized from the same Q8_0 via the same
path. tg128 / pp512, 8 threads on physical cores, 3 repetitions (mean ± sd); PPL on 40×512 wikitext-2 tokens.

| ssm_in | file | bytes/token | predicted tg128 | measured tg128 | error | achieved BW | pp512 | PPL |
|---|---|---|---|---|---|---|---|---|
| Q4_0 | 846 MB | 993 MB | 18.53 | 17.24 ± 0.43 | −7.0 % | 17.1 GB/s | 126.1 ± 0.8 | 11.64 ± 0.31 |
| **Q4_K** (baseline, calibrates BW_eff) | 846 MB | 993 MB | — | 18.53 ± 0.42 | — | 18.4 GB/s | 119.8 ± 23.0 | 11.56 ± 0.31 |
| Q5_K | 951 MB | 1098 MB | 16.76 | 16.62 ± 0.47 | −0.8 % | 18.2 GB/s | 121.5 ± 20.3 | 11.49 ± 0.31 |
| Q6_K | 1062 MB | 1209 MB | 15.22 | 14.94 ± 1.23 | −1.8 % | 18.1 GB/s | 133.9 ± 1.2 | 11.48 ± 0.31 |
| Q8_0 | 1265 MB | 1412 MB | 13.03 | 14.26 ± 0.02 | +9.4 % | 20.1 GB/s | 105.9 ± 2.6 | 11.45 ± 0.30 |

predicted = BW_eff / bytes, BW_eff = 18.53 tok/s × 993.4 MB = 18.4 GB/s (from the Q4_K baseline).
achieved BW = measured tok/s × bytes/token.

**H1 — bytes predict decode within ±10 %: PASS** (−7.0, −0.8, −1.8, +9.4 %). Going from 4.5 to 8.5 bits on
the biggest tensor costs decode speed almost exactly in proportion to the extra bytes.

**H2 — same bytes ⇒ < 5 % decode difference (Q4_0 vs Q4_K): FAIL.** Q4_0 is 7 % slower at identical bytes,
so at equal bytes the dot-product kernel matters: on this AVX2 build Q4_K gets the repacked 8×8 GEMV
(`ggml_gemv_q4_K_8x8_q8_K` dominates the decode perf profile), which streams better than the Q4_0 path.

Q8_0 beating its prediction (+9.4 %) has two candidate explanations we **cannot separate** with this data:
(a) Q8_0's simpler kernel streams more efficiently, or (b) part of the per-token cost is *not* proportional to
weight bytes (the gap analysis shows ~17 % barrier waits and ~200–400 MB of state traffic that a bigger
`ssm_in` does not grow), so heavier variants beat a purely proportional prediction. What we can rule out is
"4-bit kernels cannot saturate the bus": Llama-3.2-1B's Q4_K matmuls reach 29.1 GB/s ≈ 95 % of read
bandwidth on the same machine (results/summary_gap.csv).

**H3 — quality: monotone, but NOT RESOLVED.** PPL falls steadily 11.64 → 11.45 as `ssm_in` precision rises,
but every step is smaller than llama-perplexity's ±0.30 error on this sample, so by the pre-registered rule
no pairwise difference is claimed.

Prefill (pp512) varies non-monotonically with large spreads (±20 on two variants): prefill is compute-bound
and sensitive to which repacked GEMM kernel each type gets; it was not a pre-registered outcome.

**Take-away.** On this laptop, `ssm_in` at Q4_K is the best trade: Q4_0 saves no bytes and is slower; Q5_K/Q6_K
cost 10–19 % decode speed for a PPL change below the noise of our sample; Q8_0 costs 23 %.
