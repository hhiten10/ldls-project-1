# Own experiment: pre-registration (written 2026-11-10, BEFORE any run)

## Change exactly one thing
The ggml quantization type of the `ssm_in` tensors of **mamba2-1.3b** (the largest weight: 55.7 % of the
bytes read per decoded token at Q4_K_M). Everything else is identical: every variant is produced by the
same `llama-quantize --allow-requantize` path from the same Q8_0 GGUF with base type Q4_K_M, and only
`--tensor-type ssm_in=<T>` differs. The baseline (`ssm_in=q4_K`) goes through the same path, so the
double-quantization effect is common to all variants.

Variants: `ssm_in` ∈ {Q4_0, Q4_K, Q5_K, Q6_K, Q8_0}.

## Hypotheses and pass/fail bars (decided now)
1. **Decode is memory-bandwidth bound.** Predicted decode tok/s for each variant =
   `BW_eff / bytes_per_token(variant)`, where `BW_eff` is calibrated ONCE from the Q4_K baseline
   (measured tok/s × its bytes/token). **Pass** if every variant's measured tg128 is within ±10 % of
   this prediction. A systematic miss in one direction for higher-precision types would mean
   dequantization compute, not bytes, sets the pace.
2. **Same bytes, different kernel.** Q4_0 and Q4_K store ~4.5 bits/weight. If decode is bandwidth
   bound their decode speeds differ by < 5 %. Prefill (pp512, compute bound) is expected to differ more.
3. **Quality cost.** Perplexity (same 40×512-token wikitext sample) is non-increasing as `ssm_in`
   precision rises; reported with llama-perplexity's ± error, so differences smaller than the error
   are called "not resolved", not "better/worse".

## Measurements
tg128 (d=0) and pp512 at 8 threads pinned to physical cores, `-r 3` (mean ± sd); peak RSS; perplexity.
