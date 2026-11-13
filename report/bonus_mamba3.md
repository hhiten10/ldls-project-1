# Bonus: a Mamba-3 decode step on CPU — does MIMO pay off?

## What was built
* **CPU reference** (`bonus/ref/mamba3_ref.py`): one Mamba-3 layer decode step (SISO and MIMO) in pure PyTorch,
  assembled from the official repo's own pure-PyTorch reference functions
  (`selective_state_update_fused_ref_v2`, `apply_rotary_qk_inference_reference`, `Mamba3.step` preprocessing,
  state-spaces/mamba @ e9594ce1). The released `Mamba3.step` itself only runs on CUDA (CuTe DSL + Triton).
* **Pretrained checkpoints are public** (`state-spaces/mamba3-{siso,mimo}-1.5b`), so everything below uses
  **real weights and real hidden states** (wikitext-2 text through layers 0–11 of the 1.5B model), not random tensors.
* **C++ kernels** (`bonus/kernel/`): scalar and AVX2/FMA SISO + MIMO (rank 4) decode steps, and a Mamba-2 step
  at the same state size. No AVX-512 needed (the laptop has none).
* **ggml operator** (stretch): the Mamba-3 step wrapped with `ggml_custom_4d` and run inside a ggml graph
  (`bonus/ggml/ggml_driver.cpp`); ggml's own `SSM_SCAN` timed at the same shape as the Mamba-2 baseline.

## Correctness (bars fixed before running)
| check | result | bar |
|---|---|---|
| our step vs upstream `mamba3_siso_step_ref` (f64, real layer, 128 steps) | 2.2e-16 | 1e-5 |
| our step vs upstream chunked `mamba3_siso_fwd_ref` (independent formulation) | 1.3e-7 | 1e-4 |
| our MIMO step vs upstream `mamba3_MIMO_step_ref` | 2.2e-16 | 1e-5 |
| full 1.5B model, token-by-token CPU decode, wikitext-2 PPL (1024 tok) | SISO 12.23, MIMO 11.79 | sane |
| MIMO with the *other* rotary layout (interleaved instead of halved) | PPL 24.74 → wrong layout | — |
| **SISO AVX2 kernel vs float64 reference, 1000 real decode steps** | worst per-step rel-L2 **1.45e-5** (PyTorch fp32 floor 1.44e-5) | 1e-4 |
| **MIMO AVX2 kernel vs float64 reference, 1000 steps** | **1.60e-5** (floor 1.59e-5) | 1e-4 |
| our Mamba-2 AVX2 step vs ggml `SSM_SCAN`, 1000 steps | 4.9e-6 | 1e-5 |
| Mamba-3 ggml custom op vs direct kernel call, 1000 steps | bit-exact | 0 |

The kernels' error equals what PyTorch itself gets in float32: the remaining error is float32 rounding, not a bug.

## The update rule (what costs what)
Per head h, per token (P = 64, N = 128, R = 1 or 4):
`α = exp(A·Δt)`, `β = (1−λ)·Δt·α`, `γ = λ·Δt` (trapezoidal discretization; λ = sigmoid(trap)),
B and C get a bias and a data-dependent rotation (the complex-valued state, realized as RoPE with a
cumulative angle), then
`S ← α·S + Σ_r γ·(x·w_r) ⊗ B_r + Σ_r β·(x_prev·w_r) ⊗ B_prev,r`,  `y_r = S·C_r + D·x_r`.

FLOPs per state element: Mamba-2 5, SISO 7, **MIMO 25 (3.6× SISO)**. Bytes per element: the fp32 state is
read and written once (8 B) for all three — **MIMO adds compute without adding state traffic**.
Arithmetic intensity: Mamba-2 0.63, SISO 0.89, MIMO 3.16 FLOP/B.

## Timing (µs per layer decode step, mean ± sd of 3 repetitions, 64×64×128 fp32 state = 2 MB)
| regime | threads | Mamba-2 | M3 SISO | M3 MIMO | MIMO/SISO |
|---|---|---|---|---|---|
| cache-hot: 1 layer (2 MB stays in L2/L3) | 1 | 52.4 ± 0.5 | 144.7 ± 6.2 | 423.4 ± 13.1 | **2.93×** |
| cache-hot | 8 | 16.7 ± 0.8 | 37.1 ± 0.9 | 110.5 ± 2.8 | **2.98×** |
| DRAM-cold: 24 layers round-robin (48 MB ≫ 8 MB L3) | 1 | 337.2 ± 3.2 | 420.2 ± 5.1 | 587.3 ± 2.2 | **1.40×** |
| DRAM-cold | 8 | 250.7 ± 1.1 | 267.8 ± 0.5 | 319.5 ± 0.8 | **1.19×** |

"DRAM-cold" is the regime a real model decodes in: between two tokens of the same layer, the other 23 layers'
states and ~1 GB of weights stream through the 8 MB L3, so every layer's state comes from DRAM. Caveat: the
microbenchmark runs the 24 state updates back to back, so state reads and dirty write-backs compete with each
other; in a full model they interleave with weight streaming. The cold numbers are therefore an upper bound on
the per-step cost, but the *ratio* MIMO/SISO is measured under identical conditions.

Hardware counters (1 thread, `perf stat`): IPC hot → cold: Mamba-2 2.03 → **0.32**, SISO 1.94 → **0.67**,
MIMO 1.91 → **1.40**. Cold Mamba-2/SISO move state at 16–17 GB/s ≈ STREAM *scale* (read-modify-write,
15 GB/s), i.e. they sit on the memory roof; MIMO keeps the core busy during the same memory time.

## Answer
**On this laptop CPU, MIMO's extra compute comes at little cost in the regime that matters, but not in general.**
In a real decode, each layer's 2 MB state is cold, the step is memory-bound (SISO runs at 0.67 IPC, on the
read-modify-write bandwidth roof), and MIMO's 3.6× FLOPs raise the step time by only **19 % with 8 threads**
(40 % with 1 thread, where a single Zen 2 core has less spare FMA throughput per byte). When the state is
cache-resident the step is compute-bound and MIMO costs the full **~3×**. Per token of a 24-layer 1.5B model,
the SSM steps are ~6.4 ms (SISO) vs ~7.7 ms (MIMO) at 8 threads, next to ~30 ms of weight streaming
(≈0.9 GB at Q4 over 30.5 GB/s read bandwidth) — an estimated **≈3–4 % slower token** for a model that scores
better (PPL 11.79 vs 12.23 on our sample). This mirrors the GPU argument, with one CPU-specific caveat:
the Zen 2 machine balance (~20 FLOP/B all-core) leaves less headroom than a GPU, so a higher MIMO rank
(or a cache-resident state, e.g. batch-1 tiny models) would cross into compute-bound territory.

## Side finding: llama.cpp's Mamba-2 state update is out-of-place
Code-verified (`src/models/mamba-base.cpp`): `ggml_ssm_scan` writes y *and the new state* into a fresh output
tensor, and the graph then `ggml_cpy`s the state back into the recurrent cache. Besides reading the old state and
writing back the new one (the 2 passes an in-place update needs), this allocates and later writes back a second
2 MB buffer per layer per token.
* **Microbenchmark** (24 layers back-to-back, nothing else running, so every byte is charged to the op):
  ggml `SSM_SCAN`+`CPY` **563 µs** vs our in-place Mamba-2 step **251 µs** per layer-step (8 threads), 2.2×.
* **Inside the real model** the per-op profiler charges `SSM_SCAN` only ~159 µs per layer and the copies ~34 µs:
  the copy reads a source that is still in cache, and the extra dirty lines are written back later, during the
  next weight-streaming `MUL_MAT`s, which absorb the cost. So the in-model penalty is real but hidden inside
  MUL_MAT time, and smaller than the microbenchmark suggests (it overlaps with weight streaming).
  Quantifying it in-model would need a patched in-place scan — listed as future work, not claimed as a speed-up.
