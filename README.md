# CPU inference of Mamba-2 (Option 3) + a Mamba-3 decode step on CPU (bonus)

Laptop CPU inference study of **Mamba-2** (1.3B, 2.7B) with **llama.cpp (CPU only)**, compared with
**Mamba-1** (1.4B) and a dense **Llama-3.2-1B** contrast, plus a hand-written **Mamba-3 SISO / MIMO
decode step** (C++/AVX2) validated against the official reference and wrapped as a ggml operator.

Everything here is scripted; another team can rerun it end to end with the commands below.

## Hardware & software
| | |
|---|---|
| CPU | AMD Ryzen 7 4800H (Zen 2, 8 cores / 16 threads, 2 CCX × 4 MB L3, 512 KB L2/core), AVX2 + FMA, **no AVX-512**, not hybrid |
| Memory | 2 × 8 GB DDR4-3200, dual channel (51.2 GB/s theoretical) — measured: see `results/hw/stream.csv` |
| OS | Windows 11 + WSL2 Ubuntu 24.04 (kernel 6.6.87.2), WSL sees 7 GB RAM, 2 GB swap |
| Engine | llama.cpp commit in `docs/llama_cpp_commit.txt`, CPU-only build (`GGML_NATIVE`, OpenMP) |
| Power | charger connected, Windows "Turbo"/performance plan, browser & IDE closed; CPU frequency logged during runs (`results/hw/cpu_freq_log.csv`) |
| Threads | pinned one per **physical** core (logical CPUs 0,2,…,14) via OpenMP places |

## Models (quantization rules of the spec)
| model | source | quantizations |
|---|---|---|
| Mamba-2 1.3B | `state-spaces/mamba2-1.3b`, converted with the pinned llama.cpp | Q4_K_M, Q8_0 |
| Mamba-2 2.7B | `state-spaces/mamba2-2.7b` | Q4_K_M (Q8_0 not allowed > 1.5B) |
| Mamba-1 1.4B | `state-spaces/mamba-1.4b-hf` | Q4_K_M, Q8_0 |
| Llama-3.2-1B (base) | `unsloth/Llama-3.2-1B` (ungated mirror of `meta-llama/Llama-3.2-1B`), converted with the pinned llama.cpp (`scripts/02b_llama_base.sh`) | Q4_K_M (shared reference model) |

An earlier pass used the Instruct variant (`bartowski/Llama-3.2-1B-Instruct-GGUF`); those results are kept in
`results/llama_instruct/` and `results/rudra/instruct/` for comparison, but every reported Llama number is the base model.
| Mamba-3 1.5B SISO / MIMO | `state-spaces/mamba3-{siso,mimo}-1.5b` (bonus, PyTorch + C++) | bf16 checkpoints |

Conversion uses a lossless 16-bit intermediate (`--outtype auto`) that is deleted after quantizing.

## Reproduce
All scripts run inside WSL2 from the repo root; heavy files live in `~/p1work` (ext4, not `/mnt/c`).
```bash
bash scripts/run_logged.sh 00_setup_venv.sh     # python venv (CPU torch, gguf, plotting)
bash scripts/run_logged.sh 01_build.sh          # llama.cpp CPU-only, pinned commit
bash scripts/run_logged.sh 01b_build_prof.sh    # same commit + opt-in per-op profiler (docs/ggml-op-profile.patch)
bash scripts/run_logged.sh 02_get_models.sh     # download, convert, quantize
bash scripts/run_logged.sh 04_correctness.sh    # perplexity + sample outputs
bash scripts/run_all_timed.sh                   # STREAM, roofline, bonus timing, sweeps, profiling, experiment, plots
```
Bonus only:
```bash
bash bonus/kernel/build.sh && bash bonus/ggml/build.sh
python bonus/ref/verify_ref.py ~/p1work/models/src            # our reference == upstream references
python bonus/ref/make_trace.py <ckpt> <text> <tok> 12 1000 <out>   # real-weight 1000-step trace
bonus/kernel/m3bench check <trace>                             # C++ kernels vs float64 reference
bash bonus/run_bench.sh                                        # timing, hot vs cold cache, perf counters
```
PARAM Rudra shared reference run: `rudra/setup_login.sh <commit>` on the login node, then
`sbatch rudra/reference.sbatch` (QoS: 4 CPUs, 5 min).

Report and slides: `report/report.html` → `report/report.pdf` (Chrome headless print, 4 pages A4);
`report/slides.pptx` from `node report/make_slides.js` (pptxgenjs 3.12.0; reads the CSVs in `results/`).

## Headline results (8 threads, mean ± sd of 3)
| | Mamba-2 1.3B Q4 | Mamba-2 2.7B Q4 | Mamba-1 1.4B Q4 | Llama-3.2 1B Q4 |
|---|---|---|---|---|
| decode tok/s, ctx 0 → max | 18.2 → 18.1 (32k) | 10.0 → 10.1 (32k) | 23.2 → 24.6 (32k) | 29.5* → 3.7 (16k) |
| prefill tok/s, 512 → max | 122 → 128 | 64 → 65 | 91 → 93 | 205 → 90 |
| % of read-bandwidth roofline (decode) | 59 % | 62 % | 65 % | 77 %* |
| SSM_SCAN share, decode / prefill | 13 % / 9 % | 12 % / 7 % | 9 % / 33 % | — |

\* Llama-3.2-1B base; its 0-context decode is bimodal (one slow repetition in 3, twice): the identical test in the thread sweep gives 34.4 ± 0.2 tok/s = 90 % of the roofline.

* Read bandwidth 30.5 GB/s (8 threads), saturated by 3–4 cores; decode stops scaling there, prefill keeps scaling.
* Gap explained in `results/summary_gap.csv`: state traffic (out-of-place SSM update) slows the weight stream, ~1150 graph nodes/token → 17 % barrier waits; Llama's Q4_K GEMV reaches 95 % of read bandwidth.
* Threads × context × quantization cross sweep: `results/summary_cross.csv`, `plots/threads_x_context.png` — Mamba is flat in context at every thread count and saturates by 4 threads; Llama at 8k keeps scaling (2.1 → 5.1 → 8.3 tok/s).
* Own experiment (`docs/experiment_preregistration.md`, `report/experiment.md`): bytes predict decode within ±10 % (pass); same-bytes Q4_0 vs Q4_K differ 7 % (fail).
* PARAM Rudra reference (base Llama-3.2-1B, 4 cores, QoS limit): pp512 298.5 / 300.2, tg128 39.27 / 39.30 tok/s, peak RSS 1.32 GiB (`results/rudra/`).
* Bonus (`report/bonus_mamba3.md`): SISO / MIMO AVX2 kernels match the float64 reference over 1000 real steps
  (1.45e-5 / 1.60e-5 = PyTorch fp32 floor); MIMO costs +19 % per step when the state is DRAM-cold (8 thr), ~3× when cache-hot.

## Methodology notes (things that went wrong, kept on record)
* A model conversion filled the C: drive (WSL's disk lives there) via an F32 intermediate; fixed by a lossless 16-bit
  intermediate, deleting checkpoints after quantizing, and a free-space guard in `02_get_models.sh`.
* Swap guard: originally "any swap-in invalidates a run". Changed post hoc to "> 256 pages (1 MB)" after an idle
  daemon's 9–22 page swap-ins tripped it on one config; raw counts for every run are in `results/raw/runs_meta.csv`.
* Two pipeline crashes (retry logic under `set -e`; SIGPIPE from `perf report | head`) were fixed and the affected
  stages rerun (`scripts/run_rest.sh`).
* Requirement audit pass (`scripts/run_gaps.sh`, `run_final.sh`, `run_battery_fix.sh`): switched to the base Llama-3.2-1B,
  added the threads × context cross sweep, and logged peak RSS for every perplexity / sample / profiling run. That
  logging caught sample generation running at llama.cpp's default context (Llama's trained 131k → 5.25 GiB, over the
  4.5 GB budget); re-run with `-c 2048` (1.31 GiB). Two runs overlapped brief off-AC windows (frequency log) and were
  redone on AC; originals kept as `*.battery.jsonl`.
* A first, standalone "llama.cpp moves the state twice, 2.2× slower" estimate was narrowed after the in-model
  profile showed the copy is mostly hidden behind weight streaming (see `report/bonus_mamba3.md`).

## Layout
`scripts/` benchmark & analysis pipeline · `results/` raw jsonl/CSV + summaries · `plots/` figures ·
`bonus/` Mamba-3 reference, kernels, ggml driver · `rudra/` cluster scripts · `docs/` notes, patch, pre-registration · `report/` report & slides
