"""
Bonus analysis: does MIMO's extra decode compute come at little cost on this CPU?
Reads results/bonus/kernel_timing.csv (+ check_errors_*.csv) and writes:
  plots/bonus_step_time.png      time per layer-step: Mamba-2 vs M3 SISO vs M3 MIMO, hot vs cold, 1 vs 8 threads
  plots/bonus_roofline.png       achieved GFLOP/s vs arithmetic intensity against measured ceilings
  plots/bonus_errors.png         per-step error over 1000 steps vs the torch fp32 floor
  results/bonus/summary.csv      MIMO/SISO time ratios per regime
usage: python -I analyze_bonus.py <repo>
"""
import csv, os, sys
from collections import defaultdict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = sys.argv[1]
RES, PLOTS = os.path.join(REPO, "results", "bonus"), os.path.join(REPO, "plots")
INK, INK2, MUTED, GRID, AXIS, SURF = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF, "font.size": 10,
                     "text.color": INK, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.edgecolor": AXIS, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
                     "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False})
COL = {"mamba2": "#2a78d6", "m3_siso": "#eb6834", "m3_mimo": "#1baf7a"}
LBL = {"mamba2": "Mamba-2 step", "m3_siso": "Mamba-3 SISO", "m3_mimo": "Mamba-3 MIMO (R=4)"}

rows = []
for r in csv.reader(open(os.path.join(RES, "kernel_timing.csv"))):
    if not r or r[0] == "variant":
        continue
    v, impl, R, th, L, us, sd, gf, gbs, ai = r
    base = "mamba2" if v.startswith("mamba2") else v.split("_ggml")[0]
    rows.append(dict(variant=base, src=v, impl=impl, R=int(R), threads=int(th), layers=int(L), us=float(us),
                     sd=float(sd), gflops=float(gf), gbs=float(gbs), ai=float(ai)))

# ---- summary: our AVX2 kernels, MIMO/SISO ratio per regime
summ = []
for th in (1, 8):
    for L in (1, 24):
        pick = {x["variant"]: x for x in rows if x["impl"] == "avx2" and x["threads"] == th and x["layers"] == L}
        gg = {x["src"]: x for x in rows if x["impl"] == "ggml" and x["threads"] == th and x["layers"] == L}
        if {"m3_siso", "m3_mimo", "mamba2"} <= pick.keys():
            s, m, m2 = pick["m3_siso"], pick["m3_mimo"], pick["mamba2"]
            summ.append(dict(threads=th, layers=L, regime="cache-hot (1 layer, 2 MB)" if L == 1 else "DRAM-cold (24 layers, 48 MB)",
                             mamba2_us=m2["us"], siso_us=s["us"], mimo_us=m["us"], mimo_over_siso=m["us"] / s["us"],
                             siso_over_mamba2=s["us"] / m2["us"], flop_ratio=m["gflops"] * m["us"] / (s["gflops"] * s["us"]),
                             siso_GBs=s["gbs"], mimo_GBs=m["gbs"], mimo_gflops=m["gflops"],
                             ggml_ssm_scan_us=gg.get("mamba2_ggml_ssm_scan+cpy", {}).get("us", float("nan"))))
with open(os.path.join(RES, "summary.csv"), "w", newline="") as f:
    if summ:
        w = csv.DictWriter(f, fieldnames=list(summ[0].keys())); w.writeheader(); w.writerows(summ)
for s in summ:
    print(f"t={s['threads']} {s['regime']:30s} M2 {s['mamba2_us']:7.1f}us  SISO {s['siso_us']:7.1f}us  MIMO {s['mimo_us']:7.1f}us"
          f"  MIMO/SISO {s['mimo_over_siso']:.2f}x (FLOP ratio {s['flop_ratio']:.2f}x)  ggml SSM_SCAN {s['ggml_ssm_scan_us']:.1f}us")

# ---- plot 1: grouped bars, time per layer-step
fig, axs = plt.subplots(1, 2, figsize=(10, 4), sharey=False)
for ax, th in zip(axs, (1, 8)):
    regs = [(1, "cache-hot\n1 layer (2 MB)"), (24, "DRAM-cold\n24 layers (48 MB)")]
    w = 0.26
    for i, v in enumerate(["mamba2", "m3_siso", "m3_mimo"]):
        ys, es = [], []
        for L, _ in regs:
            x = [r for r in rows if r["variant"] == v and r["impl"] == "avx2" and r["threads"] == th and r["layers"] == L]
            ys.append(x[0]["us"] if x else 0); es.append(x[0]["sd"] if x else 0)
        bars = ax.bar([j + (i - 1) * w for j in range(len(regs))], ys, w, yerr=es, capsize=3, color=COL[v],
                      label=LBL[v], edgecolor=SURF, linewidth=1.5)
        for b, y in zip(bars, ys):
            ax.text(b.get_x() + b.get_width() / 2, y, f"{y:.0f}", ha="center", va="bottom", fontsize=7.5, color=INK2)
    ax.set_xticks(range(len(regs))); ax.set_xticklabels([r[1] for r in regs])
    ax.set_ylabel("µs per layer decode step (mean ± sd, 3 reps)"); ax.grid(axis="x", visible=False)
    ax.set_title(f"{th} thread{'s' if th > 1 else ''}", loc="left", fontweight="bold", fontsize=11)
axs[0].legend(fontsize=8)
fig.suptitle("Decode-step cost at equal state size (64 heads × 64 × 128, fp32), AVX2 kernels", x=0.01, ha="left",
             fontweight="bold", fontsize=12)
fig.tight_layout(); fig.savefig(os.path.join(PLOTS, "bonus_step_time.png"), dpi=180); plt.close(fig)

# ---- plot 2: roofline (single thread + 8 threads ceilings from STREAM / kernel peaks)
stream = defaultdict(list)
sp = os.path.join(REPO, "results", "hw", "stream.csv")
if os.path.exists(sp):
    for r in csv.DictReader(open(sp)):
        stream[r["threads"]].append(float(r["triad_MBs"]) / 1e3)
fig, ax = plt.subplots(figsize=(7.5, 4.6))
import numpy as np
xs = np.logspace(-1.3, 1.6, 100)
for th, ls in (("1", ":"), ("8", "-")):
    if stream.get(th):
        bw = sum(stream[th]) / len(stream[th])
        peak = 32 * 3.0 * int(th)          # Zen 2: 2 FMA pipes x 8 fp32 lanes x 2 flop = 32 flop/cycle/core; ~3.0 GHz all-core
        ax.plot(xs, np.minimum(peak, bw * xs), color=MUTED, linestyle=ls, linewidth=1.5,
                label=f"{th} thr roof: STREAM triad {bw:.0f} GB/s (read+write), FP32 ≈{peak:.0f} GFLOP/s")
for r in rows:
    if r["impl"] != "avx2":
        continue
    ax.scatter(r["ai"], r["gflops"], s=70 if r["layers"] == 24 else 40, color=COL[r["variant"]],
               marker="o" if r["threads"] == 8 else "^", edgecolor=SURF, linewidth=1.5, zorder=3,
               alpha=1.0 if r["layers"] == 24 else 0.55)
for v in COL:
    ax.scatter([], [], color=COL[v], label=LBL[v])
ax.scatter([], [], marker="o", color=MUTED, label="8 threads"); ax.scatter([], [], marker="^", color=MUTED, label="1 thread")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("arithmetic intensity (FLOP per byte of state read+written)"); ax.set_ylabel("achieved GFLOP/s")
ax.set_title("Where each decode step sits on the roofline", loc="left", fontweight="bold", fontsize=12, pad=20)
ax.text(0, 1.015, "solid markers = DRAM-cold (24 layers), faded = cache-hot (1 layer; above the DRAM roof = served from cache)",
        transform=ax.transAxes, fontsize=8, color=INK2)
ax.legend(fontsize=7.5, loc="lower right")
fig.tight_layout(); fig.savefig(os.path.join(PLOTS, "bonus_roofline.png"), dpi=180); plt.close(fig)

# ---- plot 3: correctness over 1000 steps
fig, ax = plt.subplots(figsize=(7.5, 3.8))
for tr, c in (("siso", COL["m3_siso"]), ("mimo", COL["m3_mimo"])):
    p = os.path.join(RES, f"check_errors_{tr}.csv")
    if not os.path.exists(p):
        continue
    d = list(csv.DictReader(open(p)))
    st = [int(x["step"]) for x in d]
    ax.plot(st, [float(x["relL2_avx2"]) for x in d], color=c, linewidth=1.2, label=f"{tr.upper()} AVX2 kernel vs float64 ref")
    ax.plot(st, [float(x["relL2_torch32"]) for x in d], color=c, linewidth=1.0, linestyle=":", label=f"{tr.upper()} PyTorch fp32 vs float64 (floor)")
ax.axhline(1e-4, color="#d03b3b", linewidth=1.2, linestyle="--"); ax.text(5, 1.15e-4, "pass bar 1e-4 (fixed before running)", fontsize=8, color=INK2)
ax.set_yscale("log"); ax.set_xlabel("decode step (real wikitext tokens, layer 12 of the 1.5B model)")
ax.set_ylabel("per-step relative L2 error")
ax.set_title("Kernel correctness over 1000 decode steps", loc="left", fontweight="bold", fontsize=12, pad=20)
ax.text(0, 1.015, "solid (our AVX2 kernel) and dotted (PyTorch fp32) lines coincide: the kernel's error IS float32 rounding",
        transform=ax.transAxes, fontsize=8, color=INK2)
ax.set_ylim(5e-8, 3e-4)
ax.legend(fontsize=7.5, loc="lower right", ncol=2)
fig.tight_layout(); fig.savefig(os.path.join(PLOTS, "bonus_errors.png"), dpi=180); plt.close(fig)
print("bonus plots written")
