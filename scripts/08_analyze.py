"""
Turn raw llama-bench jsonl + run metadata + op profiles into summary CSVs and the five required plots:
  decode vs context | prefill vs prompt length | thread scaling | operator breakdown | predicted vs measured
usage: python -I 08_analyze.py <repo_dir>
"""
import csv, glob, json, os, re, statistics as stats, sys
from collections import defaultdict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = sys.argv[1]
RES, PLOTS = os.path.join(REPO, "results"), os.path.join(REPO, "plots")
os.makedirs(PLOTS, exist_ok=True)

# ---- style: validated categorical palette (dataviz reference instance), fixed entity -> colour
INK, INK2, MUTED, GRID, AXIS, SURF = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
COLOR = {"m2-1.3b": "#2a78d6", "m2-2.7b": "#eb6834", "m1-1.4b": "#1baf7a", "llama-1b": "#eda100"}
NAME = {"m2-1.3b": "Mamba-2 1.3B", "m2-2.7b": "Mamba-2 2.7B", "m1-1.4b": "Mamba-1 1.4B", "llama-1b": "Llama-3.2 1B"}
plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF, "font.size": 10,
    "font.family": ["DejaVu Sans", "sans-serif"],"text.color": INK, "axes.labelcolor": INK2,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.edgecolor": AXIS, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
    "lines.linewidth": 2, "lines.markersize": 7, "legend.frameon": False,
})


def split_tag(tag):           # "m2-1.3b-q4" -> ("m2-1.3b", "q4")
    base, q = tag.rsplit("-", 1)
    return base, q


def style(tag):
    base, q = split_tag(tag)
    return dict(color=COLOR[base], linestyle="-" if q == "q4" else "--", marker="o" if q == "q4" else "s",
                label=f"{NAME[base]} {'Q4_K_M' if q == 'q4' else 'Q8_0'}")


# ---- load raw benchmark runs
meta = {}
mp = os.path.join(RES, "raw", "runs_meta.csv")
if os.path.exists(mp):
    for r in csv.DictReader(open(mp)):
        meta[(r["tag"], r["id"])] = r
rows = []
for f in sorted(glob.glob(os.path.join(RES, "raw", "*.jsonl"))):
    tag, rid = os.path.basename(f)[:-6].split("__")
    if "." in rid:          # archived runs (*.swapped.jsonl = invalidated by swap, *.firstpass.jsonl = superseded)
        continue
    for line in open(f):
        line = line.strip()
        if not line.startswith("{"):
            continue
        j = json.loads(line)
        m = meta.get((tag, rid), {})
        rows.append(dict(tag=tag, id=rid, n_prompt=j["n_prompt"], n_gen=j["n_gen"], n_depth=j.get("n_depth", 0),
                         threads=j["n_threads"], mean_tps=j["avg_ts"], sd_tps=j["stddev_ts"],
                         samples=";".join(f"{x:.3f}" for x in j.get("samples_ts", [])),
                         peak_rss_gb=float(m.get("peak_rss_kb", "nan")) / 1048576,
                         swap_in_pages=m.get("swap_in_pages", ""), swap_out_pages=m.get("swap_out_pages", ""),
                         major_faults=m.get("major_faults", "")))
with open(os.path.join(RES, "summary_runs.csv"), "w", newline="") as fh:
    if rows:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
print(f"{len(rows)} benchmark rows")
by = defaultdict(list)
for r in rows:
    by[r["tag"]].append(r)
tags = [t for t in ["m2-1.3b-q4", "m2-1.3b-q8", "m2-2.7b-q4", "m1-1.4b-q4", "m1-1.4b-q8", "llama-1b-q4"] if t in by]


def finish(ax, fig, name, title, sub=None, legend_out=True):
    ax.set_title(title, loc="left", fontsize=12, fontweight="bold", color=INK, pad=18 if sub else 8)
    if sub:
        ax.text(0, 1.02, sub, transform=ax.transAxes, fontsize=9, color=INK2)
    if legend_out:   # line charts: legend outside the plot area so it never covers data
        ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=8.5)
    else:
        ax.legend(loc="upper right", fontsize=8.5)
    fig.tight_layout(); fig.savefig(os.path.join(PLOTS, name), dpi=180); plt.close(fig)
    print("plot", name)


# ---- 1. decode vs context
fig, ax = plt.subplots(figsize=(9.2, 4.2))
for t in tags:
    pts = sorted((r["n_depth"], r["mean_tps"], r["sd_tps"]) for r in by[t] if r["n_gen"] > 0 and r["id"].startswith("tg128_d"))
    if pts:
        x, y, e = zip(*pts)
        ax.errorbar([max(v, 256) for v in x], y, yerr=e, capsize=3, **style(t))
ax.set_xscale("log", base=2); ax.set_xlabel("context already in cache (tokens; 0 plotted at 256)")
ax.set_ylabel("decode speed (tokens/s, mean ± sd of 3)"); ax.set_ylim(bottom=0)
finish(ax, fig, "decode_vs_context.png", "Decode speed vs context length",
       "tg128 after -d tokens of prefilled context, 8 threads, one per physical core (CPUs 0,2,…,14)")

# ---- 2. prefill vs prompt length
fig, ax = plt.subplots(figsize=(9.2, 4.2))
for t in tags:
    pts = sorted((r["n_prompt"], r["mean_tps"], r["sd_tps"]) for r in by[t] if r["n_gen"] == 0 and r["id"].startswith("pp") and "_t" not in r["id"])
    if pts:
        x, y, e = zip(*pts); ax.errorbar(x, y, yerr=e, capsize=3, **style(t))
ax.set_xscale("log", base=2); ax.set_xlabel("prompt length (tokens)"); ax.set_ylabel("prefill speed (tokens/s)")
ax.set_ylim(bottom=0)
finish(ax, fig, "prefill_vs_prompt.png", "Prefill speed vs prompt length", "ubatch 512, 8 threads")

# ---- 3. thread scaling (two panels: one measure each, no dual axis)
fig, axs = plt.subplots(1, 2, figsize=(10, 4))
for ax, kind in zip(axs, ["pp512", "tg128"]):
    for t in tags:
        pts = sorted((r["threads"], r["mean_tps"], r["sd_tps"]) for r in by[t] if re.fullmatch(kind + r"_t\d+", r["id"]))
        if pts:
            x, y, e = zip(*pts); ax.errorbar(x, y, yerr=e, capsize=3, **style(t))
    ax.set_xlabel("threads (one per physical core)"); ax.set_ylim(bottom=0)
    ax.set_ylabel("tokens/s")
    ax.set_title("Prefill pp512" if kind == "pp512" else "Decode tg128", loc="left", fontweight="bold", fontsize=11)
h, l = axs[0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=3, fontsize=8.5, bbox_to_anchor=(0.5, -0.01))
fig.suptitle("Thread scaling", x=0.01, ha="left", fontweight="bold", fontsize=12)
fig.tight_layout(rect=(0, 0.1, 1, 1)); fig.savefig(os.path.join(PLOTS, "thread_scaling.png"), dpi=180); plt.close(fig); print("plot thread_scaling.png")

# ---- 3b. threads x context (decode) and threads x prompt (prefill): the cross sweep
def cell(tag, th, d):
    rid = f"tg128_d{d}" if th == 8 else f"tg128_t{th}_d{d}"
    x = [r for r in by.get(tag, []) if r["id"] == rid]
    return (x[0]["mean_tps"], x[0]["sd_tps"]) if x else None
cross = []
for t in tags:
    for th in (1, 4, 8):
        for d in (1024, 8192):
            c = cell(t, th, d)
            if c:
                cross.append(dict(tag=t, kind="decode tg128", threads=th, length=d, mean_tps=round(c[0], 3), sd_tps=round(c[1], 3)))
    for th in (1, 4, 8):
        rid = "pp4096" if th == 8 else f"pp4096_t{th}"
        x = [r for r in by.get(t, []) if r["id"] == rid]
        if x and (th == 8 and any(r["id"] == "pp4096_t1" for r in by[t]) or th != 8):
            cross.append(dict(tag=t, kind="prefill pp", threads=th, length=4096, mean_tps=round(x[0]["mean_tps"], 3), sd_tps=round(x[0]["sd_tps"], 3)))
if cross:
    with open(os.path.join(RES, "summary_cross.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(cross[0].keys())); w.writeheader(); w.writerows(cross)
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    for ax, d in zip(axs, (1024, 8192)):
        for t in tags:
            pts = [(c["threads"], c["mean_tps"], c["sd_tps"]) for c in cross if c["tag"] == t and c["kind"] == "decode tg128" and c["length"] == d]
            if pts:
                x, y, e = zip(*sorted(pts)); ax.errorbar(x, y, yerr=e, capsize=3, **style(t))
        ax.set_xticks([1, 4, 8]); ax.set_xlabel("threads"); ax.set_ylabel("decode tokens/s"); ax.set_ylim(bottom=0)
        ax.set_title(f"Decode after {d // 1024}k tokens of context", loc="left", fontweight="bold", fontsize=11)
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, fontsize=8.5, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle("Threads × context", x=0.01, ha="left", fontweight="bold", fontsize=12)
    fig.tight_layout(rect=(0, 0.1, 1, 1)); fig.savefig(os.path.join(PLOTS, "threads_x_context.png"), dpi=180); plt.close(fig)
    print("plot threads_x_context.png")

# ---- 4. operator breakdown (from GGML_OP_PROFILE runs; mean over 3 reps)
GROUP = {"MUL_MAT": "MUL_MAT (weights)", "SSM_SCAN": "SSM_SCAN", "SSM_CONV": "SSM_CONV",
         "RMS_NORM": "norms", "NORM": "norms", "GROUP_NORM": "norms", "FLASH_ATTN_EXT": "attention",
         "SOFT_MAX": "attention", "ROPE": "attention", "GET_ROWS": "other", "CPY": "copies/concat",
         "CONCAT": "copies/concat", "SET_ROWS": "copies/concat"}
COLOR_TAGS = ["m2-1.3b-q4", "m2-1.3b-q8", "m2-2.7b-q4", "m1-1.4b-q4", "m1-1.4b-q8", "llama-1b-q4"]
op_rows = []
for f in glob.glob(os.path.join(RES, "profile", "ops_*_r*.csv")):
    stem = os.path.basename(f)[4:-4].rsplit("_r", 1)[0]          # "<tag>_<phase>"
    tag = next(t for t in COLOR_TAGS if stem.startswith(t + "_"))
    phase = stem[len(tag) + 1:]
    for r in csv.DictReader(open(f)):
        op_rows.append(dict(tag=tag, phase=phase, rep=f[-5], op=r["op"], wall_ms=float(r["wall_ms"]),
                            barrier_ms=float(r["barrier_ms_t0"]), count=int(r["count"])))
if op_rows:
    # Llama decode at 8k context = (prefill 8k + decode) - (prefill 8k), per op and per rep (estimate by differencing)
    pre = defaultdict(float); dec = defaultdict(float); cnt = {}
    for r in op_rows:
        if r["tag"].startswith("llama") and r["phase"] in ("prefill8192", "decode_d8192"):
            (pre if r["phase"] == "prefill8192" else dec)[(r["op"], r["rep"])] += r["wall_ms"]
    for (op, rep), v in dec.items():
        op_rows.append(dict(tag="llama-1b-q4", phase="decode@8k", rep=rep, op=op,
                            wall_ms=max(v - pre.get((op, rep), 0.0), 0.0), barrier_ms=0.0, count=0))
    op_rows = [r for r in op_rows if r["phase"] not in ("prefill8192", "decode_d8192")]
    agg = defaultdict(lambda: defaultdict(list))
    for r in op_rows:
        agg[(r["tag"], r["phase"])][(r["op"], r["rep"])].append(r["wall_ms"])
    table = []
    for (tag, ph), d in agg.items():
        per_op = defaultdict(list)
        reps = sorted({k[1] for k in d})
        for (op, rep), v in d.items():
            per_op[op].append(sum(v))
        tot = [sum(sum(v) for (o, rp), v in d.items() if rp == rep) for rep in reps]
        for op, v in per_op.items():
            table.append(dict(tag=tag, phase=ph, op=op, mean_ms=stats.mean(v), sd_ms=stats.pstdev(v),
                              share=stats.mean(v) / stats.mean(tot)))
    table.sort(key=lambda r: (r["tag"], r["phase"], -r["share"]))
    with open(os.path.join(RES, "summary_ops.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(table[0].keys())); w.writeheader(); w.writerows(table)
    groups = ["MUL_MAT (weights)", "SSM_SCAN", "SSM_CONV", "attention", "norms", "copies/concat", "other"]
    gcol = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#6250d6", "#898781"]
    keys = sorted({(r["tag"], r["phase"]) for r in table}, key=lambda k: (k[0], k[1]))
    fig, ax = plt.subplots(figsize=(9, 0.42 * len(keys) + 1.6))
    left = [0.0] * len(keys)
    for g, c in zip(groups, gcol):
        vals = [100 * sum(r["share"] for r in table if (r["tag"], r["phase"]) == k and GROUP.get(r["op"], "other") == g) for k in keys]
        ax.barh(range(len(keys)), vals, left=left, color=c, label=g, edgecolor=SURF, linewidth=1.5, height=0.7)
        for i, (v, l0) in enumerate(zip(vals, left)):
            if v >= 7:
                ax.text(l0 + v / 2, i, f"{v:.0f}%", ha="center", va="center", fontsize=7.5,
                        color="#0b0b0b" if c == "#eda100" else "#ffffff")
        left = [a + b for a, b in zip(left, vals)]
    ax.set_yticks(range(len(keys))); ax.set_yticklabels([f"{NAME[split_tag(t)[0]]} · {p}" for t, p in keys], fontsize=8.5)
    ax.invert_yaxis(); ax.set_xlim(0, 100); ax.set_xlabel("share of wall time (%)"); ax.grid(axis="y", visible=False)
    ax.legend(ncol=4, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.18))
    ax.set_title("Operator time breakdown (wall clock, incl. barrier waits)", loc="left", fontweight="bold", fontsize=12)
    fig.tight_layout(); fig.savefig(os.path.join(PLOTS, "op_breakdown.png"), dpi=180); plt.close(fig); print("plot op_breakdown.png")

# ---- 5. predicted vs measured decode
rp = os.path.join(RES, "roofline.csv")
if os.path.exists(rp):
    roof = list(csv.DictReader(open(rp)))
    name_to_tag = {"mamba2-1.3b-Q4_K_M": "m2-1.3b-q4", "mamba2-1.3b-Q8_0": "m2-1.3b-q8", "mamba2-2.7b-Q4_K_M": "m2-2.7b-q4",
                   "mamba-1.4b-Q4_K_M": "m1-1.4b-q4", "mamba-1.4b-Q8_0": "m1-1.4b-q8", "Llama-3.2-1B-Q4_K_M": "llama-1b-q4"}
    out = []
    for r in roof:
        tag = name_to_tag.get(r["model"])
        if not tag or r["threads"] != "8":
            continue
        meas = [x for x in by.get(tag, []) if x["id"] == f"tg128_d{r['ctx']}"]
        if meas:
            out.append(dict(tag=tag, ctx=int(r["ctx"]), pred_ro=float(r["pred_tps_readonly"]), pred_triad=float(r["pred_tps_triad"]),
                            measured=meas[0]["mean_tps"], sd=meas[0]["sd_tps"], mb_per_token=float(r["total_MB"]),
                            eff_bw_GBs=meas[0]["mean_tps"] * float(r["total_MB"]) / 1e3,
                            frac_of_ro=meas[0]["mean_tps"] / float(r["pred_tps_readonly"])))
    if out:
        with open(os.path.join(RES, "summary_roofline.csv"), "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(out[0].keys())); w.writeheader(); w.writerows(out)
        d0 = [o for o in out if o["ctx"] == 0]
        fig, ax = plt.subplots(figsize=(8, 4.2))
        xs = range(len(d0)); wd = 0.26
        ax.bar([x - wd for x in xs], [o["pred_ro"] for o in d0], wd, color="#86b6ef", label="predicted: read-only BW ÷ bytes/token", edgecolor=SURF)
        ax.bar(list(xs), [o["pred_triad"] for o in d0], wd, color="#3987e5", label="predicted: STREAM triad ÷ bytes/token", edgecolor=SURF)
        ax.bar([x + wd for x in xs], [o["measured"] for o in d0], wd, yerr=[o["sd"] for o in d0], capsize=3,
               color="#0d366b", label="measured tg128 (mean ± sd)", edgecolor=SURF)
        for x, o in zip(xs, d0):
            ax.text(x + wd, o["measured"] + o["sd"] + 0.4, f"{100*o['frac_of_ro']:.0f}%", ha="center", va="bottom",
                    fontsize=8, color=INK2)
        ax.set_xticks(list(xs)); ax.set_xticklabels([style(o["tag"])["label"].replace(" ", "\n", 1) for o in d0], fontsize=8)
        ax.set_ylabel("decode tokens/s (8 threads)"); ax.grid(axis="x", visible=False)
        finish(ax, fig, "predicted_vs_measured.png", "Roofline: predicted vs measured decode",
               "label = measured as % of read-only-bandwidth prediction (context 0)", legend_out=False)
