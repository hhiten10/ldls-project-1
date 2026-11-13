"""
Explain the roofline gap (spec step 7) for decode at context 0, per model, from three independent sources:
  * per-op wall-clock profile (decode, 256 tokens, 3 reps): where each token's milliseconds go, incl. barriers
  * GGUF byte counts (roofline.csv): what each token must read
  * perf stat counters: IPC and cache-miss ratio, decode vs prefill
Writes results/summary_gap.csv and prints a table.
usage: python -I 10_gap.py <repo>
"""
import csv, glob, os, statistics as st, sys
from collections import defaultdict

REPO = sys.argv[1]; RES = os.path.join(REPO, "results")
TOK = 256
MODELS = {"m2-1.3b-q4": "mamba2-1.3b-Q4_K_M", "m2-2.7b-q4": "mamba2-2.7b-Q4_K_M",
          "m1-1.4b-q4": "mamba-1.4b-Q4_K_M", "llama-1b-q4": "Llama-3.2-1B-Q4_K_M"}
roof = {(r["model"], r["ctx"], r["threads"]): r for r in csv.DictReader(open(os.path.join(RES, "roofline.csv")))}
bw_ro = float(roof[("mamba2-1.3b-Q4_K_M", "0", "8")]["bw_readonly_GBs"])


def perf_stat(tag, phase):
    p = os.path.join(RES, "profile", f"stat_{tag}_{phase}.csv")
    v = {}
    for line in open(p):
        parts = line.strip().split(",")
        if len(parts) > 2 and parts[0].replace(".", "").isdigit():
            v[parts[2].split(":")[0]] = float(parts[0])
    return v


rows = []
for tag, name in MODELS.items():
    per_rep = defaultdict(lambda: defaultdict(float))
    nodes = []
    for f in sorted(glob.glob(os.path.join(RES, "profile", f"ops_{tag}_decode_r*.csv"))):
        rep = f[-5]; n = 0
        for r in csv.DictReader(open(f)):
            op = r["op"]; wall = float(r["wall_ms"]); bar = float(r["barrier_ms_t0"])
            per_rep[rep]["total"] += wall; per_rep[rep]["barrier"] += bar
            key = "mm" if op == "MUL_MAT" else "scan" if op == "SSM_SCAN" else \
                  "attn" if op in ("FLASH_ATTN_EXT", "SOFT_MAX", "ROPE") else "other"
            per_rep[rep][key] += wall
            per_rep[rep][key + "_bar"] += bar
            n += int(r["count"])
        nodes.append(n / TOK)
    m = lambda k: st.mean(per_rep[r][k] for r in per_rep) / TOK            # ms per token
    sd = lambda k: st.pstdev([per_rep[r][k] / TOK for r in per_rep])
    rf = roof[(name, "0", "8")]
    w_mb, s_mb, tot_mb = float(rf["weights_MB"]), float(rf["state_MB"]), float(rf["total_MB"])
    ideal_ms = tot_mb / bw_ro                                                # MB / (GB/s) = ms
    sd_ = perf_stat(tag, "decode"); sp = perf_stat(tag, "prefill512")
    rows.append(dict(
        model=tag, ms_per_token=round(m("total"), 2), ms_sd=round(sd("total"), 2),
        ideal_ms_at_readBW=round(ideal_ms, 2), achieved_frac=round(ideal_ms / m("total"), 3),
        mulmat_ms=round(m("mm"), 2), mulmat_barrier_ms=round(m("mm_bar"), 2),
        mulmat_GBs=round(w_mb / m("mm"), 1), scan_ms=round(m("scan"), 2), attn_ms=round(m("attn"), 2),
        other_ops_ms=round(m("other"), 2), barrier_total_ms=round(m("barrier"), 2),
        barrier_share=round(m("barrier") / m("total"), 3), graph_nodes_per_token=round(st.mean(nodes)),
        weights_MB=w_mb, state_MB=s_mb,
        ipc_decode=round(sd_["instructions"] / sd_["cycles"], 2), ipc_prefill=round(sp["instructions"] / sp["cycles"], 2),
        cache_miss_ratio_decode=round(sd_["cache-misses"] / sd_["cache-references"], 3),
        cache_miss_ratio_prefill=round(sp["cache-misses"] / sp["cache-references"], 3)))

with open(os.path.join(RES, "summary_gap.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
for r in rows:
    print(r)
