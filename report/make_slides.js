// Builds report/slides.pptx (10-minute talk) from the result CSVs, with native PowerPoint charts.
// usage:  NODE_PATH=<dir with node_modules/pptxgenjs> APPLY_THEME=<path to apply_theme.js> node report/make_slides.js
//         (pptxgenjs 3.12.0; APPLY_THEME optional - without it theme colours fall back to Office defaults)
const fs = require("fs");
const path = require("path");
const pptxgen = require("pptxgenjs");

const REPO = path.resolve(__dirname, "..");
const R = (p) => path.join(REPO, "results", p);

// ------------------------------------------------------------------ data from CSVs
function csv(p) {
  const [head, ...lines] = fs.readFileSync(p, "utf8").replace(/^﻿/, "").trim().split(/\r?\n/);
  const cols = head.split(",");
  return lines.map((l) => {
    const v = l.match(/("[^"]*"|[^,]*)(,|$)/g).map((s) => s.replace(/,$/, "").replace(/^"|"$/g, ""));
    return Object.fromEntries(cols.map((c, i) => [c, v[i]]));
  });
}
const runs = csv(R("summary_runs.csv"));
const run = (tag, id) => runs.find((r) => r.tag === tag && r.id === id);
const tps = (tag, id) => { const r = run(tag, id); return r ? +(+r.mean_tps).toFixed(1) : null; };
const ops = csv(R("summary_ops.csv"));
const roof = csv(R("summary_roofline.csv")).filter((r) => r.ctx === "0");
const gap = csv(R("summary_gap.csv"));
const bonus = csv(R("bonus/summary.csv"));
const exp = csv(R("experiment/speed.csv"));
const expRoof = csv(R("experiment/roofline_variants.csv")).filter((r) => r.ctx === "0" && r.threads === "8");
const stream = csv(R("hw/stream.csv"));
const ppl = csv(R("correctness/perplexity.csv"));

const NAME = { "m2-1.3b-q4": "Mamba-2 1.3B", "m2-2.7b-q4": "Mamba-2 2.7B", "m1-1.4b-q4": "Mamba-1 1.4B", "llama-1b-q4": "Llama-3.2 1B" };
const HEX = { "m2-1.3b-q4": "2A78D6", "m2-2.7b-q4": "EB6834", "m1-1.4b-q4": "1BAF7A", "llama-1b-q4": "EDA100" };
const MODELS4 = ["m2-1.3b-q4", "m2-2.7b-q4", "m1-1.4b-q4", "llama-1b-q4"];

// ------------------------------------------------------------------ theme
const THEME = {
  name: "Laptop Silicon",
  headFontFace: "Calibri", bodyFontFace: "Calibri",
  colors: {
    dk1: "15191E", lt1: "FFFFFF", dk2: "1E2A36", lt2: "EEF2F5",
    accent1: "2A78D6", accent2: "EB6834", accent3: "1BAF7A", accent4: "EDA100", accent5: "6250D6", accent6: "E34948",
    hlink: "2A78D6", folHlink: "6250D6",
  },
};
const INK2 = "52514E", MUTED = "898781", GRID = "E1E0D9";

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE";               // 13.33 x 7.5 in
pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
pres.title = "Mamba-2 on a Laptop CPU";
pres.author = "Project 1 team";
const C = pres.SchemeColor;

pres.defineSlideMaster({
  title: "TITLE_DARK",
  background: { color: THEME.colors.dk2 },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: 0.8, y: 2.1, w: 11.7, h: 1.6, fontSize: 44, bold: true, color: C.background1, valign: "bottom", align: "left" }, text: "" } },
    { placeholder: { options: { name: "body", type: "body", x: 0.8, y: 3.85, w: 11.7, h: 1.4, fontSize: 18, color: "C3CEDA", valign: "top" }, text: "" } },
  ],
});
pres.defineSlideMaster({
  title: "CLOSING_DARK",
  background: { color: THEME.colors.dk2 },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: 0.8, y: 0.6, w: 11.7, h: 1.2, fontSize: 40, bold: true, color: C.background1, valign: "middle", align: "left" }, text: "" } },
  ],
});
pres.defineSlideMaster({
  title: "CONTENT",
  background: { color: THEME.colors.lt1 },
  margin: [0.5, 0.6, 0.6, 0.6],
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: 0.6, y: 0.35, w: 12.1, h: 0.85, fontSize: 32, bold: true, color: C.text2, valign: "middle", align: "left" }, text: "" } },
    { text: { text: "Project 1 · Option 3 (Mamba-2) + Mamba-3 bonus · Ryzen 7 4800H", options: { x: 0.6, y: 7.0, w: 8, h: 0.3, fontSize: 10, color: MUTED } } },
  ],
  slideNumber: { x: 12.2, y: 7.0, w: 0.6, h: 0.3, fontSize: 10, color: MUTED, align: "right" },
});

// helpers ------------------------------------------------------------------
const chartBase = () => ({
  catAxisLabelColor: INK2, valAxisLabelColor: INK2, catAxisLabelFontSize: 12, valAxisLabelFontSize: 12,
  catAxisLabelFontFace: "+mn-lt", valAxisLabelFontFace: "+mn-lt",
  valGridLine: { color: GRID, size: 0.75 }, catGridLine: { style: "none" },
  legendFontSize: 12, legendFontFace: "+mn-lt", legendColor: INK2,
  titleFontFace: "+mn-lt", titleColor: THEME.colors.dk2, titleFontSize: 14,
  valAxisTitleFontFace: "+mn-lt", catAxisTitleFontFace: "+mn-lt", valAxisTitleColor: INK2, catAxisTitleColor: INK2,
  valAxisTitleFontSize: 12, catAxisTitleFontSize: 12,
});
function stat(slide, x, y, w, big, small, color, name) {
  slide.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h: 1.45, rectRadius: 0.12, fill: { color: THEME.colors.lt2 }, line: { color: THEME.colors.lt2 }, objectName: name + " card" });
  slide.addText(big, { x: x + 0.2, y: y + 0.1, w: w - 0.4, h: 0.75, fontSize: 36, bold: true, color: color, margin: 0, isTextBox: true, objectName: name + " value" });
  slide.addText(small, { x: x + 0.2, y: y + 0.85, w: w - 0.4, h: 0.5, fontSize: 13, color: INK2, margin: 0, valign: "top", isTextBox: true, objectName: name + " label" });
}
function note(slide, x, y, w, h, runs, name) {
  slide.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, rectRadius: 0.1, fill: { color: THEME.colors.lt2 }, line: { color: THEME.colors.lt2 }, objectName: name + " panel" });
  slide.addText(runs, { x: x + 0.25, y: y + 0.15, w: w - 0.5, h: h - 0.3, fontSize: 15, color: C.text1, valign: "top", margin: 0, paraSpaceAfter: 6, isTextBox: true, objectName: name + " text" });
}
const B = (t) => ({ text: t, options: { bold: true } });
const T = (t) => ({ text: t });
const BR = (t, o = {}) => ({ text: t, options: { breakLine: true, ...o } });

// ================================================================== 1 title
pres.addSection({ title: "Intro" });
let s = pres.addSlide({ masterName: "TITLE_DARK", sectionTitle: "Intro" });
s.addText("Mamba-2 on a Laptop CPU", { placeholder: "title" });
s.addText("Why it runs at the speed it does — and a Mamba-3 MIMO decode step on CPU\nOption 3 + bonus · llama.cpp CPU-only · AMD Ryzen 7 4800H (Zen 2, AVX2, DDR4-3200)", { placeholder: "body" });
s.addNotes("We studied Mamba-2 on an ordinary 8-core laptop CPU with llama.cpp, compared it with Mamba-1 and a dense Llama-3.2-1B, explained the speed with a roofline and profiling, and for the bonus wrote a CPU implementation of a Mamba-3 decode step (none is publicly known), SISO and MIMO, to test whether MIMO pays off on a CPU.");

// ================================================================== 2 setup
pres.addSection({ title: "Setup" });
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Setup" });
s.addText("The machine: memory bandwidth is the budget", { placeholder: "title" });
const ro8 = stream.filter((r) => r.threads === "8").reduce((a, r) => a + +r.readonly_MBs, 0) / 3 / 1000;
stat(s, 0.6, 1.5, 3.8, ro8.toFixed(1) + " GB/s", "measured read bandwidth, 8 threads (60 % of 51.2 peak)", C.accent1, "bandwidth");
stat(s, 4.75, 1.5, 3.8, "3–4 cores", "already saturate the memory bus", C.accent2, "saturation");
stat(s, 8.9, 1.5, 3.8, "3.53 GHz", "mean clock under load, on AC, no throttling", C.accent3, "clock");
const thr = ["1", "2", "3", "4", "5", "6", "7", "8"];
const roByT = thr.map((t) => +(stream.filter((r) => r.threads === t).reduce((a, r) => a + +r.readonly_MBs, 0) / 3 / 1000).toFixed(1));
s.addChart(pres.charts.LINE, [{ name: "read-only bandwidth (GB/s)", labels: thr, values: roByT }], {
  x: 0.6, y: 3.2, w: 6.4, h: 3.6, ...chartBase(), chartColors: ["2A78D6"], lineSize: 2.5, lineDataSymbolSize: 8,
  showTitle: true, title: "Read bandwidth vs threads (GB/s)", showLegend: false, catAxisTitle: "threads (one per physical core)", showCatAxisTitle: true,
  showValue: true, dataLabelPosition: "t", dataLabelFontSize: 11, dataLabelColor: INK2, valAxisMinVal: 0, objectName: "bandwidth chart",
});
note(s, 7.35, 3.2, 5.35, 3.6, [
  BR("Rules we followed", { bold: true, fontSize: 16 }),
  BR("Q4_K_M everywhere, Q8_0 only ≤ 1.5 B, no F16"),
  BR("Mamba converted ourselves from state-spaces checkpoints"),
  BR("Threads pinned one per physical core; 3 reps, mean ± sd"),
  BR("Peak RSS and swap logged for every run"),
  T("Correctness: wikitext PPL — Mamba-2 1.3B " + (+ppl.find((r) => r.model === "mamba2-1.3b-Q4_K_M").ppl).toFixed(2) + " (Q4) / " + (+ppl.find((r) => r.model === "mamba2-1.3b-Q8_0").ppl).toFixed(2) + " (Q8)"),
], "rules");
s.addNotes("Setup. STREAM-style tests show 30.5 GB/s of read bandwidth and that three to four cores already saturate it, which will matter for decode. We followed the quantization and context rules, pinned threads to physical cores, repeated everything three times, and logged peak memory and swap for every run. Perplexity and sample outputs confirm the converted models work.");

// ================================================================== 3 decode vs context
pres.addSection({ title: "Results" });
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Results" });
s.addText("Mamba decode speed stays flat from 0 to 32 k tokens", { placeholder: "title" });
const ctxs = ["0", "1024", "8192", "16384", "32768"];
const ctxLab = ["0", "1 k", "8 k", "16 k", "32 k"];
s.addChart(pres.charts.LINE, MODELS4.map((m) => ({ name: NAME[m], labels: ctxLab, values: ctxs.map((c) => tps(m, "tg128_d" + c)) })), {
  x: 0.6, y: 1.4, w: 8.1, h: 5.4, ...chartBase(), chartColors: MODELS4.map((m) => HEX[m]), lineSize: 2.5, lineDataSymbolSize: 9,
  displayBlanksAs: "span", showLegend: true, legendPos: "b", showTitle: true, title: "Decode tokens/s vs context already in cache (8 threads)",
  valAxisMinVal: 0, catAxisTitle: "context length (categories, not to scale)", showCatAxisTitle: true, objectName: "decode vs context chart",
});
stat(s, 9.05, 1.4, 3.65, "18.2 → 18.1", "Mamba-2 1.3B tok/s at 0 → 32 k context", C.accent1, "mamba flat");
stat(s, 9.05, 3.05, 3.65, "34.4 → 3.7", "Llama-3.2 1B tok/s at 0 → 16 k (−89 %; tight 0-ctx run)", C.accent4, "llama drop");
note(s, 9.05, 4.7, 3.65, 2.1, [BR("Memory too:", { bold: true }), T("Mamba peak RSS flat (1.7 GB); Llama grows 1.3 → 2.3 GiB with its KV cache.")], "memory");
s.addNotes("The headline result: a recurrent model's state does not grow, so decode speed and memory stay flat out to 32 thousand tokens for all three Mamba models. Llama starts faster but loses about 89 percent of its speed by 16 k. Note Mamba-1 is faster than Mamba-2 at similar size; we explain that on the roofline slide.");

// ================================================================== 4 threads
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Results" });
s.addText("Prefill scales with cores; decode stops at 3–4 threads", { placeholder: "title" });
const thrSeries = (kind) => MODELS4.map((m) => ({ name: NAME[m], labels: thr, values: thr.map((t) => tps(m, `${kind}_t${t}`)) }));
s.addChart(pres.charts.LINE, thrSeries("pp512"), {
  x: 0.6, y: 1.4, w: 6.0, h: 5.4, ...chartBase(), chartColors: MODELS4.map((m) => HEX[m]), lineSize: 2.5, lineDataSymbolSize: 8,
  showTitle: true, title: "Prefill pp512 — compute-bound (tok/s)", showLegend: true, legendPos: "b", valAxisMinVal: 0,
  catAxisTitle: "threads", showCatAxisTitle: true, objectName: "prefill threads chart",
});
s.addChart(pres.charts.LINE, thrSeries("tg128"), {
  x: 6.75, y: 1.4, w: 6.0, h: 5.4, ...chartBase(), chartColors: MODELS4.map((m) => HEX[m]), lineSize: 2.5, lineDataSymbolSize: 8,
  showTitle: true, title: "Decode tg128 — bandwidth-bound (tok/s)", showLegend: true, legendPos: "b", valAxisMinVal: 0,
  catAxisTitle: "threads", showCatAxisTitle: true, objectName: "decode threads chart",
});
s.addNotes("Thread scaling separates the two regimes. Prefill keeps scaling with cores, Mamba-2 1.3B goes from 31 to 132 tokens per second, four times faster on eight cores: it is compute-bound. Decode flattens at three to four threads, exactly where the memory bus saturated on the previous slide. Decode is memory-bound.");

// ================================================================== 4b threads x context
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Results" });
s.addText("Threads × context: only attention keeps wanting cores", { placeholder: "title" });
const cross = csv(R("summary_cross.csv"));
const cv = (tag, th, d) => { const r = cross.find((c) => c.tag === tag && c.kind === "decode tg128" && c.threads === String(th) && c.length === String(d)); return r ? +(+r.mean_tps).toFixed(1) : null; };
for (const [i, d] of [[0, 1024], [1, 8192]].values()) {
  s.addChart(pres.charts.LINE, MODELS4.map((m) => ({ name: NAME[m], labels: ["1", "4", "8"], values: [1, 4, 8].map((t) => cv(m, t, d)) })), {
    x: 0.6 + i * 6.15, y: 1.4, w: 6.0, h: 4.4, ...chartBase(), chartColors: MODELS4.map((m) => HEX[m]), lineSize: 2.5, lineDataSymbolSize: 9,
    showTitle: true, title: `Decode tok/s after ${d / 1024}k tokens of context`, showLegend: true, legendPos: "b", valAxisMinVal: 0,
    catAxisTitle: "threads", showCatAxisTitle: true, objectName: `cross chart ${d}`,
  });
}
note(s, 0.6, 5.95, 12.1, 0.9, [
  B("Mamba: flat in context at every thread count, saturated by 4 threads (all 5 model/quant combos). "),
  T("Llama @ 8k: 2.1 → 5.1 → 8.3 tok/s — attention over the KV cache is compute-bound, so it keeps scaling."),
], "cross note");
s.addNotes("Step five asks for the cross of context, threads and quantization. For every Mamba model and quantization, context makes no difference at any thread count and decode saturates by four threads. The one decode case that keeps scaling with cores is Llama at 8 k context, because attention over the cache is compute-bound rather than memory-bound.");

// ================================================================== 5 op breakdown
pres.addSection({ title: "Profiling" });
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Profiling" });
s.addText("Where the time goes: weights first, then the scan", { placeholder: "title" });
const GROUP = { MUL_MAT: "Weight GEMV/GEMM", SSM_SCAN: "SSM scan", FLASH_ATTN_EXT: "Attention" };
const bars = [["m2-1.3b-q4", "decode", "Mamba-2 decode"], ["m2-1.3b-q4", "prefill4096", "Mamba-2 prefill"], ["m1-1.4b-q4", "decode", "Mamba-1 decode"],
              ["m1-1.4b-q4", "prefill4096", "Mamba-1 prefill"], ["llama-1b-q4", "decode", "Llama decode"], ["llama-1b-q4", "decode@8k", "Llama decode @8k"]];
const groups = ["Weight GEMV/GEMM", "SSM scan", "Attention", "Other ops + sync"];
const share = (tag, ph, g) => 100 * ops.filter((r) => r.tag === tag && r.phase === ph && (GROUP[r.op] || "Other ops + sync") === g).reduce((a, r) => a + +r.share, 0);
s.addChart(pres.charts.BAR, groups.map((g) => ({ name: g, labels: bars.map((b) => b[2]), values: bars.map((b) => +share(b[0], b[1], g).toFixed(0)) })), {
  x: 0.6, y: 1.4, w: 7.6, h: 5.4, ...chartBase(), barDir: "bar", barGrouping: "percentStacked", chartColors: ["2A78D6", "EB6834", "EDA100", "B4B2A9"],
  showValue: true, dataLabelPosition: "ctr", dataLabelFontSize: 11, dataLabelColor: "FFFFFF", dataLabelFormatCode: '[>=5]0"%";""',
  valAxisLabelFormatCode: "0%",
  showLegend: true, legendPos: "b", showTitle: true, title: "Share of wall time per operator (incl. barrier waits)", catAxisOrientation: "maxMin",
  objectName: "operator breakdown chart",
});
note(s, 8.5, 1.4, 4.2, 5.4, [
  BR("Mamba-2 vs Mamba-1 scan", { bold: true, fontSize: 17 }),
  BR("Prefill: 0.70 vs 3.49 ms/token", { bold: true }),
  BR("Mamba-2 is 5× cheaper: one exp per head, not per state element, and the state stays in cache within a batch."),
  BR("Decode: 8.5 vs 3.5 ms/token", { bold: true }),
  T("Order flips: Mamba-2 streams an 8× larger state from DRAM every token (≈ 24 GB/s — near the roof)."),
], "scan answer");
s.addNotes("We profiled three ways: a per-operator timer we patched into ggml, perf sampling, and hardware counters. Weight matrix-vector products dominate everywhere. The option's question is how Mamba-2's scan compares with Mamba-1's: in prefill Mamba-2's scan is five times cheaper because its decay is one scalar per head; in decode the order flips because Mamba-2 must stream an eight times larger state from memory every token. For Llama at 8 k, attention takes three quarters of the time.");

// ================================================================== 6 roofline
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Profiling" });
s.addText("Roofline: Mamba-2 decodes at 59 % of the bandwidth limit", { placeholder: "title" });
const order = ["llama-1b-q4", "m1-1.4b-q4", "m2-2.7b-q4", "m2-1.3b-q4"];
const rr = (t) => roof.find((r) => r.tag === t);
s.addChart(pres.charts.BAR, [
  { name: "predicted (read BW ÷ bytes/token)", labels: order.map((t) => NAME[t]), values: order.map((t) => +(+rr(t).pred_ro).toFixed(1)) },
  { name: "measured", labels: order.map((t) => NAME[t]), values: order.map((t) => +(+rr(t).measured).toFixed(1)) },
], {
  x: 0.6, y: 1.4, w: 6.6, h: 5.4, ...chartBase(), barDir: "col", barGrouping: "clustered", chartColors: ["9EC5F4", "184F95"], barGapWidthPct: 60,
  showValue: true, dataLabelPosition: "outEnd", dataLabelFontSize: 11, dataLabelColor: INK2, dataLabelFormatCode: "0.0",
  showLegend: true, legendPos: "b", showTitle: true, title: "Decode tok/s, context 0, 8 threads", valAxisMinVal: 0, objectName: "roofline chart",
});
const g = (t) => gap.find((r) => r.model === t);
s.addTable([
  [{ text: "", options: { fill: { color: THEME.colors.lt2 } } }, ...order.map((t) => ({ text: NAME[t].replace(" ", "\n"), options: { bold: true, fill: { color: THEME.colors.lt2 }, align: "center" } }))],
  ["% of prediction", ...order.map((t) => ({ text: Math.round(100 * +rr(t).frac_of_ro) + " %", options: { align: "center" } }))],
  ["weight GEMV GB/s", ...order.map((t) => ({ text: (+g(t).mulmat_GBs).toFixed(1), options: { align: "center" } }))],
  ["state ÷ weights", ...order.map((t) => ({ text: (+g(t).state_MB / +g(t).weights_MB).toFixed(2), options: { align: "center" } }))],
  ["barrier waits", ...order.map((t) => ({ text: Math.round(100 * +g(t).barrier_share) + " %", options: { align: "center" } }))],
  ["IPC decode", ...order.map((t) => ({ text: (+g(t).ipc_decode).toFixed(2), options: { align: "center" } }))],
], { x: 7.45, y: 1.4, w: 5.25, colW: [1.65, 0.9, 0.9, 0.9, 0.9], fontSize: 12, color: THEME.colors.dk1, border: { type: "solid", pt: 0.5, color: GRID }, rowH: 0.42, fontFace: THEME.bodyFontFace, objectName: "gap table" });
note(s, 7.45, 4.25, 5.25, 2.55, [
  BR("The GEMV kernel is fine: Llama's hits 95 % of read BW.", { bold: true }),
  BR("Mamba-2's state traffic (206 MB/token, updated out-of-place) slows the weight stream; ~1150 tiny ops/token add 17 % barrier waits."),
  BR("Low IPC (0.4) = memory-bound, not throttling (3.5 GHz)."),
  T("Llama's 77 % is a bimodal mean (one slow rep in 3); its tight runs reach 90 %."),
], "gap explanation");
s.addNotes("Prediction is bandwidth divided by bytes per token, counted from the GGUF, including the recurrent state. Llama reaches 77 percent on its recorded mean, 90 in its tight runs; Mamba-2 1.3B 59. The weight kernel is not the problem since Llama's reaches 99 percent of bandwidth. The weight stream's effective bandwidth falls exactly as the state-to-weight ratio rises: llama.cpp updates the 2 MB per-layer state out of place, and the dirty data competes with the weights. Mamba also runs four times more graph nodes per token, each ending in a barrier.");

// ================================================================== 7 experiment
pres.addSection({ title: "Experiment" });
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Experiment" });
s.addText("Experiment: change one tensor's quantization (pre-registered)", { placeholder: "title" });
const types = ["q4_0", "q4_K", "q5_K", "q6_K", "q8_0"];
const meas = types.map((t) => +exp.find((r) => r.ssm_in_type === t && r.test === "tg128").mean_tps);
const bytesOf = (t) => +expRoof.find((r) => r.model === "m2-1.3b-ssm_in-" + t).total_MB;
const bwEff = meas[1] * bytesOf("q4_K");
const pred = types.map((t) => +(bwEff / bytesOf(t)).toFixed(2));
s.addChart(pres.charts.BAR, [
  { name: "predicted from bytes", labels: types.map((t) => "ssm_in " + t.toUpperCase()), values: pred },
  { name: "measured tg128", labels: types.map((t) => "ssm_in " + t.toUpperCase()), values: meas.map((v) => +v.toFixed(2)) },
], {
  x: 0.6, y: 1.4, w: 7.4, h: 5.4, ...chartBase(), barDir: "col", barGrouping: "clustered", chartColors: ["9EC5F4", "184F95"], barGapWidthPct: 60,
  showValue: true, dataLabelPosition: "outEnd", dataLabelFontSize: 11, dataLabelColor: INK2, dataLabelFormatCode: "0.0",
  showLegend: true, legendPos: "b", showTitle: true, title: "Mamba-2 1.3B decode tok/s by ssm_in type (55.7 % of bytes)", valAxisMinVal: 0, objectName: "experiment chart",
});
note(s, 8.3, 1.4, 4.4, 5.4, [
  BR("H1  bytes predict decode ±10 %", { bold: true }), BR("PASS (−7.0, −0.8, −1.8, +9.4 %)", { color: "006300" }),
  BR("H2  same bytes ⇒ < 5 % apart", { bold: true }), BR("FAIL: Q4_0 is 7 % slower than Q4_K — the repacked Q4_K kernel streams better", { color: "B03A2E" }),
  BR("H3  quality", { bold: true }), BR("PPL 11.64 → 11.45, but each step < ±0.30 error: not resolved"),
  T("Q4_K is the right choice for ssm_in here."),
], "verdicts");
s.addNotes("Our own experiment changed exactly one thing: the quantization type of ssm_in, which is 56 percent of the bytes, with hypotheses and pass bars written down before running. Bytes predicted decode speed within ten percent, so that passed. Same bytes did not mean same speed: Q4_0 was seven percent slower than Q4_K, which fails our second hypothesis and shows the kernel matters. Quality differences were within the error bars, so we do not claim them.");

// ================================================================== 8 rudra
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Experiment" });
s.addText("Shared reference run on PARAM Rudra (Llama-3.2-1B)", { placeholder: "title" });
stat(s, 0.6, 1.5, 3.8, "298.5 / 300.2", "pp512 tok/s, Rudra jobs 186433 / 186434 (4 cores)", C.accent1, "rudra pp");
stat(s, 4.75, 1.5, 3.8, "39.27 / 39.30", "tg128 tok/s, same jobs", C.accent1, "rudra tg");
stat(s, 8.9, 1.5, 3.8, "152 / 31.7", "laptop pp512 / tg128 at 4 threads", C.accent2, "laptop ref");
const hd = (t) => ({ text: t, options: { bold: true, fill: { color: THEME.colors.lt2 }, align: "center" } });
const cc = (t) => ({ text: t, options: { align: "center" } });
s.addTable([
  [{ text: "Base Q4_K_M, 4 cores, same commit", options: { bold: true, fill: { color: THEME.colors.lt2 } } }, hd("pp512 tok/s"), hd("tg128 tok/s"), hd("peak RSS")],
  ["Rudra job 186433 (Xeon 6240R)", cc("298.5 ± 0.6"), cc("39.27 ± 0.02"), cc("1.32 GiB")],
  ["Rudra job 186434 (Xeon 6240R)", cc("300.2 ± 0.8"), cc("39.30 ± 0.04"), cc("1.32 GiB")],
  ["Laptop (Ryzen 4800H)", cc("152.3 ± 0.7"), cc("31.7 ± 0.9"), cc("1.32 GiB")],
  [{ text: "Rudra ÷ laptop", options: { bold: true } }, cc("2.0×"), cc("1.2×"), cc("")],
], { x: 0.6, y: 3.2, w: 6.4, colW: [2.8, 1.25, 1.25, 1.1], fontSize: 13, color: THEME.colors.dk1, border: { type: "solid", pt: 0.5, color: GRID }, rowH: 0.55, fontFace: THEME.bodyFontFace, objectName: "rudra table" });
note(s, 7.35, 3.2, 5.35, 3.6, [
  BR("QoS limit: 4 CPUs per job", { bold: true }), BR("48-core requests pend (QOSMaxCpuPerUserLimit) → shared node."),
  BR("Prefill 2.0× the laptop", { bold: true }), BR("AVX-512 doubles vector width for compute-bound GEMM."),
  BR("Decode only 1.2×", { bold: true }), T("Both bandwidth-limited. The two base-model jobs agree within 0.6 %."),
], "rudra notes");
s.addNotes("The shared reference test ran on a PARAM Rudra CPU node with the same llama.cpp commit and a byte-identical model. Our account is limited to four CPUs per job, so this is four cores on a shared node. We used the base Llama-3.2-1B, converted on the laptop and copied with an identical checksum. Prefill is 2 times the laptop thanks to AVX-512; decode only 1.2 times because both are bandwidth-limited. The two jobs agree within 0.6 percent; an earlier pair with the Instruct file differed by 3 to 5 percent because of neighbours on the node.");

// ================================================================== 9 bonus build
pres.addSection({ title: "Bonus" });
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Bonus" });
s.addText("Bonus: a Mamba-3 decode step on CPU, on real weights", { placeholder: "title" });
stat(s, 0.6, 1.5, 3.8, "2e-16", "our PyTorch step vs official references (f64)", C.accent3, "ref match");
stat(s, 4.75, 1.5, 3.8, "1.5e-5", "AVX2 SISO / MIMO error over 1000 real steps (= fp32 floor; bar 1e-4)", C.accent3, "kernel match");
stat(s, 8.9, 1.5, 3.8, "12.2 / 11.8", "wikitext PPL, full 1.5B SISO / MIMO decoded on CPU", C.accent3, "ppl");
note(s, 0.6, 3.2, 6.0, 3.6, [
  BR("What we built", { bold: true, fontSize: 17 }),
  BR("Official Mamba3.step is CUDA-only; we assembled a CPU reference from the repo's own PyTorch refs"),
  BR("Pretrained 1.5B SISO + MIMO checkpoints are public → real weights, real hidden states"),
  BR("C++ scalar + AVX2 kernels: SISO and MIMO rank 4, plus Mamba-2 at the same state"),
  T("Stretch: wrapped as a ggml custom op — bit-exact inside a ggml graph"),
], "built");
note(s, 6.9, 3.2, 5.8, 3.6, [
  BR("Mamba-3 update per head", { bold: true, fontSize: 17 }),
  BR("α = exp(AΔt), β = (1−λ)Δtα, γ = λΔt  (trapezoidal)"),
  BR("B, C rotated by a data-dependent angle (complex state)"),
  BR("S ← αS + Σᵣ γ(x·wᵣ)⊗Bᵣ + Σᵣ β(x_prev·wᵣ)⊗B_prev,ᵣ"),
  T("FLOPs per state element: Mamba-2 5, SISO 7, MIMO 25 — same bytes"),
], "update rule");
s.addNotes("No public CPU implementation of Mamba-3 is known to us. The official step runs only on CUDA, so we built a CPU reference from the repo's own PyTorch reference functions and matched them to machine precision. Pretrained checkpoints turned out to be public, so we decoded the full 1.5 billion parameter models token by token on the CPU and got sensible perplexities, which also settled a rotation-layout ambiguity. Our AVX2 kernels match the float64 reference over 1000 real decode steps exactly as well as PyTorch's own float32 does.");

// ================================================================== 10 MIMO answer
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Bonus" });
s.addText("Does MIMO pay off on CPU? Yes, where decode runs", { placeholder: "title" });
const bl = bonus.map((b) => `${b.layers === "1" ? "cache-hot" : "DRAM-cold"}, ${b.threads} thr`);
s.addChart(pres.charts.BAR, [
  { name: "Mamba-2 step", labels: bl, values: bonus.map((b) => +(+b.mamba2_us).toFixed(0)) },
  { name: "Mamba-3 SISO", labels: bl, values: bonus.map((b) => +(+b.siso_us).toFixed(0)) },
  { name: "Mamba-3 MIMO (R=4)", labels: bl, values: bonus.map((b) => +(+b.mimo_us).toFixed(0)) },
], {
  x: 0.6, y: 1.4, w: 7.6, h: 5.4, ...chartBase(), barDir: "col", barGrouping: "clustered", chartColors: ["2A78D6", "EB6834", "1BAF7A"], barGapWidthPct: 50,
  showValue: true, dataLabelPosition: "outEnd", dataLabelFontSize: 11, dataLabelColor: INK2, dataLabelFormatCode: "0",
  showLegend: true, legendPos: "b", showTitle: true, title: "µs per layer decode step, 2 MB fp32 state", valAxisMinVal: 0, objectName: "mimo chart",
});
const cold8 = bonus.find((b) => b.threads === "8" && b.layers === "24"), hot8 = bonus.find((b) => b.threads === "8" && b.layers === "1");
stat(s, 8.5, 1.4, 4.2, "+" + Math.round(100 * (+cold8.mimo_over_siso - 1)) + " %", "MIMO vs SISO per step when the state comes from DRAM (real decode, 8 thr)", C.accent3, "cold ratio");
stat(s, 8.5, 3.0, 4.2, (+hot8.mimo_over_siso).toFixed(1) + "×", "when the state sits in cache: compute-bound", C.accent2, "hot ratio");
note(s, 8.5, 4.6, 4.2, 2.2, [
  BR("3.6× the FLOPs hide in memory stalls", { bold: true }),
  T("IPC cold: SISO 0.67 vs MIMO 1.40. Per token ≈ 3–4 % slower for lower PPL (11.8 vs 12.2)."),
], "mimo answer");
s.addNotes("Here is the answer. MIMO does 3.6 times the arithmetic of SISO on the same state. When the state fits in cache, the step is compute-bound and MIMO costs the full 3 times. But in a real model every layer's state is evicted between tokens, so the step waits on DRAM; then MIMO's extra math hides in the memory stalls and costs only 19 percent per step at 8 threads. Per token that is roughly 3 to 4 percent, for a model with better perplexity. So MIMO pays off on this CPU too, with less headroom than on a GPU.");

// ================================================================== 11 takeaways
pres.addSection({ title: "Close" });
s = pres.addSlide({ masterName: "CLOSING_DARK", sectionTitle: "Close" });
s.addText("What we learned", { placeholder: "title" });
const pts = [
  "Decode speed = bytes per token. Mamba-2's 206 MB state is part of those bytes.",
  "Flat speed and memory to 32 k tokens is real on a laptop CPU.",
  "Mamba-2's scan: 5× cheaper than Mamba-1 in prefill, dearer in decode.",
  "Beyond 3–4 threads decode gains nothing; tiny ops cost ~17 % in barriers.",
  "Mamba-3 MIMO: 3.6× the FLOPs for ~19 % per step — compute is cheap when memory is the wall.",
];
s.addText(pts.map((t, i) => ({ text: t, options: { bullet: true, breakLine: i < pts.length - 1 } })),
  { x: 0.8, y: 2.3, w: 11.7, h: 4.3, fontSize: 22, color: "E6ECF2", paraSpaceAfter: 14, valign: "top", isTextBox: true, objectName: "takeaways" });
s.addNotes("To close: on a laptop CPU, decode speed is bytes per token, and for Mamba-2 the recurrent state is a real part of those bytes. The flat-context promise holds. Mamba-2's scan trades cheaper prefill for dearer decode. Decode stops scaling at three to four threads. And for Mamba-3, MIMO's extra compute is nearly free in the memory-bound regime a real decode runs in. Everything is scripted and reproducible from the repository.");

// ------------------------------------------------------------------ write + theme
(async () => {
  const out = path.join(REPO, "report", "slides.pptx");
  await pres.writeFile({ fileName: out });
  if (process.env.APPLY_THEME) {
    const { applyTheme } = require(process.env.APPLY_THEME);
    await applyTheme(out, THEME);
  }
  console.log("wrote", out);
})();
