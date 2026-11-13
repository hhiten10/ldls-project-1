// Small ggml driver for the bonus:
//  (1) times ggml's own SSM_SCAN (Mamba-2 decode, 1 token) at the SAME state shape as the Mamba-3 layer
//      (64 heads x headdim 64 x d_state 128, ngroups 1) - the baseline the bonus asks for;
//  (2) checks our hand-written Mamba-2 AVX2 step against ggml's SSM_SCAN over 1000 steps;
//  (3) stretch: wraps our Mamba-3 AVX2 step as a ggml custom operator (ggml_custom_4d), runs it inside a
//      ggml graph, checks it bit-for-bit against the direct kernel call, and times it.
// Graphs contain L independent layers (own state each); L=24 -> 48 MB of state = DRAM-cold like a real model.
//
// usage: ggml_driver <m3_trace_dir> <threads> <layers> <steps>
#include "../kernel/m3.h"
#include "ggml.h"
#include "ggml-cpu.h"
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <random>
#include <string>
#include <vector>

static double now_s() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

// ---- trace loading (same format as m3bench)
struct Trace { M3Params p; int T = 0; std::vector<float> proj; };
static Trace load_trace(const std::string& d) {
    Trace t; std::map<std::string, double> m; std::ifstream mf(d + "/meta.txt"); std::string k; double v;
    while (mf >> k >> v) m[k] = v;
    M3Params& p = t.p; p.H = m["H"]; p.P = m["P"]; p.N = m["N"]; p.R = m["R"]; p.S = m["S"];
    p.rotate_pairwise = m["pairwise"] != 0; p.A_floor = m["A_floor"]; t.T = m["T"];
    const size_t H = p.H, R = p.R, N = p.N, P = p.P, tot = 2*H + 2*N + 2*H*R*N + 3*H*R*P;
    std::vector<float> all(tot); std::ifstream pf(d + "/params.f32", std::ios::binary); pf.read((char*)all.data(), tot * 4);
    size_t o = 0; auto take = [&](std::vector<float>& dst, size_t n) { dst.assign(all.begin() + o, all.begin() + o + n); o += n; };
    take(p.dt_bias, H); take(p.D, H); take(p.Bn_w, N); take(p.Cn_w, N); take(p.B_bias, H*R*N); take(p.C_bias, H*R*N);
    take(p.mimo_x, H*R*P); take(p.mimo_z, H*R*P); take(p.mimo_o, H*R*P);
    t.proj.resize((size_t)t.T * p.d_in_proj()); std::ifstream jf(d + "/proj.f32", std::ios::binary);
    jf.read((char*)t.proj.data(), t.proj.size() * 4);
    return t;
}

// ---- Mamba-3 as a ggml custom op
struct M3OpCtx { const M3Params* p; M3State* st; };
static void m3_custom_op(ggml_tensor* dst, int ith, int nth, void* ud) {
    const M3OpCtx* c = (const M3OpCtx*)ud;
    const float* proj = (const float*)dst->src[0]->data;
    thread_local M3Shared sh;                     // per-thread prologue (tiny: R*N*2 + S values)
    m3_prologue(*c->p, proj, sh);
    const int H = c->p->H, h0 = H * ith / nth, h1 = H * (ith + 1) / nth;
    m3_step_avx2_heads(*c->p, *c->st, proj, sh, (float*)dst->data, h0, h1);
}

int main(int argc, char** argv) {
    if (argc < 5) { fprintf(stderr, "usage: ggml_driver <m3_trace> <threads> <layers> <steps>\n"); return 2; }
    Trace tr = load_trace(argv[1]); const int nth = atoi(argv[2]), L = atoi(argv[3]), steps = atoi(argv[4]);
    const M3Params& p = tr.p; const int H = p.H, P = p.P, N = p.N, HP = H * P;
    const double state_bytes = (double)HP * N * 4;

    ggml_init_params ip = { (size_t)1 << 30, nullptr, false };
    ggml_context* ctx = ggml_init(ip);

    // ------------------------------------------------ (1)+(2): ggml SSM_SCAN, Mamba-2, 1 token, L layers
    ggml_tensor* x   = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, P, H, 1, 1);
    ggml_tensor* dt  = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, H, 1, 1);
    ggml_tensor* A   = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1, H);
    ggml_tensor* B   = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, N, 1, 1, 1);
    ggml_tensor* C   = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, N, 1, 1, 1);
    ggml_tensor* ids = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);
    std::mt19937 rng(7); std::normal_distribution<float> nd;
    for (int i = 0; i < HP; i++) ((float*)x->data)[i] = nd(rng);
    for (int h = 0; h < H; h++) { ((float*)dt->data)[h] = -3.f + nd(rng); ((float*)A->data)[h] = -std::exp(0.5f * nd(rng)); }
    for (int n = 0; n < N; n++) { ((float*)B->data)[n] = nd(rng); ((float*)C->data)[n] = nd(rng); }
    ((int32_t*)ids->data)[0] = 0;
    std::vector<ggml_tensor*> s(L), out(L);
    ggml_cgraph* g2 = ggml_new_graph_custom(ctx, 4096, false);
    for (int l = 0; l < L; l++) {
        s[l] = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, N, P, H, 1);
        memset(s[l]->data, 0, ggml_nbytes(s[l]));
        out[l] = ggml_ssm_scan(ctx, s[l], x, dt, A, B, C, ids, 1);
        // write the new state back into s[l] (what llama.cpp does with its recurrent cache)
        ggml_tensor* new_s = ggml_view_1d(ctx, out[l], (int64_t)HP * N, (size_t)HP * sizeof(float));
        ggml_build_forward_expand(g2, ggml_cpy(ctx, new_s, ggml_view_1d(ctx, s[l], (int64_t)HP * N, 0)));
    }
    // (2) correctness of our m2_step_avx2 vs ggml on layer 0, 1000 steps (D = 0: ggml applies D*x outside SSM_SCAN)
    {
        std::vector<float> st((size_t)HP * N, 0.f), y(HP), Dz(H, 0.f), dts(H);
        for (int h = 0; h < H; h++) { float v = ((float*)dt->data)[h]; dts[h] = v > 20.f ? v : std::log1p(std::exp(v)); }
        M2Inputs in{(float*)x->data, dts.data(), (float*)B->data, (float*)C->data};
        double worst = 0;
        for (int i = 0; i < 1000; i++) {
            ggml_graph_compute_with_ctx(ctx, g2, nth);
            m2_step_avx2(H, P, N, (float*)A->data, Dz.data(), st.data(), in, y.data(), 0, H);
            const float* gy = (const float*)out[0]->data;
            double e = 0, nr = 0;
            for (int j = 0; j < HP; j++) { e += (y[j] - gy[j]) * (y[j] - gy[j]); nr += (double)gy[j] * gy[j]; }
            worst = std::max(worst, std::sqrt(e / nr));
        }
        printf("check,m2_avx2_vs_ggml_ssm_scan,1000 steps,worst_relL2=%.2e,%s\n", worst, worst < 1e-5 ? "PASS" : "FAIL");
    }
    auto bench = [&](ggml_cgraph* g, const char* name, int R, double flops) {
        for (int i = 0; i < 20; i++) ggml_graph_compute_with_ctx(ctx, g, nth);
        double reps[3];
        for (int r = 0; r < 3; r++) {
            const double t0 = now_s();
            for (int i = 0; i < steps; i++) ggml_graph_compute_with_ctx(ctx, g, nth);
            reps[r] = (now_s() - t0) / steps / L;               // per layer-step
        }
        const double mean = (reps[0] + reps[1] + reps[2]) / 3;
        double sd = 0; for (double v : reps) sd += (v - mean) * (v - mean); sd = std::sqrt(sd / 2);
        printf("%s,ggml,%d,%d,%d,%.3f,%.3f,%.2f,%.2f,%.3f\n", name, R, nth, L, mean * 1e6, sd * 1e6,
               flops / mean / 1e9, 2 * state_bytes / mean / 1e9, flops / (2 * state_bytes));
    };
    bench(g2, "mamba2_ggml_ssm_scan+cpy", 1, m2_flops(H, P, N));

    // ------------------------------------------------ (3) Mamba-3 step as ggml custom op
    ggml_tensor* proj = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, p.d_in_proj());
    std::vector<M3State> st(L); std::vector<M3OpCtx> oc(L);
    ggml_cgraph* g3 = ggml_new_graph_custom(ctx, 4096, false);
    std::vector<ggml_tensor*> y3(L);
    for (int l = 0; l < L; l++) {
        st[l].init(p); oc[l] = {&p, &st[l]};
        ggml_tensor* args[1] = {proj};
        y3[l] = ggml_custom_4d(ctx, GGML_TYPE_F32, HP, 1, 1, 1, args, 1, m3_custom_op, GGML_N_TASKS_MAX, &oc[l]);
        ggml_build_forward_expand(g3, y3[l]);
    }
    {   // bit-exactness vs direct kernel call on layer 0 over the trace
        M3State ref; ref.init(p); std::vector<float> yref(HP); double worst = 0;
        for (int i = 0; i < tr.T; i++) {
            memcpy(proj->data, &tr.proj[(size_t)i * p.d_in_proj()], p.d_in_proj() * 4);
            ggml_graph_compute_with_ctx(ctx, g3, nth);
            m3_step_avx2(p, ref, &tr.proj[(size_t)i * p.d_in_proj()], yref.data(), 0, H);
            const float* g = (const float*)y3[0]->data;
            for (int j = 0; j < HP; j++) worst = std::max(worst, (double)std::fabs(g[j] - yref[j]));
        }
        printf("check,m3_custom_op_vs_direct_kernel,%d steps,max_abs_diff=%.2e,%s\n", tr.T, worst, worst == 0 ? "BIT-EXACT" : "DIFF");
    }
    bench(g3, p.R == 1 ? "m3_siso_ggml_custom_op" : "m3_mimo_ggml_custom_op", p.R, m3_flops(p));
    ggml_free(ctx);
    return 0;
}
