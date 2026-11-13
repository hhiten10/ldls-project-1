// m3bench: correctness (vs float64 PyTorch reference, 1000 steps) + timing of decode-step kernels.
//
//   m3bench check <trace_dir>                       -> scalar & AVX2 vs reference, per-step errors CSV
//   m3bench bench <trace_dir> <threads> <layers> <steps> [variant...]
//        layers=1  : "hot"  - one layer's 2 MB state stays cache resident (best case)
//        layers=24 : "cold" - round-robin over 24 layer states (48 MB >> 8 MB L3) = what a real
//                    model decode sees, since every layer's state is evicted between tokens.
//   variants: m3scalar m3avx2 m2scalar m2avx2   (default: m3avx2 m2avx2)
#include "m3.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <random>
#include <string>
#include <omp.h>

template <class T> static std::vector<T> read_bin(const std::string& f, size_t n) {
    std::vector<T> v(n); std::ifstream in(f, std::ios::binary);
    if (!in) { fprintf(stderr, "cannot open %s\n", f.c_str()); exit(1); }
    in.read((char*)v.data(), n * sizeof(T));
    if ((size_t)in.gcount() != n * sizeof(T)) { fprintf(stderr, "short read %s\n", f.c_str()); exit(1); }
    return v;
}

struct Trace { M3Params p; int T = 0; std::vector<float> proj; std::vector<double> y64; std::vector<float> y32; };

static Trace load_trace(const std::string& d) {
    Trace t; std::map<std::string, double> m;
    std::ifstream mf(d + "/meta.txt"); std::string k; double v;
    while (mf >> k >> v) m[k] = v;
    M3Params& p = t.p;
    p.H = m["H"]; p.P = m["P"]; p.N = m["N"]; p.R = m["R"]; p.S = m["S"];
    p.rotate_pairwise = m["pairwise"] != 0; p.A_floor = m["A_floor"]; t.T = m["T"];
    if ((int)m["d_in_proj"] != p.d_in_proj()) { fprintf(stderr, "d_in_proj mismatch\n"); exit(1); }
    const size_t H = p.H, R = p.R, N = p.N, P = p.P;
    size_t tot = 2*H + 2*N + 2*H*R*N + 3*H*R*P;
    auto all = read_bin<float>(d + "/params.f32", tot); size_t o = 0;
    auto take = [&](std::vector<float>& dst, size_t n) { dst.assign(all.begin() + o, all.begin() + o + n); o += n; };
    take(p.dt_bias, H); take(p.D, H); take(p.Bn_w, N); take(p.Cn_w, N);
    take(p.B_bias, H*R*N); take(p.C_bias, H*R*N); take(p.mimo_x, H*R*P); take(p.mimo_z, H*R*P); take(p.mimo_o, H*R*P);
    t.proj = read_bin<float>(d + "/proj.f32", (size_t)t.T * p.d_in_proj());
    t.y64  = read_bin<double>(d + "/y_ref64.f64", (size_t)t.T * H * P);
    t.y32  = read_bin<float>(d + "/y_ref32.f32", (size_t)t.T * H * P);
    return t;
}

static void step_parallel(const M3Params& p, M3State& st, const float* proj, float* y, int nth) {
    static M3Shared sh;   // written by the calling thread, then read-only inside the parallel region
    m3_prologue(p, proj, sh);
    if (nth == 1) { m3_step_avx2_heads(p, st, proj, sh, y, 0, p.H); return; }
    #pragma omp parallel num_threads(nth)
    {
        int t = omp_get_thread_num(), n = omp_get_num_threads();
        int h0 = p.H * t / n, h1 = p.H * (t + 1) / n;
        m3_step_avx2_heads(p, st, proj, sh, y, h0, h1);
    }
}

// --------------------------------------------------------------------------- check
static int cmd_check(const std::string& dir) {
    Trace t = load_trace(dir); const M3Params& p = t.p;
    const int HP = p.H * p.P;
    // Pass bar fixed before running: per-step relative L2 error <= 1e-4 for every one of the T steps.
    const double BAR = 1e-4;
    printf("trace %s: H=%d P=%d N=%d R=%d S=%d pairwise=%d T=%d\n", dir.c_str(), p.H, p.P, p.N, p.R, p.S, p.rotate_pairwise, t.T);
    std::string csv = dir + "/check_errors.csv";
    FILE* f = fopen(csv.c_str(), "w");
    fprintf(f, "step,relL2_scalar,relL2_avx2,relL2_avx2_8t,relL2_torch32,maxabs_avx2\n");
    M3State s1, s2, s3; s1.init(p); s2.init(p); s3.init(p);
    std::vector<float> y1(HP), y2(HP), y3(HP);
    double worst[4] = {0, 0, 0, 0}; double worst_abs = 0, ymax = 0;
    for (int i = 0; i < t.T; i++) {
        const float* pr = &t.proj[(size_t)i * p.d_in_proj()];
        m3_step_scalar(p, s1, pr, y1.data(), 0, p.H);
        m3_step_avx2(p, s2, pr, y2.data(), 0, p.H);
        step_parallel(p, s3, pr, y3.data(), 8);
        const double* r = &t.y64[(size_t)i * HP]; const float* r32 = &t.y32[(size_t)i * HP];
        double e[4] = {0, 0, 0, 0}, nr = 0, ma = 0;
        for (int j = 0; j < HP; j++) {
            nr += r[j] * r[j];
            e[0] += (y1[j] - r[j]) * (y1[j] - r[j]); e[1] += (y2[j] - r[j]) * (y2[j] - r[j]);
            e[2] += (y3[j] - r[j]) * (y3[j] - r[j]); e[3] += (r32[j] - r[j]) * (r32[j] - r[j]);
            ma = std::max(ma, std::fabs(y2[j] - r[j])); ymax = std::max(ymax, std::fabs(r[j]));
        }
        for (int k = 0; k < 4; k++) { e[k] = std::sqrt(e[k] / nr); worst[k] = std::max(worst[k], e[k]); }
        worst_abs = std::max(worst_abs, ma);
        fprintf(f, "%d,%.3e,%.3e,%.3e,%.3e,%.3e\n", i, e[0], e[1], e[2], e[3], ma);
    }
    fclose(f);
    bool ok = worst[0] <= BAR && worst[1] <= BAR && worst[2] <= BAR;
    printf("worst per-step rel-L2 over %d steps: scalar %.2e | avx2 %.2e | avx2 8 threads %.2e | (torch fp32 floor %.2e)\n",
           t.T, worst[0], worst[1], worst[2], worst[3]);
    printf("max |avx2 - ref| = %.2e  (max |ref| = %.2e)   bar %.0e -> %s\n", worst_abs, ymax, BAR, ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}

// --------------------------------------------------------------------------- bench
static double now_s() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

static int cmd_bench(const std::string& dir, int nth, int layers, int steps, std::vector<std::string> vars) {
    Trace t = load_trace(dir); const M3Params& p = t.p;
    const int HP = p.H * p.P;
    if (vars.empty()) vars = {"m3avx2", "m2avx2"};
    std::vector<M3State> st(layers); for (auto& s : st) s.init(p);
    // Mamba-2 at the same state shape (H x P x N), ngroups = 1
    std::vector<std::vector<float>> m2st(layers, std::vector<float>((size_t)HP * p.N, 0.f));
    std::mt19937 rng(1); std::normal_distribution<float> nd;
    std::vector<float> A(p.H), D(p.H), x(HP), dt(p.H), B(p.N), C(p.N);
    for (int h = 0; h < p.H; h++) { A[h] = -std::exp(nd(rng) * 0.5f); D[h] = 1.f; dt[h] = 0.01f + 0.05f * std::fabs(nd(rng)); }
    for (auto& v : x) v = nd(rng);
    for (auto& v : B) v = nd(rng);
    for (auto& v : C) v = nd(rng);
    std::vector<float> y(HP);
    const double state_bytes = (double)HP * p.N * 4;
    for (const auto& var : vars) {
        const bool is_m2 = var.rfind("m2", 0) == 0, scalar = var.find("scalar") != std::string::npos;
        const double flops = is_m2 ? m2_flops(p.H, p.P, p.N) : m3_flops(p);
        M2Inputs in{x.data(), dt.data(), B.data(), C.data()};
        auto one = [&](int i) {
            const int L = i % layers;
            const float* pr = &t.proj[(size_t)(i % t.T) * p.d_in_proj()];
            if (!is_m2) {
                if (scalar) m3_step_scalar(p, st[L], pr, y.data(), 0, p.H);
                else step_parallel(p, st[L], pr, y.data(), nth);
            } else {
                float* s = m2st[L].data();
                if (nth == 1) (scalar ? m2_step_scalar : m2_step_avx2)(p.H, p.P, p.N, A.data(), D.data(), s, in, y.data(), 0, p.H);
                else {
                    #pragma omp parallel num_threads(nth)
                    {
                        int tt = omp_get_thread_num(), n = omp_get_num_threads();
                        (scalar ? m2_step_scalar : m2_step_avx2)(p.H, p.P, p.N, A.data(), D.data(), s, in, y.data(),
                                                                 p.H * tt / n, p.H * (tt + 1) / n);
                    }
                }
            }
        };
        for (int i = 0; i < std::max(layers * 3, 50); i++) one(i);           // warm-up (also faults pages in)
        // 3 repetitions (spec), each = `steps` layer-steps; report per-step time of each rep
        double reps[3];
        for (int r = 0; r < 3; r++) {
            const double t0 = now_s();
            for (int i = 0; i < steps; i++) one(i);
            reps[r] = (now_s() - t0) / steps;
        }
        double mean = (reps[0] + reps[1] + reps[2]) / 3, sd = 0;
        for (double v : reps) sd += (v - mean) * (v - mean);
        sd = std::sqrt(sd / 2);
        const char* variant = is_m2 ? "mamba2" : (p.R == 1 ? "m3_siso" : "m3_mimo");
        // CSV: variant,impl,R,threads,layers,us_per_step,sd_us,gflops,state_GBs(read+write),flop_per_byte
        printf("%s,%s,%d,%d,%d,%.3f,%.3f,%.2f,%.2f,%.3f\n", variant, scalar ? "scalar" : "avx2", is_m2 ? 1 : p.R, nth,
               layers, mean * 1e6, sd * 1e6, flops / mean / 1e9, 2 * state_bytes / mean / 1e9, flops / (2 * state_bytes));
        fflush(stdout);
    }
    return 0;
}

int main(int argc, char** argv) {
    if (argc >= 3 && !strcmp(argv[1], "check")) return cmd_check(argv[2]);
    if (argc >= 6 && !strcmp(argv[1], "bench")) {
        std::vector<std::string> v; for (int i = 6; i < argc; i++) v.push_back(argv[i]);
        return cmd_bench(argv[2], atoi(argv[3]), atoi(argv[4]), atoi(argv[5]), v);
    }
    fprintf(stderr, "usage: m3bench check <trace> | bench <trace> <threads> <layers> <steps> [variants]\n");
    return 2;
}
