// Mamba-3 / Mamba-2 decode-step kernels: scalar reference path + AVX2/FMA path.
#include "m3.h"
#include <cmath>
#include <cstring>
#include <immintrin.h>

static inline float softplus(float x) { return x > 20.f ? x : std::log1p(std::exp(x)); }   // torch threshold=20
static inline float sigmoidf(float x) { return 1.f / (1.f + std::exp(-x)); }
static inline float heavy_tail(float x) { return x >= 0.f ? 1.f + x : 1.f / (1.f - x); }
static constexpr float PI_F = 3.14159265358979323846f;

void M3State::init(const M3Params& p) {
    angle.assign((size_t)p.H * p.S, 0.f);
    ssm.assign((size_t)p.H * p.P * p.N, 0.f);
    k.assign((size_t)p.R * p.H * p.N, 0.f);
    v.assign((size_t)p.H * p.P, 0.f);
}

// ---------------------------------------------------------------------------------- prologue
void m3_prologue(const M3Params& p, const float* proj, M3Shared& sh) {
    const int HP = p.H * p.P, RN = p.R * p.N;
    const float* Bsrc = proj + 2 * HP;
    const float* Csrc = Bsrc + RN;
    const float* ang  = Csrc + RN + 3 * p.H;
    sh.Bn.resize(RN); sh.Cn.resize(RN); sh.tang.resize(p.S);
    for (int r = 0; r < p.R; r++) {
        double sb = 0, sc = 0;
        for (int n = 0; n < p.N; n++) { sb += (double)Bsrc[r*p.N+n]*Bsrc[r*p.N+n]; sc += (double)Csrc[r*p.N+n]*Csrc[r*p.N+n]; }
        const float ib = 1.f / std::sqrt((float)(sb / p.N) + 1e-5f), ic = 1.f / std::sqrt((float)(sc / p.N) + 1e-5f);
        for (int n = 0; n < p.N; n++) {
            sh.Bn[r*p.N+n] = Bsrc[r*p.N+n] * ib * p.Bn_w[n];
            sh.Cn[r*p.N+n] = Csrc[r*p.N+n] * ic * p.Cn_w[n];
        }
    }
    for (int s = 0; s < p.S; s++) sh.tang[s] = std::tanh(ang[s]);
}

// Per-head scalar pieces shared by both kernels: gates + rotated (biased) B and C for this head.
struct HeadCoef { float alpha, beta, gamma; };
static HeadCoef head_coef_and_rotate(const M3Params& p, M3State& st, const float* proj, const M3Shared& sh,
                                     int h, float* kB, float* Cq /* [R][N] each */) {
    const int HP = p.H * p.P, RN = p.R * p.N;
    const float* dd_dt = proj + 2 * HP + 2 * RN;
    const float* dd_A  = dd_dt + p.H;
    const float* trp   = dd_A + p.H;
    const float dt   = softplus(dd_dt[h] + p.dt_bias[h]);
    const float A    = std::fmin(-heavy_tail(dd_A[h]), -p.A_floor);
    const float trap = sigmoidf(trp[h]);
    float* ang = &st.angle[(size_t)h * p.S];
    float cs[256], sn[256];                      // N/2 <= 256
    const int half = p.N / 2;
    for (int s = 0; s < p.S; s++) {
        ang[s] += sh.tang[s] * dt * PI_F;
        cs[s] = std::cos(ang[s]); sn[s] = std::sin(ang[s]);
    }
    for (int s = p.S; s < half; s++) { cs[s] = 1.f; sn[s] = 0.f; }
    for (int r = 0; r < p.R; r++) {
        const float* bb = &p.B_bias[((size_t)h * p.R + r) * p.N];
        const float* cb = &p.C_bias[((size_t)h * p.R + r) * p.N];
        float* kb = kB + r * p.N; float* cq = Cq + r * p.N;
        for (int n = 0; n < p.N; n++) { kb[n] = sh.Bn[r*p.N+n] + bb[n]; cq[n] = sh.Cn[r*p.N+n] + cb[n]; }
        if (p.rotate_pairwise) {
            for (int i = 0; i < p.S; i++) {
                float a0 = kb[2*i], a1 = kb[2*i+1]; kb[2*i] = a0*cs[i] - a1*sn[i]; kb[2*i+1] = a0*sn[i] + a1*cs[i];
                float c0 = cq[2*i], c1 = cq[2*i+1]; cq[2*i] = c0*cs[i] - c1*sn[i]; cq[2*i+1] = c0*sn[i] + c1*cs[i];
            }
        } else {
            for (int i = 0; i < p.S; i++) {      // i >= S: cos=1, sin=0 -> identity, skip
                float a0 = kb[i], a1 = kb[i+half]; kb[i] = a0*cs[i] - a1*sn[i]; kb[i+half] = a0*sn[i] + a1*cs[i];
                float c0 = cq[i], c1 = cq[i+half]; cq[i] = c0*cs[i] - c1*sn[i]; cq[i+half] = c0*sn[i] + c1*cs[i];
            }
        }
    }
    const float alpha = std::exp(A * dt);
    return { alpha, (1.f - trap) * dt * alpha, trap * dt };
}

// ---------------------------------------------------------------------------------- scalar
void m3_step_scalar(const M3Params& p, M3State& st, const float* proj, float* y, int h0, int h1) {
    M3Shared sh; m3_prologue(p, proj, sh);
    const int HP = p.H * p.P, R = p.R, N = p.N, P = p.P;
    const float* z = proj; const float* x = proj + HP;
    std::vector<float> kB(R * N), Cq(R * N);
    for (int h = h0; h < h1; h++) {
        HeadCoef c = head_coef_and_rotate(p, st, proj, sh, h, kB.data(), Cq.data());
        for (int pp = 0; pp < P; pp++) {
            float* row = &st.ssm[((size_t)h * P + pp) * N];
            float xr[16], xs[16], acc[16];
            for (int r = 0; r < R; r++) {
                const float m = p.mimo_x[((size_t)h * R + r) * P + pp];
                xr[r] = x[h*P+pp] * m; xs[r] = st.v[h*P+pp] * m; acc[r] = 0.f;
            }
            for (int n = 0; n < N; n++) {
                float s = row[n] * c.alpha;
                for (int r = 0; r < R; r++) s += c.gamma * xr[r] * kB[r*N+n];
                for (int r = 0; r < R; r++) s += c.beta * xs[r] * st.k[((size_t)r * p.H + h) * N + n];
                row[n] = s;
                for (int r = 0; r < R; r++) acc[r] += s * Cq[r*N+n];
            }
            float out = 0.f;
            for (int r = 0; r < R; r++) {
                float o = acc[r] + p.D[h] * xr[r];
                float zr = z[h*P+pp] * p.mimo_z[((size_t)h * R + r) * P + pp];
                o *= zr * sigmoidf(zr);
                out += o * p.mimo_o[((size_t)h * R + r) * P + pp];
            }
            y[h*P+pp] = out;
        }
        for (int r = 0; r < R; r++) std::memcpy(&st.k[((size_t)r * p.H + h) * N], &kB[r*N], N * sizeof(float));
        std::memcpy(&st.v[h*P], &x[h*P], P * sizeof(float));
    }
}

// ---------------------------------------------------------------------------------- AVX2
static inline float hsum8(__m256 v) {
    __m128 lo = _mm256_castps256_ps128(v), hi = _mm256_extractf128_ps(v, 1);
    lo = _mm_add_ps(lo, hi); lo = _mm_add_ps(lo, _mm_movehl_ps(lo, lo));
    lo = _mm_add_ss(lo, _mm_shuffle_ps(lo, lo, 1)); return _mm_cvtss_f32(lo);
}

// ROWS state rows are processed together so the ROWS*R dot-product chains are independent
// (one chain per row would be FMA-latency bound: 16 dependent FMAs x 5 cycles on Zen 2).
template <int R, int ROWS>
static void m3_head_avx2(const M3Params& p, float* __restrict ssm_h, const float* __restrict kB,
                         const float* __restrict kprev /* [R] rows with stride H*N */, size_t kprev_stride,
                         const float* __restrict Cq, const HeadCoef& c,
                         const float* xr_all, const float* xs_all /* [P][R] */, float* acc_out /* [P][R] */) {
    const int N = p.N, P = p.P;
    const __m256 va = _mm256_set1_ps(c.alpha);
    for (int p0 = 0; p0 < P; p0 += ROWS) {
        __m256 g[ROWS][R], b[ROWS][R], acc[ROWS][R];
        for (int j = 0; j < ROWS; j++)
            for (int r = 0; r < R; r++) {
                g[j][r] = _mm256_set1_ps(c.gamma * xr_all[(p0 + j) * R + r]);
                b[j][r] = _mm256_set1_ps(c.beta * xs_all[(p0 + j) * R + r]);
                acc[j][r] = _mm256_setzero_ps();
            }
        for (int n = 0; n < N; n += 8) {
            __m256 vB[R], vBp[R], vC[R];
            for (int r = 0; r < R; r++) {
                vB[r]  = _mm256_load_ps(kB + r * N + n);
                vBp[r] = _mm256_loadu_ps(kprev + r * kprev_stride + n);
                vC[r]  = _mm256_load_ps(Cq + r * N + n);
            }
            for (int j = 0; j < ROWS; j++) {
                float* rowp = ssm_h + (size_t)(p0 + j) * N + n;
                __m256 s = _mm256_mul_ps(_mm256_loadu_ps(rowp), va);
                for (int r = 0; r < R; r++) s = _mm256_fmadd_ps(g[j][r], vB[r], s);
                for (int r = 0; r < R; r++) s = _mm256_fmadd_ps(b[j][r], vBp[r], s);
                _mm256_storeu_ps(rowp, s);
                for (int r = 0; r < R; r++) acc[j][r] = _mm256_fmadd_ps(s, vC[r], acc[j][r]);
            }
        }
        for (int j = 0; j < ROWS; j++)
            for (int r = 0; r < R; r++) acc_out[(p0 + j) * R + r] = hsum8(acc[j][r]);
    }
}

void m3_step_avx2_heads(const M3Params& p, M3State& st, const float* proj, const M3Shared& sh,
                        float* y, int h0, int h1) {
    const int HP = p.H * p.P, R = p.R, N = p.N, P = p.P;
    const float* z = proj; const float* x = proj + HP;
    alignas(64) float kB[4 * 256], Cq[4 * 256];     // R<=4, N<=256
    alignas(64) float xr[64 * 4], xs[64 * 4], acc[64 * 4];   // P<=64, R<=4
    for (int h = h0; h < h1; h++) {
        HeadCoef c = head_coef_and_rotate(p, st, proj, sh, h, kB, Cq);
        for (int pp = 0; pp < P; pp++)
            for (int r = 0; r < R; r++) {
                const float m = p.mimo_x[((size_t)h * R + r) * P + pp];
                xr[pp * R + r] = x[h*P+pp] * m; xs[pp * R + r] = st.v[h*P+pp] * m;
            }
        float* ssm_h = &st.ssm[(size_t)h * P * N];
        const float* kprev = &st.k[(size_t)h * N];
        const size_t kstride = (size_t)p.H * N;
        if (R == 1)      m3_head_avx2<1, 4>(p, ssm_h, kB, kprev, kstride, Cq, c, xr, xs, acc);
        else if (R == 2) m3_head_avx2<2, 2>(p, ssm_h, kB, kprev, kstride, Cq, c, xr, xs, acc);
        else             m3_head_avx2<4, 2>(p, ssm_h, kB, kprev, kstride, Cq, c, xr, xs, acc);
        for (int pp = 0; pp < P; pp++) {
            float out = 0.f;
            for (int r = 0; r < R; r++) {
                float o = acc[pp * R + r] + p.D[h] * xr[pp * R + r];
                float zr = z[h*P+pp] * p.mimo_z[((size_t)h * R + r) * P + pp];
                o *= zr * sigmoidf(zr);
                out += o * p.mimo_o[((size_t)h * R + r) * P + pp];
            }
            y[h*P+pp] = out;
        }
        for (int r = 0; r < R; r++) std::memcpy(&st.k[((size_t)r * p.H + h) * N], &kB[r*N], N * sizeof(float));
        std::memcpy(&st.v[h*P], &x[h*P], P * sizeof(float));
    }
}

void m3_step_avx2(const M3Params& p, M3State& st, const float* proj, float* y, int h0, int h1) {
    M3Shared sh; m3_prologue(p, proj, sh);
    m3_step_avx2_heads(p, st, proj, sh, y, h0, h1);
}

// ---------------------------------------------------------------------------------- Mamba-2
void m2_step_scalar(int H, int P, int N, const float* A, const float* D, float* state,
                    const M2Inputs& in, float* y, int h0, int h1) {
    for (int h = h0; h < h1; h++) {
        const float dt = in.dt[h], dA = std::exp(dt * A[h]);
        for (int pp = 0; pp < P; pp++) {
            float* row = state + ((size_t)h * P + pp) * N;
            const float xdt = in.x[h*P+pp] * dt;
            float acc = 0.f;
            for (int n = 0; n < N; n++) { float s = row[n] * dA + xdt * in.B[n]; row[n] = s; acc += s * in.C[n]; }
            y[h*P+pp] = acc + D[h] * in.x[h*P+pp];
        }
    }
}

void m2_step_avx2(int H, int P, int N, const float* A, const float* D, float* state,
                  const M2Inputs& in, float* y, int h0, int h1) {
    constexpr int ROWS = 4;
    for (int h = h0; h < h1; h++) {
        const float dt = in.dt[h];
        const __m256 vdA = _mm256_set1_ps(std::exp(dt * A[h]));
        for (int p0 = 0; p0 < P; p0 += ROWS) {
            __m256 g[ROWS], acc[ROWS];
            for (int j = 0; j < ROWS; j++) { g[j] = _mm256_set1_ps(in.x[h*P+p0+j] * dt); acc[j] = _mm256_setzero_ps(); }
            for (int n = 0; n < N; n += 8) {
                const __m256 vB = _mm256_loadu_ps(in.B + n), vC = _mm256_loadu_ps(in.C + n);
                for (int j = 0; j < ROWS; j++) {
                    float* rowp = state + ((size_t)h * P + p0 + j) * N + n;
                    __m256 s = _mm256_fmadd_ps(g[j], vB, _mm256_mul_ps(_mm256_loadu_ps(rowp), vdA));
                    _mm256_storeu_ps(rowp, s);
                    acc[j] = _mm256_fmadd_ps(s, vC, acc[j]);
                }
            }
            for (int j = 0; j < ROWS; j++) y[h*P+p0+j] = hsum8(acc[j]) + D[h] * in.x[h*P+p0+j];
        }
    }
}

// ---------------------------------------------------------------------------------- FLOPs
// Per state element: 1 mul (decay) + 2R FMA (current + previous rank-1 updates) + R FMA (readout).
// Per head, rotary: R*N*(2 bias adds) + 2*R*S*6 (rotate B and C pairs) - small, counted anyway.
double m3_flops(const M3Params& p) {
    const double per_elem = 1.0 + 2.0 * (2 * p.R) + 2.0 * p.R;
    return (double)p.H * ((double)p.P * p.N * per_elem + p.R * (2.0 * p.N + 12.0 * p.S));
}
double m2_flops(int H, int P, int N) { return (double)H * P * N * (1.0 + 2.0 + 2.0); }
