// Mamba-3 (SISO / MIMO) and Mamba-2 single-layer decode-step kernels for CPU.
// Semantics follow bonus/ref/mamba3_ref.py::Mamba3LayerRef.core_step exactly, which is checked
// against the upstream state-spaces/mamba references (bonus/ref/verify_ref.py).
#pragma once
#include <cstddef>
#include <cstdint>
#include <vector>

struct M3Params {
    int H = 64, P = 64, N = 128, R = 1, S = 32;   // heads, headdim, d_state, mimo rank, rope angles
    bool rotate_pairwise = true;                  // SISO: pairwise (2i,2i+1); MIMO: halved (i, i+N/2)
    float A_floor = 1e-4f;
    std::vector<float> dt_bias, D;                // [H]
    std::vector<float> Bn_w, Cn_w;                // [N]   RMSNorm weights for B, C
    std::vector<float> B_bias, C_bias;            // [H][R][N]
    std::vector<float> mimo_x, mimo_z, mimo_o;    // [H][R][P]  (all ones for SISO)
    int d_inner() const { return H * P; }
    int d_in_proj() const { return 2 * H * P + 2 * N * R + 3 * H + S; }
};

struct M3State {
    std::vector<float> angle;  // [H][S]
    std::vector<float> ssm;    // [H][P][N]   (64B aligned)
    std::vector<float> k;      // [R][H][N]   rotated B of previous step
    std::vector<float> v;      // [H][P]      x of previous step
    void init(const M3Params& p);
};

// One decode step. proj = in_proj(u) for this token ([d_in_proj]); y = [H*P] (before out_proj).
// h_begin/h_end: head range (for multi-threading across heads).
void m3_step_scalar(const M3Params& p, M3State& st, const float* proj, float* y, int h_begin, int h_end);
void m3_step_avx2  (const M3Params& p, M3State& st, const float* proj, float* y, int h_begin, int h_end);

// Shared per-token prologue (normalised B, C; tanh(angle proj)) computed once per token, not per head.
struct M3Shared { std::vector<float> Bn, Cn, tang; };   // [R][N], [R][N], [S]
void m3_prologue(const M3Params& p, const float* proj, M3Shared& sh);
void m3_step_avx2_heads(const M3Params& p, M3State& st, const float* proj, const M3Shared& sh,
                        float* y, int h_begin, int h_end);

// Mamba-2 core step at the same state shape (ngroups=1), no z-gate (matches ggml_ssm_scan's job):
//   state = exp(dt*A) * state + (dt*x) (x) B ;  y = state . C + D*x
struct M2Inputs { const float *x, *dt, *B, *C; };       // x[H*P], dt[H] (already softplus'd), B[N], C[N]
void m2_step_scalar(int H, int P, int N, const float* A, const float* D, float* state, const M2Inputs& in, float* y, int h0, int h1);
void m2_step_avx2  (int H, int P, int N, const float* A, const float* D, float* state, const M2Inputs& in, float* y, int h0, int h1);

// FLOP counts per step (FMA = 2), used for arithmetic intensity.
double m3_flops(const M3Params& p);
double m2_flops(int H, int P, int N);
