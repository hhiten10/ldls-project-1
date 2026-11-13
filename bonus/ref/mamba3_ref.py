"""
CPU-only PyTorch reference for ONE Mamba-3 layer decode step (SISO and MIMO) and the
Mamba-2 decode step at the same state size.

The official state-spaces/mamba Mamba3.step() calls CUDA-only kernels (CuTe DSL + Triton).
Its math is defined by two pure-PyTorch reference functions in the same repo, which we copy
here verbatim in logic (Apache-2.0, (c) 2026 Dao AI Lab, Goombalab), pinned to commit e9594ce1:
  - apply_rotary_qk_inference_reference      (mamba_ssm/ops/triton/mamba3/mamba3_mimo_rotary_step.py)
  - selective_state_update_fused_ref_v2      (mamba_ssm/ops/cute/mamba3/mamba3_step_fn.py)
and the pre-processing from Mamba3._preprocess / Mamba3.step (mamba_ssm/modules/mamba3.py).

Everything runs in a chosen dtype (float64 = ground truth, float32 = what kernels use).
"""
import math
import torch
import torch.nn.functional as F


# ----------------------------------------------------------------------------- upstream refs
def apply_rotary_qk_inference_reference(q, k, angle_state, angle_proj, dt,
                                        bias_q=None, bias_k=None, conjugate=False,
                                        rotate_pairwise=True):
    batch, mimo_dim, nheads, headdim = q.shape
    rotary_dim = angle_state.shape[-1] * 2
    angle_proj = torch.tanh(angle_proj)
    angle = angle_state + angle_proj * dt[:, :, None] * math.pi
    angle_state_new = angle
    angle = angle.unsqueeze(1).expand(-1, mimo_dim, -1, -1)
    if bias_q is not None:
        q = q + bias_q[None]
    if bias_k is not None:
        k = k + bias_k[None]
    cos, sin = torch.cos(angle), torch.sin(angle)
    if conjugate:
        sin = -sin
    if rotate_pairwise:
        q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
        k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]
        q_rot = q_rot.reshape(batch, mimo_dim, nheads, rotary_dim // 2, 2)
        k_rot = k_rot.reshape(batch, mimo_dim, nheads, rotary_dim // 2, 2)
        q0, q1 = q_rot[..., 0], q_rot[..., 1]
        k0, k1 = k_rot[..., 0], k_rot[..., 1]
        qo = torch.stack([q0 * cos - q1 * sin, q0 * sin + q1 * cos], -1).reshape(batch, mimo_dim, nheads, rotary_dim)
        ko = torch.stack([k0 * cos - k1 * sin, k0 * sin + k1 * cos], -1).reshape(batch, mimo_dim, nheads, rotary_dim)
        if rotary_dim < headdim:
            qo, ko = torch.cat([qo, q_pass], -1), torch.cat([ko, k_pass], -1)
        return qo, ko, angle_state_new
    half = headdim // 2
    q0, q1 = q[..., :half], q[..., half:]
    k0, k1 = k[..., :half], k[..., half:]
    rh = rotary_dim // 2
    if half > rh:
        pad = list(cos.shape); pad[-1] = half - rh
        cos = torch.cat([cos, torch.ones(pad, dtype=cos.dtype)], -1)
        sin = torch.cat([sin, torch.zeros(pad, dtype=sin.dtype)], -1)
    qo = torch.cat([q0 * cos - q1 * sin, q0 * sin + q1 * cos], -1)
    ko = torch.cat([k0 * cos - k1 * sin, k0 * sin + k1 * cos], -1)
    return qo, ko, angle_state_new


def selective_state_update_fused_ref_v2(state, A, B, C, xproj, x, zproj, z, dt, B_state,
                                        x_state, trap, D, outproj, compute_dtype=torch.float32):
    cd = compute_dtype
    A_f, dt_f, trap_f, D_f = A.to(cd), dt.to(cd), trap.to(cd), D.to(cd)
    x_f, xst_f = x.to(cd), x_state.to(cd)
    B_f, C_f, Bst_f = B.to(cd), C.to(cd), B_state.to(cd)
    Xp_f, st_f = xproj.to(cd), state.to(cd)
    alpha = torch.exp(A_f * dt_f)
    beta = (1.0 - trap_f) * dt_f * alpha
    gamma = trap_f * dt_f
    x_vals = x_f[:, None] * Xp_f[None]
    xs_vals = xst_f[:, None] * Xp_f[None]
    xBt_state = torch.einsum('brnh,brns->bnhs', x_vals * gamma.unsqueeze(-1).unsqueeze(1), B_f)
    xBt_prev = torch.einsum('brnh,brns->bnhs', xs_vals * beta.unsqueeze(-1).unsqueeze(1), Bst_f)
    new_state = st_f * alpha[:, :, None, None] + xBt_state + xBt_prev
    out_r = torch.einsum('bnhs,brns->brnh', new_state, C_f)
    out_r = out_r + x_vals * D_f[None, :, None]
    if z is not None:
        z_vals = z.to(cd)[:, None] * zproj.to(cd)[None]
        out_r = out_r * z_vals * torch.sigmoid(z_vals)
    out = torch.einsum('brnh,rnh->bnh', out_r, outproj.to(cd)) if outproj is not None else out_r
    return out, new_state


# ----------------------------------------------------------------------------- Mamba-3 layer
def heavy_tail_activation(x):
    return x.clamp_min(0) + torch.reciprocal(1 - x.clamp_max(0))


def rmsnorm(x, w, eps=1e-5):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * w


class Mamba3LayerRef:
    """Holds one layer's parameters (from a real checkpoint or random) in `dtype`."""

    def __init__(self, p, cfg, dtype=torch.float64):
        self.dt = dtype
        self.d_model = cfg["d_model"]
        s = cfg["ssm_cfg"]
        self.N = s["d_state"]; self.P = s["headdim"]
        self.d_inner = s["expand"] * self.d_model
        self.H = self.d_inner // self.P
        self.G = s.get("ngroups", 1)
        self.is_mimo = bool(s.get("is_mimo", False))
        self.R = s.get("mimo_rank", 4) if self.is_mimo else 1
        self.A_floor = s.get("A_floor", 1e-4)
        split = int(self.N * s.get("rope_fraction", 0.5))
        split -= split % 2
        self.S = split // 2                       # num_rope_angles
        g = lambda k: p[k].to(dtype)
        self.W_in = g("in_proj.weight")           # (d_in_proj, d_model)
        self.W_out = g("out_proj.weight")         # (d_model, d_inner)
        self.dt_bias, self.D = g("dt_bias"), g("D")
        self.B_bias, self.C_bias = g("B_bias"), g("C_bias")           # (H, R, N)
        self.Bn_w, self.Cn_w = g("B_norm.weight"), g("C_norm.weight")
        if self.is_mimo:
            self.mimo_x, self.mimo_z, self.mimo_o = g("mimo_x"), g("mimo_z"), g("mimo_o")  # (H,R,P)
        else:
            ones = torch.ones(self.H, 1, self.P, dtype=dtype)
            self.mimo_x = self.mimo_z = self.mimo_o = ones

    def split_sizes(self):
        nbc = self.N * self.G * self.R
        return [self.d_inner, self.d_inner, nbc, nbc, self.H, self.H, self.H, self.S]

    def init_state(self, batch=1):
        z = lambda *s: torch.zeros(*s, dtype=self.dt)
        return dict(angle=z(batch, self.H, self.S), ssm=z(batch, self.H, self.P, self.N),
                    k=z(batch, self.R, self.H, self.N), v=z(batch, self.H, self.P))

    def core_step(self, proj, st):
        """proj: in_proj output (batch, d_in_proj). Returns y (batch, d_inner) before out_proj,
        and updates `st` in place. This is exactly what the C++ kernel implements."""
        b = proj.shape[0]
        z, x, B, C, dd_dt, dd_A, trap, ang = torch.split(proj.to(self.dt), self.split_sizes(), -1)
        A = torch.clamp(-heavy_tail_activation(dd_A), max=-self.A_floor)
        DT = F.softplus(dd_dt + self.dt_bias)
        trap = torch.sigmoid(trap)
        B = rmsnorm(B.reshape(b, self.R, self.G, self.N), self.Bn_w).expand(-1, -1, self.H, -1)
        C = rmsnorm(C.reshape(b, self.R, self.G, self.N), self.Cn_w).expand(-1, -1, self.H, -1)
        x = x.reshape(b, self.H, self.P); z = z.reshape(b, self.H, self.P)
        ang = ang.unsqueeze(-2).expand(-1, self.H, -1)
        C, B, new_angle = apply_rotary_qk_inference_reference(
            C, B, st["angle"], ang, DT,
            bias_q=self.C_bias.permute(1, 0, 2), bias_k=self.B_bias.permute(1, 0, 2),
            conjugate=False,
            rotate_pairwise=getattr(self, "rot_pairwise_override", not self.is_mimo))  # Mamba3.step rule
        xpj = self.mimo_x.permute(1, 0, 2); zpj = self.mimo_z.permute(1, 0, 2); opj = self.mimo_o.permute(1, 0, 2)
        y, new_ssm = selective_state_update_fused_ref_v2(
            st["ssm"], A, B, C, xpj, x, zpj, z, DT, st["k"], st["v"], trap, self.D, opj,
            compute_dtype=self.dt)
        st["angle"], st["ssm"], st["k"], st["v"] = new_angle, new_ssm, B.contiguous(), x.contiguous()
        return y.reshape(b, self.d_inner)

    def step(self, u, st):
        return self.core_step(u.to(self.dt) @ self.W_in.T, st) @ self.W_out.T


# ----------------------------------------------------------------------------- Mamba-2 step
def mamba2_core_step(state, x, dt, A, B, C, D):
    """Mamba-2 selective_state_update (mamba_ssm reference semantics), no z gate.
    state (b,H,P,N); x (b,H,P); dt (b,H) already softplus'd; A (H,) negative; B,C (b,N) (ngroups=1)."""
    dA = torch.exp(dt * A[None])                                      # (b,H)
    state = state * dA[..., None, None] + (dt[..., None] * x)[..., None] * B[:, None, None, :]
    y = torch.einsum('bhpn,bn->bhp', state, C) + D[None, :, None] * x
    return y, state


# ----------------------------------------------------------------------------- loading
def load_layer(ckpt_path, layer_idx):
    sd = torch.load(ckpt_path, map_location="cpu", mmap=True, weights_only=True)
    pre = None
    for k in sd:
        if k.endswith(f"layers.{layer_idx}.mixer.in_proj.weight"):
            pre = k[: -len("in_proj.weight")]
    assert pre is not None, f"layer {layer_idx} not found"
    p = {k[len(pre):]: v.float().clone() for k, v in sd.items() if k.startswith(pre)}
    return p
