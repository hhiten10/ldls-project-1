"""
Step 0 of the bonus: prove our CPU reference (mamba3_ref.Mamba3LayerRef.core_step) reproduces
the official Mamba-3 math, using REAL pretrained layer weights, before any C++ is written.

  SISO: our step loop  vs  upstream mamba3_siso_fwd_ref (chunked/quadratic "attention" form,
        an independent formulation)  and  upstream mamba3_siso_step_ref.
  MIMO: our step loop (pairwise rotation)  vs  upstream mamba3_MIMO_step_ref.
        (The deployed Mamba3.step uses halved rotation for MIMO; the upstream test ref uses pairwise.
         Our code supports both; we check the pairwise mode against upstream to validate the math.)

Pass bar, fixed before running: max relative error (max|ours-ref| / max|ref|) <= 1e-5 in float64
for step-vs-step, and <= 1e-4 vs the fwd ref (which internally casts ADT/DT/state to float32).
Usage: python -I verify_ref.py <models/src dir> [layer] [T]
"""
import json, sys, os, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mamba3_ref import Mamba3LayerRef, load_layer, heavy_tail_activation, rmsnorm
import torch.nn.functional as F
import upstream_refs as up

torch.manual_seed(0)
src = sys.argv[1]; layer = int(sys.argv[2]) if len(sys.argv) > 2 else 3
T = int(sys.argv[3]) if len(sys.argv) > 3 else 128
f64 = torch.float64
ok_all = True


def relerr(a, b):
    return ((a - b).abs().max() / b.abs().max()).item()


def report(name, e, bar):
    global ok_all
    ok = e <= bar; ok_all &= ok
    print(f"  {name:55s} max-rel-err = {e:.3e}  (bar {bar:.0e})  {'PASS' if ok else 'FAIL'}")


for variant in ["siso", "mimo"]:
    d = f"{src}/mamba3-{variant}-1.5b"
    cfg = json.load(open(f"{d}/config.json"))
    p = load_layer(f"{d}/pytorch_model.bin", layer)
    L = Mamba3LayerRef(p, cfg, f64)
    print(f"== {variant}: layer {layer}, H={L.H} P={L.P} N={L.N} R={L.R} S={L.S}, T={T}")
    # realistic-scale inputs: unit-RMS hidden states (what the pre-norm feeds the mixer)
    u = torch.randn(1, T, L.d_model, dtype=f64)
    proj = u @ L.W_in.T                                    # (1,T,d_in_proj)
    z, x, B, C, dd_dt, dd_A, trap, ang = torch.split(proj, L.split_sizes(), -1)
    A = torch.clamp(-heavy_tail_activation(dd_A), max=-L.A_floor)
    DT = F.softplus(dd_dt + L.dt_bias)
    Bn = rmsnorm(B.reshape(1, T, L.R, L.G, L.N), L.Bn_w)
    Cn = rmsnorm(C.reshape(1, T, L.R, L.G, L.N), L.Cn_w)
    X = x.reshape(1, T, L.H, L.P); Z = z.reshape(1, T, L.H, L.P)
    Ang = ang.unsqueeze(-2).expand(-1, -1, L.H, -1)

    if variant == "siso":
        st = L.init_state()
        ours = torch.stack([L.core_step(proj[:, t], st) for t in range(T)], 1).reshape(1, T, L.H, L.P)
        Q = Cn[:, :, 0].expand(-1, -1, L.H, -1); K = Bn[:, :, 0].expand(-1, -1, L.H, -1)
        args = dict(Q=Q, K=K, V=X, ADT=(A * DT).transpose(1, 2), DT=DT.transpose(1, 2),
                    Trap=trap.transpose(1, 2), Q_bias=L.C_bias[:, 0], K_bias=L.B_bias[:, 0],
                    Angles=Ang, D=L.D, Z=Z)
        ref_step, _ = up.mamba3_siso_step_ref(**args)
        ref_fwd, _ = up.mamba3_siso_fwd_ref(**args, dtype=f64)
        report("ours vs upstream mamba3_siso_step_ref (f64)", relerr(ours, ref_step.to(f64)), 1e-5)
        report("ours vs upstream mamba3_siso_fwd_ref (chunked form)", relerr(ours, ref_fwd.to(f64)), 1e-4)
    else:
        L.is_mimo = False  # -> rotate_pairwise=True inside core_step (matches upstream test ref)
        st = L.init_state()
        ours = torch.stack([L.core_step(proj[:, t], st) for t in range(T)], 1).reshape(1, T, L.H, L.P)
        L.is_mimo = True
        Q = Cn.expand(-1, -1, -1, L.H, -1); K = Bn.expand(-1, -1, -1, L.H, -1)  # (b,t,r,h,n)
        ref_step, _ = up.mamba3_MIMO_step_ref(
            Q=Q, K=K, V=X, ADT=(A * DT).transpose(1, 2), DT=DT.transpose(1, 2),
            Trap=trap.transpose(1, 2), Q_bias=L.C_bias, K_bias=L.B_bias, Angles=Ang,
            MIMO_V=L.mimo_x, MIMO_O=L.mimo_o, D=L.D, Z=Z, MIMO_Z=L.mimo_z)
        report("ours(pairwise) vs upstream mamba3_MIMO_step_ref", relerr(ours, ref_step.to(f64)), 1e-5)

print("ALL PASS" if ok_all else "SOME FAILED")
sys.exit(0 if ok_all else 1)
