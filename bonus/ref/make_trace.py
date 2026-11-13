"""
Dump a 1000-step decode trace of ONE real Mamba-3 layer for the C++ kernel check.

Inputs to the layer are the true hidden states produced by the pretrained model on real text
(wikitext-2 test), so the kernel is checked on realistic value ranges (dt, decay, angles), not
synthetic noise. The in_proj output is rounded to float32 ONCE; both the float64 reference and the
kernel consume exactly these float32 values, so the measured error is the kernel's own arithmetic.

usage: python -I make_trace.py <ckpt_dir> <text_file> <tokenizer_dir> <layer> <T> <out_dir>
writes: meta.txt params.f32 proj.f32 y_ref64.f64 y_ref32.f32 ssm_final64.f64
"""
import os, sys, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mamba3_model_ref import Mamba3ModelRef
from transformers import AutoTokenizer

ckpt, text_file, tok_dir, layer, T, out = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5]), sys.argv[6]
os.makedirs(out, exist_ok=True)
tok = AutoTokenizer.from_pretrained(tok_dir)
ids = torch.tensor(tok(open(text_file, encoding="utf-8").read())["input_ids"][:T])
assert len(ids) == T, f"text too short: {len(ids)} tokens"

M = Mamba3ModelRef(ckpt, torch.float64)
u, L = M.hidden_before(ids, layer)                        # (T, d) f64
proj32 = (u @ L.W_in.T).float()                           # rounded once -> shared input

# float64 reference on the float32-rounded inputs
st = L.init_state()
y64 = torch.stack([L.core_step(proj32[t:t+1].double(), st)[0] for t in range(T)])
# PyTorch float32 run of the same math = the floating-point floor any fp32 kernel can expect
from mamba3_ref import Mamba3LayerRef, load_layer
L32 = Mamba3LayerRef(load_layer(f"{ckpt}/pytorch_model.bin", layer), M.cfg, torch.float32)
st32 = L32.init_state()
y32 = torch.stack([L32.core_step(proj32[t:t+1], st32)[0] for t in range(T)])

H, P, N, R, S = L.H, L.P, L.N, L.R, L.S
f = lambda t: t.detach().float().contiguous().reshape(-1)
params = torch.cat([f(L.dt_bias), f(L.D), f(L.Bn_w), f(L.Cn_w), f(L.B_bias), f(L.C_bias),
                    f(L.mimo_x), f(L.mimo_z), f(L.mimo_o)])
params.numpy().tofile(f"{out}/params.f32")
proj32.contiguous().numpy().tofile(f"{out}/proj.f32")
y64.contiguous().numpy().tofile(f"{out}/y_ref64.f64")
y32.contiguous().numpy().tofile(f"{out}/y_ref32.f32")
st["ssm"].contiguous().numpy().tofile(f"{out}/ssm_final64.f64")
with open(f"{out}/meta.txt", "w") as fh:
    for k, v in dict(H=H, P=P, N=N, R=R, S=S, pairwise=int(not L.is_mimo), A_floor=L.A_floor, T=T,
                     d_in_proj=L.W_in.shape[0], layer=layer).items():
        fh.write(f"{k} {v}\n")
dt_stats = torch.nn.functional.softplus(proj32[:, 2*H*P + 2*N*R: 2*H*P + 2*N*R + H].double() + L.dt_bias)
print(f"trace -> {out}: T={T} H={H} P={P} N={N} R={R} S={S} pairwise={int(not L.is_mimo)}; "
      f"dt range [{dt_stats.min():.2e}, {dt_stats.max():.2e}], |y| max {y64.abs().max():.3f}, "
      f"torch-fp32 vs fp64 rel err {((y32.double()-y64).norm()/y64.norm()).item():.2e}")
