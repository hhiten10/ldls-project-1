"""
Full Mamba-3 language model on CPU, decode-style (token by token through each layer's recurrence),
built on mamba3_ref.Mamba3LayerRef. Used for:
  (1) real hidden-state inputs for the kernel traces (make_trace.py),
  (2) a perplexity sanity check of our CPU decode semantics on real text (SISO and MIMO).

Memory: processes the network LAYER BY LAYER over all T tokens, so only one layer's weights are ever
up-cast (the full model in f64 would need 12 GB). This is mathematically identical to token-by-token
decoding because each layer's recurrence only needs that layer's own state.

Block structure follows mamba_ssm (models/mixer_seq_simple.py + modules/block.py + modules/mlp.py):
  residual = hidden (+ residual);  hidden = RMSNorm(residual);  hidden = Mixer(hidden)
  residual = hidden + residual;    hidden = RMSNorm2(residual); hidden = GatedMLP(hidden)
  end: residual = hidden + residual; hidden = RMSNorm_f(residual); logits = hidden @ embedding.T
"""
import json, torch
import torch.nn.functional as F
from mamba3_ref import Mamba3LayerRef, rmsnorm


class Mamba3ModelRef:
    def __init__(self, ckpt_dir, dtype=torch.float64):
        self.cfg = json.load(open(f"{ckpt_dir}/config.json"))
        self.sd = torch.load(f"{ckpt_dir}/pytorch_model.bin", map_location="cpu", mmap=True, weights_only=True)
        self.dt = dtype
        self.n_layer = self.cfg["n_layer"]
        self.eps = 1e-5
        self.pre = "backbone."

    def w(self, k):
        return self.sd[self.pre + k].to(self.dt)

    def layer(self, i):
        pre = f"{self.pre}layers.{i}.mixer."
        p = {k[len(pre):]: v.float() for k, v in self.sd.items() if k.startswith(pre)}
        return Mamba3LayerRef(p, self.cfg, self.dt)

    def embed(self, ids):
        return self.sd[self.pre + "embedding.weight"][ids].to(self.dt)          # (T, d)

    def run_layer(self, i, hidden, residual, capture=None):
        """hidden/residual: (T, d). Returns new (hidden, residual). capture: dict to store mixer inputs."""
        residual = hidden if residual is None else hidden + residual
        h = rmsnorm(residual, self.w(f"layers.{i}.norm.weight"), self.eps)
        L = self.layer(i)
        if capture is not None:
            capture["u"] = h.clone(); capture["layer"] = L
        st = L.init_state()
        proj = h @ L.W_in.T                                                     # (T, d_in_proj)
        y = torch.stack([L.core_step(proj[t:t + 1], st)[0] for t in range(h.shape[0])])
        h = y @ L.W_out.T
        if f"{self.pre}layers.{i}.mlp.fc1.weight" in self.sd:
            residual = h + residual
            h = rmsnorm(residual, self.w(f"layers.{i}.norm2.weight"), self.eps)
            y, gate = (h @ self.w(f"layers.{i}.mlp.fc1.weight").T).chunk(2, dim=-1)
            h = (y * F.silu(gate)) @ self.w(f"layers.{i}.mlp.fc2.weight").T
        return h, residual

    def hidden_before(self, ids, layer_idx):
        """Mixer input of layer `layer_idx` for all tokens (T, d)."""
        h, r = self.embed(ids), None
        cap = {}
        for i in range(layer_idx + 1):
            h, r = self.run_layer(i, h, r, capture=cap if i == layer_idx else None)
        return cap["u"], cap["layer"]

    @torch.no_grad()
    def nll(self, ids, chunk_vocab=16384):
        """Mean next-token NLL over ids[1:] (natural log)."""
        h, r = self.embed(ids), None
        for i in range(self.n_layer):
            h, r = self.run_layer(i, h, r)
        h = rmsnorm(h + r, self.w("norm_f.weight"), self.eps).float()
        E = self.sd[self.pre + "embedding.weight"]                              # tied lm_head (bf16)
        logits = torch.cat([h @ E[j:j + chunk_vocab].float().T for j in range(0, E.shape[0], chunk_vocab)], -1)
        logits = logits[:, : self.cfg["vocab_size"]].double()
        return F.cross_entropy(logits[:-1], ids[1:]).item()
