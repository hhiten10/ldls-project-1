"""
Roofline prediction for decode (spec step 4): predicted tok/s = measured bandwidth / bytes per token.

Bytes per decoded token, counted from the GGUF itself:
  * every weight tensor is read once per token (batch 1 => matrix-vector, no reuse),
    except token_embd, of which only 1 row is read -- UNLESS the model ties embeddings (no
    `output.weight`), in which case the LM head multiplies by the whole token_embd matrix;
  * recurrent state (llama.cpp keeps it in F32): read + written once per token per layer
      Mamba-2: ssm state  d_inner*N  + conv state (K-1)*(d_inner + 2*G*N)
      Mamba-1: ssm state  d_inner*N  + conv state (K-1)*d_inner
  * attention KV cache (F16 by default): 2 * n_layer * n_head_kv * head_dim * 2 B * context.

usage: python -I 06_roofline.py <results_dir> <gguf>...    -> results/roofline.csv
"""
import csv, os, sys
from gguf import GGUFReader


def field(r, key, default=None):
    f = r.fields.get(key)
    if f is None:
        return default
    v = f.parts[f.data[0]]
    return v.tobytes().decode() if v.dtype.kind == "u" and len(v) > 1 else v[0].item() if len(v) == 1 else v


def bytes_per_token(path, ctx):
    r = GGUFReader(path)
    arch = field(r, "general.architecture")
    tensors = {t.name: int(t.n_bytes) for t in r.tensors}
    tied = "output.weight" not in tensors
    emb = tensors["token_embd.weight"]
    n_vocab = [t for t in r.tensors if t.name == "token_embd.weight"][0].shape[-1]
    w = sum(v for k, v in tensors.items() if k != "token_embd.weight")
    w += emb if tied else emb / n_vocab
    nl = field(r, f"{arch}.block_count")
    state = 0.0; kv = 0.0
    if arch in ("mamba2", "mamba"):
        d_inner = field(r, f"{arch}.ssm.inner_size"); N = field(r, f"{arch}.ssm.state_size")
        K = field(r, f"{arch}.ssm.conv_kernel"); G = field(r, f"{arch}.ssm.group_count", 0) or 0
        conv_ch = d_inner + 2 * G * N if arch == "mamba2" else d_inner
        state = 2 * 4 * nl * (d_inner * N + (K - 1) * conv_ch)          # read + write, f32
    else:
        n_head = field(r, f"{arch}.attention.head_count"); n_kv = field(r, f"{arch}.attention.head_count_kv")
        emb_len = field(r, f"{arch}.embedding_length")
        hd = field(r, f"{arch}.attention.key_length", emb_len // n_head)
        kv = 2 * nl * n_kv * hd * 2 * ctx                                   # K and V, f16, read
    return dict(arch=arch, weights=w, state=state, kv=kv, total=w + state + kv, tied=tied, layers=nl)


def load_bw(results):
    """mean over reps of STREAM triad and read-only bandwidth per thread count (MB/s -> GB/s)."""
    acc = {}
    with open(os.path.join(results, "hw", "stream.csv")) as f:
        for row in csv.DictReader(f):
            t = row["threads"]
            a = acc.setdefault(t, {"triad": [], "ro": []})
            a["triad"].append(float(row["triad_MBs"]) / 1e3); a["ro"].append(float(row["readonly_MBs"]) / 1e3)
    return {t: {k: sum(v) / len(v) for k, v in d.items()} for t, d in acc.items()}


if __name__ == "__main__":
    results, ggufs = sys.argv[1], sys.argv[2:]
    bw = load_bw(results)
    out = os.environ.get("ROOFLINE_OUT", os.path.join(results, "roofline.csv"))
    with open(out, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["model", "arch", "ctx", "threads", "weights_MB", "state_MB", "kv_MB", "total_MB",
                     "bw_readonly_GBs", "bw_triad_GBs", "pred_tps_readonly", "pred_tps_triad"])
        for g in ggufs:
            name = os.path.basename(g).replace(".gguf", "")
            for ctx in (0, 1024, 8192, 16384, 32768):
                b = bytes_per_token(g, ctx)
                if b["arch"] == "llama" and ctx > 16384:
                    continue
                for t, v in sorted(bw.items(), key=lambda kv: (len(kv[0]), kv[0])):
                    wr.writerow([name, b["arch"], ctx, t, round(b["weights"] / 1e6, 1), round(b["state"] / 1e6, 2),
                                 round(b["kv"] / 1e6, 1), round(b["total"] / 1e6, 1), round(v["ro"], 2),
                                 round(v["triad"], 2), round(v["ro"] * 1e9 / b["total"], 2),
                                 round(v["triad"] * 1e9 / b["total"], 2)])
            b0 = bytes_per_token(g, 0)
            print(f"{name:34s} arch={b0['arch']:7s} tied={b0['tied']!s:5s} weights={b0['weights']/1e6:8.1f} MB "
                  f"state={b0['state']/1e6:6.1f} MB/token")
    print("wrote", out)
