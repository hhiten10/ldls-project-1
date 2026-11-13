"""
End-to-end check of our CPU decode semantics on the full pretrained Mamba-3 models:
perplexity on the first T tokens of wikitext-2 test, every token decoded through every layer's
recurrence (no chunked/parallel form anywhere).

Also settles the MIMO rotary layout empirically: Mamba3.step() (deployed decode path) uses
"halved" pairing (i, i+N/2) for MIMO, the upstream MIMO test reference uses interleaved pairs.
The layout the model was trained with gives a sane perplexity; the other one does not.

usage: python -I ppl_check.py <models/src> <text> <tokenizer_dir> [T]
"""
import math, os, sys, time, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mamba3_model_ref import Mamba3ModelRef
import mamba3_ref
from transformers import AutoTokenizer

src, text, tokd = sys.argv[1:4]
T = int(sys.argv[4]) if len(sys.argv) > 4 else 1024
tok = AutoTokenizer.from_pretrained(tokd)
ids = torch.tensor(tok(open(text, encoding="utf-8").read())["input_ids"][:T])
print(f"T={T} tokens of {os.path.basename(text)}")

orig_init = mamba3_ref.Mamba3LayerRef.__init__
def run(variant, force_pairwise=None):
    def patched(self, *a, **k):
        orig_init(self, *a, **k)
        if force_pairwise is not None:
            self.rot_pairwise_override = force_pairwise
    mamba3_ref.Mamba3LayerRef.__init__ = patched
    M = Mamba3ModelRef(f"{src}/mamba3-{variant}-1.5b", torch.float32)
    t0 = time.time(); nll = M.nll(ids); dt = time.time() - t0
    mamba3_ref.Mamba3LayerRef.__init__ = orig_init
    label = variant + ("" if force_pairwise is None else f" (rotation forced {'pairwise' if force_pairwise else 'halved'})")
    print(f"  {label:45s} NLL {nll:.4f}  PPL {math.exp(nll):8.2f}   [{dt:.0f}s]", flush=True)
    return nll

run("siso")
run("mimo")                      # deployed Mamba3.step layout (halved)
run("mimo", force_pairwise=True) # upstream test-reference layout
