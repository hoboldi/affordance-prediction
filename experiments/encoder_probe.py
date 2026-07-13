"""#1 Text-encoder geometry probe (no training).
Question: is the open-vocab failure specific to CLIP's text geometry, or fundamental?
For each held-out synonym, rank the 8 trained verbs by cosine similarity under several text encoders.
If a better encoder routes the FAILED synonyms (decant->pour, click->press, elevate->lift) to the
right verb where CLIP does not, an encoder swap would help open-vocab. CPU."""
import sys, os
sys.path.insert(0, "src")
import numpy as np, torch

VERBS = ["contain","sit","pour","move","display","grasp","press","lift"]
HELD = {"grasp":"seize","sit":"rest on","pour":"decant","display":"showcase",
        "move":"reposition","press":"click","lift":"elevate","contain":"keep inside"}
FAILED = {"decant","click","elevate"}   # the ones CLIP could not generalize to (5-fold)

def l2(x):
    return x / (x.norm(dim=-1, keepdim=True) + 1e-8)

# ---- encoder loaders (each returns enc(list[str]) -> (N,D) tensor) ----
def clip_encoder():
    from vlm.vlm_wrapper import VLMWrapper, VLMConfig
    vlm = VLMWrapper(VLMConfig(device="cpu"))
    return lambda texts: torch.stack([vlm.encode_text([t])[0] for t in texts])

def hf_meanpool(name):
    from transformers import AutoTokenizer, AutoModel
    tok = AutoTokenizer.from_pretrained(name)
    mdl = AutoModel.from_pretrained(name).eval()
    def enc(texts):
        b = tok(list(texts), padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad():
            out = mdl(**b).last_hidden_state              # (N,T,D)
        m = b["attention_mask"].unsqueeze(-1).float()
        return (out * m).sum(1) / m.sum(1).clamp(min=1e-9)  # mean pool
    return enc

ENCODERS = [
    ("CLIP-ViT-B/32 (current)", clip_encoder),
    ("all-mpnet-base-v2",       lambda: hf_meanpool("sentence-transformers/all-mpnet-base-v2")),
    ("all-MiniLM-L6-v2",        lambda: hf_meanpool("sentence-transformers/all-MiniLM-L6-v2")),
    ("roberta-base (raw)",      lambda: hf_meanpool("roberta-base")),
]

def run_encoder(name, build):
    try:
        enc = build()
    except Exception as e:
        print(f"\n### {name}: FAILED to load ({e!r}) — skipped"); return None
    verb_emb = l2(enc(VERBS))                                   # (8,D)
    syns = [HELD[v] for v in VERBS]
    syn_emb = l2(enc(syns))                                     # (8,D)
    sims = syn_emb @ verb_emb.T                                 # (8,8) cosine to each verb
    print(f"\n### {name}")
    print(f"{'synonym':12s} {'true_verb':9s} {'nearest':9s} {'cos_true':>8s} {'ok?':>4s}")
    correct = 0; hard_ok = 0
    for i, v in enumerate(VERBS):
        s = HELD[v]
        near = VERBS[int(sims[i].argmax())]
        cos_true = float(sims[i, i])
        ok = (near == v)
        correct += ok
        if s in FAILED and ok: hard_ok += 1
        mark = "Y" if ok else ("*" if s in FAILED else "n")
        print(f"{s:12s} {v:9s} {near:9s} {cos_true:8.3f} {mark:>4s}")
    print(f"  nearest-verb accuracy: {correct}/8   |   FAILED-word rescues (decant/click/elevate): {hard_ok}/3")
    return correct, hard_ok

print("=== TEXT-ENCODER GEOMETRY PROBE ===")
print("Y=correct nearest verb, n=wrong, *=one of the CLIP-failed words got it right")
summary = {}
for name, build in ENCODERS:
    r = run_encoder(name, build)
    if r is not None: summary[name] = r
print("\n=== SUMMARY (nearest-verb acc / hard-word rescues) ===")
for name, (c, h) in summary.items():
    print(f"  {name:26s}  acc {c}/8   hard {h}/3")
print("\nInterpretation: if an alternative encoder beats CLIP on acc AND rescues hard words,")
print("an encoder swap is worth one fold and reopens open-vocab. If none do, the CLIP bound is fundamental.")
