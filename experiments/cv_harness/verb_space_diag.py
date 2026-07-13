"""Cross-verb diagnostic: where do the unseen verbs land in verb space?
(1) CLIP-text cosine similarity of each NOVEL verb (lift/open/press) to the 6 TRAINED verbs.
(2) Same after the model's verb_proj (does the learned projection preserve or destroy those relations?).
Tells us if the failure is the ENCODER (novel verb lands far from the right trained verb) or the
HEAD (lands near, but the head still doesn't route it there). CPU, no training."""
import sys
sys.path.insert(0, "src")
import numpy as np, torch
from models.mlp_head import AffordanceMLP, mlp_head_config_from_model_cfg
from vlm.vlm_wrapper import VLMWrapper, VLMConfig

TRAINED = ["contain", "sit", "pour", "move", "display", "grasp"]
NOVEL = ["lift", "open", "press"]
ALLV = TRAINED + NOVEL

ck = torch.load("outputs/basefull_fold0/last.pt", map_location="cpu", weights_only=False)
model = AffordanceMLP(mlp_head_config_from_model_cfg(ck["model_cfg"])); model.load_state_dict(ck["model"]); model.eval()
vlm = VLMWrapper(VLMConfig(device="cpu"))
clip = {v: vlm.encode_text([v])[0].float() for v in ALLV}

def cos(a, b): return float(torch.nn.functional.cosine_similarity(a.reshape(1, -1), b.reshape(1, -1)).item())

# projected-space vectors (what the model actually conditions on)
with torch.no_grad():
    proj = {v: model.verb_proj(clip[v]).reshape(-1) for v in ALLV}

for space, vecs in [("CLIP-text", clip), ("after verb_proj", proj)]:
    print(f"\n=== {space}: cosine(novel verb, trained verb) — closest trained verb in bold-ish (*) ===")
    print(f"{'novel':7s} | " + " ".join(f"{t[:5]:>6s}" for t in TRAINED) + " |  closest")
    for nv in NOVEL:
        sims = {t: cos(vecs[nv], vecs[t]) for t in TRAINED}
        best = max(sims, key=sims.get)
        row = " ".join((f"*{sims[t]:.2f}" if t == best else f"{sims[t]:6.2f}") for t in TRAINED)
        print(f"{nv:7s} | {row} |  {best} ({sims[best]:.2f})")
