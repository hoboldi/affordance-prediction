"""Cheap gate for richer verb encoding: do affordance DESCRIPTIONS place the novel verbs closer to the
right trained verb (in CLIP-text space) than the bare words do? If descriptions don't even improve
placement, a retrain won't help. Compares bare-word vs description cosine(novel, its target trained verb)."""
import sys
sys.path.insert(0, "src")
import torch
from vlm.vlm_wrapper import VLMWrapper, VLMConfig

# action-style descriptions (describe the verb's action, not the answer region)
DESC = {
    "contain": "put things inside this object to hold or store them",
    "sit":     "sit down and rest your weight on this object",
    "pour":    "pour liquid out through the opening of this object",
    "move":    "push or grab this object to move it somewhere else",
    "display": "show or present content on the front of this object",
    "grasp":   "grip and hold this object in your hand",
    "lift":    "grip and raise this object up off the surface",
    "open":    "open this object by moving its lid or door",
    "press":   "press down on the surface of this object with a finger",
}
# each novel verb's intuitively-correct target trained verb (from the diagnostic)
TARGET = {"lift": "grasp", "open": "move", "press": "sit"}
vlm = VLMWrapper(VLMConfig(device="cpu"))
def cos(a, b): return float(torch.nn.functional.cosine_similarity(a.reshape(1, -1).float(), b.reshape(1, -1).float()).item())

word = {v: vlm.encode_text([v])[0] for v in DESC}
desc = {v: vlm.encode_text([DESC[v]])[0] for v in DESC}

print(f"{'novel->target':16s} {'bare-word cos':>13s} {'description cos':>15s} {'change':>8s}")
for nv, tv in TARGET.items():
    w = cos(word[nv], word[tv]); d = cos(desc[nv], desc[tv])
    print(f"{nv+' -> '+tv:16s} {w:13.3f} {d:15.3f} {d-w:+8.3f}")

# also: do descriptions SPREAD the trained verbs (less compressed = more discriminative)?
import itertools, numpy as np
tw = np.mean([cos(word[a], word[b]) for a, b in itertools.combinations(['contain','sit','pour','move','display','grasp'], 2)])
td = np.mean([cos(desc[a], desc[b]) for a, b in itertools.combinations(['contain','sit','pour','move','display','grasp'], 2)])
print(f"\nmean pairwise cos among TRAINED verbs:  bare-word={tw:.3f}  description={td:.3f}  (lower=more spread=better)")
