"""Precompute all-mpnet-base-v2 embeddings for verbs + synonyms → a {word: tensor(768)} file.
L2-normalized mean-pooled (same recipe as encoder_probe.py). Used by the mpnet-verb retrain + eval."""
import sys, torch
sys.path.insert(0, "src")
from transformers import AutoTokenizer, AutoModel

VERBS = ["contain","sit","pour","move","display","grasp","press","lift"]
HELD = {"grasp":"seize","sit":"rest on","pour":"decant","display":"showcase",
        "move":"reposition","press":"click","lift":"elevate","contain":"keep inside"}
SEEN = {"grasp":"grip","sit":"sit down","pour":"pour out","display":"show",
        "move":"slide","press":"tap","lift":"raise","contain":"store"}
MODEL = "sentence-transformers/all-mpnet-base-v2"

tok = AutoTokenizer.from_pretrained(MODEL); mdl = AutoModel.from_pretrained(MODEL).eval()
def enc(texts):
    b = tok(list(texts), padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        out = mdl(**b).last_hidden_state
    m = b["attention_mask"].unsqueeze(-1).float()
    e = (out * m).sum(1) / m.sum(1).clamp(min=1e-9)
    return e / (e.norm(dim=-1, keepdim=True) + 1e-8)          # L2-normalize

words = list(VERBS) + list(HELD.values()) + list(SEEN.values())
E = enc(words)
d = {w: E[i].float().contiguous() for i, w in enumerate(words)}
torch.save(d, "experiments/verb_emb_mpnet.pt")
print(f"saved {len(d)} embeddings (dim={E.shape[-1]}) → experiments/verb_emb_mpnet.pt")
print("verbs:", VERBS)
