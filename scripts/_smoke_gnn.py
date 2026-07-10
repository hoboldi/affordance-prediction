"""Smoke test for the AffordanceGNN backbone — LARGEST-mesh memory-bounding check.

Finds the largest mesh (max V) in the cmr2 finepatch manifest, builds an AffordanceGNN matching the
flagship recipe (vlm 128, dino 128, pos/normals 0, text verb embeddings), and runs forward + backward
on THAT object: checks output shape (V,) and finiteness, computes a masked BCE loss against the GEAL
pseudolabels, backprops, verifies a finite non-None grad exists, and reports peak CUDA memory.

The EdgeConv layers are row-chunked (gnn_chunk vertices at a time) + gradient-checkpointed, so peak
memory is bounded regardless of V. Originally the un-chunked (V,k,2C) intermediate + stored backward
activations OOM'd on the >1M-vertex meshes in this manifest.

Run (use expandable_segments to avoid allocator fragmentation on the big mesh):
    CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
        PYTHONPATH=src .venv-affordance/bin/python scripts/_smoke_gnn.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
if (_root / "src").is_dir():
    sys.path.insert(0, str(_root / "src"))

import torch

from models.gnn_head import AffordanceGNN, AffordanceGNNConfig
from models.mlp_head import affordance_bce_loss
from utils.config import load_config

MANIFEST = Path("/home/datasets/customDatasets/cmr2/manifest.finepatch.jsonl")
DINO_FILENAME = "vertex_dino_fine.pt"


def fail(reason: str) -> None:
    print(f"FAILED: {reason}")
    sys.exit(1)


def _find_largest_index(ds) -> tuple[int, int]:
    """Return (dataset_index, V) of the mesh with the most vertices.

    Cheap pass: estimate V from each object's vertex_positions.pt file size (V*3*float32), pick the
    biggest, then return its index in the dataset (the row whose recon dir matches).
    """
    import numpy as np  # noqa: F401  (kept local; not strictly needed)

    best_idx, best_v, best_sz = -1, -1, -1
    for i in range(len(ds)):
        recon = ds.rows[i].sam3d_reconstruction_dir
        if recon is None:
            continue
        p = recon / "vertex_positions.pt"
        if not p.is_file():
            continue
        sz = p.stat().st_size
        if sz > best_sz:
            best_sz, best_idx = sz, i
    if best_idx < 0:
        fail("could not locate any vertex_positions.pt to size the largest mesh")
    # Exact V from the chosen object's labels (avoids loading all positions).
    item = ds[best_idx]
    V = int(item["vertex_affordance"].shape[0]) if item.get("vertex_affordance") is not None else -1
    return best_idx, V


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")
    if device.type == "cuda":
        import os
        print(f"PYTORCH_CUDA_ALLOC_CONF={os.environ.get('PYTORCH_CUDA_ALLOC_CONF', '<unset>')}")

    cfg = load_config()

    # Import load_split lazily (it lives in the training script alongside main()).
    sys.path.insert(0, str(_root / "scripts"))
    from train_affordance import load_split

    ds = load_split(
        MANIFEST, "train", cfg,
        filter_degenerate=False,
        load_vertex_dino=True, dino_filename=DINO_FILENAME,
    )
    if len(ds) == 0:
        fail("no samples loaded from manifest")
    print(f"dataset: {len(ds)} samples")

    verbs = sorted({row.verb for row in ds.rows})
    verb_to_idx = {v: i for i, v in enumerate(verbs)}
    print(f"verbs ({len(verbs)}): {verbs}")

    print("scanning for the largest mesh …")
    idx, V_est = _find_largest_index(ds)
    print(f"largest mesh: index={idx}  sample_id={ds.rows[idx].sample_id}  V={V_est}")

    # Verb text embeddings: try the real CLIP encoder (matches train_affordance); fall back to zeros.
    try:
        from vlm.vlm_wrapper import VLMWrapper, build_vlm_config
        prompts = [v.replace("_", " ") for v in verbs]
        verb_text_emb = VLMWrapper(build_vlm_config(cfg)).encode_text(prompts)
        print(f"verb text emb (CLIP): {tuple(verb_text_emb.shape)}")
    except Exception as e:  # noqa: BLE001
        verb_text_emb = torch.zeros(len(verbs), 512)
        print(f"verb text emb (zeros fallback, {e!r}): {tuple(verb_text_emb.shape)}")

    gcfg = AffordanceGNNConfig(
        vlm_dim=128, dino_vertex_dim=128, sam3d_dim=0, normals_dim=0, pos_dim=0,
        verb_embedding="text", verb_dim=512, num_verbs=len(verbs), verb_text_dim=512,
        gnn_hidden=128, gnn_layers=3, knn_k=8, dropout=0.1, gnn_chunk=32768,
    )
    model = AffordanceGNN(gcfg, verb_text_embeddings=verb_text_emb).to(device)
    model.train()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model params: {n_params:,}  gnn_chunk={gcfg.gnn_chunk}")

    opt = torch.optim.Adam(model.parameters(), lr=1e-4)

    item = ds[idx]
    vlm = item.get("vertex_features")
    dino = item.get("dino_vertex_features")
    pos = item.get("vertex_positions")
    aff = item.get("vertex_affordance")
    mask = item.get("vertex_visible_mask")
    verb = item["verb"]

    if vlm is None:
        fail(f"largest object ({item['sample_id']}) missing vertex_features (CLIP)")
    if dino is None:
        fail(f"largest object ({item['sample_id']}) missing dino_vertex_features")
    if pos is None:
        fail(f"largest object ({item['sample_id']}) missing vertex_positions (needed to build kNN)")
    if aff is None:
        fail(f"largest object ({item['sample_id']}) missing vertex_affordance labels")

    vlm = vlm.float().to(device)
    dino = dino.float().to(device)
    pos = pos.float().to(device)
    aff = aff.float().to(device)
    mask = mask.bool().to(device) if mask is not None else None
    V = vlm.shape[0]
    print(f"running forward+backward on V={V} vertices …")

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    t0 = time.time()

    opt.zero_grad()
    logits = model(
        verb_to_idx[verb],
        vlm_features=vlm,
        dino_vertex=dino,
        vertex_positions=pos,   # kNN built on the fly
    )

    if tuple(logits.shape) != (V,):
        fail(f"output shape {tuple(logits.shape)} != ({V},)")
    if not torch.isfinite(logits).all():
        fail("non-finite logits")

    loss = affordance_bce_loss(logits, aff, mask=mask, pos_weight=5.0)
    if not torch.isfinite(loss):
        fail(f"non-finite loss {loss.item()}")
    loss.backward()

    if device.type == "cuda":
        torch.cuda.synchronize()
    dt_ms = (time.time() - t0) * 1000.0

    grad_ok = any(
        (p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0)
        for p in model.parameters()
    )
    if not grad_ok:
        fail("no finite non-zero grad found")

    opt.step()

    peak_mb = (torch.cuda.max_memory_allocated() / 1e6) if device.type == "cuda" else float("nan")
    print(
        f"  LARGEST {item['sample_id']:<40} verb={verb:<8} V={V:>9}  "
        f"loss={loss.item():.4f}  fwd+bwd={dt_ms:8.1f} ms  peak={peak_mb:8.1f} MB  grad_ok={grad_ok}"
    )

    if device.type == "cuda" and peak_mb > 40000:
        fail(f"peak memory {peak_mb:.0f} MB exceeds the ~40 GB budget")

    print("SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
