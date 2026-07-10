"""Shared helpers: load the poster GNN model + an object's vertex features directly, run open-vocab verbs."""
from __future__ import annotations
import os, sys
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
from pathlib import Path
import numpy as np, torch

ROOT = Path("/home/kraum/Prototype"); sys.path.insert(0, str(ROOT / "src"))
DATA = Path("/home/datasets/customDatasets/cmr2/reconstructions")
CKPT = ROOT / "outputs/gnn_geom/best.pt"


def _feat(t):
    """Unwrap {'features': ...} dicts, return float tensor."""
    if isinstance(t, dict):
        t = t["features"]
    return t.float()


def load_model(ckpt_path=CKPT):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    mcfg = ck["model_cfg"]
    if mcfg.get("backbone") == "gnn":
        from models.gnn_head import AffordanceGNN, gnn_head_config_from_model_cfg
        cfg = gnn_head_config_from_model_cfg(mcfg); model = AffordanceGNN(cfg)
    else:
        from models.mlp_head import AffordanceMLP, mlp_head_config_from_model_cfg
        cfg = mlp_head_config_from_model_cfg(mcfg); model = AffordanceMLP(cfg)
    model.load_state_dict(ck["model"]); model.eval()
    cfg._backbone = mcfg.get("backbone"); cfg._dino_filename = mcfg.get("dino_filename", "vertex_dino_fine.pt")
    return model, cfg


def load_object(obj: str, dino_filename: str = "vertex_dino_fine.pt"):
    """Return (mesh_trimesh, positions np, kw dict of features) for the model forward.
    NOTE: the trained ckpts expect dino_filename=vertex_dino_fine.pt (NOT vertex_dino.pt)."""
    import trimesh
    d = DATA / obj
    pos = torch.load(d / "vertex_positions.pt", map_location="cpu", weights_only=False).float()
    kw = dict(
        vlm_features=_feat(torch.load(d / "vertex_semantics_clean.pt", map_location="cpu", weights_only=False)),
        dino_vertex=_feat(torch.load(d / dino_filename, map_location="cpu", weights_only=False)),
        vertex_geom=torch.load(d / "vertex_geom.pt", map_location="cpu", weights_only=False).float(),
        slat_vertex=torch.load(d / "slat_vertex_features.pt", map_location="cpu", weights_only=False).float(),
        vertex_normals=torch.load(d / "vertex_normals.pt", map_location="cpu", weights_only=False).float(),
        knn_idx=torch.load(d / "vertex_knn_k8.pt", map_location="cpu", weights_only=False).long(),
        vertex_positions=pos,
        dino_cls=torch.load(d / "dino_cls.pt", map_location="cpu", weights_only=False).float(),
        ss_dino_cls=torch.load(d / "ss_dino_cls.pt", map_location="cpu", weights_only=False).float(),
    )
    mesh = trimesh.load(d / "mesh.glb", process=False, force="mesh")
    return mesh, pos.numpy(), kw


_VLM = None
def encode_verbs(verbs):
    global _VLM
    from vlm.vlm_wrapper import VLMWrapper, VLMConfig
    if _VLM is None:
        _VLM = VLMWrapper(VLMConfig(device="cpu"))
    embs = _VLM.encode_text(list(verbs))
    return {v: e for v, e in zip(verbs, embs)}


def predict(model, kw, emb, backbone="gnn"):
    kw = dict(kw)
    if backbone != "gnn":
        kw.pop("knn_idx", None)          # MLP forward has no knn_idx arg
    with torch.no_grad():
        return torch.sigmoid(model(emb, **kw)).numpy()
