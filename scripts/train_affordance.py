"""Train AffordanceMLP on preprocessed SAM3D manifests.

Usage:
    python scripts/train_affordance.py [--manifest PATH] [--output_dir DIR] [--epochs N] [--lr LR] [--device cuda]
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

_cwd = Path(__file__).resolve().parent.parent
if (_cwd / "src").is_dir():
    sys.path.insert(0, str(_cwd / "src"))

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from tqdm import tqdm

from datasets.data_root_dataset import DataRootDataset
from models.mlp_head import AffordanceMLP, affordance_bce_loss, build_affordance_mlp
from training.vertex_affordance_train import eval_vertex_bce, training_epoch_vertex_bce
from utils.config import load_config

logging.basicConfig(
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
    handlers=[logging.StreamHandler()],
)
log = logging.getLogger(__name__)


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_metrics(
    model: AffordanceMLP,
    dataset: Dataset,
    verb_to_idx: dict[str, int],
    device: torch.device,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Compute mIoU@0.5 and AUPRC over the dataset."""
    from sklearn.metrics import average_precision_score

    model.eval()
    all_probs: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []

    with torch.no_grad():
        for i in range(len(dataset)):
            item = dataset[i]
            if item.get("vertex_affordance") is None:
                continue
            y = item["vertex_affordance"].float()
            mask = item.get("vertex_visible_mask")

            def _to(t):
                return t.float().to(device) if t is not None else None

            logits = model(
                verb_to_idx[item["verb"]],
                slat_vertex=_to(item.get("slat_vertex_features")),
                vlm_features=_to(item.get("vertex_features")),
                dino_cls=_to(item.get("dino_cls")),
                ss_dino_cls=_to(item.get("ss_dino_cls")),
                vertex_normals=_to(item.get("vertex_normals")),
            )
            probs = torch.sigmoid(logits).cpu()

            if mask is not None:
                probs = probs[mask]
                y = y[mask]

            all_probs.append(probs)
            all_labels.append(y)

    probs_cat = torch.cat(all_probs).numpy()
    labels_cat = torch.cat(all_labels).numpy()

    preds = (probs_cat >= threshold).astype(float)
    tp = ((preds == 1) & (labels_cat == 1)).sum()
    fp = ((preds == 1) & (labels_cat == 0)).sum()
    fn = ((preds == 0) & (labels_cat == 1)).sum()
    iou = tp / max(tp + fp + fn, 1)

    auprc = float(average_precision_score(labels_cat, probs_cat)) if labels_cat.sum() > 0 else 0.0

    return {"miou": float(iou), "auprc": auprc}


# ── Verb embeddings ────────────────────────────────────────────────────────────

# ── Dataset helpers ────────────────────────────────────────────────────────────

def _degenerate_label_filter(row, lo: float = 0.005, hi: float = 0.5) -> bool:
    cache = row.sam3d_reconstruction_dir / f"affordance_{row.verb}.pt"
    if not cache.is_file():
        return True
    rate = float(torch.load(cache, weights_only=True).float().mean())
    return lo <= rate <= hi


def load_split(
    manifest: Path,
    split: str,
    cfg: dict,
    *,
    filter_degenerate: bool = True,
) -> DataRootDataset:
    return DataRootDataset(
        manifest_path=manifest,
        cfg=cfg,
        split=split,
        load_mesh_eager=False,
        load_vertex_labels_eager=True,
        load_vertex_semantics_eager=False,
        row_filter=_degenerate_label_filter if filter_degenerate else None,
    )


# ── Checkpoint helpers ─────────────────────────────────────────────────────────

def save_checkpoint(
    path: Path,
    *,
    epoch: int,
    model: AffordanceMLP,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    train_losses: list[float],
    val_losses: list[float],
    val_metrics: list[dict],
    best_auprc: float,
    verb_to_idx: dict[str, int],
) -> None:
    torch.save(
        {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "train_losses": train_losses,
            "val_losses": val_losses,
            "val_metrics": val_metrics,
            "best_auprc": best_auprc,
            "model_cfg": {
                "vlm_dim": model.cfg.vlm_dim,
                "verb_dim": model.cfg.verb_dim,
                "num_verbs": model.cfg.num_verbs,
                "sam3d_dim": model.cfg.sam3d_dim,
                "dino_cls_dim": model.cfg.dino_cls_dim,
                "ss_dino_cls_dim": model.cfg.ss_dino_cls_dim,
                "normals_dim": model.cfg.normals_dim,
                "cond_dim": model.cfg.cond_dim,
                "hidden_dims": list(model.cfg.hidden_dims),
                "dropout": model.cfg.dropout,
            },
            "verb_to_idx": verb_to_idx,
        },
        path,
    )


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(description="Train AffordanceMLP on SAM3D manifests")
    p.add_argument("--manifest", type=Path, default=Path("/workspace/data/sam3d/manifest.jsonl"))
    p.add_argument("--output_dir", type=Path, default=Path("outputs/affordance_mlp"))
    p.add_argument("--config", type=Path, default=None)
    p.add_argument("--epochs", type=int, default=None, help="Override training.num_epochs from config")
    p.add_argument("--lr", type=float, default=None, help="Override training.learning_rate from config")
    p.add_argument("--pos_weight", type=float, default=None, help="Override training.pos_weight from config")
    p.add_argument("--grad_accum", type=int, default=8, help="Samples to accumulate gradients over before each optimizer step")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--max_train_samples", type=int, default=None)
    p.add_argument("--no_val", action="store_true")
    p.add_argument("--resume", action="store_true", help="Resume from latest checkpoint in output_dir")
    p.add_argument("--dino_cls_dim", type=int, default=None, help="Override model.dino_cls_dim from config")
    p.add_argument("--ss_dino_cls_dim", type=int, default=None, help="Override model.ss_dino_cls_dim from config")
    args = p.parse_args()

    cfg = load_config(args.config)
    tr_cfg = cfg.get("training", {})

    n_epochs   = args.epochs    or int(tr_cfg.get("num_epochs", 50))
    lr         = args.lr        or float(tr_cfg.get("learning_rate", 1e-4))
    pos_weight = args.pos_weight or float(tr_cfg.get("pos_weight", 5.0))
    device     = torch.device(args.device)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # File logger alongside stdout
    fh = logging.FileHandler(out_dir / "train.log")
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%H:%M:%S"))
    log.addHandler(fh)

    log.info("Config: epochs=%d lr=%g pos_weight=%g device=%s", n_epochs, lr, pos_weight, device)
    log.info("Manifest: %s", args.manifest)
    log.info("Output:   %s", out_dir)

    # ── Datasets ──────────────────────────────────────────────────────────────
    log.info("Loading datasets …")
    ds_train = load_split(args.manifest, "train", cfg)
    ds_val   = None if args.no_val else load_split(args.manifest, "val", cfg)
    log.info("train: %d samples  val: %s", len(ds_train), len(ds_val) if ds_val else "—")

    # ── Verbs (learned embedding table inside the model) ────────────────────────
    # Read verbs from manifest rows directly — avoids loading all labels just to collect verb names.
    verbs = sorted({row.verb for row in ds_train.rows} | ({row.verb for row in ds_val.rows} if ds_val else set()))
    verb_to_idx = {v: i for i, v in enumerate(verbs)}
    log.info("Verbs (%d): %s", len(verbs), verbs)

    model_cfg_raw = dict(cfg.get("model", {}))
    if args.dino_cls_dim is not None:
        model_cfg_raw["dino_cls_dim"] = args.dino_cls_dim
    if args.ss_dino_cls_dim is not None:
        model_cfg_raw["ss_dino_cls_dim"] = args.ss_dino_cls_dim
    model_cfg_raw["num_verbs"] = len(verbs)
    cfg = {**cfg, "model": model_cfg_raw}

    # ── Model ─────────────────────────────────────────────────────────────────
    model = build_affordance_mlp(cfg).to(device)
    log.info("Model input_dim=%d  params=%s", model.cfg.input_dim,
             f"{sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=max(1, n_epochs // 5), gamma=0.5)

    train_losses: list[float] = []
    val_losses:   list[float] = []
    val_metrics:  list[dict]  = []
    best_auprc   = 0.0
    start_epoch  = 0

    # ── Resume ────────────────────────────────────────────────────────────────
    if args.resume:
        ckpts = sorted(out_dir.glob("epoch_*.pt"))
        if ckpts:
            ckpt = torch.load(ckpts[-1], map_location=device, weights_only=True)
            model.load_state_dict(ckpt["model"])
            optimizer.load_state_dict(ckpt["optimizer"])
            scheduler.load_state_dict(ckpt["scheduler"])
            train_losses  = ckpt["train_losses"]
            val_losses    = ckpt.get("val_losses", [])
            val_metrics   = ckpt.get("val_metrics", [])
            best_auprc    = ckpt.get("best_auprc", 0.0)
            start_epoch   = ckpt["epoch"]
            log.info("Resumed from %s (epoch %d)", ckpts[-1].name, start_epoch)
        else:
            log.warning("--resume set but no checkpoints found — starting from scratch")

    # ── Training loop ─────────────────────────────────────────────────────────
    for ep in range(start_epoch, n_epochs):
        t0 = time.time()

        tr_loss = training_epoch_vertex_bce(
            model, optimizer, ds_train,
            verb_to_idx=verb_to_idx,
            device=device,
            max_samples=args.max_train_samples,
            pos_weight=pos_weight,
            grad_accum=args.grad_accum,
            progress=lambda r: tqdm(r, desc=f"train {ep+1}/{n_epochs}", leave=False),
        )
        train_losses.append(tr_loss)
        scheduler.step()

        va_loss: float | None = None
        metrics: dict | None = None
        if ds_val is not None:
            va_loss = eval_vertex_bce(
                model, ds_val,
                verb_to_idx=verb_to_idx,
                device=device,
                pos_weight=pos_weight,
                progress=lambda r: tqdm(r, desc=f"val   {ep+1}/{n_epochs}", leave=False),
            )
            val_losses.append(va_loss)
            metrics = compute_metrics(model, ds_val, verb_to_idx, device)
            val_metrics.append(metrics)

        elapsed = time.time() - t0
        msg = (
            f"epoch {ep+1:>3}/{n_epochs}"
            f"  train={tr_loss:.4f}"
            + (f"  val={va_loss:.4f}  mIoU={metrics['miou']:.3f}  AUPRC={metrics['auprc']:.3f}" if metrics else "")
            + f"  lr={scheduler.get_last_lr()[0]:.2e}  {elapsed:.0f}s"
        )
        log.info(msg)

        # Periodic checkpoint (every epoch)
        ckpt_path = out_dir / f"epoch_{ep+1:04d}.pt"
        save_checkpoint(
            ckpt_path,
            epoch=ep + 1,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            train_losses=train_losses,
            val_losses=val_losses,
            val_metrics=val_metrics,
            best_auprc=best_auprc,
            verb_to_idx=verb_to_idx,
        )

        # Best model by AUPRC
        if metrics and metrics["auprc"] > best_auprc:
            best_auprc = metrics["auprc"]
            save_checkpoint(
                out_dir / "best.pt",
                epoch=ep + 1,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                train_losses=train_losses,
                val_losses=val_losses,
                val_metrics=val_metrics,
                best_auprc=best_auprc,
                verb_to_idx=verb_to_idx,
            )
            log.info("  ↑ new best AUPRC=%.3f — saved best.pt", best_auprc)

        # Keep only last 3 epoch checkpoints to save disk
        old_ckpts = sorted(out_dir.glob("epoch_*.pt"))[:-3]
        for c in old_ckpts:
            c.unlink()

    log.info("Training complete. Best AUPRC=%.3f", best_auprc)
    log.info("Checkpoints: %s", out_dir)


if __name__ == "__main__":
    main()
