import logging
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, ConstantLR
from torch.utils.data import DataLoader, Dataset

from affordance.evaluation.metrics import compute_metrics
from affordance.evaluation.visualize import save_prediction_plys

log = logging.getLogger(__name__)


def _kl_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    bce = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    t = targets.clamp(1e-7, 1 - 1e-7)
    entropy = -(t * t.log() + (1 - t) * (1 - t).log())
    return (bce - entropy).mean()


def _balanced_sample(y: torch.Tensor, n_per_class: int, mask: torch.Tensor | None = None) -> torch.Tensor:
    candidates = mask.nonzero(as_tuple=True)[0] if mask is not None else torch.arange(y.shape[0], device=y.device)
    pos = candidates[y[candidates] > 0.5]
    neg = candidates[y[candidates] <= 0.5]

    def _pick(idx: torch.Tensor) -> torch.Tensor:
        if idx.numel() <= n_per_class:
            return idx
        return idx[torch.randperm(idx.numel(), device=idx.device)[:n_per_class]]

    return torch.cat([_pick(pos), _pick(neg)])


def _item_to_device(item: dict, device: torch.device) -> dict:
    out = dict(item)
    for key in (
        "vertex_features", "vertex_affordance", "slat_vertex_features",
        "vertex_normals", "vertex_positions", "dino_cls", "ss_dino_cls",
        "dino_vertex_features",
    ):
        if out.get(key) is not None:
            out[key] = out[key].float().to(device)
    if out.get("vertex_visible_mask") is not None:
        out["vertex_visible_mask"] = out["vertex_visible_mask"].bool().to(device)
    if out.get("verb_idx") is not None:
        out["verb_idx"] = out["verb_idx"].to(device)
    return out


def _forward(model: nn.Module, item: dict) -> torch.Tensor:
    return model(
        verb_idx=item.get("verb_idx"),
        vlm_features=item.get("vertex_features"),
        slat_vertex=item.get("slat_vertex_features"),
        vertex_normals=item.get("vertex_normals"),
        vertex_positions=item.get("vertex_positions"),
        dino_cls=item.get("dino_cls"),
        ss_dino_cls=item.get("ss_dino_cls"),
        dino_vertex=item.get("dino_vertex_features"),
    )


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        train_dataset: Dataset,
        val_dataset: Dataset,
        lr: float = 1e-4,
        epochs: int = 100,
        checkpoint_dir: str | Path = "checkpoints",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        early_stopping_patience: int = 0,
        early_stopping_min_delta: float = 1e-4,
        lr_schedule: str = "cosine",
        target_loss: float = 0.0,
        grad_accum: int = 8,
        n_vertices_per_class: int = 2048,
    ):
        self.model = model.to(device)
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.epochs = epochs
        self.device = torch.device(device)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_min_delta = early_stopping_min_delta
        self.target_loss = target_loss
        self.grad_accum = grad_accum
        self.n_vertices_per_class = n_vertices_per_class

        self.optimizer = AdamW(model.parameters(), lr=lr)
        if lr_schedule == "cosine":
            self.scheduler = CosineAnnealingLR(self.optimizer, T_max=epochs)
        else:
            self.scheduler = ConstantLR(self.optimizer, factor=1.0, total_iters=epochs)

        self.best_val_loss = float("inf")
        self._epochs_without_improvement = 0

    def train_epoch(self, epoch: int) -> float:
        self.model.train()
        order = torch.randperm(len(self.train_dataset)).tolist()
        self.optimizer.zero_grad()
        pending = 0
        total_loss = 0.0

        for step, i in enumerate(order):
            item = _item_to_device(self.train_dataset[i], self.device)
            logits = _forward(self.model, item)
            y = item["vertex_affordance"]
            idx = _balanced_sample(y, self.n_vertices_per_class, item.get("vertex_visible_mask"))
            loss = _kl_loss(logits[idx], y[idx])
            (loss / self.grad_accum).backward()
            pending += 1
            if pending >= self.grad_accum:
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                self.optimizer.zero_grad()
                pending = 0
            total_loss += loss.item()
            if step % 10 == 0:
                log.info(f"epoch {epoch} step {step}/{len(order)} loss={loss.item():.4f}")

        if pending > 0:
            nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            self.optimizer.zero_grad()

        return total_loss / len(order)

    @torch.no_grad()
    def val_epoch(self) -> float:
        self.model.eval()
        total_loss = 0.0
        for i in range(len(self.val_dataset)):
            item = _item_to_device(self.val_dataset[i], self.device)
            logits = _forward(self.model, item)
            y = item["vertex_affordance"]
            mask = item.get("vertex_visible_mask")
            if mask is not None:
                logits, y = logits[mask], y[mask]
            total_loss += _kl_loss(logits, y).item()
        return total_loss / max(len(self.val_dataset), 1)

    def save_checkpoint(self, epoch: int, val_loss: float, name: str = "checkpoint.pt") -> None:
        torch.save({
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "val_loss": val_loss,
        }, self.checkpoint_dir / name)

    def load_checkpoint(self, path: str | Path) -> int:
        ckpt = torch.load(path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        self.best_val_loss = ckpt["val_loss"]
        return ckpt["epoch"]

    def train(self) -> None:
        for epoch in range(1, self.epochs + 1):
            train_loss = self.train_epoch(epoch)
            val_loss = self.val_epoch()
            self.scheduler.step()

            log.info(f"epoch {epoch}/{self.epochs} train={train_loss:.4f} val={val_loss:.4f} lr={self.scheduler.get_last_lr()[0]:.2e}")

            self.save_checkpoint(epoch, val_loss, name="last.pt")
            if val_loss < self.best_val_loss - self.early_stopping_min_delta:
                self.best_val_loss = val_loss
                self._epochs_without_improvement = 0
                self.save_checkpoint(epoch, val_loss, name="best.pt")
                log.info(f"new best val loss {val_loss:.4f} — saved best.pt")
            else:
                self._epochs_without_improvement += 1

            if self.target_loss > 0 and val_loss < self.target_loss:
                log.info(f"target loss {self.target_loss} reached — stopping")
                break

            if self.early_stopping_patience > 0 and self._epochs_without_improvement >= self.early_stopping_patience:
                log.info(f"early stopping: no improvement for {self.early_stopping_patience} epochs")
                break

    @torch.no_grad()
    def evaluate(self, dataset: Dataset) -> dict[str, float]:
        best_ckpt = self.checkpoint_dir / "best.pt"
        if best_ckpt.exists():
            ckpt = torch.load(best_ckpt, map_location=self.device, weights_only=True)
            self.model.load_state_dict(ckpt["model_state_dict"])
            log.info(f"evaluate: loaded best.pt (epoch {ckpt['epoch']}, val_loss={ckpt['val_loss']:.4f})")
        self.model.eval()
        all_probs, all_targets = [], []
        for i in range(len(dataset)):
            item = _item_to_device(dataset[i], self.device)
            logits = _forward(self.model, item)
            mask = item.get("vertex_visible_mask")
            probs = torch.sigmoid(logits)
            y = item["vertex_affordance"]
            if mask is not None:
                probs, y = probs[mask], y[mask]
            all_probs.append(probs)
            all_targets.append(y)
        metrics = compute_metrics(torch.cat(all_probs), torch.cat(all_targets))
        log.info("test  " + "  ".join(f"{k}={v:.4f}" for k, v in metrics.items()))
        return metrics

    @torch.no_grad()
    def visualize(self, dataset: Dataset, output_dir: Path) -> None:
        self.model.eval()
        for i in range(len(dataset)):
            item = _item_to_device(dataset[i], self.device)
            logits = _forward(self.model, item)
            pred = torch.sigmoid(logits)
            name = f"{item['sample_id']}_{item['verb']}"
            save_prediction_plys(
                item["vertex_positions"], item["vertex_affordance"], pred,
                item.get("vertex_visible_mask", torch.ones(pred.shape[0], dtype=torch.bool)),
                output_dir, name,
            )
        log.info(f"saved visualizations to {output_dir}")
