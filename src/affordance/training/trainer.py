import logging
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, ConstantLR
from torch.utils.data import DataLoader

log = logging.getLogger(__name__)


def _masked_kl_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    # KL(targets || sigmoid(logits)) = BCE(logits, targets) - H(targets)
    bce = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    t = targets.clamp(1e-7, 1 - 1e-7)
    entropy = -(t * t.log() + (1 - t) * (1 - t).log())
    return ((bce - entropy) * mask).sum() / mask.sum().clamp(min=1)


def _run_batch(batch: dict, model: nn.Module) -> torch.Tensor:
    logits_list = model(batch)
    total_loss = torch.tensor(0.0, device=logits_list[0].device)
    for logits, targets, visible in zip(logits_list, batch["pseudolabels"], batch["vsem_visible"]):
        targets = targets.to(logits.device)
        visible = visible.float().to(logits.device)
        total_loss = total_loss + _masked_kl_loss(logits, targets, visible)
    return total_loss / len(logits_list)


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        lr: float = 1e-4,
        epochs: int = 100,
        checkpoint_dir: str | Path = "checkpoints",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        early_stopping_patience: int = 0,  # 0 = disabled
        early_stopping_min_delta: float = 1e-4,
        lr_schedule: str = "cosine",  # "cosine" | "constant"
        target_loss: float = 0.0,  # stop early if val loss drops below this
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.epochs = epochs
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_min_delta = early_stopping_min_delta

        self.optimizer = AdamW(model.parameters(), lr=lr)
        if lr_schedule == "cosine":
            self.scheduler = CosineAnnealingLR(self.optimizer, T_max=epochs)
        else:
            self.scheduler = ConstantLR(self.optimizer, factor=1.0, total_iters=epochs)
        self.target_loss = target_loss
        self.best_val_loss = float("inf")
        self._epochs_without_improvement = 0

    def _move_batch(self, batch: dict) -> dict:
        out = {}
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                out[k] = v.to(self.device)
            elif isinstance(v, list) and isinstance(v[0], torch.Tensor):
                out[k] = [t.to(self.device) for t in v]
            else:
                out[k] = v
        return out

    def train_epoch(self, epoch: int) -> float:
        self.model.train()
        total_loss = 0.0
        for step, batch in enumerate(self.train_loader):
            batch = self._move_batch(batch)
            loss = _run_batch(batch, self.model)
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            total_loss += loss.item()
            if step % 10 == 0:
                log.info(f"epoch {epoch} step {step}/{len(self.train_loader)} loss={loss.item():.4f}")
        return total_loss / len(self.train_loader)

    @torch.no_grad()
    def val_epoch(self) -> float:
        self.model.eval()
        total_loss = 0.0
        for batch in self.val_loader:
            batch = self._move_batch(batch)
            total_loss += _run_batch(batch, self.model).item()
        return total_loss / len(self.val_loader)

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
