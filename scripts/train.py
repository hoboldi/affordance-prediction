import logging
from pathlib import Path

import hydra
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from affordance.data.dataset import AffordanceDataset
from affordance.models.combined import PN2MLPHead
from affordance.models.mlp_head import AffordanceMLP, mlp_head_config_from_model_cfg
from affordance.training.trainer import Trainer

log = logging.getLogger(__name__)


def _setup_file_logging(output_dir: str) -> None:
    log_path = Path(output_dir) / "train.log"
    root = logging.getLogger()
    if any(isinstance(h, logging.FileHandler) and h.baseFilename == str(log_path.resolve()) for h in root.handlers):
        return
    fh = logging.FileHandler(log_path)
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter("[%(asctime)s][%(name)s][%(levelname)s] - %(message)s"))
    root.addHandler(fh)


def build_dataset(cfg: DictConfig, split: str) -> AffordanceDataset:
    match cfg.data.subset:
        case "single":
            return AffordanceDataset.single(cfg.data.data_root, split)
        case "nine":
            return AffordanceDataset.nine(cfg.data.data_root, split)
        case "ninety":
            return AffordanceDataset.ninety(cfg.data.data_root, split)
        case _:
            return AffordanceDataset(cfg.data.data_root, split)


def build_model(cfg: DictConfig) -> torch.nn.Module:
    raw = OmegaConf.to_container(cfg.model, resolve=True)
    mlp_cfg = mlp_head_config_from_model_cfg(raw)
    if cfg.model.name == "pn2_mlp":
        return PN2MLPHead(
            cfg=mlp_cfg,
            sa1_n=int(raw.get("sa1_n", 256)),
            sa2_n=int(raw.get("sa2_n", 64)),
            k=int(raw.get("k", 16)),
        )
    return AffordanceMLP(mlp_cfg)


@hydra.main(config_path="../configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    _setup_file_logging(HydraConfig.get().runtime.output_dir)
    log.info("Hyperparameters:\n" + OmegaConf.to_yaml(cfg))
    torch.manual_seed(cfg.training.seed)

    train_ds = build_dataset(cfg, split="train")
    val_ds = train_ds if cfg.training.overfit else build_dataset(cfg, split="val")
    log.info(f"train={len(train_ds)} samples, val={len(val_ds)} samples")

    model = build_model(cfg)
    log.info(f"model parameters: {sum(p.numel() for p in model.parameters()):,}")

    checkpoint_dir = Path(HydraConfig.get().runtime.output_dir) / "checkpoints"
    trainer = Trainer(
        model=model,
        train_dataset=train_ds,
        val_dataset=val_ds,
        lr=cfg.training.lr,
        epochs=cfg.training.epochs,
        early_stopping_patience=cfg.training.early_stopping_patience,
        target_loss=cfg.training.target_loss,
        lr_schedule=cfg.training.lr_schedule,
        grad_accum=int(cfg.training.grad_accum),
        checkpoint_dir=checkpoint_dir,
    )
    trainer.train()

    test_ds = train_ds if cfg.training.overfit else build_dataset(cfg, split="test")
    log.info(f"evaluating on {len(test_ds)} test samples")
    trainer.evaluate(test_ds)
    vis_dir = Path(HydraConfig.get().runtime.output_dir) / "visualizations"
    trainer.visualize(test_ds, vis_dir)


if __name__ == "__main__":
    main()
