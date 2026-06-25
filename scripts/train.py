import logging
from pathlib import Path

import hydra
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from affordance.data.dataloader import get_dataloader
from affordance.data.dataset import AffordanceDataset
from affordance.models.combined import PN2MLPHead
from affordance.models.mlp_head import MLPHead
from affordance.models.pointnet_plus_plus import PointNetPlusPlusHead
from affordance.training.trainer import Trainer

log = logging.getLogger(__name__)


def _setup_file_logging(output_dir: str) -> None:
    log_path = Path(output_dir) / "train.log"
    root = logging.getLogger()
    # avoid duplicate handlers if Hydra already added one to the same file
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


@hydra.main(config_path="../configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    _setup_file_logging(HydraConfig.get().runtime.output_dir)

    log.info("Hyperparameters:\n" + OmegaConf.to_yaml(cfg))

    torch.manual_seed(cfg.training.seed)

    train_ds = build_dataset(cfg, split="train")
    val_ds = train_ds if cfg.training.overfit else build_dataset(cfg, split="val")
    log.info(f"train={len(train_ds)} samples, val={len(val_ds)} samples")

    train_loader = get_dataloader(train_ds, batch_size=cfg.data.batch_size, shuffle=True, num_workers=cfg.data.num_workers)
    val_loader = get_dataloader(val_ds, batch_size=cfg.data.batch_size, shuffle=False, num_workers=cfg.data.num_workers)

    if cfg.model.name == "pointnet++":
        model = PointNetPlusPlusHead(
            sa1_n=cfg.model.sa1_n,
            sa2_n=cfg.model.sa2_n,
            k=cfg.model.k,
            verb_embedding_dim=cfg.model.verb_embedding_dim,
        )
    elif cfg.model.name == "pn2_mlp":
        model = PN2MLPHead(
            sa1_n=cfg.model.sa1_n,
            sa2_n=cfg.model.sa2_n,
            k=cfg.model.k,
            verb_embedding_dim=cfg.model.verb_embedding_dim,
            layer_dims=list(cfg.model.layer_dims),
            dropout=cfg.model.dropout,
        )
    else:
        model = MLPHead(
            layer_dims=list(cfg.model.layer_dims),
            verb_embedding_dim=cfg.model.verb_embedding_dim,
            dropout=cfg.model.dropout,
        )
    log.info(f"model parameters: {sum(p.numel() for p in model.parameters()):,}")

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        lr=cfg.training.lr,
        epochs=cfg.training.epochs,
        early_stopping_patience=cfg.training.early_stopping_patience,
        target_loss=cfg.training.target_loss,
        lr_schedule=cfg.training.lr_schedule,
    )
    trainer.train()


if __name__ == "__main__":
    main()
