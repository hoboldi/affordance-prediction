import torch
from torch.utils.data import DataLoader

from affordance.data.dataset import AffordanceDataset

# Keys where tensors have a variable first dimension (num_vertices or num_latents)
_VARIABLE_LENGTH_KEYS = {
    "vertex_positions",
    "vertex_normals",
    "slat_coords",
    "slat_feats",
    "slat_vertex_features",
    "vertex_affordance",
    "vertex_features",
    "vertex_visible_mask",
}


def _collate(batch: list[dict]) -> dict:
    out = {}
    for key in batch[0]:
        values = [sample[key] for sample in batch]
        if key in _VARIABLE_LENGTH_KEYS:
            out[key] = values  # list of tensors — pad if needed downstream
        elif isinstance(values[0], torch.Tensor):
            out[key] = torch.stack(values)
        else:
            out[key] = values
    return out


def get_dataloader(
    dataset: AffordanceDataset,
    batch_size: int = 8,
    shuffle: bool = True,
    num_workers: int = 4,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=_collate,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )
