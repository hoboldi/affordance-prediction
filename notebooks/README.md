# Notebooks

## `reverb_demo.ipynb`

A minimal, end-to-end demo of **ReVerb**: load a trained model and a pre-reconstructed object, predict a
**verb-conditioned per-vertex affordance map**, and visualise **verb routing** — the same object lighting up
different, near-disjoint regions for *contain* / *pour* / *grasp*.

Run it from the repo root (the first cell adds `src/` to the path). The paths at the top
(`RECON_DIR`, `CKPT`) are environment-specific — point them at your own reconstructions and a trained fold
checkpoint.

To predict on a **new image**: reconstruct it with `scripts/generate_sam3d.py`, compute per-vertex features
with `scripts/generate_vertex_dino.py` and `scripts/precompute_geom.py`, then run the notebook's prediction
cells.

For the method, results, and analyses, see [`docs/RESULTS_consolidated.md`](../docs/RESULTS_consolidated.md)
and [`docs/OVERNIGHT_FINDINGS.md`](../docs/OVERNIGHT_FINDINGS.md).
