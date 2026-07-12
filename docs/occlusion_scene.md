# Occlusion scene

`affordance.visualization.occlusion_scene` renders two pipeline objects together in
one lit MuJoCo still life, with one object placed *in front of* another so it
visually occludes it. From the exact same camera it writes two images:

- `plain.png` — the SAM3D appearance only (you can see the occlusion)
- `heatmap.png` — both objects overlaid with their affordance heatmaps

The intended figure is the "our model reasons about occluded parts" poster panel: a
**laptop occluding a mug**, where the laptop's `press` heatmap lights up the keyboard
and the mug's `grasp` heatmap lights up the handle — with the grasp region running
*behind* the laptop screen and back out again.

Like `robot_demo`, it consumes the standard pipeline outputs
(2D image → SAM3D reconstruction → per-vertex affordance scores) and is
score-agnostic: point `--*-scores` at pseudolabels **or** model predictions, or use
`--labels human` to colour from the per-vertex **human annotations**
(`data/human_gt_labels/<id>/vertex_manuallabels_<verb>.pt`). Heatmap baking (decimate
→ UV-unwrap → gouraud-rasterize) and the "hot red" colormap are reused from
`robot_demo`; this module adds only scene composition and cameras.

Two layouts:

- `--layout scene` (default) — both objects together, from one shared camera:
  `plain.png` + `heatmap.png` (the occlusion figure).
- `--layout separate` — each object **alone**, centred and framed by its own camera,
  heatmap only: one `<category>_<verb>.png` per object (e.g. `laptop_press.png`,
  `cup_grasp.png`). `--layout both` does both.

## Setup

Same as `robot_demo` — the vendored Franka Panda model is not needed here, but MuJoCo
renders offscreen through its GL context, so run it in a desktop session (over plain
SSH there is no GL context on macOS).

## Usage

```bash
# occlusion figure (laptop occludes mug), coloured by GEAL pseudolabels
uv run python -m affordance.visualization.occlusion_scene \
    --front data/reconstructions/laptop__112_13277_23636__v000 --front-verb press \
    --back  data/reconstructions/cup__12_100_593__v000        --back-verb grasp

# separate per-object figures, coloured by human annotations
uv run python -m affordance.visualization.occlusion_scene \
    --front data/reconstructions/laptop__112_13277_23636__v000 --front-verb press \
    --back  data/reconstructions/cup__12_100_593__v000        --back-verb grasp \
    --labels human --layout separate --output-dir outputs/occlusion_scene/human
```

Output goes to `outputs/occlusion_scene/`: the `plain.png` / `heatmap.png` (scene) and
`<category>_<verb>.png` (separate) images, plus the baked per-object assets
(`front.obj`, `front_plain.png`, `front_heat.png`, `back.*`) and the generated scene
XMLs. The default poses are tuned for exactly this laptop/mug pair.

Options:

| flag | meaning |
| --- | --- |
| `--front / --back` | reconstruction dirs for the occluding (front) and occluded (back) object |
| `--front-verb / --back-verb` | picks `vertex_pseudolabels_<verb>.pt` from each dir (default `press` / `grasp`) |
| `--front-scores / --back-scores path.pt` | use an arbitrary per-vertex score tensor instead (e.g. model predictions) |
| `--labels pseudo\|human` | default score source when `--*-scores` is omitted: GEAL pseudolabels or human annotations |
| `--layout scene\|separate\|both` | one combined occlusion scene, or one heatmap image per object |
| `--front-size / --back-size` | largest object dimension in metres after rescaling (default 0.30 / 0.17) |
| `--front-pos / --back-pos x y z` | where the object's bounding-box bottom-centre is placed on the table |
| `--front-euler / --back-euler rx ry rz` | extrinsic X→Y→Z rotation in degrees (SAM3D meshes are view-aligned, so tune this to orient) |
| `--width / --height` | image size (default 1600×1200) |

## How it works

1. **Load & place** — each object is loaded via `robot_demo.load_object` (glTF Y-up →
   MuJoCo Z-up, outward normals), rotated by `--*-euler`, scaled so its largest
   dimension is `--*-size`, and dropped onto the table (`z=0`) at `--*-pos`.
2. **Bake once, colour twice** — the mesh is decimated + UV-unwrapped a single time;
   a *plain* texture (SAM3D appearance) and a *heatmap* texture (appearance blended
   with the `hot` affordance colormap) are baked off the same UVs.
3. **Compose** — a floor, two static mesh geoms (already in world coordinates), two
   lights and a look-at camera. Real z-buffering gives correct occlusion between the
   two separate objects.
4. **Render** — the plain and heatmap scenes differ only by which texture each
   material points at, so both images share pixel-identical geometry and camera.

## Notes / limitations

- SAM3D reconstructions are view-aligned (`vertex_world_rotation: None`), so the
  per-object `--*-euler` is how you get the laptop upright and the mug's handle where
  you want it. The shipped defaults are hand-tuned for the example pair above.
- Purely visual: no physics, no contact — objects are placed, not simulated.
