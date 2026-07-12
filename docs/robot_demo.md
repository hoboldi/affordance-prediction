# Robot visualization demo

`affordance.visualization.robot_demo` renders a video of a Franka Panda acting on a
predicted 3D affordance, consuming the standard pipeline outputs
(2D image → SAM3D reconstruction → per-vertex affordance scores):

- `<reconstruction dir>/mesh.glb` — faces + appearance (vertex colors)
- `<reconstruction dir>/vertex_positions.pt`, `vertex_normals.pt`
- any per-vertex score tensor: `vertex_pseudolabels_<verb>.pt` or model predictions

The affordance heatmap is baked onto the object as a texture (MuJoCo cannot render
vertex colors), the highest-scoring reachable vertex becomes the contact target, and
the arm approaches along its surface normal. Everything is kinematic — no physics —
so the arbitrary-scale, non-watertight SAM3D meshes need no contact tuning.

## Setup

The Franka Panda model is vendored from
[mujoco_menagerie](https://github.com/google-deepmind/mujoco_menagerie) (gitignored):

```bash
git clone --depth 1 --filter=blob:none --sparse \
    https://github.com/google-deepmind/mujoco_menagerie.git third_party/mujoco_menagerie
git -C third_party/mujoco_menagerie sparse-checkout set franka_emika_panda
```

Rendering happens offscreen through MuJoCo's GL context, so run it in a desktop
session (over plain SSH there is no GL context on macOS).

## Usage

```bash
uv run python -m affordance.visualization.robot_demo \
    --reconstruction data/reconstructions/bottle__34_1397_4376__v000 \
    --verb grasp
```

Output goes to `outputs/robot_demo/<sample>__<verb>/`: `demo.mp4`, per-phase key
frames (`key_phase*.png`), and the baked object assets (`object.obj`,
`object_texture.png`).

Options:

| flag | meaning |
| --- | --- |
| `--verb` | picks `vertex_pseudolabels_<verb>.pt` from the reconstruction dir (default `grasp`) |
| `--scores path.pt` | use an arbitrary per-vertex score tensor instead (e.g. model predictions) |
| `--action pick\|touch\|pour` | robot behaviour; defaults per verb — prehensile verbs (`grasp`, `wrap_grasp`, `lift`, `move`) pick and lift, `pour` picks then tilts, everything else (`contain`, `press`, ...) approaches and touches the region with closed fingertips |
| `--object-size` | largest object dimension in metres after rescaling (SAM3D scale is arbitrary), default 0.18 |
| `--width/--height/--fps` | video settings (default 1280×720 @ 30) |

## How it works

1. **Load** — `vertex_positions.pt` + `mesh.glb` faces/colors are index-aligned
   (the GLB is the same vertex buffer, centered and unit-scaled). Converted from
   glTF Y-up to MuJoCo Z-up, scaled to `--object-size`, placed on a pedestal, and
   yawed so the hot region faces between the robot and the camera.
2. **Heatmap texture** — scores blend into the vertex colors ("hot" colormap),
   the mesh is decimated (`fast-simplification`), UV-unwrapped (`xatlas`), and the
   colors are gouraud-rasterized into a PNG texture in UV space.
3. **Target selection** — among the top-2 % scoring vertices, pick the one whose
   outward normal best faces the robot/camera (approaches from below are excluded).
4. **Kinematics** — damped least-squares IK for the TCP; since SAM3D normals often
   demand poses at the Panda's joint limits, a small search over finger flips
   (the gripper is symmetric) and downward approach tilts picks the reachable
   orientation. Waypoints: home → pre-grasp (11 cm back along the approach) →
   contact → act (close/lift/tilt or touch/retreat).
5. **Fake grasp** — on closure the object (a mocap body) is rigidly parented to the
   hand frame; finger closure width comes from the local object width around the
   contact point.

## Notes / limitations

- SAM3D reconstructions are view-aligned (`vertex_world_rotation: None` in
  `meta.json`), so objects are not guaranteed upright on the pedestal.
- The grasp is visual only: no physics, no collision checking; the fingers can
  intersect the mesh slightly.
- The demo scene XML is written into the menagerie directory
  (`_affordance_demo_scene.xml`) so the Panda's relative asset paths resolve.
