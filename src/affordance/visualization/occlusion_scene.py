"""Static two-object occlusion scene for posters/figures.

Renders several pipeline objects (2D image -> SAM3D reconstruction -> per-vertex
affordance scores) together in one MuJoCo scene, with one object placed in front
of another so it visually *occludes* it. Two images are produced from the exact
same camera:

  - ``plain.png``   : the SAM3D appearance only (you can see the occlusion)
  - ``heatmap.png`` : both objects overlaid with their affordance heatmaps

The intended use is the "our model reasons about occluded parts" figure: a
laptop occluding a mug, where the mug's affordance heatmap (e.g. grasp on the
handle) is still shown on the parts hidden behind the laptop.

Geometry is baked to textured OBJs exactly like ``robot_demo`` (MuJoCo cannot
render vertex colors); the affordance colormap and UV baking helpers are reused
from that module. No robot, no kinematics -- just a lit still life.

Usage (the default poses are tuned for exactly this laptop-occludes-mug pair):
  uv run python -m affordance.visualization.occlusion_scene \\
      --front data/reconstructions/laptop__112_13277_23636__v000 --front-verb press \\
      --back  data/reconstructions/cup__12_100_593__v000        --back-verb grasp

Pass ``--*-scores path.pt`` to colour the heatmap with model predictions instead
of the pseudolabels, and tune ``--*-size/--*-pos/--*-euler`` to re-compose.
"""

import argparse
from dataclasses import dataclass, field
from pathlib import Path

import imageio.v2 as imageio
import numpy as np

from affordance.visualization.robot_demo import (
    DemoConfig,
    ObjectData,
    _rasterize_uv_colors,
    _write_obj,
    heatmap_colors,
    load_object,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
PANDA_DIR = PROJECT_ROOT / "third_party/mujoco_menagerie/franka_emika_panda"


@dataclass
class SceneObject:
    reconstruction_dir: Path
    scores_path: Path
    name: str
    size: float  # largest dimension after rescaling [m]
    position: tuple[float, float, float]  # bounding-box bottom-centre goes here [m]
    euler_deg: tuple[float, float, float] = (0.0, 0.0, 0.0)  # extrinsic X, Y, Z
    verb: str = ""  # affordance verb, only used to name the per-object output file

    @property
    def label_name(self) -> str:
        category = self.reconstruction_dir.name.split("__")[0] or self.name
        return f"{category}_{self.verb}" if self.verb else category


@dataclass
class OcclusionConfig:
    objects: list[SceneObject]
    output_dir: Path
    texture_res: int = 1024
    target_faces: int = 60_000
    width: int = 1600
    height: int = 1200
    camera_pos: tuple[float, float, float] = (0.0, -0.75, 0.42)
    camera_target: tuple[float, float, float] = (0.0, 0.05, 0.11)
    fov_deg: float = 45.0


@dataclass
class _BakedObject:
    name: str
    obj_path: Path
    plain_png: Path
    heat_png: Path


def _euler_matrix(euler_deg: tuple[float, float, float]) -> np.ndarray:
    """Extrinsic X->Y->Z rotation matrix (degrees)."""
    rx, ry, rz = np.deg2rad(euler_deg)
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)
    mx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    my = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    mz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return mz @ my @ mx


def _place(obj: ObjectData, spec: SceneObject) -> None:
    """Rotate, scale to metric size, and drop onto the table at spec.position (in place)."""
    rot = _euler_matrix(spec.euler_deg)
    obj.positions = obj.positions @ rot.T
    obj.normals = obj.normals @ rot.T

    extent = obj.positions.max(0) - obj.positions.min(0)
    obj.positions *= spec.size / extent.max()

    lo, hi = obj.positions.min(0), obj.positions.max(0)
    shift = np.array(
        [
            spec.position[0] - (lo[0] + hi[0]) / 2,
            spec.position[1] - (lo[1] + hi[1]) / 2,
            spec.position[2] - lo[2],
        ]
    )
    obj.positions += shift


def _bake_object(obj: ObjectData, name: str, cfg: OcclusionConfig) -> _BakedObject:
    """Decimate + UV-unwrap once; bake a plain and a heatmap texture off the same geometry."""
    import fast_simplification
    import xatlas
    from scipy.spatial import cKDTree

    reduction = 1.0 - min(1.0, cfg.target_faces / len(obj.faces))
    if reduction > 0.01:
        dec_v, dec_f = fast_simplification.simplify(
            obj.positions.astype(np.float32), obj.faces.astype(np.int32), target_reduction=reduction
        )
    else:
        dec_v, dec_f = obj.positions.astype(np.float32), obj.faces.astype(np.int32)

    nearest = cKDTree(obj.positions).query(dec_v, k=1)[1]
    dec_normals = obj.normals[nearest]

    vmapping, indices, uvs = xatlas.parametrize(dec_v.astype(np.float64), dec_f.astype(np.uint32))
    verts = dec_v[vmapping]
    vnorm = dec_normals[vmapping]
    sample_idx = nearest[vmapping]  # original-vertex index feeding each unwrapped vertex
    indices = indices.astype(np.int64)

    obj_path = cfg.output_dir / f"{name}.obj"
    _write_obj(obj_path, verts, vnorm, uvs, indices)

    plain_png = cfg.output_dir / f"{name}_plain.png"
    heat_png = cfg.output_dir / f"{name}_heat.png"
    imageio.imwrite(plain_png, _rasterize_uv_colors(uvs, indices, obj.rgb[sample_idx], cfg.texture_res))
    imageio.imwrite(heat_png, _rasterize_uv_colors(uvs, indices, heatmap_colors(obj)[sample_idx], cfg.texture_res))
    return _BakedObject(name, obj_path, plain_png, heat_png)


def _camera_xyaxes(pos: np.ndarray, target: np.ndarray) -> str:
    """MuJoCo camera x/y axes (right, up) for a look-at, world +z up."""
    forward = target - pos
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, np.array([0.0, 0.0, 1.0]))
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    return " ".join(f"{v:.5f}" for v in np.concatenate([right, up]))


def _frame_camera(center: np.ndarray, size: float) -> tuple[np.ndarray, np.ndarray]:
    """A 3/4 front view that frames a single object: from -y, slightly +x and above."""
    direction = np.array([0.35, -1.0, 0.55])
    direction /= np.linalg.norm(direction)
    return center + direction * (2.3 * size), center.copy()


def _build_scene_xml(
    baked: list[_BakedObject], variant: str, cfg: OcclusionConfig, cam_pos, cam_target
) -> str:
    assets, geoms = [], []
    for b in baked:
        png = b.plain_png if variant == "plain" else b.heat_png
        assets.append(
            f'    <texture type="2d" name="{b.name}_tex" file="{png.resolve()}"/>\n'
            f'    <material name="{b.name}_mat" texture="{b.name}_tex" specular="0.2" shininess="0.3"/>\n'
            f'    <mesh name="{b.name}_mesh" file="{b.obj_path.resolve()}"/>'
        )
        geoms.append(
            f'    <geom name="{b.name}_geom" type="mesh" mesh="{b.name}_mesh" '
            f'material="{b.name}_mat" contype="0" conaffinity="0"/>'
        )

    pos = np.asarray(cam_pos)
    xyaxes = _camera_xyaxes(pos, np.asarray(cam_target))
    return f"""
<mujoco model="occlusion scene">
  <visual>
    <headlight diffuse="0.55 0.55 0.55" ambient="0.35 0.35 0.35" specular="0.1 0.1 0.1"/>
    <rgba haze="0.15 0.25 0.35 1"/>
    <global offwidth="{cfg.width}" offheight="{cfg.height}" fovy="{cfg.fov_deg}"/>
    <quality shadowsize="4096"/>
  </visual>

  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.4 0.55 0.75" rgb2="0.1 0.12 0.18" width="512" height="3072"/>
    <texture type="2d" name="groundplane" builtin="checker" mark="edge" rgb1="0.24 0.30 0.38" rgb2="0.16 0.21 0.28"
      markrgb="0.7 0.7 0.75" width="300" height="300"/>
    <material name="groundplane" texture="groundplane" texuniform="true" texrepeat="6 6" reflectance="0.15"/>
{chr(10).join(assets)}
  </asset>

  <worldbody>
    <light pos="0.4 -0.6 1.8" dir="-0.2 0.35 -1" directional="true" diffuse="0.55 0.55 0.55"/>
    <light pos="-0.7 -0.3 1.2" dir="0.5 0.25 -1" diffuse="0.3 0.3 0.3"/>
    <geom name="floor" size="0 0 0.05" type="plane" material="groundplane"/>
{chr(10).join(geoms)}
    <camera name="main" pos="{pos[0]:.4f} {pos[1]:.4f} {pos[2]:.4f}" xyaxes="{xyaxes}"/>
  </worldbody>
</mujoco>
"""


def _load_placed(spec: SceneObject, cfg: OcclusionConfig) -> ObjectData:
    print(f"Loading {spec.name} from {spec.reconstruction_dir.name} ...")
    oc = DemoConfig(
        reconstruction_dir=spec.reconstruction_dir,
        scores_path=spec.scores_path,
        output_dir=cfg.output_dir,
    )
    obj = load_object(oc)
    _place(obj, spec)
    return obj


def _render(cfg: OcclusionConfig, baked: list[_BakedObject], variant: str, cam_pos, cam_target, out: Path) -> Path:
    import mujoco

    xml = _build_scene_xml(baked, variant, cfg, cam_pos, cam_target)
    scene_path = cfg.output_dir / f"_scene_{out.stem}.xml"
    scene_path.write_text(xml)
    model = mujoco.MjModel.from_xml_path(str(scene_path))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    renderer = mujoco.Renderer(model, height=cfg.height, width=cfg.width)
    renderer.update_scene(data, camera="main")
    img = renderer.render()
    renderer.close()
    imageio.imwrite(out, img)
    print(f"Wrote {out}")
    return out


def render_scene(cfg: OcclusionConfig) -> tuple[Path, Path]:
    """All objects in one scene, viewed from the shared cfg camera: plain.png + heatmap.png."""
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    baked: list[_BakedObject] = []
    for spec in cfg.objects:
        obj = _load_placed(spec, cfg)
        print(f"  baking textures for {spec.name} ...")
        baked.append(_bake_object(obj, spec.name, cfg))

    outputs = []
    for variant, fname in (("plain", "plain.png"), ("heat", "heatmap.png")):
        outputs.append(_render(cfg, baked, variant, cfg.camera_pos, cfg.camera_target, cfg.output_dir / fname))
    return outputs[0], outputs[1]


def render_separate(cfg: OcclusionConfig) -> list[Path]:
    """Each object alone, centred on the table and framed by its own camera, heatmap only.

    Produces one ``<category>_<verb>.png`` per object -- the per-object affordance figures.
    """
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    outputs = []
    for spec in cfg.objects:
        solo = SceneObject(
            reconstruction_dir=spec.reconstruction_dir,
            scores_path=spec.scores_path,
            name=spec.name,
            size=spec.size,
            position=(0.0, 0.0, 0.0),  # centred on the table origin
            euler_deg=spec.euler_deg,
            verb=spec.verb,
        )
        obj = _load_placed(solo, cfg)
        print(f"  baking textures for {spec.name} ...")
        baked = [_bake_object(obj, spec.name, cfg)]
        center = (obj.positions.min(0) + obj.positions.max(0)) / 2
        cam_pos, cam_target = _frame_camera(center, spec.size)
        outputs.append(_render(cfg, baked, "heat", cam_pos, cam_target, cfg.output_dir / f"{spec.label_name}.png"))
    return outputs


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _scores_for(rec: Path, verb: str, explicit: Path | None, source: str) -> Path:
    if explicit is not None:
        scores = explicit
    elif source == "human":
        scores = PROJECT_ROOT / "data/human_gt_labels" / rec.name / f"vertex_manuallabels_{verb}.pt"
    else:
        scores = rec / f"vertex_pseudolabels_{verb}.pt"
    if not scores.exists():
        raise FileNotFoundError(f"no {source} affordance scores at {scores}")
    return scores


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--front", type=Path, required=True, help="occluding object reconstruction dir")
    parser.add_argument("--front-verb", default="press")
    parser.add_argument("--front-scores", type=Path, default=None)
    parser.add_argument("--front-size", type=float, default=0.30)
    parser.add_argument("--front-pos", type=float, nargs=3, default=(0.0, -0.14, 0.0))
    parser.add_argument("--front-euler", type=float, nargs=3, default=(0.0, 0.0, 0.0))
    parser.add_argument("--back", type=Path, required=True, help="occluded object reconstruction dir")
    parser.add_argument("--back-verb", default="grasp")
    parser.add_argument("--back-scores", type=Path, default=None)
    parser.add_argument("--back-size", type=float, default=0.17)
    parser.add_argument("--back-pos", type=float, nargs=3, default=(0.19, 0.12, 0.0))
    parser.add_argument("--back-euler", type=float, nargs=3, default=(0.0, 0.0, 80.0))
    parser.add_argument("--labels", choices=["pseudo", "human"], default="pseudo",
                        help="score source when --*-scores is not given: GEAL pseudolabels or human annotations")
    parser.add_argument("--layout", choices=["scene", "separate", "both"], default="scene",
                        help="scene: both objects together (plain+heatmap); separate: one heatmap image per object")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs/occlusion_scene")
    parser.add_argument("--width", type=int, default=1600)
    parser.add_argument("--height", type=int, default=1200)
    args = parser.parse_args()

    objects = [
        SceneObject(
            reconstruction_dir=args.front,
            scores_path=_scores_for(args.front, args.front_verb, args.front_scores, args.labels),
            name="front",
            size=args.front_size,
            position=tuple(args.front_pos),
            euler_deg=tuple(args.front_euler),
            verb=args.front_verb,
        ),
        SceneObject(
            reconstruction_dir=args.back,
            scores_path=_scores_for(args.back, args.back_verb, args.back_scores, args.labels),
            name="back",
            size=args.back_size,
            position=tuple(args.back_pos),
            euler_deg=tuple(args.back_euler),
            verb=args.back_verb,
        ),
    ]
    cfg = OcclusionConfig(objects=objects, output_dir=args.output_dir, width=args.width, height=args.height)
    if args.layout in ("scene", "both"):
        render_scene(cfg)
    if args.layout in ("separate", "both"):
        render_separate(cfg)


if __name__ == "__main__":
    main()
