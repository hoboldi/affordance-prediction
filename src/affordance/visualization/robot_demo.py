"""Kinematic MuJoCo demo: a Franka Panda acts on a predicted 3D affordance.

Consumes the outputs of the existing pipeline (2D image -> SAM3D reconstruction
-> per-vertex affordance scores) and renders a video in which the robot
approaches the highest-affordance point along its surface normal, grasps and
lifts the object. Everything is animated kinematically -- no physics -- so the
arbitrary-scale, non-watertight SAM3D meshes need no contact tuning.

Inputs per sample:
  - reconstruction dir: mesh.glb (faces + appearance), vertex_positions.pt,
    vertex_normals.pt
  - a per-vertex score tensor (.pt): pseudolabels or model predictions

Usage:
  uv run python -m affordance.visualization.robot_demo \\
      --reconstruction data/reconstructions/bottle__34_1397_4376__v000 \\
      --verb grasp
"""

import argparse
from dataclasses import dataclass, field
from pathlib import Path

import fast_simplification
import imageio.v2 as imageio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np
import torch
import trimesh
import xatlas
from scipy.ndimage import distance_transform_edt
from scipy.spatial import cKDTree

PROJECT_ROOT = Path(__file__).resolve().parents[3]
PANDA_DIR = PROJECT_ROOT / "third_party/mujoco_menagerie/franka_emika_panda"

# glTF is Y-up, MuJoCo is Z-up: (x, y, z) -> (x, -z, y)
_YUP_TO_ZUP = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])

# Distance from the "hand" body frame to the point between the fingertips.
_TCP_OFFSET = np.array([0.0, 0.0, 0.1034])

# How the robot acts on the affordance region, per verb. Prehensile verbs get
# pick-and-lift; the rest are demonstrated by approaching and touching.
_ACTION_BY_VERB = {
    "grasp": "pick",
    "wrap_grasp": "pick",
    "lift": "pick",
    "move": "pick",
    "pour": "pour",
}


@dataclass
class DemoConfig:
    reconstruction_dir: Path
    scores_path: Path
    output_dir: Path
    action: str = "pick"  # pick | touch | pour
    object_size: float = 0.18  # largest object dimension after rescaling [m]
    object_xy: tuple[float, float] = (0.58, 0.0)
    pedestal_half: tuple[float, float, float] = (0.12, 0.12, 0.20)
    texture_res: int = 1024
    target_faces: int = 60_000
    fps: int = 30
    width: int = 1280
    height: int = 720
    # horizontal direction the grasp normal is rotated to face: between the
    # robot base (-x) and the camera (-y), so the approach is both reachable
    # and visible.
    facing_azimuth_deg: float = 225.0


@dataclass
class ObjectData:
    positions: np.ndarray  # (N, 3) z-up, metric, placed in the scene
    normals: np.ndarray  # (N, 3) unit, outward
    faces: np.ndarray  # (F, 3)
    rgb: np.ndarray  # (N, 3) in [0, 1]
    scores: np.ndarray  # (N,) raw affordance scores


@dataclass
class GraspTarget:
    point: np.ndarray  # contact point on the surface (world)
    normal: np.ndarray  # outward unit normal at that point (world)
    rotation: np.ndarray = field(default=None)  # (3,3) hand orientation, cols=[x y z]


# ---------------------------------------------------------------------------
# Pipeline outputs -> scene-ready object
# ---------------------------------------------------------------------------


def load_object(cfg: DemoConfig) -> ObjectData:
    rec = cfg.reconstruction_dir
    positions = torch.load(rec / "vertex_positions.pt", map_location="cpu", weights_only=True).numpy()
    normals = torch.load(rec / "vertex_normals.pt", map_location="cpu", weights_only=True).numpy()
    scores = torch.load(cfg.scores_path, map_location="cpu", weights_only=True).float().numpy().reshape(-1)

    mesh = trimesh.load(rec / "mesh.glb", force="mesh")
    faces = np.asarray(mesh.faces, dtype=np.int64)
    rgb = np.asarray(mesh.visual.vertex_colors)[:, :3].astype(np.float64) / 255.0

    n = len(positions)
    if len(mesh.vertices) != n or len(scores) != n:
        raise ValueError(
            f"vertex count mismatch: positions={n}, mesh={len(mesh.vertices)}, scores={len(scores)}"
        )

    positions = positions @ _YUP_TO_ZUP.T
    normals = normals @ _YUP_TO_ZUP.T
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = normals / np.clip(norms, 1e-9, None)

    # stored normals follow the face winding; flip globally if it is inward
    if mesh.volume < 0:
        normals = -normals

    return ObjectData(positions, normals, faces, rgb, scores)


def normalized_scores(scores: np.ndarray) -> np.ndarray:
    lo, hi = np.percentile(scores, [5, 99])
    return np.clip((scores - lo) / max(hi - lo, 1e-9), 0.0, 1.0)


def place_object(obj: ObjectData, cfg: DemoConfig) -> None:
    """Scale to metric size and rotate/translate onto the pedestal, in place."""
    extent = obj.positions.max(0) - obj.positions.min(0)
    obj.positions *= cfg.object_size / extent.max()

    # yaw the object so the horizontal component of the best-scoring outward
    # normal faces between the robot and the camera (reachable + visible)
    s = normalized_scores(obj.scores)
    top = np.argsort(s)[-max(1, int(0.02 * len(s))):]
    mean_n = obj.normals[top[s[top] > 0.8 * s[top].max()]].mean(0)
    horiz = mean_n[:2]
    if np.linalg.norm(horiz) > 0.3:
        cur = np.arctan2(horiz[1], horiz[0])
        tgt = np.deg2rad(cfg.facing_azimuth_deg)
        c, si = np.cos(tgt - cur), np.sin(tgt - cur)
        yaw = np.array([[c, -si, 0.0], [si, c, 0.0], [0.0, 0.0, 1.0]])
        obj.positions = obj.positions @ yaw.T
        obj.normals = obj.normals @ yaw.T

    lo, hi = obj.positions.min(0), obj.positions.max(0)
    pedestal_top = 2 * cfg.pedestal_half[2]
    shift = np.array(
        [cfg.object_xy[0] - (lo[0] + hi[0]) / 2, cfg.object_xy[1] - (lo[1] + hi[1]) / 2, pedestal_top - lo[2]]
    )
    obj.positions += shift


# ---------------------------------------------------------------------------
# Grasp target selection
# ---------------------------------------------------------------------------


def select_grasp(obj: ObjectData, facing_dir: np.ndarray) -> GraspTarget:
    """Highest-affordance vertex whose outward normal allows a feasible approach."""
    s = normalized_scores(obj.scores)
    k = max(1, int(0.02 * len(s)))
    cand = np.argsort(s)[-k:]

    n = obj.normals[cand]
    reach = n @ facing_dir  # surface faces the robot/camera side
    utility = s[cand] + 0.3 * np.clip(reach, 0.0, None)
    from_below = n[:, 2] < -0.3
    if not from_below.all():
        utility[from_below] = -np.inf  # avoid approaching from below if possible

    best = cand[np.argmax(utility)]
    normal = obj.normals[best]

    # hand frame: z = approach (into the surface), y = finger-opening axis
    z = -normal
    up = np.array([0.0, 0.0, 1.0]) if abs(z[2]) < 0.95 else np.array([1.0, 0.0, 0.0])
    y = np.cross(z, up)
    y /= np.linalg.norm(y)
    x = np.cross(y, z)
    rotation = np.column_stack([x, y, z])

    return GraspTarget(point=obj.positions[best].copy(), normal=normal.copy(), rotation=rotation)


def estimate_half_gap(obj: ObjectData, grasp: GraspTarget, radius: float = 0.03) -> float:
    """Half-width of the object around the grasp point along the finger axis."""
    d = np.linalg.norm(obj.positions - grasp.point, axis=1)
    near = obj.positions[d < radius]
    if len(near) < 10:
        return 0.02
    proj = np.abs((near - grasp.point) @ grasp.rotation[:, 1])
    return float(np.clip(np.percentile(proj, 90), 0.004, 0.04))


# ---------------------------------------------------------------------------
# Heatmap texture baking (MuJoCo meshes cannot use vertex colors)
# ---------------------------------------------------------------------------


def heatmap_colors(obj: ObjectData) -> np.ndarray:
    """Overlay the SAM3D appearance with a red affordance heatmap."""
    s = normalized_scores(obj.scores)
    red = np.array([1.0, 0.0, 0.0])
    w = (s**1.5 * 0.9)[:, None]
    return obj.rgb * (1 - w) + red * w


def bake_textured_obj(obj: ObjectData, cfg: DemoConfig) -> tuple[Path, Path]:
    """Decimate, UV-unwrap and bake per-vertex colors into an OBJ + PNG pair."""
    colors = heatmap_colors(obj)

    reduction = 1.0 - min(1.0, cfg.target_faces / len(obj.faces))
    if reduction > 0.01:
        dec_v, dec_f = fast_simplification.simplify(
            obj.positions.astype(np.float32), obj.faces.astype(np.int32), target_reduction=reduction
        )
    else:
        dec_v, dec_f = obj.positions.astype(np.float32), obj.faces.astype(np.int32)

    nearest = cKDTree(obj.positions).query(dec_v, k=1)[1]
    dec_colors = colors[nearest]
    dec_normals = obj.normals[nearest]

    vmapping, indices, uvs = xatlas.parametrize(dec_v.astype(np.float64), dec_f.astype(np.uint32))
    verts = dec_v[vmapping]
    vcols = dec_colors[vmapping]
    vnorm = dec_normals[vmapping]

    texture = _rasterize_uv_colors(uvs, indices.astype(np.int64), vcols, cfg.texture_res)

    obj_path = cfg.output_dir / "object.obj"
    png_path = cfg.output_dir / "object_texture.png"
    imageio.imwrite(png_path, texture)
    _write_obj(obj_path, verts, vnorm, uvs, indices)
    return obj_path, png_path


def _rasterize_uv_colors(uvs: np.ndarray, faces: np.ndarray, colors: np.ndarray, res: int) -> np.ndarray:
    """Gouraud-rasterize per-vertex RGB into UV space, one channel at a time."""
    dpi = 100
    fig = plt.figure(figsize=(res / dpi, res / dpi), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])

    def rasterize(values: np.ndarray) -> np.ndarray:
        ax.clear()
        ax.set_xlim(0, 1), ax.set_ylim(0, 1), ax.axis("off")
        ax.set_facecolor("black")
        ax.tripcolor(uvs[:, 0], uvs[:, 1], faces, values, shading="gouraud", cmap="gray", vmin=0, vmax=1)
        fig.canvas.draw()
        return np.asarray(fig.canvas.buffer_rgba())[..., 0].copy()

    channels = [rasterize(colors[:, c]) for c in range(3)]
    coverage = rasterize(np.ones(len(colors)))
    plt.close(fig)

    img = np.stack(channels, axis=-1)
    mask = coverage > 40
    _, (iy, ix) = distance_transform_edt(~mask, return_indices=True)
    return img[iy, ix]  # dilate charts into the background to hide seams


def _write_obj(path: Path, verts: np.ndarray, normals: np.ndarray, uvs: np.ndarray, faces: np.ndarray) -> None:
    with open(path, "w") as f:
        f.write("mtllib object.mtl\n")
        for v in verts:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for vt in uvs:
            f.write(f"vt {vt[0]:.6f} {vt[1]:.6f}\n")
        for vn in normals:
            f.write(f"vn {vn[0]:.4f} {vn[1]:.4f} {vn[2]:.4f}\n")
        for a, b, c in faces + 1:
            f.write(f"f {a}/{a}/{a} {b}/{b}/{b} {c}/{c}/{c}\n")


# ---------------------------------------------------------------------------
# Scene construction
# ---------------------------------------------------------------------------


def build_scene_xml(obj_path: Path, png_path: Path, grasp: GraspTarget, obj_center: np.ndarray, cfg: DemoConfig) -> Path:
    """Write the demo scene next to panda.xml so its relative assets resolve."""
    if not (PANDA_DIR / "panda.xml").exists():
        raise FileNotFoundError(
            "Franka Panda model not found. Fetch it with:\n"
            "  git clone --depth 1 --filter=blob:none --sparse "
            "https://github.com/google-deepmind/mujoco_menagerie.git third_party/mujoco_menagerie\n"
            "  git -C third_party/mujoco_menagerie sparse-checkout set franka_emika_panda"
        )
    ph = cfg.pedestal_half
    marker = grasp.point - obj_center
    arrow_tip = marker + grasp.normal * 0.06
    xml = f"""
<mujoco model="affordance demo">
  <include file="panda.xml"/>

  <statistic center="{cfg.object_xy[0]} 0 {2 * ph[2]:.3f}" extent="1"/>

  <visual>
    <headlight diffuse="0.6 0.6 0.6" ambient="0.3 0.3 0.3" specular="0 0 0"/>
    <rgba haze="0.15 0.25 0.35 1"/>
    <global offwidth="{cfg.width}" offheight="{cfg.height}"/>
  </visual>

  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0" width="512" height="3072"/>
    <texture type="2d" name="groundplane" builtin="checker" mark="edge" rgb1="0.2 0.3 0.4" rgb2="0.1 0.2 0.3"
      markrgb="0.8 0.8 0.8" width="300" height="300"/>
    <material name="groundplane" texture="groundplane" texuniform="true" texrepeat="5 5" reflectance="0.2"/>
    <texture type="2d" name="object_tex" file="{png_path.resolve()}"/>
    <material name="object_mat" texture="object_tex" specular="0.2" shininess="0.2"/>
    <mesh name="object_mesh" file="{obj_path.resolve()}"/>
  </asset>

  <worldbody>
    <light pos="0 0 2.5" dir="0 0 -1" directional="true"/>
    <light mode="targetbody" target="object" pos="1.2 0.8 1.8" diffuse="0.45 0.45 0.45"/>
    <geom name="floor" size="0 0 0.05" type="plane" material="groundplane"/>
    <geom name="pedestal" type="box" size="{ph[0]} {ph[1]} {ph[2]}" pos="{cfg.object_xy[0]} {cfg.object_xy[1]} {ph[2]}"
      rgba="0.35 0.35 0.38 1"/>

    <body name="object" mocap="true" pos="{obj_center[0]:.4f} {obj_center[1]:.4f} {obj_center[2]:.4f}">
      <geom name="object_geom" type="mesh" mesh="object_mesh" material="object_mat"
        pos="{-obj_center[0]:.4f} {-obj_center[1]:.4f} {-obj_center[2]:.4f}" contype="0" conaffinity="0"/>
      <geom name="grasp_marker" type="sphere" size="0.007" pos="{marker[0]:.4f} {marker[1]:.4f} {marker[2]:.4f}"
        rgba="0.1 1.0 0.4 1" contype="0" conaffinity="0"/>
      <geom name="approach_arrow" type="cylinder" size="0.0022"
        fromto="{marker[0]:.4f} {marker[1]:.4f} {marker[2]:.4f} {arrow_tip[0]:.4f} {arrow_tip[1]:.4f} {arrow_tip[2]:.4f}"
        rgba="0.1 1.0 0.4 0.8" contype="0" conaffinity="0"/>
    </body>

    <camera name="main" mode="targetbody" target="object" pos="1.45 -1.25 0.85"/>
  </worldbody>
</mujoco>
"""
    scene_path = PANDA_DIR / "_affordance_demo_scene.xml"
    scene_path.write_text(xml)
    return scene_path


# ---------------------------------------------------------------------------
# Kinematics
# ---------------------------------------------------------------------------


class PandaKinematics:
    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData):
        self.model, self.data = model, data
        self.hand_id = model.body("hand").id
        arm_joints = [model.joint(f"joint{i}") for i in range(1, 8)]
        self.arm_qpos = np.array([j.qposadr[0] for j in arm_joints])
        self.arm_dofs = np.array([j.dofadr[0] for j in arm_joints])
        self.limits = np.array([j.range for j in arm_joints])
        self.finger_qpos = np.array(
            [model.joint("finger_joint1").qposadr[0], model.joint("finger_joint2").qposadr[0]]
        )
        self.q_home = model.key("home").qpos[self.arm_qpos].copy()

    def set_arm(self, q: np.ndarray) -> None:
        self.data.qpos[self.arm_qpos] = q
        mujoco.mj_forward(self.model, self.data)

    def hand_pose(self) -> tuple[np.ndarray, np.ndarray]:
        pos = self.data.xpos[self.hand_id].copy()
        rot = self.data.xmat[self.hand_id].reshape(3, 3).copy()
        return pos, rot

    def tcp_pose(self) -> tuple[np.ndarray, np.ndarray]:
        pos, rot = self.hand_pose()
        return pos + rot @ _TCP_OFFSET, rot

    def solve_ik(
        self,
        target_pos: np.ndarray,
        target_rot: np.ndarray,
        q_init: np.ndarray,
        iters: int = 300,
        damping: float = 1e-2,
    ) -> np.ndarray:
        """Damped least-squares IK for the TCP, biased toward the home pose."""
        model, data = self.model, self.data
        q = q_init.copy()
        jacp = np.zeros((3, model.nv))
        jacr = np.zeros((3, model.nv))
        quat_err = np.empty(4)
        vel_err = np.empty(3)

        for _ in range(iters):
            self.set_arm(q)
            tcp, rot = self.tcp_pose()
            err_p = target_pos - tcp
            mujoco.mju_mat2Quat(quat_err, (target_rot @ rot.T).flatten())
            mujoco.mju_quat2Vel(vel_err, quat_err, 1.0)
            if np.linalg.norm(err_p) < 5e-4 and np.linalg.norm(vel_err) < 5e-3:
                break

            mujoco.mj_jac(model, data, jacp, jacr, tcp, self.hand_id)
            J = np.vstack([jacp[:, self.arm_dofs], jacr[:, self.arm_dofs]])
            err = np.concatenate([err_p, vel_err])
            far = np.linalg.norm(err_p) > 5e-3
            JJt = J @ J.T + (damping if far else 1e-4) * np.eye(6)
            dq = J.T @ np.linalg.solve(JJt, err)

            if far:  # null-space bias toward home keeps the elbow natural,
                # but its damped projection leaks into task space, so drop
                # it for the final convergence
                dq_null = 0.05 * (self.q_home - q)
                dq += dq_null - J.T @ np.linalg.solve(JJt, J @ dq_null)

            step = np.linalg.norm(dq)
            if step > 0.2:
                dq *= 0.2 / step
            q = np.clip(q + dq, self.limits[:, 0], self.limits[:, 1])

        self.set_arm(q)
        tcp, rot = self.tcp_pose()
        mujoco.mju_mat2Quat(quat_err, (target_rot @ rot.T).flatten())
        mujoco.mju_quat2Vel(vel_err, quat_err, 1.0)
        return q, float(np.linalg.norm(target_pos - tcp)), float(np.linalg.norm(vel_err))

    def best_grasp_orientation(
        self, tcp_contact: np.ndarray, base_rot: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Search over finger flips and downward approach tilts for the pose
        the arm can actually reach (SAM3D normals often demand horizontal
        approaches that pin the elbow/wrist at joint limits)."""
        flip_z = np.diag([-1.0, -1.0, 1.0])  # gripper is symmetric under 180 deg
        seeds = [self.q_home, np.array([0.0, 0.35, 0.0, -1.9, 0.0, 2.25, -0.785])]

        best = None
        for allow_from_below in (False, True):
            for tilt_deg in (0.0, 20.0, -20.0, 40.0, -40.0):
                quat = np.empty(4)
                mujoco.mju_axisAngle2Quat(quat, base_rot[:, 1], np.deg2rad(tilt_deg))  # tilt about finger axis
                tilt_mat = np.zeros(9)
                mujoco.mju_quat2Mat(tilt_mat, quat)
                rot_tilted = tilt_mat.reshape(3, 3) @ base_rot
                if rot_tilted[2, 2] > 0.1 and not allow_from_below:
                    continue
                for rot in (rot_tilted, rot_tilted @ flip_z):
                    for seed in seeds:
                        q, ep, er = self.solve_ik(tcp_contact, rot, seed)
                        cost = ep + 0.05 * er + 0.0002 * abs(tilt_deg)
                        if best is None or cost < best[0]:
                            best = (cost, q, rot, ep, er)
            if best is not None:
                break
        _, q, rot, ep, er = best
        print(f"  grasp IK: residual {ep * 1000:.1f} mm / {er:.3f} rad")
        return q, rot


# ---------------------------------------------------------------------------
# Animation + rendering
# ---------------------------------------------------------------------------


def _ease(t: np.ndarray) -> np.ndarray:
    return 0.5 - 0.5 * np.cos(np.pi * t)


def run_demo(cfg: DemoConfig) -> Path:
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading pipeline outputs from {cfg.reconstruction_dir.name} ...")
    obj = load_object(cfg)
    place_object(obj, cfg)
    az = np.deg2rad(cfg.facing_azimuth_deg)
    facing = np.array([np.cos(az), np.sin(az), 0.25])
    grasp = select_grasp(obj, facing / np.linalg.norm(facing))
    half_gap = estimate_half_gap(obj, grasp)
    print(f"  grasp point {np.round(grasp.point, 3)}, normal {np.round(grasp.normal, 2)}, half-gap {half_gap * 1000:.0f} mm")

    print("Baking affordance heatmap texture ...")
    obj_path, png_path = bake_textured_obj(obj, cfg)

    obj_center = (obj.positions.min(0) + obj.positions.max(0)) / 2
    scene_path = build_scene_xml(obj_path, png_path, grasp, obj_center, cfg)
    model = mujoco.MjModel.from_xml_path(str(scene_path))
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
    mujoco.mj_forward(model, data)

    kin = PandaKinematics(model, data)
    mocap_id = model.body("object").mocapid[0]
    obj_pos0 = model.body("object").pos.copy()
    obj_quat0 = model.body("object").quat.copy()
    data.mocap_pos[mocap_id] = obj_pos0
    data.mocap_quat[mocap_id] = obj_quat0
    mujoco.mj_forward(model, data)

    print("Solving IK waypoints ...")
    inset = 0.010 if cfg.action in ("pick", "pour") else 0.0
    tcp_contact = grasp.point + (-grasp.normal) * inset  # fingertips slightly past the surface
    q_grasp, hand_rot = kin.best_grasp_orientation(tcp_contact, grasp.rotation)
    approach = hand_rot[:, 2]  # actual approach after tilt/flip search
    q_pre, _, _ = kin.solve_ik(tcp_contact - approach * 0.11, hand_rot, q_grasp)

    open_w = 0.04
    # phases: (seconds, q_from, q_to, finger_from, finger_to, attached)
    if cfg.action == "touch":
        # approach with closed fingertips, touch the affordance region, retreat
        phases = [
            (0.6, kin.q_home, kin.q_home, open_w, 0.0, False),
            (2.2, kin.q_home, q_pre, 0.0, 0.0, False),
            (1.3, q_pre, q_grasp, 0.0, 0.0, False),
            (1.2, q_grasp, q_grasp, 0.0, 0.0, False),
            (1.0, q_grasp, q_pre, 0.0, 0.0, False),
            (0.8, q_pre, q_pre, 0.0, 0.0, False),
        ]
    else:
        closed_w = half_gap
        q_lift, _, _ = kin.solve_ik(tcp_contact + np.array([0.0, 0.0, 0.18]), hand_rot, q_grasp)
        phases = [
            (0.6, kin.q_home, kin.q_home, open_w, open_w, False),
            (2.2, kin.q_home, q_pre, open_w, open_w, False),
            (1.3, q_pre, q_grasp, open_w, open_w, False),
            (0.7, q_grasp, q_grasp, open_w, closed_w, False),
            (1.6, q_grasp, q_lift, closed_w, closed_w, True),
        ]
        if cfg.action == "pour":
            # tilt the object about the finger axis, as if emptying it
            quat = np.empty(4)
            mujoco.mju_axisAngle2Quat(quat, hand_rot[:, 1], np.deg2rad(65))
            tilt = np.zeros(9)
            mujoco.mju_quat2Mat(tilt, quat)
            q_tilt, ep, _ = kin.solve_ik(
                tcp_contact + np.array([0.0, 0.0, 0.18]), tilt.reshape(3, 3) @ hand_rot, q_lift
            )
            q_final = q_tilt if ep < 0.02 else q_lift
            phases += [(1.5, q_lift, q_final, closed_w, closed_w, True)]
        else:
            q_show = q_lift.copy()
            q_show[0] -= 0.45
            phases += [(1.8, q_lift, q_show, closed_w, closed_w, True)]
        q_end = phases[-1][2]
        phases += [(1.0, q_end, q_end, closed_w, closed_w, True)]

    print("Rendering ...")
    renderer = mujoco.Renderer(model, height=cfg.height, width=cfg.width)
    video_path = cfg.output_dir / "demo.mp4"
    writer = imageio.get_writer(video_path, fps=cfg.fps, quality=8)

    T_rel = None
    key_frames: list[tuple[str, np.ndarray]] = []
    for phase_idx, (seconds, q0, q1, f0, f1, attached) in enumerate(phases):
        steps = max(1, int(seconds * cfg.fps))
        for i in range(steps):
            a = _ease(np.array((i + 1) / steps))
            kin.data.qpos[kin.arm_qpos] = q0 + a * (q1 - q0)
            kin.data.qpos[kin.finger_qpos] = f0 + a * (f1 - f0)
            mujoco.mj_forward(model, data)

            if attached:
                if T_rel is None:
                    hp, hr = kin.hand_pose()
                    R0 = np.zeros(9)
                    mujoco.mju_quat2Mat(R0, obj_quat0)
                    T_rel = (hr.T @ (obj_pos0 - hp), hr.T @ R0.reshape(3, 3))
                hp, hr = kin.hand_pose()
                data.mocap_pos[mocap_id] = hp + hr @ T_rel[0]
                quat = np.empty(4)
                mujoco.mju_mat2Quat(quat, (hr @ T_rel[1]).flatten())
                data.mocap_quat[mocap_id] = quat
                mujoco.mj_forward(model, data)

            renderer.update_scene(data, camera="main")
            frame = renderer.render()
            writer.append_data(frame)
        key_frames.append((f"phase{phase_idx}", frame))

    writer.close()
    renderer.close()

    for name, frame in key_frames:
        imageio.imwrite(cfg.output_dir / f"key_{name}.png", frame)
    print(f"Wrote {video_path}")
    return video_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--reconstruction", type=Path, required=True, help="SAM3D reconstruction directory")
    parser.add_argument("--verb", default="grasp", help="affordance verb; picks vertex_pseudolabels_<verb>.pt")
    parser.add_argument("--scores", type=Path, default=None, help="per-vertex score .pt (overrides --verb file)")
    parser.add_argument("--action", choices=["pick", "touch", "pour"], default=None,
                        help="robot behaviour; defaults per verb (prehensile verbs pick, others touch)")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--object-size", type=float, default=0.18, help="largest object dimension in metres")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    args = parser.parse_args()

    scores = args.scores or args.reconstruction / f"vertex_pseudolabels_{args.verb}.pt"
    if not scores.exists():
        raise FileNotFoundError(f"no affordance scores at {scores}")
    out = args.output_dir or PROJECT_ROOT / "outputs/robot_demo" / f"{args.reconstruction.name}__{args.verb}"

    cfg = DemoConfig(
        reconstruction_dir=args.reconstruction,
        scores_path=scores,
        output_dir=out,
        action=args.action or _ACTION_BY_VERB.get(args.verb, "touch"),
        object_size=args.object_size,
        fps=args.fps,
        width=args.width,
        height=args.height,
    )
    run_demo(cfg)


if __name__ == "__main__":
    main()
