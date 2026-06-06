# Docker: SAM3D + affordance pipeline

This image follows [SAM 3D Objects setup](https://github.com/facebookresearch/sam-3d-objects/blob/main/doc/setup.md): CUDA **12.1**, conda env from `sam-3d-objects/environments/default.yml`, then `pip install -e '.[dev]'`, `'.[p3d]'`, `'.[inference]'`, and the Hydra patch. The affordance repo is installed with **`--no-deps`** so PyTorch stays on SAM3D’s **torch 2.5.1+cu121** stack (do not run a plain `pip install -e .` in `sam3d` without `--no-deps`, or pip may still try to “fix” pins and break **xformers** / **torchaudio**).

- **Single GPU (default):** Compose passes **only one** GPU into `autonomous-pipeline`. Set on the host before `docker compose` / Dev Container:

  ```bash
  export AFFORDANCE_CUDA_DEVICE=2   # physical GPU index (default: 0)
  ```

  This sets both **`gpus.device_ids`** (Docker only injects that device) and **`NVIDIA_VISIBLE_DEVICES`** (CUDA / `nvidia-smi` inside the container). PyTorch then uses **`cuda:0`**, which is that physical GPU.

  To expose **all** GPUs again (not recommended unless you know you need it), edit `docker-compose.yml`: restore `gpus: all` and `NVIDIA_VISIBLE_DEVICES: all`.

## Requirements

- **Linux** host with [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) and a recent driver.
- Meta documents **≥32 GB VRAM** for SAM3D; smaller GPUs may OOM.
- **Network during `docker compose build`** so the Dockerfile can `git clone` [facebookresearch/sam-3d-objects](https://github.com/facebookresearch/sam-3d-objects). You do **not** need `git submodule update` for the image build.
- **Optional local submodule**: if you keep `sam-3d-objects` in your tree, the `.:/workspace` mount overlays the clone at runtime (useful for pins or local patches).
- **Checkpoints**: request access on [Hugging Face — facebook/sam-3d-objects](https://huggingface.co/facebook/sam-3d-objects), then either:
  - download on the host under `sam-3d-objects/checkpoints/` (repo mount makes them visible), or
  - run the [HF download steps](https://github.com/facebookresearch/sam-3d-objects/blob/main/doc/setup.md) inside the container (`HF_TOKEN` is passed through compose when set).

## Build

**Enable BuildKit** (needed for Dockerfile cache mounts that speed rebuilds):

```bash
cd /path/to/Prototype
DOCKER_BUILDKIT=1 docker compose build
```

BuildKit is on by default in recent Docker Desktop / Engine; the variable is harmless if already enabled.

### Why the first build is slow (and why “use conda instead of mamba” does not help)

* **Mamba is already faster than classic `conda`** for the same `environment.yml` (better solver + implementation). Switching to `conda env create` would usually make the **conda** phase *slower*, not faster.
* Most wall time is dominated by:
  * **Huge upstream `default.yml`** — hundreds of pinned CUDA / compiler packages to download.
  * **`pip install -e '.[p3d]'` / `[inference]`** — compiling or building **flash-attn**, **pytorch3d** (git), **gsplat** (git), **xformers**, etc. That is CPU– and I/O–heavy on every clean build.
* **Rebuilds:** the Dockerfile uses **cache mounts** for `/opt/conda/pkgs` and `/root/.cache/pip` so repeated `docker compose build` can reuse downloaded conda packages and pip wheels when those layers need to re-run. **`mamba clean` must not run against that mount** (see Dockerfile comment) or the build fails with `Device or resource busy`.

If you need a *much* faster path, you would have to **change what gets installed** (e.g. inference-only slim env, prebuilt wheels, or a private base image that already contains SAM3D deps) — not the mamba vs conda choice alone.

### “It’s been 20+ minutes — is it stuck?”

Usually **no.** A **first** full build of this image is often **~45–120+ minutes** on a typical workstation, depending on CPU, disk, and network. **20 minutes in** you may still be inside:

1. **`mamba env create`** (large CUDA stack download), or  
2. **`pip install -e ".[dev]"`** (huge dependency tree), or especially  
3. **`pip install -e ".[p3d]"`** — **flash-attn** and related builds can sit with **little console output for tens of minutes** while `nvcc` runs.

**See live compiler output** (recommended if you’re unsure what it’s doing):

```bash
DOCKER_BUILDKIT=1 docker compose build --progress=plain 2>&1 | tee /tmp/docker-sam3d-build.log
```

The Dockerfile now prints **`>>> [n/6] …`** markers between major phases so you can tell which step is running.

Optional: pin the SAM3D revision (branch, tag, or commit SHA) or use a fork:

```bash
export SAM3D_REF=main                    # or a specific tag/commit from upstream
export SAM3D_REPO=https://github.com/facebookresearch/sam-3d-objects.git
DOCKER_BUILDKIT=1 docker compose build
```

Compose forwards these as build args when set (see `docker-compose.yml`).

### `gsplat` build fails: `IndexError` in `_get_cuda_arch_flags` (empty arch list)

During **`docker compose build` there is normally no GPU** in the build context. PyTorch then cannot infer CUDA architectures for compiling `gsplat`, and you can see:

`IndexError: list index out of range` in `torch/utils/cpp_extension.py` → `_get_cuda_arch_flags`.

The Dockerfile sets **`TORCH_CUDA_ARCH_LIST`** to a conservative default (`7.5;8.0;8.6;8.9+PTX`). Tighten it to your hardware to speed up the compile, for example:

```bash
export TORCH_CUDA_ARCH_LIST="8.9+PTX"   # e.g. Ada Lovelace (RTX 40xx)
DOCKER_BUILDKIT=1 docker compose build
```

Or pass a build arg once:

```bash
DOCKER_BUILDKIT=1 docker compose build --build-arg TORCH_CUDA_ARCH_LIST="8.6+PTX"
```

For **Hopper (H100)**, you may need `9.0a+PTX` or `9.0+PTX` depending on your PyTorch / nvcc pair (check [CUDA arch list](https://developer.nvidia.com/cuda-gpus)).

## Develop in VS Code / Cursor (container + notebooks)

**Recommended:** use the **Dev Container** so the editor runs *inside* the same environment as SAM3D / gsplat (no host conda mismatch).

1. Install the **Dev Containers** extension (VS Code) or use Cursor’s built-in dev-container support.
2. From the repo root: Command Palette → **Dev Containers: Reopen in Container** (or **Rebuild and Reopen in Container** after a Dockerfile change).
3. Wait for Compose to start the **`autonomous-pipeline`** service; the workspace folder is **`/workspace`** (your bind-mounted repo).
4. Open a notebook under **`notebooks/`** → **Select Kernel** → **`/opt/conda/envs/sam3d/bin/python`** (or *Python Environments…* → that interpreter). `.vscode/settings.json` already points `python.defaultInterpreterPath` there when you are inside the container.
5. Optional: start Jupyter in a terminal and use port **8888** (forwarded by `.devcontainer/devcontainer.json`):

   ```bash
   jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root
   ```

   Then open the printed URL in a browser, or use the **Jupyter** view in VS Code with the **same** conda interpreter as the kernel.

**Alternative (no Dev Container):** start a long-lived container, then **Attach to Running Container**:

```bash
docker compose run --rm --name sam3d-dev -p 8888:8888 autonomous-pipeline bash -lc 'sleep infinity'
```

In VS Code: Command Palette → **Dev Containers: Attach to Running Container** → pick `sam3d-dev`. Open folder **`/workspace`**. Same interpreter path as above.

---

## VS Code / Cursor: “Python not found” in the container

1. **Workspace settings** — this repo includes **`.vscode/settings.json`** so the Python extension uses  
   **`/opt/conda/envs/sam3d/bin/python`** and the integrated terminal gets the same **`PATH`** / conda env vars after attach. **Reload the window** once after attach: Command Palette → **Developer: Reload Window**.

2. **Interpreter** — Command Palette → **Python: Select Interpreter** → choose **Enter interpreter path** →  
   **`/opt/conda/envs/sam3d/bin/python`**

3. In a **new** integrated terminal, check:

   ```bash
   echo "$PATH"
   which python
   python -V
   ```

4. `docker-compose.yml` also sets **`PATH`**, **`CONDA_PREFIX`**, and **`CONDA_DEFAULT_ENV`** for the service. Recreate the container after pulling changes:

   ```bash
   docker compose up -d --force-recreate
   ```

5. If the Python extension still mis-detects: Command Palette → **Python: Clear Cache and Reload Window**.

## Run an interactive shell

```bash
docker compose run --rm autonomous-pipeline bash
```

Inside the container:

```bash
# Smoke test imports
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
PYTHONPATH=src python -c "from reconstruction.sam3d_wrapper import SAM3DWrapper; print('wrapper ok')"

# Batch reconstruction (needs checkpoints under sam-3d-objects/checkpoints/hf/)
python scripts/generate_sam3d.py --help
```

## Jupyter (optional)

```bash
docker compose run --rm -p 8888:8888 autonomous-pipeline \
  jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root
```

## Notes

- **`PYOPENGL_PLATFORM=egl`** is set in Compose for headless rendering. If `pyrender` fails to create a context, try `osmesa` (software) in your override: `PYOPENGL_PLATFORM=osmesa`.
- **Volume mount** (`.:/workspace` in `docker-compose.yml`) overlays the image’s `/workspace` with your working tree. Keep the submodule and checkpoints on the host under the same paths the code expects.
- **Data directory**: compose may bind-mount a host folder onto **`/workspace/data`** (see `docker-compose.yml`). That hides `./data` from the repo bind mount at that path.
- **`shm_size`**: PyTorch DataLoader and some libs use shared memory; 4 GB is set in compose; increase if you see `/dev/shm` errors.
- If **`docker compose build`** fails on `flash-attn` or OOM during compile, rebuild with a lower parallelism: edit `docker-compose.yml` `build.args.MAX_JOBS` (e.g. `2`).
