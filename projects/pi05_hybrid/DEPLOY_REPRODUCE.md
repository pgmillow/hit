# Deployment Reproduction Guide

This guide is for reproducing Pi0.5 deployment on another computer from a
fresh checkout. The repository stores code and configs only. Large artifacts
such as datasets, checkpoints, exported bundles, and Hugging Face caches should
be copied separately.

## 1. What To Move

Move by GitHub:

- Source code in this repository
- `environment.yml`
- `requirements.txt`
- `deploy/config/deploy.yaml`
- Documentation and tests

Move by USB drive, LAN, or `scp`:

- `outputs/exports/<run_name>` deployment bundle
- Optional local base model cache if the target computer cannot access Hugging Face
- Optional datasets only if you also plan to train on the target computer

Do not upload these to GitHub:

- `outputs/checkpoints/`
- `outputs/exports/`
- `outputs/logs/`
- `data/raw/`, `data/interim/`, `data/processed/`
- `~/.cache/huggingface`

## 2. Target Computer Prerequisites

Recommended baseline:

- Ubuntu 24.04 or a system compatible with ROS 2 Jazzy
- NVIDIA driver new enough for the PyTorch CUDA wheel you install
- Miniconda or Miniforge
- ROS 2 Jazzy installed at `/opt/ros/jazzy`
- The same robot-side drivers and camera publishers used by your deployment

Check GPU driver:

```bash
nvidia-smi
```

Check ROS:

```bash
source /opt/ros/jazzy/setup.bash
ros2 --version
```

## 3. Clone Code

```bash
mkdir -p ~/pi05_ws/projects
cd ~/pi05_ws/projects
git clone <YOUR_GITHUB_REPO_URL> pi05
cd pi05
```

If you use a local LeRobot checkout instead of an installed package, put it in
one of the paths discovered by the project scripts, for example:

```bash
mkdir -p ~/pi05_ws/third_party
cd ~/pi05_ws/third_party
git clone <LEROBOT_REPO_URL> lerobot
```

## 4. Create Conda Environment

Recommended clean setup:

```bash
source ~/miniconda3/etc/profile.d/conda.sh
cd ~/pi05_ws/projects/pi05
conda env create -f environment.yml
conda activate lerobot312
pip install -e .
```

If the environment already exists:

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate lerobot312
cd ~/pi05_ws/projects/pi05
pip install -U -r requirements.txt
pip install -e .
```

Do not directly copy `~/miniconda3/envs/lerobot312` between machines unless the
OS, driver stack, usernames, and paths are effectively identical. If you must
move the current environment, use `conda-pack` rather than a raw directory copy.

Source machine:

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate base
pip install conda-pack
conda pack -n lerobot312 -o lerobot312.tar.gz
```

Target machine:

```bash
mkdir -p ~/miniconda3/envs/lerobot312
tar -xzf lerobot312.tar.gz -C ~/miniconda3/envs/lerobot312
~/miniconda3/envs/lerobot312/bin/conda-unpack
```

## 5. Copy Deployment Bundle

Copy the exported bundle from the training machine:

```text
outputs/exports/<run_name>
```

Example target location:

```bash
mkdir -p ~/pi05_ws/projects/pi05/outputs/exports
rsync -av /path/to/pi05_grasp_generalization_v1 \
  ~/pi05_ws/projects/pi05/outputs/exports/
```

Then update `deploy/config/deploy.yaml`:

```yaml
bundle:
  bundle_dir: ../../outputs/exports/pi05_grasp_generalization_v1
```

## 6. Verify Install

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate lerobot312
cd ~/pi05_ws/projects/pi05
python -m compileall common/src deploy/src tests/deploy
python -m pytest tests/train/test_config.py tests/deploy
```

Verify ROS entrypoints:

```bash
source /opt/ros/jazzy/setup.bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate lerobot312
cd ~/pi05_ws/projects/pi05
python -m pi05.deploy.cli.deploy_ros --help
python -m pi05.deploy.cli.bridge_ros --help
```

If `python -m pi05...` cannot find the package, run `pip install -e .` again
from the repository root.

## 7. Run Deployment

Start with `dry-run` or `shadow-run` in `deploy/config/deploy.yaml`.

```bash
source /opt/ros/jazzy/setup.bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate lerobot312
cd ~/pi05_ws/projects/pi05
bash deploy/scripts/run_inference.sh
```

Start the optional bridge only when you want to forward `/pi05_vla/command/*`
to the downstream execution stack:

```bash
bash deploy/scripts/run_bridge.sh
```

## 8. Common Issues

`torch.cuda.is_available()` is false:

- Check `nvidia-smi`.
- Reinstall a PyTorch wheel compatible with the target driver.

`rclpy` import fails inside conda:

- Source ROS first: `source /opt/ros/jazzy/setup.bash`.
- Make sure the target machine has ROS 2 Jazzy installed.

Bundle path error:

- Check `bundle.bundle_dir` in `deploy/config/deploy.yaml`.
- Confirm the target directory contains `manifest.json`, `normalizers.json`,
  `experiment_config.yaml`, and `adapter/`.

Robot does not move:

- In `dry-run`, command topics are not published.
- In `shadow-run`, `/pi05_vla/command/*` is published but bridge forwarding may
  be disabled.
- For hardware forwarding, use `safe-run` and enable `bridge.enabled` plus
  `bridge.publish_to_picotele` only after topic and safety checks pass.
