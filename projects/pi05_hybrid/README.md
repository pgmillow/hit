# PI05 Project

Standalone repository for PI0.5 dataset preparation, training, export, and deployment.

## Layout

- `common/src`: shared config, model, robot spec, ROS topic names, data codecs, and path helpers
- `train/src`: training CLI and engine code
- `deploy/src`: deployment runtime, model loading, ROS nodes, bridge, and safety checks
- `train/config`: training configs
- `deploy/config`: deployment configs
- `train/scripts`: dataset prep, training, and export wrappers
- `deploy/scripts`: inference and packaging wrappers
- `data/raw`: raw recordings or imported source files
- `data/interim`: temporary conversion files
- `data/processed`: local processed datasets or symlinks
- `outputs/checkpoints`: training checkpoints
- `outputs/exports`: deploy bundles and packaged artifacts
- `outputs/logs`: TensorBoard and runtime logs
- `docs`: workflow notes

## Public Repo Defaults

This repository is intentionally configured for public sharing:

- dataset path defaults to `data/processed/lerobot_data`
- pretrained PI0.5 base defaults to `lerobot/pi05_base`
- checkpoints, exports, and logs are written under `outputs/`
- generated data, model weights, logs, and caches are ignored by Git

If your dataset or weights live elsewhere, edit the YAML configs or point them to a local path.
Relative paths in configs are resolved against the config file location.

## Quick Start

Install in editable mode:

```bash
cd /path/to/pi05
pip install -e .
```

For the local robot environment:

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate lerobot312
pip install -e .
```

Launch training:

```bash
bash train/scripts/train_lora.sh
```

Re-export a deploy bundle from a finished run:

```bash
RUN_DIR=outputs/checkpoints/pi05_grasp_generalization_v1 bash train/scripts/export_policy.sh
```

Run deployment from a bundle configured in `deploy/config/deploy.yaml`:

```bash
source /opt/ros/jazzy/setup.bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate lerobot312
bash deploy/scripts/run_inference.sh
```

See `docs/training.md`, `docs/deployment.md`, and `DEPLOY_REPRODUCE.md` for the intended workflow.


export CUDA_VISIBLE_DEVICES=0,1
accelerate launch --multi_gpu --num_processes 2 --mixed_precision bf16 \
  -m pi05.train.cli.train --config train/config/lora.yaml



cd ~/hit/pi05_ws/projects/pi05
source ~/miniconda3/etc/profile.d/conda.sh
conda activate lerobot-pi

export PYTHONNOUSERSITE=1
export PYTHONPATH="common/src:train/src:deploy/src:../../third_party/lerobot/src"

accelerate launch --multi_gpu --num_processes 4 --mixed_precision bf16 \
  -m pi05.train.cli.train --config train/config/lora.yaml