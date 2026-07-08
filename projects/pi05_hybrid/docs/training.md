# Training Workflow

## 1. Prepare Data

Default dataset location for this repo:

```text
data/processed/lerobot_data
```

If you already have a converted LeRobot dataset elsewhere, update `train/config/lora.yaml` or `train/config/data.yaml` to point at it.

To run the converter wrapper, either keep a local converter script in one of the searched locations or pass it explicitly:

```bash
CONVERTER_SCRIPT=/abs/path/to/mcap_to_lerobot_v3.py bash train/scripts/prepare_dataset.sh mcap-dir /path/to/mcap_dir
```

## 2. Check Config

Primary config:

- `train/config/lora.yaml`

Related references:

- `train/config/data.yaml`
- `train/config/train.yaml`
- `common/config/paths.yaml`

Relative paths inside these configs are resolved against the config file location.

## 3. Launch Training

Default local entrypoint:

```bash
bash train/scripts/train_lora.sh
```

Equivalent Python module command from repo root:

```bash
PYTHONPATH=common/src:train/src:deploy/src${PYTHONPATH:+:$PYTHONPATH} python -m pi05.train.cli.train --config train/config/lora.yaml
```

If `lerobot` is not installed as a package, you can prepend a local checkout manually:

```bash
PYTHONPATH=/abs/path/to/lerobot/src:common/src:train/src:deploy/src python -m pi05.train.cli.train --config train/config/lora.yaml
```

## 4. Outputs

Training artifacts are written under:

- `outputs/checkpoints/<run_name>`
- `outputs/logs`
- `outputs/exports/<run_name>`

TensorBoard logs live under `outputs/logs/tensorboard`.

After training finishes, the code automatically writes a deploy bundle to `outputs/exports/<run_name>`.

## 5. Manual Re-Export

```bash
RUN_DIR=outputs/checkpoints/pi05_grasp_generalization_v1 bash train/scripts/export_policy.sh
```

The deploy bundle includes:

- `adapter/`
- `experiment_config.yaml`
- `manifest.json`
- `normalizers.json`
