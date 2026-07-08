# Deployment Workflow

The real robot deployment entry only reads an exported bundle:

```text
outputs/exports/<run_name>
```

It does not accept `dataset_path` or `final_adapter` directly. Debug scripts can
add fast local checks later, but robot deployment should stay bundle-only.

## Layout

```text
common/src/pi05/common/
├── data/                  # image/state/action codecs and normalization helpers
├── robot/                 # action layout and joint limit primitives
└── ros/                   # Pi0.5 topic naming helpers

deploy/src/pi05/deploy/
├── config/                # deploy.yaml schema
├── models/                # bundle-only policy loader
├── runtime/               # shared buffer, inference worker, control loop, safety guard
├── ros_nodes/             # pi05_vla_deploy_node and pi05_bridge_node
└── cli/                   # console entrypoints
```

## 1. Export a Bundle

Training exports a deployment bundle automatically to:

```text
outputs/exports/<run_name>
```

You can also re-export it manually:

```bash
RUN_DIR=outputs/checkpoints/pi05_grasp_generalization_v1 bash train/scripts/export_policy.sh
```

Optional packaging step:

```bash
bash deploy/scripts/package_model.sh /abs/path/to/deploy_bundle
```

## 2. Check Deployment Config

Deployment config lives at:

- `deploy/config/deploy.yaml`

This file captures all deployment-facing parameters:

- `bundle.bundle_dir`
- `runtime.mode`: `dry-run`, `shadow-run`, or `safe-run`
- `runtime.inference_hz`, `runtime.control_hz`, `runtime.chunk_size`, `runtime.execute_horizon`
- `runtime.prefetch_steps`, `runtime.blend_steps`, `runtime.max_action_age_sec`
- `runtime.max_inference_requests`, `runtime.max_pending_chunks`, `runtime.fallback_policy`
- `image.transport`: `raw`, `compressed`, or `both`
- `runtime.device`, `runtime.secondary_device`, `runtime.multi_gpu_strategy`
- `topics.observation.*`, `topics.command.*`, `topics.bridge_output.*`
- `safety.*`
- `bridge.*`

## 3. Run Modes

`dry-run` loads the model, receives observations, infers actions, and logs commands without publishing command topics.

`shadow-run` publishes `/pi05_vla/command/*` topics but does not require the bridge to forward commands to hardware.

`safe-run` publishes `/pi05_vla/command/*` for the bridge to consume. The bridge is still separately controlled by `bridge.enabled` and `bridge.publish_to_picotele`.

## 4. Rolling Prediction Runtime

The deploy runtime uses asynchronous rolling prediction:

```text
ROS callbacks -> ObservationCollector -> SharedBuffer.latest_observation
ControlLoop   -> inference_request_queue -> InferenceWorker
InferenceWorker -> chunk_result_queue -> ControlLoop
ControlLoop   -> /pi05_vla/command/*
```

Both `inference_request_queue` and `chunk_result_queue` are bounded latest-only queues. Their default `maxsize` is `1`, so stale observations and stale chunks are dropped instead of accumulating latency.

The control loop executes `active_chunk` at `runtime.control_hz`. When `active_cursor >= execute_horizon - prefetch_steps`, it submits a new inference request using the latest observation. The inference worker runs in the background and never publishes robot commands.

When a `pending_chunk` arrives, it waits for the execute-horizon boundary. The new chunk starts from `ActionChunk.aligned_index(now)` instead of blindly starting from action `0`. The transition uses smoothstep blending for `runtime.blend_steps` ticks:

```text
alpha = 3 * s^2 - 2 * s^3
cmd = (1 - alpha) * blend_start_command + alpha * new_action
```

All final commands, including blended commands and fallback holds, still pass through `SafetyGuard`.

Recommended first-pass values:

```yaml
runtime:
  control_hz: 30
  inference_hz: 10
  chunk_size: 30
  execute_horizon: 10
  prefetch_steps: 5
  blend_steps: 3
  max_action_age_sec: 0.45
  max_inference_requests: 1
  max_pending_chunks: 1
  fallback_policy: hold_last_action
```

## 5. Image Transport

Deployment defaults to `image.transport: raw`. Raw images avoid compressed-image decode latency on the deployment machine. Use `compressed` only when camera bandwidth is the bottleneck, or `both` while transitioning topic publishers.

## 6. Start Deployment

```bash
source /opt/ros/jazzy/setup.bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate lerobot312
cd /home/hit/pi05_ws/projects/pi05
bash deploy/scripts/run_inference.sh
```

Use a different config file:

```bash
bash deploy/scripts/run_inference.sh deploy/config/deploy.yaml
```

Start the optional bridge in another terminal:

```bash
source /opt/ros/jazzy/setup.bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate lerobot312
cd /home/hit/pi05_ws/projects/pi05
bash deploy/scripts/run_bridge.sh
```

## 7. Command Topics

The policy runtime publishes only Pi0.5-owned topics:

```text
/pi05_vla/command/left_arm/joint_target
/pi05_vla/command/right_arm/joint_target
/pi05_vla/command/left_hand/target
/pi05_vla/command/right_hand/target
/pi05_vla/status
/pi05_vla/metrics
```

`pi05_bridge_node` is the only place that adapts those commands to an existing execution stack. It performs topic adaptation, hand command conversion, NaN/Inf rejection, and max-step limiting; it does not run model inference.

For current deployment, choose the active GPU with `runtime.device` such as `cuda:0` or `cuda:1`. `runtime.multi_gpu_strategy` is reserved and guarded so future model sharding is explicit instead of silently changing runtime behavior.

## 8. Verification

```bash
python -m compileall common/src deploy/src tests/deploy
python -m pytest tests/train/test_config.py tests/deploy/test_deploy_config.py
```
