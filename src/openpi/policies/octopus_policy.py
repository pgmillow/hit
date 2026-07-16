import dataclasses

import einops
import numpy as np

from openpi import transforms


# Action layout (from data/convert_lerobot_v3_to_v2.py, 14-dim):
#   [0:6]  left_arm_cmd_pos    [6:12] right_arm_cmd_pos    [12] left_hand_cmd_pos    [13] right_hand_cmd_pos
# Only right arm + right hand (7 dims total) are used as model action input/output; left part is zeroed.
RIGHT_ACTION_DIMS = (6, 7, 8, 9, 10, 11, 13)

# State layout (26-dim, from convert_lerobot_v3_to_v2.py STATE_NAMES):
#   [0:6]  left_arm_qpos    [6:12] right_arm_qpos    [12] left_hand_qpos    [13] right_hand_qpos
#   [14:20] left_ee_pose    [20:26] right_ee_pose
# Only right side (right arm + right hand + right ee pose = 13 dims) is fed to the model; left side is zeroed.
RIGHT_STATE_DIMS = (6, 7, 8, 9, 10, 11, 13, 20, 21, 22, 23, 24, 25)


def _right_action_mask(num_dims: int) -> np.ndarray:
    mask = np.zeros(num_dims, dtype=np.float32)
    mask[list(RIGHT_ACTION_DIMS)] = 1.0
    return mask


def _right_state_mask(num_dims: int) -> np.ndarray:
    mask = np.zeros(num_dims, dtype=np.float32)
    mask[list(RIGHT_STATE_DIMS)] = 1.0
    return mask


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.ndim == 3 and image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class OctopusInputs(transforms.DataTransformFn):
    """Inputs for Octopus MCAP datasets converted to LeRobot format.

    By default (``bimanual=False``) left-arm / left-hand / left-ee dims are zeroed so the
    model only learns the right side (legacy single-arm pour setups).
    Set ``bimanual=True`` to keep the full 26-D state and 14-D action.
    """

    bimanual: bool = False

    def __call__(self, data: dict) -> dict:
        images = data["images"]

        state = np.asarray(data["state"], dtype=np.float32)
        if not self.bimanual:
            # Zero out left arm / left hand / left ee pose so only the 13 right-side state dims
            # are fed to the model.
            state = state * _right_state_mask(state.shape[-1])

        inputs = {
            "state": state,
            "image": {
                "base_0_rgb": _parse_image(images["top"]),
                "left_wrist_0_rgb": _parse_image(images["left_wrist"]),
                "right_wrist_0_rgb": _parse_image(images["right_wrist"]),
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.True_,
            },
        }

        if "actions" in data:
            actions = np.asarray(data["actions"], dtype=np.float32)
            if not self.bimanual:
                # Keep only right arm + right hand; zero out left arm and left_hand so the model
                # only sees/learns the 7 right-side action dims (others are pad 0).
                actions = actions * _right_action_mask(actions.shape[-1])
            inputs["actions"] = actions

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class OctopusOutputs(transforms.DataTransformFn):
    """Outputs for Octopus policy inference."""

    action_dim: int = 14
    bimanual: bool = False

    def __call__(self, data: dict) -> dict:
        actions = np.asarray(data["actions"][..., : self.action_dim])
        if not self.bimanual:
            # Zero out left arm and left_hand so only the 7 right-side dims are returned.
            actions = actions * _right_action_mask(self.action_dim)
        return {"actions": actions}
