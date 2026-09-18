"""xArm7 / GELLO real-task transforms for pi0.5 (SUAVE real-robot baseline).

Mirror of franka_pair_policy.py for the 7-dim SUAVE real-task action space
[dx, dy, dz, d_axis_angle(3), gripper] (world-frame per-step deltas, native
30 Hz, gripper 0=open/1=closed) and 7-dim state [ee_xyz, ee_axis_angle,
measured_gripper_position]. Two cameras: cam_high (front RealSense) and
cam_left_wrist (wrist RealSense); the right-wrist slot is masked.

Actions in the LEROBOT_REAL_<task> datasets are ALREADY deltas, so the data
config must not push DeltaActions/AbsoluteActions.
"""

import dataclasses
from typing import ClassVar

import einops
import numpy as np

from openpi import transforms

XARM_DIM = 7


def make_xarm_example() -> dict:
    return {
        "state": np.ones((XARM_DIM,)),
        "images": {
            "cam_high": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
            "cam_left_wrist": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
        },
        "prompt": "do something",
    }


def _to_hwc(img):
    img = np.asarray(img)
    if np.issubdtype(img.dtype, np.floating):
        img = (255 * img).astype(np.uint8)
    return einops.rearrange(img, "c h w -> h w c")


@dataclasses.dataclass(frozen=True)
class XArmInputs(transforms.DataTransformFn):
    """images: dict[name, img(c,h,w)]; state: [7]; actions: [horizon, 7] (train only)."""

    EXPECTED_CAMERAS: ClassVar[tuple[str, ...]] = ("cam_high", "cam_left_wrist")

    def __call__(self, data: dict) -> dict:
        in_images = {k: _to_hwc(v) for k, v in data["images"].items()}
        if set(in_images) - set(self.EXPECTED_CAMERAS):
            raise ValueError(f"Expected images in {self.EXPECTED_CAMERAS}, got {tuple(in_images)}")
        base_image = in_images["cam_high"]
        images = {"base_0_rgb": base_image}
        image_masks = {"base_0_rgb": np.True_}
        if "cam_left_wrist" in in_images:
            images["left_wrist_0_rgb"], image_masks["left_wrist_0_rgb"] = in_images["cam_left_wrist"], np.True_
        else:
            images["left_wrist_0_rgb"], image_masks["left_wrist_0_rgb"] = np.zeros_like(base_image), np.False_
        images["right_wrist_0_rgb"], image_masks["right_wrist_0_rgb"] = np.zeros_like(base_image), np.False_
        state = np.asarray(data["state"], dtype=np.float32)
        assert state.shape[-1] == XARM_DIM, state.shape
        inputs = {"image": images, "image_mask": image_masks, "state": state}
        if "actions" in data:
            actions = np.asarray(data["actions"], dtype=np.float32)
            assert actions.shape[-1] == XARM_DIM, actions.shape
            inputs["actions"] = actions
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]
        return inputs


@dataclasses.dataclass(frozen=True)
class XArmOutputs(transforms.DataTransformFn):
    def __call__(self, data: dict) -> dict:
        # model action_dim is 32 (padded); the first 7 are the xArm deltas + gripper.
        return {"actions": np.asarray(data["actions"][:, :XARM_DIM])}
