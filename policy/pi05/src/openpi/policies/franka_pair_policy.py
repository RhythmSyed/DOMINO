"""Franka-pair (RoboTwin/DOMINO franka-panda rig) transforms for pi0/pi0.5.

Mirror of aloha_policy.py for a 16-dim [L arm7, L grip, R arm7, R grip] joint
space (DOMINO's `joint_action/vector` order, same as the LEROBOT_DOMINO_FRANKA*
datasets). No ALOHA-specific joint flips or gripper remaps: the DOMINO sim
consumes raw joint targets, and the same dataset trains PUMA/SUAVE.
"""

import dataclasses
from typing import ClassVar

import einops
import numpy as np

from openpi import transforms

FRANKA_PAIR_DIM = 16


def make_franka_pair_example() -> dict:
    return {
        "state": np.ones((FRANKA_PAIR_DIM,)),
        "images": {
            "cam_high": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
            "cam_left_wrist": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
            "cam_right_wrist": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
        },
        "prompt": "do something",
    }


def _to_hwc(img):
    img = np.asarray(img)
    if np.issubdtype(img.dtype, np.floating):
        img = (255 * img).astype(np.uint8)
    return einops.rearrange(img, "c h w -> h w c")


@dataclasses.dataclass(frozen=True)
class FrankaPairInputs(transforms.DataTransformFn):
    """images: dict[name, img(c,h,w)]; state: [16]; actions: [horizon, 16] (train only)."""

    EXPECTED_CAMERAS: ClassVar[tuple[str, ...]] = ("cam_high", "cam_left_wrist", "cam_right_wrist")

    def __call__(self, data: dict) -> dict:
        in_images = {k: _to_hwc(v) for k, v in data["images"].items()}
        if set(in_images) - set(self.EXPECTED_CAMERAS):
            raise ValueError(f"Expected images in {self.EXPECTED_CAMERAS}, got {tuple(in_images)}")
        base_image = in_images["cam_high"]
        images = {"base_0_rgb": base_image}
        image_masks = {"base_0_rgb": np.True_}
        for dest, source in {"left_wrist_0_rgb": "cam_left_wrist", "right_wrist_0_rgb": "cam_right_wrist"}.items():
            if source in in_images:
                images[dest], image_masks[dest] = in_images[source], np.True_
            else:
                images[dest], image_masks[dest] = np.zeros_like(base_image), np.False_
        state = np.asarray(data["state"], dtype=np.float32)
        assert state.shape[-1] == FRANKA_PAIR_DIM, state.shape
        inputs = {"image": images, "image_mask": image_masks, "state": state}
        if "actions" in data:
            actions = np.asarray(data["actions"], dtype=np.float32)
            assert actions.shape[-1] == FRANKA_PAIR_DIM, actions.shape
            inputs["actions"] = actions
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]
        return inputs


@dataclasses.dataclass(frozen=True)
class FrankaPairOutputs(transforms.DataTransformFn):
    def __call__(self, data: dict) -> dict:
        # model action_dim is 32 (padded); the first 16 are the franka pair.
        return {"actions": np.asarray(data["actions"][:, :FRANKA_PAIR_DIM])}
