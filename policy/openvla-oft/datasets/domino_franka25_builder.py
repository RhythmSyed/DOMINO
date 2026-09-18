"""
RLDS builder: DOMINO 25 single-arm tasks, franka-panda pair, for OpenVLA-OFT (SUAVE same-data baseline).

Reads the DOMINO release zips DIRECTLY (datasets/domino/dataset/<task>/franka-panda_*.zip),
no intermediate HDF5s (the ALOHA preprocess step would need ~1 TB of uncompressed images).
Same episode set as SUAVE / PUMA* / pi* (single-arm filter: exactly one arm with total
|dq| > 0.5 rad -> 18,231 episodes), same OFT-ALOHA conventions as move_can_pot_builder.py:
    state_t  = joint_action/vector[t]      (16: L arm7, L grip, R arm7, R grip)
    action_t = joint_action/vector[t+1]    (absolute joint targets; T-1 steps)
    images   = head / left wrist / right wrist, 320x240 -> 256x256 bicubic, JPEG-encoded
    language = the episode's first "seen" instruction (as for the LeRobot/UWM conversions)
No low_cam (front) view: the OFT loader with num_images_in_input=3 uses primary+left+right.

Build (local openvla env; TFDS_DATA_DIR selects the output root):
    cd third_party/domino/policy/openvla-oft
    DOMINO_DATASET_DIR=~/Desktop/suave/datasets/domino/dataset TFDS_DATA_DIR=~/Desktop/suave/outputs/rlds \\
        python -m datasets.domino_franka25_builder
"""
from typing import Any, Iterator, Tuple
import glob
import io
import json
import os
import re
import zipfile

import h5py
import numpy as np
import tensorflow_datasets as tfds
from PIL import Image

from datasets.conversion_utils import MultiThreadedDatasetBuilder

EMBODIMENT = "franka-panda"
ZIP_RE = re.compile(rf"{EMBODIMENT}_(clean|randomized)_level(\d)_(\d+)\.zip$")
IMG = 256
TASKS25 = [  # == run_parallel_domino_eval.TRAINED_TASKS / LEROBOT_DOMINO_FRANKA task dirs
    "adjust_bottle", "beat_block_hammer", "click_alarmclock", "click_bell", "dump_bin_bigbin",
    "move_can_pot", "move_pillbottle_pad", "move_playingcard_away", "move_stapler_pad",
    "place_a2b_left", "place_a2b_right", "place_bread_basket", "place_container_plate",
    "place_empty_cup", "place_fan", "place_mouse_pad", "place_object_scale",
    "place_object_stand", "place_phone_stand", "place_shoe", "press_stapler",
    "rotate_qrcode", "shake_bottle", "shake_bottle_horizontally", "stamp_seal",
]


def _active_arm(h, min_motion=0.5):
    moving = [s for s in ("left", "right") if np.abs(np.diff(h[f"joint_action/{s}_arm"][:], axis=0)).sum() > min_motion]
    return moving[0] if len(moving) == 1 else None


def _resize(raw_jpegs):
    out = np.empty((len(raw_jpegs), IMG, IMG, 3), dtype=np.uint8)
    for i, b in enumerate(raw_jpegs):
        out[i] = np.asarray(Image.open(io.BytesIO(bytes(b))).convert("RGB").resize((IMG, IMG), Image.BICUBIC))
    return out


def _generate_examples(paths) -> Iterator[Tuple[str, Any]]:
    """One work unit = one zip (50 or 200 episodes)."""
    for zp in paths:
        m = ZIP_RE.search(os.path.basename(zp)); task = os.path.basename(os.path.dirname(zp))
        prefix = f"{EMBODIMENT}_{m.group(1)}_level{m.group(2)}"
        with zipfile.ZipFile(zp) as zf:
            eps = sorted(n.split("/")[-1].removesuffix(".hdf5") for n in zf.namelist() if n.endswith(".hdf5"))
            for ep in eps:
                try:
                    with h5py.File(io.BytesIO(zf.read(f"{prefix}/data/{ep}.hdf5"))) as h:
                        if _active_arm(h) is None:
                            continue
                        vec = h["joint_action/vector"][:].astype(np.float32)          # (T, 16)
                        head = h["observation/head_camera/rgb"][:]
                        left = h["observation/left_camera/rgb"][:]
                        right = h["observation/right_camera/rgb"][:]
                    T = min(len(vec), len(head), len(left), len(right))
                    if T < 2:
                        continue
                    instr = json.loads(zf.read(f"{prefix}/instructions/{ep}.json"))["seen"][0]
                except (KeyError, OSError) as e:
                    print(f"[skip] {zp}:{ep}: {type(e).__name__}: {e}", flush=True)
                    continue
                states, actions = vec[: T - 1], vec[1:T]
                head_i, left_i, right_i = _resize(head[: T - 1]), _resize(left[: T - 1]), _resize(right[: T - 1])
                n = T - 1
                steps = [{
                    "observation": {"image": head_i[i], "left_wrist_image": left_i[i], "right_wrist_image": right_i[i],
                                    "state": states[i]},
                    "action": actions[i],
                    "discount": np.float32(1.0),
                    "reward": np.float32(1.0 if i == n - 1 else 0.0),
                    "is_first": np.bool_(i == 0), "is_last": np.bool_(i == n - 1), "is_terminal": np.bool_(i == n - 1),
                    "language_instruction": instr,
                } for i in range(n)]
                key = f"{task}/{prefix}/{ep}"
                yield key, {"steps": steps, "episode_metadata": {"file_path": key, "task": task,
                                                                  "scene": m.group(1), "dynamic_level": int(m.group(2))}}


class domino_franka25(MultiThreadedDatasetBuilder):
    VERSION = tfds.core.Version("1.0.0")
    RELEASE_NOTES = {"1.0.0": "DOMINO 25 single-arm tasks, franka-panda pair, all scenes/levels (18,231 eps)."}
    N_WORKERS = int(os.environ.get("RLDS_WORKERS", "24"))
    MAX_PATHS_IN_MEMORY = 8          # one path = one zip (50-200 episodes)
    PARSE_FCN = _generate_examples

    def _info(self) -> tfds.core.DatasetInfo:
        img = lambda: tfds.features.Image(shape=(IMG, IMG, 3), dtype=np.uint8, encoding_format="jpeg")
        return self.dataset_info_from_configs(
            features=tfds.features.FeaturesDict({
                "steps": tfds.features.Dataset({
                    "observation": tfds.features.FeaturesDict({
                        "image": img(), "left_wrist_image": img(), "right_wrist_image": img(),
                        "state": tfds.features.Tensor(shape=(16,), dtype=np.float32),
                    }),
                    "action": tfds.features.Tensor(shape=(16,), dtype=np.float32),
                    "discount": tfds.features.Scalar(dtype=np.float32),
                    "reward": tfds.features.Scalar(dtype=np.float32),
                    "is_first": tfds.features.Scalar(dtype=np.bool_),
                    "is_last": tfds.features.Scalar(dtype=np.bool_),
                    "is_terminal": tfds.features.Scalar(dtype=np.bool_),
                    "language_instruction": tfds.features.Text(),
                }),
                "episode_metadata": tfds.features.FeaturesDict({
                    "file_path": tfds.features.Text(), "task": tfds.features.Text(),
                    "scene": tfds.features.Text(), "dynamic_level": tfds.features.Scalar(dtype=np.int64),
                }),
            }))

    def _split_paths(self):
        root = os.environ["DOMINO_DATASET_DIR"]
        zips = sorted(z for t in TASKS25 for z in glob.glob(os.path.join(root, t, f"{EMBODIMENT}_*.zip")) if ZIP_RE.search(z))
        lim = int(os.environ.get("RLDS_ZIP_LIMIT", "0"))      # smoke builds: first N zips only (separate TFDS_DATA_DIR!)
        if lim:
            zips = zips[:lim]
        else:
            assert len(zips) == 150, f"expected 25 tasks x 6 zips, found {len(zips)}"
        print(f"[INFO] {len(zips)} zips over {len(TASKS25)} tasks")
        return {"train": zips}


if __name__ == "__main__":
    domino_franka25().download_and_prepare()
