"""
DOMINO policy module for a π0/π0.5 served by openpi's websocket policy server.

Mirrors policy/pi05/deploy_policy.py (the authors' in-process π0.5 adapter) but
talks to `scripts/serve_policy.py` over openpi-client, so the JAX stack lives in
its own venv (/venv/openpi) and the sim keeps the debugged /opt/domino venv —
the same two-process shape as the PUMA and SUAVE DOMINO evals.

Observation → openpi input (FrankaPairInputs / AlohaInputs contract):
    images: {cam_high, cam_left_wrist, cam_right_wrist} uint8 (3, H, W)
    state:  observation["joint_action"]["vector"]  = [L arm, L grip, R arm, R grip]
    prompt: TASK_ENV.get_instruction()
Action: server returns (action_horizon, D) absolute joint targets in the same
vector order; the first `pi0_step` are executed open-loop with take_action('qpos')
exactly like the authors' deploy_policy.py (their deploy_policy.yml: pi0_step 50).

Loaded by script/eval_policy.py via --policy_name pi05_client_policy (module on
PYTHONPATH); usr_args come from the yml + --overrides.
"""

import numpy as np

from openpi_client import websocket_client_policy


class Pi05Client:
    def __init__(self, host, port, pi0_step):
        self.client = websocket_client_policy.WebsocketClientPolicy(host=host, port=port)
        self.pi0_step = int(pi0_step)
        self.instruction = None
        self.n_infer = 0
        print(f"[pi05_client] connected {host}:{port} | server metadata: "
              f"{self.client.get_server_metadata()} | pi0_step={self.pi0_step}", flush=True)

    def infer(self, observation):
        obs = observation["observation"]
        chw = lambda img: np.ascontiguousarray(np.transpose(np.asarray(img, dtype=np.uint8), (2, 0, 1)))
        payload = {
            "state": np.asarray(observation["joint_action"]["vector"], dtype=np.float32),
            "images": {
                "cam_high": chw(obs["head_camera"]["rgb"]),
                "cam_left_wrist": chw(obs["left_camera"]["rgb"]),
                "cam_right_wrist": chw(obs["right_camera"]["rgb"]),
            },
            "prompt": self.instruction,
        }
        actions = np.asarray(self.client.infer(payload)["actions"], dtype=np.float64)
        self.n_infer += 1
        return actions


def get_model(usr_args):
    return Pi05Client(host=usr_args.get("host", "127.0.0.1"),
                      port=int(usr_args.get("port", 8000)),
                      pi0_step=usr_args.get("pi0_step", 50))


def eval(TASK_ENV, model, observation):
    if model.instruction is None:
        model.instruction = str(TASK_ENV.get_instruction())
    actions = model.infer(observation)
    state_dim = len(observation["joint_action"]["vector"])
    assert actions.shape[1] >= state_dim, (actions.shape, state_dim)
    for action in actions[: model.pi0_step]:
        TASK_ENV.take_action(action[:state_dim])
        if TASK_ENV.eval_success or TASK_ENV.take_action_cnt >= TASK_ENV.step_lim:
            break


def reset_model(model):
    model.instruction = None
    try:
        model.client.reset()
    except Exception:
        pass
