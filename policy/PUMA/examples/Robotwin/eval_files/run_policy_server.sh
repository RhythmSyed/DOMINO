#!/bin/bash
export PYTHONPATH=$(pwd):${PYTHONPATH}
export puma_python=/home/rhythmsyed/miniconda3/envs/puma/bin/python
your_ckpt=/home/rhythmsyed/Desktop/suave/third_party/domino/policy/PUMA/playground/Pretrained_models/PUMA-ckpt/checkpoints/steps_100000_pytorch_model.pt
gpu_id=0
port=9001

CUDA_VISIBLE_DEVICES=$gpu_id ${puma_python} deployment/model_server/server_policy.py \
    --ckpt_path ${your_ckpt} \
    --port ${port} \
    --use_bf16
