#!/bin/bash
# PUMA on LIBERO — local 3xA6000 launch. Mirrors examples/Robotwin's
# run_robotwin_train.sh but with LIBERO config + local resource settings:
#   3 GPUs x per-device 8 x grad-accum 5 = global batch 120 (~ StarVLA's
#   LIBERO global batch of 128), 30k steps, DeepSpeed ZeRO-2.
set -euo pipefail
set -x

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

export PYTHONPATH="${REPO_ROOT}"
export TORCH_DISTRIBUTED_DEFAULT_TIMEOUT=3600
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2}"
NUM_GPUS="${NUM_GPUS:-3}"
PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-8}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-5}"
MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-30000}"
CONFIG_YAML="${CONFIG_YAML:-${SCRIPT_DIR}/puma_train_libero.yaml}"

RUN_ID="${RUN_ID:-$(date +%Y%m%d)_libero_puma}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/results/Checkpoints/${RUN_ID}}"
mkdir -p "${OUTPUT_DIR}"
cp "${BASH_SOURCE[0]}" "${CONFIG_YAML}" "${OUTPUT_DIR}/"

export WANDB_PROJECT="${WANDB_PROJECT:-uwm-baseline}"
export WANDB_RUN_NAME="${WANDB_RUN_NAME:-suave_puma-scratch-libero-all-local}"
export WANDB_MODE="${WANDB_MODE:-online}"

TRAIN_SCRIPT="${REPO_ROOT}/PUMA/training/train_puma.py"

accelerate launch \
  --config_file "${REPO_ROOT}/PUMA/config/deepseeds/deepspeed_zero2.yaml" \
  --num_processes "${NUM_GPUS}" \
  "${TRAIN_SCRIPT}" \
  --config_yaml "${CONFIG_YAML}" \
  --run_id "${RUN_ID}" \
  --run_root_dir "${REPO_ROOT}/results/Checkpoints" \
  --datasets.vla_data.per_device_batch_size "${PER_DEVICE_BATCH_SIZE}" \
  --trainer.gradient_accumulation_steps "${GRAD_ACCUM_STEPS}" \
  --trainer.max_train_steps "${MAX_TRAIN_STEPS}" \
  --wandb_project "${WANDB_PROJECT}" \
  --wandb_entity "${WANDB_ENTITY:-tri}" \
  "$@"
