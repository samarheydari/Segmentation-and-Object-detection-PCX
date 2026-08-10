#!/usr/bin/env bash
set -euo pipefail

cd /home/heydari/paper/LCRP
export PERSON_CAR_DATA_ROOT=/home/heydari/FHHI-XAI/data/person_car_detection_data/original_BRK
# Required import roots for this mixed layout:
# - experiments/datasets/models at /home/heydari/paper/LCRP
# - utils.* under /home/heydari/paper/LCRP/LCRP
# - yolov6 package under /home/heydari/paper
export PYTHONPATH="/home/heydari/paper/LCRP:/home/heydari/paper/LCRP/LCRP:/home/heydari/paper:${PYTHONPATH:-}"
PYTHON_BIN="${PYTHON_BIN:-/home/heydari/miniconda3/envs/v-310/bin/python}"
CKPT_SRC="${CKPT_SRC:-/home/heydari/paper/12-supp/code/models/yolob6s_ckpt.pt}"
CKPT_DST="/home/heydari/paper/LCRP/models/yolob6s_ckpt.pt"

mkdir -p "$(dirname "${CKPT_DST}")"
ln -sfn "${CKPT_SRC}" "${CKPT_DST}"

for class_id in 0 1; do
  "${PYTHON_BIN}" -m experiments.global_class_concepts \
    --model_name yolov6s6 \
    --dataset_name person_car \
    --class_id "${class_id}" \
    --batch_size 4 \
    --rel_init ones
done
