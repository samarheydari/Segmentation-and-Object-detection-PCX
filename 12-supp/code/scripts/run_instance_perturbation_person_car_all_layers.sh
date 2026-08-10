#!/usr/bin/env bash
set -uo pipefail

CODE_ROOT="/home/heydari/paper/12-supp/code"
EXP_DIR="${CODE_ROOT}/experiments"
PYTHON_BIN="${PYTHON_BIN:-/usr/bin/python3}"

MODEL_NAME="${MODEL_NAME:-yolov6s6}"
DATASET_NAME="${DATASET_NAME:-person_car}"
REL_INIT="${REL_INIT:-logits}"
NUM_SAMPLES="${NUM_SAMPLES:-100}"
BATCH_SIZE="${BATCH_SIZE:-10}"

# Optional: set RUN_INSERTION=1 to run both deletion and insertion modes.
RUN_INSERTION="${RUN_INSERTION:-0}"
RESUME="${RESUME:-1}"
STOP_ON_ERROR="${STOP_ON_ERROR:-0}"

cd "${EXP_DIR}"

BASE_DIR="results/global_class_concepts/${DATASET_NAME}/${MODEL_NAME}/${REL_INIT}"
if [[ ! -d "${BASE_DIR}" ]]; then
  echo "Concept directory not found: ${BASE_DIR}"
  exit 1
fi

mapfile -t LAYERS < <(
  find "${BASE_DIR}" -maxdepth 1 -type f -name '*_class_0.pth' \
    -printf '%f\n' \
    | sed -E 's/_class_0\.pth$//' \
    | sort -u
)

if [[ ${#LAYERS[@]} -eq 0 ]]; then
  echo "No layer concept files found under: ${BASE_DIR}"
  exit 1
fi

echo "Found ${#LAYERS[@]} layers."

OUT_DIR="results/instance_perturbation/${DATASET_NAME}/${MODEL_NAME}/data"
mkdir -p "${OUT_DIR}"
FAILED_LOG="${OUT_DIR}/failed_layers.log"
touch "${FAILED_LOG}"

for layer in "${LAYERS[@]}"; do
  if [[ "${RESUME}" == "1" && -f "${OUT_DIR}/instance_perturbation_${layer}.pth" ]]; then
    echo "Skipping completed layer: ${layer}"
    continue
  fi

  echo "Running layer: ${layer}"
  if ! "${PYTHON_BIN}" "${EXP_DIR}/instance_perturbation_od.py" \
    --model_name "${MODEL_NAME}" \
    --dataset_name "${DATASET_NAME}" \
    --layer_name "${layer}" \
    --num_samples "${NUM_SAMPLES}" \
    --batch_size "${BATCH_SIZE}" \
    --insertion True; then
    echo "${layer}" | tee -a "${FAILED_LOG}"
    echo "Layer failed (logged): ${layer}"
    if [[ "${STOP_ON_ERROR}" == "1" ]]; then
      exit 1
    fi
    continue
  fi

  if [[ "${RUN_INSERTION}" == "1" ]]; then
    if ! "${PYTHON_BIN}" "${EXP_DIR}/instance_perturbation_od.py" \
      --model_name "${MODEL_NAME}" \
      --dataset_name "${DATASET_NAME}" \
      --layer_name "${layer}" \
      --num_samples "${NUM_SAMPLES}" \
      --batch_size "${BATCH_SIZE}" \
      --insertion True; then
      echo "${layer} (insertion)" | tee -a "${FAILED_LOG}"
      echo "Layer failed in insertion mode (logged): ${layer}"
      if [[ "${STOP_ON_ERROR}" == "1" ]]; then
        exit 1
      fi
    fi
  fi
done
