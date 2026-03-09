#!/usr/bin/env bash
set -euo pipefail

# Run concept perturbation for PIDNet/flood using only class-1 concepts.
# Outputs are saved under:
#   results/instance_perturbation/flood/pidnet/${REL_INIT}_${RESULT_TAG}/
#
# Usage:
#   bash scripts/run_instance_perturbation_only_flood.sh
#   NUM_SAMPLES=120 RESULT_TAG=only_flood_120 bash scripts/run_instance_perturbation_only_flood.sh
#   NUM_SAMPLES=120 RESULT_TAG=only_flood_120 RUN_INSERTION=1 bash scripts/run_instance_perturbation_only_flood.sh

CODE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$CODE_DIR"

MODEL_NAME="${MODEL_NAME:-pidnet}"
DATASET_NAME="${DATASET_NAME:-flood}"
REL_INIT="${REL_INIT:-ones}"
RESULT_TAG="${RESULT_TAG:-only_flood_120}"
NUM_SAMPLES="${NUM_SAMPLES:-120}"
BATCH_SIZE="${BATCH_SIZE:-2}"
NUM_WORKERS="${NUM_WORKERS:-0}"
RUN_INSERTION="${RUN_INSERTION:-1}"

OUT_BASE="results/instance_perturbation/${DATASET_NAME}/${MODEL_NAME}/${REL_INIT}_${RESULT_TAG}"
LOG_DIR="${OUT_BASE}/run_logs"
mkdir -p "$LOG_DIR"

# Use class-1 concept files as canonical layer list for strict flood-only concepts.
mapfile -t LAYERS < <(
  ls "results/global_class_concepts/${DATASET_NAME}/${MODEL_NAME}/${REL_INIT}"/*_class_1.pth 2>/dev/null \
    | sed -E 's#^.*/##' \
    | sed -E 's/_class_1\.pth$//' \
    | sort -u
)

if [ "${#LAYERS[@]}" -eq 0 ]; then
  echo "No class-1 concept files found under results/global_class_concepts/${DATASET_NAME}/${MODEL_NAME}/${REL_INIT}"
  exit 1
fi

: > "${LOG_DIR}/flip_success_layers.txt"
: > "${LOG_DIR}/flip_failed_layers.txt"
: > "${LOG_DIR}/insertion_success_layers.txt"
: > "${LOG_DIR}/insertion_failed_layers.txt"

for layer in "${LAYERS[@]}"; do
  echo "Running flip layer: ${layer}"
  if python3 -m experiments.instance_perturbation_only_flood \
      --model_name "${MODEL_NAME}" \
      --dataset_name "${DATASET_NAME}" \
      --layer_name "${layer}" \
      --rel_init "${REL_INIT}" \
      --result_tag "${RESULT_TAG}" \
      --num_samples "${NUM_SAMPLES}" \
      --batch_size "${BATCH_SIZE}" \
      --num_workers "${NUM_WORKERS}" \
      --insertion False > "${LOG_DIR}/last_flip.log" 2>&1; then
    echo "${layer}" >> "${LOG_DIR}/flip_success_layers.txt"
  else
    echo "${layer}" >> "${LOG_DIR}/flip_failed_layers.txt"
    tail -n 20 "${LOG_DIR}/last_flip.log" || true
  fi

  if [ "${RUN_INSERTION}" = "1" ]; then
    echo "Running insertion layer: ${layer}"
    if python3 -m experiments.instance_perturbation_only_flood \
        --model_name "${MODEL_NAME}" \
        --dataset_name "${DATASET_NAME}" \
        --layer_name "${layer}" \
        --rel_init "${REL_INIT}" \
        --result_tag "${RESULT_TAG}" \
        --num_samples "${NUM_SAMPLES}" \
        --batch_size "${BATCH_SIZE}" \
        --num_workers "${NUM_WORKERS}" \
        --insertion True > "${LOG_DIR}/last_insertion.log" 2>&1; then
      echo "${layer}" >> "${LOG_DIR}/insertion_success_layers.txt"
    else
      echo "${layer}" >> "${LOG_DIR}/insertion_failed_layers.txt"
      tail -n 20 "${LOG_DIR}/last_insertion.log" || true
    fi
  fi
done

echo "Done."
echo "Flip success:      $(wc -l < "${LOG_DIR}/flip_success_layers.txt")"
echo "Flip failed:       $(wc -l < "${LOG_DIR}/flip_failed_layers.txt")"
if [ "${RUN_INSERTION}" = "1" ]; then
  echo "Insertion success: $(wc -l < "${LOG_DIR}/insertion_success_layers.txt")"
  echo "Insertion failed:  $(wc -l < "${LOG_DIR}/insertion_failed_layers.txt")"
fi
echo "Outputs: ${OUT_BASE}"
