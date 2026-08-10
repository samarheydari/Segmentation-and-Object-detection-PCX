#!/usr/bin/env bash
set -euo pipefail

# Rerun only stale flip layers reported by audit JSON.
#
# Usage:
#   bash scripts/rerun_stale_flip_only_flood.sh
#   RESULT_TAG=only_flood_120 NUM_SAMPLES=120 bash scripts/rerun_stale_flip_only_flood.sh

CODE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$CODE_DIR"

MODEL_NAME="${MODEL_NAME:-pidnet}"
DATASET_NAME="${DATASET_NAME:-flood}"
REL_INIT="${REL_INIT:-ones}"
RESULT_TAG="${RESULT_TAG:-only_flood_120}"
NUM_SAMPLES="${NUM_SAMPLES:-120}"
BATCH_SIZE="${BATCH_SIZE:-2}"
NUM_WORKERS="${NUM_WORKERS:-0}"

OUT_BASE="results/instance_perturbation/${DATASET_NAME}/${MODEL_NAME}/${REL_INIT}_${RESULT_TAG}"
DATA_DIR="${OUT_BASE}/data"
LOG_DIR="${OUT_BASE}/run_logs"
mkdir -p "$LOG_DIR"

AUDIT_JSON="${DATA_DIR}/audit_${RESULT_TAG}.json"
if [ ! -f "$AUDIT_JSON" ]; then
  echo "Missing audit file: $AUDIT_JSON"
  echo "Run audit first:"
  echo "python3 -m experiments.audit_only_flood_perturbation --model_name ${MODEL_NAME} --dataset_name ${DATASET_NAME} --rel_init ${REL_INIT} --result_tag ${RESULT_TAG} --expected_samples ${NUM_SAMPLES} --strict_min_samples ${NUM_SAMPLES}"
  exit 1
fi

mapfile -t STALE_LAYERS < <(
  python3 - <<PY
import json
with open("${AUDIT_JSON}", "r") as f:
    data = json.load(f)
for layer in data["result_integrity"]["stale_flip_layers_(n<expected_samples)"]:
    print(layer)
PY
)

if [ "${#STALE_LAYERS[@]}" -eq 0 ]; then
  echo "No stale flip layers found in ${AUDIT_JSON}"
  exit 0
fi

: > "${LOG_DIR}/stale_flip_rerun_success_layers.txt"
: > "${LOG_DIR}/stale_flip_rerun_failed_layers.txt"

for layer in "${STALE_LAYERS[@]}"; do
  echo "Rerun stale flip layer: ${layer}"
  if python3 -m experiments.instance_perturbation_only_flood \
      --model_name "${MODEL_NAME}" \
      --dataset_name "${DATASET_NAME}" \
      --layer_name "${layer}" \
      --rel_init "${REL_INIT}" \
      --result_tag "${RESULT_TAG}" \
      --num_samples "${NUM_SAMPLES}" \
      --batch_size "${BATCH_SIZE}" \
      --num_workers "${NUM_WORKERS}" \
      --insertion False > "${LOG_DIR}/last_stale_flip.log" 2>&1; then
    echo "${layer}" >> "${LOG_DIR}/stale_flip_rerun_success_layers.txt"
  else
    echo "${layer}" >> "${LOG_DIR}/stale_flip_rerun_failed_layers.txt"
    tail -n 20 "${LOG_DIR}/last_stale_flip.log" || true
  fi
done

echo "Stale flip rerun done."
echo "success: $(wc -l < "${LOG_DIR}/stale_flip_rerun_success_layers.txt")"
echo "failed:  $(wc -l < "${LOG_DIR}/stale_flip_rerun_failed_layers.txt")"
