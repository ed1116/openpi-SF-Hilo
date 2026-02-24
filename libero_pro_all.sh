#!/usr/bin/env bash
set -euo pipefail

#[COPILOT] Sweep script for 20 LIBERO-PRO evaluations:
#[COPILOT] 4 base suites x (base + swap + object + language + task).

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_PATH="${ROOT_DIR}/third_party/LIBERO-PRO/evaluation_config.yaml"
MAIN_PY="${ROOT_DIR}/examples/libero-pro/main.py"

#[COPILOT] Runtime knobs (override via env vars when needed).
PY_BIN="${PY_BIN:-${ROOT_DIR}/examples/libero-pro/.venv-pro/bin/python}" #[COPILOT] Prefer libero-pro venv interpreter by default.
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8001}"
NUM_TRIALS="${NUM_TRIALS:-10}"
NUM_STEPS_WAIT="${NUM_STEPS_WAIT:-10}"
REPLAN_STEPS="${REPLAN_STEPS:-5}"
RESIZE_SIZE="${RESIZE_SIZE:-224}"
SEED="${SEED:-7}"
VIDEO_ROOT="${VIDEO_ROOT:-${ROOT_DIR}/data/libero-pro/videos}"
LOG_ROOT="${LOG_ROOT:-${ROOT_DIR}/data/libero-pro/logs}"
DRY_RUN="${DRY_RUN:-0}"
#[COPILOT] Resume support: 1-based run index (1=libero_spatial base, 2=libero_spatial swap, ...).
START_RUN_IDX="${START_RUN_IDX:-1}"

#[COPILOT] Ensure LIBERO-PRO benchmark package is imported first so *_temp/*_swap suites are registered.
export PYTHONPATH="${ROOT_DIR}/third_party/LIBERO-PRO:${PYTHONPATH:-}"
export LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-${ROOT_DIR}/third_party/LIBERO-PRO/.libero}" #[COPILOT] Use LIBERO-PRO-specific LIBERO config directory.

mkdir -p "${VIDEO_ROOT}" "${LOG_ROOT}"

if [[ ! -f "${CONFIG_PATH}" ]]; then
  echo "[ERROR] evaluation_config.yaml not found: ${CONFIG_PATH}"
  exit 1
fi

if [[ ! -f "${MAIN_PY}" ]]; then
  echo "[ERROR] main.py not found: ${MAIN_PY}"
  exit 1
fi

#[COPILOT] Backup and always restore original evaluation_config.yaml.
TMP_BACKUP="$(mktemp)"
cp "${CONFIG_PATH}" "${TMP_BACKUP}"
restore_config() {
  cp "${TMP_BACKUP}" "${CONFIG_PATH}"
  rm -f "${TMP_BACKUP}"
}
trap restore_config EXIT

set_flags() {
  local use_swap="$1"
  local use_object="$2"
  local use_language="$3"
  local use_task="$4"

  #[COPILOT] Keep environment perturbation disabled by design for this sweep.
  CONFIG_PATH="${CONFIG_PATH}" \
  USE_SWAP="${use_swap}" \
  USE_OBJECT="${use_object}" \
  USE_LANGUAGE="${use_language}" \
  USE_TASK="${use_task}" \
  python3 - <<'PY'
import os
from pathlib import Path
import yaml

config_path = Path(os.environ["CONFIG_PATH"])
cfg = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}

def to_bool(v: str) -> bool:
    return v.strip().lower() in {"1", "true", "yes", "y"}

cfg["use_swap"] = to_bool(os.environ["USE_SWAP"])
cfg["use_object"] = to_bool(os.environ["USE_OBJECT"])
cfg["use_language"] = to_bool(os.environ["USE_LANGUAGE"])
cfg["use_task"] = to_bool(os.environ["USE_TASK"])
cfg["use_environment"] = False

config_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
PY
}

BASE_SUITES=("libero_spatial" "libero_object" "libero_goal" "libero_10")
MODES=("base" "swap" "object" "language" "task")
TOTAL_RUNS=$(( ${#BASE_SUITES[@]} * ${#MODES[@]} ))
RUN_IDX=0

for base_suite in "${BASE_SUITES[@]}"; do
  for mode in "${MODES[@]}"; do
    case "${mode}" in
      base)
        set_flags "false" "false" "false" "false"
        suite_tag="${base_suite}"
        ;;
      swap)
        set_flags "true" "false" "false" "false"
        suite_tag="${base_suite}_swap"
        ;;
      object)
        set_flags "false" "true" "false" "false"
        suite_tag="${base_suite}_object"
        ;;
      language)
        set_flags "false" "false" "true" "false"
        suite_tag="${base_suite}_lan"
        ;;
      task)
        set_flags "false" "false" "false" "true"
        suite_tag="${base_suite}_task"
        ;;
      *)
        echo "[ERROR] Unknown mode: ${mode}"
        exit 1
        ;;
    esac

    RUN_IDX=$((RUN_IDX + 1))
    #[COPILOT] Skip already-finished runs before START_RUN_IDX.
    if (( RUN_IDX < START_RUN_IDX )); then
      echo "[SKIP ${RUN_IDX}/${TOTAL_RUNS}] base=${base_suite} mode=${mode}"
      continue
    fi

    video_path="${VIDEO_ROOT}/${suite_tag}"
    log_path="${LOG_ROOT}/${suite_tag}.txt"
    mkdir -p "${video_path}"

    echo "[${RUN_IDX}/${TOTAL_RUNS}] base=${base_suite} mode=${mode} suite_tag=${suite_tag}"
    echo "  video_out_path=${video_path}"
    echo "  log_out_path=${log_path}"

    cmd=(
      "${PY_BIN}" "${MAIN_PY}"
      --args.host "${HOST}"
      --args.port "${PORT}"
      --args.task_suite_name "${base_suite}"
      --args.num_trials_per_task "${NUM_TRIALS}"
      --args.num_steps_wait "${NUM_STEPS_WAIT}"
      --args.replan_steps "${REPLAN_STEPS}"
      --args.resize_size "${RESIZE_SIZE}"
      --args.seed "${SEED}"
      --args.video_out_path "${video_path}"
      --args.log_out_path "${log_path}"
      --args.evaluation_config_path "${CONFIG_PATH}"
    )

    if [[ "${DRY_RUN}" == "1" ]]; then
      printf '  [DRY_RUN] %q ' "${cmd[@]}"
      printf '\n'
    else
      "${cmd[@]}"
    fi
  done
done

echo "[DONE] Completed ${TOTAL_RUNS} LIBERO-PRO evaluation runs."
