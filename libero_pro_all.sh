#!/usr/bin/env bash
set -euo pipefail

#[COPILOT] Sweep script for 20 LIBERO-PRO evaluations:
#[COPILOT] 4 main suites x (base + swap + object + language + task).

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_PATH="$(realpath "${BASH_SOURCE[0]}")"
CONFIG_PATH="${ROOT_DIR}/third_party/LIBERO-PRO/evaluation_config.yaml"
MAIN_PY="${ROOT_DIR}/examples/libero-pro/main.py"

#[COPILOT] Runtime knobs (override via env vars when needed).
PY_BIN="${PY_BIN:-${ROOT_DIR}/third_party/LIBERO-PRO/.venv-pro/bin/python}" #[COPILOT] Fixed LIBERO-PRO venv interpreter path.
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
NUM_WORKERS="${NUM_WORKERS:-1}" #[COPILOT] Number of parallel workers. NUM_WORKERS=6 yields 4/4/3/3/3/3 split for 20 runs.
WORKER_IDX="${WORKER_IDX:-0}" #[COPILOT] 0-based worker id; auto-managed when NUM_WORKERS>1.
INTERNAL_WORKER="${INTERNAL_WORKER:-0}"
GPU_IDS="${GPU_IDS:-0,1,2,3,4,5}" #[COPILOT] Comma-separated GPU ids to bind workers in order.
PORTS="${PORTS:-8001,8002,8003,8004,8005,8006}" #[COPILOT] Comma-separated policy-server ports to bind workers in order.
RUN_IDS="${RUN_IDS:-}" #[COPILOT] Optional comma-separated 1-based run ids from the 20-run sweep (e.g., 1,2,7,13).

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

if (( NUM_WORKERS < 1 )); then
  echo "[ERROR] NUM_WORKERS must be >= 1 (got ${NUM_WORKERS})"
  exit 1
fi

if (( INTERNAL_WORKER == 0 )) && (( NUM_WORKERS > 1 )); then
  IFS=',' read -r -a GPU_ARR <<< "${GPU_IDS}"
  IFS=',' read -r -a PORT_ARR <<< "${PORTS}"
  if (( ${#GPU_ARR[@]} < NUM_WORKERS )); then
    echo "[ERROR] GPU_IDS count (${#GPU_ARR[@]}) is smaller than NUM_WORKERS (${NUM_WORKERS})."
    exit 1
  fi
  if (( ${#PORT_ARR[@]} < NUM_WORKERS )); then
    echo "[ERROR] PORTS count (${#PORT_ARR[@]}) is smaller than NUM_WORKERS (${NUM_WORKERS})."
    exit 1
  fi

  pids=()
  for ((i=0; i<NUM_WORKERS; i++)); do
    echo "[SPAWN] worker=${i}/${NUM_WORKERS} gpu=${GPU_ARR[$i]} port=${PORT_ARR[$i]}"
    CUDA_VISIBLE_DEVICES="${GPU_ARR[$i]}" \
    INTERNAL_WORKER=1 \
    WORKER_IDX="${i}" \
    NUM_WORKERS="${NUM_WORKERS}" \
    PORT="${PORT_ARR[$i]}" \
    bash "${SCRIPT_PATH}" &
    pids+=("$!")
  done

  failed=0
  for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then
      failed=1
    fi
  done

  if (( failed != 0 )); then
    echo "[ERROR] One or more workers failed."
    exit 1
  fi
  echo "[DONE] All ${NUM_WORKERS} workers completed."
  exit 0
fi

if (( WORKER_IDX < 0 )) || (( WORKER_IDX >= NUM_WORKERS )); then
  echo "[ERROR] WORKER_IDX must satisfy 0 <= WORKER_IDX < NUM_WORKERS (got ${WORKER_IDX}, ${NUM_WORKERS})."
  exit 1
fi

WORK_TMP_DIR="$(mktemp -d)"
WORK_CONFIG_PATH="${WORK_TMP_DIR}/evaluation_config.worker_${WORKER_IDX}.yaml"
cp "${CONFIG_PATH}" "${WORK_CONFIG_PATH}"
cleanup_worker_tmp() {
  rm -rf "${WORK_TMP_DIR}"
}
trap cleanup_worker_tmp EXIT

set_flags() {
  local config_path="$1"
  local use_swap="$2"
  local use_object="$3"
  local use_language="$4"
  local use_task="$5"

  #[COPILOT] Keep environment perturbation disabled by design for this sweep.
  CONFIG_PATH="${config_path}" \
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
ASSIGNED_RUNS=0
declare -A RUN_ID_SET=()

if [[ -n "${RUN_IDS}" ]]; then
  IFS=',' read -r -a RUN_ID_ARR <<< "${RUN_IDS}"
  for raw_id in "${RUN_ID_ARR[@]}"; do
    run_id="$(echo "${raw_id}" | tr -d '[:space:]')"
    if [[ -z "${run_id}" ]]; then
      continue
    fi
    if ! [[ "${run_id}" =~ ^[0-9]+$ ]]; then
      echo "[ERROR] RUN_IDS must be comma-separated positive integers (got '${raw_id}')."
      exit 1
    fi
    RUN_ID_SET["${run_id}"]=1
  done
fi

echo "[WORKER] idx=${WORKER_IDX}/${NUM_WORKERS} gpu=${CUDA_VISIBLE_DEVICES:-unset} port=${PORT}"

for base_suite in "${BASE_SUITES[@]}"; do
  for mode in "${MODES[@]}"; do
    case "${mode}" in
      base)
        use_swap="false"
        use_object="false"
        use_language="false"
        use_task="false"
        suite_tag="${base_suite}"
        ;;
      swap)
        use_swap="true"
        use_object="false"
        use_language="false"
        use_task="false"
        suite_tag="${base_suite}_swap"
        ;;
      object)
        use_swap="false"
        use_object="true"
        use_language="false"
        use_task="false"
        suite_tag="${base_suite}_object"
        ;;
      language)
        use_swap="false"
        use_object="false"
        use_language="true"
        use_task="false"
        suite_tag="${base_suite}_lan"
        ;;
      task)
        use_swap="false"
        use_object="false"
        use_language="false"
        use_task="true"
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

    if [[ -n "${RUN_IDS}" ]] && [[ -z "${RUN_ID_SET[${RUN_IDX}]+x}" ]]; then
      continue
    fi

    #[COPILOT] Deterministic static sharding: with NUM_WORKERS=6 and TOTAL_RUNS=20 => 4/4/3/3/3/3.
    if (( ((RUN_IDX - 1) % NUM_WORKERS) != WORKER_IDX )); then
      continue
    fi

    set_flags "${WORK_CONFIG_PATH}" "${use_swap}" "${use_object}" "${use_language}" "${use_task}"
    ASSIGNED_RUNS=$((ASSIGNED_RUNS + 1))

    video_path="${VIDEO_ROOT}/${suite_tag}"
    log_path="${LOG_ROOT}/${suite_tag}.txt"
    mkdir -p "${video_path}"

    echo "[${RUN_IDX}/${TOTAL_RUNS}] worker=${WORKER_IDX} base=${base_suite} mode=${mode} suite_tag=${suite_tag}"
    echo "  video_out_path=${video_path}"
    echo "  log_out_path=${log_path}"
    echo "  evaluation_config_path=${WORK_CONFIG_PATH}"

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
      --args.evaluation_config_path "${WORK_CONFIG_PATH}"
    )

    if [[ "${DRY_RUN}" == "1" ]]; then
      printf '  [DRY_RUN] %q ' "${cmd[@]}"
      printf '\n'
    else
      "${cmd[@]}"
    fi
  done
done

echo "[DONE] worker=${WORKER_IDX} assigned_runs=${ASSIGNED_RUNS}"
