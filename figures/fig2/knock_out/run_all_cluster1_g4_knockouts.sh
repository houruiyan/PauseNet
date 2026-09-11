#!/usr/bin/env bash
set -uo pipefail

# Run G4-targeted in-silico knockout for every cluster-1 count/profile pattern.
# Patterns are processed sequentially so that only one PauseNet model occupies
# the selected GPU at a time.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_SCRIPT="${SCRIPT_DIR}/pausenet_pattern_g4_knockout.py"

PYTHON_BIN="${PYTHON_BIN:-/mnt/HDD8TB/houruiyan/env/conda_envs/PY310/bin/python}"
N_SHUFFLES="${N_SHUFFLES:-1000}"
JOBS="${JOBS:-16}"
DEVICE="${DEVICE:-cuda:0}"
G4_WINDOW="${G4_WINDOW:-25}"
G4_THRESHOLD="${G4_THRESHOLD:-1.2}"
DRY_RUN="${DRY_RUN:-0}"

RESULTS_DIR="${RESULTS_DIR:-${SCRIPT_DIR}/results}"
LOG_DIR="${LOG_DIR:-${SCRIPT_DIR}/logs}"
mkdir -p "${RESULTS_DIR}" "${LOG_DIR}"

PATTERNS=(
    count_pattern_2
    count_pattern_4
    count_pattern_5
    count_pattern_6
    count_pattern_7
    count_pattern_8
    count_pattern_9
    count_pattern_10
    count_pattern_11
    profile_pattern_6
    profile_pattern_7
    profile_pattern_8
    profile_pattern_13
    profile_pattern_19
    profile_pattern_35
    profile_pattern_37
)

successful=()
failed=()

echo "G4 knockout batch"
echo "  patterns:    ${#PATTERNS[@]}"
echo "  shuffles:    ${N_SHUFFLES} per seqlet"
echo "  workers:     ${JOBS}"
echo "  device:      ${DEVICE}"
echo "  G4 window:   ${G4_WINDOW} nt"
echo "  threshold:   ${G4_THRESHOLD}"
echo "  results:     ${RESULTS_DIR}"
echo "  logs:        ${LOG_DIR}"

if [[ ! -f "${PYTHON_SCRIPT}" ]]; then
    echo "ERROR: Python script not found: ${PYTHON_SCRIPT}" >&2
    exit 1
fi
if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "ERROR: Python executable not found: ${PYTHON_BIN}" >&2
    exit 1
fi

for pattern in "${PATTERNS[@]}"; do
    pattern_dir="${RESULTS_DIR}/${pattern}"
    output_prefix="${pattern_dir}/${pattern}_g4_knockout"
    log_file="${LOG_DIR}/${pattern}.log"
    mkdir -p "${pattern_dir}"

    command=(
        "${PYTHON_BIN}"
        "${PYTHON_SCRIPT}"
        --pattern "${pattern}"
        --n-shuffles "${N_SHUFFLES}"
        --jobs "${JOBS}"
        --device "${DEVICE}"
        --g4-window "${G4_WINDOW}"
        --g4-threshold "${G4_THRESHOLD}"
        --plot-left -500
        --plot-right 500
        --failure-policy skip
        --output-prefix "${output_prefix}"
    )

    echo
    echo "[$(date '+%F %T')] START ${pattern}"
    if [[ "${DRY_RUN}" == "1" ]]; then
        printf '  %q' "${command[@]}"
        printf '\n'
        successful+=("${pattern} (dry-run)")
        continue
    fi

    if "${command[@]}" 2>&1 | tee "${log_file}"; then
        successful+=("${pattern}")
        echo "[$(date '+%F %T')] DONE  ${pattern}"
    else
        exit_code=${PIPESTATUS[0]}
        failed+=("${pattern} (exit ${exit_code})")
        echo "[$(date '+%F %T')] FAILED ${pattern}, exit=${exit_code}" >&2
    fi
done

echo
echo "========== Batch summary =========="
echo "Successful: ${#successful[@]}"
for item in "${successful[@]}"; do
    echo "  + ${item}"
done

echo "Failed: ${#failed[@]}"
for item in "${failed[@]}"; do
    echo "  - ${item}"
done

if (( ${#failed[@]} > 0 )); then
    exit 1
fi

echo "All G4 knockout runs completed successfully."
