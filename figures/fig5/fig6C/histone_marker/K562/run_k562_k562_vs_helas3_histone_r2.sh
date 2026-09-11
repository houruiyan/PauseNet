#!/usr/bin/env bash
set -euo pipefail

readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly FIG_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
readonly PYTHON_BIN="/mnt/HDD8TB/houruiyan/env/conda_envs/testPauseNet/bin/python"
readonly CALCULATE_SCRIPT="${FIG_DIR}/calculate_model_tf_r2.py"

readonly DEFAULT_NPZ="/mnt/HDD8TB/houruiyan/pausing_site/2_train_model/evaluate/k562_mnetseq/test_profiles.npz"
readonly DEFAULT_TRACKS="${SCRIPT_DIR}/histone_tracks.tsv"
readonly DEFAULT_RESULT_DIR="${SCRIPT_DIR}/k562_histone_r2_results"

usage() {
    cat <<EOF
Usage:
  bash $(basename "$0") [K562_TEST_NPZ] [HISTONE_TRACKS_TSV] [RESULT_DIR]

Arguments are optional. Defaults:
  K562_TEST_NPZ      ${DEFAULT_NPZ}
  HISTONE_TRACKS_TSV ${DEFAULT_TRACKS}
  RESULT_DIR         ${DEFAULT_RESULT_DIR}

The track table must be tab-separated and contain:
  histone_marker  helas3_bigwig  k562_bigwig

Optional environment variables:
  TRAIN_CHR=chr8 TEST_CHR=chr9 THREADS=2 EXPECTED_ASSEMBLY=hg19
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi
if (( $# > 3 )); then
    usage >&2
    exit 2
fi

readonly NPZ="${1:-${DEFAULT_NPZ}}"
readonly TRACKS_TSV="${2:-${DEFAULT_TRACKS}}"
readonly RESULT_DIR="${3:-${DEFAULT_RESULT_DIR}}"
readonly TRAIN_CHR="${TRAIN_CHR:-chr8}"
readonly TEST_CHR="${TEST_CHR:-chr9}"
readonly THREADS="${THREADS:-2}"
readonly EXPECTED_ASSEMBLY="${EXPECTED_ASSEMBLY:-hg19}"

[[ -f "${NPZ}" ]] || { echo "ERROR: NPZ not found: ${NPZ}" >&2; exit 1; }
[[ -f "${TRACKS_TSV}" ]] || {
    echo "ERROR: track table not found: ${TRACKS_TSV}" >&2
    exit 1
}
[[ -f "${CALCULATE_SCRIPT}" ]] || {
    echo "ERROR: calculator not found: ${CALCULATE_SCRIPT}" >&2
    exit 1
}
[[ -x "${PYTHON_BIN}" ]] || {
    echo "ERROR: Python not executable: ${PYTHON_BIN}" >&2
    exit 1
}
[[ "${THREADS}" =~ ^[1-9][0-9]*$ ]] || {
    echo "ERROR: THREADS must be a positive integer." >&2
    exit 1
}

mkdir -p "${RESULT_DIR}"
readonly USED_TRACKS="${RESULT_DIR}/used_histone_tracks.tsv"
printf "histone_marker\tslug\thelas3_bigwig\tk562_bigwig\n" > "${USED_TRACKS}"

declare -a MARKERS=()
declare -a SLUGS=()
declare -a HELAS3_SPECS=()
declare -a K562_SPECS=()
declare -A SEEN_SLUGS=()

while IFS=$'\t' read -r marker helas3_bigwig k562_bigwig extra; do
    marker="${marker%$'\r'}"
    helas3_bigwig="${helas3_bigwig%$'\r'}"
    k562_bigwig="${k562_bigwig%$'\r'}"

    [[ -z "${marker}" || "${marker}" == \#* ]] && continue
    [[ "${marker}" == "histone_marker" ]] && continue
    if [[ -n "${extra:-}" || -z "${helas3_bigwig}" || -z "${k562_bigwig}" ]]; then
        echo "ERROR: malformed row for '${marker}' in ${TRACKS_TSV}" >&2
        echo "Expected exactly 3 tab-separated columns." >&2
        exit 1
    fi
    [[ -f "${helas3_bigwig}" ]] || {
        echo "ERROR: HeLaS3 bigWig not found for ${marker}: ${helas3_bigwig}" >&2
        exit 1
    }
    [[ -f "${k562_bigwig}" ]] || {
        echo "ERROR: K562 bigWig not found for ${marker}: ${k562_bigwig}" >&2
        exit 1
    }

    slug="$(
        printf '%s' "${marker}" |
            tr '[:upper:]' '[:lower:]' |
            sed -E 's/[^a-z0-9]+/_/g; s/^_+//; s/_+$//'
    )"
    [[ -n "${slug}" ]] || {
        echo "ERROR: marker name cannot be converted to a filename: ${marker}" >&2
        exit 1
    }
    [[ -z "${SEEN_SLUGS[${slug}]+x}" ]] || {
        echo "ERROR: duplicate marker/slug '${marker}' -> '${slug}'." >&2
        exit 1
    }
    SEEN_SLUGS["${slug}"]=1

    MARKERS+=("${marker}")
    SLUGS+=("${slug}")
    HELAS3_SPECS+=(--tf "${marker}=${helas3_bigwig}")
    K562_SPECS+=(--tf "${marker}=${k562_bigwig}")
    printf "%s\t%s\t%s\t%s\n" \
        "${marker}" "${slug}" "${helas3_bigwig}" "${k562_bigwig}" \
        >> "${USED_TRACKS}"
done < "${TRACKS_TSV}"

if (( ${#MARKERS[@]} == 0 )); then
    echo "ERROR: no histone tracks found in ${TRACKS_TSV}" >&2
    exit 1
fi

echo "Preflight: validating ${#MARKERS[@]} HeLaS3/K562 histone pairs"
"${PYTHON_BIN}" - "${USED_TRACKS}" "${EXPECTED_ASSEMBLY}" <<'PY'
import csv
import sys
from pathlib import Path

import pyBigWig

tracks_path = Path(sys.argv[1])
expected_assembly = sys.argv[2]
expected_chr1_lengths = {
    "hg19": 249_250_621,
    "hg38": 248_956_422,
}
expected_chr1 = expected_chr1_lengths.get(expected_assembly)
errors = []

with tracks_path.open("r", encoding="utf-8", newline="") as handle:
    tracks = list(csv.DictReader(handle, delimiter="\t"))

for track in tracks:
    marker = track["histone_marker"]
    for cell_type, column in (
        ("HeLaS3", "helas3_bigwig"),
        ("K562", "k562_bigwig"),
    ):
        path = Path(track[column])
        try:
            with pyBigWig.open(str(path)) as bigwig:
                if bigwig is None:
                    raise RuntimeError("pyBigWig returned no file handle")
                chr1_length = bigwig.chroms().get("chr1")
                if expected_chr1 is not None and chr1_length != expected_chr1:
                    raise ValueError(
                        f"expected {expected_assembly} chr1 length "
                        f"{expected_chr1}, found {chr1_length}"
                    )
        except Exception as exc:
            errors.append(f"{marker} {cell_type}: {path}: {exc}")

if errors:
    print("ERROR: invalid histone bigWig input(s):", file=sys.stderr)
    for error in errors:
        print(f"  - {error}", file=sys.stderr)
    raise SystemExit(1)

print(
    f"Preflight passed: {len(tracks) * 2} bigWigs; "
    f"assembly={expected_assembly}"
)
PY

run_one() {
    local marker="$1"
    local bigwig="$2"
    local output_tsv="$3"

    echo "Running ${marker}: ${bigwig}"
    "${PYTHON_BIN}" "${CALCULATE_SCRIPT}" \
        --npz "${NPZ}" \
        --train-chr "${TRAIN_CHR}" \
        --test-chr "${TEST_CHR}" \
        --tf "${marker}=${bigwig}" \
        --expected-assembly "${EXPECTED_ASSEMBLY}" \
        --threads "${THREADS}" \
        --output-tsv "${output_tsv}"
}

for i in "${!MARKERS[@]}"; do
    marker="${MARKERS[$i]}"
    slug="${SLUGS[$i]}"
    helas3_bigwig="${HELAS3_SPECS[$((i * 2 + 1))]#*=}"
    k562_bigwig="${K562_SPECS[$((i * 2 + 1))]#*=}"

    run_one \
        "${marker}" \
        "${k562_bigwig}" \
        "${RESULT_DIR}/k562_${slug}_k562.tsv"
    run_one \
        "${marker}" \
        "${helas3_bigwig}" \
        "${RESULT_DIR}/k562_${slug}_helas3.tsv"
done

if (( ${#MARKERS[@]} >= 2 )); then
    echo "Running PauseNet + all K562 histone tracks"
    "${PYTHON_BIN}" "${CALCULATE_SCRIPT}" \
        --npz "${NPZ}" \
        --train-chr "${TRAIN_CHR}" \
        --test-chr "${TEST_CHR}" \
        "${K562_SPECS[@]}" \
        --expected-assembly "${EXPECTED_ASSEMBLY}" \
        --fit-all-tfs \
        --only-all-tfs \
        --threads "${THREADS}" \
        --output-tsv "${RESULT_DIR}/k562_all_k562_histones.tsv"

    echo "Running PauseNet + all HeLaS3 histone tracks"
    "${PYTHON_BIN}" "${CALCULATE_SCRIPT}" \
        --npz "${NPZ}" \
        --train-chr "${TRAIN_CHR}" \
        --test-chr "${TEST_CHR}" \
        "${HELAS3_SPECS[@]}" \
        --expected-assembly "${EXPECTED_ASSEMBLY}" \
        --fit-all-tfs \
        --only-all-tfs \
        --threads "${THREADS}" \
        --output-tsv "${RESULT_DIR}/k562_all_helas3_histones.tsv"
fi

"${PYTHON_BIN}" - "${RESULT_DIR}" "${USED_TRACKS}" <<'PY'
import csv
import sys
from pathlib import Path

result_dir = Path(sys.argv[1])
used_tracks = Path(sys.argv[2])


def read_values(path):
    with path.open("r", encoding="utf-8", newline="") as handle:
        return {
            row["condition"]: float(row["r2_adj_procapnet"])
            for row in csv.DictReader(handle, delimiter="\t")
        }


with used_tracks.open("r", encoding="utf-8", newline="") as handle:
    tracks = list(csv.DictReader(handle, delimiter="\t"))

summary_path = result_dir / "k562_histone_r2_summary.tsv"
report_path = result_dir / "k562_histone_r2_report.txt"

with (
    summary_path.open("w", encoding="utf-8", newline="") as summary_handle,
    report_path.open("w", encoding="utf-8") as report_handle,
):
    writer = csv.writer(summary_handle, delimiter="\t")
    writer.writerow(
        [
            "histone_marker",
            "PauseNet_only_R2_adj",
            "K562_histone_only_R2_adj",
            "PauseNet_plus_K562_histone_R2_adj",
            "HeLaS3_histone_only_R2_adj",
            "PauseNet_plus_HeLaS3_histone_R2_adj",
            "matched_minus_mismatched_R2_adj",
        ]
    )

    for track in tracks:
        marker = track["histone_marker"]
        slug = track["slug"]
        k562 = read_values(result_dir / f"k562_{slug}_k562.tsv")
        helas3 = read_values(result_dir / f"k562_{slug}_helas3.tsv")
        baseline = 0.5 * (k562["model_only"] + helas3["model_only"])
        matched_advantage = (
            k562["model_plus_tf"] - helas3["model_plus_tf"]
        )

        writer.writerow(
            [
                marker,
                f"{baseline:.6f}",
                f"{k562['tf_only']:.6f}",
                f"{k562['model_plus_tf']:.6f}",
                f"{helas3['tf_only']:.6f}",
                f"{helas3['model_plus_tf']:.6f}",
                f"{matched_advantage:.6f}",
            ]
        )
        report_handle.write(f"{marker}\n")
        report_handle.write(f"  PauseNet only:                    {baseline:.6f}\n")
        report_handle.write(
            f"  K562 histone only:                "
            f"{k562['tf_only']:.6f}\n"
        )
        report_handle.write(
            f"  PauseNet + K562 histone:          "
            f"{k562['model_plus_tf']:.6f}\n"
        )
        report_handle.write(
            f"  HeLaS3 histone only:              "
            f"{helas3['tf_only']:.6f}\n"
        )
        report_handle.write(
            f"  PauseNet + HeLaS3 histone:        "
            f"{helas3['model_plus_tf']:.6f}\n"
        )
        report_handle.write(
            f"  Matched - mismatched adjusted R2: "
            f"{matched_advantage:.6f}\n\n"
        )

if len(tracks) >= 2:
    k562_all = read_values(result_dir / "k562_all_k562_histones.tsv")
    helas3_all = read_values(result_dir / "k562_all_helas3_histones.tsv")
    all_summary = result_dir / "k562_all_histones_r2_summary.tsv"
    with all_summary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(
            [
                "PauseNet_only_R2_adj",
                "All_K562_histones_only_R2_adj",
                "PauseNet_plus_all_K562_histones_R2_adj",
                "All_HeLaS3_histones_only_R2_adj",
                "PauseNet_plus_all_HeLaS3_histones_R2_adj",
                "matched_minus_mismatched_R2_adj",
            ]
        )
        baseline = 0.5 * (
            k562_all["model_only"] + helas3_all["model_only"]
        )
        matched_advantage = (
            k562_all["model_plus_all_tf"]
            - helas3_all["model_plus_all_tf"]
        )
        writer.writerow(
            [
                f"{baseline:.6f}",
                f"{k562_all['all_tf_only']:.6f}",
                f"{k562_all['model_plus_all_tf']:.6f}",
                f"{helas3_all['all_tf_only']:.6f}",
                f"{helas3_all['model_plus_all_tf']:.6f}",
                f"{matched_advantage:.6f}",
            ]
        )
    print(f"All-histone summary: {all_summary}")

print(f"Summary: {summary_path}")
print(f"Report:  {report_path}")
PY

echo "Completed:"
echo "  Summary: ${RESULT_DIR}/k562_histone_r2_summary.tsv"
echo "  Report:  ${RESULT_DIR}/k562_histone_r2_report.txt"
if (( ${#MARKERS[@]} >= 2 )); then
    echo "  All:     ${RESULT_DIR}/k562_all_histones_r2_summary.tsv"
fi
