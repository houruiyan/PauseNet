#!/usr/bin/env bash
set -euo pipefail

readonly FIG_DIR="/mnt/HDD8TB/houruiyan/pausing_site/4_plot_figure/fig6/fig6C"
readonly PYTHON_BIN="/mnt/HDD8TB/houruiyan/env/conda_envs/testPauseNet/bin/python"
readonly CALCULATE_SCRIPT="${FIG_DIR}/calculate_model_tf_r2.py"
readonly NPZ="/mnt/HDD8TB/houruiyan/pausing_site/2_train_model/evaluate/helas3_netseq/test_profiles.npz"
readonly TF_ROOT="/mnt/HDD8TB/houruiyan/pausing_site/data/TF"
readonly RESULT_DIR="${FIG_DIR}/helas3_tf_r2_results"

mkdir -p "${RESULT_DIR}"

run_one() {
    local tf_name="$1"
    local bigwig="$2"
    local output_tsv="$3"

    echo "Running ${tf_name}: ${bigwig}"
    "${PYTHON_BIN}" "${CALCULATE_SCRIPT}" \
        --npz "${NPZ}" \
        --train-chr chr8 \
        --test-chr chr9 \
        --tf "${tf_name}=${bigwig}" \
        --expected-assembly hg19 \
        --threads 2 \
        --output-tsv "${output_tsv}"
}

# CTCF
run_one \
    CTCF \
    "${TF_ROOT}/CTCF/ENCFF179RSE_HelaS3_hg19.bigWig" \
    "${RESULT_DIR}/helas3_ctcf_helas3.tsv"
run_one \
    CTCF \
    "${TF_ROOT}/CTCF/ENCFF979PWH_K562_hg19.bigWig" \
    "${RESULT_DIR}/helas3_ctcf_k562.tsv"

# CHD2
run_one \
    CHD2 \
    "${TF_ROOT}/CHD2/ENCFF239ZWG_CHD2_HeLaS3_hg19.bigWig" \
    "${RESULT_DIR}/helas3_chd2_helas3.tsv"
run_one \
    CHD2 \
    "${TF_ROOT}/CHD2/ENCFF032HVZ_CHD2_K562_hg19.bigWig" \
    "${RESULT_DIR}/helas3_chd2_k562.tsv"

# SWI
run_one \
    SWI \
    "${TF_ROOT}/SWI/ENCFF805DTS_HelaS3_hg19.bigWig" \
    "${RESULT_DIR}/helas3_swi_helas3.tsv"
run_one \
    SWI \
    "${TF_ROOT}/SWI/ENCFF424PDB_K562_hg19.bigWig" \
    "${RESULT_DIR}/helas3_swi_k562.tsv"

# TFIIF
run_one \
    TFIIF \
    "${TF_ROOT}/TFIIF/ENCFF541OKE_TFIIF_HelaS3_hg19.bigWig" \
    "${RESULT_DIR}/helas3_tfiif_helas3.tsv"
run_one \
    TFIIF \
    "${TF_ROOT}/TFIIF/ENCFF050ZBY_TFIIF_K562_hg19.bigWig" \
    "${RESULT_DIR}/helas3_tfiif_k562.tsv"

echo "Running PauseNet + all selected HeLaS3 TFs"
"${PYTHON_BIN}" "${CALCULATE_SCRIPT}" \
    --npz "${NPZ}" \
    --train-chr chr8 \
    --test-chr chr9 \
    --tf "CTCF=${TF_ROOT}/CTCF/ENCFF179RSE_HelaS3_hg19.bigWig" \
    --tf "CHD2=${TF_ROOT}/CHD2/ENCFF239ZWG_CHD2_HeLaS3_hg19.bigWig" \
    --tf "SWI=${TF_ROOT}/SWI/ENCFF805DTS_HelaS3_hg19.bigWig" \
    --tf "TFIIF=${TF_ROOT}/TFIIF/ENCFF541OKE_TFIIF_HelaS3_hg19.bigWig" \
    --expected-assembly hg19 \
    --fit-all-tfs \
    --only-all-tfs \
    --threads 1 \
    --output-tsv "${RESULT_DIR}/helas3_all_helas3_tfs.tsv"

"${PYTHON_BIN}" - "${RESULT_DIR}" <<'PY'
import csv
import sys
from pathlib import Path

result_dir = Path(sys.argv[1])
tfs = ["CTCF", "CHD2", "SWI", "TFIIF"]


def read_values(path):
    values = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            values[row["condition"]] = float(row["r2_adj_procapnet"])
    return values


output = result_dir / "helas3_tf_r2_summary.tsv"
with output.open("w", encoding="utf-8", newline="") as handle:
    writer = csv.writer(handle, delimiter="\t")
    writer.writerow(
        [
            "TF",
            "PauseNet_only_R2_adj",
            "HeLaS3_TF_only_R2_adj",
            "PauseNet_plus_HeLaS3_TF_R2_adj",
            "K562_TF_only_R2_adj",
            "PauseNet_plus_K562_TF_R2_adj",
        ]
    )
    for tf in tfs:
        slug = tf.lower()
        helas3 = read_values(result_dir / f"helas3_{slug}_helas3.tsv")
        k562 = read_values(result_dir / f"helas3_{slug}_k562.tsv")
        baseline = 0.5 * (
            helas3["model_only"] + k562["model_only"]
        )
        writer.writerow(
            [
                tf,
                f"{baseline:.6f}",
                f"{helas3['tf_only']:.6f}",
                f"{helas3['model_plus_tf']:.6f}",
                f"{k562['tf_only']:.6f}",
                f"{k562['model_plus_tf']:.6f}",
            ]
        )
print(f"Summary: {output}")

all_helas3 = read_values(result_dir / "helas3_all_helas3_tfs.tsv")
all_output = result_dir / "helas3_all_helas3_tfs_summary.tsv"
with all_output.open("w", encoding="utf-8", newline="") as handle:
    writer = csv.writer(handle, delimiter="\t")
    writer.writerow(
        [
            "PauseNet_only_R2_adj",
            "All_HeLaS3_TF_only_R2_adj",
            "PauseNet_plus_all_HeLaS3_TF_R2_adj",
        ]
    )
    writer.writerow(
        [
            f"{all_helas3['model_only']:.6f}",
            f"{all_helas3['all_tf_only']:.6f}",
            f"{all_helas3['model_plus_all_tf']:.6f}",
        ]
    )
print(f"All-TF summary: {all_output}")

report = result_dir / "helas3_tf_r2_report.txt"
with report.open("w", encoding="utf-8") as handle:
    for tf in tfs:
        slug = tf.lower()
        helas3 = read_values(result_dir / f"helas3_{slug}_helas3.tsv")
        k562 = read_values(result_dir / f"helas3_{slug}_k562.tsv")
        baseline = 0.5 * (
            helas3["model_only"] + k562["model_only"]
        )
        handle.write(f"{tf}\n")
        handle.write(f"  PauseNet only:          {baseline:.6f}\n")
        handle.write(
            f"  HeLaS3 TF only:         "
            f"{helas3['tf_only']:.6f}\n"
        )
        handle.write(
            f"  PauseNet + HeLaS3 TF:   "
            f"{helas3['model_plus_tf']:.6f}\n"
        )
        handle.write(
            f"  K562 TF only:           {k562['tf_only']:.6f}\n"
        )
        handle.write(
            f"  PauseNet + K562 TF:     "
            f"{k562['model_plus_tf']:.6f}\n\n"
        )
    handle.write("ALL SELECTED HeLaS3 TFs\n")
    handle.write(
        f"  PauseNet only:             "
        f"{all_helas3['model_only']:.6f}\n"
    )
    handle.write(
        f"  All HeLaS3 TF only:        "
        f"{all_helas3['all_tf_only']:.6f}\n"
    )
    handle.write(
        f"  PauseNet + all HeLaS3 TF:  "
        f"{all_helas3['model_plus_all_tf']:.6f}\n"
    )
print(f"Report:  {report}")
PY

echo "Completed:"
echo "  Summary: ${RESULT_DIR}/helas3_tf_r2_summary.tsv"
echo "  All TF:  ${RESULT_DIR}/helas3_all_helas3_tfs_summary.tsv"
echo "  Report:  ${RESULT_DIR}/helas3_tf_r2_report.txt"
