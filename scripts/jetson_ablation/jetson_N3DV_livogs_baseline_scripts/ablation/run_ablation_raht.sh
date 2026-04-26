#!/bin/bash
# Run RAHT ablation study (PyTorch vs CUDA) on flame_salmon_1 and sear_steak,
# for both SH degree 0 and SH degree 2.
#
# Usage:
#   bash scripts/livogs_baseline/ablation/run_ablation_raht.sh
#
# Outputs per (sequence, sh_degree):
#   <ply_path>/ablation/livogs_raht_sh<d>/ablation_raht.csv
#   <ply_path>/ablation/livogs_raht_sh<d>/ablation_raht_config.json
#
# Plots per SH degree in:
#   scripts/livogs_baseline/ablation/plots/raht_sh<d>/

set -euo pipefail

DATASET_NAME="Neural_3D_Video"
DATA_ROOT="/home/rajrup/Queen"
FRAME_START=1
FRAME_END=300
INTERVAL=10   # sample every 10th frame; set to 1 for full benchmark

PLOT_BASE_DIR="scripts/livogs_baseline/ablation/plots"

# Sequence-specific tuned LiVoGS params.
# J and sh_color_space affect voxelization / attribute preparation (RAHT inputs).
# QP and RLGR params are saved to config for reproducibility but do not affect RAHT timing.
declare -A SEQ_J=(         ["flame_salmon_1"]=15    ["sear_steak"]=12  )
declare -A SEQ_SH_CS=(     ["flame_salmon_1"]="klt" ["sear_steak"]="klt" )
declare -A SEQ_QPS=(       ["flame_salmon_1"]=0.01  ["sear_steak"]=0.01 )
declare -A SEQ_QPQ=(       ["flame_salmon_1"]=0.04  ["sear_steak"]=0.04 )
declare -A SEQ_QPO=(       ["flame_salmon_1"]=0.01  ["sear_steak"]=0.04 )
declare -A SEQ_QPDC=(      ["flame_salmon_1"]=0.03  ["sear_steak"]=0.01 )
declare -A SEQ_QPAC=(      ["flame_salmon_1"]=0.39  ["sear_steak"]=0.06 )
declare -A SEQ_RLGR_BS=(   ["flame_salmon_1"]=512   ["sear_steak"]=512  )

SEQUENCES=("flame_salmon_1" "sear_steak")
SEQ_LABELS=("Flame Salmon" "Sear Steak")
SH_DEGREES=(0 2)

# Accumulate CSV paths per SH degree for the combined plot
declare -A SH_CSV_PATHS   # SH_CSV_PATHS[0]="csv1 csv2", SH_CSV_PATHS[2]="csv1 csv2"

echo "======================================================================"
echo "RAHT Ablation Study: PyTorch vs CUDA"
echo "  Sequences:  ${SEQUENCES[*]}"
echo "  SH degrees: ${SH_DEGREES[*]}"
echo "  Frames:     ${FRAME_START} to ${FRAME_END} (interval=${INTERVAL})"
echo "======================================================================"

for SH_DEG in "${SH_DEGREES[@]}"; do
    echo ""
    echo "======================================================================"
    echo "  SH degree: ${SH_DEG}"
    echo "======================================================================"

    CSV_PATHS=()
    SH_CSV_PATHS[$SH_DEG]=""

    for SEQ in "${SEQUENCES[@]}"; do
        PLY_PATH="${DATA_ROOT}/pretrained_output/${DATASET_NAME}/queen_compressed_${SEQ}"
        OUTPUT_FOLDER="${PLY_PATH}/ablation/livogs_raht_sh${SH_DEG}"
        J="${SEQ_J[$SEQ]}"
        SH_CS="${SEQ_SH_CS[$SEQ]}"
        QPS="${SEQ_QPS[$SEQ]}"
        QPQ="${SEQ_QPQ[$SEQ]}"
        QPO="${SEQ_QPO[$SEQ]}"
        QPDC="${SEQ_QPDC[$SEQ]}"
        QPAC="${SEQ_QPAC[$SEQ]}"
        RLGR_BS="${SEQ_RLGR_BS[$SEQ]}"

        echo ""
        echo "----------------------------------------------------------------------"
        echo "  Sequence:  ${SEQ}  |  SH${SH_DEG}"
        echo "  PLY path:  ${PLY_PATH}"
        echo "  Output:    ${OUTPUT_FOLDER}"
        echo "  J=${J}, sh_color_space=${SH_CS}"
        echo "  QP: scales=${QPS}, quats=${QPQ}, opacity=${QPO}, dc=${QPDC}, ac=${QPAC}"
        echo "  RLGR block size: ${RLGR_BS}"
        echo "----------------------------------------------------------------------"

        python scripts/livogs_baseline/ablation/ablation_raht.py \
            --ply_path        "${PLY_PATH}" \
            --output_folder   "${OUTPUT_FOLDER}" \
            --frame_start     ${FRAME_START} \
            --frame_end       ${FRAME_END} \
            --interval        ${INTERVAL} \
            --sh_degree       ${SH_DEG} \
            --J               ${J} \
            --sh_color_space  ${SH_CS} \
            --qps             ${QPS} \
            --qpq             ${QPQ} \
            --qpo             ${QPO} \
            --qpdc            ${QPDC} \
            --qpac            ${QPAC} \
            --rlgr_block_size ${RLGR_BS}

        CSV_PATHS+=("${OUTPUT_FOLDER}/ablation_raht.csv")
        SH_CSV_PATHS[$SH_DEG]+="${OUTPUT_FOLDER}/ablation_raht.csv "
    done

    # Per-SH single plot
    PLOT_DIR="${PLOT_BASE_DIR}/raht_sh${SH_DEG}"
    mkdir -p "${PLOT_DIR}"
    echo ""
    echo "Generating single-SH plots for SH${SH_DEG}..."
    for FMT in png pdf; do
        python scripts/livogs_baseline/ablation/plot_ablation_raht.py \
            --input_csvs "${CSV_PATHS[@]}" \
            --seq_labels "${SEQ_LABELS[@]}" \
            --output_folder "${PLOT_DIR}" \
            --format ${FMT}
    done
    echo "  Plots saved to: ${PLOT_DIR}"
done

# Combined SH0 + SH* plot (QUEEN sequences + Actor1 pre-existing data)
echo ""
echo "======================================================================"
echo "Generating combined SH0 + SH* plots (including Actor1)..."
echo "======================================================================"

# Actor1 uses SH0 and SH3 (= SH*)
ACTOR1_SH0_CSV="/home/rajrup/VideoGS/train_output/HiFi4G_Dataset/4K_Actor1_Greeting/ablation/livogs_raht_sh0/ablation_raht.csv"
ACTOR1_SHSTAR_CSV="/home/rajrup/VideoGS/train_output/HiFi4G_Dataset/4K_Actor1_Greeting/ablation/livogs_raht_sh3/ablation_raht.csv"

# Build arrays from the space-separated strings; Actor1 goes first (leftmost)
read -ra QUEEN_SH0_CSVS    <<< "${SH_CSV_PATHS[0]}"
read -ra QUEEN_SHSTAR_CSVS <<< "${SH_CSV_PATHS[2]}"

SH0_CSVS=(    "${ACTOR1_SH0_CSV}"    "${QUEEN_SH0_CSVS[@]}"    )
SHSTAR_CSVS=( "${ACTOR1_SHSTAR_CSV}" "${QUEEN_SHSTAR_CSVS[@]}" )
ALL_SEQ_LABELS=("Actor1" "${SEQ_LABELS[@]}")

COMBINED_PLOT_DIR="${PLOT_BASE_DIR}"
mkdir -p "${COMBINED_PLOT_DIR}"

for FMT in png pdf; do
    python scripts/livogs_baseline/ablation/plot_ablation_raht.py \
        --sh0_csvs    "${SH0_CSVS[@]}" \
        --shstar_csvs "${SHSTAR_CSVS[@]}" \
        --seq_labels  "${ALL_SEQ_LABELS[@]}" \
        --output_folder "${COMBINED_PLOT_DIR}" \
        --format ${FMT}
done

echo ""
echo "======================================================================"
echo "Done!"
echo "  Per-SH plots: ${PLOT_BASE_DIR}/raht_sh0/  and  ${PLOT_BASE_DIR}/raht_sh2/"
echo "  Combined plot: ${COMBINED_PLOT_DIR}/raht_latency_sh_combined.*"
echo "                 ${COMBINED_PLOT_DIR}/raht_speedup_sh_combined.*"
echo "======================================================================"
