#!/bin/bash
# Run RLGR ablation study on a QUEEN-trained Neural_3D_Video sequence.
#
# Usage:
#   bash scripts/livogs_baseline/ablation/run_ablation_rlgr.sh [OPTIONS]
#     --sequence_name   (default: flame_salmon_1)
#     --frame_start     (default: 1)
#     --frame_end       (default: 300)
#     --interval        (default: 1)

set -euo pipefail

DATASET_NAME="Neural_3D_Video"
FRAME_START=1
FRAME_END=300
INTERVAL=50

# LiVoGS defaults for flame_salmon_1
# SEQUENCE_NAME="flame_salmon_1"
# J=15
# QPS=0.01
# QPQ=0.04
# QPO=0.01
# QPDC=0.03
# QPAC=0.39

# LiVoGS defaults for sear_steak
SEQUENCE_NAME="sear_steak"
J=12
QPS=0.01
QPQ=0.04
QPO=0.04
QPDC=0.01
QPAC=0.06

SH_COLOR_SPACE="klt"
NVCOMP="ANS"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --sequence_name)  SEQUENCE_NAME="$2";  shift 2 ;;
        --frame_start)    FRAME_START="$2";    shift 2 ;;
        --frame_end)      FRAME_END="$2";      shift 2 ;;
        --interval)       INTERVAL="$2";       shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

DATA_ROOT="/home/rajrup/Queen"
PLY_PATH="${DATA_ROOT}/pretrained_output/${DATASET_NAME}/queen_compressed_${SEQUENCE_NAME}"
OUTPUT_FOLDER="${PLY_PATH}/ablation/livogs_rlgr"

echo "======================================================================"
echo "RLGR Ablation Study"
echo "  Sequence:  ${SEQUENCE_NAME}"
echo "  Frames:    ${FRAME_START} to ${FRAME_END} (interval=${INTERVAL})"
echo "  Output:    ${OUTPUT_FOLDER}"
echo "======================================================================"

python scripts/livogs_baseline/ablation/ablation_rlgr.py \
    --ply_path "${PLY_PATH}" \
    --output_folder "${OUTPUT_FOLDER}" \
    --frame_start ${FRAME_START} --frame_end ${FRAME_END} --interval ${INTERVAL} \
    --J ${J} \
    --qps ${QPS} --qpq ${QPQ} --qpo ${QPO} --qpdc ${QPDC} --qpac ${QPAC} \
    --sh_color_space ${SH_COLOR_SPACE} \
    --nvcomp_algorithm ${NVCOMP}

echo ""
echo "Generating plots..."
python scripts/livogs_baseline/ablation/plot_ablation_rlgr.py \
    --input_csv "${OUTPUT_FOLDER}/ablation_rlgr.csv" \
    --output_folder scripts/livogs_baseline/ablation/plots"

echo ""
echo "======================================================================"
echo "Done! Results in: ${OUTPUT_FOLDER}"
echo "======================================================================"
