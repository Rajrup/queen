#!/bin/bash
# Run RAHT ablation study (PyTorch vs CUDA) on HiFi4G sequences.
#
# Usage:
#   bash scripts/livogs_baseline/ablation/run_ablation_raht.sh
#
# Outputs per (sequence, sh_degree):
#   <data_root>/train_output/<dataset>/<seq>/ablation/livogs_raht_sh<d>/ablation_raht.csv
#   <data_root>/train_output/<dataset>/<seq>/ablation/livogs_raht_sh<d>/ablation_raht_config.json
#
# Plots per SH degree in:
#   scripts/livogs_baseline/ablation/plots/raht_sh<d>/

set -euo pipefail

DATASET_NAME="HiFi4G_Dataset"
DATA_ROOT="/home/rajrup/VideoGS"
FRAME_START=0
FRAME_END=200
INTERVAL=10   # sample every 10th frame; set to 1 for full benchmark

PLOT_BASE_DIR="scripts/livogs_baseline/ablation/plots"

# Sequence-specific tuned LiVoGS params.
# J and sh_color_space affect voxelization / attribute preparation (RAHT inputs).
# QP and RLGR params are saved to config for reproducibility but do not affect RAHT timing.
declare -A SEQ_J=(         ["4K_Actor1_Greeting"]=12  )
declare -A SEQ_SH_CS=(     ["4K_Actor1_Greeting"]="klt" )
declare -A SEQ_QPS=(       ["4K_Actor1_Greeting"]=0.0001 )
declare -A SEQ_QPQ=(       ["4K_Actor1_Greeting"]=0.01 )
declare -A SEQ_QPO=(       ["4K_Actor1_Greeting"]=0.1 )
declare -A SEQ_QPDC=(      ["4K_Actor1_Greeting"]=0.02 )
declare -A SEQ_QPAC=(      ["4K_Actor1_Greeting"]=0.06 )
declare -A SEQ_RLGR_BS=(   ["4K_Actor1_Greeting"]=512 )

SEQUENCES=("4K_Actor1_Greeting")
SEQ_LABELS=("Actor1 Greeting")
SH_DEGREES=(0 3)

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

    for SEQ in "${SEQUENCES[@]}"; do
        PLY_PATH="${DATA_ROOT}/train_output/${DATASET_NAME}/${SEQ}/checkpoint"
        OUTPUT_FOLDER="${DATA_ROOT}/train_output/${DATASET_NAME}/${SEQ}/ablation/livogs_raht_sh${SH_DEG}"
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
    done

    # Plot for this SH degree
    PLOT_DIR="${PLOT_BASE_DIR}/raht_sh${SH_DEG}"
    mkdir -p "${PLOT_DIR}"

    echo ""
    echo "Generating plots for SH${SH_DEG}..."

   # python scripts/livogs_baseline/ablation/plot_ablation_raht.py \
   #     --input_csvs "${CSV_PATHS[@]}" \
   #     --seq_labels "${SEQ_LABELS[@]}" \
    #    --output_folder "${PLOT_DIR}" \
     #   --format png

    # python scripts/livogs_baseline/ablation/plot_ablation_raht.py \
      #  --input_csvs "${CSV_PATHS[@]}" \
       # --seq_labels "${SEQ_LABELS[@]}" \
       # --output_folder "${PLOT_DIR}" \
       # --format pdf

    echo "  Plots saved to: ${PLOT_DIR}"
done

echo ""
echo "======================================================================"
echo "Done! All results under:"
for SH_DEG in "${SH_DEGREES[@]}"; do
    echo "  SH${SH_DEG}: ${PLOT_BASE_DIR}/raht_sh${SH_DEG}/"
done
echo "======================================================================"
