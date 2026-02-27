#!/bin/bash

# Evaluate MesonGS compression pipeline for QUEEN-trained models
# Dataset: Neural_3D_Video (DyNeRF)
#
# Usage: evaluate_mesongs_compression.sh [OPTIONS]
#   --sequence_name    Sequence name          (default: cook_spinach)
#   --frame_start      Start frame            (default: 1)
#   --frame_end        End frame              (default: 20)
#   --interval         Frame interval         (default: 1)
#   --depth            Octree depth           (default: from config)
#   --n_block          Block quant count      (default: from config)
#   --codebook_size    VQ codebook size       (default: from config)
#   --prune            Enable pruning         (flag)

DATASET_NAME="Neural_3D_Video"
# SEQUENCE_NAME="coffee_martini"    #
# SEQUENCE_NAME="cook_spinach"      #
# SEQUENCE_NAME="cut_roasted_beef"  #
SEQUENCE_NAME="flame_salmon_1"    #
# SEQUENCE_NAME="flame_steak"       #
# SEQUENCE_NAME="sear_steak"        #

START_FRAME=1
END_FRAME=300
INTERVAL=10
SH_DEGREE=2

# MesonGS-specific parameters (empty = use config defaults)
DEPTH=""
N_BLOCK=""
CODEBOOK_SIZE=""
PRUNE=""

# --- Parse named arguments ---
while [[ $# -gt 0 ]]; do
    case "$1" in
        --sequence_name)   SEQUENCE_NAME="$2";   shift 2 ;;
        --frame_start)     START_FRAME="$2";     shift 2 ;;
        --frame_end)       END_FRAME="$2";       shift 2 ;;
        --interval)        INTERVAL="$2";        shift 2 ;;
        --depth)           DEPTH="$2";           shift 2 ;;
        --n_block)         N_BLOCK="$2";         shift 2 ;;
        --codebook_size)   CODEBOOK_SIZE="$2";   shift 2 ;;
        --prune)           PRUNE="--prune";      shift 1 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

data_path="/synology/rajrup/Queen"
dataset_path="${data_path}/${DATASET_NAME}/${SEQUENCE_NAME}"
gt_model_path="${data_path}/pretrained_output/${DATASET_NAME}/queen_compressed_${SEQUENCE_NAME}"

# Build output folder name from parameters
output_tag="depth_${DEPTH:-cfg}_nblock_${N_BLOCK:-cfg}_cb_${CODEBOOK_SIZE:-cfg}"
if [ -n "$PRUNE" ]; then
    output_tag="${output_tag}_pruned"
fi
output_folder="${gt_model_path}/compression/mesongs/${output_tag}"

# Build optional args for the Python script
OPTIONAL_ARGS=""
[ -n "$DEPTH" ]         && OPTIONAL_ARGS="${OPTIONAL_ARGS} --depth ${DEPTH}"
[ -n "$N_BLOCK" ]       && OPTIONAL_ARGS="${OPTIONAL_ARGS} --n_block ${N_BLOCK}"
[ -n "$CODEBOOK_SIZE" ] && OPTIONAL_ARGS="${OPTIONAL_ARGS} --codebook_size ${CODEBOOK_SIZE}"
[ -n "$PRUNE" ]         && OPTIONAL_ARGS="${OPTIONAL_ARGS} ${PRUNE}"

QUEEN_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MESONGS_ROOT="${QUEEN_ROOT}/MesonGS"

eval "$(conda shell.bash hook 2>/dev/null)"
conda activate mesongs

### 1. MesonGS Compress + Decompress (from MesonGS dir for raht_torch etc.)
echo "======================================================================"
echo "Step 1: MesonGS Compress + Decompress"
echo "======================================================================"
echo "  Dataset:      ${dataset_path}"
echo "  GT model:     ${gt_model_path}"
echo "  Output:       ${output_folder}"
echo "  Scene:        ${SEQUENCE_NAME}"
echo "  Optional:     ${OPTIONAL_ARGS}"
echo "======================================================================"

cd "${MESONGS_ROOT}"

python "${QUEEN_ROOT}/scripts/mesongs_baseline/compression_decompress_pipeline.py" \
    --ply_path "${gt_model_path}" \
    --dataset_path "${dataset_path}" \
    --output_folder "${output_folder}" \
    --output_ply_folder "${output_folder}/decompressed_ply" \
    --frame_start ${START_FRAME} --frame_end ${END_FRAME} --interval ${INTERVAL} \
    --sh_degree ${SH_DEGREE} \
    --scene_name "${SEQUENCE_NAME}" \
    ${OPTIONAL_ARGS}

### 2. Evaluate Decompression Quality (PSNR/SSIM vs GT)
echo ""
echo "======================================================================"
echo "Step 2: Evaluate Decompression Quality"
echo "======================================================================"

conda activate queen
cd "${QUEEN_ROOT}"

python scripts/evaluate_decompress.py \
    --config configs/dynerf.yaml \
    -s "${dataset_path}" \
    -m "${gt_model_path}" \
    --decompressed_ply_path "${output_folder}/decompressed_ply" \
    --output_render_path "${output_folder}/evaluation" \
    --save_renders \
    --frame_start ${START_FRAME} --frame_end ${END_FRAME} --interval ${INTERVAL}

echo ""
echo "======================================================================"
echo "Done! Results in: ${output_folder}"
echo "======================================================================"
