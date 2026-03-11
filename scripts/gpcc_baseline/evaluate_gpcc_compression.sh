#!/bin/bash

# Evaluate GPCC compression pipeline for VideoGS-trained models
#
# Usage: evaluate_gpcc_compression.sh [OPTIONS]
#   --dataset_path     Path to VideoGS checkpoint dir  (required)
#   --output_dir       Output directory                (required)
#   --output_ply_dir   Output PLY directory            (default: output_dir/decompressed_ply)
#   --gt_ply_path      GT model path for evaluation    (default: dataset_path)
#   --dataset_eval_path  Dataset path for rendering    (required for eval)
#   --tmc3_path        Path to tmc3 binary             (default: /ssd1/haodongw/workspace/3dstream/mpeg-pcc-tmc13/build/tmc3/tmc3)
#   --voxel_depth      Voxelization depth J            (default: 15)
#   --qp_rest          QP for rest attributes          (default: 40)
#   --qp_dc            QP for dc attributes            (default: 4)
#   --qp_opacity       QP for opacity                  (default: 4)
#   --frame_start      Start frame                     (default: 0)
#   --num_frames       Number of frames                (default: 1)
#   --sh_degree        SH degree                       (default: 3)
#   --resolution       Render resolution               (default: 2)

# Default values
DATASET_PATH=""
OUTPUT_DIR=""
OUTPUT_PLY_DIR=""
GT_PLY_PATH=""
DATASET_EVAL_PATH=""
TMC3_PATH="/ssd1/haodongw/workspace/3dstream/mpeg-pcc-tmc13/build/tmc3/tmc3"
VOXEL_DEPTH=15
QP_REST=40
QP_DC=4
QP_OPACITY=4
FRAME_START=0
NUM_FRAMES=1
SH_DEGREE=3
RESOLUTION=2

# Parse named arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dataset_path)      DATASET_PATH="$2";      shift 2 ;;
        --output_dir)        OUTPUT_DIR="$2";         shift 2 ;;
        --output_ply_dir)    OUTPUT_PLY_DIR="$2";     shift 2 ;;
        --gt_ply_path)       GT_PLY_PATH="$2";        shift 2 ;;
        --dataset_eval_path) DATASET_EVAL_PATH="$2";  shift 2 ;;
        --tmc3_path)         TMC3_PATH="$2";          shift 2 ;;
        --voxel_depth)       VOXEL_DEPTH="$2";        shift 2 ;;
        --qp_rest)           QP_REST="$2";            shift 2 ;;
        --qp_dc)             QP_DC="$2";              shift 2 ;;
        --qp_opacity)        QP_OPACITY="$2";         shift 2 ;;
        --frame_start)       FRAME_START="$2";        shift 2 ;;
        --num_frames)        NUM_FRAMES="$2";         shift 2 ;;
        --sh_degree)         SH_DEGREE="$2";          shift 2 ;;
        --resolution)        RESOLUTION="$2";         shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

# Validate required args
if [[ -z "$DATASET_PATH" ]]; then
    echo "Error: --dataset_path is required"
    exit 1
fi
if [[ -z "$OUTPUT_DIR" ]]; then
    echo "Error: --output_dir is required"
    exit 1
fi

# Set defaults
if [[ -z "$OUTPUT_PLY_DIR" ]]; then
    OUTPUT_PLY_DIR="${OUTPUT_DIR}/decompressed_ply"
fi
if [[ -z "$GT_PLY_PATH" ]]; then
    GT_PLY_PATH="${DATASET_PATH}"
fi

# Get script directory for relative paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPTS_DIR="$(dirname "$SCRIPT_DIR")"

### Step 1: GPCC Compress + Decompress
echo "======================================================================"
echo "Step 1: GPCC Compress + Decompress"
echo "======================================================================"
python "${SCRIPT_DIR}/compress_decompress_pipeline.py" \
    --input_dir "${DATASET_PATH}" \
    --output_dir "${OUTPUT_DIR}" \
    --output_ply_dir "${OUTPUT_PLY_DIR}" \
    --tmc3_path "${TMC3_PATH}" \
    --voxel_depth ${VOXEL_DEPTH} \
    --qp_rest ${QP_REST} \
    --qp_dc ${QP_DC} \
    --qp_opacity ${QP_OPACITY} \
    --frame_start ${FRAME_START} \
    --num_frames ${NUM_FRAMES}

### Step 2: Evaluate Decompression Quality (if dataset_eval_path provided)
if [[ -n "$DATASET_EVAL_PATH" ]]; then
    echo ""
    echo "======================================================================"
    echo "Step 2: Evaluate Decompression Quality"
    echo "======================================================================"
    python "${SCRIPTS_DIR}/evaluate_decompress.py" \
        --gt_ply_path "${GT_PLY_PATH}" \
        --decompressed_ply_path "${OUTPUT_PLY_DIR}" \
        --dataset_path "${DATASET_EVAL_PATH}" \
        --output_render_path "${OUTPUT_DIR}/evaluation" \
        --save_renders \
        --sh_degree ${SH_DEGREE} \
        --resolution ${RESOLUTION} \
        --frame_start ${FRAME_START} \
        --frame_end $((FRAME_START + NUM_FRAMES)) \
        --interval 1
fi

echo ""
echo "======================================================================"
echo "Done! Results in: ${OUTPUT_DIR}"
echo "======================================================================"
