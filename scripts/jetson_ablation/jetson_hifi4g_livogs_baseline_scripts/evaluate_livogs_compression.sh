#!/bin/bash

# Evaluate LiVoGS compression pipeline for VideoGS-trained models
#
# Usage: evaluate_livogs_compression.sh [OPTIONS]
#   --dataset_name     Dataset name           (default: HiFi4G_Dataset)
#   --sequence_name    Sequence name          (default: 4K_Actor1_Greeting)
#   --quantize_step    Quantization step      (default: 0.0001)
#   --j                Octree depth           (default: 12)
#   --sh_degree        SH degree (0-3)        (default: 3)
#   --sh_color_space   Color space            (default: klt)
#   --frame_start      Start frame            (default: 0)
#   --frame_end        End frame              (default: 200)
#   --interval         Frame interval         (default: 1)
#   --nvcomp           nvCOMP algorithm       (default: ANS, 'None' to disable)
#   --optimized        Use optimized decoder  (flag, default: off)

# # Full SH3 with original pipeline
# bash scripts/livogs_baseline/evaluate_livogs_compression.sh --sh_degree 3

# # SH0 only (DC) with optimized decoder
# bash scripts/livogs_baseline/evaluate_livogs_compression.sh --sh_degree 0 --optimized

# # SH1 with specific sequence
# bash scripts/livogs_baseline/evaluate_livogs_compression.sh --sh_degree 1 --sequence_name 4K_Actor2_Dancing

DATASET_NAME="HiFi4G_Dataset"
SEQUENCE_NAME="4K_Actor1_Greeting"
RESOLUTION=2

START_FRAME=0
END_FRAME=200
INTERVAL=1
SH_DEGREE=3

# LiVoGS compression parameters
J=12                    # Octree depth for voxelization
QUANTIZE_STEP=0.0001      # Uniform quantization step
SH_COLOR_SPACE="klt"    # Color space: rgb, yuv, klt
RLGR_BLOCK_SIZE=4096    # RLGR parallel block size
NVCOMP_ALGORITHM="None"  # nvCOMP algorithm for position compression (None to disable) -> Jetson doesn't support nvcomp
USE_OPTIMIZED=0         # 0 = original, 1 = optimized decoder (Doesn't show improvement on Jetson Orin)

# --- Parse named arguments ---
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dataset_name)   DATASET_NAME="$2";   shift 2 ;;
        --sequence_name)  SEQUENCE_NAME="$2";  shift 2 ;;
        --quantize_step)  QUANTIZE_STEP="$2";  shift 2 ;;
        --j)              J="$2";              shift 2 ;;
        --sh_degree)      SH_DEGREE="$2";      shift 2 ;;
        --sh_color_space) SH_COLOR_SPACE="$2"; shift 2 ;;
        --frame_start)    START_FRAME="$2";    shift 2 ;;
        --frame_end)      END_FRAME="$2";      shift 2 ;;
        --interval)       INTERVAL="$2";       shift 2 ;;
        --nvcomp)         NVCOMP_ALGORITHM="$2"; shift 2 ;;
        --optimized)      USE_OPTIMIZED=1;     shift 1 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

data_path="/home/rajrup/VideoGS"
dataset_path="${data_path}/${DATASET_NAME}_processed/${SEQUENCE_NAME}"
gt_model_path="${data_path}/train_output/${DATASET_NAME}/${SEQUENCE_NAME}/checkpoint"
output_folder="${data_path}/train_output/${DATASET_NAME}/${SEQUENCE_NAME}/compression/livogs/J_${J}_qstep_${QUANTIZE_STEP}_${SH_COLOR_SPACE}_sh${SH_DEGREE}_nvcomp_${NVCOMP_ALGORITHM}"

if [[ ${USE_OPTIMIZED} -eq 1 ]]; then
    PIPELINE_SCRIPT="scripts/livogs_baseline/compress_decompress_pipeline_optimized.py"
    output_folder="${output_folder}_optimized"
else
    PIPELINE_SCRIPT="scripts/livogs_baseline/compress_decompress_pipeline.py"
fi

echo "======================================================================"
echo "LiVoGS Compression Evaluation"
echo "======================================================================"
echo "  Dataset:          ${DATASET_NAME} / ${SEQUENCE_NAME}"
echo "  SH degree:        ${SH_DEGREE}"
echo "  J (octree depth): ${J}"
echo "  Quantize step:    ${QUANTIZE_STEP}"
echo "  Color space:      ${SH_COLOR_SPACE}"
echo "  nvCOMP:           ${NVCOMP_ALGORITHM}"
echo "  Pipeline:         $(basename ${PIPELINE_SCRIPT})"
echo "  Frames:           ${START_FRAME} to ${END_FRAME} (interval=${INTERVAL})"
echo "  Output:           ${output_folder}"
echo "======================================================================"

### 1. LiVoGS Compress + Decompress (encode → bytestream on GPU → decode → save PLY)
echo ""
echo "======================================================================"
echo "Step 1: LiVoGS Compress + Decompress"
echo "======================================================================"
python ${PIPELINE_SCRIPT} \
    --ply_path "${gt_model_path}" \
    --output_folder "${output_folder}" \
    --output_ply_folder "${output_folder}/decompressed_ply" \
    --frame_start ${START_FRAME} --frame_end ${END_FRAME} --interval ${INTERVAL} \
    --sh_degree ${SH_DEGREE} \
    --J ${J} \
    --quantize_step ${QUANTIZE_STEP} \
    --sh_color_space ${SH_COLOR_SPACE} \
    --rlgr_block_size ${RLGR_BLOCK_SIZE} \
    --nvcomp_algorithm ${NVCOMP_ALGORITHM}

# ### 2. Evaluate Decompression Quality (PSNR/SSIM vs GT)
# echo ""
# echo "======================================================================"
# echo "Step 2: Evaluate Decompression Quality"
# echo "======================================================================"
# python scripts/evaluate_decompress.py \
#     --gt_ply_path "${gt_model_path}" \
#     --decompressed_ply_path "${output_folder}/decompressed_ply" \
#     --dataset_path "${dataset_path}" \
#     --output_render_path "${output_folder}/evaluation" \
#     --save_renders \
#     --sh_degree ${SH_DEGREE} \
#     --resolution ${RESOLUTION} \
#     --frame_start ${START_FRAME} --frame_end ${END_FRAME} --interval ${INTERVAL}

# echo ""
# echo "======================================================================"
# echo "Done! Results in: ${output_folder}"
# echo "======================================================================"
