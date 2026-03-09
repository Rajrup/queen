#!/bin/bash

# Evaluate LiVoGS compression pipeline for QUEEN-trained models
#
# Usage: evaluate_livogs_compression.sh [OPTIONS]
#   --dataset_name     Dataset name           (default: Neural_3D_Video)
#   --sequence_name    Sequence name          (default: cook_spinach)
#   --quantize_step    Quantization step      (default: 0.0001)
#   --j                Octree depth           (default: 12)
#   --sh_degree        SH degree (0-3)        (default: 3)
#   --sh_color_space   Color space            (default: klt)
#   --frame_start      Start frame            (default: 0)
#   --frame_end        End frame              (default: 200)
#   --interval         Frame interval         (default: 1)
#   --nvcomp           nvCOMP algorithm       (default: ANS, 'None' to disable)
#   --optimized        Use optimized decoder  (flag, default: off)

# # Full SH2 with original pipeline
# bash scripts/livogs_baseline/evaluate_livogs_compression.sh --sh_degree 2

# # SH0 only (DC) with optimized decoder
# bash scripts/livogs_baseline/evaluate_livogs_compression.sh --sh_degree 0 --optimized

# # SH1 with specific sequence
# bash scripts/livogs_baseline/evaluate_livogs_compression.sh --sh_degree 1 --sequence_name flame_salmon_1

# Evaluate LiVoGS compression pipeline for QUEEN-trained models
# Dataset: cook_spinach from Neural_3D_Video (DyNeRF)
DATASET_NAME="Neural_3D_Video"
# SEQUENCE_NAME="coffee_martini"    # Failed at Frame: 52
# SEQUENCE_NAME="cook_spinach"      # Done
# SEQUENCE_NAME="cut_roasted_beef"  # Done
SEQUENCE_NAME="flame_salmon_1"    # Done
# SEQUENCE_NAME="flame_steak"       # Done
# SEQUENCE_NAME="sear_steak"        # Done

# LiVoGS compression parameters
START_FRAME=1
END_FRAME=20
INTERVAL=1
SH_DEGREE=2

J=17                    # Octree depth for voxelization
QUANTIZE_STEP=0.0001    # Uniform quantization step
SH_COLOR_SPACE="klt"    # Color space: rgb, yuv, klt
RLGR_BLOCK_SIZE=4096    # RLGR parallel block size
NVCOMP_ALGORITHM="ANS"  # nvCOMP algorithm for position compression (None to disable)
USE_OPTIMIZED=0         # 0 = original, 1 = optimized decoder

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
data_path="/synology/rajrup/Queen"
dataset_path="${data_path}/${DATASET_NAME}/${SEQUENCE_NAME}"
gt_model_path="${data_path}/pretrained_output/${DATASET_NAME}/queen_compressed_${SEQUENCE_NAME}"
output_folder="${gt_model_path}/compression/livogs/J_${J}_qstep_${QUANTIZE_STEP}_${SH_COLOR_SPACE}_sh${SH_DEGREE}_nvcomp_${NVCOMP_ALGORITHM}"

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

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate queen

### 1. LiVoGS Compress + Decompress (encode → bytestream on GPU → decode → save PLY)
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

### 2. Evaluate Decompression Quality (PSNR/SSIM vs GT)
echo ""
echo "======================================================================"
echo "Step 2: Evaluate Decompression Quality"
echo "======================================================================"
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
