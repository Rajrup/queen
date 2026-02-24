#!/bin/bash

# Evaluate LiVoGS compression pipeline for QUEEN-trained models
# Dataset: cook_spinach from Neural_3D_Video (DyNeRF)
DATASET_NAME="Neural_3D_Video"
SEQUENCE_NAME="cook_spinach"

# LiVoGS compression parameters
START_FRAME=1
END_FRAME=300
INTERVAL=1
SH_DEGREE=2

J=15                    # Octree depth for voxelization
QUANTIZE_STEP=0.0001    # Uniform quantization step
SH_COLOR_SPACE="klt"    # Color space: rgb, yuv, klt
RLGR_BLOCK_SIZE=4096    # RLGR parallel block size

QUEEN_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
data_path="/synology/rajrup/Queen"
dataset_path="${data_path}/${DATASET_NAME}/${SEQUENCE_NAME}"
gt_model_path="${data_path}/pretrained_output/${DATASET_NAME}/queen_compressed_${SEQUENCE_NAME}"
output_folder="${gt_model_path}/compression/livogs/J_${J}_qstep_${QUANTIZE_STEP}_${SH_COLOR_SPACE}"

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate queen

### 1. LiVoGS Compress + Decompress (encode → bytestream on GPU → decode → save PLY)
echo "======================================================================"
echo "Step 1: LiVoGS Compress + Decompress"
echo "======================================================================"
echo "  QUEEN root:   ${QUEEN_ROOT}"
echo "  Model path:   ${gt_model_path}"
echo "  Output:       ${output_folder}"
echo "======================================================================"
python "${QUEEN_ROOT}/compression/livogs/compress_decompress.py" \
    --ply_path "${gt_model_path}" \
    --output_folder "${output_folder}" \
    --output_ply_folder "${output_folder}/decompressed_ply" \
    --frame_start ${START_FRAME} --frame_end ${END_FRAME} --interval ${INTERVAL} \
    --sh_degree ${SH_DEGREE} \
    --J ${J} \
    --quantize_step ${QUANTIZE_STEP} \
    --sh_color_space ${SH_COLOR_SPACE} \
    --rlgr_block_size ${RLGR_BLOCK_SIZE}

### 2. Evaluate Decompression Quality (PSNR/SSIM/LPIPS vs GT)
echo ""
echo "======================================================================"
echo "Step 2: Evaluate Decompression Quality"
echo "======================================================================"
python "${QUEEN_ROOT}/compression/evaluate_decompress.py" \
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
