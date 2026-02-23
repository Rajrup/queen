#!/bin/bash

# Evaluate VideoGS compression pipeline for QUEEN-trained models
# Dataset: cook_spinach from Neural_3D_Video (DyNeRF)
SEQUENCE_NAME="cook_spinach"

# VideoGS compression parameters
START_FRAME=1
END_FRAME=300
GROUP_SIZE=20
INTERVAL=1
SH_DEGREE=2
QP=22

QUEEN_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
dataset_path="${QUEEN_ROOT}/data/multipleview/${SEQUENCE_NAME}"
gt_model_path="${QUEEN_ROOT}/output/${SEQUENCE_NAME}_trained_compressed"
output_folder="${gt_model_path}/compression/videogs/qp_${QP}"

echo "======================================================================"
echo "Step 1: VideoGS Compress + Decompress"
echo "======================================================================"
### 1. Compress PLY to PNG (Full SH)
echo "Compressing PLY to PNG (Full SH)..."
python "${QUEEN_ROOT}/compression/videogs/compress_to_png_full_sh.py" --frame_start ${START_FRAME} --frame_end ${END_FRAME} --group_size ${GROUP_SIZE} --interval ${INTERVAL} --ply_path ${gt_model_path} --output_folder "${output_folder}/compressed_png" --sh_degree ${SH_DEGREE}

### 2. Compress PNG to MP4 (H.264)
echo "Compressing PNG to MP4 (H.264)..."
python "${QUEEN_ROOT}/compression/videogs/compress_png_2_video.py" --input_folder "${output_folder}/compressed_png" --output_folder "${output_folder}/compressed_video" --qp ${QP} --sh_degree ${SH_DEGREE}

### 3. Decompress MP4 to PNG
echo "Decompressing MP4 to PNG..."
python "${QUEEN_ROOT}/compression/videogs/decompress_video_2_png.py" --input_folder "${output_folder}/compressed_video" --output_folder "${output_folder}/decompressed_png"

### 4. Decompress PNG to PLY (Full SH)
echo "Decompressing PNG to PLY (Full SH)..."
python "${QUEEN_ROOT}/compression/videogs/decompress_from_png_full_sh.py" --compressed_folder "${output_folder}/decompressed_png" --output_ply_folder "${output_folder}/decompressed_ply" --sh_degree ${SH_DEGREE}

### 5. Evaluate Decompression Quality
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