#!/bin/bash
eval "$(conda shell.bash hook)"
conda activate queen

BASEDIR=/synology/rajrup/Queen/pretrained_output/Neural_3D_Video/queen_compressed_sear_steak
GPCC_DIR=${BASEDIR}/compression/gpcc_test/J17_dc16_rest8_qp4/frame1
QUANT_DIR=${BASEDIR}/compression/gpcc_test/J17_dc16_rest8_quantize_only/frame1

EVAL_ARGS="--config configs/dynerf.yaml -s /synology/rajrup/Queen/Neural_3D_Video/sear_steak -m ${BASEDIR} --frame_start 1 --frame_end 1 --interval 1 --save_renders"

echo "===== [1/2] QUANTIZE ONLY: DC=uint16, rest=uint8 YUV round-trip (no TMC3) ====="
CUDA_VISIBLE_DEVICES=0 python /home/rajrup/Project/Queen/scripts/gpcc_baseline/compress_decompress_pipeline.py \
  --input_dir ${BASEDIR} --output_dir ${QUANT_DIR} --output_ply_dir ${QUANT_DIR}/decompressed_ply \
  --voxel_depth 17 --frame_start 1 --num_frames 1 --quantize_only
CUDA_VISIBLE_DEVICES=0 python /home/rajrup/Project/Queen/scripts/evaluate_decompress.py ${EVAL_ARGS} \
  --decompressed_ply_path ${QUANT_DIR}/decompressed_ply --output_render_path ${QUANT_DIR}/evaluation

echo ""
echo "===== [2/2] FULL PIPELINE: DC=uint16, rest=uint8 YUV + GPCC QP=(4,4,4) ====="
CUDA_VISIBLE_DEVICES=0 python /home/rajrup/Project/Queen/scripts/gpcc_baseline/compress_decompress_pipeline.py \
  --input_dir ${BASEDIR} --output_dir ${GPCC_DIR} --output_ply_dir ${GPCC_DIR}/decompressed_ply \
  --tmc3_path /home/rajrup/Project/mpeg-pcc-tmc13/build/tmc3/tmc3 \
  --voxel_depth 17 --qp_opacity 4 --qp_dc 4 --qp_rest 4 --frame_start 1 --num_frames 1
CUDA_VISIBLE_DEVICES=0 python /home/rajrup/Project/Queen/scripts/evaluate_decompress.py ${EVAL_ARGS} \
  --decompressed_ply_path ${GPCC_DIR}/decompressed_ply --output_render_path ${GPCC_DIR}/evaluation
