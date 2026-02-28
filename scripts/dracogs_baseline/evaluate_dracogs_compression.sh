#!/bin/bash

# Evaluate DracoGS compression pipeline for QUEEN-trained models
# Dataset: Neural_3D_Video (DyNeRF)
#
# Usage: evaluate_dracogs_compression.sh [OPTIONS]
#   --sequence_name    Sequence name          (default: cook_spinach)
#   --frame_start      Start frame            (default: 1)
#   --frame_end        End frame              (default: 20)
#   --interval         Frame interval         (default: 1)
#   --qp               Position quantization  (default: 16)
#   --qfd              SH DC quantization     (default: 16)
#   --qfr1             SH band1 quantization  (default: 16)
#   --qfr2             SH band2 quantization  (default: 16)
#   --qfr3             SH band3 quantization  (default: 16)
#   --qo               Opacity quantization   (default: 16)
#   --qs               Scale quantization     (default: 16)
#   --qr               Rotation quantization  (default: 16)
#   --cl               Compression level      (default: 10)

DATASET_NAME="Neural_3D_Video"
# SEQUENCE_NAME="coffee_martini"    #
# SEQUENCE_NAME="cook_spinach"      #
# SEQUENCE_NAME="cut_roasted_beef"  #
SEQUENCE_NAME="flame_salmon_1"    #
# SEQUENCE_NAME="flame_steak"       #
# SEQUENCE_NAME="sear_steak"        #

START_FRAME=1
END_FRAME=300
INTERVAL=1
SH_DEGREE=2

# DracoGS quantization parameters
QP=16
QFD=16
QFR1=16
QFR2=16
QFR3=16
QO=16
QS=16
QR=16
CL=10

# --- Parse named arguments ---
while [[ $# -gt 0 ]]; do
    case "$1" in
        --sequence_name)   SEQUENCE_NAME="$2";   shift 2 ;;
        --frame_start)     START_FRAME="$2";     shift 2 ;;
        --frame_end)       END_FRAME="$2";       shift 2 ;;
        --interval)        INTERVAL="$2";        shift 2 ;;
        --qp)              QP="$2";              shift 2 ;;
        --qfd)             QFD="$2";             shift 2 ;;
        --qfr1)            QFR1="$2";            shift 2 ;;
        --qfr2)            QFR2="$2";            shift 2 ;;
        --qfr3)            QFR3="$2";            shift 2 ;;
        --qo)              QO="$2";              shift 2 ;;
        --qs)              QS="$2";              shift 2 ;;
        --qr)              QR="$2";              shift 2 ;;
        --cl)              CL="$2";              shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

data_path="/synology/rajrup/Queen"
dataset_path="${data_path}/${DATASET_NAME}/${SEQUENCE_NAME}"
gt_model_path="${data_path}/pretrained_output/${DATASET_NAME}/queen_compressed_${SEQUENCE_NAME}"

# Build output folder name from quantization parameters
output_tag="qp_${QP}_qfd_${QFD}_qfr1_${QFR1}_qfr2_${QFR2}_qfr3_${QFR3}_qo_${QO}_qs_${QS}_qr_${QR}_cl_${CL}"
output_folder="${gt_model_path}/compression/dracogs/${output_tag}"

QUEEN_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

eval "$(conda shell.bash hook 2>/dev/null)"
conda activate queen

### 1. DracoGS Compress + Decompress
echo "======================================================================"
echo "Step 1: DracoGS Compress + Decompress"
echo "======================================================================"
echo "  Dataset:      ${dataset_path}"
echo "  GT model:     ${gt_model_path}"
echo "  Output:       ${output_folder}"
echo "  Scene:        ${SEQUENCE_NAME}"
echo "  Quant:        qp=${QP} qfd=${QFD} qfr1=${QFR1} qfr2=${QFR2} qfr3=${QFR3} qo=${QO} qs=${QS} qr=${QR} cl=${CL}"
echo "======================================================================"

cd "${QUEEN_ROOT}"

python scripts/dracogs_baseline/compress_decompress_pipeline.py \
    --ply_path "${gt_model_path}" \
    --output_folder "${output_folder}" \
    --output_ply_folder "${output_folder}/decompressed_ply" \
    --frame_start ${START_FRAME} --frame_end ${END_FRAME} --interval ${INTERVAL} \
    --sh_degree ${SH_DEGREE} \
    --scene_name "${SEQUENCE_NAME}" \
    --qp ${QP} --qfd ${QFD} --qfr1 ${QFR1} --qfr2 ${QFR2} --qfr3 ${QFR3} \
    --qo ${QO} --qs ${QS} --qr ${QR} --cl ${CL}

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
