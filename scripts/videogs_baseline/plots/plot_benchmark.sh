#!/bin/bash

# Plot benchmark results for VideoGS compression on QUEEN-trained models
DATASET_NAME="Neural_3D_Video"
# SEQUENCE_NAME="coffee_martini"    #
SEQUENCE_NAME="cook_spinach"      # Done
# SEQUENCE_NAME="cut_roasted_beef"  # Done
# SEQUENCE_NAME="flame_salmon_1"    # Done
# SEQUENCE_NAME="flame_steak"       # Done
# SEQUENCE_NAME="sear_steak"        # Done

# H.264 QP (0=lossless, 51=worst)
QP=25

QUEEN_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
data_path="/synology/rajrup/Queen"
input_folder="${data_path}/pretrained_output/${DATASET_NAME}/queen_compressed_${SEQUENCE_NAME}/compression/videogs"
plot_script_folder="${QUEEN_ROOT}/scripts/videogs_baseline/plots"

QP_ARGS="--qp ${QP}"

# Plot compressed size breakdown + point counts
python ${plot_script_folder}/plot_compressed_size.py \
    --input_folder ${input_folder} \
    --dataset_name ${DATASET_NAME} \
    --sequence_name ${SEQUENCE_NAME} \
    ${QP_ARGS} \
    --output_folder ${plot_script_folder}

# Plot compression/decompression time
python ${plot_script_folder}/plot_compression_time.py \
    --input_folder ${input_folder} \
    --dataset_name ${DATASET_NAME} \
    --sequence_name ${SEQUENCE_NAME} \
    ${QP_ARGS} \
    --output_folder ${plot_script_folder}

# Plot quality
python ${plot_script_folder}/plot_quality.py \
    --input_folder ${input_folder} \
    --dataset_name ${DATASET_NAME} \
    --sequence_name ${SEQUENCE_NAME} \
    ${QP_ARGS} \
    --output_folder ${plot_script_folder}
