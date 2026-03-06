#!/bin/bash

# Plot benchmark results for MesonGS compression on QUEEN-trained models
DATASET_NAME="Neural_3D_Video"
# SEQUENCE_NAME="coffee_martini"    #
# SEQUENCE_NAME="cook_spinach"      #
# SEQUENCE_NAME="cut_roasted_beef"  #
SEQUENCE_NAME="flame_salmon_1"    #
# SEQUENCE_NAME="flame_steak"       #
# SEQUENCE_NAME="sear_steak"        #

CONFIG_NAME="params_default"

data_path="/synology/rajrup/Queen"
gt_model_path="${data_path}/pretrained_output/${DATASET_NAME}/queen_compressed_${SEQUENCE_NAME}"
input_folder="${gt_model_path}/compression/mesongs"

QUEEN_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
plot_script_folder="${QUEEN_ROOT}/scripts/mesongs_baseline/plots"

# Plot compressed size + point counts
python ${plot_script_folder}/plot_compressed_size.py --input_folder ${input_folder} --dataset_name ${DATASET_NAME} --sequence_name ${SEQUENCE_NAME} --config_name ${CONFIG_NAME} --output_folder ${plot_script_folder}

# Plot encode/decode time (stacked area)
python ${plot_script_folder}/plot_compression_time.py --input_folder ${input_folder} --dataset_name ${DATASET_NAME} --sequence_name ${SEQUENCE_NAME} --config_name ${CONFIG_NAME} --output_folder ${plot_script_folder}

# Plot quality (PSNR / SSIM)
python ${plot_script_folder}/plot_quality.py --input_folder ${input_folder} --dataset_name ${DATASET_NAME} --sequence_name ${SEQUENCE_NAME} --config_name ${CONFIG_NAME} --output_folder ${plot_script_folder}
