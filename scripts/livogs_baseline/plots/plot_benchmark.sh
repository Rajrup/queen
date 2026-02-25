#!/bin/bash

# Plot benchmark results for LiVoGS compression on QUEEN-trained models
DATASET_NAME="Neural_3D_Video"
SEQUENCE_NAME="coffee_martini"    #
# SEQUENCE_NAME="cook_spinach"      # Done
# SEQUENCE_NAME="cut_roasted_beef"  # Done
# SEQUENCE_NAME="flame_salmon_1"    # Done
# SEQUENCE_NAME="flame_steak"       # Done
# SEQUENCE_NAME="sear_steak"        # Done
J=15
QUANTIZE_STEP=0.0001
SH_COLOR_SPACE="klt"
CONFIG_NAME="J_${J}_qstep_${QUANTIZE_STEP}_${SH_COLOR_SPACE}"

QUEEN_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
data_path="/synology/rajrup/Queen"
input_folder="${data_path}/pretrained_output/${DATASET_NAME}/queen_compressed_${SEQUENCE_NAME}/compression/livogs"
plot_script_folder="${QUEEN_ROOT}/scripts/livogs_baseline/plots"

# Plot compressed size breakdown + point counts
python ${plot_script_folder}/plot_compressed_size.py --input_folder ${input_folder} --dataset_name ${DATASET_NAME} --sequence_name ${SEQUENCE_NAME} --j ${J} --qstep ${QUANTIZE_STEP} --sh_color_space ${SH_COLOR_SPACE} --output_folder ${plot_script_folder}

# Plot encode/decode time (stacked area)
python ${plot_script_folder}/plot_compression_time.py --input_folder ${input_folder} --dataset_name ${DATASET_NAME} --sequence_name ${SEQUENCE_NAME} --j ${J} --qstep ${QUANTIZE_STEP} --sh_color_space ${SH_COLOR_SPACE} --output_folder ${plot_script_folder}

# Plot quality (PSNR / SSIM)
python ${plot_script_folder}/plot_quality.py --input_folder ${input_folder} --dataset_name ${DATASET_NAME} --sequence_name ${SEQUENCE_NAME} --j ${J} --qstep ${QUANTIZE_STEP} --sh_color_space ${SH_COLOR_SPACE} --output_folder ${plot_script_folder}
