#!/bin/bash

# Plot benchmark results for LiVoGS compression on QUEEN-trained models
SEQUENCE_NAME="cook_spinach"
J=15
QUANTIZE_STEP=0.0001
SH_COLOR_SPACE="klt"
CONFIG_NAME="J_${J}_qstep_${QUANTIZE_STEP}_${SH_COLOR_SPACE}"

QUEEN_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
input_folder="${QUEEN_ROOT}/output/${SEQUENCE_NAME}_trained_compressed/compression/livogs"
plot_script_folder="${QUEEN_ROOT}/scripts/livogs_baseline/plots"

# Plot compressed size breakdown + point counts
python ${plot_script_folder}/plot_compressed_size.py --input_folder ${input_folder} --j ${J} --qstep ${QUANTIZE_STEP} --sh_color_space ${SH_COLOR_SPACE} --output_folder ${plot_script_folder}

# Plot encode/decode time (stacked area)
python ${plot_script_folder}/plot_compression_time.py --input_folder ${input_folder} --j ${J} --qstep ${QUANTIZE_STEP} --sh_color_space ${SH_COLOR_SPACE} --output_folder ${plot_script_folder}

# Plot quality (PSNR / SSIM)
python ${plot_script_folder}/plot_quality.py --input_folder ${input_folder} --j ${J} --qstep ${QUANTIZE_STEP} --sh_color_space ${SH_COLOR_SPACE} --output_folder ${plot_script_folder}
