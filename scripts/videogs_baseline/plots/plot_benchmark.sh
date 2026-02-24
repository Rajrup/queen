#!/bin/bash

# Plot benchmark results for VideoGS compression on QUEEN-trained models
DATASET_NAME="Neural_3D_Video"
SEQUENCE_NAME="cook_spinach"
QP=22

QUEEN_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
data_path="/synology/rajrup/Queen"
input_folder="${data_path}/pretrained_output/${DATASET_NAME}/queen_compressed_${SEQUENCE_NAME}/compression/videogs"
plot_script_folder="${QUEEN_ROOT}/scripts/videogs_baseline/plots"

# Plot compressed size breakdown + point counts
python ${plot_script_folder}/plot_compressed_size.py --input_folder ${input_folder} --qp ${QP} --output_folder ${plot_script_folder}

# Plot compression/decompression time
python ${plot_script_folder}/plot_compression_time.py --input_folder ${input_folder} --qp ${QP} --output_folder ${plot_script_folder}

# Plot quality
python ${plot_script_folder}/plot_quality.py --input_folder ${input_folder} --qp ${QP} --output_folder ${plot_script_folder}
