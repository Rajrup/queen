#!/bin/bash
# Download QUEEN pre-trained models from GitHub release v1.0-compressed
# For each sequence: download parts -> cat into tar.gz -> extract -> remove parts

set -e

OUTPUT_DIR="/synology/rajrup/Queen/pretrained_output"
BASE_URL="https://github.com/NVlabs/queen/releases/download/v1.0-compressed"
mkdir -p "$OUTPUT_DIR"
cd "$OUTPUT_DIR"

SEQUENCES=(
    "cook_spinach:8"
    "coffee_martini:14"
    "cut_roasted_beef:8"
    "flame_salmon_1:13"
    "flame_steak:8"
    "sear_steak:9"
)

for entry in "${SEQUENCES[@]}"; do
    SEQ="${entry%%:*}"
    NUM_PARTS="${entry##*:}"
    PREFIX="queen_compressed_${SEQ}"
    TAR_FILE="${PREFIX}.tar.gz"

    echo ""
    echo "======================================================================"
    echo "Sequence: ${SEQ} (${NUM_PARTS} parts)"
    echo "======================================================================"

    # Skip if already extracted
    if [ -d "${OUTPUT_DIR}/${SEQ}" ] || [ -d "${OUTPUT_DIR}/queen_compressed_${SEQ}" ]; then
        echo "  Already extracted, skipping download."
        continue
    fi

    # Skip if tar already exists
    if [ -f "${TAR_FILE}" ]; then
        echo "  Tar file exists, skipping download. Extracting..."
    else
        # Download parts
        for i in $(seq -w 0 $((NUM_PARTS - 1))); do
            PART="${PREFIX}.$(printf '%03d' $((10#$i)))"
            if [ -f "${PART}" ]; then
                echo "  Part ${PART} already downloaded, skipping."
            else
                echo "  Downloading ${PART}..."
                wget -q --show-progress -c "${BASE_URL}/${PART}" -O "${PART}"
            fi
        done

        # Concatenate parts into tar.gz
        echo "  Concatenating parts into ${TAR_FILE}..."
        cat ${PREFIX}.* > "${TAR_FILE}"
    fi

    # Extract
    echo "  Extracting ${TAR_FILE}..."
    tar -xzf "${TAR_FILE}"

    # Remove part files (keep tar)
    echo "  Removing part files..."
    for i in $(seq -w 0 $((NUM_PARTS - 1))); do
        PART="${PREFIX}.$(printf '%03d' $((10#$i)))"
        rm -f "${PART}"
    done

    echo "  Done: ${SEQ}"
done

echo ""
echo "======================================================================"
echo "All downloads complete!"
echo "Contents of ${OUTPUT_DIR}:"
ls -la "${OUTPUT_DIR}"
echo "======================================================================"
