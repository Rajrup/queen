# VideoGS baseline

Run from the **project root**.

**`evaluate_videogs_compression.sh`** — Full pipeline: PLY → PNG → H.264 → decompress → PLY, then quality evaluation. Edit variables at the top (dataset, QP, frame range, etc.). Writes under `train_output/.../videogs_compression/qp_<QP>/`.

**`plots/plot_benchmark.sh`** — Generates all benchmark plots (compression time bar and area, compressed size, quality) for the default QP and paths. Set `input_folder` and `QP` in the script to match your run. Plots go to `plots/plots/videogs_compression/qp_<qp>/`.
