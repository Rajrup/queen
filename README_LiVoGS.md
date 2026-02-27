## LiVoGS

Setup and running instructions for LiVoGS on QUEEN trained models.

## Setup LiVoGS

```bash
conda activate queen

# Clone LiVoGS (no submodules)
git submodule add https://github.com/haodongw101/LiVoGS.git
cd LiVoGS

# Selectively clone only the PyRLGR pybind11 submodule
git submodule update --init compression/PyRLGR/thirdparty/pybind11

# Install Octree Compression
cd compression/Octree_Compression_GPU
make pybind

# Install RAHT-3DGS-codec
cd ../RAHT-3DGS-codec/cuda
pip install . --no-build-isolation

# Install PyRLGR
cd ../../PyRLGR
pip install . --no-build-isolation

# We won't install gsplat as we are using the QUEEN trained models.
cd ../../../
```

## Compression

### Run LiVoGS Compression on DyNeRF Dataset

```bash
# Full RD-curve sweep (multi-QP, multi-GPU)
python scripts/livogs_baseline/run_rd_pipeline.py

# Single experiment
python scripts/livogs_baseline/rd_pipeline/worker.py --dataset_name Neural_3D_Video --sequence_name cook_spinach
```

### Generate plots for HiFi4G Dataset

```bash
bash scripts/livogs_baseline/plots/plot_benchmark.sh
```