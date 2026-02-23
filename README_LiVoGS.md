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
bash install_dependencies.sh
make

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

### Run LiVoGS Compression on HiFi4G Dataset

```bash
bash scripts/livogs_baseline/evaluate_livogs_compression.sh
```

### Generate plots for HiFi4G Dataset

```bash
bash scripts/livogs_baseline/plots/plot_benchmark.sh
```