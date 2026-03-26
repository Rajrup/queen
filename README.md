## Setup

System details:
```
Jetpack 6.2.1
CUDA 12.6
Ubuntu 22.04
CMake version 3.22.1
GCC version 11.4.0
Python version 3.10
PyTorch version 2.5.0
Torchvision version 0.20.1
```

## Install Dependencies

```bash
git clone --recurse-submodules https://github.com/Rajrup/queen.git Queen
cd Queen
```

### Conda Environment

```bash
conda create -n queen python=3.10 -y
conda activate queen

# Install the package and its dependencies
sudo apt-get install libglm-dev libgl1 -y
pip install six
pip install numpy==1.26.4
pip install opencv-python==4.11.0.86
pip install Cython

# Install PyTorch and torchvision
pip install https://developer.download.nvidia.com/compute/redist/jp/v61/pytorch/torch-2.5.0a0+872d972e41.nv24.08.17622132-cp310-cp310-linux_aarch64.whl
# Install torchvision from source: https://forums.developer.nvidia.com/t/pytorch-for-jetson/72048
sudo apt install libjpeg-dev zlib1g-dev libpython3-dev libopenblas-dev libavcodec-dev libavformat-dev libswscale-dev
pip install pillow
git clone --branch <version> https://github.com/pytorch/vision torchvision
cd torchvision
git checkout tags/v0.20.1
export BUILD_VERSION=0.20.1
python setup.py install

sudo apt-get install python3-dev libxml2-dev libxslt1-dev zlib1g-dev libjsoncpp-dev

# Install the package (use --no-deps so pip does not reinstall PyTorch/torchvision from PyPI)
pip install plyfile tqdm pillow scipy wandb torchmetrics imutils matplotlib torchac timm==0.6.13 einops==0.6.0
pip install -e . --no-deps
pip install torchmetrics[image]
pip install tensorboard

# Install CUDA-dependent submodules (must be installed with --no-build-isolation)
pip install --no-build-isolation ./submodules/simple-knn
pip install --no-build-isolation ./submodules/diff-gaussian-rasterization
pip install --no-build-isolation ./submodules/gaussian-rasterization-grad

# Other optional dependencies
pip install pynvml psutil

# Apply timm package patch (required for compatibility)
python scripts/patch_timm.py
```

**Dependencies for LiVoGS:**

Follow the instructions in `README_LiVoGS.md` to set up LiVoGS.

```

### LiVoGS 3D Codec

```bash
# Full RD-curve sweep (multi-QP, multi-GPU)
python scripts/livogs_baseline/run_rd_pipeline.py

# Single experiment
python scripts/livogs_baseline/rd_pipeline/worker.py --dataset_name Neural_3D_Video --sequence_name cook_spinach
# Generate plots
bash scripts/livogs_baseline/plots/plot_benchmark.sh
```