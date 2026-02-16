## Setup

- Ubuntu 24.04
- GCC 12.4.0
- CUDA 12.1
- CuDNN 9.17.1

## Install Dependencies

```bash
git clone --recurse-submodules git@github.com:NVlabs/queen.git Queen
cd Queen
# set up some relevant directories
mkdir -p data
mkdir -p logs
mkdir -p output
```

### Conda Environment

```bash
conda create -n queen python=3.11 -y
conda activate queen

# Install the package and its dependencies
sudo apt-get install libglm-dev libgl1 -y
pip install six
pip install -e .
pip install tensorboard

# Install CUDA-dependent submodules (must be installed with --no-build-isolation)
pip install --no-build-isolation ./submodules/simple-knn
pip install --no-build-isolation ./submodules/diff-gaussian-rasterization
pip install --no-build-isolation ./submodules/gaussian-rasterization-grad

# Apply timm package patch (required for compatibility)
python scripts/patch_timm.py
```

### Download weights for MiDaS

```bash
wget -P MiDaS/weights https://github.com/isl-org/MiDaS/releases/download/v3_1/dpt_beit_large_512.pt
```

## Data Preparation

### Neural 3D Video (or DyNeRF)
```bash
mkdir -p data/multipleview

mkdir -p /synology/rajrup/Queen/Neural_3D_Video/cook_spinach
for i in $(seq -w 0 20); do ln -s /synology/rajrup/4DGaussians/Neural_3D_Video/cook_spinach/cam${i} /synology/rajrup/Queen/Neural_3D_Video/cook_spinach/; done
ln -s /synology/rajrup/4DGaussians/Neural_3D_Video/cook_spinach/points3D_downsample2.ply /synology/rajrup/Queen/Neural_3D_Video/cook_spinach/
ln -s /synology/rajrup/4DGaussians/Neural_3D_Video/cook_spinach/poses_bounds.npy /synology/rajrup/Queen/Neural_3D_Video/cook_spinach/
ln -s /synology/rajrup/4DGaussians/Neural_3D_Video/cook_spinach/colmap /synology/rajrup/Queen/Neural_3D_Video/cook_spinach/
ln -s /synology/rajrup/4DGaussians/Neural_3D_Video/cook_spinach/image_colmap /synology/rajrup/Queen/Neural_3D_Video/cook_spinach/
ln -s /synology/rajrup/4DGaussians/Neural_3D_Video/cook_spinach/sparse_ /synology/rajrup/Queen/Neural_3D_Video/cook_spinach/
ln -s /synology/rajrup/Queen/Neural_3D_Video/cook_spinach/ data/multipleview/
```

### Google Immersive

## Training

### Training DyNeRF

```bash
python train.py --config configs/dynerf.yaml --log_images --log_ply -s data/multipleview/cook_spinach -m ./output/cook_spinach_trained

# Saving compressed model
python train.py --config configs/dynerf.yaml --log_images --log_compressed --log_ply -s data/multipleview/cook_spinach -m ./output/cook_spinach_trained_compressed
```

### Evaluation

```bash
python metrics_video.py -m ./output/cook_spinach_trained

# Render static camera viewpoints and spiral if trained w/o --log_compressed
python render.py -s data/multipleview/cook_spinach -m ./output/cook_spinach_trained
python render_fvv.py --config configs/dynerf.yaml  -s data/multipleview/cook_spinach -m ./output/cook_spinach_trained

# Render static camera viewpoints and spiral if trained w/ --log_compressed
python render_fvv_compressed.py --config configs/dynerf.yaml  -s data/multipleview/cook_spinach -m ./output/cook_spinach_trained_compressed
```

