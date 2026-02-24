## Setup

- Ubuntu 24.04
- GCC 12.4.0
- CUDA 12.1
- CuDNN 9.17.1

## Install Dependencies

```bash
git clone --recurse-submodules https://github.com/Rajrup/queen.git Queen
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

- Use 4DGaussians to preprocess the Neural 3D Video dataset.
- Follow instructions in `scripts/preprocess_dynerf.sh` to preprocess the Neural 3D Video dataset or follow: [here](https://github.com/hustvl/4DGaussians?tab=readme-ov-file#training)
- Preprocessed data: `/synology/rajrup/Queen/Neural_3D_Video`

```bash
conda activate Gaussians4D
cd /home/rajrup/project/4DGaussians/
./scripts/preprocess_dynerf.sh # Change sequence name before running.
```

### Google Immersive

## Training

### Training DyNeRF (Neural 3D Video)

```bash
python train.py --config configs/dynerf.yaml --log_images --log_ply -s /synology/rajrup/Queen/Neural_3D_Video/cook_spinach -m /synology/rajrup/Queen/train_output/Neural_3D_Video/cook_spinach

# Saving compressed model
python train.py --config configs/dynerf.yaml --log_images --log_compressed --log_ply -s /synology/rajrup/Queen/Neural_3D_Video/cook_spinach -m /synology/rajrup/Queen/train_output/Neural_3D_Video/cook_spinach
```

### Evaluation

```bash
python metrics_video.py -m /synology/rajrup/Queen/train_output/Neural_3D_Video/cook_spinach

# Render static camera viewpoints and spiral if trained w/o --log_compressed
python render.py -s /synology/rajrup/Queen/Neural_3D_Video/cook_spinach -m /synology/rajrup/Queen/train_output/Neural_3D_Video/cook_spinach
python render_fvv.py --config configs/dynerf.yaml  -s /synology/rajrup/Queen/Neural_3D_Video/cook_spinach -m /synology/rajrup/Queen/train_output/Neural_3D_Video/cook_spinach

# Render static camera viewpoints and spiral if trained w/ --log_compressed
python render_fvv_compressed.py --config configs/dynerf.yaml  -s /synology/rajrup/Queen/Neural_3D_Video/cook_spinach -m /synology/rajrup/Queen/train_output/Neural_3D_Video/cook_spinach_compressed --render_compare
```

## Compression

### VideoGS 2D Codec

```bash
bash scripts/videogs_baseline/evaluate_videogs_compression.sh

# Generate plots
bash scripts/videogs_baseline/plots/plot_benchmark.sh
```

### LiVoGS 3D Codec

```bash
bash scripts/livogs_baseline/evaluate_livogs_compression.sh

# Generate plots
bash scripts/livogs_baseline/plots/plot_benchmark.sh
```