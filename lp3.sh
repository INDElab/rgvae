#!/bin/bash
# Job requirements:

#SBATCH -N 1
#SBATCH -t 11:00:00
#SBATCH --mem=15.6G
#SBATCH -p gpu_titanrtx_shared
#SBATCH --gres=gpu:1

# Loading modules
module purge
module load 2019
module load Python/3.6.6-foss-2019b
module load CUDA/10.1.243
module load cuDNN/7.6.5.32-CUDA-10.1.243

echo "RGVAE training"
export DATA_TO_UNZIP_PATH=$HOME/rgvae.tar.gz
export DATASET_DIR="datasets"
export EXPERIMENT_NAME="GCVAE_h60_wn"
export PATH_TO_SOURCE="rgvae"
export STORE_DIR=$HOME/results

mkdir -p "$STORE_DIR"
# Copy input data from home to scratch
cp -R $HOME/"$PATH_TO_SOURCE" "$TMPDIR"
cd "$TMPDIR"/"rgvae"

# pip3 install --user -r requirements.txt
# pip3 install --user torch==1.7.0+cu101 torchvision==0.8.1+cu101 torchaudio==0.7.0 -f https://download.pytorch.org/whl/torch_stable.html
pip3 install --user -e .
python3 -u run_lp.py --ds wn18rr --m_path GCVAE_wn18rr_20e_12l_20201111.pt

tar -czf "$STORE_DIR"/"$EXPERIMENT_NAME".tar.gz "$TMPDIR"/"rgvae"/"data"
