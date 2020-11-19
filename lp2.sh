#!/bin/bash
# Job requirements:

#SBATCH -N 1
#SBATCH -t 66:00:00
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
export EXPERIMENT_NAME="GCVAE_h60_fb"
export PATH_TO_SOURCE="rgvae"
export STORE_DIR=$HOME/results1117

mkdir -p "$STORE_DIR"
# Copy input data from home to scratch
cp -R $HOME/"$PATH_TO_SOURCE" "$TMPDIR"
cd "$TMPDIR"/"rgvae"

# pip3 install --user -r requirements.txt
# pip3 install --user torch==1.7.0+cu101 torchvision==0.8.1+cu101 torchaudio==0.7.0 -f https://download.pytorch.org/whl/torch_stable.html
pip3 install --user -e .
python3 -u run_lp.py --ds fb15k --m_path GCVAE_fb15k_89e_14l_20201111.pt --bs_2 13

tar -czf "$STORE_DIR"/"$EXPERIMENT_NAME".tar.gz "$TMPDIR"/"rgvae"/"data"

export USER_AT_HOST="fwolf@login-gpu.lisa.surfsara.nl"
export PUBKEYPATH="$HOME/.ssh/id_ed25519.pub"

ssh-copy-id -i "$PUBKEYPATH" "$USER_AT_HOST"